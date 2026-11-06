"""Runtime-only observation; never changes model, input or sampling parameters.

The installed processor creates effective controls exactly as before. This
wrapper records/asserts them. Original native returns are fsynced by the existing
return journal before bind_returns validates target identities. No retries.
"""
import hashlib,json,os,sys,time
from pathlib import Path

def compact(x):return json.dumps(x,ensure_ascii=False,separators=(',',':'))
def digest(x):return hashlib.sha256(x.encode()).hexdigest()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def durable(path,obj):
 with Path(path).open('a',encoding='utf8') as f:f.write(compact(obj)+'\n');f.flush();os.fsync(f.fileno())
def write(path,obj):
 with Path(path).open('x',encoding='utf8') as f:json.dump(obj,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())

def worker_identity(worker):
 """Trusted root-frozen callable: only inspect the already loaded local model."""
 model=worker.model_runner.model
 parameters=list(model.parameters())
 return {'worker_pid':os.getpid(),'model_class':type(model).__name__,'model_module':type(model).__module__,
  'parameter_dtypes':sorted({str(p.dtype) for p in parameters}),'parameters':sum(p.numel() for p in parameters),
  'remote_model_modules_loaded':[n for n in sys.modules if n.startswith('transformers_modules.')]}

LAUNCH_ENV_KEYS=('CUDA_VISIBLE_DEVICES','TMPDIR','XDG_CACHE_HOME','XDG_CONFIG_HOME','HF_HOME','HF_MODULES_CACHE',
 'TORCH_HOME','VLLM_CACHE_ROOT','VLLM_CONFIG_ROOT','NUMBA_CACHE_DIR','TORCHINDUCTOR_CACHE_DIR','TRITON_CACHE_DIR',
 'CUDA_CACHE_PATH','TORCH_EXTENSIONS_DIR','FLASHINFER_WORKSPACE_BASE','PYTHONPATH','INTENTSAT_SINGLE_RANK_NCCL_COMPAT',
 'VLLM_USE_V1','VLLM_RPC_BASE_PATH','VLLM_ALLOW_INSECURE_SERIALIZATION','VLLM_NO_USAGE_STATS','DO_NOT_TRACK','PYTHONDONTWRITEBYTECODE')

def launch_environment(root,model):
 """Only experiment-owned caches and local communication addresses differ."""
 root=Path(root);base=root/'runtime_cache'/model
 paths={'TMPDIR':'tmp','XDG_CACHE_HOME':'xdg','XDG_CONFIG_HOME':'config','HF_HOME':'hf','HF_MODULES_CACHE':'hf_modules',
  'TORCH_HOME':'torch','VLLM_CACHE_ROOT':'vllm','VLLM_CONFIG_ROOT':'vllm_config','NUMBA_CACHE_DIR':'numba',
  'TORCHINDUCTOR_CACHE_DIR':'inductor','TRITON_CACHE_DIR':'triton','CUDA_CACHE_PATH':'cuda',
  'TORCH_EXTENSIONS_DIR':'extensions','FLASHINFER_WORKSPACE_BASE':'flashinfer_base'}
 env={k:str(base/v) for k,v in paths.items()}
 env['VLLM_RPC_BASE_PATH']='/home/gpu/i2sr1106/'+{'qwen':'tq','smol':'ts'}[model]
 env['VLLM_ALLOW_INSECURE_SERIALIZATION']='1'
 return env

class RuntimeObserver:
 def __init__(self,llm,out,model,contract):
  self.llm=llm;self.out=Path(out);self.model=model;self.contract=contract;self.expected=contract['models'][model]
  self.active=None;self.processed=[];self.seen_engine_ids=set();self.bound=0;self.journal_offset=0
  engine=llm.llm_engine;cfg=engine.vllm_config;mc=engine.model_config;tok=llm.get_tokenizer()
  native=llm.collective_rpc(worker_identity)
  actual={'created_unix':time.time(),'worker_identity':native,'engine_class':type(engine).__module__+'.'+type(engine).__name__,
   'native_hf_config_class':type(mc.hf_config).__module__+'.'+type(mc.hf_config).__name__,
   'model_impl_requested':getattr(mc.model_impl,'value',mc.model_impl),'dtype':str(mc.dtype),'max_model_len':mc.max_model_len,
   'max_num_seqs':cfg.scheduler_config.max_num_seqs,'tensor_parallel_size':cfg.parallel_config.tensor_parallel_size,
   'enforce_eager':mc.enforce_eager,'prefix_caching':cfg.cache_config.enable_prefix_caching,
   'gpu_memory_utilization':cfg.cache_config.gpu_memory_utilization,'tokenizer_class':type(tok).__name__,
   'primary_eos_token_id':tok.eos_token_id,'generation_config_fields':engine.processor.generation_config_fields,
   'guided_decoding_backend':cfg.decoding_config.backend,'guided_decoding_disable_fallback':cfg.decoding_config.disable_fallback,
   'root_trusted_callable_sha256':contract['worker_identity_source_sha256'],
   'serialization_scope':'This experiment process and its local children permit generic pickle/cloudpickle RPC fallback. The actual callable supplied here is the reviewed frozen worker_identity; this is not a one-function allowlist.',
   'launch_environment':{k:os.environ.get(k) for k in LAUNCH_ENV_KEYS}}
  # Persist actual identity even if a check fails, before any scientific generate.
  write(self.out/'EFFECTIVE_NATIVE_RUNTIME.json',actual)
  e=self.expected
  assert len(native)==1 and native[0]['model_class']==e['native_class'] and native[0]['model_module']==e['native_module']
  assert native[0]['parameter_dtypes']==['torch.bfloat16'] and native[0]['parameters']>0 and not native[0]['remote_model_modules_loaded']
  assert actual['native_hf_config_class']==e['native_hf_config_class'] and actual['dtype']=='torch.bfloat16'
  # Preserve the original model_impl=AUTO request. Native implementation is
  # established by the exact loaded worker class/module, not inferred from AUTO.
  assert actual['model_impl_requested'] in ('auto','vllm')
  assert actual['max_model_len']==4096 and actual['max_num_seqs']==16 and actual['tensor_parallel_size']==1
  assert actual['enforce_eager'] is True and actual['prefix_caching'] is False and actual['gpu_memory_utilization']==.70
  assert actual['guided_decoding_backend']=='xgrammar' and actual['guided_decoding_disable_fallback'] is True
  assert actual['primary_eos_token_id']==e['primary_eos_token_id']
  original=engine.processor.process_inputs
  def observed(*args,**kwargs):
   result=original(*args,**kwargs);request=result[1];p=request.sampling_params
   if p is None:raise RuntimeError('Transfer experiment unexpectedly processed a non-generation request')
   position=len(self.processed);expected=self.active['jobs'][position] if self.active and position<len(self.active['jobs']) else None
   controls={k:getattr(p,k) for k in ('n','temperature','top_k','top_p','min_p','max_tokens','ignore_eos','repetition_penalty','presence_penalty','frequency_penalty','seed','skip_special_tokens','spaces_between_special_tokens')}
   controls.update(stop=list(p.stop),stop_token_ids=sorted(p.stop_token_ids),all_stop_token_ids=sorted(p.all_stop_token_ids),
    primary_eos_token_id=request.eos_token_id,sampling_type=p.sampling_type.name,
    guided_backend=p.guided_decoding.backend if p.guided_decoding else None,
    guided_disable_fallback=p.guided_decoding.disable_fallback if p.guided_decoding else None,
    grammar_sha256=digest(p.guided_decoding.regex) if p.guided_decoding else None)
   row={'observed_unix':time.time(),'engine_request_id':request.request_id,'model':model,
    'stage':self.active['stage'] if self.active else None,'block':self.active['block'] if self.active else None,
    'position':position,'candidate_logical_request_id':expected['request_id'] if expected else None,
    'prompt_token_ids_sha256':digest(compact(list(request.prompt_token_ids))),'effective_controls':controls,
    'validation_status':'UNBOUND until the full native return batch is persisted'}
   durable(self.out/'effective_requests.jsonl',row);self.processed.append(row)
   assert expected is not None,'Unexpected processed request outside the submitted batch'
   assert row['engine_request_id'] not in self.seen_engine_ids;self.seen_engine_ids.add(row['engine_request_id'])
   assert row['prompt_token_ids_sha256']==digest(compact(expected['prompt_token_ids']))
   assert (p.n,p.temperature,p.top_p,p.top_k,p.min_p,p.max_tokens)==(1,0,1,0,0,128)
   assert not p.ignore_eos and p.sampling_type.name=='GREEDY'
   assert (p.repetition_penalty,p.presence_penalty,p.frequency_penalty,p.seed)==(1,0,0,None)
   assert controls['stop']==[], 'Unplanned custom stop strings'
   assert controls['stop_token_ids']==sorted(i for i in e['all_stop_token_ids'] if i!=e['primary_eos_token_id']), 'Unexpected non-primary EOS stop token IDs'
   assert p.skip_special_tokens is True and p.spaces_between_special_tokens is True, 'Output formatting controls changed'
   assert sorted(p.all_stop_token_ids)==e['all_stop_token_ids'] and request.eos_token_id==e['primary_eos_token_id']
   guided=expected.get('guided') is not False
   assert (p.guided_decoding is not None)==guided
   if guided:
    assert p.guided_decoding.backend=='xgrammar' and p.guided_decoding.disable_fallback is True
    assert controls['grammar_sha256']==digest(expected['grammar'])
   return result
  engine.processor.process_inputs=observed
 def begin(self,jobs,stage,block=None):
  if self.active is not None:raise RuntimeError('Previous generate batch was not fully bound; no retry allowed')
  self.active={'jobs':jobs,'stage':stage,'block':block};self.processed=[]
  durable(self.out/'effective_batch_submissions.jsonl',{'stage':stage,'block':block,'logical_request_ids':[j['request_id'] for j in jobs],'time_unix':time.time()})
 def bind_returns(self,jobs,values,journal_path):
  # Called ONLY after original return_journal.generate_preserving_returns has
  # fsynced every actual output. Failed binding never loses an observed return.
  assert self.active is not None and len(values)==len(jobs)==len(self.processed)
  assert [j['request_id'] for j in jobs]==[j['request_id'] for j in self.active['jobs']]
  receipts=[]
  for j,value,p in zip(jobs,values,self.processed):
   receipts.append({'logical_request_id':j['request_id'],'engine_request_id':getattr(value,'request_id',None),
    'processed_engine_request_id':p['engine_request_id'],'processed_prompt_token_ids_sha256':p['prompt_token_ids_sha256'],
    'returned_prompt_token_ids_sha256':digest(compact(list(value.prompt_token_ids))) if value.prompt_token_ids is not None else None,
    'grammar_sha256':p['effective_controls']['grammar_sha256'],'stage':self.active['stage'],'block':self.active['block']})
  with Path(journal_path).open('rb') as stream:
   stream.seek(self.journal_offset);segment=stream.read()
  # Exactly the new, already-fsynced journal line; O(total bytes), not a hash of
  # the entire growing journal at every batch.
  journal=json.loads(segment);assert journal['stage']==self.active['stage'] and journal['block']==self.active['block']
  assert journal['expected_batch_request_ids']==[j['request_id'] for j in jobs]
  durable(self.out/'effective_return_bindings.jsonl',{'stage':self.active['stage'],'block':self.active['block'],
   'journal_byte_offset':self.journal_offset,'journal_byte_length':len(segment),
   'journal_segment_sha256':hashlib.sha256(segment).hexdigest(),'bindings':receipts,'validated':False})
  for j,row in zip(jobs,receipts):
   assert row['engine_request_id']==row['processed_engine_request_id']
   assert row['returned_prompt_token_ids_sha256']==row['processed_prompt_token_ids_sha256']==digest(compact(j['prompt_token_ids']))
  durable(self.out/'effective_binding_validation.jsonl',{'stage':self.active['stage'],'block':self.active['block'],'logical_request_ids':[j['request_id'] for j in jobs],'validated':True})
  self.journal_offset+=len(segment);self.bound+=len(jobs);self.active=None;self.processed=[]
 def receipts(self):
  names=('EFFECTIVE_NATIVE_RUNTIME.json','effective_requests.jsonl','effective_batch_submissions.jsonl','effective_return_bindings.jsonl','effective_binding_validation.jsonl')
  return {'model':self.model,'bound_science_and_control_requests':self.bound,'active_batch_unbound':self.active is not None,
   'files':{n:sha(self.out/n) for n in names if (self.out/n).exists()}}
