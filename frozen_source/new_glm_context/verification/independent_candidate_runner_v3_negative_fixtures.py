"""CPU fake-engine execution of the actual runner; no model/runtime GPU import."""
import os,sys,json,pathlib,types,importlib.util,traceback,hashlib,re
os.environ['CUDA_VISIBLE_DEVICES']=''
sys.dont_write_bytecode=True
BASE=pathlib.Path('/home/gpu/IntentSat2S_revision_20261106/new_glm_context')
OUT=pathlib.Path(__file__).resolve().parent/os.environ.get('REVIEW_OUTPUT_SUBDIR','runner_v3_candidate_checks')
OUT.mkdir(exist_ok=False)
sys.path.insert(0,str(BASE))
import common
spec=importlib.util.spec_from_file_location('review_runner',BASE/'run_gpu.py')
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
S=types.SimpleNamespace
write=lambda p,d:p.write_text(json.dumps(d)+'\n')
lines=lambda p,rs:p.write_text(''.join(json.dumps(r)+'\n' for r in rs))
class Guided:
 def __init__(self,**kw):self.__dict__.update(kw)
class Sample:
 def __init__(self,**kw):
  self.__dict__.update(dict(n=1,temperature=0,top_k=0,top_p=1,min_p=0,max_tokens=128,ignore_eos=False,repetition_penalty=1,presence_penalty=0,frequency_penalty=0,seed=None,skip_special_tokens=True,spaces_between_special_tokens=True));self.__dict__.update(kw)
  self.all_stop_token_ids=set(self.stop_token_ids);self.sampling_type=S(name='GREEDY')
RESULTS=[]
for fault in ('prompt_token_mismatch','response_count_mismatch','reordered_same_prompt','extra_return','extra_completion','empty_return','invalid_raw','parser_exception','generate_raises','diagnostic_token_mismatch'):
 root=OUT/('runner_v3_fixture_'+fault);root.mkdir(exist_ok=False)
 for d in ('prepared','runtime_override','model'): (root/d).mkdir()
 write(root/'model/config.json',{})
 cfg=json.loads((BASE/'prepared/runtime_configuration_glm.json').read_text())
 cfg.update(packages={},python='.'.join(map(str,sys.version_info[:3])),python_executable=sys.executable,vllm_source_sha256={})
 cfg['llm_kwargs']={}
 write(root/'prepared/runtime_configuration_glm.json',cfg)
 write(root/'prepared/MODEL_FILE_SHA256.json',{'glm':{'path':str(root/'model'),'files':{'config.json':common.sha(root/'model/config.json')}}})
 write(root/'PREINFERENCE_FROZEN_SHA256.json',{'files':{}})
 write(root/'GPU_GO_glm.json',{'authorized_by':'root','device':0,'model':'glm','freeze_sha256':common.sha(root/'PREINFERENCE_FROZEN_SHA256.json')})
 def job(rid,idx=0):
  return dict(request_id=rid,ordinal=idx,block=0,task='boolq',target_id='fixture-'+str(idx),selected_ordinal=idx,scope='target-only',context='near',repeat=1,requested_ids=[1],output_ids=[1],prompt_token_ids=[42],prompt_sha256='fixture',grammar=common.grammar([1]),grammar_sha256=common.digest(common.grammar([1])),id_map_sha256='fixture',wrapped_prompt='CPU FIXTURE')
 dg=[dict(job('ordinary'),diagnostic_group='ordinary_banana',guided=False,output_ids=[]),dict(job('guided'),diagnostic_group='guided_banana',guided=True),dict(job('smoke'),diagnostic_group='synthetic_format_smoke',guided=True)]
 dg[0].pop('grammar',None);dg[0].pop('grammar_sha256',None)
 lines(root/'prepared/glm_diagnostic_requests.jsonl',dg)
 science=[job('science/'+str(i),i) for i in range(3)]
 lines(root/'prepared/science.jsonl',science)
 write(root/'prepared/REQUEST_ENVELOPE_glm.json',{'unique_all_calls':6,'execution_order':['mixed'],'modes':{'mixed':{'file':'prepared/science.jsonl','requests_per_submission':64}}})
 INSTANCES=[]
 class FakeLLM:
  def __init__(self,**kw):
   self.calls=0;self.returned=[];self.last_scientific_returns=[];INSTANCES.append(self)
   mc=S(hf_config=S(),model_impl='vllm',dtype='torch.bfloat16',max_model_len=4096,enforce_eager=True)
   ec=S(scheduler_config=S(max_num_seqs=16),parallel_config=S(tensor_parallel_size=1),cache_config=S(enable_prefix_caching=False,gpu_memory_utilization=.7),decoding_config=S(backend='xgrammar',disable_fallback=True))
   def process(*args,**kwargs):return None,args[0]
   self.llm_engine=S(model_config=mc,vllm_config=ec,processor=S(generation_config_fields={},process_inputs=process),engine_core=S(shutdown=lambda:None))
  def collective_rpc(self,*args):return [{'model_class':cfg['expected_native']['class'],'model_module':cfg['expected_native']['module'],'parameter_dtypes':['torch.bfloat16'],'remote_model_modules_loaded':[]}]
  def get_tokenizer(self):return S(eos_token_id=cfg['primary_eos_token_id'])
  def generate(self,prompts,params,**kwargs):
   self.calls+=1;responses=[]
   for i,(p,s) in enumerate(zip(prompts,params)):
    eid=f'fake-engine/{self.calls}/{i}'
    self.llm_engine.processor.process_inputs(S(request_id=eid,prompt_token_ids=p['prompt_token_ids'],eos_token_id=cfg['primary_eos_token_id'],sampling_params=s))
    if s.guided_decoding:
     ids=re.findall(r'"(\d+)"',s.guided_decoding.regex);raw=json.dumps({key:'Y' for key in ids},separators=(',',':'))
    else:raw='BANANA'
    if self.calls==4 and fault=='invalid_raw' and i==1:raw=' {"1":"Y"}'
    if self.calls==4 and fault=='parser_exception' and i==1:raw='RAISE_CPU_PARSER'
    if self.calls==4 and fault=='generate_raises':raise RuntimeError('CPU fake generate exception after processing input')
    tokens=[99] if (self.calls==4 and fault=='prompt_token_mismatch' and i==1) or (self.calls==2 and fault=='diagnostic_token_mismatch') else p['prompt_token_ids']
    responses.append(S(request_id=eid,prompt_token_ids=tokens,outputs=[S(text=raw,token_ids=[100],finish_reason='stop',stop_reason=None)]))
   if self.calls==4:
    if fault=='response_count_mismatch':responses=responses[:-1]
    if fault=='reordered_same_prompt':responses=[responses[1],responses[0],responses[2]]
    if fault=='extra_return':responses.append(S(request_id='UNEXPECTED_ENGINE_ID',prompt_token_ids=[42],outputs=[S(text='EXTRA_RETURN_SENTINEL',token_ids=[101,102],finish_reason='stop',stop_reason=None)]))
    if fault=='extra_completion':responses[1].outputs.append(S(text='EXTRA_COMPLETION_SENTINEL',token_ids=[103,104,105],finish_reason='stop',stop_reason=None))
    if fault=='empty_return':responses=[]
    self.last_scientific_returns=responses
   self.returned.extend(responses);return responses
 fakev=types.ModuleType('vllm');fakev.LLM=FakeLLM;fakev.SamplingParams=Sample
 fakev.__spec__=importlib.util.spec_from_file_location('vllm',root/'fake_vllm/__init__.py')
 fakeparams=types.ModuleType('vllm.sampling_params');fakeparams.GuidedDecodingParams=Guided
 adapter=types.ModuleType('single_rank_nccl_compat');adapter.ACTIVE=True;adapter.__file__=str(root/'runtime_override/single_rank_nccl_compat.py')
 for name,mod in [('vllm',fakev),('vllm.sampling_params',fakeparams),('single_rank_nccl_compat',adapter)]:sys.modules[name]=mod
 runner.ROOT=root;runner.gpu_processes=lambda:[];runner.check_owned=lambda:[]
 original_parse=common.parse
 def faultparse(raw,ids):
  if raw=='RAISE_CPU_PARSER':raise RuntimeError('CPU injected parser exception')
  return original_parse(raw,ids)
 runner.parse=faultparse
 sys.argv=['run_gpu.py','--model','glm','--device','0'];os.environ['CUDA_VISIBLE_DEVICES']='0'
 caught=None
 try:runner.main()
 except BaseException as e:caught=type(e).__name__+': '+str(e)
 response_path=root/'run_glm/responses.jsonl'
 saved=common.read_lines(response_path) if response_path.exists() else []
 failure_path=root/'run_glm/FAILURE.json'
 receipts=common.read_lines(root/'run_glm/raw_return_receipts.jsonl')
 science_receipts=[r for r in receipts if r['mode']=='mixed']
 actual_science=INSTANCES[-1].last_scientific_returns
 expected_raw=[a.text for r in actual_science for a in r.outputs]
 captured_raw=[a['raw'] for r in science_receipts for ret in r['actual_returns'] for a in (ret['outputs'] or [])]
 accounting=json.loads((root/'run_glm/RAW_RETURN_ACCOUNTING.json').read_text())
 assert captured_raw==expected_raw,(fault,captured_raw,expected_raw)
 if fault=='invalid_raw':assert caught is None and len(saved)==3 and any(not r['strict_valid'] for r in saved)
 else:assert caught is not None and failure_path.exists()
 if fault in ('prompt_token_mismatch','response_count_mismatch','reordered_same_prompt','extra_return','extra_completion','empty_return','generate_raises','diagnostic_token_mismatch'):
  assert accounting['complete_total_input_tokens'] is None and accounting['complete_total_output_tokens'] is None
 if fault=='parser_exception':assert accounting['complete_total_input_tokens']==6 and accounting['observed_returned_requests']==6
 if fault=='diagnostic_token_mismatch':assert receipts[-1]['mode']=='diagnostic/guided_banana' and receipts[-1]['actual_returns'][0]['actual_prompt_token_ids']==[99]
 results={'fault':fault,'exception':caught,'saved_canonical_scientific_rows':len(saved),'all_actual_return_raw_completions_captured_before_failure':captured_raw==expected_raw,'captured_scientific_requests':sum(len(r['actual_returns']) for r in science_receipts),'captured_scientific_completions':len(captured_raw),'all_observed_returned_requests':accounting['observed_returned_requests'],'complete_input_total':accounting['complete_total_input_tokens'],'complete_output_total':accounting['complete_total_output_tokens'],'failed_generate_events':accounting['failed_generate_events'],'used_actual_runner_source_sha256':common.sha(BASE/'run_gpu.py'),'used_actual_generation_receipts_sha256':common.sha(BASE/'generation_receipts.py'),'GPU_execution':False,'vllm_and_worker_are_fake':True}
 RESULTS.append(results);print(results)
 os.environ['CUDA_VISIBLE_DEVICES']=''
write(OUT/'runner_v3_negative_results.json',RESULTS)
assert len(RESULTS)==10 and all(r['all_actual_return_raw_completions_captured_before_failure'] for r in RESULTS)
