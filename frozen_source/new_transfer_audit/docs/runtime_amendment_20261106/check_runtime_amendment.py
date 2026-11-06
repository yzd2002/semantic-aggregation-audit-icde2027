#!/usr/bin/env python3
"""CPU fake-engine execution of the actual amended runner, isolated output only.

Every output from this checker is synthetic and must never be presented as a
scientific model judgment. The checker never starts CUDA or loads model weights.
"""
import argparse,collections,contextlib,copy,importlib.machinery,importlib.util,io,json,os,re,sys,types
from pathlib import Path
from unittest.mock import patch
os.environ['CUDA_VISIBLE_DEVICES']=''
def compact(x):return json.dumps(x,ensure_ascii=False,separators=(',',':'))
def write(p,x):
 with p.open('x') as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
def rows(p,x):
 with p.open('x') as f:
  for r in x:f.write(compact(r)+'\n')
def lines(p):return [json.loads(t) for t in p.read_text().splitlines() if t.strip()] if p.exists() else []
def load(p):return json.loads(p.read_text())
def main():
 p=argparse.ArgumentParser();p.add_argument('--base-science-root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();producer=Path(__file__).resolve().parent;base=a.base_science_root;out=a.out;out.mkdir(parents=True,exist_ok=False)
 sys.path.insert(0,str(base));sys.path.insert(0,str(producer))
 import runtime_observer as observe
 spec=importlib.util.spec_from_file_location('amended_actual_runner',producer/'run_gpu.py');run=importlib.util.module_from_spec(spec);spec.loader.exec_module(run)
 import analyze_transfer as analyzer
 contract_path=producer/'prepared/runtime_observation_contract.json'
 if not contract_path.is_file():contract_path=producer/'runtime_observation_contract.json'
 contract=load(contract_path);vllm_origin=importlib.util.find_spec('vllm').origin
 cases=[]
 for model,fault in [('qwen','none'),('smol','none'),('qwen','native_class'),('qwen','native_dtype'),('qwen','effective_greedy'),('qwen','effective_eos'),('qwen','effective_grammar'),('qwen','effective_prompt'),('qwen','return_id_second'),('qwen','return_prompt_second'),('qwen','short_returns'),('qwen','extra_returns'),('qwen','multiple_completions'),('qwen','invalid_raw')]:
  f=out/f'{model}_{fault}';f.mkdir();(f/'prepared').mkdir();(f/'runtime_override').mkdir()
  science=lines(base/'prepared'/f'{model}_requests.jsonl')[:3];schedule=lines(base/'prepared'/f'{model}_schedule.jsonl')[:3];controls=lines(base/'prepared'/f'{model}_controls.jsonl')
  write(f/'model_locations.json',{model:str(f/'CPU_SYNTHETIC_NO_MODEL')});write(f/'prepared/MODEL_FILE_SHA256.json',{model:{'files':{}}})
  write(f/'prepared/runtime_configuration.json',{'packages':{},'python':__import__('platform').python_version(),'python_executable':sys.executable,'FLASHINFER_WORKSPACE_BASE':'CPU_BASELINE95_SUPERSEDED_SHARED_PATH'})
  write(f/'prepared/runtime_observation_contract.json',contract);(f/'runtime_observer.py').symlink_to(producer/'runtime_observer.py')
  rows(f/'prepared'/f'{model}_requests.jsonl',science);rows(f/'prepared'/f'{model}_schedule.jsonl',schedule);rows(f/'prepared'/f'{model}_controls.jsonl',controls)
  expected=contract['models'][model];nativehf=type(expected['native_hf_config_class'].split('.')[-1],(),{'__module__':'.'.join(expected['native_hf_config_class'].split('.')[:-1])})()
  class Param:
   def __init__(self,**kw):self.__dict__.update(kw)
  calls=[]
  class Processor:
   generation_config_fields={'eos_token_id':expected['all_stop_token_ids']}
   def process_inputs(self,request_id,prompt,params):
    q=copy.deepcopy(params);q.n=1;q.top_p=1.;q.top_k=0;q.min_p=0.;q.ignore_eos=False;q.repetition_penalty=1.;q.presence_penalty=q.frequency_penalty=0.;q.seed=None;q.skip_special_tokens=True;q.spaces_between_special_tokens=True;q.stop_token_ids=[i for i in expected['all_stop_token_ids'] if i!=expected['primary_eos_token_id']];q.all_stop_token_ids=list(expected['all_stop_token_ids']);q.sampling_type=types.SimpleNamespace(name='GREEDY')
    if len(calls)==3:
     if fault=='effective_greedy':q.temperature=.7;q.sampling_type.name='RANDOM'
     if fault=='effective_eos':q.all_stop_token_ids=[9999999]
     if fault=='effective_grammar' and q.guided_decoding:q.guided_decoding.regex='CPU_WRONG_GRAMMAR'
    tokens=list(prompt['prompt_token_ids'])
    if len(calls)==3 and fault=='effective_prompt':tokens.append(9999999)
    return None,types.SimpleNamespace(request_id=request_id,prompt_token_ids=tokens,sampling_params=q,eos_token_id=expected['primary_eos_token_id'])
  class Engine:
   def __init__(self,**kw):
    calls.append({'LLM_constructor':kw})
    cfg=types.SimpleNamespace(scheduler_config=types.SimpleNamespace(max_num_seqs=16),parallel_config=types.SimpleNamespace(tensor_parallel_size=1),cache_config=types.SimpleNamespace(enable_prefix_caching=False,gpu_memory_utilization=.70),decoding_config=types.SimpleNamespace(backend='xgrammar',disable_fallback=True))
    mc=types.SimpleNamespace(hf_config=nativehf,model_impl=types.SimpleNamespace(value='auto'),dtype='torch.bfloat16',max_model_len=4096,enforce_eager=True)
    self.llm_engine=types.SimpleNamespace(vllm_config=cfg,model_config=mc,processor=Processor());self.generated=0;self.eid=0
   def get_tokenizer(self):return types.SimpleNamespace(eos_token_id=expected['primary_eos_token_id'])
   def collective_rpc(self,fn):
    assert fn is observe.worker_identity
    return [{'worker_pid':12345,'model_class':'CPU_WRONG_NATIVE' if fault=='native_class' else expected['native_class'],'model_module':expected['native_module'],'parameter_dtypes':['torch.float16'] if fault=='native_dtype' else ['torch.bfloat16'],'parameters':12345,'remote_model_modules_loaded':[]}]
   def generate(self,prompts,parameters,use_tqdm=False):
    self.generated+=1;values=[]
    for i,(prompt,params) in enumerate(zip(prompts,parameters)):
     eid=f'CPU_SYNTHETIC_EID_{self.eid}';self.eid+=1
     self.llm_engine.processor.process_inputs(eid,prompt,params)
     guided=params.guided_decoding;ids=[] if guided is None else re.findall(r'"(\d+)":',guided.regex);raw='BANANA' if guided is None else compact({i:'Y' for i in ids})
     if self.generated==4 and fault=='invalid_raw' and i==1:raw+='\n'
     tokens=list(prompt['prompt_token_ids']);output=[types.SimpleNamespace(text=raw,token_ids=[1001,1002],finish_reason='stop',stop_reason=None)]
     if self.generated==4 and fault=='return_prompt_second' and i==1:tokens.append(9999999)
     if self.generated==4 and fault=='return_id_second' and i==1:eid='CPU_WRONG_RETURN_EID'
     if self.generated==4 and fault=='multiple_completions' and i==1:output.append(copy.deepcopy(output[0]))
     values.append(types.SimpleNamespace(request_id=eid,prompt_token_ids=tokens,outputs=output))
    calls.append({'generate':self.generated,'returned':values})
    if self.generated==4 and fault=='short_returns':values=values[:-1];calls[-1]['returned']=values
    if self.generated==4 and fault=='extra_returns':values.append(copy.deepcopy(values[-1]));values[-1].request_id+='EXTRA'
    return values
  fake=types.ModuleType('vllm');fake.__spec__=importlib.machinery.ModuleSpec('vllm',loader=None,origin=vllm_origin);fake.LLM=Engine;fake.SamplingParams=Param
  sampling=types.ModuleType('vllm.sampling_params');sampling.GuidedDecodingParams=Param;compat=types.ModuleType('single_rank_nccl_compat');compat.ACTIVE=True;compat.__file__=str(f/'runtime_override/single_rank_nccl_compat.py')
  origsha=run.sha
  def fixture_sha(path):return 'CPU_SYNTHETIC_NO_SCIENTIFIC_FREEZE' if Path(path).name=='PREINFERENCE_FROZEN_SHA256.json' else origsha(path)
  env=observe.launch_environment(f,model);env.update(CUDA_VISIBLE_DEVICES='CPU_SYNTHETIC',PYTHONDONTWRITEBYTECODE='1')
  failure=None;stdout=io.StringIO()
  with patch.object(run,'ROOT',f),patch.object(run,'verify',lambda *args:{'CPU_SYNTHETIC_NO_GPU_GO':True}),patch.object(run,'processes',lambda *args:[]),patch.object(run,'owned',lambda *args:[]),patch.object(run,'sha',fixture_sha),patch.object(analyzer,'analyze',lambda *args:None),patch.dict(sys.modules,{'vllm':fake,'vllm.sampling_params':sampling,'single_rank_nccl_compat':compat}),patch.dict(os.environ,env),contextlib.redirect_stdout(stdout):
   try:run.main(model,'CPU_SYNTHETIC')
   except BaseException as exc:failure=type(exc).__name__+': '+str(exc)
  (f/'stdout.log').write_text(stdout.getvalue());runout=f/f'run_{model}';native=lines(runout/'engine_returns.jsonl');effective=lines(runout/'effective_requests.jsonl');canonical=lines(runout/'responses.jsonl')
  assert (failure is None)==(fault in ('none','invalid_raw')),(model,fault,failure)
  returned=[z for b in calls if 'generate' in b for z in b['returned']];saved=[z for b in native for z in b['actual_returns']]
  assert len(saved)==len(returned)
  for savedrow,actual in zip(saved,returned):assert savedrow['engine_request_id']==actual.request_id and [z['raw'] for z in savedrow['outputs']]==[z.text for z in actual.outputs]
  if fault=='none':
   assert len(canonical)==3 and len(effective)==23
   s=load(runout/'summary.json');assert s['effective_runtime_observation_receipts']['bound_science_and_control_requests']==23 and not s['effective_runtime_observation_receipts']['active_batch_unbound']
  if fault=='invalid_raw':assert len(canonical)==3 and canonical[1]['strict_valid'] is False
  if fault in ('native_class','native_dtype'):assert not effective and not native and not (runout/'DATA_START.json').exists()
  if fault in ('effective_greedy','effective_eos','effective_grammar','effective_prompt'):assert (runout/'FAILURE.json').exists() and effective and not canonical
  if fault in ('return_id_second','return_prompt_second','short_returns','extra_returns'):assert len(native)==4 and not canonical
  if fault=='multiple_completions':assert len(native[-1]['actual_returns'][1]['outputs'])==2
  cases.append({'model':model,'fault':fault,'status':'PASS','exception':failure,'actual_returned_engine_records':len(returned),'journaled_engine_records':len(saved),'effective_processed_receipts':len(effective),'canonical_science_rows':len(canonical)})
 result={'status':'PASS','test_count':len(cases),'source_sha256':{n:observe.sha(producer/n) for n in ('run_gpu.py','launch_gpu.py','runtime_observer.py','verify_effective_runtime.py','check_runtime_amendment.py')},'runtime_observation_contract_sha256':observe.sha(contract_path),'all_fixtures_CPU_SYNTHETIC':True,'GPU_or_model_loaded':False,'cases':cases}
 write(out/'RUNTIME_AMENDMENT_CPU_CHECK.json',result);print(compact({'status':'PASS','test_count':len(cases),'report':str(out/'RUNTIME_AMENDMENT_CPU_CHECK.json')}))
if __name__=='__main__':main()
