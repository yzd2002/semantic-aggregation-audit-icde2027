#!/usr/bin/env python3
"""Owned-GPU runner. Raw strings are never trimmed, repaired, or retried."""
import argparse,collections,importlib.metadata as package_metadata,os,platform,subprocess,sys,time,traceback
from common import *

def gpu_processes():
 text=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader,nounits'],text=True)
 return [{'pid':int(line.split(',')[0]),'record':line} for line in text.splitlines() if line.strip()]
def check_owned():
 import psutil
 owned={os.getpid()}|{p.pid for p in psutil.Process().children(recursive=True)}
 running=gpu_processes()
 if any(r['pid'] not in owned for r in running):raise RuntimeError('Unowned GPU process detected; no further request submitted: '+str(running))
 return running
def append(path,obj):
 with path.open('a',encoding='utf8') as f:f.write(compact(obj)+'\n');f.flush()
def synthetic_records(task,context):
 if task=='dblp_acm':
  a={'title':'Database systems','authors':'Alice Smith','venue':'Example Conference','year':'2020'}
  b={'title':'Cloud formations','authors':'Bob Jones','venue':'Weather Journal','year':'2019'}
  c={'title':'Graph algorithms','authors':'Carol Brown','venue':'Computing Journal','year':'2018'}
  pairs=[(a,a),(b,b),(b,c),(c,c)] if context=='near' else [(a,a),(c,b),(a,c),(b,b)]
  return [{'features':{'left':x,'right':y}} for x,y in pairs]
 target={'question':'Is water a liquid at room temperature?','passage':'At ordinary room temperature and pressure, water is a liquid.'}
 others=[{'question':'Can a triangle have four sides?','passage':'A triangle has exactly three sides.'},
  {'question':'Is the Moon a star?','passage':'The Moon is a natural satellite, not a star.'},
  {'question':'Do birds have feathers?','passage':'Birds have feathers.'}]
 if context=='random':others=list(reversed(others))
 return [{'features':r} for r in [target,*others]]

def main(model):
 out=ROOT/f'run_{model}';out.mkdir(exist_ok=False)
 try:
  frozen=json.loads((ROOT/'PREINFERENCE_FROZEN_SHA256.json').read_text())
  for rel,want in frozen['files'].items():
   if sha(ROOT/rel)!=want:raise RuntimeError('Frozen input/code differs: '+rel)
  import single_rank_nccl_compat as compat
  if not compat.ACTIVE or Path(compat.__file__).resolve()!=ROOT/'runtime_override/single_rank_nccl_compat.py':
   raise RuntimeError('Frozen process-local adapter is not active')
  cfg=json.loads((ROOT/'prepared/runtime_configuration.json').read_text())
  versions={p:package_metadata.version(p) for p in cfg['packages']}
  if versions!=cfg['packages'] or platform.python_version()!=cfg['python'] or sys.executable!=cfg['python_executable']:
   raise RuntimeError('Observed runtime differs from the frozen preparation')
  model_entry=json.loads((ROOT/'prepared/MODEL_FILE_SHA256.json').read_text())[model]
  model_path=Path(json.loads((ROOT/'model_locations.json').read_text())[model])
  for rel,want in model_entry['files'].items():
   if sha(model_path/rel)!=want:raise RuntimeError('Model asset changed: '+rel)
  if gpu_processes():raise RuntimeError('GPU was not idle; refusing to load')
  jobs=read_lines(ROOT/'prepared'/f'{model}_requests.jsonl')
  write_json(out/'START.json',{'pid':os.getpid(),'started_unix':time.time(),'model':model,'model_path':str(model_path),
   'planned_scientific_jobs':len(jobs),'frozen_manifest_sha256':sha(ROOT/'PREINFERENCE_FROZEN_SHA256.json'),
   'request_file_sha256':sha(ROOT/'prepared'/f'{model}_requests.jsonl'),'runtime_versions':versions,
   'launch_environment':{k:os.environ.get(k) for k in ('TMPDIR','XDG_CACHE_HOME','VLLM_CACHE_ROOT','VLLM_CONFIG_ROOT','NUMBA_CACHE_DIR',
    'TORCHINDUCTOR_CACHE_DIR','TRITON_CACHE_DIR','CUDA_CACHE_PATH','TORCH_EXTENSIONS_DIR','HF_HOME','PYTHONPATH',
    'INTENTSAT_SINGLE_RANK_NCCL_COMPAT','VLLM_USE_V1','VLLM_NO_USAGE_STATS','DO_NOT_TRACK','PYTHONDONTWRITEBYTECODE')}})
  import torch,vllm
  from vllm import LLM,SamplingParams
  from vllm.sampling_params import GuidedDecodingParams
  llm=LLM(model=str(model_path),tokenizer=str(model_path),trust_remote_code=False,dtype='bfloat16',
   tensor_parallel_size=1,max_model_len=4096,gpu_memory_utilization=.70,max_num_seqs=16,enforce_eager=True,
   enable_prefix_caching=False,disable_log_stats=True,guided_decoding_backend='xgrammar',guided_decoding_disable_fallback=True)
  tokenizer=llm.get_tokenizer()
  def parameters(rule=None):
   return SamplingParams(temperature=0,max_tokens=128,guided_decoding=None if rule is None else
    GuidedDecodingParams(regex=rule,backend='xgrammar',disable_fallback=True))
  def collect(job,response,diagnostic=False):
   if list(response.prompt_token_ids)!=job['prompt_token_ids']:raise RuntimeError('Engine prompt tokens differ from preflight: '+job['request_id'])
   answer=response.outputs[0];raw=answer.text
   parsed=parse(raw,job['output_ids']) if job.get('output_ids') else None
   meta={k:job[k] for k in ('request_id','ordinal','block','task','target_id','selected_ordinal','scope','context','repeat',
    'requested_ids','output_ids','prompt_sha256','grammar_sha256','id_map_sha256','execution_mode','source_main_request_id') if k in job}
   return {**meta,'engine_request_id':response.request_id,'raw':raw,'strict_valid':parsed is not None,
    'target_label':int(parsed[1]=='Y') if parsed is not None and 1 in parsed else None,'parsed':parsed,
    'finish_reason':answer.finish_reason,'stop_reason':answer.stop_reason,'actual_input_tokens':len(response.prompt_token_ids),
    'actual_output_tokens':len(answer.token_ids),'output_token_ids':list(answer.token_ids),
    'input_token_ids_sha256':digest(compact(list(response.prompt_token_ids))),'diagnostic':diagnostic}
  def diagnostic_call(name,requests,guided=True):
   check_owned();append(out/'diagnostic_submissions.jsonl',{'name':name,'request_ids':[j['request_id'] for j in requests],'submitted_unix':time.time()})
   values=llm.generate([{'prompt_token_ids':j['prompt_token_ids']} for j in requests],
    [parameters(j['grammar'] if guided else None) for j in requests],use_tqdm=False)
   if len(values)!=len(requests):raise RuntimeError('Diagnostic response count mismatch')
   records=[]
   for j,v in zip(requests,values):
    row=collect(j,v,True);row['wrapped_prompt']=j['wrapped_prompt'];row['grammar']=j.get('grammar');records.append(row)
    append(out/'diagnostics.jsonl',row)
   return records
  parser_checks()
  wrapped=tokenizer.apply_chat_template([{'role':'system','content':'Follow the user instruction exactly.'},
   {'role':'user','content':'Reply with only the word BANANA.'}],tokenize=False,add_generation_prompt=True)
  banana={'wrapped_prompt':wrapped,'prompt_token_ids':tokenizer.encode(wrapped,add_special_tokens=False),'prompt_sha256':digest(wrapped)}
  ordinary=diagnostic_call('ordinary_banana',[{**banana,'request_id':'diagnostic/ordinary_banana','output_ids':[]}],False)[0]
  specs=([1],[1,2,3,4],[2,3,4,1],[9,10,11],[11])
  guided=diagnostic_call('guided_banana',[{**banana,'request_id':f'diagnostic/guided_banana/{i}',
   'output_ids':ids,'grammar':grammar(ids),'grammar_sha256':digest(grammar(ids))} for i,ids in enumerate(specs)])
  negative_ok=all(r['strict_valid'] for r in guided) and 'BANANA' in ordinary['raw'] and all(parse(ordinary['raw'],ids) is None for ids in specs)
  write_json(out/'negative_controls.json',{'valid':negative_ok,'ordinary':ordinary,'guided':guided,'parser_checks':parser_checks()})
  if not negative_ok:raise RuntimeError('Incompatible-instruction guidance controls failed; scientific inference not started')
  synthetic=[]
  for task in TASKS:
   for scope in (*SCOPES,'singleton'):
    for context in (CONTEXTS if scope!='singleton' else ('none',)):
     records=synthetic_records(task,'near' if context=='none' else context)
     if scope=='singleton':records=records[:1]
     p=prepared_prompt(tokenizer,task,records,scope)
     synthetic.append({**p,'request_id':f'diagnostic/smoke/{task}/{scope}/{context}','task':task,'scope':scope,'context':context})
  smoke=diagnostic_call('synthetic_format_smoke',synthetic)
  smoke_ok=all(r['strict_valid'] for r in smoke)
  write_json(out/'synthetic_smoke.json',{'valid':smoke_ok,'records':smoke,'semantic_labels_not_used_as_gate':True})
  if not smoke_ok:raise RuntimeError('Synthetic format smoke failed; scientific inference not started')
  write_json(out/'DATA_START.json',{'time_unix':time.time(),'owned_gpu_processes':check_owned(),'planned_requests':len(jobs),
   'negative_controls_sha256':sha(out/'negative_controls.json'),'synthetic_smoke_sha256':sha(out/'synthetic_smoke.json')})
  count=valid=0;start=time.perf_counter();totals=collections.Counter()
  with (out/'responses.jsonl').open('x',encoding='utf8') as sink:
   for offset in range(0,len(jobs),64):
    owned=check_owned();batch=jobs[offset:offset+64];t=time.perf_counter()
    append(out/'block_submissions.jsonl',{'block':offset//64,'first_ordinal':offset,'last_ordinal':offset+len(batch)-1,
     'request_ids':[j['request_id'] for j in batch],'submitted_unix':time.time()})
    values=llm.generate([{'prompt_token_ids':j['prompt_token_ids']} for j in batch],[parameters(j['grammar']) for j in batch],use_tqdm=False)
    if len(values)!=len(batch):raise RuntimeError('Scientific response count mismatch')
    for job,response in zip(batch,values):
     row=collect(job,response);sink.write(compact(row)+'\n');count+=1;valid+=row['strict_valid']
     totals['input_tokens']+=row['actual_input_tokens'];totals['output_tokens']+=row['actual_output_tokens']
    sink.flush();seconds=time.perf_counter()-t
    append(out/'blocks.jsonl',{'block':offset//64,'requests':len(batch),'seconds':seconds,'completed_unix':time.time(),'owned_gpu_processes':owned})
    print(model,'block',offset//64,'returned',count,'/',len(jobs),'valid',valid,'elapsed',round(time.perf_counter()-start,2),flush=True)
  diagnostics=read_lines(out/'diagnostics.jsonl')
  diagnostic_tokens={k:sum(r['actual_'+k] for r in diagnostics) for k in ('input_tokens','output_tokens')}
  write_json(out/'summary.json',{'status':'complete','model':model,'scientific_calls':count,'valid_scientific_calls':valid,
   'invalid_scientific_calls':count-valid,'diagnostic_calls':len(diagnostics),'all_calls':count+len(diagnostics),
   'scientific_observed_tokens':dict(totals),'diagnostic_observed_tokens':diagnostic_tokens,
   'all_observed_tokens':{k:totals[k]+diagnostic_tokens[k] for k in diagnostic_tokens},
   'interleaved_execution_seconds':time.perf_counter()-start,'completed_unix':time.time(),
   'response_sha256':sha(out/'responses.jsonl'),'block_sha256':sha(out/'blocks.jsonl')})
  # Pre-inference supplement B: same loaded model and parameters; one blocking request at a time.
  serial=out/'serial';serial.mkdir(exist_ok=False)
  serial_jobs=read_lines(ROOT/'prepared'/f'{model}_serial_requests.jsonl')
  main_by_id={j['request_id']:j for j in jobs}
  for j in serial_jobs:
   original=main_by_id[j['source_main_request_id']]
   for key in ('wrapped_prompt','prompt_token_ids','grammar','output_ids','requested_ids','id_map'):
    if j[key]!=original[key]:raise RuntimeError('Serial request differs from its main counterpart: '+key)
  write_json(serial/'START.json',{'started_unix':time.time(),'planned_requests':len(serial_jobs),
   'same_loaded_engine':True,'follows_complete_main_run':True,'main_summary_sha256':sha(out/'summary.json'),
   'request_sha256':sha(ROOT/'prepared'/f'{model}_serial_requests.jsonl')})
  serial_count=serial_valid=0;serial_totals=collections.Counter();serial_start=time.perf_counter()
  with (serial/'responses.jsonl').open('x',encoding='utf8') as sink:
   for job in serial_jobs:
    owned=check_owned();t=time.perf_counter()
    append(serial/'block_submissions.jsonl',{'block':job['block'],'first_ordinal':job['ordinal'],'last_ordinal':job['ordinal'],
     'request_ids':[job['request_id']],'submitted_unix':time.time()})
    values=llm.generate([{'prompt_token_ids':job['prompt_token_ids']}],[parameters(job['grammar'])],use_tqdm=False)
    if len(values)!=1:raise RuntimeError('Serial generate returned a count other than one')
    row=collect(job,values[0]);sink.write(compact(row)+'\n');sink.flush()
    serial_count+=1;serial_valid+=row['strict_valid']
    serial_totals['input_tokens']+=row['actual_input_tokens'];serial_totals['output_tokens']+=row['actual_output_tokens']
    append(serial/'blocks.jsonl',{'block':job['block'],'requests':1,'seconds':time.perf_counter()-t,'completed_unix':time.time(),'owned_gpu_processes':owned})
    if serial_count%32==0 or serial_count==len(serial_jobs):
     print(model,'serial returned',serial_count,'/',len(serial_jobs),'valid',serial_valid,'elapsed',round(time.perf_counter()-serial_start,2),flush=True)
  write_json(serial/'summary.json',{'status':'complete','model':model,'scientific_calls':serial_count,'valid_scientific_calls':serial_valid,
   'invalid_scientific_calls':serial_count-serial_valid,'observed_tokens':dict(serial_totals),
   'serial_execution_seconds':time.perf_counter()-serial_start,'completed_unix':time.time(),
   'response_sha256':sha(serial/'responses.jsonl'),'block_sha256':sha(serial/'blocks.jsonl')})
  write_json(out/'ALL_WORK_COMPLETE.json',{'status':'complete','main_calls':count,'serial_calls':serial_count,'diagnostic_calls':len(diagnostics),
   'all_calls':count+serial_count+len(diagnostics),'all_observed_tokens':{k:totals[k]+serial_totals[k]+diagnostic_tokens[k] for k in diagnostic_tokens},
   'main_summary_sha256':sha(out/'summary.json'),'serial_summary_sha256':sha(serial/'summary.json'),'completed_unix':time.time()})
 except BaseException as error:
  write_json(out/'FAILURE.json',{'time_unix':time.time(),'type':type(error).__name__,'message':str(error),'traceback':traceback.format_exc()})
  raise

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--model',choices=MODELS,required=True);main(parser.parse_args().model)
