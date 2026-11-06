#!/usr/bin/env python3
"""One owned GPU, one phase/model, no retries or answer-based changes."""
import argparse,collections,importlib.metadata as package_metadata,os,platform,subprocess,sys,time,traceback
from common import *

def append(path,record):
 with path.open('a',encoding='utf8') as f:f.write(compact(record)+'\n');f.flush()
def gpu_processes():
 raw=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader,nounits'],text=True)
 return [{'pid':int(line.split(',')[0]),'record':line} for line in raw.splitlines() if line.strip()]
def check_owned():
 import psutil
 owned={os.getpid()}|{p.pid for p in psutil.Process().children(recursive=True)};running=gpu_processes()
 if any(r['pid'] not in owned for r in running):raise RuntimeError('Unowned GPU process; no further submission: '+str(running))
 return running
def synthetics(task):
 identifiers=[1,8,9,10,11,64,127,128];records=[]
 for index,identifier in enumerate(identifiers):
  if task=='dblp_acm':
   a={'title':'Example database study '+str(index),'authors':'Alice Example','venue':'Illustrative Conference','year':'2020'}
   b=a.copy() if index%2==0 else {'title':'Distinct weather study '+str(index),'authors':'Bob Example','venue':'Illustrative Journal','year':'2019'}
   features={'left':a,'right':b}
  else:features={'question':'Is the lamp on?','passage':'The lamp is '+('on.' if index%2==0 else 'off.')}
  records.append({'task':task,'phase':'diagnostic','source_id':'synthetic:'+str(index),'reserve_ordinal':None,
   'occurrence_id':identifier,'features':features,'feature_sha256':digest(compact(features))})
 return records

def main(phase,model):
 out=ROOT/f'run_{phase}_{model}';out.mkdir(exist_ok=False)
 try:
  freeze=json.loads((ROOT/'PREINFERENCE_FROZEN_SHA256.json').read_text())
  for rel,want in freeze['files'].items():
   if sha(ROOT/rel)!=want:raise RuntimeError('Frozen scientific file differs: '+rel)
  release=None
  if phase=='test':
   from test_gate import verify_test_release
   release=verify_test_release(ROOT)
  import single_rank_nccl_compat as compat
  if not compat.ACTIVE or Path(compat.__file__).resolve()!=ROOT/'runtime_override/single_rank_nccl_compat.py':raise RuntimeError('Frozen process-local adapter not active')
  config=json.loads((ROOT/'prepared/runtime_configuration.json').read_text())
  current={p:package_metadata.version(p) for p in config['packages']}
  if current!=config['packages'] or platform.python_version()!=config['python'] or sys.executable!=config['python_executable']:raise RuntimeError('Runtime version mismatch')
  if os.environ.get('FLASHINFER_WORKSPACE_BASE')!=config['FLASHINFER_WORKSPACE_BASE']:raise RuntimeError('FlashInfer cache path differs from frozen config')
  locations=json.loads((ROOT/'model_locations.json').read_text());model_path=Path(locations[model])
  model_manifest=json.loads((ROOT/'prepared/MODEL_FILE_SHA256.json').read_text())[model]
  for rel,want in model_manifest['files'].items():
   if sha(model_path/rel)!=want:raise RuntimeError('Model asset changed: '+rel)
  if gpu_processes():raise RuntimeError('GPU not idle before loading')
  jobs=read_lines(ROOT/'prepared'/f'{model}_{phase}_requests.jsonl')
  write_json(out/'START.json',{'pid':os.getpid(),'phase':phase,'model':model,'started_unix':time.time(),'planned_scientific_jobs':len(jobs),
   'scientific_freeze_sha256':sha(ROOT/'PREINFERENCE_FROZEN_SHA256.json'),'request_sha256':sha(ROOT/'prepared'/f'{model}_{phase}_requests.jsonl'),
   'test_release':release,'runtime_versions':current,'launch_environment':{k:os.environ.get(k) for k in
    ('TMPDIR','XDG_CACHE_HOME','XDG_CONFIG_HOME','VLLM_CACHE_ROOT','VLLM_CONFIG_ROOT','HF_HOME','TORCH_HOME','NUMBA_CACHE_DIR',
     'TORCHINDUCTOR_CACHE_DIR','TRITON_CACHE_DIR','CUDA_CACHE_PATH','TORCH_EXTENSIONS_DIR','FLASHINFER_WORKSPACE_BASE',
     'PYTHONPATH','INTENTSAT_SINGLE_RANK_NCCL_COMPAT','VLLM_USE_V1','VLLM_NO_USAGE_STATS','PYTHONDONTWRITEBYTECODE')}})
  from vllm import LLM,SamplingParams
  from vllm.sampling_params import GuidedDecodingParams
  llm=LLM(model=str(model_path),tokenizer=str(model_path),trust_remote_code=False,dtype='bfloat16',tensor_parallel_size=1,
   max_model_len=4096,gpu_memory_utilization=.70,max_num_seqs=16,enforce_eager=True,enable_prefix_caching=False,
   disable_log_stats=True,guided_decoding_backend='xgrammar',guided_decoding_disable_fallback=True)
  tokenizer=llm.get_tokenizer()
  def parameters(rule=None):return SamplingParams(temperature=0,max_tokens=128,guided_decoding=None if rule is None else
   GuidedDecodingParams(regex=rule,backend='xgrammar',disable_fallback=True))
  def collect(job,response,diagnostic=False):
   if list(response.prompt_token_ids)!=job['prompt_token_ids']:raise RuntimeError('Actual prompt differs from preflight')
   answer=response.outputs[0];parsed=parse(answer.text,job['output_ids']) if job.get('output_ids') else None
   meta={k:job[k] for k in ('phase','task','configuration_id','k','grouping','execution_type','repeat','batch_index','occurrence_ids',
    'ordinal','block','request_id','output_ids','requested_ids','prompt_sha256','grammar_sha256','id_map_sha256') if k in job}
   return {**meta,'engine_request_id':response.request_id,'raw':answer.text,'strict_valid':parsed is not None,'labels':parsed,
    'finish_reason':answer.finish_reason,'stop_reason':answer.stop_reason,'actual_input_tokens':len(response.prompt_token_ids),
    'actual_output_tokens':len(answer.token_ids),'output_token_ids':list(answer.token_ids),
    'input_token_ids_sha256':digest(compact(list(response.prompt_token_ids))),'diagnostic':diagnostic}
  def diagnostic(name,requests,guided=True):
   check_owned();append(out/'diagnostic_submissions.jsonl',{'name':name,'request_ids':[r['request_id'] for r in requests],'submitted_unix':time.time()})
   values=llm.generate([{'prompt_token_ids':r['prompt_token_ids']} for r in requests],[parameters(r['grammar'] if guided else None) for r in requests],use_tqdm=False)
   if len(values)!=len(requests):raise RuntimeError('Diagnostic response count mismatch')
   records=[]
   for j,v in zip(requests,values):
    row=collect(j,v,True);row.update(wrapped_prompt=j['wrapped_prompt'],grammar=j.get('grammar'));append(out/'diagnostics.jsonl',row);records.append(row)
   return records
  parser_checks()
  wrapped=tokenizer.apply_chat_template([{'role':'system','content':'Follow the user instruction exactly.'},{'role':'user','content':'Reply with only the word BANANA.'}],tokenize=False,add_generation_prompt=True)
  base={'wrapped_prompt':wrapped,'prompt_token_ids':tokenizer.encode(wrapped,add_special_tokens=False),'prompt_sha256':digest(wrapped)}
  ordinary=diagnostic('ordinary_banana',[{**base,'request_id':'diagnostic/ordinary_banana','output_ids':[]}],False)[0]
  identifiers=[1,8,9,10,11,64,127,128]
  specs=([128],[9,10,127,128],[128,127,10,9],identifiers,list(reversed(identifiers)))
  guided=diagnostic('guided_banana',[{**base,'request_id':f'diagnostic/guided_banana/{i}','output_ids':ids,
   'grammar':grammar(ids),'grammar_sha256':digest(grammar(ids))} for i,ids in enumerate(specs)])
  okay='BANANA' in ordinary['raw'] and all(parse(ordinary['raw'],ids) is None for ids in specs) and all(r['strict_valid'] for r in guided)
  write_json(out/'negative_controls.json',{'valid':okay,'ordinary':ordinary,'guided':guided,'parser_checks':parser_checks()})
  if not okay:raise RuntimeError('Guidance negative controls failed before scientific execution')
  smoke_jobs=[]
  for task in TASKS:
   records=synthetics(task)
   for k,kind in [(2,'C'),(2,'I'),(2,'O'),(4,'C'),(4,'I'),(4,'O'),(8,'C')]:
    selected=records[-k:];p=prepare_prompt(tokenizer,task,selected,kind)
    smoke_jobs.append({**p,'request_id':f'diagnostic/smoke/{task}/k{k}/{kind}','task':task,'execution_type':kind})
  smoke=diagnostic('synthetic_format_smoke',smoke_jobs);okay=all(r['strict_valid'] for r in smoke)
  write_json(out/'synthetic_smoke.json',{'valid':okay,'records':smoke,'semantic_labels_not_used_as_gate':True})
  if not okay:raise RuntimeError('Format smoke failed before scientific execution')
  write_json(out/'DATA_START.json',{'phase':phase,'model':model,'time_unix':time.time(),'planned_requests':len(jobs),'owned_gpu_processes':check_owned(),
   'negative_control_sha256':sha(out/'negative_controls.json'),'synthetic_smoke_sha256':sha(out/'synthetic_smoke.json'),'test_release':release})
  count=valid=0;totals=collections.Counter();started=time.perf_counter()
  with (out/'responses.jsonl').open('x',encoding='utf8') as f:
   for offset in range(0,len(jobs),64):
    owned=check_owned();batch=jobs[offset:offset+64];t=time.perf_counter()
    append(out/'block_submissions.jsonl',{'block':offset//64,'first_ordinal':offset,'last_ordinal':offset+len(batch)-1,
     'request_ids':[j['request_id'] for j in batch],'submitted_unix':time.time()})
    responses=llm.generate([{'prompt_token_ids':j['prompt_token_ids']} for j in batch],[parameters(j['grammar']) for j in batch],use_tqdm=False)
    if len(responses)!=len(batch):raise RuntimeError('Scientific response count mismatch')
    for j,v in zip(batch,responses):
     row=collect(j,v);f.write(compact(row)+'\n');count+=1;valid+=row['strict_valid']
     totals['input_tokens']+=row['actual_input_tokens'];totals['output_tokens']+=row['actual_output_tokens']
    f.flush();append(out/'blocks.jsonl',{'block':offset//64,'requests':len(batch),'seconds':time.perf_counter()-t,'completed_unix':time.time(),'owned_gpu_processes':owned})
    print(phase,model,'block',offset//64,'returned',count,'/',len(jobs),'valid',valid,flush=True)
  diagnostics=read_lines(out/'diagnostics.jsonl');assert len(diagnostics)==20
  diag_tokens={k:sum(r['actual_'+k] for r in diagnostics) for k in ('input_tokens','output_tokens')}
  write_json(out/'summary.json',{'status':'complete','phase':phase,'model':model,'scientific_calls':count,'valid_scientific_calls':valid,
   'invalid_scientific_calls':count-valid,'diagnostic_calls':20,'all_calls':count+20,'scientific_observed_tokens':dict(totals),
   'diagnostic_observed_tokens':diag_tokens,'all_observed_tokens':{k:totals[k]+diag_tokens[k] for k in diag_tokens},
   'interleaved_seconds':time.perf_counter()-started,'completed_unix':time.time(),'response_sha256':sha(out/'responses.jsonl'),
   'block_sha256':sha(out/'blocks.jsonl'),'test_release':release})
  if phase=='dev' and all((ROOT/f'run_dev_{m}/summary.json').is_file() for m in MODELS):
   from lock_policies import main as lock_completed_development
   lock_completed_development(ROOT)
  if phase=='test':
   from analyze_test import analyze
   analyze(ROOT,model,ROOT/f'analysis_test_{model}')
 except BaseException as error:
  write_json(out/'FAILURE.json',{'phase':phase,'model':model,'time_unix':time.time(),'type':type(error).__name__,'message':str(error),'traceback':traceback.format_exc()})
  raise

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--phase',choices=PHASES,required=True);p.add_argument('--model',choices=MODELS,required=True)
 a=p.parse_args();main(a.phase,a.model)
