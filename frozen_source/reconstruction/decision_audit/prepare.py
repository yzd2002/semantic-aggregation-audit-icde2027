#!/usr/bin/env python3
"""CPU preparation of both complete phases; no new model judgments or gold selection."""
import collections,csv,importlib.metadata as package_metadata,os,platform,random,subprocess,sys,time
from common import *
for key,rel in {'TMPDIR':'tmp','XDG_CACHE_HOME':'cache','HF_HOME':'cache/hf','TORCH_HOME':'cache/torch',
 'VLLM_CACHE_ROOT':'cache/vllm','NUMBA_CACHE_DIR':'cache/numba','TORCHINDUCTOR_CACHE_DIR':'cache/inductor',
 'TRITON_CACHE_DIR':'cache/triton','CUDA_CACHE_PATH':'cache/cuda','TORCH_EXTENSIONS_DIR':'cache/extensions',
 'FLASHINFER_WORKSPACE_BASE':'cache/flashinfer_base'}.items():
 path=ROOT/rel;path.mkdir(parents=True,exist_ok=True);os.environ[key]=str(path)
os.environ.update(PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',PIP_DISABLE_PIP_VERSION_CHECK='1')
import numpy as np
from lexical import LexicalIndex

def relations_from_reserves():
 reserves=read_lines(ROOT/'sources/reserves.jsonl');rows=[]
 for task in TASKS:
  taskrows=sorted([r for r in reserves if r['task']==task],key=lambda r:r['reserve_ordinal'])
  assert [r['reserve_ordinal'] for r in taskrows]==list(range(256))
  for item in taskrows:
   rec=item['record'];ordinal=item['reserve_ordinal'];phase='dev' if ordinal<128 else 'test'
   assert rec['task']==task and digest(feature_string(rec))==rec['feature_sha256']
   rows.append({'task':task,'phase':phase,'source_id':rec['source_id'],'source_split':rec['source_split'],
    'source_index':rec['source_index'],'reserve_ordinal':ordinal,'occurrence_id':ordinal%128+1,
    'features':rec['features'],'feature_sha256':rec['feature_sha256'],'old_feature_token_lengths':rec['token_lengths']})
 return rows

def partitions(rows):
 outputs=[];index_audits={};random_bases={}
 for task in TASKS:
  bank=read_lines(ROOT/'sources'/f'{task}_background_bank.jsonl');index=LexicalIndex(bank);index_audits[task]=index.audit
  for phase in PHASES:
   records=sorted([r for r in rows if r['task']==task and r['phase']==phase],key=lambda r:r['occurrence_id'])
   assert [r['occurrence_id'] for r in records]==list(range(1,129))
   matrix=index.encode_counts([index.terms(feature_string(r)) for r in records]);cosine=(matrix@matrix.T).toarray()
   seed=digest(compact([SEEDS['grouping'],'random',task,phase]));shuffled=list(range(1,129));random.Random(int(seed,16)).shuffle(shuffled)
   random_bases[task+'/'+phase]={'seed_sha256':seed,'permutation':shuffled}
   ranks={i:digest(compact([SEEDS['grouping'],'diversity-anchor',task,phase,i])) for i in range(1,129)}
   for k in KS:
    for grouping in GROUPS:
     if grouping=='random':batches=[shuffled[start:start+k] for start in range(0,128,k)]
     else:
      available=set(range(1,129));batches=[]
      while available:
       anchor=min(available,key=lambda i:(ranks[i],i));batch=[anchor];available.remove(anchor)
       while available and len(batch)<k:
        chosen=min(available,key=lambda i:(max(float(cosine[i-1,j-1]) for j in batch),i))
        batch.append(chosen);available.remove(chosen)
       batches.append(batch)
     assert sorted(i for batch in batches for i in batch)==list(range(1,129)) and all(len(batch)==k for batch in batches)
     outputs.append({'task':task,'phase':phase,'k':k,'grouping':grouping,'batches':batches,
      'batch_membership_sha256':digest(compact(batches)),
      'mean_within_batch_cosine':float(np.mean([cosine[i-1,j-1] for batch in batches for index_i,i in enumerate(batch) for j in batch[index_i+1:]]))})
 return outputs,index_audits,random_bases

def get_gold(rows):
 lookup={}
 with (ROOT/'sources/dblp_acm/test.csv').open(newline='',encoding='utf8') as f:
  for i,r in enumerate(csv.DictReader(f)):
   assert r['label'] in ('0','1');lookup['dblp_acm',f'test:{i:05d}:L{r["ltable_id"]}:R{r["rtable_id"]}']=int(r['label'])
 for r in read_lines(ROOT/'sources/boolq/validation.jsonl'):
  assert type(r['answer']) is bool;lookup['boolq',f'validation:{r["row_idx"]:05d}']=int(r['answer'])
 return [{'task':r['task'],'phase':r['phase'],'source_id':r['source_id'],'occurrence_id':r['occurrence_id'],'gold':lookup[r['task'],r['source_id']]} for r in rows]

def schedules(partition_rows):
 all_jobs={}
 for phase in PHASES:
  jobs=[]
  for task in TASKS:
   for k in KS:
    for grouping in GROUPS:
     partition=next(p for p in partition_rows if (p['task'],p['phase'],p['k'],p['grouping'])==(task,phase,k,grouping))
     for kind in TYPES:
      config=config_id(k,grouping,kind)
      for repeat in repeats(phase,kind):
       for batch_index,ids in enumerate(partition['batches']):
        jobs.append({'phase':phase,'task':task,'configuration_id':config,'k':k,'grouping':grouping,'execution_type':kind,
         'repeat':repeat,'batch_index':batch_index,'occurrence_ids':sorted(ids),'batch_membership_sha256':partition['batch_membership_sha256'],
         'request_id':f'{phase}/{task}/{config}/r{repeat}/b{batch_index:03d}'})
   for repeat in (1,2):
    for i in range(1,129):
     jobs.append({'phase':phase,'task':task,'configuration_id':'singleton','k':1,'grouping':'none','execution_type':'S',
      'repeat':repeat,'batch_index':i-1,'occurrence_ids':[i],'batch_membership_sha256':None,
      'request_id':f'{phase}/{task}/singleton/r{repeat}/b{i-1:03d}'})
  random.Random(SEEDS[phase+'_schedule']).shuffle(jobs)
  for ordinal,j in enumerate(jobs):j.update(ordinal=ordinal,block=ordinal//64)
  all_jobs[phase]=jobs
 return all_jobs

def budget_audit(jobs):
 result={}
 for phase,phasejobs in jobs.items():
  result[phase]={}
  for task in TASKS:
   current=[j for j in phasejobs if j['task']==task]
   expected=2048 if phase=='dev' else 1600
   assert len(current)==expected
   result[phase][task]={'research_calls':len(current),'requested_item_judgments':sum(len(j['occurrence_ids']) for j in current)}
   if phase=='dev':
    selectors={
     'Input':[j for j in current if j['execution_type']=='S' or (j['execution_type'] in ('C','I') and j['repeat']<=2)],
     'Output':[j for j in current if j['execution_type']=='S' or (j['execution_type'] in ('C','O') and j['repeat']<=2)],
     'Repeat':[j for j in current if j['execution_type'] in ('C','S')]}
    result[phase][task]['selectors']={}
    for name,subset in selectors.items():
     assert len(subset)==1152 and sum(len(j['occurrence_ids']) for j in subset)==3328
     result[phase][task]['selectors'][name]={'calls':len(subset),'requested_item_judgments':sum(len(j['occurrence_ids']) for j in subset),
      'request_ids_sha256':digest(compact([j['request_id'] for j in subset]))}
 assert sum(len(v) for v in jobs.values())*2==14592
 return result

def main():
 from transformers import AutoTokenizer
 inputs=ROOT/'inputs';prepared=ROOT/'prepared'
 if (inputs/'relations.jsonl').exists():raise RuntimeError('Refusing to overwrite existing prepared inputs')
 rows=relations_from_reserves();parts,index_audits,random_bases=partitions(rows);jobs=schedules(parts)
 write_lines(inputs/'relations.jsonl',rows);write_lines(inputs/'partitions.jsonl',parts)
 write_json(inputs/'LEXICAL_INDEX_AUDIT.json',index_audits);write_json(inputs/'RANDOM_BASE_PERMUTATIONS.json',random_bases)
 feature_selection_hash=digest(compact({'relations':rows,'partitions':parts,'schedules':jobs}))
 # Gold is attached separately only after every feature-only physical plan is fixed.
 gold=get_gold(rows);write_lines(inputs/'gold.jsonl',gold)
 assert digest(compact({'relations':rows,'partitions':parts,'schedules':jobs}))==feature_selection_hash
 write_json(inputs/'FEATURE_ONLY_PLAN_HASH.json',{'sha256':feature_selection_hash,'gold_used_for_plan_construction':False})
 write_json(inputs/'SOURCE_FILE_SHA256.json',{str(p.relative_to(ROOT)):sha(p) for p in sorted((ROOT/'sources').rglob('*')) if p.is_file()})
 for phase,phasejobs in jobs.items():
  write_lines(prepared/f'{phase}_schedule.jsonl',phasejobs)
  write_json(prepared/f'{phase}_block_membership.json',[[j['request_id'] for j in phasejobs[start:start+64]] for start in range(0,len(phasejobs),64)])
 write_json(prepared/'ROLE_BUDGETS.json',budget_audit(jobs));write_json(prepared/'PARSER_CHECKS.json',{'status':'PASS','checks':parser_checks()})
 lookup={(r['task'],r['phase'],r['occurrence_id']):r for r in rows}
 locations=json.loads((ROOT/'model_locations.json').read_text());model_hashes={};preflight={}
 for model in MODELS:
  tok=AutoTokenizer.from_pretrained(locations[model],local_files_only=True,trust_remote_code=False)
  for r in rows:
   actual=len(tok.encode(feature_string(r),add_special_tokens=False))
   assert actual==r['old_feature_token_lengths'][model] and actual<=512
  preflight[model]={}
  for phase,phasejobs in jobs.items():
   output=[];max_output=0
   for j in phasejobs:
    records=[lookup[j['task'],phase,i] for i in j['occurrence_ids']]
    p=prepare_prompt(tok,j['task'],records,j['execution_type'])
    assert p['prompt_tokens_preflight']+128<=4096
    # All grammar outputs use ASCII, and even single-character emission is below the cap.
    longest=compact({str(i):'Y' for i in p['output_ids']});assert len(longest)+1<=128
    output_token_lengths=[len(tok.encode(compact({str(i):value for i in p['output_ids']}),add_special_tokens=False)) for value in ('Y','N')]
    max_output=max(max_output,*output_token_lengths)
    output.append({**j,**p,'canonical_output_token_lengths_allY_allN':output_token_lengths,'max_ascii_output_characters':len(longest)})
   write_lines(prepared/f'{model}_{phase}_requests.jsonl',output)
   bykey={(j['task'],j['k'],j['grouping'],j['execution_type'],j['repeat'],j['batch_index']):j for j in output}
   for j in output:
    if j['execution_type']=='O':
     c=bykey[j['task'],j['k'],j['grouping'],'C',j['repeat'],j['batch_index']]
     assert j['wrapped_prompt']==c['wrapped_prompt'] and j['prompt_token_ids']==c['prompt_token_ids']
    if j['execution_type']=='I':
     c=bykey[j['task'],j['k'],j['grouping'],'C',j['repeat'],j['batch_index']]
     assert j['grammar']==c['grammar'] and j['requested_ids']==c['requested_ids']
   preflight[model][phase]={'requests':len(output),'max_prompt_plus_cap':max(j['prompt_tokens_preflight']+128 for j in output),
    'max_canonical_output_tokens':max_output,'request_sha256':sha(prepared/f'{model}_{phase}_requests.jsonl'),'C_O_prompt_equality':'PASS','C_I_grammar_equality':'PASS'}
  model_hashes[model]={'path':locations[model],'files':{p.name:sha(p) for p in sorted(Path(locations[model]).iterdir()) if p.is_file()}}
  print(model,'both phase preflight PASS',flush=True)
 write_json(prepared/'MODEL_FILE_SHA256.json',model_hashes);write_json(prepared/'BOTH_MODEL_CONTEXT_PREFLIGHT.json',preflight)
 checks=subprocess.run([sys.executable,'-m','pip','check'],capture_output=True,text=True)
 (prepared/'pip_check.log').write_text(checks.stdout+checks.stderr)
 config={'python':platform.python_version(),'python_executable':sys.executable,'packages':{p:package_metadata.version(p) for p in PACKAGES},
  'pip_check_exit':checks.returncode,'environment_status':'Inherited dependency conflicts, explicit same singleton NCCL adapter; not clean-machine replication',
  'dtype':'bfloat16','temperature':0,'max_model_len':4096,'max_tokens':128,'max_num_seqs':16,'tensor_parallel_size':1,
  'enforce_eager':True,'prefix_caching':False,'guided_backend':'xgrammar','disable_guidance_fallback':True,'gpu_memory_utilization':.70,
  'block_requests':64,'seeds':SEEDS,'FLASHINFER_WORKSPACE_BASE':str(ROOT/'cache/flashinfer_base'),
  'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv,noheader'],text=True)}
 write_json(prepared/'runtime_configuration.json',config)
 print('Both phases prepared; no inference. Final code freeze and independent review still required.',flush=True)

if __name__=='__main__':main()
