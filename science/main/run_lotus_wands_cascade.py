#!/usr/bin/env python3
"""Run a real, traced LOTUS helper-LM cascade on the exposed WANDS-52 sample."""
import argparse,dataclasses,hashlib,importlib,json,os,random,time,zipfile
from pathlib import Path
from urllib.parse import urlparse
import numpy as np,pandas as pd,psutil
import lotus
from lotus.models import LM
from lotus.types import CascadeArgs,ProxyModel,SerializationFormat
from lotus.templates import task_instructions
from transformers import AutoTokenizer
from importlib.metadata import version
from run_compact_prompt_ablation import metrics
from analyze_order_fallback_validation import paired
HERE=Path(__file__).resolve().parent
INSTRUCTION=('The {product} fully matches the shopper query {query}, including the requested product type '
             'and every stated essential modifier; use only supplied information and do not assume unknown properties.')
SEED_SHUFFLE=20271305;SEED_CALIBRATION=20271231;SEED_BOOT=20271306
WHEEL_HASH='4803bcba7a30d50095b0c6fc5036942ebd1cfceb42a3f907454892c11171fa89'
def lines(p):
 with p.open(encoding='utf8') as f:return [json.loads(x) for x in f]
def digest_bytes(x):return hashlib.sha256(x).hexdigest()
def message_hash(m):return digest_bytes(json.dumps(m,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode())
def safe_endpoint(x):
 u=urlparse(x)
 if u.scheme!='http' or u.hostname not in ('localhost','127.0.0.1') or u.path.rstrip('/')!='/v1' or u.username or u.password:raise ValueError('Only unauthenticated loopback /v1 endpoints are allowed')
 return x.rstrip('/')
def check_gpu_ownership():
 import subprocess
 parents=[]
 for p in psutil.process_iter(['pid','cmdline']):
  a=p.info['cmdline'] or []
  if '-m' in a and 'vllm.entrypoints.openai.api_server' in a and '--served-model-name' in a:
   if a[a.index('--served-model-name')+1] in ('intentsat-oracle','intentsat-helper'):parents.append(p)
 if len(parents)!=2:raise RuntimeError('Expected exactly two own benchmark servers')
 owned={p.pid for p in parents}
 for p in parents:owned.update(c.pid for c in p.children(recursive=True))
 used={int(x.strip()) for x in subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).splitlines() if x.strip().isdigit()}
 if not used or not used<=owned:raise RuntimeError('Unexpected GPU process; stopping before another measured phase')
def check_wheel(wheel):
 if version('lotus-ai')!='1.2.4' or digest_bytes(wheel.read_bytes())!=WHEEL_HASH:raise ValueError('Unexpected LOTUS wheel or installed version')
 site=Path(lotus.__file__).resolve().parent.parent
 with zipfile.ZipFile(wheel) as z:
  names=[n for n in z.namelist() if n.startswith('lotus/') and n.endswith('.py')]
  if len(names)!=69 or any(not (site/n).exists() or digest_bytes((site/n).read_bytes())!=digest_bytes(z.read(n)) for n in names):raise ValueError('Installed source does not match recorded wheel')
 return len(names)
def parser(text):
 answer=text.split('Answer:',1)[1].strip() if 'Answer:' in text else text.strip()
 low=answer.lower()
 return True if 'true' in low else False if 'false' in low else True
def parse_audit(texts):
 out={'neither':0,'both':0,'empty':0}
 for text in texts:
  answer=text.split('Answer:',1)[1].strip() if 'Answer:' in text else text.strip()
  lo=answer.lower();t='true' in lo;f='false' in lo
  out['neither']+=int(not t and not f);out['both']+=int(t and f);out['empty']+=int(not answer)
 return out
class CapturedLM(LM):
 def __init__(self,*args,trace_file,role,**kwargs):
  super().__init__(*args,**kwargs);self.trace_file=trace_file;self.role=role;self.phase='unset';self.batches={}
 def _update_stats(self,response,is_cached=False):
  u=getattr(response,'usage',None)
  self._pending_usage.append(None if not u else {k:int(getattr(u,k) or 0) for k in ('prompt_tokens','completion_tokens','total_tokens')})
  return super()._update_stats(response,is_cached=is_cached)
 def __call__(self,messages,**kwargs):
  batch=self.batches.get(self.phase,0);self.batches[self.phase]=batch+1;self._pending_usage=[]
  t=time.perf_counter();result=super().__call__(messages,**kwargs);duration=time.perf_counter()-t
  assert len(result.outputs)==len(messages) and len(self._pending_usage)==len(messages)
  if result.logprobs is not None:assert len(result.logprobs)==len(messages)
  with self.trace_file.open('a',encoding='utf8') as f:
   for i,(msg,out) in enumerate(zip(messages,result.outputs)):
    probs=None if result.logprobs is None else [p.model_dump(mode='json') for p in result.logprobs[i]]
    event={'phase':self.phase,'role':self.role,'batch':batch,'ordinal':i,'messages':msg,'prompt_sha256':message_hash(msg),
           'output':out,'logprobs':probs,'usage':self._pending_usage[i],'batch_seconds':duration}
    f.write(json.dumps(event,ensure_ascii=False,allow_nan=False)+'\n')
  return result
def usage(m):
 x=dataclasses.asdict(m.stats)
 return {'physical':{k:int(x['physical_usage'][k]) for k in ('prompt_tokens','completion_tokens','total_tokens')},
         'virtual':{k:int(x['virtual_usage'][k]) for k in ('prompt_tokens','completion_tokens','total_tokens')},
         'cache_hits':int(x['cache_hits']),'operator_cache_hits':int(x['operator_cache_hits'])}
def difference(a,b):
 return {k:{x:b[k][x]-a[k][x] for x in a[k]} for k in ('physical','virtual')}|{'cache_hits':b['cache_hits']-a['cache_hits'],'operator_cache_hits':b['operator_cache_hits']-a['operator_cache_hits']}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--main-endpoint',required=True);ap.add_argument('--helper-endpoint',required=True)
 ap.add_argument('--main-model',required=True);ap.add_argument('--helper-model',required=True);ap.add_argument('--wheel',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
 main_endpoint=safe_endpoint(a.main_endpoint);helper_endpoint=safe_endpoint(a.helper_endpoint)
 checked_files=check_wheel(a.wheel);check_gpu_ownership();assert lotus.settings.serialization_format==SerializationFormat.DEFAULT
 rows=lines(HERE/'wands_selective_execution_selection/inputs.jsonl');previous=lines(HERE/'run_wands_selective_execution_qwen_v1/selective_outputs.jsonl')
 assert len(rows)==len(previous)==52 and all(len(r['items'])==32 and r['query_id']==p['query_id'] and len(p['pred'])==32 for r,p in zip(rows,previous))
 flat=[{'query_id':r['query_id'],'candidate_id':item['id'],'pool_index':i,'candidate_index':j,'query':r['query'],'product':item['text']}
       for i,r in enumerate(rows) for j,item in enumerate(r['items'])]
 n=len(flat);assert n==1664
 perm=np.random.default_rng(SEED_SHUFFLE).permutation(n);shuffled=[flat[int(i)] for i in perm];df=pd.DataFrame(shuffled)
 a.out.mkdir(parents=True,exist_ok=False)
 with (a.out/'shuffled_rows.jsonl').open('w',encoding='utf8') as f:
  for r in shuffled:f.write(json.dumps(r,ensure_ascii=False)+'\n')
 tok_main=AutoTokenizer.from_pretrained(a.main_model,local_files_only=True);tok_helper=AutoTokenizer.from_pretrained(a.helper_model,local_files_only=True)
 for tok in (tok_main,tok_helper):
  assert all(len(tok.encode(x,add_special_tokens=False))==1 for x in ('True','False'))
 trace=a.out/'model_requests.jsonl'
 def make(role,endpoint,tok,served):return CapturedLM(model='hosted_vllm/'+served,api_base=endpoint,api_key='local',tokenizer=tok,max_ctx_len=4096,max_tokens=16,max_batch_size=16,temperature=0,trace_file=trace,role=role)
 main_lm=make('main',main_endpoint,tok_main,'intentsat-oracle');helper_lm=make('helper',helper_endpoint,tok_helper,'intentsat-helper')
 lotus.settings.configure(lm=main_lm,helper_lm=helper_lm,enable_cache=False)
 cols=lotus.nl_expression.parse_cols(INSTRUCTION);formatted=lotus.nl_expression.nle2str(INSTRUCTION,cols)
 docs=task_instructions.df2multimodal_info(df,cols)
 maxima={};
 for name,tok,model in [('main',tok_main,main_lm),('helper',tok_helper,helper_lm)]:
  lengths=[len(tok.apply_chat_template(task_instructions.filter_formatter(model,doc,formatted),tokenize=True,add_generation_prompt=True)) for doc in docs]
  maxima[name]=max(lengths)+16
  if maxima[name]>4096:raise RuntimeError(name+' exceeds context with output cap')
 preflight={'input_sha256':digest_bytes((HERE/'wands_selective_execution_selection/inputs.jsonl').read_bytes()),
  'prior_selective_sha256':digest_bytes((HERE/'run_wands_selective_execution_qwen_v1/selective_outputs.jsonl').read_bytes()),
  'rows':n,'pools':len(rows),'shuffle_seed':SEED_SHUFFLE,'lotus_version':version('lotus-ai'),'wheel_sha256':WHEEL_HASH,
  'installed_lotus_python_files_verified':checked_files,'columns':cols,'instruction':INSTRUCTION,'formatted_instruction':formatted,
  'max_prompt_plus_output_tokens':maxima,'model_ids':{'main':'Qwen3-4B-Instruct-2507','helper':'SmolLM2-1.7B-Instruct'},
  'model_versions':{k:version(k) for k in ('lotus-ai','litellm','pandas','numpy','transformers')},
  'server':{'vllm':'0.10.2 V1','gpu':'RTX 5090','dtype':'bfloat16','max_model_len':4096,'max_num_seqs_per_server':16,'eager':True,'prefix_caching':False,'memory_fractions':{'main':0.49,'helper':0.28}},
  'max_tokens':16,'client_batch_size':16,'temperature':0,'default':True,'cache_enabled':False}
 (a.out/'preflight.json').write_text(json.dumps(preflight,indent=2)+'\n')
 sem=importlib.import_module('lotus.sem_ops.sem_filter');original_sampling=sem.importance_sampling;captured=[]
 def traced_sampling(scores,args):
  indices,factors=original_sampling(scores,args)
  captured.append({'proxy_scores':[float(x) for x in scores],'sample_indices':[int(x) for x in indices],
                   'correction_factors':[float(x) for x in factors]})
  return indices,factors
 sem.importance_sampling=traced_sampling
 start_usage=usage(main_lm);check_gpu_ownership();main_lm.phase='base';t=time.perf_counter()
 base,base_stats=df.sem_filter(INSTRUCTION,return_all=True,return_raw_outputs=True,default=True,suffix='_lotus',return_stats=True)
 base_seconds=time.perf_counter()-t;after_base=usage(main_lm)
 assert len(base)==n and list(base['query_id'])==list(df['query_id']) and list(base['candidate_id'])==list(df['candidate_id'])
 check_gpu_ownership();main_lm.phase='cascade';helper_lm.phase='cascade';t=time.perf_counter()
 args=CascadeArgs(recall_target=.8,precision_target=.8,sampling_percentage=.1,failure_probability=.2,proxy_model=ProxyModel.HELPER_LM,
                  cascade_IS_random_seed=SEED_CALIBRATION,cascade_IS_max_sample_range=n,cascade_IS_weight=.9,cascade_num_calibration_quantiles=50)
 cascade,stats=df.sem_filter(INSTRUCTION,return_all=True,return_raw_outputs=True,default=True,suffix='_lotus',cascade_args=args,return_stats=True)
 cascade_seconds=time.perf_counter()-t;after_cascade=usage(main_lm);helper_usage=usage(helper_lm)
 assert len(captured)==1 and len(cascade)==n and list(cascade['query_id'])==list(df['query_id']) and list(cascade['candidate_id'])==list(df['candidate_id'])
 raw_base=[str(x) if x is not None else '' for x in base['raw_output_lotus']];raw_cascade=[str(x) if x is not None else '' for x in cascade['raw_output_lotus']]
 bool_base=[bool(x) for x in base['_lotus']];bool_cascade=[bool(x) for x in cascade['_lotus']]
 assert all(parser(r)==v for r,v in zip(raw_base,bool_base))
 vectors={'lotus_base':[[None]*32 for _ in rows],'lotus_cascade':[[None]*32 for _ in rows],
          'selective':[p['pred'] for p in previous],
          'lexical':[['Y' if set(r['query'].casefold().split()) and set(r['query'].casefold().split())<=set(it['text'].casefold().split()) else 'N' for it in r['items']] for r in rows]}
 # Use the existing reference's regex-based tokenization to ensure exact comparability.
 import re
 def terms(x):return set(re.findall(r'\w+',x.casefold()))
 vectors['lexical']=[['Y' if terms(r['query']) and terms(r['query'])<=terms(it['text']) else 'N' for it in r['items']] for r in rows]
 with (a.out/'decisions.jsonl').open('w',encoding='utf8') as f:
  for i,r in enumerate(shuffled):
   pi,ci=r['pool_index'],r['candidate_index'];vectors['lotus_base'][pi][ci]='Y' if bool_base[i] else 'N';vectors['lotus_cascade'][pi][ci]='Y' if bool_cascade[i] else 'N'
   f.write(json.dumps({'shuffled_index':i,'query_id':r['query_id'],'candidate_id':r['candidate_id'],'pool_index':pi,'candidate_index':ci,
      'base':bool_base[i],'cascade':bool_cascade[i],'base_raw':raw_base[i],'cascade_raw':raw_cascade[i]},ensure_ascii=False)+'\n')
 assert all(len(v)==32 and all(x in ('Y','N') for x in v) for ps in vectors.values() for v in ps)
 common=list(range(len(rows)));quality={k:metrics(rows,v,common) for k,v in vectors.items()}
 draw=np.random.default_rng(SEED_BOOT).integers(0,len(rows),size=(4000,len(rows)))
 ref=[x for v in vectors['lotus_base'] for x in v];cas=[x for v in vectors['lotus_cascade'] for x in v]
 tp=sum(a==b=='Y' for a,b in zip(ref,cas));fp=sum(a=='N' and b=='Y' for a,b in zip(ref,cas));fn=sum(a=='Y' and b=='N' for a,b in zip(ref,cas))
 helper_events=[json.loads(line) for line in trace.open(encoding='utf8') if '"role": "helper"' in line]
 assert len(helper_events)==n and all(e['logprobs'] is not None for e in helper_events)
 fallback_count=0;first_positions={}
 for e in helper_events:
  found=next((i for i,t in enumerate(e['logprobs']) if {'True','False'}<={q['token'] for q in t['top_logprobs']}),None)
  if found is None:fallback_count+=1
  else:first_positions[str(found)]=first_positions.get(str(found),0)+1
 proxy_diagnostics={'missing_both_label_logprobs':fallback_count,'first_both_label_position_counts':first_positions,
                    'helper_raw':parse_audit([e['output'] for e in helper_events])}
 summary={'protocol':'PROTOCOL_lotus_wands_cascade.md','pools':len(rows),'candidates':n,'quality':quality,
  'cascade_minus_lotus_base':paired(rows,vectors,common,'lotus_base','lotus_cascade',draw),'query_bootstrap_seed':SEED_BOOT,'resamples':4000,
  'cascade_vs_separately_run_base':{'candidate_agreement':sum(a==b for a,b in zip(ref,cas))/n,'precision':tp/(tp+fp) if tp+fp else None,'recall':tp/(tp+fn) if tp+fn else None,'tp':tp,'fp':fp,'fn':fn},
  'timing_seconds':{'base':base_seconds,'cascade':cascade_seconds},'logical_requests':{'base_main':n,'cascade_helper':n,'cascade_calibration_main':len(captured[0]['sample_indices']),
   'cascade_routed_main':int(stats['filters_resolved_by_large_model'])},
  'usage':{'base_main':difference(start_usage,after_base),'cascade_main':difference(after_base,after_cascade),'cascade_helper':helper_usage},
  'routing':{'helper':int(stats['filters_resolved_by_helper_model']),'main':int(stats['filters_resolved_by_large_model']),'thresholds':stats['cascade_args'].model_dump(mode='json')},
  'raw_diagnostics':{'base':parse_audit(raw_base),'cascade':parse_audit(raw_cascade)},'proxy_diagnostics':proxy_diagnostics}
 (a.out/'sampling_trace.json').write_text(json.dumps(captured[0],indent=2,allow_nan=False)+'\n')
 (a.out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
 print('LOTUS base F1/MAE',quality['lotus_base']['f1'],quality['lotus_base']['ratio_mae'])
 print('LOTUS cascade F1/MAE',quality['lotus_cascade']['f1'],quality['lotus_cascade']['ratio_mae'])
 print('Routed helper/main',summary['routing']['helper'],summary['routing']['main'],'calibration main calls',summary['logical_requests']['cascade_calibration_main'])
 print('Main-only/cascade wall seconds',base_seconds,cascade_seconds)
if __name__=='__main__':main()
