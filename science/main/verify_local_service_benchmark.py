#!/usr/bin/env python3
"""Independently reparse local-service records, scheduling and reported metrics."""
import argparse,hashlib,json,random,re
from pathlib import Path
import numpy as np
import run_experiment as base
import run_full_pools as harness
from run_keyed_order_audit import keyed_prompt
from run_selective_execution import point_prompt,SUFFIX

def lines(path):
 with path.open(encoding='utf8') as f:return [json.loads(s) for s in f]
def digest(s):return hashlib.sha256(s.encode()).hexdigest()
def pairs(p):
 out={}
 for k,v in p:
  if k in out:raise ValueError('duplicate key')
  out[k]=v
 return out
def spec_parse(s):
 try:
  x=json.loads(s,object_pairs_hook=pairs)
  if not isinstance(x,dict) or set(x)!={'product_type','required','preferred'}:return None
  if not isinstance(x['product_type'],str) or not x['product_type'].strip():return None
  if any(not isinstance(x[k],list) or any(not isinstance(y,str) for y in x[k]) for k in ('required','preferred')):return None
  return x
 except (ValueError,TypeError,AttributeError):return None
def key_parse(s):
 try:
  if not re.fullmatch(r'\{"1":"[YN]","2":"[YN]","3":"[YN]","4":"[YN]","5":"[YN]","6":"[YN]","7":"[YN]","8":"[YN]"\}',s):return None
  x=json.loads(s,object_pairs_hook=pairs);return [x[str(i)] for i in range(1,9)]
 except (ValueError,TypeError):return None
def point_parse(s):return s if s in ('Y','N') else None
def checked(req,stage,prompt,parser):
 assert req['stage']==stage and 1<=len(req['attempts'])<=2
 attempts=req['attempts']
 for i,a in enumerate(attempts):
  expected=prompt if i==0 else prompt+('\nReturn valid JSON only.' if stage=='spec' else SUFFIX)
  assert a['prompt_sha256']==digest(expected) and a['seconds']>=0
  assert isinstance(a['prompt_tokens'],int) and isinstance(a['completion_tokens'],int)
  assert a['prompt_tokens']>=0 and a['completion_tokens']>=0
  if i==1:assert attempts[0]['error'] is None and parser(attempts[0]['output']) is None
  if a['error'] is not None:assert i==len(attempts)-1 and a['output'] is None
  else:assert a['http_status']==200 and isinstance(a['output'],str)
  if i==0 and len(attempts)==1 and a['error'] is None:assert parser(a['output']) is not None or len(attempts)==2
 last=attempts[-1];return None if last['error'] is not None else parser(last['output'])
def verify_pool(row,record):
 reqs=record['requests'];assert reqs and record['query_id']==row['query_id']
 spec=checked(reqs[0],'spec',base.p_spec(row),spec_parse)
 if reqs[0]['attempts'][-1]['error'] is not None:
  assert len(reqs)==1 and record['prediction'] is None and not record['specification_fallback'];return
 assert record['specification_fallback']==(spec is None)
 chunks=harness.chunks(row);plan=record['plan']
 if plan=='pointwise':
  assert len(reqs)==33;v=[]
  for i,r in enumerate(reqs[1:]):
   assert r['index']==i;v.append(checked(r,'pointwise',point_prompt(row,row['items'][i],spec),point_parse))
  assert record['prediction']==(None if None in v else v);return
 modes=['original'] if plan=='keyed' else ['original','reverse'];kreq=reqs[1:1+4*len(modes)];assert len(kreq)==4*len(modes)
 vectors=[]
 for mode_index,mode in enumerate(modes):
  v=[]
  for j,part in enumerate(chunks):
   r=kreq[mode_index*4+j];cur=part if mode=='original' else {**part,'items':[dict(x) for x in reversed(part['items'])]}
   assert r['start']==part['start'] and r['mode']==mode and r['presented_ids']==[x['id'] for x in cur['items']]
   x=checked(r,'keyed',keyed_prompt(cur,spec),key_parse)
   if x is None:v=None
   elif v is not None:v.extend(x)
  vectors.append(v)
 if any(x is None for x in vectors):assert record['prediction'] is None and len(reqs)==1+4*len(modes);return
 if plan=='keyed':assert len(reqs)==5 and record['prediction']==vectors[0];return
 need=[i for i,(a,b) in enumerate(zip(*vectors)) if a!=b];points=reqs[9:];assert [x['index'] for x in points]==need
 h={}
 for i,r in zip(need,points):h[i]=checked(r,'pointwise',point_prompt(row,row['items'][i],spec),point_parse)
 pred=[vectors[0][i] if vectors[0][i]==vectors[1][i] else h[i] for i in range(32)]
 assert record['prediction']==(None if None in pred else pred)
def quality(rows,recs):
 tp=fp=tn=fn=0;mae=[]
 for row,r in zip(rows,recs):
  p=r['prediction']
  if p is None:continue
  y=[i['gold'] for i in row['items']];assert len(y)==len(p)==32
  for a,b in zip(y,p):tp+=a==b=='Y';fp+=a=='N' and b=='Y';tn+=a==b=='N';fn+=a=='Y' and b=='N'
  mae.append(abs(p.count('Y')-y.count('Y'))/32)
 return {'tp':tp,'fp':fp,'tn':tn,'fn':fn,'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,
         'balanced_accuracy':(tp/(tp+fn)+tn/(tn+fp))/2 if tp+fn and tn+fp else None,
         'ratio_mae':sum(mae)/len(mae) if mae else None}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);ap.add_argument('--selection',type=Path,required=True);a=ap.parse_args()
 rows=lines(a.selection/'inputs.jsonl');summary=json.loads((a.run/'summary.json').read_text());records=lines(a.run/'pool_records.jsonl')
 assert hashlib.sha256((a.selection/'inputs.jsonl').read_bytes()).hexdigest()==summary['selection_inputs_sha256']
 control=json.loads((a.run/'guidance_sanity.json').read_text());assert control['valid'] and point_parse(control['unconstrained']['output']) is None and key_parse(control['unconstrained']['output']) is None
 assert point_parse(control['point_guided']['output']) is not None and key_parse(control['keyed_guided']['output']) is not None
 combos=[(p,c) for p in ('pointwise','keyed','selective') for c in (1,4)];random.Random(20271231).shuffle(combos)
 order=[(0,*x) for x in combos]+[(1,*x) for x in reversed(combos)]
 assert [tuple(x) for x in summary['condition_order']]==order and len(summary['conditions'])==12 and len(records)==52*12
 for j,(repeat,plan,concurrency) in enumerate(order):
  src=rows[:];random.Random(20271231+repeat).shuffle(src);recs=records[j*52:(j+1)*52];st=summary['conditions'][j]
  assert (st['repeat'],st['plan'],st['concurrency'])==(repeat,plan,concurrency)
  assert all((r['repeat'],r['plan'],r['concurrency'])==(repeat,plan,concurrency) for r in recs)
  for row,r in zip(src,recs):
   assert 0<=r['start_offset_seconds']<=r['end_offset_seconds']<=st['condition_wall_seconds']+1e-8
   assert abs(r['end_offset_seconds']-r['start_offset_seconds']-r['active_seconds'])<1e-8
   verify_pool(row,r)
  events=sorted([(r['start_offset_seconds'],1) for r in recs]+[(r['end_offset_seconds'],-1) for r in recs],key=lambda x:(x[0],x[1]));active=peak=0
  for _,change in events:active+=change;peak=max(peak,active);assert active>=0
  assert active==0 and peak<=concurrency
  complete=[r for r in recs if r['prediction'] is not None];lat=[r['active_seconds'] for r in complete]
  attempts=[x for r in recs for req in r['requests'] for x in req['attempts']]
  assert st['attempted_pools']==len(recs) and st['complete_pools']==len(complete)
  assert st['specification_fallbacks']==sum(r['specification_fallback'] for r in recs)
  assert st['model_http_attempts']==len(attempts) and st['http_failures']==sum(x['error'] is not None for x in attempts)
  assert st['logical_tokens']=={'input':sum(x['prompt_tokens'] for x in attempts),'output':sum(x['completion_tokens'] for x in attempts)}
  assert abs(st['complete_pools_per_second']-len(complete)/st['condition_wall_seconds'])<1e-12
  if complete:
   assert abs(st['active_latency_p50_seconds']-float(np.quantile(lat,.5)))<1e-12
   assert abs(st['active_latency_p95_seconds']-float(np.quantile(lat,.95)))<1e-12
   q=quality(src,recs)
   for k,v in q.items():
    if v is None:assert st['quality_complete'][k] is None
    else:assert abs(st['quality_complete'][k]-v)<1e-12,(j,k)
  print('repeat',repeat,plan,'concurrency',concurrency,'complete',len(complete),'peak active',peak,'verified')
 print('Responses, retries, candidate associations, overlap limits, latency, quality, calls and throughput independently verified.')
if __name__=='__main__':main()
