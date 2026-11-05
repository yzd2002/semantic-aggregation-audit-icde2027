#!/usr/bin/env python3
"""Closed-loop, local HTTP comparison; see PROTOCOL_local_service_benchmark.md."""
import argparse,asyncio,hashlib,json,random,time
from pathlib import Path
from urllib.parse import urlparse
import httpx,numpy as np
import run_experiment as base
import run_full_pools as harness
import run_wands
from order_fallback import combine
from run_keyed_order_audit import keyed_prompt,keyed_regex,parse_keyed
from run_selective_execution import point_prompt,parse_one,SUFFIX
from run_compact_prompt_ablation import metrics
from select_selective_execution import lines

PROTOCOL='PROTOCOL_local_service_benchmark.md'
SEED=20271231
IDS=list(range(1,9))
RULE=keyed_regex(IDS)

def digest(text):return hashlib.sha256(text.encode('utf8')).hexdigest()
def conditions():
 c=[(p,n) for p in ('pointwise','keyed','selective') for n in (1,4)]
 random.Random(SEED).shuffle(c)
 return [(0,*x) for x in c]+[(1,*x) for x in reversed(c)]
async def call(client,model,text,system,cap,rule):
 body={'model':model,'messages':[{'role':'system','content':system},{'role':'user','content':text}],
       'temperature':0,'max_tokens':cap,'stream':False}
 if rule is not None:body['guided_regex']=rule
 started=time.perf_counter()
 try:
  r=await client.post('/v1/chat/completions',json=body);duration=time.perf_counter()-started
  if r.status_code!=200:return {'http_status':r.status_code,'error':'http_status','seconds':duration,'output':None,'prompt_tokens':0,'completion_tokens':0}
  payload=r.json();choices=payload['choices'];usage=payload['usage'];assert len(choices)==1
  output=choices[0]['message']['content'];assert isinstance(output,str)
  return {'http_status':200,'error':None,'seconds':duration,'output':output.strip(),
          'finish_reason':choices[0]['finish_reason'],'prompt_tokens':int(usage['prompt_tokens']),
          'completion_tokens':int(usage['completion_tokens'])}
 except (httpx.HTTPError,ValueError,KeyError,TypeError,AssertionError) as e:
  return {'http_status':None,'error':type(e).__name__,'seconds':time.perf_counter()-started,'output':None,'prompt_tokens':0,'completion_tokens':0}
async def request(client,model,stage,text,parser,cap,rule,meta):
 attempts=[];msg=text
 for attempt in range(2):
  out=await call(client,model,msg,run_wands.SYSTEM,cap,rule);out['prompt_sha256']=digest(msg);attempts.append(out)
  if out['error'] is not None:break
  if parser(out['output']) is not None:break
  msg=text+('\nReturn valid JSON only.' if stage=='spec' else SUFFIX)
 return {'stage':stage,**meta,'attempts':attempts}
def final(r,parser):
 a=r['attempts'][-1]
 return None if a['error'] is not None else parser(a['output'])
async def pipeline(client,model,row,plan):
 records=[];spec=await request(client,model,'spec',base.p_spec(row),base.parse_spec,160,None,{})
 records.append(spec)
 if spec['attempts'][-1]['error'] is not None:return None,records,False
 obj=final(spec,base.parse_spec);fallback=obj is None
 chunks=harness.chunks(row)
 async def keyed(part,mode):
  current=part if mode=='original' else {**part,'items':[dict(x) for x in reversed(part['items'])]}
  return await request(client,model,'keyed',keyed_prompt(current,obj),lambda s:parse_keyed(s,IDS),128,RULE,
       {'start':part['start'],'mode':mode,'presented_ids':[x['id'] for x in current['items']]})
 async def point(i):
  return await request(client,model,'pointwise',point_prompt(row,row['items'][i],obj),parse_one,4,'[YN]',{'index':i})
 if plan=='pointwise':
  pts=await asyncio.gather(*(point(i) for i in range(32)));records.extend(pts)
  v=[final(r,parse_one) for r in pts];return (None if None in v else v),records,fallback
 modes=['original'] if plan=='keyed' else ['original','reverse']
 ks=await asyncio.gather(*(keyed(part,mode) for mode in modes for part in chunks));records.extend(ks)
 vectors=[]
 for mode_i in range(len(modes)):
  v=[]
  for r in ks[mode_i*4:(mode_i+1)*4]:
   d=final(r,lambda s:parse_keyed(s,IDS))
   if d is None:return None,records,fallback
   v.extend(d[i] for i in IDS)
  vectors.append(v)
 if plan=='keyed':return vectors[0],records,fallback
 need=[i for i,(a,b) in enumerate(zip(*vectors)) if a!=b]
 pts=await asyncio.gather(*(point(i) for i in need));records.extend(pts)
 h=[None]*32
 for i,r in zip(need,pts):h[i]=final(r,parse_one)
 return combine(vectors[0],vectors[1],h),records,fallback
async def sanity(client,model):
 msg='Reply with only the word BANANA.';system='Follow the user instruction exactly.'
 raw=await call(client,model,msg,system,128,None);point=await call(client,model,msg,system,4,'[YN]');keyed=await call(client,model,msg,system,128,RULE)
 valid=(not any(x['error'] for x in (raw,point,keyed)) and parse_one(raw['output']) is None and parse_keyed(raw['output'],IDS) is None
        and parse_one(point['output']) is not None and parse_keyed(keyed['output'],IDS) is not None)
 return {'unconstrained':raw,'point_guided':point,'keyed_guided':keyed,'valid':valid}
def summarise(rows,recs,wall):
 complete=[i for i,r in enumerate(recs) if r['prediction'] is not None]
 pred=[r['prediction'] for r in recs];lat=[recs[i]['active_seconds'] for i in complete]
 attempts=[a for r in recs for req in r['requests'] for a in req['attempts']]
 return {'attempted_pools':len(recs),'complete_pools':len(complete),'specification_fallbacks':sum(r['specification_fallback'] for r in recs),
         'quality_complete':metrics(rows,pred,complete) if complete else None,
         'active_latency_p50_seconds':float(np.quantile(lat,.5)) if lat else None,
         'active_latency_p95_seconds':float(np.quantile(lat,.95)) if lat else None,
         'condition_wall_seconds':wall,'complete_pools_per_second':len(complete)/wall,
         'model_http_attempts':len(attempts),'http_failures':sum(a['error'] is not None for a in attempts),
         'logical_tokens':{'input':sum(a['prompt_tokens'] for a in attempts),'output':sum(a['completion_tokens'] for a in attempts)}}
async def main():
 ap=argparse.ArgumentParser();ap.add_argument('--endpoint',required=True);ap.add_argument('--model',default='intentsat-local');ap.add_argument('--selection',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
 url=urlparse(a.endpoint)
 if url.scheme!='http' or url.hostname not in ('localhost','127.0.0.1') or url.path not in ('','/'):raise ValueError('Only a local loopback HTTP endpoint is allowed')
 rows=lines(a.selection/'inputs.jsonl')
 if len(rows)!=52 or any(len(x['items'])!=32 for x in rows):raise ValueError('Expected the fixed 52-query selection')
 selection_hash=hashlib.sha256((a.selection/'inputs.jsonl').read_bytes()).hexdigest()
 a.out.mkdir(parents=True,exist_ok=False)
 async with httpx.AsyncClient(base_url=a.endpoint,timeout=120,limits=httpx.Limits(max_connections=256,max_keepalive_connections=256),trust_env=False) as client:
  modelcheck=await client.get('/v1/models');modelcheck.raise_for_status()
  if a.model not in [x['id'] for x in modelcheck.json()['data']]:raise ValueError('Unexpected served model')
  control=await sanity(client,a.model);(a.out/'guidance_sanity.json').write_text(json.dumps(control,indent=2)+'\n')
  if not control['valid']:raise RuntimeError('Constraint negative control failed')
  dummy={'query_id':'synthetic','query':'white ceramic cup','items':[{'id':i+1,'text':f'Name: white ceramic cup, item {i+1}'} for i in range(32)]}
  for plan in ('pointwise','keyed','selective'):
   result,_,_=await pipeline(client,a.model,dummy,plan)
   if result is None:raise RuntimeError('Synthetic warmup incomplete for '+plan)
  summary={'protocol':PROTOCOL,'selection_inputs_sha256':selection_hash,'model':a.model,'server':'vLLM 0.10.2 V1 local loopback',
           'prefix_caching':False,'max_num_seqs':16,'condition_order':conditions(),'conditions':[]}
  with (a.out/'pool_records.jsonl').open('w',encoding='utf8') as out:
   for repeat,plan,concurrency in conditions():
    shuffled=rows[:];random.Random(SEED+repeat).shuffle(shuffled);semaphore=asyncio.Semaphore(concurrency)
    t=time.perf_counter()
    async def one(row):
     async with semaphore:
      started=time.perf_counter();pred,req,fallback=await pipeline(client,a.model,row,plan);ended=time.perf_counter()
      return {'repeat':repeat,'plan':plan,'concurrency':concurrency,'query_id':row['query_id'],
              'start_offset_seconds':started-t,'end_offset_seconds':ended-t,'active_seconds':ended-started,
              'prediction':pred,'specification_fallback':fallback,'requests':req}
    recs=await asyncio.gather(*(one(row) for row in shuffled));wall=time.perf_counter()-t
    stats=summarise(shuffled,recs,wall);stats.update({'repeat':repeat,'plan':plan,'concurrency':concurrency})
    summary['conditions'].append(stats)
    for r in recs:out.write(json.dumps(r,ensure_ascii=False)+'\n')
    out.flush();(a.out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print('condition',repeat,plan,concurrency,'complete',stats['complete_pools'],'p50',stats['active_latency_p50_seconds'],'p95',stats['active_latency_p95_seconds'],'wall',wall,flush=True)
    if stats['http_failures']>0:raise RuntimeError('HTTP failures recorded; aborting later conditions')
  print('Completed all service conditions.',flush=True)
if __name__=='__main__':asyncio.run(main())
