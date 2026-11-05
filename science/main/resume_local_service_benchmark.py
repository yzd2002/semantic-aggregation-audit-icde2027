#!/usr/bin/env python3
"""Continue the fixed schedule after its first recorded condition; see amendment."""
import argparse,asyncio,hashlib,json,random,time
from pathlib import Path
import httpx
import run_local_service_benchmark as benchmark

async def main():
 p=argparse.ArgumentParser()
 p.add_argument('--endpoint',required=True);p.add_argument('--model',default='intentsat-local')
 p.add_argument('--selection',type=Path,required=True);p.add_argument('--run',type=Path,required=True)
 a=p.parse_args();u=benchmark.urlparse(a.endpoint)
 if u.scheme!='http' or u.hostname not in ('localhost','127.0.0.1') or u.path not in ('','/'):
  raise ValueError('Only a local loopback HTTP endpoint is allowed')
 rows=benchmark.lines(a.selection/'inputs.jsonl')
 summary=json.loads((a.run/'summary.json').read_text())
 records=benchmark.lines(a.run/'pool_records.jsonl')
 assert len(rows)==52 and all(len(r['items'])==32 for r in rows)
 assert len(summary['conditions'])==1 and len(records)==52
 assert [tuple(x) for x in summary['condition_order']]==benchmark.conditions()
 assert (summary['conditions'][0]['repeat'],summary['conditions'][0]['plan'],summary['conditions'][0]['concurrency'])==benchmark.conditions()[0]
 assert summary['selection_inputs_sha256']==hashlib.sha256((a.selection/'inputs.jsonl').read_bytes()).hexdigest()
 assert summary['model']==a.model
 async with httpx.AsyncClient(base_url=a.endpoint,timeout=120,limits=httpx.Limits(max_connections=256,max_keepalive_connections=256),trust_env=False) as client:
  modelcheck=await client.get('/v1/models');modelcheck.raise_for_status()
  if a.model not in [x['id'] for x in modelcheck.json()['data']]:raise ValueError('Unexpected served model')
  with (a.run/'pool_records.jsonl').open('a',encoding='utf8') as out:
   for repeat,plan,concurrency in benchmark.conditions()[1:]:
    shuffled=rows[:];random.Random(benchmark.SEED+repeat).shuffle(shuffled);semaphore=asyncio.Semaphore(concurrency)
    t=time.perf_counter()
    async def one(row):
     async with semaphore:
      started=time.perf_counter();pred,req,fallback=await benchmark.pipeline(client,a.model,row,plan);ended=time.perf_counter()
      return {'repeat':repeat,'plan':plan,'concurrency':concurrency,'query_id':row['query_id'],
              'start_offset_seconds':started-t,'end_offset_seconds':ended-t,'active_seconds':ended-started,
              'prediction':pred,'specification_fallback':fallback,'requests':req}
    recs=await asyncio.gather(*(one(row) for row in shuffled));wall=time.perf_counter()-t
    stats=benchmark.summarise(shuffled,recs,wall);stats.update({'repeat':repeat,'plan':plan,'concurrency':concurrency})
    summary['conditions'].append(stats)
    for r in recs:out.write(json.dumps(r,ensure_ascii=False)+'\n')
    out.flush();(a.run/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print('condition',repeat,plan,concurrency,'complete',stats['complete_pools'],'http failures',stats['http_failures'],
          'p50',stats['active_latency_p50_seconds'],'p95',stats['active_latency_p95_seconds'],'wall',wall,flush=True)
    if not stats['complete_pools']:raise RuntimeError('All pools failed; stopping after preserving this condition')
 print('Completed all service conditions.',flush=True)
if __name__=='__main__':asyncio.run(main())
