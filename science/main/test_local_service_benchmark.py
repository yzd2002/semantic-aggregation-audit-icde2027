#!/usr/bin/env python3
"""Check client decisions on controlled response, retry and HTTP-failure cases."""
import asyncio,json
from run_local_service_benchmark import pipeline,summarise
class Response:
 def __init__(self,status,content=''):
  self.status_code=status;self.content=content
 def json(self):
  return {'choices':[{'message':{'content':self.content},'finish_reason':'stop'}],
          'usage':{'prompt_tokens':10,'completion_tokens':1}}
class FakeClient:
 def __init__(self,mode):self.mode=mode;self.count=0
 async def post(self,path,json):
  self.count+=1;prompt=json['messages'][1]['content'];regex=json.get('guided_regex')
  if self.mode=='httpfail' and regex is None:return Response(503)
  if regex is None:return Response(200,'{"product_type":"cup","required":[],"preferred":[]}')
  if regex=='[YN]':return Response(200,'Y')
  if self.mode=='invalid':return Response(200,'bad')
  reverse='{"id": 8,' in prompt and prompt.index('{"id": 8,')<prompt.index('{"id": 1,')
  labels={str(i):('Y' if i==(2 if reverse else 1) else 'N') for i in range(1,9)}
  return Response(200,__import__('json').dumps(labels,separators=(',',':')))
row={'query_id':'sample','query':'white cup','items':[{'id':i+1,'text':f'cup {i+1}','gold':'Y' if i in (0,1) else 'N'} for i in range(32)]}
async def check():
 c=FakeClient('ok');pred,requests,fallback=await pipeline(c,'local',row,'selective')
 assert pred is not None and not fallback and [r['index'] for r in requests if r['stage']=='pointwise']==[0,1,8,9,16,17,24,25]
 assert len(requests)==17 and pred[:3]==['Y','Y','N']
 c=FakeClient('invalid');pred,requests,_=await pipeline(c,'local',row,'selective')
 assert pred is None and all(r['stage']!='pointwise' for r in requests) and all(len(r['attempts'])==2 for r in requests[1:])
 c=FakeClient('httpfail');pred,requests,fallback=await pipeline(c,'local',row,'pointwise')
 assert pred is None and not fallback and len(requests)==1 and len(requests[0]['attempts'])==1
asyncio.run(check())
rows=[row,row];recs=[{'prediction':['Y']*32,'active_seconds':1.,'specification_fallback':False,'requests':[]},
                     {'prediction':None,'active_seconds':.1,'specification_fallback':True,'requests':[]}]
s=summarise(rows,recs,2.);assert s['complete_pools']==1 and s['active_latency_p50_seconds']==1. and s['active_latency_p95_seconds']==1. and s['complete_pools_per_second']==.5
print('Selective fan-out, parsing retry, HTTP failure and complete-only latency tests passed.')
