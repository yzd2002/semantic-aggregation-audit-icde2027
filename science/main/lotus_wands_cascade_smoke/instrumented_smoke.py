import json,os,hashlib
from pathlib import Path
import numpy as np,pandas as pd,lotus
from lotus.models import LM
from lotus.types import CascadeArgs,ProxyModel
from transformers import AutoTokenizer
class CaptureLM(LM):
 def __call__(self,messages,**kwargs):
  result=super().__call__(messages,**kwargs)
  events=[]
  for i,(msg,out) in enumerate(zip(messages,result.outputs)):
   lp=None if result.logprobs is None else [x.model_dump() for x in result.logprobs[i]]
   events.append({'messages':msg,'output':out,'logprobs':lp})
  if not hasattr(self,'capture'):self.capture=[]
  self.capture.append(events)
  return result
p=lambda n:int(Path('/tmp/intentsat-lotus-cascade-'+n+'-port.txt').read_text())
def model(n,path,served):
 tok=AutoTokenizer.from_pretrained(path,local_files_only=True)
 return CaptureLM(model='hosted_vllm/'+served,api_base=f'http://127.0.0.1:{p(n)}/v1',api_key='local',tokenizer=tok,max_ctx_len=4096,max_tokens=16,max_batch_size=16,temperature=0)
main=model('main','/root/models/Qwen/Qwen3-4B-Instruct-2507','intentsat-oracle')
helper=model('helper','/home/gpu/intent2s_models/SmolLM2-1.7B-Instruct','intentsat-helper')
lotus.settings.configure(lm=main,helper_lm=helper,enable_cache=False)
rows=[{'query':'white ceramic mug','product':('A white ceramic coffee mug.' if i%2==0 else 'A plastic mug lid.')} for i in range(32)]
df=pd.DataFrame(rows)
args=CascadeArgs(recall_target=.8,precision_target=.8,sampling_percentage=.25,failure_probability=.2,proxy_model=ProxyModel.HELPER_LM,cascade_IS_random_seed=20271231,cascade_IS_max_sample_range=32)
out,stats=df.sem_filter('The {product} is a full exact match for the query {query} and satisfies all stated requirements.',return_all=True,return_raw_outputs=True,default=True,suffix='_lotus',cascade_args=args,return_stats=True)
assert len(out)==len(df) and '_lotus' in out.columns and 'raw_output_lotus' in out.columns
v={'pools':1,'rows':len(out),'positive':int(out['_lotus'].sum()),'resolved_helper':stats.get('filters_resolved_by_helper_model'),'resolved_main':stats.get('filters_resolved_by_large_model'),'thresholds':stats['cascade_args'].model_dump(mode='json')}
assert v['resolved_helper']+v['resolved_main']==len(df)
Path('/tmp/intentsat-lotus-cascade-smoke-result.json').write_text(json.dumps(v,indent=2)+'\n')
Path('/tmp/intentsat-lotus-cascade-smoke-events.json').write_text(json.dumps({'main':main.capture,'helper':helper.capture},indent=2)+'\n')
print('Synthetic cascade completed:',v['rows'],'rows; helper',v['resolved_helper'],'main',v['resolved_main'])
