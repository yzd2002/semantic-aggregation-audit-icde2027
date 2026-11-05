import json,tempfile
from pathlib import Path
import pandas as pd,lotus
from lotus.types import CascadeArgs,ProxyModel
from transformers import AutoTokenizer
from run_lotus_wands_cascade import CapturedLM,usage
p=lambda n:int(Path('/tmp/intentsat-lotus-cascade-'+n+'-port.txt').read_text())
trace=Path(tempfile.mktemp(prefix='intentsat-lotus-usage-smoke-',suffix='.jsonl'))
def model(n,path,served):
 tok=AutoTokenizer.from_pretrained(path,local_files_only=True)
 x=CapturedLM(model='hosted_vllm/'+served,api_base=f'http://127.0.0.1:{p(n)}/v1',api_key='local',tokenizer=tok,max_ctx_len=4096,max_tokens=16,max_batch_size=16,temperature=0,trace_file=trace,role=n)
 x.phase='synthetic';return x
main=model('main','/root/models/Qwen/Qwen3-4B-Instruct-2507','intentsat-oracle')
helper=model('helper','/home/gpu/intent2s_models/SmolLM2-1.7B-Instruct','intentsat-helper')
lotus.settings.configure(lm=main,helper_lm=helper,enable_cache=False)
df=pd.DataFrame([{'query':'white ceramic mug','product':('A white ceramic coffee mug.' if i%2==0 else 'A plastic mug lid.')} for i in range(32)])
a=CascadeArgs(recall_target=.8,precision_target=.8,sampling_percentage=.25,failure_probability=.2,proxy_model=ProxyModel.HELPER_LM,cascade_IS_random_seed=20271231,cascade_IS_max_sample_range=32)
out,stats=df.sem_filter('The {product} is a full exact match for the query {query} and satisfies all stated requirements.',return_all=True,return_raw_outputs=True,default=True,suffix='_lotus',cascade_args=a,return_stats=True)
events=[json.loads(x) for x in trace.read_text().splitlines()]
for role,model in [('main',main),('helper',helper)]:
 rows=[r for r in events if r['role']==role];assert rows and all(r['usage'] is not None for r in rows)
 totals={k:sum(r['usage'][k] for r in rows) for k in ('prompt_tokens','completion_tokens','total_tokens')}
 assert totals==usage(model)['physical'] and usage(model)['cache_hits']==0
print('Synthetic trace captured',len(events),'responses with per-request token usage and reconciled LOTUS totals.')
