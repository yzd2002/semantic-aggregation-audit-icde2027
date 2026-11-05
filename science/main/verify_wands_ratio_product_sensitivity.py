#!/usr/bin/env python3
"""Independent loop-based query/product reweighting check for WANDS-52."""
import argparse,collections,json,math
from pathlib import Path
import numpy as np

def lines(p):
 with p.open(encoding='utf8') as f:return [json.loads(x) for x in f if x.strip()]
def main():
 p=argparse.ArgumentParser();p.add_argument('--artifact',type=Path,required=True);p.add_argument('--result',type=Path,required=True);a=p.parse_args();root=a.artifact
 rows=lines(root/'wands_selective_execution_selection/inputs.jsonl');saved=[s for s in lines(root/'run_wands_supervised_reference/predictions.jsonl') if s['sample']=='selective52']
 ref=json.loads(a.result.read_text())['samples']['selective52'];threshold=json.loads((root/'run_wands_ratio_baseline/summary.json').read_text())['threshold_selection']['threshold']
 if len(rows)!=52 or len(saved)!=52 or any(x['query_id']!=y['query_id'] for x,y in zip(rows,saved)):raise ValueError('Query alignment')
 products=sorted({int(x['product_id']) for r in rows for x in r['items']});pid={v:i for i,v in enumerate(products)};rng=np.random.default_rng(20271201);harddiff=[];lexdiff=[]
 for _ in range(4000):
  qc=collections.Counter(int(x) for x in rng.integers(0,52,52));pc=collections.Counter(int(x) for x in rng.integers(0,len(products),len(products)));h=l=0.0
  for q,count in qc.items():
   r=rows[q];s=saved[q];den=ng=nq=nh=nl=0
   for i,item in enumerate(r['items']):
    w=pc.get(pid[int(item['product_id'])],0);den+=w;ng+=w*(item['gold']=='Y');nq+=w*(s['qwen'][i]=='Y');nh+=w*(s['positive_probability'][i]>=threshold);nl+=w*(s['lexical'][i]=='Y')
   if not den:raise ValueError('Selected query has zero product weight')
   g=ng/den;err=abs(nq/den-g);h+=count*(abs(nh/den-g)-err);l+=count*(abs(nl/den-g)-err)
  harddiff.append(h/52);lexdiff.append(l/52)
 for name,values in [('hard_minus_qwen',harddiff),('lexical_minus_qwen',lexdiff)]:
  interval=np.quantile(values,[.025,.975])
  if any(not math.isclose(x,y,abs_tol=1e-12,rel_tol=0) for x,y in zip(interval,ref['ranges'][name])):raise ValueError(name+' mismatch')
 print('Independent loop-based 52-query product sensitivity ranges verified.')
if __name__=='__main__':main()
