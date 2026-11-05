#!/usr/bin/env python3
"""Paired query-ID cluster bootstrap on common complete original ESCI pools."""
import argparse,json,re
from pathlib import Path
import numpy as np
ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,default=Path(__file__).resolve().parent/'run_full_pools_250');p=ap.parse_args().run
rows=[json.loads(s) for s in (p/'inputs.jsonl').open(encoding='utf8')]
outputs=[json.loads(s) for s in (p/'outputs.jsonl').open(encoding='utf8')]
methods=['pointwise','batch_json','two_stage_json','two_stage_compact_fallback']
assert len(rows)==len(outputs) and all(r['query_id']==o['query_id'] for r,o in zip(rows,outputs))
shared=[i for i,o in enumerate(outputs) if all(o['plans'][m]['pred'] is not None for m in methods)]
assert shared
records={}
constants=['always_yes','always_no','lexical_all_terms']
for method in methods+constants:
 a=[]
 for i in shared:
  row=rows[i];gold=[x['gold'] for x in row['items']]
  if method in methods: pred=outputs[i]['plans'][method]['pred']
  elif method=='lexical_all_terms':
   query=set(re.findall(r'\w+',row['query'].casefold()))
   pred=['Y' if query and query.issubset(set(re.findall(r'\w+',item['text'].casefold()))) else 'N' for item in row['items']]
  else: pred=[('Y' if method=='always_yes' else 'N')]*len(gold)
  assert len(pred)==len(gold)
  tp=sum(g=='Y' and y=='Y' for g,y in zip(gold,pred));fp=sum(g=='N' and y=='Y' for g,y in zip(gold,pred))
  fn=sum(g=='Y' and y=='N' for g,y in zip(gold,pred));tn=sum(g=='N' and y=='N' for g,y in zip(gold,pred))
  mae=abs(sum(y=='Y' for y in pred)/len(pred)-sum(g=='Y' for g in gold)/len(gold))
  a.append((tp,fp,fn,tn,mae))
 records[method]=np.array(a,dtype=float)
B=4000;seed=20271104;rng=np.random.default_rng(seed);draw=rng.integers(0,len(shared),size=(B,len(shared)))
boot={};result={'description':'Query-cluster bootstrap on the common complete subset. Intervals are descriptive for the original labeled query pools, not a user-satisfaction confidence interval.',
 'common_complete_pools':len(shared),'resamples':B,'seed':seed,'methods':{},'differences':{}}
def measures(a):
 sums=a[:,:4].sum(axis=0);tp,fp,fn,tn=sums
 return {'f1':float(2*tp/(2*tp+fp+fn)) if 2*tp+fp+fn else 0.0,'mae':float(a[:,4].mean()),'accuracy':float((tp+tn)/sum(sums)), 'balanced_accuracy':float((tp/(tp+fn)+tn/(tn+fp))/2)}
def ci(v):return [float(x) for x in np.quantile(v,[.025,.975])]
for method,a in records.items():
 totals=a[draw,:4].sum(axis=1);maes=a[draw,4].mean(axis=1)
 f1=2*totals[:,0]/(2*totals[:,0]+totals[:,1]+totals[:,2]);boot[method]={'f1':f1,'mae':maes}
 result['methods'][method]={**measures(a),'f1_ci':ci(f1),'mae_ci':ci(maes)}
for method in methods[1:]+constants:
 result['differences'][method]={k+'_ci':ci(boot[method][k]-boot['pointwise'][k]) for k in ('f1','mae')}
(p/'paired_analysis.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
