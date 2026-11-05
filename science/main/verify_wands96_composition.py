#!/usr/bin/env python3
"""Verify composition deltas from saved fallback predictions and raw keyed JSON."""
import argparse,collections,hashlib,json,math
from pathlib import Path
import numpy as np

def read(p):
 with p.open(encoding='utf8') as f:return [json.loads(x) for x in f if x.strip()]
def check(x,y,label):
 if not math.isclose(float(x),float(y),abs_tol=1e-12,rel_tol=0):raise ValueError(label+' mismatch')
def main():
 p=argparse.ArgumentParser();p.add_argument('--artifact',type=Path,required=True);p.add_argument('--result',type=Path,required=True);a=p.parse_args();r=a.artifact;out=json.loads(a.result.read_text())
 files={'selection':'wands_fallback_validation_selection/inputs.jsonl','original':'run_wands_fallback_validation_qwen_v1/original_responses.jsonl','reverse':'run_wands_fallback_validation_qwen_v1/stable_reverse_responses.jsonl','pointwise':'run_wands_fallback_validation_qwen_v1/pointwise_responses.jsonl','paired':'run_wands_fallback_validation_qwen_v1/paired_analysis.json','saved_reference':'run_wands_supervised_reference/predictions.jsonl'}
 for k,path in files.items():
  with (r/path).open('rb') as f:actual=hashlib.file_digest(f,'sha256').hexdigest()
  if actual!=out['input_sha256'][k]:raise ValueError('Input digest mismatch: '+k)
 rows=read(r/files['selection']);responses=read(r/files['original']);reversals=read(r/files['reverse']);pred=[v for v in read(r/files['saved_reference']) if v['sample']=='fallback96'];grouped=collections.defaultdict(list);routes=collections.Counter()
 if len(rows)!=96 or len(pred)!=96 or len(responses)!=384 or len(reversals)!=384:raise ValueError('Length mismatch')
 for i,(row,prediction) in enumerate(zip(rows,pred)):
  if row['query_id']!=prediction['query_id'] or len(row['items'])!=32 or len(prediction['qwen'])!=32:raise ValueError('Sample mismatch')
  original=[];reverse=[]
  for c in range(4):
   for values,dest in ((responses,original),(reversals,reverse)):
    event=values[i*4+c];pairs=json.loads(event['final'],object_pairs_hook=lambda x:x)
    if event['query_id']!=row['query_id'] or event['start']!=8*c or len(pairs)!=8 or [x[0] for x in pairs]!=list(map(str,range(1,9))) or any(x[1] not in ('Y','N') for x in pairs):raise ValueError('Recorded keyed output mismatch')
    dest.extend(v for _,v in pairs)
  k=sum(it['gold']=='Y' for it in row['items']);group='all_negative' if k==0 else 'all_positive' if k==32 else 'mixed';older=abs(original.count('Y')-k);newer=abs(prediction['qwen'].count('Y')-k)
  grouped[group].append((older,newer,newer-older));routes[group]+=sum(x!=y for x,y in zip(original,reverse))
 for j,name in enumerate(('all_negative','mixed','all_positive')):
  if name not in grouped:
   if name in out['groups']:raise ValueError('Unexpected group')
   continue
  g=grouped[name];rep=out['groups'][name];n=len(g);d=np.array([x[2] for x in g],dtype=np.int64)
  if rep['pools']!=n or rep['routed_candidates']!=routes[name] or rep['bootstrap_seed']!=20271224+j:raise ValueError('Group size or route mismatch')
  check(sum(x[0] for x in g)/(32*n),rep['keyed_mae'],name+' keyed');check(sum(x[1] for x in g)/(32*n),rep['pointwise_fallback_mae'],name+' fallback');check(sum(d)/(32*n),rep['fallback_minus_keyed'],name+' difference');check(sum(d)/(32*96),rep['weighted_contribution_to_overall'],name+' contribution')
  draws=np.random.default_rng(20271224+j).integers(0,n,size=(4000,n));boot=np.array([sum(int(d[x]) for x in sample)/(32*n) for sample in draws]);ci=np.quantile(boot,[.025,.975])
  for x,y in zip(ci,rep['unadjusted_posthoc_query_bootstrap_95']):check(x,y,name+' interval')
 check(sum(sum(x[2] for x in g) for g in grouped.values())/(32*96),out['overall']['difference'],'overall');print('Independent integer-count and query-resampling checks passed for all 96 pools.')
if __name__=='__main__':main()
