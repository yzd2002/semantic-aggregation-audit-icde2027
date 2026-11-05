#!/usr/bin/env python3
"""Independently recompute paired schema-guided versus unguided results."""
import argparse,hashlib,json,math
from fractions import Fraction
from pathlib import Path
import numpy as np

def read(p):
 with p.open(encoding='utf8') as f:return [json.loads(x) for x in f]
def close(a,b):return math.isclose(float(a),float(b),rel_tol=1e-11,abs_tol=1e-11)
def audit(rows,old,new,old_ids,new_ids,saved):
 od={x['query_id']:x['pred'] for x in old};nd={x['query_id']:x['pred'] for x in new}
 if len(od)!=len(old) or len(nd)!=len(new):raise ValueError('Duplicate response ID')
 old_set=set(old_ids);new_set=set(new_ids);errors=[];old_e=[];new_e=[];old_s=[];new_s=[];base=[];ids=[]
 for row in rows:
  q=row['query_id']
  if q not in old_set or q not in new_set:continue
  ref=[v['gold'] for v in row['items']];a=od[q];b=nd[q]
  if a is None or b is None or len(a)!=len(ref) or len(b)!=len(ref):raise ValueError('Invalid common-pool lengths')
  r=Fraction(ref.count('Y'),len(ref));x=Fraction(a.count('Y'),len(ref));y=Fraction(b.count('Y'),len(ref))
  old_e.append(abs(x-r));new_e.append(abs(y-r));old_s.append(x-r);new_s.append(y-r);base.append(1-r);errors.append(abs(y-r)-abs(x-r));ids.append(q)
 if saved['common_query_ids']!=ids or saved['common_pools']!=len(ids):raise ValueError('Common query IDs disagree')
 if not ids:
  if saved['comparison'] is not None:raise ValueError('Empty intersection has a comparison')
  return
 n=len(ids);c=saved['comparison']
 for k,v in [('old_unguided_mae',sum(old_e)/n),('new_schema_guided_mae',sum(new_e)/n),('difference_new_minus_old',sum(errors)/n),('all_y_mae',sum(base)/n),('old_signed_ratio_bias',sum(old_s)/n),('new_signed_ratio_bias',sum(new_s)/n)]:
  if not close(c[k],v):raise ValueError('Metric differs: '+k)
 rng=np.random.default_rng(saved['seed']);draws=[]
 for _ in range(saved['resamples']):
  ix=rng.integers(0,n,size=n);draws.append(sum(float(errors[j]) for j in ix)/n)
 interval=np.quantile(draws,[.025,.975])
 if not all(close(a,b) for a,b in zip(c['paired_query_95_percentile_ci'],interval)):raise ValueError('Bootstrap interval differs')

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for name in ('selection','old-run','old-analysis','new-run','new-analysis','result'):p.add_argument('--'+name,type=Path,required=True)
 a=p.parse_args();rows=read(a.selection/'inputs.jsonl');digest=hashlib.sha256((a.selection/'inputs.jsonl').read_bytes()).hexdigest()
 old=json.loads(a.old_analysis.read_text());new=json.loads(a.new_analysis.read_text());saved=json.loads(a.result.read_text())
 if any(x['selection_inputs_sha256']!=digest for x in (old,new,saved)):raise ValueError('Selection hash mismatch')
 audit(rows,read(a.old_run/'selective_outputs.jsonl'),read(a.new_run/'selective_outputs.jsonl'),old['common_query_ids'],new['common_query_ids'],saved)
 print('Independent rational-arithmetic comparison and paired bootstrap passed.')
if __name__=='__main__':main()
