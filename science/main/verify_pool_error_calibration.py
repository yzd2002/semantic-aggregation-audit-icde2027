#!/usr/bin/env python3
"""Independent score-count reconstruction and finite conformal-rank enumeration."""
import argparse,hashlib,itertools,json,math
from fractions import Fraction
from pathlib import Path

def read(p):
 with p.open(encoding='utf8') as f:return [json.loads(x) for x in f if x.strip()]
def close(x,y,name):
 if not math.isclose(float(x),float(y),abs_tol=1e-12,rel_tol=0):raise ValueError(name+' mismatch')
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--artifact',type=Path,required=True);ap.add_argument('--result',type=Path,required=True);a=ap.parse_args();root=a.artifact;ref=json.loads(a.result.read_text())
 for name,expected in ref['input_sha256'].items():
  with (root/name).open('rb') as f:actual=hashlib.file_digest(f,'sha256').hexdigest()
  if actual!=expected:raise ValueError('Input SHA-256 mismatch: '+name)
 specs=[('cal160','wands_holdout_selection/inputs.jsonl','run_keyed_order_qwen_v1/original_outputs.jsonl'),('test96','wands_fallback_validation_selection/inputs.jsonl','run_wands_fallback_validation_qwen_v1/original_responses.jsonl'),('test52','wands_selective_execution_selection/inputs.jsonl','run_wands_selective_execution_qwen_v1/original_responses.jsonl')];samples={}
 for name,src,out in specs:
  rows=read(root/src);events=read(root/out);records=[]
  for i,row in enumerate(rows):
   chunks=events[i]['raw'] if name=='cal160' else events[4*i:4*i+4];v=[]
   for j,c in enumerate(chunks):
    pairs=json.loads(c['final'],object_pairs_hook=lambda x:x)
    if c['start']!=8*j or c['size']!=8 or [x[0] for x in pairs]!=list(map(str,range(1,9))) or any(x[1] not in ('Y','N') for x in pairs):raise ValueError('Keyed output')
    v.extend(x[1] for x in pairs)
   g=[z['gold'] for z in row['items']]
   if len(v)!=32 or len(g)!=32:raise ValueError('Pool size')
   p=v.count('Y');actual=g.count('Y');mistakes=sum(x!=y for x,y in zip(v,g));res=abs(p-actual)
   if res>mistakes:raise ValueError('Candidate count inequality')
   records.append((p,actual,mistakes,res))
  samples[name]=records
 cal=samples['cal160'];rank=math.ceil(Fraction(9,10)*(len(cal)+1));k1=sorted(x[2] for x in cal)[rank-1];k2=sorted(x[3] for x in cal)[rank-1]
 if ref['calibration']['order_statistic_rank']!=rank or (k1,k2)!=(ref['calibration']['candidate_error_threshold'],ref['calibration']['ratio_residual_threshold']):raise ValueError('Quantile mismatch')
 for name in ('test96','test52'):
  s=samples[name]
  for label,k,idx in [('candidate_error',k1,2),('ratio_residual',k2,3)]:
   expected=ref['test'][name][label];hits=budget=width=full=0
   for p,g,e,res in s:
    lo=max(0,p-k);hi=min(32,p+k);hits+=lo<=g<=hi;budget+=(e if idx==2 else res)<=k;width+=hi-lo;full+=hi-lo==32
   if (expected['interval_covered'],expected['score_at_most_threshold'],expected['full_width_pools'])!=(hits,budget,full):raise ValueError(name+' coverage mismatch')
   close(expected['mean_width'],width/(32*len(s)),name+' width')
  for p,g,e,res in s:
   if not max(0,p-k1)<=max(0,p-k2)<=min(32,p+k2)<=min(32,p+k1):raise ValueError('Containment')
 # For each possible vector, uniformly holding out one position is exchangeable.
 # Check discrete ties, finite correction and the conservative bounded fallback.
 rank_cases=0
 for n in range(1,6):
  for full in itertools.product(range(4),repeat=n+1):
   for alpha in (Fraction(1,10),Fraction(1,4),Fraction(1,2)):
    k=math.ceil((1-alpha)*(n+1));hits=0
    for j in range(n+1):
     calibration=sorted(full[:j]+full[j+1:]);threshold=32 if k>n else calibration[k-1];hits+=full[j]<=threshold
    if Fraction(hits,n+1)<1-alpha:raise ValueError('Finite-rank coverage')
    rank_cases+=1
 # Verify the residual inequality and interval nesting for every binary vector pair up to n=6.
 vector_cases=0
 for n in range(1,7):
  for v in itertools.product((0,1),repeat=n):
   for y in itertools.product((0,1),repeat=n):
    p=sum(v);g=sum(y);e=sum(x!=z for x,z in zip(v,y));res=abs(p-g)
    if res>e:raise ValueError('Binary error bound')
    for ke in range(n+1):
     for kr in range(ke+1):
      if not max(0,p-ke)<=max(0,p-kr)<=min(n,p+kr)<=min(n,p+ke):raise ValueError('Interval nesting')
      if e<=ke and not max(0,p-ke)<=g<=min(n,p+ke):raise ValueError('Deterministic bound')
    vector_cases+=1
 print('Independent WANDS scores verified; finite-rank/tie cases:',rank_cases,'binary-vector cases:',vector_cases)
if __name__=='__main__':main()
