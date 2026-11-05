#!/usr/bin/env python3
"""Compare two observed 7B selective pipelines on their shared complete pools."""
import argparse,hashlib,json
from fractions import Fraction
from pathlib import Path
import numpy as np

def lines(path):
 with path.open(encoding='utf8') as f:return [json.loads(x) for x in f]

def compare(rows,old,new,old_common,new_common,resamples=4000,seed=20271101):
 ids=[r['query_id'] for r in rows]
 if len(ids)!=len(set(ids)):raise ValueError('Duplicate selected query ID')
 def mapping(records):
  d={x['query_id']:x['pred'] for x in records}
  if len(d)!=len(records) or set(d)!=set(ids):raise ValueError('Output query IDs do not match sample')
  return d
 om,nm=mapping(old),mapping(new);common=set(old_common)&set(new_common)
 if not common.issubset(ids):raise ValueError('A common query is not in selection')
 selected=[r for r in rows if r['query_id'] in common]
 out={'attempted_pools':len(rows),'common_pools':len(selected),'common_query_ids':[x['query_id'] for x in selected],
      'resamples':resamples,'seed':seed,'comparison':None}
 if not selected:return out
 old_error=[];new_error=[];baseline=[];old_signed=[];new_signed=[]
 for row in selected:
  gold=[x['gold'] for x in row['items']];left=om[row['query_id']];right=nm[row['query_id']]
  if not gold or any(x not in ('Y','N') for x in gold) or any(v is None or len(v)!=len(gold) or any(x not in ('Y','N') for x in v) for v in (left,right)):
   raise ValueError('A common pool has invalid or misaligned outputs')
  n=len(gold);r=Fraction(gold.count('Y'),n);a=Fraction(left.count('Y'),n);b=Fraction(right.count('Y'),n)
  old_error.append(float(abs(a-r)));new_error.append(float(abs(b-r)));old_signed.append(float(a-r));new_signed.append(float(b-r));baseline.append(float(1-r))
 old_error=np.asarray(old_error);new_error=np.asarray(new_error);delta=new_error-old_error
 idx=np.random.default_rng(seed).integers(0,len(selected),size=(resamples,len(selected)))
 out['comparison']={'old_unguided_mae':float(old_error.mean()),'new_schema_guided_mae':float(new_error.mean()),
  'difference_new_minus_old':float(delta.mean()),'paired_query_95_percentile_ci':[float(x) for x in np.quantile(delta[idx].mean(axis=1),[.025,.975])],
  'all_y_mae':float(np.mean(baseline)),'old_signed_ratio_bias':float(np.mean(old_signed)),'new_signed_ratio_bias':float(np.mean(new_signed))}
 return out

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for name in ('selection','old-run','old-analysis','new-run','new-analysis','out'):p.add_argument('--'+name,type=Path,required=True)
 a=p.parse_args()
 if a.out.exists():raise FileExistsError(a.out)
 rows=lines(a.selection/'inputs.jsonl');digest=hashlib.sha256((a.selection/'inputs.jsonl').read_bytes()).hexdigest()
 analyses=[json.loads(x.read_text()) for x in (a.old_analysis,a.new_analysis)]
 if any(x['selection_inputs_sha256']!=digest for x in analyses):raise ValueError('Input selection hash mismatch')
 result=compare(rows,lines(a.old_run/'selective_outputs.jsonl'),lines(a.new_run/'selective_outputs.jsonl'),analyses[0]['common_query_ids'],analyses[1]['common_query_ids'])
 result.update({'selection_inputs_sha256':digest,'interpretation':'Post-hoc whole-pipeline comparison on exposed queries; not a causal or user-satisfaction claim.'})
 a.out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf8');print('Guided-versus-unguided comparison: common pools',result['common_pools'])
if __name__=='__main__':main()
