#!/usr/bin/env python3
"""Exploratory shared-product exponential reweighting for 7B pipelines."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np

def lines(path):
 with path.open(encoding='utf8') as f:return [json.loads(x) for x in f]
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def assemble(rows,old,new,common):
 ids=[str(x['query_id']) for x in rows]
 if len(ids)!=len(set(ids)):raise ValueError('Duplicate query ID')
 def by_id(records):
  result={str(r['query_id']):r['pred'] for r in records}
  if len(records)!=len(result) or set(result)!=set(ids):raise ValueError('Output IDs differ from selected queries')
  return result
 left,right=by_id(old),by_id(new);wanted=set(map(str,common));chosen=[r for r in rows if str(r['query_id']) in wanted]
 if [str(x['query_id']) for x in chosen]!=list(map(str,common)):raise ValueError('Common ID order differs')
 products=sorted({str(i['product_id']) for row in chosen for i in row['items']});index={x:i for i,x in enumerate(products)}
 offsets=[];pids=[];gold=[];a=[];b=[];pos=0
 for row in chosen:
  q=str(row['query_id']);items=row['items'];x=left[q];y=right[q]
  if not items or len(x)!=len(items) or len(y)!=len(items):raise ValueError('Incomplete query')
  if any(v not in ('Y','N') for v in x+y+[it['gold'] for it in items]):raise ValueError('Nonbinary labels')
  if [it['id'] for it in items]!=list(range(1,len(items)+1)):raise ValueError('Candidate positions differ')
  offsets.append(pos);pos+=len(items)
  for it,aa,bb in zip(items,x,y):pids.append(index[str(it['product_id'])]);gold.append(it['gold']=='Y');a.append(aa=='Y');b.append(bb=='Y')
 return chosen,products,np.asarray(offsets),np.asarray(pids),np.asarray(gold,float),np.asarray(a,float),np.asarray(b,float)
def run(rows,old,new,comparison,seed,draws=4000):
 chosen,products,offsets,pids,gold,left,right=assemble(rows,old,new,comparison['common_query_ids'])
 lengths=np.diff(np.r_[offsets,len(pids)]);gr=np.add.reduceat(gold,offsets)/lengths;lr=np.add.reduceat(left,offsets)/lengths;rr=np.add.reduceat(right,offsets)/lengths
 old_mae=float(np.mean(np.abs(lr-gr)));new_mae=float(np.mean(np.abs(rr-gr)));saved=comparison['comparison']
 if abs(old_mae-saved['old_unguided_mae'])>1e-12 or abs(new_mae-saved['new_schema_guided_mae'])>1e-12:raise ValueError('Unweighted MAE differs from verified comparison')
 rng=np.random.default_rng(seed);values=[]
 for _ in range(draws):
  qdraw=rng.integers(0,len(chosen),len(chosen));weight=rng.exponential(1.,len(products));w=weight[pids];den=np.add.reduceat(w,offsets)
  if np.any(den<=0):raise RuntimeError('Product weight missing')
  g=np.add.reduceat(w*gold,offsets)/den;l=np.add.reduceat(w*left,offsets)/den;r=np.add.reduceat(w*right,offsets)/den
  values.append(float((np.abs(r-g)-np.abs(l-g))[qdraw].mean()))
 counts=np.bincount(pids,minlength=len(products));lo,hi=np.percentile(values,[2.5,97.5])
 return {'pools':len(chosen),'candidates':len(pids),'distinct_products':len(products),'repeated_products':int(np.sum(counts>1)),
         'old_unguided_mae':old_mae,'new_schema_guided_mae':new_mae,'difference_new_minus_old':new_mae-old_mae,
         'sensitivity_range_95':[float(lo),float(hi)],'draws':draws,'seed':seed,'reweighted_differences':values}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--artifact',type=Path,required=True);ap.add_argument('--work',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
 if a.out.exists():raise FileExistsError(a.out)
 result={'protocol':'PROTOCOL_product_weight_sensitivity.md','interpretation':'Post-hoc product weighting changes the estimand; not a fixed-pool confidence interval.','datasets':{}}
 for label,sel,old,new,comparison,seed in [('ESCI','esci_selective_holdout_selection','run_esci_7b','run_esci_7b_specguided','esci_specguided_comparison.json',20271104),('WANDS','wands_selective_execution_selection','run_wands_7b','run_wands_7b_specguided','wands_specguided_comparison.json',20271105)]:
  inp=a.artifact/sel/'inputs.jsonl';comp=json.loads((a.work/comparison).read_text());assert sha(inp)==comp['selection_inputs_sha256']
  result['datasets'][label]=run(lines(inp),lines(a.work/old/'selective_outputs.jsonl'),lines(a.work/new/'selective_outputs.jsonl'),comp,seed)
 a.out.write_text(json.dumps(result,indent=2)+'\n');
 for label,x in result['datasets'].items():print(label,'pools',x['pools'],'candidates',x['candidates'],'shared products',x['repeated_products'],'weighted range',x['sensitivity_range_95'])
if __name__=='__main__':main()
