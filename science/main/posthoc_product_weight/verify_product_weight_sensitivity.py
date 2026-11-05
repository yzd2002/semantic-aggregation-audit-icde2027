#!/usr/bin/env python3
"""Independent recomputation of every query-product weighted contrast."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np

def read(path):
 with path.open(encoding='utf8') as f:return [json.loads(x) for x in f]
def main():
 p=argparse.ArgumentParser();p.add_argument('--artifact',type=Path,required=True);p.add_argument('--work',type=Path,required=True);p.add_argument('--result',type=Path,required=True);a=p.parse_args();result=json.loads(a.result.read_text())
 for name,selection,oldname,newname,comparison in [('ESCI','esci_selective_holdout_selection','run_esci_7b','run_esci_7b_specguided','esci_specguided_comparison.json'),('WANDS','wands_selective_execution_selection','run_wands_7b','run_wands_7b_specguided','wands_specguided_comparison.json')]:
  path=a.artifact/selection/'inputs.jsonl';raw=read(path);saved=json.loads((a.work/comparison).read_text());assert hashlib.sha256(path.read_bytes()).hexdigest()==saved['selection_inputs_sha256']
  old={str(x['query_id']):x['pred'] for x in read(a.work/oldname/'selective_outputs.jsonl')};new={str(x['query_id']):x['pred'] for x in read(a.work/newname/'selective_outputs.jsonl')}
  ids=list(map(str,saved['common_query_ids']));rows={str(x['query_id']):x for x in raw};assert len(rows)==len(raw) and set(rows)==set(old)==set(new)
  products=sorted({str(item['product_id']) for q in ids for item in rows[q]['items']});lookup={p:i for i,p in enumerate(products)}
  q_indices=[];p_indices=[];reference=[];before=[];after=[]
  for n,q in enumerate(ids):
   items=rows[q]['items'];assert len(items)==len(old[q])==len(new[q]) and [i['id'] for i in items]==list(range(1,len(items)+1))
   for item,x,y in zip(items,old[q],new[q]):
    assert item['gold'] in ('Y','N') and x in ('Y','N') and y in ('Y','N');q_indices.append(n);p_indices.append(lookup[str(item['product_id'])]);reference.append(float(item['gold']=='Y'));before.append(float(x=='Y'));after.append(float(y=='Y'))
  q_indices=np.array(q_indices);p_indices=np.array(p_indices);reference=np.array(reference);before=np.array(before);after=np.array(after);n=len(ids)
  lens=np.bincount(q_indices,minlength=n);gold=np.bincount(q_indices,weights=reference,minlength=n)/lens
  x=np.bincount(q_indices,weights=before,minlength=n)/lens;y=np.bincount(q_indices,weights=after,minlength=n)/lens
  old_mae=float(np.mean(abs(x-gold)));new_mae=float(np.mean(abs(y-gold)));entry=result['datasets'][name]
  assert entry['pools']==n and entry['candidates']==len(reference) and entry['distinct_products']==len(products)
  assert entry['repeated_products']==sum(v>1 for v in np.bincount(p_indices,minlength=len(products)))
  assert abs(entry['old_unguided_mae']-old_mae)<1e-12 and abs(entry['new_schema_guided_mae']-new_mae)<1e-12
  rng=np.random.default_rng(entry['seed']);values=[]
  for _ in range(entry['draws']):
   qdraw=rng.integers(0,n,n);weights=rng.exponential(1.,len(products))[p_indices];den=np.bincount(q_indices,weights=weights,minlength=n);assert np.all(den>0)
   g=np.bincount(q_indices,weights=weights*reference,minlength=n)/den
   x=np.bincount(q_indices,weights=weights*before,minlength=n)/den;y=np.bincount(q_indices,weights=weights*after,minlength=n)/den
   counts=np.bincount(qdraw,minlength=n);values.append(float(np.dot(counts,abs(y-g)-abs(x-g))/n))
  if not np.allclose(values,entry['reweighted_differences'],rtol=0,atol=1e-12):raise ValueError('Weighted draws differ for '+name)
  if not np.allclose(np.percentile(values,[2.5,97.5]),entry['sensitivity_range_95'],rtol=0,atol=1e-12):raise ValueError('Weighted percentiles differ for '+name)
  print(name,'verified',n,'queries and',len(values),'weighted draws')
if __name__=='__main__':main()
