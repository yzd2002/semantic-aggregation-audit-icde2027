#!/usr/bin/env python3
"""Independently verify the training-ratio threshold and WANDS-52 ranges."""
import argparse,bisect,collections,json,math
from pathlib import Path
import numpy as np

def lines(p):
 with p.open(encoding='utf8') as f:return [json.loads(x) for x in f if x.strip()]
def close(x,y,label):
 if not math.isclose(float(x),float(y),abs_tol=1e-12,rel_tol=0):raise ValueError(label+' mismatch')
def main():
 p=argparse.ArgumentParser();p.add_argument('--artifact',type=Path,required=True);p.add_argument('--result',type=Path,required=True);a=p.parse_args();r=a.artifact;ref=json.loads(a.result.read_text())
 train=lines(r/'run_wands_160/inputs.jsonl');ix=lines(r/'run_wands_embedding_reference/embedding_index.jsonl')[:160]
 with np.load(r/'run_wands_embedding_reference/embeddings.npz',allow_pickle=False) as z:q=z['queries'];d=z['documents']
 scores=[];truth=[];values=set()
 for row,rec in zip(train,ix):
  if rec['sample']!='train160' or rec['query_id']!=row['query_id'] or len(rec['document_indices'])!=32:raise ValueError('Training alignment')
  vals=sorted(float(x) for x in np.sum(d[rec['document_indices']]*q[rec['query_index']],axis=1,dtype=np.float32));scores.append(vals);values.update(vals);truth.append(sum(i['gold']=='Y' for i in row['items']))
 candidates=[math.nextafter(max(values),math.inf)]+sorted(values,reverse=True)
 loss=[sum(abs((32-bisect.bisect_left(s,t))-g) for s,g in zip(scores,truth)) for t in candidates]
 best=min(range(len(candidates)),key=lambda i:loss[i]);threshold=candidates[best]
 close(threshold,ref['threshold_selection']['threshold'],'training threshold');close(loss[best]/(160*32),ref['threshold_selection']['training_selected_ratio_mae'],'training error')
 rows=lines(r/'wands_selective_execution_selection/inputs.jsonl');em=[x for x in lines(r/'run_wands_embedding_reference/predictions.jsonl') if x['sample']=='selective52'];qp=[x for x in lines(r/'run_wands_supervised_reference/predictions.jsonl') if x['sample']=='selective52']
 if len(rows)!=52 or len(em)!=52 or len(qp)!=52 or any(x['query_id']!=y['query_id'] or x['query_id']!=z['query_id'] for x,y,z in zip(rows,em,qp)):raise ValueError('Test alignment')
 gold=[];qwen=[];hard=[]
 for row,e,pred in zip(rows,em,qp):
  gold.append(sum(x['gold']=='Y' for x in row['items'])/32);qwen.append(sum(x=='Y' for x in pred['qwen'])/32);hard.append(sum(x>=threshold for x in e['scores'])/32)
 err=[abs(h-g)-abs(qw-g) for g,qw,h in zip(gold,qwen,hard)];rng=np.random.default_rng(20271223);draw=rng.integers(0,52,size=(4000,52));qinterval=np.quantile([math.fsum(err[int(i)] for i in a)/52 for a in draw],[.025,.975])
 for x,y in zip(qinterval,ref['samples']['selective52']['query_bootstrap']['ratio_minus_qwen']['interval_95']):close(x,y,'query interval')
 products=sorted({int(x['product_id']) for row in rows for x in row['items']});lookup={v:i for i,v in enumerate(products)};rng=np.random.default_rng(20271201);vals=[]
 for _ in range(4000):
  qc=collections.Counter(int(x) for x in rng.integers(0,52,52));pc=collections.Counter(int(x) for x in rng.integers(0,len(products),len(products)));total=0.0
  for index,count in qc.items():
   row=rows[index];e=em[index];pred=qp[index];den=ng=nq=nh=0
   for k,item in enumerate(row['items']):
    weight=pc.get(lookup[int(item['product_id'])],0);den+=weight;ng+=weight*(item['gold']=='Y');nq+=weight*(pred['qwen'][k]=='Y');nh+=weight*(e['scores'][k]>=threshold)
   if not den:raise ValueError('Zero-weight selected query')
   total+=count*(abs(nh/den-ng/den)-abs(nq/den-ng/den))
  vals.append(total/52)
 for x,y in zip(np.quantile(vals,[.025,.975]),ref['samples']['selective52']['product_reweighted_ranges']['ratio_minus_qwen']):close(x,y,'product range')
 print('Independent sorted-score threshold and 52-query bootstrap/product ranges verified.')
if __name__=='__main__':main()
