#!/usr/bin/env python3
"""Independently verify stored embedding associations, scores and calibration."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
SPLITS=[('train160','run_wands_160',None),('holdout160','wands_holdout_selection',20271221),
        ('fallback96','wands_fallback_validation_selection',20271222),('selective52','wands_selective_execution_selection',20271223)]
PREFIX='Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:'

def lines(p):
 with p.open(encoding='utf8') as f:return [json.loads(x) for x in f]
def counts(rows,values):
 out=[]
 for row,pred in zip(rows,values):
  g=np.asarray([it['gold']=='Y' for it in row['items']]);p=np.asarray([x=='Y' for x in pred]);assert len(g)==len(p)==32
  out.append([int((g&p).sum()),int((~g&p).sum()),int((~g&~p).sum()),int((g&~p).sum()),abs(int(g.sum())-int(p.sum()))/32])
 return np.asarray(out,dtype=float)
def quality(a):
 tp,fp,tn,fn=a[:,:4].sum(axis=0);return {'tp':int(tp),'fp':int(fp),'tn':int(tn),'fn':int(fn),
  'f1':float(2*tp/(2*tp+fp+fn)) if (2*tp+fp+fn)>0 else 0.0,
  'balanced_accuracy':float((tp/(tp+fn)+tn/(tn+fp))/2),'ratio_mae':float(a[:,4].mean())}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);a=ap.parse_args()
 summary=json.loads((a.run/'summary.json').read_text());index=lines(a.run/'embedding_index.jsonl');saved=lines(a.run/'predictions.jsonl')
 with np.load(a.run/'embeddings.npz',allow_pickle=False) as f:qv=f['queries'];dv=f['documents']
 assert qv.dtype==dv.dtype==np.float32 and qv.shape[1]==dv.shape[1]==1024
 assert np.allclose(np.linalg.norm(qv,axis=1),1,atol=1e-5) and np.allclose(np.linalg.norm(dv,axis=1),1,atol=1e-5)
 datasets=[(n,lines(HERE/p/'inputs.jsonl'),seed) for n,p,seed in SPLITS]
 qmap={};dmap={};idx=0;scores={}
 for name,rows,seed in datasets:
  block=[]
  for row in rows:
   q=PREFIX+row['query']
   if q not in qmap:qmap[q]=len(qmap)
   did=[]
   for it in row['items']:
    text=it['text']
    if text not in dmap:dmap[text]=len(dmap)
    did.append(dmap[text])
   rec=index[idx];idx+=1
   assert rec=={'sample':name,'query_id':row['query_id'],'query_index':qmap[q],'document_indices':did}
   block.append(np.sum(dv[did]*qv[qmap[q]],axis=1,dtype=np.float32))
  scores[name]=np.stack(block)
 assert idx==len(index) and len(qmap)==qv.shape[0] and len(dmap)==dv.shape[0]
 train=datasets[0][1];trainq={r['query'].strip().casefold() for r in train};trainid={i['product_id'] for r in train for i in r['items']};traintext={i['text'] for r in train for i in r['items']}
 y=np.asarray([i['gold']=='Y' for r in train for i in r['items']]);vals=scores['train160'].ravel();best=(-1.0,None)
 for t in np.unique(vals)[::-1]:
  p=vals>=t;tp=int((y&p).sum());fp=int((~y&p).sum());fn=int((y&~p).sum());f=2*tp/(2*tp+fp+fn)
  if f>best[0]:best=(f,float(t),tp,fp,fn)
 threshold=summary['training']['threshold'];assert best[1]==threshold and best[2:]==(summary['training']['tp'],summary['training']['fp'],summary['training']['fn']) and abs(best[0]-summary['training']['f1'])<1e-12
 prior=lines(HERE/'run_wands_supervised_reference/predictions.jsonl')
 for name,rows,seed in datasets[1:]:
  assert not trainq&{r['query'].strip().casefold() for r in rows}
  assert not trainid&{i['product_id'] for r in rows for i in r['items']}
  assert not traintext&{i['text'] for r in rows for i in r['items']}
  p=[r for r in prior if r['sample']==name];actual=[r for r in saved if r['sample']==name]
  assert len(p)==len(actual)==len(rows) and all(x['query_id']==y['query_id']==z['query_id'] for x,y,z in zip(p,actual,rows))
  embedding=[['Y' if float(v)>=threshold else 'N' for v in row] for row in scores[name]]
  for i,x in enumerate(actual):
   assert x['embedding']==embedding[i] and np.allclose(x['scores'],scores[name][i],rtol=0,atol=1e-7)
  methods={'embedding':embedding,'qwen':[r['qwen'] for r in p],'lexical':[r['lexical'] for r in p],'supervised_lexical':[r['supervised'] for r in p]}
  z={k:counts(rows,v) for k,v in methods.items()}
  for k,arr in z.items():
   for k2,v in quality(arr).items():assert abs(v-summary['samples'][name]['quality'][k][k2])<1e-12,(name,k,k2)
  draw=np.random.default_rng(seed).integers(0,len(rows),size=(4000,len(rows)))
  def f1(v):
   c=v[draw,:4].sum(axis=1);num=2*c[:,0];den=num+c[:,1]+c[:,3]
   return np.divide(num,den,out=np.zeros_like(num),where=den!=0)
  df=f1(z['embedding'])-f1(z['qwen']);dm=z['embedding'][draw,4].mean(axis=1)-z['qwen'][draw,4].mean(axis=1)
  cmp=summary['samples'][name]['embedding_minus_qwen']
  assert np.allclose(np.quantile(df,[.025,.975]),cmp['f1_difference_ci'],rtol=0,atol=1e-12)
  assert np.allclose(np.quantile(dm,[.025,.975]),cmp['ratio_mae_difference_ci'],rtol=0,atol=1e-12)
  print(name,'queries',len(rows),'no train query, product or exact-text overlap; metrics verified')
 print('Saved vector norms, index reconstruction, all scores, training threshold and bootstrap independently verified.')
if __name__=='__main__':main()
