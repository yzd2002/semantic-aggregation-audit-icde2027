#!/usr/bin/env python3
"""Evaluate a training-ratio threshold for saved WANDS embeddings."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
SETS=[('holdout160','wands_holdout_selection',20271221,20261031,160),
      ('fallback96','wands_fallback_validation_selection',20271222,20271128,96),
      ('selective52','wands_selective_execution_selection',20271223,20271201,52)]
def lines(path):
 with path.open(encoding='utf8') as f:return [json.loads(x) for x in f if x.strip()]
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def metrics(g,p):
 g=np.asarray(g,dtype=bool);p=np.asarray(p,dtype=bool)
 if g.shape!=p.shape:raise ValueError('Prediction shape')
 tp=int((g&p).sum());tn=int((~g&~p).sum());fp=int((~g&p).sum());fn=int((g&~p).sum())
 return {'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.0,
         'balanced_accuracy':(tp/(tp+fn)+tn/(tn+fp))/2 if tp+fn and tn+fp else None,
         'ratio_mae':float(np.abs(p.mean(axis=1)-g.mean(axis=1)).mean()),
         'tp':tp,'fp':fp,'tn':tn,'fn':fn}
def select_threshold(scores,gold):
 if scores.ndim!=2 or scores.shape!=gold.shape or not np.isfinite(scores).all():raise ValueError('Unexpected training scores')
 scores=np.asarray(scores,dtype=np.float64);n,k=scores.shape;counts=np.zeros(n,dtype=np.int64);g=gold.sum(axis=1).astype(np.int64)
 order=np.argsort(-scores.ravel(),kind='stable');flat=scores.ravel();best=int(g.sum());t=float(np.nextafter(flat[order[0]],np.inf));i=0
 while i<len(order):
  j=i+1;value=flat[order[i]]
  while j<len(order) and flat[order[j]]==value:j+=1
  for position in order[i:j]:counts[int(position)//k]+=1
  loss=int(np.abs(counts-g).sum())
  if loss<best:best=loss;t=float(value)
  i=j
 values=np.concatenate(([np.nextafter(flat[order[0]],np.inf)],np.unique(flat)[::-1]));direct=np.asarray([np.abs((scores>=v).sum(axis=1)-g).sum() for v in values]);idx=int(np.argmin(direct))
 if int(direct[idx])!=best or float(values[idx])!=t:raise ValueError('Independent direct threshold sweep failed')
 return t,best/(n*k),len(values)
def main():
 p=argparse.ArgumentParser();p.add_argument('--artifact',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();root=a.artifact.resolve()
 emb=root/'run_wands_embedding_reference';trainpath=root/'run_wands_160/inputs.jsonl';idxpath=emb/'embedding_index.jsonl';npzpath=emb/'embeddings.npz';testpredpath=emb/'predictions.jsonl';priorpath=emb/'summary.json';qpath=root/'run_wands_supervised_reference/predictions.jsonl';crosspath=root/'run_wands_ratio_product_sensitivity/summary.json'
 train=lines(trainpath);indices=lines(idxpath);testpred=lines(testpredpath);prior=json.loads(priorpath.read_text());qpred=lines(qpath);cross=json.loads(crosspath.read_text())
 sources=[trainpath,idxpath,npzpath,testpredpath,priorpath,qpath,crosspath];data=[('train160',train)]
 for name,selection,seed,productseed,n in SETS:
  path=root/selection/'inputs.jsonl';sources.append(path);data.append((name,lines(path)))
 if len(train)!=160 or len(indices)!=468:raise ValueError('Unexpected dataset counts')
 trainq={r['query'].strip().casefold() for r in train};trainp={str(i['product_id']) for r in train for i in r['items']};traint={i['text'] for r in train for i in r['items']}
 if len(trainq)!=160:raise ValueError('Repeated training queries')
 for name,rows in data:
  if any(len(r['items'])!=32 or any(i['gold'] not in ('Y','N') or (i['gold']=='Y')!=(i['grade']=='Exact') for i in r['items']) for r in rows):raise ValueError('Gold or pool mismatch '+name)
  if name!='train160' and (trainq&{r['query'].strip().casefold() for r in rows} or trainp&{str(i['product_id']) for r in rows for i in r['items']} or traint&{i['text'] for r in rows for i in r['items']}):raise ValueError('Train/test overlap '+name)
 with np.load(npzpath,allow_pickle=False) as z:qv=z['queries'];dv=z['documents']
 if qv.ndim!=2 or dv.ndim!=2 or qv.shape[1]!=dv.shape[1] or qv.dtype!=np.float32 or dv.dtype!=np.float32:raise ValueError('Unexpected vector representation')
 scores={};offset=0
 for name,rows in data:
  vals=[]
  for row in rows:
   ix=indices[offset];offset+=1
   if ix['sample']!=name or ix['query_id']!=row['query_id'] or len(ix['document_indices'])!=32:raise ValueError('Embedding index mismatch')
   vals.append(np.sum(dv[ix['document_indices']]*qv[ix['query_index']],axis=1,dtype=np.float32))
  scores[name]=np.stack(vals).astype(np.float64)
 gold=np.array([[i['gold']=='Y' for i in r['items']] for r in train],dtype=bool)
 t,trainmae,nthresholds=select_threshold(scores['train160'],gold);f1t=prior['training']['threshold'];oldtrain=metrics(gold,scores['train160']>=f1t)
 if abs(oldtrain['f1']-prior['training']['f1'])>1e-12:raise ValueError('Original training F1 mismatch')
 result={'scope':'post-hoc, train-only threshold applied to exposed WANDS samples; no new model inference',
         'threshold_selection':{'threshold':t,'training_selected_ratio_mae':trainmae,'candidate_thresholds':nthresholds,'train_pools':160,'training_labels':5120,'tie_rule':'largest threshold','original_f1_threshold':f1t},'samples':{},'input_sha256':{}}
 for name,selection,seed,pseed,n in SETS:
  rows=dict(data)[name];stored=[x for x in testpred if x['sample']==name];q=[x for x in qpred if x['sample']==name]
  if len(rows)!=n or len(stored)!=n or len(q)!=n or any(r['query_id']!=s['query_id'] or r['query_id']!=v['query_id'] for r,s,v in zip(rows,stored,q)):raise ValueError(name+' alignment')
  p=np.asarray(scores[name],dtype=np.float64);saved=np.asarray([x['scores'] for x in stored],dtype=np.float64)
  if not np.array_equal(p,saved):raise ValueError(name+' vector scores differ')
  gold=np.array([[i['gold']=='Y' for i in r['items']] for r in rows],dtype=bool);model=np.array([[y=='Y' for y in x['qwen']] for x in q],dtype=bool);lex=np.array([[y=='Y' for y in x['lexical']] for x in q],dtype=bool)
  old=(p>=f1t);hard=(p>=t)
  if any(s['embedding']!=['Y' if y else 'N' for y in row] for s,row in zip(stored,old)):raise ValueError(name+' prior labels mismatch')
  allm={'f1_threshold':old,'ratio_threshold':hard,'qwen':model,'lexical':lex};m={k:metrics(gold,v) for k,v in allm.items()}
  for key in ('f1_threshold','qwen','lexical'):
   earlier=prior['samples'][name]['quality']['embedding' if key=='f1_threshold' else key]
   for metric in ('f1','balanced_accuracy','ratio_mae'):
    if abs(m[key][metric]-earlier[metric])>1e-12:raise ValueError(name+' previous metric mismatch '+key)
  target=gold.mean(axis=1);errors={k:np.abs(v.mean(axis=1)-target) for k,v in allm.items()};draw=np.random.default_rng(seed).integers(0,n,size=(4000,n))
  query={}
  for k,left,right in [('ratio_minus_qwen','ratio_threshold','qwen'),('ratio_minus_lexical','ratio_threshold','lexical'),('f1_minus_qwen','f1_threshold','qwen')]:
   delta=errors[left]-errors[right];query[k]={'difference':float(delta.mean()),'interval_95':[float(x) for x in np.quantile(delta[draw].mean(axis=1),[.025,.975])]}
  oldinterval=prior['samples'][name]['embedding_minus_qwen']['ratio_mae_difference_ci']
  if not np.allclose(query['f1_minus_qwen']['interval_95'],oldinterval,rtol=0,atol=1e-12):raise ValueError(name+' old query interval mismatch')
  ids=sorted({int(i['product_id']) for r in rows for i in r['items']});lookup={v:i for i,v in enumerate(ids)};pi=np.array([lookup[int(i['product_id'])] for r in rows for i in r['items']]);qi=np.repeat(np.arange(n),32);g=gold.ravel().astype(float);vals={k:v.ravel().astype(float) for k,v in allm.items()};rng=np.random.default_rng(pseed);samples={'ratio_minus_qwen':[],'ratio_minus_lexical':[],'lexical_minus_qwen':[]}
  for _ in range(4000):
   qc=np.bincount(rng.integers(0,n,n),minlength=n);pc=np.bincount(rng.integers(0,len(ids),len(ids)),minlength=len(ids));weight=pc[pi];den=np.bincount(qi,weights=weight,minlength=n)
   if np.any((qc>0)&(den==0)):raise RuntimeError('Selected query without product weight')
   safe=np.where(den==0,1,den);ref=np.bincount(qi,weights=weight*g,minlength=n)/safe
   er={k:np.abs(np.bincount(qi,weights=weight*v,minlength=n)/safe-ref) for k,v in vals.items()}
   for k,left,right in [('ratio_minus_qwen','ratio_threshold','qwen'),('ratio_minus_lexical','ratio_threshold','lexical'),('lexical_minus_qwen','lexical','qwen')]:samples[k].append(float(np.dot(qc,er[left]-er[right])/qc.sum()))
  prod={k:[float(x) for x in np.quantile(v,[.025,.975])] for k,v in samples.items()}
  if not np.allclose(prod['lexical_minus_qwen'],cross['samples'][name]['ranges']['lexical_minus_qwen'],rtol=0,atol=1e-12):raise ValueError(name+' prior product sensitivity mismatch')
  result['samples'][name]={'pools':n,'distinct_products':len(ids),'repeated_product_ids':int(sum(x>1 for x in np.bincount(pi))),'metrics':m,'query_seed':seed,'query_bootstrap':query,'product_seed':pseed,'product_reweighted_ranges':prod}
 for p in sources:result['input_sha256'][str(p.relative_to(root))]=sha(p)
 if a.out.exists() or a.out.is_symlink():raise FileExistsError(a.out)
 a.out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
 print('Ratio threshold',t,'training selected MAE',trainmae,'F1 threshold',f1t)
 for name,*_ in SETS:
  x=result['samples'][name];m=x['metrics'];print(name,'ratio threshold',round(m['ratio_threshold']['ratio_mae'],4),'F1 threshold',round(m['f1_threshold']['ratio_mae'],4),'Qwen',round(m['qwen']['ratio_mae'],4),'lexical',round(m['lexical']['ratio_mae'],4),'query delta',x['query_bootstrap']['ratio_minus_qwen']['interval_95'],'product range',x['product_reweighted_ranges']['ratio_minus_qwen'])
if __name__=='__main__':main()
