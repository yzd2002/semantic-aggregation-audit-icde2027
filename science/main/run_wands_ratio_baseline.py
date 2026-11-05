#!/usr/bin/env python3
"""Re-evaluate WANDS lexical references using ratio-oriented decision rules."""
import argparse,hashlib,json,math,sys
from pathlib import Path
import numpy as np

SETS=[('holdout160','wands_holdout_selection','run_keyed_order_qwen_v1',20261031,160),
      ('fallback96','wands_fallback_validation_selection','run_wands_fallback_validation_qwen_v1',20261032,96),
      ('selective52','wands_selective_execution_selection','run_wands_selective_execution_qwen_v1',20261033,52)]
def load(path):
 with path.open(encoding='utf8') as f:return [json.loads(x) for x in f if x.strip()]
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def binary_metrics(gold,pred):
 g=np.asarray(gold,dtype=bool);p=np.asarray(pred,dtype=bool)
 if g.shape!=p.shape:raise ValueError('Prediction shape')
 tp=int((g&p).sum());tn=int((~g&~p).sum());fp=int((~g&p).sum());fn=int((g&~p).sum())
 return {'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.0,
         'balanced_accuracy':(tp/(tp+fn)+tn/(tn+fp))/2 if tp+fn and tn+fp else None,
         'ratio_mae':float(np.abs(p.mean(axis=1)-g.mean(axis=1)).mean()),
         'tp':tp,'fp':fp,'tn':tn,'fn':fn}
def select_threshold(scores,gold):
 if scores.ndim!=2 or scores.shape!=gold.shape or not np.isfinite(scores).all() or (scores<0).any() or (scores>=1).any():raise ValueError('Unexpected training scores')
 n,k=scores.shape;counts=np.zeros(n,dtype=np.int64);g=gold.sum(axis=1).astype(np.int64)
 order=np.argsort(-scores.ravel(),kind='stable');flat=scores.ravel();best_loss=int(g.sum());threshold=float(np.nextafter(flat[order[0]],np.inf));i=0
 while i<len(order):
  j=i+1;v=flat[order[i]]
  while j<len(order) and flat[order[j]]==v:j+=1
  for position in order[i:j]:counts[int(position)//k]+=1
  loss=int(np.abs(counts-g).sum())
  if loss<best_loss:best_loss=loss;threshold=float(v)
  i=j
 # Independent direct sweep, with the same documented all-negative and highest-threshold tie rule.
 values=np.concatenate(([np.nextafter(flat[order[0]],np.inf)],np.unique(flat)[::-1]))
 direct=np.array([np.abs((scores>=t).sum(axis=1)-g).sum() for t in values],dtype=np.int64)
 best_index=int(np.argmin(direct))
 if int(direct[best_index])!=best_loss or float(values[best_index])!=threshold:raise AssertionError('Threshold sweep disagrees')
 return threshold,best_loss/(n*k),len(values)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--artifact',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
 root=a.artifact.resolve();sys.path.insert(0,str(root))
 from run_wands_supervised_reference import qwen_predictions
 trainpath=root/'run_wands_160/inputs.jsonl';oofpath=root/'run_wands_calibrated_lexical/out_of_fold.jsonl'
 predpath=root/'run_wands_supervised_reference/predictions.jsonl';oldpath=root/'run_wands_supervised_reference/summary.json'
 cvpath=root/'run_wands_calibrated_lexical/predictions.jsonl'
 sourcepaths=[trainpath,oofpath,predpath,oldpath,cvpath,root/'run_wands_calibrated_lexical/summary.json']
 train=load(trainpath);oof=load(oofpath);preds=load(predpath);cv=load(cvpath);old=json.loads(oldpath.read_text())
 if len(train)!=160 or len(oof)!=160 or len({r['query_id'] for r in train})!=160 or any(r['query_id']!=p['query_id'] or len(r['items'])!=32 or len(p['positive_probability'])!=32 for r,p in zip(train,oof)):raise ValueError('Training alignment')
 def check(rows):
  if any(len(r['items'])!=32 or any(i['gold'] not in ('Y','N') or (i['gold']=='Y')!=(i['grade']=='Exact') for i in r['items']) for r in rows):raise ValueError('WANDS labels')
 check(train)
 gold=np.asarray([[i['gold']=='Y' for i in r['items']] for r in train],dtype=bool)
 scores=np.asarray([r['positive_probability'] for r in oof],dtype=float)
 threshold,train_score,nthresholds=select_threshold(scores,gold)
 median=float(sorted(gold.mean(axis=1))[79])
 result={'scope':'post-hoc fixed-output descriptive analysis, not a new run or blind test',
         'threshold_selection':{'threshold':threshold,'oof_selected_ratio_mae':train_score,'candidate_thresholds':nthresholds,
           'training_pools':len(train),'training_labels':int(gold.size),'tie_rule':'largest threshold',
           'constant_lower_median_train_ratio':median},'samples':{},'input_sha256':{}}
 trainq={r['query'].strip().casefold() for r in train};trainp={str(i['product_id']) for r in train for i in r['items']};traint={i['text'] for r in train for i in r['items']}
 for name,selection,run,seed,n in SETS:
  sel=root/selection/'inputs.jsonl';sourcepaths.append(sel);rows=load(sel);check(rows)
  s=[x for x in preds if x['sample']==name];v=[x for x in cv if x['sample']==name]
  if len(rows)!=n or len(s)!=n or len(v)!=n or len({r['query_id'] for r in rows})!=n:raise ValueError(name+' size')
  if any(r['query_id']!=p['query_id'] or r['query_id']!=c['query_id'] or p['positive_probability']!=c['positive_probability'] for r,p,c in zip(rows,s,v)):raise ValueError(name+' alignment or saved scores')
  if trainq&{r['query'].strip().casefold() for r in rows} or trainp&{str(i['product_id']) for r in rows for i in r['items']} or traint&{i['text'] for r in rows for i in r['items']}:raise ValueError(name+' train/test overlap')
  p=np.asarray([r['positive_probability'] for r in s],dtype=float)
  if p.shape!=(n,32) or not np.isfinite(p).all() or (p<0).any() or (p>1).any():raise ValueError(name+' invalid scores')
  g=np.asarray([[i['gold']=='Y' for i in r['items']] for r in rows],dtype=bool)
  q=np.asarray([[x=='Y' for x in r['qwen']] for r in s],dtype=bool);lex=np.asarray([[x=='Y' for x in r['lexical']] for r in s],dtype=bool)
  raw=qwen_predictions(name,rows,root/run)
  if any(r['qwen']!=x or len(r['qwen'])!=32 or set(r['qwen'])-{'Y','N'} for r,x in zip(s,raw)):raise ValueError(name+' Qwen reconstruction')
  fixed=(p>=.5);hard=(p>=threshold)
  for r,row in zip(s,fixed):
   if r['supervised']!=['Y' if x else 'N' for x in row]:raise ValueError('Saved 0.5 labels mismatch')
  methods={'qwen':binary_metrics(g,q),'fixed_0_5':binary_metrics(g,fixed),'ratio_threshold':binary_metrics(g,hard),'lexical':binary_metrics(g,lex)}
  for method,key in [('qwen','qwen'),('fixed_0_5','supervised'),('lexical','lexical')]:
   for metric in ('f1','balanced_accuracy','ratio_mae'):
    if abs(methods[method][metric]-old['samples'][name]['quality'][key][metric])>1e-12:raise ValueError(name+' old metric mismatch '+method+' '+metric)
  target=g.mean(axis=1);reference=q.mean(axis=1)
  ratio_preds={'ratio_threshold':hard.mean(axis=1),'score_mean':p.mean(axis=1),'training_median':np.full(n,median)}
  draw=np.random.default_rng(seed).integers(0,n,size=(4000,n));unc={}
  for method,ratio in ratio_preds.items():
   errors=np.abs(ratio-target);diff=errors-np.abs(reference-target);dist=diff[draw].mean(axis=1)
   unc[method]={'ratio_mae':float(errors.mean()),'difference_vs_qwen':float(diff.mean()),
                'query_bootstrap_95':[float(x) for x in np.quantile(dist,[.025,.975])]}
  result['samples'][name]={'pools':n,'seed':seed,'binary_metrics':methods,'ratio_results':unc}
 for p in sourcepaths:result['input_sha256'][str(p.relative_to(root))]=sha(p)
 if a.out.exists() or a.out.is_symlink():raise FileExistsError(a.out)
 a.out.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
 print('Training-only ratio threshold:',threshold,'selected OOF MAE:',train_score,'median fraction:',median)
 for name in [x[0] for x in SETS]:
  x=result['samples'][name];u=x['ratio_results']
  print(name,'Qwen',round(x['binary_metrics']['qwen']['ratio_mae'],4),'ratio-threshold',round(u['ratio_threshold']['ratio_mae'],4),'mean-score',round(u['score_mean']['ratio_mae'],4),'train-median',round(u['training_median']['ratio_mae'],4))
if __name__=='__main__':main()
