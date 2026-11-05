#!/usr/bin/env python3
"""Independently refit and audit cross-fold threshold and exposed test results."""
import argparse,hashlib,json,random,warnings
from fractions import Fraction
from pathlib import Path
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction import FeatureHasher
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits
from verify_wands_supervised_reference import features,tokenize,qwen,query_counts,quality
HERE=Path(__file__).resolve().parent
SETS=[('holdout160','wands_holdout_selection','run_keyed_order_qwen_v1',20271241),
 ('fallback96','wands_fallback_validation_selection','run_wands_fallback_validation_qwen_v1',20271242),
 ('selective52','wands_selective_execution_selection','run_wands_selective_execution_qwen_v1',20271243)]
def load(p):
 with p.open(encoding='utf8') as f:return [json.loads(s) for s in f]
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def fit(x,y,indices):
 m=LogisticRegression(C=1,solver='lbfgs',max_iter=1000,tol=1e-4,penalty='l2',class_weight=None,random_state=20271211)
 with warnings.catch_warnings(record=True) as caught:
  warnings.simplefilter('always',ConvergenceWarning)
  with threadpool_limits(limits=2):m.fit(x[indices],y[indices])
 assert not any(issubclass(w.category,ConvergenceWarning) for w in caught)
 return m
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);args=ap.parse_args()
 summary=json.loads((args.run/'summary.json').read_text());training_path=HERE/'run_wands_160/inputs.jsonl'
 oldpath=HERE/'run_wands_supervised_reference/predictions.jsonl';train=load(training_path);old=load(oldpath)
 assert summary['training_input_sha256']==digest(training_path) and summary['existing_probabilities_sha256']==digest(oldpath)
 assert len(train)==160 and all(len(r['items'])==32 for r in train)
 assert len({r['query'].strip().casefold() for r in train})==160 and len({r['query_id'] for r in train})==160
 saved_oof=load(args.run/'out_of_fold.jsonl');saved=load(args.run/'predictions.jsonl');assert len(saved_oof)==160
 h=FeatureHasher(n_features=2**18,input_type='dict',alternate_sign=False,dtype=np.float32)
 x=h.transform(features(r['query'],i['text']) for r in train for i in r['items'])
 y=np.array([int(i['gold']=='Y') for r in train for i in r['items']]);assert len(y)==5120
 order=list(range(160));random.Random(20271229).shuffle(order);oof=np.full(5120,np.nan)
 for f in range(5):
  valq=set(order[32*f:32*(f+1)]);valid=[train[i] for i in sorted(valq)]
  valids={str(i['product_id']) for r in valid for i in r['items']};valtext={i['text'] for r in valid for i in r['items']}
  validation=[i*32+j for i in sorted(valq) for j in range(32)]
  available=[i*32+j for i in range(160) if i not in valq for j in range(32)]
  keep=[k for k in available if str(train[k//32]['items'][k%32]['product_id']) not in valids and train[k//32]['items'][k%32]['text'] not in valtext]
  assert not valids&{str(train[k//32]['items'][k%32]['product_id']) for k in keep}
  assert not valtext&{train[k//32]['items'][k%32]['text'] for k in keep}
  meta=summary['folds'][f]
  assert meta['fold']==f and meta['validation_query_indices']==sorted(valq)
  assert meta['training_candidates_before_product_text_exclusion']==len(available) and meta['training_candidates']==len(keep)
  assert meta['excluded_training_candidates']==len(available)-len(keep) and meta['training_positive']==int(y[keep].sum())
  assert meta['validation_candidates']==len(validation) and meta['validation_positive']==int(y[validation].sum()) and not meta['convergence_warning']
  m=fit(x,y,keep);assert int(m.n_iter_[0])==meta['iterations']
  with threadpool_limits(limits=2):oof[validation]=m.predict_proba(x[validation])[:,1]
  for i in valq:
   assert saved_oof[i]['query_id']==train[i]['query_id'] and saved_oof[i]['fold']==f
   assert np.allclose(saved_oof[i]['positive_probability'],oof[i*32:(i+1)*32],rtol=0,atol=1e-9)
 stored=np.array([p for r in saved_oof for p in r['positive_probability']]);assert np.isfinite(stored).all()
 best=Fraction(-1,1);best_threshold=None;best_tp=best_count=None;positive=int(y.sum())
 for threshold in np.unique(stored)[::-1]:
  mask=stored>=threshold;tp=int(y[mask].sum());predicted=int(mask.sum());score=Fraction(2*tp,positive+predicted)
  if score>best:best,best_threshold,best_tp,best_count=score,float(threshold),tp,predicted
 c=summary['calibration'];assert c['threshold']==best_threshold and c['tp']==best_tp and c['predicted_positive']==best_count and c['gold_positive']==positive
 assert c['numerator']==2*best_tp and c['denominator']==positive+best_count and c['f1']==float(best)
 full=fit(x,y,list(range(len(y))))
 trainq={r['query'].strip().casefold() for r in train};trainid={str(i['product_id']) for r in train for i in r['items']};traintext={i['text'] for r in train for i in r['items']}
 for name,sel,run,seed in SETS:
  rows=load(HERE/sel/'inputs.jsonl');n=len(rows);previous=[r for r in old if r['sample']==name];current=[r for r in saved if r['sample']==name]
  assert len(previous)==len(current)==n and all(r['query_id']==p['query_id']==c['query_id'] for r,p,c in zip(rows,previous,current))
  assert not trainq&{r['query'].strip().casefold() for r in rows} and not trainid&{str(i['product_id']) for r in rows for i in r['items']}
  assert not traintext&{i['text'] for r in rows for i in r['items']}
  tx=h.transform(features(r['query'],i['text']) for r in rows for i in r['items'])
  with threadpool_limits(limits=2):prob=full.predict_proba(tx)[:,1]
  priorprob=np.array([p for r in previous for p in r['positive_probability']]);assert np.allclose(prob,priorprob,rtol=0,atol=1e-9)
  cal=[['Y' if p>=best_threshold else 'N' for p in r['positive_probability']] for r in previous]
  fixed=[['Y' if p>=.5 else 'N' for p in r['positive_probability']] for r in previous]
  q=qwen(name,rows,HERE/run)
  lex=[['Y' if tokenize(r['query']) and tokenize(r['query'])<=tokenize(i['text']) else 'N' for i in r['items']] for r in rows]
  for i in range(n):
   assert current[i]['calibrated']==cal[i] and current[i]['positive_probability']==previous[i]['positive_probability']
   assert (previous[i]['supervised'],previous[i]['qwen'],previous[i]['lexical'])==(fixed[i],q[i],lex[i])
  vectors={'calibrated':cal,'fixed_0_5':fixed,'qwen':q,'lexical':lex}
  matrices={k:query_counts(rows,v) for k,v in vectors.items()};s=summary['samples'][name]
  assert s['pools']==n and s['selection_sha256']==digest(HERE/sel/'inputs.jsonl') and s['query_bootstrap_seed']==seed and s['resamples']==4000
  for k,z in matrices.items():
   for field,value in quality(z).items():assert abs(value-s['quality'][k][field])<1e-12,(name,k,field)
  draw=np.random.default_rng(seed).integers(0,n,size=(4000,n))
  def f1(a):
   b=a[draw,:4].sum(axis=1);den=2*b[:,0]+b[:,1]+b[:,3]
   return np.divide(2*b[:,0],den,out=np.zeros(len(draw)),where=den!=0)
  difference_f1=f1(matrices['calibrated'])-f1(matrices['qwen'])
  difference_mae=matrices['calibrated'][draw,4].mean(axis=1)-matrices['qwen'][draw,4].mean(axis=1)
  comp=s['calibrated_minus_qwen']
  assert abs(comp['f1_difference']-(quality(matrices['calibrated'])['f1']-quality(matrices['qwen'])['f1']))<1e-12
  assert abs(comp['ratio_mae_difference']-(quality(matrices['calibrated'])['ratio_mae']-quality(matrices['qwen'])['ratio_mae']))<1e-12
  assert np.allclose(np.quantile(difference_f1,[.025,.975]),comp['f1_difference_ci'],rtol=0,atol=1e-12)
  assert np.allclose(np.quantile(difference_mae,[.025,.975]),comp['ratio_mae_difference_ci'],rtol=0,atol=1e-12)
  print(name,'query, product and text isolation; predictions, metrics and intervals verified')
 assert sum(1 for _ in saved)==sum(len(load(HERE/s/'inputs.jsonl')) for _,s,_,_ in SETS)
 print('Independently refit all five product-isolated folds and full model; threshold, ties, raw Qwen outputs and paired results verified.')
if __name__=='__main__':main()
