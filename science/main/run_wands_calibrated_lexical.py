#!/usr/bin/env python3
"""Five-fold, product-disjoint threshold selection for the prior lexical reference."""
import argparse,hashlib,json,random,time,warnings
from pathlib import Path
import numpy as np
import scipy,sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction import FeatureHasher
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits
from run_supervised_baseline import features
from run_compact_prompt_ablation import metrics
from analyze_order_fallback_validation import paired
HERE=Path(__file__).resolve().parent
SETS=[('holdout160','wands_holdout_selection',20271241),('fallback96','wands_fallback_validation_selection',20271242),('selective52','wands_selective_execution_selection',20271243)]
def lines(path):
 with path.open(encoding='utf8') as f:return [json.loads(s) for s in f]
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def calibrate(scores,gold):
 order=np.argsort(-scores,kind='stable');positive=int(gold.sum());tp=0;best=None;i=0
 while i<len(order):
  j=i;value=float(scores[order[i]])
  while j<len(order) and float(scores[order[j]])==value:
   tp+=int(gold[order[j]]);j+=1
  numerator,denominator=2*tp,positive+j
  if best is None or numerator*best['denominator']>best['numerator']*denominator:
   best={'threshold':value,'tp':tp,'predicted_positive':j,'gold_positive':positive,'numerator':numerator,'denominator':denominator,'f1':numerator/denominator}
  i=j
 return best
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
 train=lines(HERE/'run_wands_160/inputs.jsonl');assert len(train)==160 and all(len(r['items'])==32 for r in train)
 tests={name:lines(HERE/path/'inputs.jsonl') for name,path,seed in SETS}
 prior=lines(HERE/'run_wands_supervised_reference/predictions.jsonl');old=json.loads((HERE/'run_wands_supervised_reference/summary.json').read_text())
 trainq={r['query'].strip().casefold() for r in train};trainid={str(i['product_id']) for r in train for i in r['items']};traintext={i['text'] for r in train for i in r['items']}
 assert len(trainq)==160 and len({r['query_id'] for r in train})==160
 for name,rows in tests.items():
  assert not trainq&{r['query'].strip().casefold() for r in rows}
  assert not trainid&{str(i['product_id']) for r in rows for i in r['items']}
  assert not traintext&{i['text'] for r in rows for i in r['items']}
 a.out.mkdir(parents=True,exist_ok=False);started=time.monotonic()
 h=FeatureHasher(n_features=2**18,input_type='dict',alternate_sign=False,dtype=np.float32)
 x=h.transform(features(r['query'],it['text']) for r in train for it in r['items'])
 gold=np.asarray([int(it['gold']=='Y') for r in train for it in r['items']]);assert len(gold)==5120
 order=list(range(160));random.Random(20271229).shuffle(order);folds=[];oof=np.full(len(gold),np.nan);assign=[None]*160
 for fold in range(5):
  valq=set(order[fold*32:(fold+1)*32]);valid=[r for i,r in enumerate(train) if i in valq]
  valids={str(it['product_id']) for r in valid for it in r['items']};valtext={it['text'] for r in valid for it in r['items']}
  val_idx=[i*32+j for i in range(160) if i in valq for j in range(32)]
  candidates=[i*32+j for i,r in enumerate(train) if i not in valq for j in range(32)]
  tr_idx=[k for k in candidates if str(train[k//32]['items'][k%32]['product_id']) not in valids and train[k//32]['items'][k%32]['text'] not in valtext]
  assert len(val_idx)==1024 and not set(tr_idx)&set(val_idx)
  model=LogisticRegression(C=1,solver='lbfgs',max_iter=1000,tol=1e-4,penalty='l2',class_weight=None,random_state=20271211)
  with warnings.catch_warnings(record=True) as ws:
   warnings.simplefilter('always',ConvergenceWarning)
   with threadpool_limits(limits=2):model.fit(x[tr_idx],gold[tr_idx]);oof[val_idx]=model.predict_proba(x[val_idx])[:,1]
  warning=any(issubclass(w.category,ConvergenceWarning) for w in ws)
  for i in valq:assign[i]=fold
  folds.append({'fold':fold,'validation_query_indices':sorted(valq),'training_candidates_before_product_text_exclusion':len(candidates),
   'training_candidates':len(tr_idx),'excluded_training_candidates':len(candidates)-len(tr_idx),
   'training_positive':int(gold[tr_idx].sum()),'validation_candidates':len(val_idx),'validation_positive':int(gold[val_idx].sum()),
   'iterations':int(model.n_iter_[0]),'convergence_warning':warning})
  if warning:
   (a.out/'failure.json').write_text(json.dumps({'folds':folds,'reason':'convergence warning'},indent=2)+'\n')
   raise RuntimeError('Fold did not converge; result retained without test evaluation.')
 assert np.isfinite(oof).all() and all(x is not None for x in assign)
 choice=calibrate(oof,gold);threshold=choice['threshold'];sample_results={};records=[]
 for name,sel,seed in SETS:
  rows=tests[name];saved=[p for p in prior if p['sample']==name]
  assert len(rows)==len(saved) and all(r['query_id']==s['query_id'] for r,s in zip(rows,saved))
  labels=[[('Y' if p>=threshold else 'N') for p in s['positive_probability']] for s in saved]
  pred={'calibrated':labels,'fixed_0_5':[s['supervised'] for s in saved],'qwen':[s['qwen'] for s in saved],'lexical':[s['lexical'] for s in saved]}
  common=list(range(len(rows)));assert all(len(x)==32 for v in pred.values() for x in v)
  q={k:metrics(rows,v,common) for k,v in pred.items()}
  for metric in ('f1','balanced_accuracy','ratio_mae'):
   assert abs(q['fixed_0_5'][metric]-old['samples'][name]['quality']['supervised'][metric])<1e-12
   assert abs(q['qwen'][metric]-old['samples'][name]['quality']['qwen'][metric])<1e-12
  draw=np.random.default_rng(seed).integers(0,len(rows),size=(4000,len(rows)))
  sample_results[name]={'pools':len(rows),'selection_sha256':sha(HERE/sel/'inputs.jsonl'),'quality':q,
   'query_bootstrap_seed':seed,'resamples':4000,'calibrated_minus_qwen':paired(rows,pred,common,'qwen','calibrated',draw)}
  for row,s,label in zip(rows,saved,labels):records.append({'sample':name,'query_id':row['query_id'],'calibrated':label,'positive_probability':s['positive_probability']})
 summary={'protocol':'PROTOCOL_wands_calibrated_lexical.md','training_input_sha256':sha(HERE/'run_wands_160/inputs.jsonl'),
  'existing_probabilities_sha256':sha(HERE/'run_wands_supervised_reference/predictions.jsonl'),'fold_seed':20271229,
  'calibration':choice,'folds':folds,'samples':sample_results,
  'runtime':{'numpy':np.__version__,'scipy':scipy.__version__,'scikit_learn':sklearn.__version__},'elapsed_seconds':time.monotonic()-started}
 with (a.out/'out_of_fold.jsonl').open('w',encoding='utf8') as f:
  for i,row in enumerate(train):f.write(json.dumps({'query_id':row['query_id'],'fold':assign[i],'positive_probability':oof[i*32:(i+1)*32].tolist()})+'\n')
 with (a.out/'predictions.jsonl').open('w',encoding='utf8') as f:
  for record in records:f.write(json.dumps(record)+'\n')
 (a.out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
 print('Out-of-fold threshold',threshold,'training OOF F1',choice['f1'])
 for name,x in sample_results.items():print(name,'calibrated F1/MAE',x['quality']['calibrated']['f1'],x['quality']['calibrated']['ratio_mae'],'Qwen',x['quality']['qwen']['f1'],x['quality']['qwen']['ratio_mae'])
if __name__=='__main__':main()
