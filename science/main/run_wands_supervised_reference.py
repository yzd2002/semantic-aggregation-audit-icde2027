#!/usr/bin/env python3
"""Post-hoc WANDS supervised reference; see fixed protocol."""
import argparse, collections, hashlib, json, re, time, warnings
from pathlib import Path
import numpy as np
import scipy, sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction import FeatureHasher
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits
from run_supervised_baseline import features
from analyze_order_fallback_validation import keyed_vectors, paired
from run_compact_prompt_ablation import metrics
from run_order_fallback_validation import parse_one
from order_fallback import combine

HERE=Path(__file__).resolve().parent
PROTOCOL='PROTOCOL_wands_supervised_reference.md'
SETS=[('holdout160','wands_holdout_selection','run_keyed_order_qwen_v1',20271211),
      ('fallback96','wands_fallback_validation_selection','run_wands_fallback_validation_qwen_v1',20271212),
      ('selective52','wands_selective_execution_selection','run_wands_selective_execution_qwen_v1',20271213)]

def lines(path):
    with path.open(encoding='utf8') as f:return [json.loads(line) for line in f]
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def ids(rows):return {str(i['product_id']) for row in rows for i in row['items']}
def queries(rows):return {r['query'].strip().casefold() for r in rows}
def qwen_predictions(name,rows,run):
    if name=='holdout160':
        r=lines(run/'original_outputs.jsonl')
        if len(r)!=len(rows) or any(a['query_id']!=b['query_id'] for a,b in zip(rows,r)):raise ValueError('holdout query alignment')
        return [x['pred'] for x in r]
    if name=='selective52':
        r=lines(run/'selective_outputs.jsonl')
        if len(r)!=len(rows) or any(a['query_id']!=b['query_id'] for a,b in zip(rows,r)):raise ValueError('selective query alignment')
        return [x['pred'] for x in r]
    if name=='fallback96':
        original=keyed_vectors(rows,lines(run/'original_responses.jsonl'))
        reverse=keyed_vectors(rows,lines(run/'stable_reverse_responses.jsonl'))
        point=lines(run/'pointwise_responses.jsonl')
        if len(point)!=len(rows)*32:raise ValueError('pointwise response count')
        vectors=[]
        for q,row in enumerate(rows):
            block=point[q*32:(q+1)*32]
            if any(x['query_id']!=row['query_id'] or x['start']!=i for i,x in enumerate(block)):raise ValueError('pointwise candidate alignment')
            vectors.append([parse_one(x['final']) for x in block])
        return [combine(a,b,c) for a,b,c in zip(original,reverse,vectors)]
    raise ValueError(name)
def lexical(row):
    q=set(re.findall(r'\w+',row['query'].casefold()))
    return ['Y' if q and q.issubset(set(re.findall(r'\w+',it['text'].casefold()))) else 'N' for it in row['items']]
def check(rows,n):
    if len(rows)!=n or len(queries(rows))!=n or len({r['query_id'] for r in rows})!=n:raise ValueError('pool count/query uniqueness')
    if any(len(r['items'])!=32 or any(i['gold'] not in ('Y','N') or (i['gold']=='Y')!=(i['grade']=='Exact') for i in r['items']) for r in rows):raise ValueError('candidate count/label mapping')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    train=lines(HERE/'run_wands_160/inputs.jsonl');check(train,160)
    samples={name:lines(HERE/selection/'inputs.jsonl') for name,selection,run,seed in SETS}
    counts={'holdout160':160,'fallback96':96,'selective52':52}
    for name,rows in samples.items():
        check(rows,counts[name])
        if queries(train)&queries(rows) or ids(train)&ids(rows):raise ValueError(name+' train/test leakage')
    for i,(name,_,_,_) in enumerate(SETS):
        for other,_,_,_ in SETS[i+1:]:
            if queries(samples[name])&queries(samples[other]) or ids(samples[name])&ids(samples[other]):raise ValueError(name+'/'+other+' evaluation overlap')
    a.out.mkdir(parents=True,exist_ok=False)
    t=time.monotonic();train_features=[features(r['query'],it['text']) for r in train for it in r['items']]
    gold=np.asarray([int(it['gold']=='Y') for r in train for it in r['items']])
    h=FeatureHasher(n_features=2**18,input_type='dict',alternate_sign=False,dtype=np.float32)
    x=h.transform(train_features)
    model=LogisticRegression(C=1,solver='lbfgs',max_iter=1000,tol=1e-4,penalty='l2',class_weight=None,random_state=20271211)
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter('always',ConvergenceWarning)
        with threadpool_limits(limits=2):model.fit(x,gold)
    warning=any(issubclass(w.category,ConvergenceWarning) for w in captured)
    result={'protocol':PROTOCOL,'training':{'pools':len(train),'candidates':len(gold),'positive':int(gold.sum()),
      'distinct_products':len(ids(train)),'iterations':int(model.n_iter_[0]),'max_iter':1000,'convergence_warning':warning,
      'selection_sha256':sha(HERE/'run_wands_160/inputs.jsonl')},
      'runtime':{'numpy':np.__version__,'scipy':scipy.__version__,'scikit_learn':sklearn.__version__},'samples':{}}
    all_records=[]
    for name,selection,run,seed in SETS:
        rows=samples[name];testx=h.transform(features(r['query'],it['text']) for r in rows for it in r['items'])
        with threadpool_limits(limits=2):p=model.predict_proba(testx)[:,1]
        sup=[['Y' if x>=.5 else 'N' for x in p[i*32:(i+1)*32]] for i in range(len(rows))]
        qwen=qwen_predictions(name,rows,HERE/run);base=[lexical(row) for row in rows]
        common=[i for i in range(len(rows)) if qwen[i] is not None]
        if len(common)!=len(rows):raise ValueError('Unexpected incomplete Qwen pool; report before updating protocol')
        if any(len(v)!=32 or set(v)-{'Y','N'} for v in qwen):raise ValueError('Invalid Qwen vector')
        pred={'qwen':qwen,'supervised':sup,'lexical':base}
        rng=np.random.default_rng(seed);draw=rng.integers(0,len(common),size=(4000,len(common)))
        quality={k:metrics(rows,v,common) for k,v in pred.items()}
        prev=json.loads((HERE/run/'paired_analysis.json').read_text())
        expected=prev['quality_common']['keyed_original' if name=='holdout160' else 'pointwise_fallback' if name=='fallback96' else 'actual_fallback']
        for m in ('f1','balanced_accuracy','ratio_mae'):
            if not np.isclose(quality['qwen'][m],expected[m],rtol=0,atol=1e-12):raise ValueError(name+' Qwen baseline mismatch '+m)
        result['samples'][name]={'selection_sha256':sha(HERE/selection/'inputs.jsonl'),'pools':len(rows),'common_pools':len(common),
          'training_query_overlap':len(queries(train)&queries(rows)),'training_product_overlap':len(ids(train)&ids(rows)),
          'query_bootstrap_seed':seed,'resamples':4000,'quality':quality,
          'supervised_minus_qwen':paired(rows,pred,common,'qwen','supervised',draw)}
        for i,row in enumerate(rows):
            all_records.append({'sample':name,'query_id':row['query_id'],'supervised':sup[i],
              'positive_probability':[float(v) for v in p[i*32:(i+1)*32]],'qwen':qwen[i],'lexical':base[i]})
    with (a.out/'predictions.jsonl').open('w',encoding='utf8') as f:
        for x in all_records:f.write(json.dumps(x,ensure_ascii=False)+'\n')
    result['elapsed_seconds_including_features_and_fit']=time.monotonic()-t
    (a.out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()
