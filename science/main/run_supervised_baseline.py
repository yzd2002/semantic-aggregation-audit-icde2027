#!/usr/bin/env python3
"""Train the frozen post-hoc lexical baseline; see PROTOCOL_supervised_baseline.md."""
import argparse, collections, json, math, re, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import scipy, sklearn
from sklearn.feature_extraction import FeatureHasher
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
SEED = 20271106
NTRAIN = 50000
FIELDS = ('product_title','product_description','product_bullet_point','product_brand','product_color')
METHODS = ('pointwise','batch_json','two_stage_json','two_stage_compact_fallback')

def read_jsonl(p):
    with p.open(encoding='utf8') as f:
        return [json.loads(line) for line in f]

def tokens(s):
    return set(re.findall(r'\w+',s.casefold()))

def features(query, text):
    q,p=tokens(query),tokens(text);shared=q&p;union=q|p
    d={}
    for prefix,words in (('q:',q),('p:',p),('shared:',shared),('missing:',q-p)):
        for w in words:d[prefix+w]=1.0
    d.update({'numeric:query_coverage':len(shared)/len(q) if q else 0.0,
              'numeric:product_coverage':len(shared)/len(p) if p else 0.0,
              'numeric:jaccard':len(shared)/len(union) if union else 0.0,
              'numeric:log_query_count':math.log1p(len(q)),
              'numeric:log_product_count':math.log1p(len(p)),
              'numeric:constant':1.0})
    return d

def train_data(examples,products,prior):
    ex=pd.read_parquet(examples,columns=['example_id','query','query_id','product_id','product_locale','esci_label','small_version','split'],
         filters=[('product_locale','=','us'),('small_version','=',1),('split','=','train')])
    ex=ex[ex.esci_label.isin(['E','S','C','I']) & ex['query'].notna()].copy()
    normalized=ex['query'].str.strip().str.casefold()
    ex=ex[normalized.ne('') & ~normalized.isin(prior)].sort_values('example_id')
    eligible=len(ex)
    if eligible<NTRAIN:raise RuntimeError(f'Only {eligible} eligible training examples')
    ex=ex.sample(n=NTRAIN,random_state=SEED).copy()
    ids=sorted(ex.product_id.dropna().unique().tolist())
    dataset=ds.dataset(products,format='parquet')
    prod=dataset.to_table(columns=['product_id','product_locale',*FIELDS],
        filter=(ds.field('product_locale')=='us') & ds.field('product_id').isin(ids)).to_pandas()
    if prod.duplicated(['product_id','product_locale']).any():raise RuntimeError('Duplicate product key')
    joined=ex.merge(prod,on=['product_id','product_locale'],how='left',indicator=True,validate='many_to_one')
    missing=int((joined['_merge']!='both').sum())
    if missing:raise RuntimeError(f'{missing} missing product joins')
    return joined,eligible

def metrics(golds,preds):
    rows=[]
    for g,p in zip(golds,preds):
        assert len(g)==len(p) and len(g)>0
        g=np.asarray(g,dtype=int);p=np.asarray(p,dtype=int)
        tp=int(((g==1)&(p==1)).sum());fp=int(((g==0)&(p==1)).sum())
        tn=int(((g==0)&(p==0)).sum());fn=int(((g==1)&(p==0)).sum())
        rows.append([tp,fp,tn,fn,abs(float(g.mean()-p.mean()))])
    return np.asarray(rows,dtype=float)

def report_subset(inds, golds, preds, qwen):
    records={name:metrics([golds[i] for i in inds],[labels[i] for i in inds])
             for name,labels in [('supervised_lexical',preds),('qwen_pointwise',qwen)]}
    rng=np.random.default_rng(20271108);draw=rng.integers(0,len(inds),size=(4000,len(inds)))
    result={'pools':len(inds),'resamples':4000,'seed':20271108,'methods':{}}
    draws={}
    def calc(a):
        tp,fp,tn,fn=a[:,:4].sum(axis=0)
        return {'f1':float(2*tp/(2*tp+fp+fn)),
                'balanced_accuracy':float((tp/(tp+fn)+tn/(tn+fp))/2),
                'ratio_mae':float(a[:,4].mean())}
    for name,a in records.items():
        c=a[draw,:4].sum(axis=1);tp,fp,tn,fn=c.T
        den=2*tp+fp+fn
        f1=np.divide(2*tp,den,out=np.zeros_like(den),where=den>0)
        sens=np.divide(tp,tp+fn,out=np.zeros_like(tp),where=(tp+fn)>0)
        spec=np.divide(tn,tn+fp,out=np.zeros_like(tn),where=(tn+fp)>0)
        ds_={'f1':f1,'balanced_accuracy':(sens+spec)/2,'ratio_mae':a[draw,4].mean(axis=1)}
        draws[name]=ds_;result['methods'][name]=calc(a)
        for k,v in ds_.items():result['methods'][name][k+'_ci']=[float(x) for x in np.quantile(v,[.025,.975])]
    result['supervised_minus_pointwise_ci']={k:[float(x) for x in np.quantile(draws['supervised_lexical'][k]-draws['qwen_pointwise'][k],[.025,.975])]
                                               for k in ('f1','ratio_mae')}
    return result

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--examples',type=Path,default=Path('/home/gpu/intent2s_esci_official_2026/shopping_queries_dataset_examples.parquet'))
    ap.add_argument('--products',type=Path,default=Path('/home/gpu/intent2s_esci_official_2026/shopping_queries_dataset_products.parquet'))
    ap.add_argument('--out',type=Path,default=Path('run_supervised_baseline'))
    ap.add_argument('--max-iter',type=int,default=200)
    a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    pools=read_jsonl(HERE/'run_full_pools_250/inputs.jsonl')
    outputs=read_jsonl(HERE/'run_full_pools_250/outputs.jsonl')
    prior=set()
    for folder in ('run_test_160','run_validation_500','run_full_pools_250'):
        prior.update(r['query'].strip().casefold() for r in read_jsonl(HERE/folder/'inputs.jsonl'))
    start=time.monotonic();data,eligible=train_data(a.examples,a.products,prior)
    assert not set(data['query'].str.strip().str.casefold()) & prior
    with (a.out/'train_manifest.jsonl').open('w',encoding='utf8') as f:
        for x in data.itertuples(index=False):
            f.write(json.dumps({'example_id':str(x.example_id),'query_id':str(x.query_id),'query':x.query,
                                'product_id':str(x.product_id),'grade':x.esci_label},ensure_ascii=False)+'\n')
    docs=[(x.query,' '.join(getattr(x,k) for k in FIELDS if isinstance(getattr(x,k),str) and getattr(x,k))[:600])
          for x in data.itertuples(index=False)]
    y=np.asarray([1 if grade in ('E','S') else 0 for grade in data.esci_label],dtype=int)
    print('Selected',len(y),'training examples;',data['query'].nunique(),'unique query strings;',len(set(y)),'classes.',flush=True)
    hasher=FeatureHasher(n_features=2**18,input_type='dict',alternate_sign=False,dtype=np.float32)
    xtrain=hasher.transform(features(q,t) for q,t in docs)
    print('Training matrix',xtrain.shape,'nonzero',xtrain.nnz,flush=True)
    classifier=LogisticRegression(C=1,solver='lbfgs',max_iter=a.max_iter,penalty='l2',class_weight=None,random_state=SEED)
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter('always',ConvergenceWarning)
        with threadpool_limits(limits=2):classifier.fit(xtrain,y)
    convergence_warning=any(issubclass(w.category,ConvergenceWarning) for w in captured)
    print('Convergence warning:',convergence_warning,flush=True)
    print('Solver iterations',classifier.n_iter_.tolist(),flush=True)
    testdocs=[(pool['query'],item['text']) for pool in pools for item in pool['items']]
    xtest=hasher.transform(features(q,t) for q,t in testdocs)
    with threadpool_limits(limits=2):probs=classifier.predict_proba(xtest)[:,1]
    labels=(probs>=0.5).astype(int);golds=[];preds=[];qwen=[];k=0
    with (a.out/'predictions.jsonl').open('w',encoding='utf8') as f:
        for pool,out in zip(pools,outputs):
            assert pool['query_id']==out['query_id']
            n=len(pool['items']);pp=probs[k:k+n];pred=labels[k:k+n];k+=n
            golds.append([int(it['gold']=='Y') for it in pool['items']]);preds.append(pred.tolist())
            qp=out['plans']['pointwise']['pred'];assert qp is not None;qwen.append([int(z=='Y') for z in qp])
            f.write(json.dumps({'query_id':pool['query_id'],'pred':['Y' if z else 'N' for z in pred],
                                'positive_probability':[float(v) for v in pp]})+'\n')
    assert k==len(labels)
    complete=[i for i,o in enumerate(outputs) if all(o['plans'][m]['pred'] is not None for m in METHODS)]
    allp=list(range(len(pools)));assert len(complete)==207 and len(allp)==250
    result={'protocol':'PROTOCOL_supervised_baseline.md','training_examples':len(y),'eligible_training_examples':eligible,
      'training_unique_queries':int(data['query'].nunique()),'training_grades':dict(collections.Counter(data.esci_label)),
      'excluded_normalized_query_strings':len(prior),'missing_join_count':0,'n_features':2**18,
      'solver_iterations':int(classifier.n_iter_[0]),'max_iter':a.max_iter,'convergence_warning':convergence_warning,
      'elapsed_seconds_including_data_loading_training_and_inference':time.monotonic()-start,
      'runtime':{'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__,'scikit_learn':sklearn.__version__},
      'all_250':report_subset(allp,golds,preds,qwen),'qwen_common_207':report_subset(complete,golds,preds,qwen)}
    (a.out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()
