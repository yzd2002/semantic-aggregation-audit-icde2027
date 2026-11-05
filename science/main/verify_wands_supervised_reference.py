#!/usr/bin/env python3
"""Independently re-fit, reparse saved Qwen outputs and recalculate metrics."""
import argparse, json, math, re, warnings
from pathlib import Path
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction import FeatureHasher
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

HERE=Path(__file__).resolve().parent
SETS=[('holdout160','wands_holdout_selection','run_keyed_order_qwen_v1',20271211),
      ('fallback96','wands_fallback_validation_selection','run_wands_fallback_validation_qwen_v1',20271212),
      ('selective52','wands_selective_execution_selection','run_wands_selective_execution_qwen_v1',20271213)]
def load(p):
    with p.open(encoding='utf8') as f:return [json.loads(x) for x in f]
def tokenize(s):return set(re.findall(r'\w+',s.casefold()))
def features(query,text):
    q=tokenize(query);p=tokenize(text);shared=q&p;d={}
    for prefix,terms in [('q:',q),('p:',p),('shared:',shared),('missing:',q-p)]:
        for term in terms:d[prefix+term]=1.0
    d['numeric:query_coverage']=len(shared)/len(q) if q else 0.0
    d['numeric:product_coverage']=len(shared)/len(p) if p else 0.0
    d['numeric:jaccard']=len(shared)/len(q|p) if q|p else 0.0
    d['numeric:log_query_count']=math.log1p(len(q));d['numeric:log_product_count']=math.log1p(len(p));d['numeric:constant']=1.0
    return d
def keyed_record(x):
    obj=json.loads(x)
    assert list(obj)==[str(i) for i in range(1,9)] and all(v in ('Y','N') for v in obj.values())
    return [obj[str(i)] for i in range(1,9)]
def keyed_blocks(rows,records):
    assert len(records)==len(rows)*4
    result=[]
    for q,row in enumerate(rows):
        v=[]
        for i in range(4):
            x=records[q*4+i];assert x['query_id']==row['query_id'] and x['start']==i*8 and x['size']==8
            v.extend(keyed_record(x['final']))
        result.append(v)
    return result
def qwen(name,rows,run):
    if name=='holdout160':
        saved=load(run/'original_outputs.jsonl');assert len(saved)==len(rows)
        result=[]
        for row,x in zip(rows,saved):
            assert x['query_id']==row['query_id'] and len(x['raw'])==4
            p=[]
            for i,chunk in enumerate(x['raw']):
                assert chunk['start']==i*8 and chunk['size']==8
                p.extend(keyed_record(chunk['final']))
            assert p==x['pred'];result.append(p)
        return result
    orig=keyed_blocks(rows,load(run/'original_responses.jsonl'))
    rev=keyed_blocks(rows,load(run/'stable_reverse_responses.jsonl'))
    if name=='fallback96':
        points=load(run/'pointwise_responses.jsonl');assert len(points)==32*len(rows)
        out=[]
        for qi,row in enumerate(rows):
            vals=[]
            for i,x in enumerate(points[qi*32:(qi+1)*32]):
                assert x['query_id']==row['query_id'] and x['start']==i and x['final'] in ('Y','N')
                vals.append(orig[qi][i] if orig[qi][i]==rev[qi][i] else x['final'])
            out.append(vals)
        return out
    points=load(run/'selective_pointwise_responses.jsonl');by={(x['query_id'],x['start']):x['final'] for x in points}
    assert len(points)==len(by)
    saved=load(run/'selective_outputs.jsonl');assert len(saved)==len(rows)
    expected={(row['query_id'],i) for qi,row in enumerate(rows) for i in range(32) if orig[qi][i]!=rev[qi][i]}
    assert set(by)==expected and all(v in ('Y','N') for v in by.values())
    result=[]
    for qi,(row,x) in enumerate(zip(rows,saved)):
        assert row['query_id']==x['query_id']
        p=[orig[qi][i] if orig[qi][i]==rev[qi][i] else by[(row['query_id'],i)] for i in range(32)]
        assert p==x['pred'];result.append(p)
    return result
def query_counts(rows,pred):
    out=[]
    for row,v in zip(rows,pred):
        y=np.asarray([x['gold']=='Y' for x in row['items']]);p=np.asarray([x=='Y' for x in v]);assert len(y)==len(p)==32
        out.append([int(np.sum(y&p)),int(np.sum(~y&p)),int(np.sum(~y&~p)),int(np.sum(y&~p)),abs(int(p.sum())-int(y.sum()))/32])
    return np.asarray(out,dtype=float)
def quality(z):
    tp,fp,tn,fn=z[:,:4].sum(axis=0)
    return {'tp':int(tp),'fp':int(fp),'tn':int(tn),'fn':int(fn),'f1':float(2*tp/(2*tp+fp+fn)),
            'balanced_accuracy':float((tp/(tp+fn)+tn/(tn+fp))/2),'ratio_mae':float(z[:,4].mean())}
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);a=ap.parse_args()
    summary=json.loads((a.run/'summary.json').read_text());saved=load(a.run/'predictions.jsonl')
    train=load(HERE/'run_wands_160/inputs.jsonl');trainq={r['query'].strip().casefold() for r in train};trainp={i['product_id'] for r in train for i in r['items']}
    assert len(train)==160 and sum(len(x['items']) for x in train)==5120
    y=np.array([int(i['gold']=='Y') for r in train for i in r['items']])
    h=FeatureHasher(n_features=2**18,input_type='dict',alternate_sign=False,dtype=np.float32)
    x=h.transform(features(r['query'],i['text']) for r in train for i in r['items'])
    model=LogisticRegression(C=1,solver='lbfgs',max_iter=1000,tol=1e-4,penalty='l2',class_weight=None,random_state=20271211)
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter('always',ConvergenceWarning)
        with threadpool_limits(limits=2):model.fit(x,y)
    assert int(model.n_iter_[0])==summary['training']['iterations']
    assert any(issubclass(w.category,ConvergenceWarning) for w in captured)==summary['training']['convergence_warning']
    for name,selection,run,seed in SETS:
        rows=load(HERE/selection/'inputs.jsonl');n=len(rows);goldq={r['query'].strip().casefold() for r in rows};goldp={i['product_id'] for r in rows for i in r['items']}
        assert not trainq&goldq and not trainp&goldp
        rec=[r for r in saved if r['sample']==name];assert len(rec)==n and all(x['query_id']==r['query_id'] for x,r in zip(rec,rows))
        tx=h.transform(features(r['query'],i['text']) for r in rows for i in r['items'])
        with threadpool_limits(limits=2):prob=model.predict_proba(tx)[:,1]
        assert np.allclose(prob,np.array([p for r in rec for p in r['positive_probability']]),rtol=0,atol=1e-9)
        sup=[['Y' if p>=.5 else 'N' for p in prob[i*32:(i+1)*32]] for i in range(n)]
        ref=qwen(name,rows,HERE/run)
        lex=[['Y' if tokenize(r['query']) and tokenize(r['query'])<=tokenize(it['text']) else 'N' for it in r['items']] for r in rows]
        assert all((rec[i]['supervised'],rec[i]['qwen'],rec[i]['lexical'])==(sup[i],ref[i],lex[i]) for i in range(n))
        z={k:query_counts(rows,v) for k,v in [('supervised',sup),('qwen',ref),('lexical',lex)]}
        for k,matrix in z.items():
            for field,value in quality(matrix).items():assert abs(value-summary['samples'][name]['quality'][k][field])<1e-12,(name,k,field)
        draw=np.random.default_rng(seed).integers(0,n,size=(4000,n))
        def bootf1(a):
            c=a[draw,:4].sum(axis=1);return np.divide(2*c[:,0],2*c[:,0]+c[:,1]+c[:,3],out=np.zeros(len(draw)),where=(2*c[:,0]+c[:,1]+c[:,3])!=0)
        cf=bootf1(z['supervised'])-bootf1(z['qwen'])
        cm=z['supervised'][draw,4].mean(axis=1)-z['qwen'][draw,4].mean(axis=1)
        actual=summary['samples'][name]['supervised_minus_qwen']
        assert np.allclose(np.quantile(cf,[.025,.975]),actual['f1_difference_ci'],rtol=0,atol=1e-12)
        assert np.allclose(np.quantile(cm,[.025,.975]),actual['ratio_mae_difference_ci'],rtol=0,atol=1e-12)
        print(name,'queries',n,'training/query and product overlap',len(trainq&goldq),len(trainp&goldp),'verified')
    print('Independent re-fit, probabilities, Qwen output reconstruction, metrics and paired bootstrap verified.')
if __name__=='__main__':main()
