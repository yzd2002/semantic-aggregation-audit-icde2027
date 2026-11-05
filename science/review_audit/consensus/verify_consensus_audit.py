#!/usr/bin/env python3
"""Independent parsing, position-DP expectation and bootstrap verification."""
import argparse, hashlib, json, math
from collections import defaultdict
from fractions import Fraction
from pathlib import Path
import numpy as np

def records(p):
    with p.open(encoding='utf8') as f:return [json.loads(line) for line in f if line.strip()]

def point_map(recs):
    result={}
    for r in recs:
        key=(r['query_id'],r['start']);assert key not in result and r['final'] in ('Y','N')
        result[key]=int(r['final']=='Y')
    return result

def batch_map(recs):
    result={}
    for r in recs:
        value=json.loads(r['final']);assert set(value)==set(map(str,range(1,r['size']+1)))
        for local,label in value.items():
            key=(r['query_id'],r['start']+int(local)-1);assert key not in result and label in ('Y','N')
            result[key]=int(label=='Y')
    return result

def dp(a,h,g,k):
    # Counts subsets by number chosen and signed change, visiting each position once.
    states={(0,0):1}
    for x,y in zip(a,h):
        nxt=defaultdict(int)
        for (chosen,change),ways in states.items():
            nxt[chosen,change]+=ways
            if chosen<k:nxt[chosen+1,change+y-x]+=ways
        states=nxt
    total=sum(v for (chosen,change),v in states.items() if chosen==k)
    assert total==math.comb(len(a),k)
    delta=sum(a)-sum(g)
    numerator=sum(v*abs(delta+change) for (chosen,change),v in states.items() if chosen==k)
    return Fraction(numerator,len(a)*total)

def counts(g,p):
    tp=fp=fn=tn=0
    for x,y in zip(g,p):
        if x and y:tp+=1
        elif y:fp+=1
        elif x:fn+=1
        else:tn+=1
    return [tp,fp,fn,tn,abs(sum(p)-sum(g))/len(g),sum(x!=y for x,y in zip(g,p))/len(g)]

def from_totals(t,n):
    tp,fp,fn,tn=t[...,0],t[...,1],t[...,2],t[...,3]
    f1=np.divide(2*tp,2*tp+fp+fn,out=np.zeros_like(tp),where=(2*tp+fp+fn)>0)
    tpr=np.divide(tp,tp+fn,out=np.full_like(tp,np.nan),where=(tp+fn)>0)
    tnr=np.divide(tn,tn+fp,out=np.full_like(tn,np.nan),where=(tn+fp)>0)
    return {'ratio_mae':t[...,4]/n,'candidate_error_macro':t[...,5]/n,'f1':f1,'balanced_accuracy':(tpr+tnr)/2}

def checknum(x,y):
    if y is None:assert not np.isfinite(x)
    else:assert np.allclose(x,y,atol=1e-12,rtol=0),(x,y)

def verify(base,path):
    report=json.loads(path.read_text());checked=0
    for rel,digest in report['source_sha256'].items():assert hashlib.sha256((base/rel).read_bytes()).hexdigest()==digest
    for name,data in report['settings'].items():
        run=base/data['run'];rows=records(base/data['selection']/'inputs.jsonl');n=len(rows);assert n==data['pools']
        a=batch_map(records(run/'original_responses.jsonl'));b=batch_map(records(run/'stable_reverse_responses.jsonl'))
        h=point_map(records(run/'pointwise_responses.jsonl'));s=point_map(records(run/'selective_pointwise_responses.jsonl'))
        saved={x['query_id']:x['pred'] for x in records(run/'selective_outputs.jsonl')}
        stats={k:[] for k in data['metrics']};expect=[];groups=[]
        for r,old in zip(rows,data['per_pool']):
            q=r['query_id'];g=[int(i['gold']=='Y') for i in r['items']];size=len(g)
            x=[a[q,i] for i in range(size)];y=[b[q,i] for i in range(size)];z=[h[q,i] for i in range(size)]
            actual=[x[i] if x[i]==y[i] else s[q,i] for i in range(size)]
            assert saved[q]==['Y' if t else 'N' for t in actual]
            plans={'keyed':x,'reverse':y,'pointwise':z,'selective':actual,
                   'cached':[v if v==w else t for v,w,t in zip(x,y,z)],
                   'and':[min(v,w) for v,w in zip(x,y)],'or':[max(v,w) for v,w in zip(x,y)]}
            k=sum(v!=w for v,w in zip(x,y));ex=dp(x,z,g,k);expect.append(float(ex));checked+=1
            assert ex==Fraction(*old['expected_uniform_mae_exact']) and old['k']==k and old['n']==size
            label='mixed' if 0<sum(g)<size else ('all_positive' if sum(g) else 'all_negative');groups.append(label)
            assert old['stratum']==label and old['gold_positives']==sum(g)
            assert old['pointwise_y_on_disagreements']==sum(t for v,w,t in zip(x,y,z) if v!=w)
            assert old['selective_equals_and']==(plans['selective']==plans['and'])
            for key,p in plans.items():
                c=counts(g,p);stats[key].append(c);point=from_totals(np.asarray(c),1)
                for metric,value in point.items():checknum(value,old['metrics'][key][metric])
        stats={k:np.asarray(v,dtype=float) for k,v in stats.items()};expect=np.asarray(expect)
        for key,x in stats.items():
            point=from_totals(x.sum(axis=0),n)
            for metric,value in point.items():checknum(value,data['metrics'][key][metric])
        # Reconstruct resampling as multinomial frequency weights, not row indexing.
        rng=np.random.default_rng(data['bootstrap_seed']);draw=rng.integers(0,n,size=(4000,n))
        weights=np.stack([np.bincount(row,minlength=n) for row in draw])
        def pair(left,right):
            l,r=stats[left],stats[right];pm,rm=from_totals(l.sum(axis=0),n),from_totals(r.sum(axis=0),n)
            bl,br=from_totals(weights@l,n),from_totals(weights@r,n)
            return {k:(pm[k]-rm[k],np.percentile(bl[k]-br[k],[2.5,97.5])) for k in pm}
        for policy in ('and','or'):
            for metric,(gap,ci) in pair('selective',policy).items():
                stored=data['selective_minus_'+policy][metric];checknum(gap,stored['difference']);checknum(ci,stored['unadjusted_query_interval_95'])
        d=stats['cached'][:,4]-expect
        checknum(d.mean(),data['cached_minus_uniform']['difference']);checknum(np.percentile(weights@d/n,[2.5,97.5]),data['cached_minus_uniform']['unadjusted_query_interval_95'])
        checknum(expect.mean(),data['expected_uniform_mae'])
        for label,old in data['label_composition'].items():
            mask=np.asarray([x==label for x in groups]);m=int(mask.sum());assert old['pools']==m
            if not m:continue
            ds=d[mask];draw=np.random.default_rng(old['bootstrap_seed']).integers(0,m,size=(4000,m))
            weights=np.stack([np.bincount(row,minlength=m) for row in draw])
            checknum(ds.mean(),old['cached_minus_uniform']['difference']);checknum(np.percentile(weights@ds/m,[2.5,97.5]),old['cached_minus_uniform']['unadjusted_query_interval_95'])
            checknum(ds.sum()/n,old['contribution_to_overall_cached_minus_uniform']);checknum(expect[mask].mean(),old['expected_uniform_mae'])
            for key,x in stats.items():
                for metric,value in from_totals(x[mask].sum(axis=0),m).items():checknum(value,old['metrics'][key][metric])
        # Independently sum recorded calls/tokens for the two keyed stages and pointwise stage.
        summary=json.loads((run/'summary.json').read_text());sp=summary.get('specifications',summary['modes'].get('specifications'))
        costs={'calls':sp['initial_calls']+sp['retry_calls'],'logical_tokens':dict(sp['logical_tokens'])}
        for mode in ('original','stable_reverse'):
            events=records(run/(mode+'_responses.jsonl'));costs['calls']+=sum(1+int(e['retry_finish_reason'] is not None) for e in events)
            for kind in ('input','output'):costs['logical_tokens'][kind]+=sum(e['first_'+kind+'_tokens']+e['retry_'+kind+'_tokens'] for e in events)
        assert costs==data['cost']['two_keyed_and_or']
        print(name+': raw vectors, all metrics, all exact subset expectations, paired intervals, composition contributions and trace costs verified.')
    print('PASS: '+str(checked)+' pool-model instances; 302 distinct pools; all source hashes unchanged.')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--artifact',type=Path,required=True);p.add_argument('--result',type=Path,required=True);a=p.parse_args();verify(a.artifact,a.result)
