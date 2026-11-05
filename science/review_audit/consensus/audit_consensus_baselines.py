#!/usr/bin/env python3
"""Post-hoc trace audit. See PROTOCOL_consensus_audit.md; no model inference."""
import argparse, hashlib, json, math
from fractions import Fraction
from pathlib import Path
import numpy as np

SETTINGS = (
    ('WANDS/Qwen','wands_selective_execution_selection','run_wands_selective_execution_qwen_v1',52),
    ('WANDS/SmolLM2','wands_selective_execution_selection','run_wands_selective_execution_smollm_v1',52),
    ('ESCI/Qwen','esci_selective_holdout_selection','run_esci_selective_holdout_qwen',250),
    ('ESCI/SmolLM2','esci_selective_holdout_selection','run_esci_selective_holdout_smollm',250),
)
FILES = ('original_responses.jsonl','stable_reverse_responses.jsonl','pointwise_responses.jsonl',
         'selective_pointwise_responses.jsonl','selective_outputs.jsonl','summary.json')
NAMES = ('keyed','reverse','pointwise','selective','cached','and','or')
METRICS = ('ratio_mae','candidate_error_macro','f1','balanced_accuracy')

def read(p):
    # Literal newline only: product texts can contain escaped/decoded Unicode line separators.
    return [json.loads(x) for x in p.read_text(encoding='utf8').split('\n') if x.strip()]

def keyed_vectors(rows, records):
    expected=[(r['query_id'],s,min(8,len(r['items'])-s)) for r in rows for s in range(0,len(r['items']),8)]
    assert [(r['query_id'],r['start'],r['size']) for r in records] == expected
    result=[];at=0
    for row in rows:
        labels=[]
        for start in range(0,len(row['items']),8):
            size=min(8,len(row['items'])-start)
            pairs=json.loads(records[at]['final'],object_pairs_hook=list);at+=1
            assert len(pairs)==size and [k for k,v in pairs]==list(map(str,range(1,size+1)))
            assert all(v in ('Y','N') for k,v in pairs)
            labels.extend(v=='Y' for k,v in pairs)
        assert len(labels)==len(row['items']);result.append(np.asarray(labels,dtype=bool))
    return result

def exact_uniform(a,h,g,k):
    plus=int(np.count_nonzero(~a&h));minus=int(np.count_nonzero(a&~h));zero=len(a)-plus-minus
    delta=int(a.sum())-int(g.sum());total=math.comb(len(a),k);weight=count=0
    for u in range(min(plus,k)+1):
        for v in range(min(minus,k-u)+1):
            z=k-u-v
            if z>zero:continue
            ways=math.comb(plus,u)*math.comb(minus,v)*math.comb(zero,z)
            count+=ways;weight+=ways*abs(delta+u-v)
    assert count==total
    return Fraction(weight,len(a)*total)

def sufficient(g,p):
    tp=np.count_nonzero(g&p);fp=np.count_nonzero(~g&p);fn=np.count_nonzero(g&~p);tn=np.count_nonzero(~g&~p)
    return np.array([tp,fp,fn,tn,abs(fp-fn)/len(g),(fp+fn)/len(g)],dtype=float)

def aggregate(x):
    # x ends in [pool, sufficient-statistic]; any leading dimensions are resamples.
    tp,fp,fn,tn=np.moveaxis(x[...,:4].sum(axis=-2),-1,0)
    f1=np.divide(2*tp,2*tp+fp+fn,out=np.zeros_like(tp),where=(2*tp+fp+fn)!=0)
    tpr=np.divide(tp,tp+fn,out=np.full_like(tp,np.nan),where=(tp+fn)!=0)
    tnr=np.divide(tn,tn+fp,out=np.full_like(tn,np.nan),where=(tn+fp)!=0)
    return {'ratio_mae':x[...,4].mean(axis=-1),'candidate_error_macro':x[...,5].mean(axis=-1),
            'f1':f1,'balanced_accuracy':(tpr+tnr)/2}

def plain(d):
    return {k:(float(v) if np.isfinite(v) else None) for k,v in d.items()}

def contrast(x,y,draw):
    left,right=aggregate(x),aggregate(y);bl,br=aggregate(x[draw]),aggregate(y[draw])
    return {k:{'difference':float(left[k]-right[k]),'unadjusted_query_interval_95':np.quantile(bl[k]-br[k],[.025,.975]).tolist()}
            for k in METRICS}

def mean_contrast(d,seed):
    d=np.asarray(d,dtype=float);draw=np.random.default_rng(seed).integers(0,len(d),size=(4000,len(d)))
    return {'difference':float(d.mean()),'unadjusted_query_interval_95':np.quantile(d[draw].mean(axis=1),[.025,.975]).tolist()}

def main(base,out):
    if out.exists():raise FileExistsError(out)
    result={'design':'Post-hoc descriptive audit of fixed, complete cached traces; see accompanying protocol.',
            'unique_pools':302,'pool_model_instances':604,'bootstrap_draws':4000,'numpy':np.__version__,
            'source_sha256':{},'settings':{}}
    old=json.loads((base/'posthoc_routing_budget/routing_budget_result.json').read_text())['settings']
    result['source_sha256']['posthoc_routing_budget/routing_budget_result.json']=hashlib.sha256((base/'posthoc_routing_budget/routing_budget_result.json').read_bytes()).hexdigest()
    for si,(name,selection,run,count) in enumerate(SETTINGS):
        seed=2026110210+si
        for rel in [selection+'/inputs.jsonl']+[run+'/'+f for f in FILES]:
            result['source_sha256'][rel]=hashlib.sha256((base/rel).read_bytes()).hexdigest()
        rows=read(base/selection/'inputs.jsonl');assert len(rows)==count
        a=keyed_vectors(rows,read(base/run/FILES[0]));b=keyed_vectors(rows,read(base/run/FILES[1]))
        raw=read(base/run/FILES[2]);wanted=[(r['query_id'],i) for r in rows for i in range(len(r['items']))]
        assert [(r['query_id'],r['start']) for r in raw]==wanted and all(r['final'] in ('Y','N') for r in raw)
        hmap={(r['query_id'],r['start']):r['final']=='Y' for r in raw}
        sparse=read(base/run/FILES[3]);needed=[(r['query_id'],i) for r,x,y in zip(rows,a,b) for i in np.flatnonzero(x!=y)]
        assert [(r['query_id'],r['start']) for r in sparse]==needed and all(r['final'] in ('Y','N') for r in sparse)
        smap={(r['query_id'],r['start']):r['final']=='Y' for r in sparse};assert len(smap)==len(sparse)
        saved=read(base/run/FILES[4]);assert [x['query_id'] for x in saved]==[x['query_id'] for x in rows]
        stats={k:[] for k in NAMES};per_pool=[]
        for qi,(row,x,y) in enumerate(zip(rows,a,b)):
            g=np.asarray([i['gold']=='Y' for i in row['items']]);n=len(g);k=int(np.count_nonzero(x!=y))
            h=np.asarray([hmap[row['query_id'],i] for i in range(n)])
            s=np.asarray([x[i] if x[i]==y[i] else smap[row['query_id'],i] for i in range(n)])
            assert saved[qi]['pred']==['Y' if t else 'N' for t in s]
            plans={'keyed':x,'reverse':y,'pointwise':h,'selective':s,'cached':np.where(x==y,x,h),'and':x&y,'or':x|y}
            for key,p in plans.items():stats[key].append(sufficient(g,p))
            exact=exact_uniform(x,h,g,k);gold=int(g.sum());stratum='all_negative' if gold==0 else ('all_positive' if gold==n else 'mixed')
            prior=old[name]['per_pool'][qi];assert prior['query_id']==row['query_id'] and prior['n']==n and prior['k']==k
            assert exact==Fraction(*map(int,prior['expected_random_mae_exact']))
            per_pool.append({'query_id':row['query_id'],'n':n,'gold_positives':gold,'stratum':stratum,'k':k,
                'expected_uniform_mae':float(exact),'expected_uniform_mae_exact':[exact.numerator,exact.denominator],
                'pointwise_y_on_disagreements':int(h[x!=y].sum()),
                'selective_equals_and':bool(np.array_equal(s,x&y)),
                'metrics':{key:plain(aggregate(np.asarray([stats[key][-1]]))) for key in NAMES}})
        stats={k:np.stack(v) for k,v in stats.items()};draw=np.random.default_rng(seed).integers(0,count,size=(4000,count))
        summary=json.loads((base/run/'summary.json').read_text());total=summary['selective_pipeline'];point=summary['modes']['selective_pointwise']
        costs={'scope':'Trace-derived calls and logical tokens; no AND/OR latency was executed.',
               'actual_selective':{'calls':total['initial_calls']+total['retry_calls'],'logical_tokens':total['logical_tokens']},
               'two_keyed_and_or':{'calls':total['initial_calls']+total['retry_calls']-point['initial_calls']-point['retry_calls'],
                   'logical_tokens':{k:total['logical_tokens'][k]-point['logical_tokens'][k] for k in ('input','output')}},
               'pointwise_calls_saved':point['initial_calls']+point['retry_calls']}
        expected=np.asarray([r['expected_uniform_mae'] for r in per_pool]);delta=stats['cached'][:,4]-expected
        groups={}
        for gi,label in enumerate(('all_negative','mixed','all_positive')):
            mask=np.asarray([r['stratum']==label for r in per_pool]);size=int(mask.sum())
            if not size:groups[label]={'pools':0};continue
            gs=seed+100+gi;groups[label]={'pools':size,'bootstrap_seed':gs,
                'metrics':{k:plain(aggregate(v[mask])) for k,v in stats.items()},
                'expected_uniform_mae':float(expected[mask].mean()),
                'cached_minus_uniform':mean_contrast(delta[mask],gs),
                'contribution_to_overall_cached_minus_uniform':float(delta[mask].sum()/count)}
        result['settings'][name]={'selection':selection,'run':run,'pools':count,'candidates':sum(len(r['items']) for r in rows),
            'bootstrap_seed':seed,'metrics':{k:plain(aggregate(v)) for k,v in stats.items()},
            'selective_minus_and':contrast(stats['selective'],stats['and'],draw),
            'selective_minus_or':contrast(stats['selective'],stats['or'],draw),
            'cached_minus_uniform':mean_contrast(delta,seed),'expected_uniform_mae':float(expected.mean()),
            'cost':costs,'label_composition':groups,'per_pool':per_pool}
        print(name,'AND/OR/selective MAE:',*[round(result['settings'][name]['metrics'][k]['ratio_mae'],6) for k in ('and','or','selective')])
    out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--artifact',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();main(a.artifact,a.out)
