#!/usr/bin/env python3
"""Independent integer reconstruction of secondary key-order and failure ledger."""
import argparse,json
from fractions import Fraction
import numpy as np
from common import ROOT,CONDITIONS,read,sha
from verify_controlled import raw_parse,close

def main(model):
    old=json.loads((ROOT/f'run_{model}/supplementary_analysis.json').read_text());planned=read(ROOT/'selection/schedule.jsonl');path=ROOT/f'run_{model}/responses.jsonl';records=read(path) if path.exists() else []
    parsed={r['request_id']:raw_parse(r['raw'],r['ids'],r['condition'].endswith('D')) for r in records}
    ledger=old['execution_ledger'];assert ledger['planned_jobs']==len(planned) and ledger['returned_jobs']==len(records)
    assert ledger['invalid_request_ids']==[r['request_id'] for r in records if parsed[r['request_id']] is None]
    assert ledger['unreturned_request_ids']==[j['request_id'] for j in planned if j['request_id'] not in parsed]
    assert ledger['unreturned_jobs']==len(ledger['unreturned_request_ids']) and ledger['invalid_returned_jobs']==len(ledger['invalid_request_ids'])
    assert ledger['valid_returned_jobs']==len(records)-ledger['invalid_returned_jobs']
    for di,d in enumerate(('esci','wands')):
        rows=read(ROOT/'selection'/f'{d}_inputs.jsonl');store={r['query_id']:r for r in rows};vec={(q,c,k):{} for q in store for c in (*CONDITIONS,'S') for k in (0,1)}
        attempted={r['query_id'] for r in records if r['dataset']==d};entry=old['datasets'][d]
        for r in records:
            if r['dataset']==d and parsed[r['request_id']]:vec[r['query_id'],r['condition'],r['repeat']].update({int(i):int(v=='Y') for i,v in parsed[r['request_id']].items()})
        done=lambda q,c,k:len(vec[q,c,k])==len(store[q]['items'])
        value=lambda q,c,k:[vec[q,c,k][i] for i in range(1,len(store[q]['items'])+1)]
        batch=[q for q in store if all(done(q,c,k) for c in CONDITIONS for k in (0,1))];single=[q for q in store if done(q,'S',0) and done(q,'S',1)]
        assert entry['all8_complete_query_ids']==batch and entry['singleton_own_complete_query_ids']==single
        assert entry['attempted_pools']==len(attempted) and entry['wholly_unattempted_query_ids']==[q for q in store if q not in attempted]
        for c in (*CONDITIONS,'S'):
            for k in (0,1):
                assert entry['completion'][c+str(k)]==sum(done(q,c,k) for q in store)
                for kind in ('input','output'):assert entry['actual_tokens'][c+str(k)][kind]==sum(r['actual_'+kind+'_tokens'] for r in records if r['dataset']==d and r['condition']==c and r['repeat']==k)
        for k in (0,1):
            tp=fp=fn=tn=0;mae=[];cer=[]
            for q in single:
                g=[int(i['gold']=='Y') for i in store[q]['items']];p=value(q,'S',k)
                tp+=sum(a==b==1 for a,b in zip(g,p));fp+=sum(a==0 and b==1 for a,b in zip(g,p));fn+=sum(a==1 and b==0 for a,b in zip(g,p));tn+=sum(a==b==0 for a,b in zip(g,p))
                mae.append(Fraction(abs(sum(g)-sum(p)),len(g)));cer.append(Fraction(sum(a!=b for a,b in zip(g,p)),len(g)))
            metrics={'tp':tp,'fp':fp,'fn':fn,'tn':tn,'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0,
                'balanced_accuracy':(tp/(tp+fn)+tn/(tn+fp))/2 if tp+fn and tn+fp else None,
                'ratio_mae':float(sum(mae)/len(single)) if single else None,'candidate_error_macro':float(sum(cer)/len(single)) if single else None}
            for metric,v in metrics.items():close(v,entry['singleton_quality_own'][str(k)][metric])
        if batch:
            n=len(batch);seed=2026110220+2*(model=='smol')+di;draw=np.random.default_rng(seed).integers(0,n,size=(4000,n));w=np.stack([np.bincount(ix,minlength=n) for ix in draw])
            def check(values,old):
                ar=np.array(values,dtype=float);close(ar.mean(),old['mean']);close(np.quantile(w@ar/n,[.025,.975]),old['unadjusted_query_interval_95'])
            for inp,ca,cd in [('A','AA','AD'),('R','RA','RD')]:
                for metric in ('candidate_disagreement','absolute_ratio_change'):
                    dist=lambda a,b:Fraction(sum(x!=y for x,y in zip(a,b)),len(a)) if metric=='candidate_disagreement' else Fraction(abs(sum(a)-sum(b)),len(a))
                    change=[sum(dist(value(q,ca,k),value(q,cd,k)) for k in (0,1))/2 for q in batch]
                    replay=[sum(dist(value(q,c,0),value(q,c,1)) for c in (ca,cd))/2 for q in batch]
                    for which,vals in [('key_change',change),('mean_replay',replay),('excess',[x-y for x,y in zip(change,replay)])]:check(vals,entry['secondary_key_order'][inp][metric][which])
                for metric in ('reference_ratio_mae_D_minus_A','candidate_error_D_minus_A'):
                    values=[]
                    for q in batch:
                        g=[int(i['gold']=='Y') for i in store[q]['items']]
                        loss=lambda p:Fraction(abs(sum(g)-sum(p)),len(g)) if metric.startswith('reference') else Fraction(sum(a!=b for a,b in zip(g,p)),len(g))
                        values.append(sum(loss(value(q,cd,k))-loss(value(q,ca,k)) for k in (0,1))/2)
                    check(values,entry['secondary_key_order'][inp][metric])
        print('PASS supplement',model,d,': raw execution ledger, singleton own-set quality, tokens and direct key-order effects/intervals independently verified.')
    manifest=json.loads((ROOT/'SUPPLEMENT_FROZEN_SHA256.json').read_text())
    for rel,h in manifest['files'].items():assert sha(ROOT/rel)==h
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',choices=['qwen','smol'],required=True);main(p.parse_args().model)
