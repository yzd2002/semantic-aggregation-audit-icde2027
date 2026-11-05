#!/usr/bin/env python3
"""Independent reconstruction and integer check for the cancellation diagnostic."""
import argparse,hashlib,json,math
from pathlib import Path
def read(p):return [json.loads(s) for s in p.read_text(encoding='utf8').split('\n') if s.strip()]
def vector(text,n):
    pairs=json.loads(text,object_pairs_hook=lambda x:x)
    if not isinstance(pairs,list) or pairs!=[(str(i),v) for i,(_,v) in enumerate(pairs,1)] or len(pairs)!=n:raise ValueError('Unexpected local IDs')
    if any(v not in ('Y','N') for _,v in pairs) or json.dumps(dict(pairs),separators=(',',':'))!=text:raise ValueError('Invalid canonical output')
    return [v for _,v in pairs]
def main(base,result):
    saved=json.loads(result.read_text())
    for n,h in saved['source_sha256'].items():assert hashlib.sha256((base/n).read_bytes()).hexdigest()==h
    choices=(('wands52','wands_selective_execution_selection','run_wands_selective_execution_qwen_v1'),('esci250','esci_selective_holdout_selection','run_esci_selective_holdout_qwen'))
    for name,sel,run in choices:
        rows=read(base/sel/'inputs.jsonl');folder=base/run
        maps={}
        for mode in ('original','stable_reverse','pointwise','selective_pointwise'):
            data=read(folder/(mode+'_responses.jsonl'));m={(r['query_id'],r['start']):r for r in data};assert len(data)==len(m);maps[mode]=m
        allpred={k:[] for k in ('keyed','selective','pointwise','always_Y','always_N')}
        for row in rows:
            q=row['query_id'];n=len(row['items']);a=[];b=[]
            for start in range(0,n,8):
                size=min(8,n-start);a+=vector(maps['original'][q,start]['final'],size);b+=vector(maps['stable_reverse'][q,start]['final'],size)
            full=[maps['pointwise'][q,i]['final'] for i in range(n)];select=[x if x==y else maps['selective_pointwise'][q,i]['final'] for i,(x,y) in enumerate(zip(a,b))]
            assert all(x in ('Y','N') for x in full+select)
            for k,v in (('keyed',a),('selective',select),('pointwise',full),('always_Y',['Y']*n),('always_N',['N']*n)):allpred[k].append(v)
        assert saved['settings'][name]['pools']==len(rows) and saved['settings'][name]['candidates']==sum(len(r['items']) for r in rows)
        for plan,pred in allpred.items():
            tp=fp=fn=tn=0;err=[];mae=[];cancel=[];signed=[];exact=hidden=opp=0
            for row,p in zip(rows,pred):
                gold=[x['gold'] for x in row['items']];n=len(p);bad=sum(x!=y for x,y in zip(gold,p));delta=p.count('Y')-gold.count('Y')
                assert bad>=abs(delta) and (bad-abs(delta))%2==0
                err.append(bad/n);mae.append(abs(delta)/n);cancel.append((bad-abs(delta))/n);signed.append(delta/n)
                exact+=delta==0;hidden+=delta==0 and bad>0;opp+=bad>abs(delta)
                tp+=sum(x==y=='Y' for x,y in zip(gold,p));tn+=sum(x==y=='N' for x,y in zip(gold,p))
                fp+=sum(x=='N' and y=='Y' for x,y in zip(gold,p));fn+=sum(x=='Y' and y=='N' for x,y in zip(gold,p))
            avg=lambda v:math.fsum(v)/len(rows)
            expected={'tp':tp,'fp':fp,'fn':fn,'tn':tn,'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,
                      'candidate_error_macro':avg(err),'ratio_mae':avg(mae),'cancelled_error_macro':avg(cancel),'signed_ratio_bias':avg(signed),
                      'exact_ratio_pools':exact,'exact_ratio_with_candidate_errors':hidden,'pools_with_opposing_errors':opp}
            actual=saved['settings'][name]['plans'][plan];assert expected.keys()==actual.keys()
            assert all(math.isclose(v,actual[k],rel_tol=1e-12,abs_tol=1e-12) for k,v in expected.items()),(name,plan)
        print(name,': all five plans independently verified against fixed candidate labels.')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--artifact',type=Path,required=True);p.add_argument('--result',type=Path,required=True);a=p.parse_args();main(a.artifact,a.result)
