#!/usr/bin/env python3
"""Independent position-DP verification, separate parsing and bootstrap check."""
import argparse,hashlib,json,math
from collections import defaultdict
from fractions import Fraction
from pathlib import Path
import numpy as np

SETTINGS=(
 ('WANDS/Qwen','wands_selective_execution_selection','run_wands_selective_execution_qwen_v1'),
 ('WANDS/SmolLM2','wands_selective_execution_selection','run_wands_selective_execution_smollm_v1'),
 ('ESCI/Qwen','esci_selective_holdout_selection','run_esci_selective_holdout_qwen'),
 ('ESCI/SmolLM2','esci_selective_holdout_selection','run_esci_selective_holdout_smollm'))
def read(p):return [json.loads(s) for s in p.read_text(encoding='utf8').split('\n') if s.strip()]
def keyed(text,n):
    pairs=json.loads(text,object_pairs_hook=lambda xs:xs)
    assert len(pairs)==n and [k for k,v in pairs]==list(map(str,range(1,n+1)))
    assert all(v in ('Y','N') for k,v in pairs)
    assert json.dumps(dict(pairs),separators=(',',':'))==text
    return [v for k,v in pairs]
def counts(deltas,k):
    states={(0,0):1}
    for delta in deltas:
        update=defaultdict(int)
        for (taken,value),weight in states.items():
            update[taken,value]+=weight
            if taken<k:update[taken+1,value+delta]+=weight
        states=update
    result={value:weight for (taken,value),weight in states.items() if taken==k}
    assert sum(result.values())==math.comb(len(deltas),k)
    return result
def main(base,path):
    result=json.loads(path.read_text())
    for rel,h in result['source_sha256'].items():assert hashlib.sha256((base/rel).read_bytes()).hexdigest()==h
    for idx,(name,sel,run) in enumerate(SETTINGS):
        rows=read(base/sel/'inputs.jsonl');saved=result['settings'][name];records={}
        for mode in ('original','stable_reverse','pointwise'):
            rec=read(base/run/(mode+'_responses.jsonl'));m={(x['query_id'],x['start']):x for x in rec};assert len(m)==len(rec);records[mode]=m
        stats=[];error_agree=error_disagree=budget=total=0
        for row,observed in zip(rows,saved['per_pool']):
            q=row['query_id'];n=len(row['items']);a=[];b=[];h=[records['pointwise'][q,i]['final'] for i in range(n)]
            assert all(x in ('Y','N') for x in h)
            for start in range(0,n,8):
                size=min(8,n-start);a+=keyed(records['original'][q,start]['final'],size);b+=keyed(records['stable_reverse'][q,start]['final'],size)
            gold=[x['gold'] for x in row['items']];k=sum(x!=y for x,y in zip(a,b));route=[h[i] if a[i]!=b[i] else a[i] for i in range(n)]
            change=[(y=='Y')-(x=='Y') for x,y in zip(a,h)]
            dp=counts(change,k);delta=a.count('Y')-gold.count('Y')
            expected=Fraction(sum(weight*abs(delta+v) for v,weight in dp.items()),n*sum(dp.values()))
            assert expected==Fraction(*map(int,observed['expected_random_mae_exact']))
            # Independent candidate expectation from integer inverse weighting.
            wrong_base=sum(x!=y for x,y in zip(a,gold));wrong_h=sum(x!=y for x,y in zip(h,gold))
            ec=Fraction((n-k)*wrong_base+k*wrong_h,n*n)
            assert ec==Fraction(*map(int,observed['expected_random_candidate_error_exact']))
            ea=sum(x!=y and x==z for x,y,z in zip(a,gold,b));ed=wrong_base-ea
            assert observed['query_id']==q and observed['n']==n and observed['k']==k
            assert observed['keyed_errors_agree']==ea and observed['keyed_errors_disagree']==ed
            actual={'routed_mae':abs(route.count('Y')-gold.count('Y'))/n,'expected_random_mae':float(expected),
                    'routed_candidate_error':sum(x!=y for x,y in zip(route,gold))/n,'expected_random_candidate_error':float(ec)}
            assert all(math.isclose(v,observed[key],abs_tol=1e-12,rel_tol=1e-12) for key,v in actual.items())
            stats.append(actual);error_agree+=ea;error_disagree+=ed;budget+=k;total+=n
        n=len(rows);assert len(stats)==len(saved['per_pool'])==saved['pools']==n
        assert saved['candidates']==total and saved['pointwise_label_budget']==budget
        assert saved['keyed_errors_agree']==error_agree and saved['keyed_errors_disagree']==error_disagree
        assert math.isclose(saved['budget_fraction'],budget/total,abs_tol=1e-12)
        for key,value in saved['means'].items():assert math.isclose(value,math.fsum(x[key] for x in stats)/n,abs_tol=1e-12)
        generator=np.random.Generator(np.random.PCG64(20261103+idx));draw=generator.integers(n,size=(4000,n))
        for metric,left,right in (('ratio_mae','routed_mae','expected_random_mae'),('candidate_error','routed_candidate_error','expected_random_candidate_error')):
            diff=np.asarray([x[left]-x[right] for x in stats]);saved_comp=saved['routed_minus_expected_random'][metric]
            assert math.isclose(float(np.mean(diff)),saved_comp['difference'],abs_tol=1e-12)
            assert np.allclose(np.percentile(diff[draw].mean(axis=1),[2.5,97.5]),saved_comp['unadjusted_query_interval'],atol=1e-12,rtol=1e-12)
        print(name+': all pools, exact subset expectations, routing, means and unadjusted intervals independently verified.')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--artifact',type=Path,required=True);p.add_argument('--result',type=Path,required=True);a=p.parse_args();main(a.artifact,a.result)
