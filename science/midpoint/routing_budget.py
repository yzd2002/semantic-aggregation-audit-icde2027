#!/usr/bin/env python3
"""Exact, conditional matched-label-budget ablation on already observed traces."""
import argparse,hashlib,json,math
from fractions import Fraction
from pathlib import Path
import numpy as np
from error_cancellation import read,vectors,keyed_pools,summarize,FILES

SETTINGS=(
 ('WANDS/Qwen','wands_selective_execution_selection','run_wands_selective_execution_qwen_v1','run_wands_selective_execution_qwen_v1/paired_analysis.json',52),
 ('WANDS/SmolLM2','wands_selective_execution_selection','run_wands_selective_execution_smollm_v1','run_wands_selective_execution_smollm_v1/paired_analysis.json',52),
 ('ESCI/Qwen','esci_selective_holdout_selection','run_esci_selective_holdout_qwen','analysis_esci_selective_holdout_qwen/summary.json',250),
 ('ESCI/SmolLM2','esci_selective_holdout_selection','run_esci_selective_holdout_smollm','analysis_esci_selective_holdout_smollm/summary.json',250))

def exact_random_mae(plus,minus,zero,k,delta):
    """E |delta + sum(selected(h-a))|/n under a uniform k-subset."""
    n=plus+minus+zero
    if min(plus,minus,zero,k)<0 or n==0 or k>n:raise ValueError('Invalid counts')
    weight=total=0
    for pos in range(min(plus,k)+1):
        for neg in range(min(minus,k-pos)+1):
            unchanged=k-pos-neg
            if unchanged>zero:continue
            count=math.comb(plus,pos)*math.comb(minus,neg)*math.comb(zero,unchanged)
            total+=count;weight+=count*abs(delta+pos-neg)
    assert total==math.comb(n,k)
    return Fraction(weight,n*total)

def pool_stat(row,a,b,h):
    gold=[x['gold'] for x in row['items']];n=len(gold)
    assert n>0 and len(a)==len(b)==len(h)==n and set(gold+a+b+h)<=set('YN')
    routed=[u if u==v else z for u,v,z in zip(a,b,h)]
    k=sum(u!=v for u,v in zip(a,b));g=gold.count('Y');pa=a.count('Y')
    err=lambda p:sum(x!=y for x,y in zip(p,gold))
    ea,eh,er=err(a),err(h),err(routed)
    plus=sum(x=='N' and y=='Y' for x,y in zip(a,h));minus=sum(x=='Y' and y=='N' for x,y in zip(a,h))
    expected_mae=exact_random_mae(plus,minus,n-plus-minus,k,pa-g)
    expected_error=Fraction(ea,n)+Fraction(k,n)*Fraction(eh-ea,n)
    exact=lambda f:[str(f.numerator),str(f.denominator)]
    return {'query_id':row['query_id'],'n':n,'k':k,
            'routed_mae':abs(routed.count('Y')-g)/n,'expected_random_mae':float(expected_mae),
            'routed_candidate_error':er/n,'expected_random_candidate_error':float(expected_error),
            'expected_random_mae_exact':exact(expected_mae),'expected_random_candidate_error_exact':exact(expected_error),
            'keyed_errors_agree':sum(x!=y and x==z for x,y,z in zip(a,gold,b)),
            'keyed_errors_disagree':sum(x!=y and x!=z for x,y,z in zip(a,gold,b))},routed

def analyze(base,out):
    if out.exists():raise FileExistsError('Use a fresh output file')
    result={'scope':'Post-hoc cached-trace ablation, conditional on observed per-pool budgets; no new inference or latency claim.',
            'random_reference':'Exact expectation over all uniform k-subsets, with the same saved full-pointwise labels on both plans.',
            'numpy':np.__version__,'bootstrap_draws':4000,'bootstrap_seed_order':[20261103+i for i in range(4)],
            'source_sha256':{},'settings':{}}
    for idx,(name,sel,run,prior,n) in enumerate(SETTINGS):
        for rel in [sel+'/inputs.jsonl',prior]+[run+'/'+f for f in FILES]:
            result['source_sha256'][rel]=hashlib.sha256((base/rel).read_bytes()).hexdigest()
        rows,pred=vectors(base,sel,run);b=keyed_pools(rows,read(base/run/'stable_reverse_responses.jsonl'))
        assert len(rows)==n
        stats=[];routed=[]
        for row,a,rev,h in zip(rows,pred['keyed'],b,pred['pointwise']):
            s,p=pool_stat(row,a,rev,h);stats.append(s);routed.append(p)
        old=json.loads((base/prior).read_text());assert old['common_complete_pools']==n
        for current,key in ((pred['keyed'],'original'),(pred['pointwise'],'pointwise'),(routed,'full_trace_fallback'),(pred['selective'],'actual_fallback')):
            observed=summarize(rows,current)
            for field in ('tp','fp','fn','tn','ratio_mae','f1'):assert abs(observed[field]-old['quality_common'][key][field])<1e-12
        means={key:math.fsum(x[key] for x in stats)/n for key in ('routed_mae','expected_random_mae','routed_candidate_error','expected_random_candidate_error')}
        draw=np.random.default_rng(20261103+idx).integers(0,n,size=(4000,n));comparisons={}
        for label,left,right in (('ratio_mae','routed_mae','expected_random_mae'),('candidate_error','routed_candidate_error','expected_random_candidate_error')):
            diff=np.array([x[left]-x[right] for x in stats]);boot=diff[draw].mean(axis=1)
            comparisons[label]={'difference':float(diff.mean()),'unadjusted_query_interval':np.quantile(boot,[.025,.975]).tolist()}
        total=sum(x['n'] for x in stats);k=sum(x['k'] for x in stats)
        ea=sum(x['keyed_errors_agree'] for x in stats);ed=sum(x['keyed_errors_disagree'] for x in stats)
        result['settings'][name]={'pools':n,'candidates':total,'pointwise_label_budget':k,'budget_fraction':k/total,
            'means':means,'routed_minus_expected_random':comparisons,
            'keyed_error_rate_agree':ea/(total-k) if total>k else None,'keyed_error_rate_disagree':ed/k if k else None,
            'keyed_errors_agree':ea,'keyed_errors_disagree':ed,'per_pool':stats}
    out.write_text(json.dumps(result,indent=2)+'\n')
    print('All four fixed settings analyzed; original, pointwise, cached-routed and actual selective metrics match prior results.')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--artifact',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();analyze(a.artifact,a.out)
