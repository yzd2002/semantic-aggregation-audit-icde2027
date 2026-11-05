#!/usr/bin/env python3
"""Apply the mixed-size deterministic floor to saved new-ESCI events."""
import argparse,hashlib,json,sys
from fractions import Fraction
from pathlib import Path
from variable_pool_floor import evaluate

MODES=('original','stable_reverse','selective_pointwise','replay','pointwise')
INPUTS=['esci_selective_holdout_selection/inputs.jsonl']+[
    f'run_esci_selective_holdout_{model}/{mode}_responses.jsonl'
    for model in ('qwen','smollm') for mode in MODES]


def lines(path):
    with path.open(encoding='utf8') as f:return [json.loads(x) for x in f]


def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def record(value):
    return {'numerator':value.numerator,'denominator':value.denominator,'decimal':float(value)}


def run(root):
    sys.path.insert(0,str(root))
    from analyze_esci_selective_holdout import rebuild
    rows=lines(root/INPUTS[0])
    if len(rows)!=250 or sum(len(r['items']) for r in rows)!=4428 or len({r['query_id'] for r in rows})!=250:
        raise ValueError('Unexpected source sample')
    if any(x['grade'] not in ('E','S','C','I') or x['gold']!=('Y' if x['grade'] in ('E','S') else 'N')
           for row in rows for x in row['items']):raise ValueError('Unexpected source grades')
    out={'analysis':'post-hoc unequal-pool stability floor',
         'pools':len(rows),'candidates':sum(len(r['items']) for r in rows),
         'input_sha256':{name:sha(root/name) for name in INPUTS},'models':{},
         'limitations':'Finite selected pools; fixed binary reference; variable-pool floor can be non-sharp; witness is synthetic and assumes unrestricted occurrence labels.'}
    for model in ('qwen','smollm'):
        base=root/f'run_esci_selective_holdout_{model}'
        vectors,_,_,_=rebuild(rows,{mode:lines(base/(mode+'_responses.jsonl')) for mode in MODES})
        a,b=vectors['original'],vectors['stable_reverse']
        if any(x is None for x in a+b):raise ValueError('An output pool is incomplete')
        pools=[(len(row['items']),x.count('Y'),y.count('Y')) for row,x,y in zip(rows,a,b,strict=True)]
        result=evaluate(pools);nq=len(rows)
        actual_a=sum((Fraction(abs(x.count('Y')-sum(t['gold']=='Y' for t in row['items'])),len(x))
                      for row,x in zip(rows,a,strict=True)),Fraction())/nq
        actual_b=sum((Fraction(abs(y.count('Y')-sum(t['gold']=='Y' for t in row['items'])),len(y))
                      for row,y in zip(rows,b,strict=True)),Fraction())/nq
        if max(actual_a,actual_b)<result['floor']:raise ValueError('Observed MAE violates deterministic floor')
        out['models'][model]={
          'pool_counts':[{'query_id':row['query_id'],'n':p[0],'original_positive':p[1],
                          'reverse_positive':p[2],'synthetic_reference_positive':r}
                         for row,p,r in zip(rows,pools,result['witness_reference_counts'],strict=True)],
          'pools_with_changed_positive_count':sum(a0!=b0 for _,a0,b0 in pools),
          'minimum_pool_size':min(n for n,_,_ in pools),'maximum_pool_size':max(n for n,_,_ in pools),
          **{k:record(result[k]) for k in ('floor','witness_mae','witness_mae_first','witness_mae_second',
                                          'guaranteed_witness_slack','weighted_count_gap_sum')},
          'observed_ES_positive_mae_original':record(actual_a),
          'observed_ES_positive_mae_reversed':record(actual_b)}
    return out


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args();result=run(args.root)
    with args.out.open('x',encoding='utf8') as f:json.dump(result,f,indent=2);f.write('\n')
    print('Computed exact rational floors and constructive witnesses for two models on',result['pools'],'pools.')


if __name__=='__main__':main()
