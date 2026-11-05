#!/usr/bin/env python3
"""Independently reconstruct raw model routes and audit mapping sensitivity."""
import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sys
import numpy as np


def readlines(path):
    with path.open(encoding='utf8') as f:
        return [json.loads(line) for line in f]


def hash_file(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def close(actual, expected):
    if abs(actual-expected)>1e-12:
        raise ValueError(f'Metric mismatch: {actual} vs {expected}')


def independently_score(rows,vectors,positive):
    counts={'tp':0,'fp':0,'tn':0,'fn':0}; errs=[]
    for row,pred in zip(rows,vectors,strict=True):
        items=row['items']
        if len(items)!=len(pred) or any(v not in ('Y','N') for v in pred):
            raise ValueError('Invalid complete prediction vector')
        positives=0; predicted=0
        for item, label in zip(items,pred,strict=True):
            actual=item['grade'] in positive; decision=label=='Y'
            positives+=actual; predicted+=decision
            key='tp' if actual and decision else 'fn' if actual else 'fp' if decision else 'tn'
            counts[key]+=1
        errs.append(Fraction(abs(predicted-positives),len(items)))
    tp,fp,tn,fn=(counts[x] for x in ('tp','fp','tn','fn'))
    counts.update({'f1':float(Fraction(2*tp,2*tp+fp+fn)) if 2*tp+fp+fn else 0.,
                   'balanced_accuracy':float((Fraction(tp,tp+fn)+Fraction(tn,tn+fp))/2),
                   'ratio_mae':float(sum(errs,Fraction())/len(errs))})
    return counts, np.array([float(x) for x in errs])


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',required=True,type=Path);ap.add_argument('--analysis-dir',required=True,type=Path)
    a=ap.parse_args(); root=a.root; here=a.analysis_dir
    expected=json.loads((here/'result.json').read_text())
    frozen=json.loads((here/'PRE_ANALYSIS_SHA256.json').read_text())
    for name,value in frozen.items():
        path=here/name if name in ('PROTOCOL_label_mapping_sensitivity.md','analyze_label_mapping.py','test_label_mapping.py') else root/name
        if hash_file(path)!=value:raise ValueError('Pre-analysis input or code hash changed: '+name)
    for name,value in expected['input_sha256'].items():
        if hash_file(root/name)!=value:raise ValueError('Result input hash mismatch: '+name)
    sys.path.insert(0,str(root))
    from analyze_esci_selective_holdout import rebuild
    rows=readlines(root/'esci_selective_holdout_selection/inputs.jsonl')
    if len(rows)!=250 or sum(len(r['items']) for r in rows)!=4428:raise ValueError('Unexpected data size')
    supervised=readlines(root/'run_esci_selective_supervised_reference/predictions.jsonl')
    if [x['query_id'] for x in supervised]!=[r['query_id'] for r in rows]:raise ValueError('Supervised IDs differ')
    modes=('original','stable_reverse','selective_pointwise','replay','pointwise')
    draws=np.random.default_rng(expected['seed']).integers(0,len(rows),size=(expected['resamples'],len(rows)))
    for model in ('qwen','smollm'):
        base=root/f'run_esci_selective_holdout_{model}'
        events={mode:readlines(base/(mode+'_responses.jsonl')) for mode in modes}
        rebuilt,_,_,_=rebuild(rows,events)
        saved=readlines(base/'selective_outputs.jsonl')
        if len(saved)!=len(rows) or any(x['query_id']!=r['query_id'] or x['pred']!=v
                                       for x,r,v in zip(saved,rows,rebuilt['actual_fallback'],strict=True)):
            raise ValueError('Raw events do not reconstruct selective output')
        plans={'original':rebuilt['original'],'selective':rebuilt['actual_fallback'],
               'pointwise':rebuilt['pointwise'],'supervised':[x['pred'] for x in supervised],
               'all_y':[['Y']*len(r['items']) for r in rows],
               'all_n':[['N']*len(r['items']) for r in rows]}
        errors={}
        for mapping,positive in [('E_or_S',{'E','S'}),('E_only',{'E'})]:
            errors[mapping]={}
            for plan,pred in plans.items():
                if any(p is None for p in pred):raise ValueError('An expected pool is incomplete')
                metrics,error=independently_score(rows,pred,positive);errors[mapping][plan]=error
                for key,value in metrics.items():close(value,expected['models'][model]['quality'][mapping][plan][key])
        effects={key:values['selective']-values['original'] for key,values in errors.items()}
        effects['E_only_minus_E_or_S']=effects['E_only']-effects['E_or_S']
        for label,values in effects.items():
            saved=expected['models'][model]['selective_minus_keyed'][label]
            close(float(np.mean(values)),saved['mean'])
            # Rebuild each resample's paired mean independently, retaining the same draws.
            sample_means=np.array([sum(values[i] for i in sample)/len(sample) for sample in draws])
            for x,y in zip(np.quantile(sample_means,[.025,.975]),saved['ci95'],strict=True):close(x,y)
    print('Verified frozen hashes, two raw selective routes, both label mappings, all metrics and paired intervals.')


if __name__=='__main__':main()
