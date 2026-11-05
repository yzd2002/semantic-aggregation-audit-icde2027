#!/usr/bin/env python3
"""Planned descriptive paired comparison of exposed ESCI model runs."""
import argparse,hashlib,json
from fractions import Fraction
from pathlib import Path
import numpy as np
from analyze_esci_selective_holdout import lines,rebuild

MODES=('original','actual_fallback','pointwise')
NEEDED=('original','stable_reverse','replay','actual_fallback','pointwise')

def signed_errors(rows,vectors,indices):
    values=[]
    for i in indices:
        truth=[x['gold'] for x in rows[i]['items']]
        pred=vectors[i]
        if not truth or pred is None or len(pred)!=len(truth) or any(x not in ('Y','N') for x in pred):
            raise ValueError('Invalid or unaligned candidate vector')
        values.append(float(Fraction(pred.count('Y')-truth.count('Y'),len(truth))))
    return np.array(values,dtype=float)

def paired(rows,older,newer,resamples=4000,seed=20271031):
    if len(rows)!=len(older['original']) or len(rows)!=len(newer['original']):raise ValueError('Pool count mismatch')
    for modes in (older,newer):
        if any(k not in modes or len(modes[k])!=len(rows) for k in NEEDED):raise ValueError('Missing model modes')
    common=[i for i in range(len(rows)) if all(modes[k][i] is not None for modes in (older,newer) for k in NEEDED)]
    result={'attempted':len(rows),'common_pools':len(common),'common_query_ids':[rows[i]['query_id'] for i in common],
            'resamples':resamples,'seed':seed,'comparisons':{},'all_y_mae':None}
    if not common:return result
    rng=np.random.default_rng(seed)
    sample=rng.integers(0,len(common),size=(resamples,len(common)))
    for mode in MODES:
        old_signed=signed_errors(rows,older[mode],common);new_signed=signed_errors(rows,newer[mode],common)
        old=np.abs(old_signed);new=np.abs(new_signed);delta=new-old
        draws=delta[sample].mean(axis=1)
        result['comparisons'][mode]={'older_4b_mae':float(old.mean()),'new_7b_mae':float(new.mean()),
              'difference_7b_minus_4b':float(delta.mean()),'paired_query_95_percentile_ci':[float(x) for x in np.quantile(draws,[.025,.975])],
              'older_signed_ratio_bias':float(old_signed.mean()),'new_signed_ratio_bias':float(new_signed.mean())}
    result['all_y_mae']=float(np.mean([sum(x['gold']=='N' for x in rows[i]['items'])/len(rows[i]['items']) for i in common]))
    return result

def read_vectors(run,rows):
    summary=json.loads((run/'summary.json').read_text(encoding='utf-8'))
    if not summary.get('completed'):raise ValueError('Run did not complete; inspect preserved output')
    events={mode:lines(run/(mode+'_responses.jsonl')) for mode in ('original','stable_reverse','replay','pointwise','selective_pointwise')}
    pred,need,matched,changed=rebuild(rows,events)
    return pred,summary

def main():
    p=argparse.ArgumentParser();p.add_argument('--selection',type=Path,required=True);p.add_argument('--older-run',type=Path,required=True);p.add_argument('--newer-run',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    rows=lines(a.selection/'inputs.jsonl');old,old_summary=read_vectors(a.older_run,rows);new,new_summary=read_vectors(a.newer_run,rows)
    digest=hashlib.sha256((a.selection/'inputs.jsonl').read_bytes()).hexdigest()
    if any(x['selection_inputs_sha256']!=digest for x in [old_summary,new_summary]):raise ValueError('Selections differ')
    if old_summary['model']!='Qwen3-4B-Instruct-2507' or new_summary['model']!='Qwen2.5-7B-Instruct':raise ValueError('Unexpected model')
    result=paired(rows,old,new);result.update({'selection_sha256':digest,'older_model':old_summary['model'],'newer_model':new_summary['model'],
        'interpretation':'Post-hoc sensitivity on an exposed sample; differences are not causal model-size effects.'})
    a.out.mkdir(parents=True,exist_ok=False)
    (a.out/'summary.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print('Paired model sensitivity complete; common pools:',result['common_pools'],flush=True)
if __name__=='__main__':main()
