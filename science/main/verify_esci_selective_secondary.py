#!/usr/bin/env python3
"""Post-hoc independent check of secondary metrics and saved request statistics."""
import argparse
import collections
import json
from pathlib import Path

import numpy as np
from verify_esci_selective_holdout import close, keyedparse, lines, one, quality, specparse


def vectorize(rows, records):
    cursor, output = 0, []
    for row in rows:
        values = [None] * len(row['items'])
        for start in range(0, len(values), 8):
            rec = records[cursor]; cursor += 1
            n = min(8, len(values)-start)
            if (rec['query_id'], rec['start'], rec['size']) != (row['query_id'], start, n):
                raise ValueError('Keyed vector association differs.')
            parsed = keyedparse(rec['final'], n)
            if parsed is not None:
                values[start:start+n] = parsed
        output.append(values if None not in values else None)
    if cursor != len(records):
        raise ValueError('Extra keyed records.')
    return output


def paired_independent(rows, left, right, common, seed=20271202):
    tuples = []
    for i in common:
        gold = [x['gold'] for x in rows[i]['items']]
        def components(pred):
            c = collections.Counter(zip(gold, pred))
            return (c['Y','Y'], c['N','Y'], c['Y','N'], abs(gold.count('Y')-pred.count('Y'))/len(gold))
        tuples.append(components(left[i]) + components(right[i]))
    values = np.asarray(tuples, dtype=float)
    draw = np.random.default_rng(seed).integers(0, len(common), size=(4000, len(common)))
    sampled = values[draw]
    totals = sampled.sum(axis=1)
    def f1(i):
        numerator = 2*totals[:, i]
        denominator = numerator + totals[:, i+1] + totals[:, i+2]
        return np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator != 0)
    diff_f1 = f1(4)-f1(0)
    diff_mae = sampled[:, :, 7].mean(axis=1)-sampled[:, :, 3].mean(axis=1)
    l = quality(rows, left, common); r = quality(rows, right, common)
    return {'f1_difference': r['f1']-l['f1'], 'f1_difference_ci': np.quantile(diff_f1,[.025,.975]).tolist(),
            'ratio_mae_difference': r['ratio_mae']-l['ratio_mae'],
            'ratio_mae_difference_ci': np.quantile(diff_mae,[.025,.975]).tolist()}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--selection',type=Path,required=True)
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--reference-analysis',type=Path,required=True)
    p.add_argument('--qwen-run',type=Path,required=True)
    p.add_argument('--qwen-analysis',type=Path,required=True)
    p.add_argument('--smollm-run',type=Path,required=True)
    p.add_argument('--smollm-analysis',type=Path,required=True)
    a=p.parse_args()
    rows=lines(a.selection/'inputs.jsonl')
    ref=lines(a.reference/'predictions.jsonl')
    secondary=json.loads((a.reference_analysis/'summary.json').read_text())
    if len(rows)!=250 or len(ref)!=len(rows) or any(x['query_id']!=r['query_id'] or len(x['pred'])!=len(r['items']) for x,r in zip(ref,rows)):
        raise ValueError('Reference alignment differs.')
    reference=[x['pred'] for x in ref]
    if not close(quality(rows,reference,list(range(len(rows)))),secondary['all_250_supervised']):
        raise ValueError('All-pool supervised quality differs.')
    for model,run,analysis_path in (('qwen',a.qwen_run,a.qwen_analysis),('smollm',a.smollm_run,a.smollm_analysis)):
        summary=json.loads((run/'summary.json').read_text())
        analysis=json.loads((analysis_path/'summary.json').read_text())
        modes=('specifications','original','stable_reverse','selective_pointwise','replay','pointwise')
        events={m:lines(run/(m+'_responses.jsonl')) for m in modes}
        for mode in modes:
            rec=events[mode]
            def parse(record,field):
                value=record[field]
                return specparse(value) if mode=='specifications' else keyedparse(value,record['size']) if mode in ('original','stable_reverse','replay') else one(value)
            expected={'valid_first':sum(parse(x,'first') is not None for x in rec),
                      'valid_final':sum(parse(x,'final') is not None for x in rec),
                      'finish_reasons':dict(collections.Counter(x['first_finish_reason'] for x in rec))}
            if any(v!=summary['modes'][mode][k] for k,v in expected.items()):
                raise ValueError('Saved format or finish-reason totals differ: '+mode)
        vectors={m:vectorize(rows,events[m]) for m in ('original','stable_reverse','replay')}
        points={(x['query_id'],x['start']):x for x in events['pointwise']}
        selected=events['selective_pointwise']
        matched=changed=0
        for s in selected:
            f=points[(s['query_id'],s['start'])]
            if s['prompt_sha256']!=f['prompt_sha256']:
                raise ValueError('Pointwise prompt association differs.')
            if one(s['final']) is not None and one(f['final']) is not None:
                matched+=1;changed+=s['final']!=f['final']
        if (matched!=analysis['selective']['matched_selective_and_full_pointwise'] or
                changed!=analysis['selective']['changed_selective_vs_full_pointwise']):
            raise ValueError('Selective/full repeat comparison differs.')
        outputs=lines(run/'selective_outputs.jsonl')
        vectors['actual_fallback']=[x['pred'] for x in outputs]
        vectors['pointwise']=[]
        for r in rows:
            v=[one(points[(r['query_id'],i)]['final']) for i in range(len(r['items']))]
            vectors['pointwise'].append(v if None not in v else None)
        vectors['supervised']=reference
        ids=set(analysis['common_query_ids'])
        common=[i for i,r in enumerate(rows) if r['query_id'] in ids]
        if len(common)!=analysis['common_complete_pools']:
            raise ValueError('Common set mismatch.')
        section=secondary['models'][model]
        if not close(quality(rows,reference,common),section['supervised_quality_common']):
            raise ValueError('Common-set supervised quality differs.')
        for mode in ('stable_reverse','replay'):
            n=sum(sum(x!=y for x,y in zip(vectors['original'][i],vectors[mode][i])) for i in common)
            if n!=analysis['keyed_'+mode+'_changed_candidates']:
                raise ValueError('Keyed comparison differs.')
        for left in ('original','actual_fallback','pointwise'):
            reconstructed=paired_independent(rows,vectors[left],reference,common)
            saved=section['secondary_comparisons']['supervised_minus_'+left]
            if any(not close(v,saved[k]) for k,v in reconstructed.items()):
                raise ValueError('Secondary paired bootstrap differs.')
    print('Format counts, reversal/replay and selective-repeat changes, supervised metrics and all secondary paired bootstraps independently verified.')


if __name__=='__main__':
    main()
