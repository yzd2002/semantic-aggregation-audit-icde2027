#!/usr/bin/env python3
"""Compute the frozen primary ratio-MAE contrast and secondary diagnostics."""
import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
from order_fallback import combine
from run_compact_prompt_ablation import metrics
from run_keyed_order_audit import parse_keyed
from run_order_fallback_validation import parse_one
from select_order_fallback_validation import N, PROTOCOL, lines


def keyed_vectors(rows, records):
    if len(records) != 4 * len(rows):
        raise ValueError('Unexpected keyed response count.')
    output = []
    for q, row in enumerate(rows):
        v = [None] * 32
        for c in range(4):
            rec = records[q * 4 + c]
            if rec['query_id'] != row['query_id'] or rec['start'] != 8*c:
                raise ValueError('Keyed response ordering mismatch.')
            parsed = parse_keyed(rec['final'], list(range(1, 9)))
            if parsed is not None:
                for k, label in parsed.items():
                    v[c * 8 + k - 1] = label
        output.append(v if None not in v else None)
    return output


def paired(rows, preds, common, left, right, draw):
    points = []
    for i in common:
        gold = [x['gold'] for x in rows[i]['items']]
        a, b = preds[left][i], preds[right][i]
        def stats(v):
            tp = sum(x == y == 'Y' for x, y in zip(v, gold))
            fp = sum(x == 'Y' and y == 'N' for x, y in zip(v, gold))
            fn = sum(x == 'N' and y == 'Y' for x, y in zip(v, gold))
            return tp, fp, fn, abs(v.count('Y') - gold.count('Y')) / len(gold)
        points.append(stats(a) + stats(b))
    values = np.array(points, dtype=float)
    if not len(common):
        return None
    sampled = values[draw]
    summed = sampled.sum(axis=1)
    def f1(k):
        numerator = 2 * summed[:, k]
        denominator = numerator + summed[:, k + 1] + summed[:, k + 2]
        return np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator != 0)
    difference_f1 = f1(4) - f1(0)
    difference_mae = sampled[:, :, 7].mean(axis=1) - sampled[:, :, 3].mean(axis=1)
    a = metrics(rows, preds[left], common)
    b = metrics(rows, preds[right], common)
    return {'left': left, 'right': right, 'f1_difference': b['f1'] - a['f1'],
            'f1_difference_ci': [float(x) for x in np.quantile(difference_f1, [.025, .975])],
            'ratio_mae_difference': b['ratio_mae'] - a['ratio_mae'],
            'ratio_mae_difference_ci': [float(x) for x in np.quantile(difference_mae, [.025, .975])]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--selection', type=Path, required=True)
    a = ap.parse_args()
    rows = lines(a.selection / 'inputs.jsonl')
    selection = json.loads((a.selection / 'selection.json').read_text())
    summary = json.loads((a.run / 'summary.json').read_text())
    sanity = json.loads((a.run / 'guidance_sanity.json').read_text())
    source_sha = hashlib.sha256((a.selection / 'inputs.jsonl').read_bytes()).hexdigest()
    modes = ['original', 'replay', 'stable_reverse', 'pointwise', 'pointwise_replay_disagreements']
    if (len(rows) != N or selection['protocol'] != PROTOCOL or summary['protocol'] != PROTOCOL
            or summary['selection_inputs_sha256'] != source_sha or summary['mode_order'] != modes
            or summary['runtime']['engine'] != 'V1 (VLLM_USE_V1=1)'
            or sanity['unconstrained_valid_json'] or sanity['unconstrained_valid_point']
            or not sanity['guided_json_valid'] or not sanity['guided_point_valid']):
        raise ValueError('Unexpected protocol, source or constraint control.')
    records = {mode: lines(a.run / (mode + '_responses.jsonl')) for mode in modes}
    preds = {name: keyed_vectors(rows, records[name]) for name in modes[:3]}
    point = records['pointwise']
    if len(point) != 32 * len(rows):
        raise ValueError('Unexpected pointwise response count.')
    point_partial = []
    for q, row in enumerate(rows):
        block = point[q*32:(q+1)*32]
        if any(rec['query_id'] != row['query_id'] or rec['start'] != i for i, rec in enumerate(block)):
            raise ValueError('Pointwise response ordering mismatch.')
        point_partial.append([parse_one(rec['final']) for rec in block])
    preds['pointwise'] = [v if None not in v else None for v in point_partial]
    preds['always_Y'] = [['Y'] * 32 for row in rows]
    preds['always_N'] = [['N'] * 32 for row in rows]
    preds['lexical'] = []
    for row in rows:
        terms = set(re.findall(r'\w+', row['query'].casefold()))
        preds['lexical'].append(['Y' if terms and terms.issubset(set(re.findall(r'\w+', x['text'].casefold()))) else 'N'
                                 for x in row['items']])
    preds['pointwise_fallback'] = [combine(x, y, h) for x, y, h in zip(preds['original'], preds['stable_reverse'], point_partial)]
    preds['lexical_fallback'] = [combine(x, y, h) for x, y, h in zip(preds['original'], preds['stable_reverse'], preds['lexical'])]
    common = [i for i in range(N) if all(preds[name][i] is not None for name in ('original', 'replay', 'stable_reverse', 'pointwise'))]
    need = []
    for q in range(N):
        for c in range(4):
            x = parse_keyed(records['original'][q*4+c]['final'], list(range(1, 9)))
            y = parse_keyed(records['stable_reverse'][q*4+c]['final'], list(range(1, 9)))
            if x is not None and y is not None:
                need.extend((q, c*8+i-1) for i in range(1, 9) if x[i] != y[i])
    all_replay = records['pointwise_replay_disagreements']
    by_key = {(r['query_id'], r['start']): r for r in all_replay}
    if len(by_key) != len(all_replay):
        raise ValueError('Duplicate pointwise replay.')
    changed, matched = 0, 0
    for q, i in need:
        rec = point[q*32+i]
        replay = by_key.get((rows[q]['query_id'], i))
        if parse_one(rec['final']) is not None:
            if replay is None or replay['prompt_sha256'] != rec['prompt_sha256']:
                raise ValueError('Missing or altered pointwise replay.')
            if parse_one(replay['final']) is not None:
                matched += 1
                changed += replay['final'] != rec['final']
    # Every replay must refer to a valid pointwise request for a keyed disagreement.
    point_index = {(r['query_id'], r['start']): r for r in point}
    assert all(k in point_index and v['prompt_sha256'] == point_index[k]['prompt_sha256'] for k, v in by_key.items())
    assert set(by_key) == {(rows[q]['query_id'], i) for q, i in need if parse_one(point[q*32+i]['final']) is not None}
    required = [point[q*32+i] for q, i in need]
    required_retry = sum(r['retry_finish_reason'] is not None for r in required)
    spec = summary['specifications']
    keyed_cost = [summary['modes'][m] for m in ('original', 'stable_reverse')]
    virtual_calls = (spec['initial_calls'] + spec['retry_calls'] + sum(x['initial_calls'] + x['retry_calls'] for x in keyed_cost)
                     + len(required) + required_retry)
    virtual_tokens = {key: spec['logical_tokens'][key] + sum(x['logical_tokens'][key] for x in keyed_cost)
                      + sum(r['first_' + key + '_tokens'] + r['retry_' + key + '_tokens'] for r in required)
                      for key in ('input', 'output')}
    result = {'protocol': PROTOCOL, 'selection_inputs_sha256': source_sha, 'model': summary['model'],
              'attempted_pools': N, 'candidate_count': 32*N, 'resamples': 4000, 'seed': 20271127,
              'grade_counts': selection['grade_counts'], 'specification_fallback_pools': spec['fallback_pools'],
              'completion': {k: sum(v is not None for v in values) for k, values in preds.items()},
              'common_complete_pools': len(common), 'common_query_ids': [rows[i]['query_id'] for i in common],
              'disagreement': {'candidates': len(need), 'valid_pointwise_replays': matched, 'pointwise_replay_changed': changed,
                               'virtual_fallback_calls': len(required), 'virtual_fallback_retries': required_retry,
                               'virtual_total_calls_with_spec_and_two_keyed_passes': virtual_calls,
                               'virtual_logical_tokens': virtual_tokens},
              'quality_common': {}, 'comparisons': {}, 'primary_upper_endpoint_below_zero': None}
    if common:
        draw = np.random.default_rng(20271127).integers(0, len(common), size=(4000, len(common)))
        result['quality_common'] = {k: metrics(rows, v, common) for k, v in preds.items()}
        for right in ('pointwise_fallback', 'lexical_fallback', 'pointwise', 'lexical'):
            result['comparisons'][right + '_minus_original'] = paired(rows, preds, common, 'original', right, draw)
        result['primary_upper_endpoint_below_zero'] = result['comparisons']['pointwise_fallback_minus_original']['ratio_mae_difference_ci'][1] < 0
        result['keyed_replay_changed_candidates'] = sum(sum(x != y for x, y in zip(preds['original'][i], preds['replay'][i])) for i in common)
    (a.run / 'paired_analysis.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
