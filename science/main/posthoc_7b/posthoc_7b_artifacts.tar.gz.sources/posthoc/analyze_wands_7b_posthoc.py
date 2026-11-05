#!/usr/bin/env python3
"""Analyze actual selective execution and its later full-pointwise comparator."""
import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
from analyze_order_fallback_validation import keyed_vectors, paired
from order_fallback import combine
from run_compact_prompt_ablation import metrics
from run_keyed_order_audit import parse_keyed
from run_order_fallback_validation import parse_one
from select_selective_execution import N, PROTOCOL as SELECTION_PROTOCOL, lines
PROTOCOL = 'PROTOCOL_7b_multidomain_posthoc.md'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--selection', type=Path, required=True)
    a = ap.parse_args()
    rows = lines(a.selection / 'inputs.jsonl')
    selection = json.loads((a.selection / 'selection.json').read_text())
    summary = json.loads((a.run / 'summary.json').read_text())
    sanity = json.loads((a.run / 'guidance_sanity.json').read_text())
    sha = hashlib.sha256((a.selection / 'inputs.jsonl').read_bytes()).hexdigest()
    modes = ['original', 'stable_reverse', 'selective_pointwise', 'replay', 'pointwise']
    if (len(rows) != N or selection['protocol'] != SELECTION_PROTOCOL or summary['protocol'] != PROTOCOL
            or summary['selection_inputs_sha256'] != sha or summary['mode_order'] != modes
            or summary['runtime']['engine'] != 'V1 (VLLM_USE_V1=1)'
            or sanity['unconstrained_valid_json'] or sanity['unconstrained_valid_point']
            or not sanity['guided_json_valid'] or not sanity['guided_point_valid']):
        raise ValueError('Unexpected protocol, selection or format control.')
    rec = {mode: lines(a.run / (mode + '_responses.jsonl')) for mode in modes}
    pred = {mode: keyed_vectors(rows, rec[mode]) for mode in ('original', 'stable_reverse', 'replay')}
    if len(rec['pointwise']) != N*32:
        raise ValueError('Unexpected full-pointwise count.')
    full_partial = []
    for q, row in enumerate(rows):
        block = rec['pointwise'][q*32:(q+1)*32]
        if any(x['query_id'] != row['query_id'] or x['start'] != i for i, x in enumerate(block)):
            raise ValueError('Full-pointwise ordering mismatch.')
        full_partial.append([parse_one(x['final']) for x in block])
    pred['pointwise'] = [v if None not in v else None for v in full_partial]
    selective = {(x['query_id'], x['start']): x for x in rec['selective_pointwise']}
    if len(selective) != len(rec['selective_pointwise']):
        raise ValueError('Duplicate selective request.')
    need = []
    for q, row in enumerate(rows):
        for c in range(4):
            x = parse_keyed(rec['original'][q*4+c]['final'], list(range(1, 9)))
            y = parse_keyed(rec['stable_reverse'][q*4+c]['final'], list(range(1, 9)))
            if x is not None and y is not None:
                need.extend((q, c*8+i-1) for i in range(1, 9) if x[i] != y[i])
    expected_keys = [(rows[q]['query_id'], i) for q, i in need]
    if [(x['query_id'], x['start']) for x in rec['selective_pointwise']] != expected_keys:
        raise ValueError('Selective requests do not exactly match valid keyed disagreements.')
    sparse = [[parse_one(selective[(row['query_id'], i)]['final']) if (row['query_id'], i) in selective else None
               for i in range(32)] for row in rows]
    pred['actual_fallback'] = [combine(x, y, h) for x, y, h in zip(pred['original'], pred['stable_reverse'], sparse)]
    pred['full_trace_fallback'] = [combine(x, y, h) for x, y, h in zip(pred['original'], pred['stable_reverse'], full_partial)]
    saved = lines(a.run / 'selective_outputs.jsonl')
    if len(saved) != N or any(row['query_id'] != x['query_id'] or x['pred'] != p
                              for row, x, p in zip(rows, saved, pred['actual_fallback'])):
        raise ValueError('Saved actual fallback vectors differ from recomputed outputs.')
    pred['always_N'] = [['N']*32 for row in rows]
    pred['lexical'] = []
    for row in rows:
        words = set(re.findall(r'\w+', row['query'].casefold()))
        pred['lexical'].append(['Y' if words and words.issubset(set(re.findall(r'\w+', x['text'].casefold()))) else 'N'
                                for x in row['items']])
    common = [i for i in range(N) if all(pred[k][i] is not None for k in
              ('original', 'stable_reverse', 'replay', 'pointwise', 'actual_fallback', 'full_trace_fallback'))]
    matched = changed = 0
    for q, i in need:
        s = selective[(rows[q]['query_id'], i)]; f = rec['pointwise'][q*32+i]
        if s['prompt_sha256'] != f['prompt_sha256']:
            raise ValueError('Selective and full pointwise prompts differ.')
        x, y = parse_one(s['final']), parse_one(f['final'])
        if x is not None and y is not None:
            matched += 1; changed += x != y
    pipeline = summary['selective_pipeline']
    result = {'protocol': PROTOCOL, 'model': summary['model'], 'selection_inputs_sha256': sha,
              'attempted_pools': N, 'candidate_count': N*32, 'grade_counts': selection['grade_counts'],
              'resamples': 4000, 'seed': 20271202,
              'completion': {k: sum(x is not None for x in v) for k, v in pred.items()},
              'common_complete_pools': len(common), 'common_query_ids': [rows[i]['query_id'] for i in common],
              'specification_fallback_pools': summary['specifications']['fallback_pools'],
              'selective': {'disagreeing_candidates': len(need), 'actual_initial_calls': summary['modes']['selective_pointwise']['initial_calls'],
                            'actual_retry_calls': summary['modes']['selective_pointwise']['retry_calls'],
                            'actual_complete_pools': pipeline['complete_pools'],
                            'actual_total_calls_with_spec_and_two_keyed_passes': pipeline['initial_calls']+pipeline['retry_calls'],
                            'actual_logical_tokens': pipeline['logical_tokens'],
                            'offline_batch_wall_seconds': pipeline['offline_batch_wall_seconds'],
                            'matched_selective_and_full_pointwise': matched, 'changed_selective_vs_full_pointwise': changed},
              'quality_common': {}, 'comparisons': {}, 'secondary_no_success_threshold': True}
    if common:
        draw = np.random.default_rng(20271202).integers(0, len(common), size=(4000, len(common)))
        result['quality_common'] = {k: metrics(rows, v, common) for k, v in pred.items()}
        for right in ('actual_fallback', 'full_trace_fallback', 'pointwise', 'lexical', 'always_N'):
            result['comparisons'][right + '_minus_original'] = paired(rows, pred, common, 'original', right, draw)
        result['comparisons']['actual_minus_full_trace'] = paired(rows, pred, common, 'full_trace_fallback', 'actual_fallback', draw)
        result['keyed_replay_changed_candidates'] = sum(sum(x != y for x,y in zip(pred['original'][i], pred['replay'][i])) for i in common)
    (a.run / 'paired_analysis.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
