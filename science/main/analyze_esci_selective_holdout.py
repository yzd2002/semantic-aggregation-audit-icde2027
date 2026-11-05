#!/usr/bin/env python3
"""Analyze actual selective fallback on the frozen variable-size ESCI pools."""
import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
from analyze_order_fallback_validation import paired
from order_fallback import combine
from run_compact_prompt_ablation import metrics
from run_keyed_order_audit import parse_keyed
from select_esci_selective_holdout import PROTOCOL


def lines(path):
    with path.open(encoding='utf8') as f:
        return [json.loads(line) for line in f]


def one(text):
    return text if text in ('Y', 'N') else None


def rebuild(rows, events):
    """Return plan vectors and request associations, preserving failed pools."""
    chunks = [(row['query_id'], start, min(8, len(row['items']) - start))
              for row in rows for start in range(0, len(row['items']), 8)]
    keyed = {}
    for mode in ('original', 'stable_reverse', 'replay'):
        saved = events[mode]
        if [(x['query_id'], x['start'], x['size']) for x in saved] != chunks:
            raise ValueError('Keyed records do not match the variable-size chunk order.')
        keyed[mode] = {(q, start): parse_keyed(x['final'], list(range(1, n+1)))
                       for (q, start, n), x in zip(chunks, saved)}
    pred = {mode: [] for mode in keyed}
    need = []
    for row in rows:
        vectors = {mode: [None] * len(row['items']) for mode in keyed}
        for start in range(0, len(row['items']), 8):
            pair = [keyed[mode][(row['query_id'], start)] for mode in ('original', 'stable_reverse')]
            if all(x is not None for x in pair):
                need.extend((row['query_id'], start + i - 1) for i in pair[0] if pair[0][i] != pair[1][i])
            for mode in keyed:
                labels = keyed[mode][(row['query_id'], start)]
                if labels is not None:
                    for local, label in labels.items():
                        vectors[mode][start + local - 1] = label
        for mode, values in vectors.items():
            pred[mode].append(values if None not in values else None)
    wanted_full = [(r['query_id'], i) for r in rows for i in range(len(r['items']))]
    full = events['pointwise']
    selected = events['selective_pointwise']
    if [(x['query_id'], x['start']) for x in full] != wanted_full:
        raise ValueError('Full pointwise records do not match candidate order.')
    if [(x['query_id'], x['start']) for x in selected] != need:
        raise ValueError('Actual selective calls do not match valid keyed disagreements.')
    full_map = {(x['query_id'], x['start']): x for x in full}
    selected_map = {(x['query_id'], x['start']): x for x in selected}
    full_vectors, sparse = [], []
    for row in rows:
        full_vectors.append([one(full_map[(row['query_id'], i)]['final']) for i in range(len(row['items']))])
        sparse.append([one(selected_map[(row['query_id'], i)]['final']) if (row['query_id'], i) in selected_map else None
                       for i in range(len(row['items']))])
    pred['pointwise'] = [v if None not in v else None for v in full_vectors]
    pred['actual_fallback'] = [combine(a, b, c) for a, b, c in zip(pred['original'], pred['stable_reverse'], sparse)]
    pred['full_trace_fallback'] = [combine(a, b, c) for a, b, c in zip(pred['original'], pred['stable_reverse'], full_vectors)]
    matched = changed = 0
    for key in need:
        s, f = selected_map[key], full_map[key]
        if s['prompt_sha256'] != f['prompt_sha256']:
            raise ValueError('Selective and later full pointwise prompts differ.')
        if one(s['final']) is not None and one(f['final']) is not None:
            matched += 1
            changed += s['final'] != f['final']
    return pred, need, matched, changed


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--selection', type=Path, required=True)
    ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    if a.out.exists():
        raise FileExistsError('Use a new analysis output directory.')
    rows = lines(a.selection / 'inputs.jsonl')
    select = json.loads((a.selection / 'selection.json').read_text(encoding='utf8'))
    summary = json.loads((a.run / 'summary.json').read_text(encoding='utf8'))
    sanity = json.loads((a.run / 'guidance_sanity.json').read_text(encoding='utf8'))
    selection_sha = hashlib.sha256((a.selection / 'inputs.jsonl').read_bytes()).hexdigest()
    modes = ('original', 'stable_reverse', 'selective_pointwise', 'replay', 'pointwise')
    if (len(rows) != 250 or select['protocol'] != PROTOCOL or summary['protocol'] != PROTOCOL or
            summary['selection_inputs_sha256'] != selection_sha or summary['mode_order'] != list(modes) or
            not summary.get('completed') or sanity['unconstrained_valid_keyed'] or sanity['unconstrained_valid_point'] or
            not sanity['guided_keyed_valid'] or not sanity['guided_point_valid']):
        raise ValueError('Unexpected source, protocol, execution or constraint control.')
    events = {mode: lines(a.run / (mode + '_responses.jsonl')) for mode in modes}
    pred, need, matched, changed = rebuild(rows, events)
    saved = lines(a.run / 'selective_outputs.jsonl')
    if len(saved) != len(rows) or any(x['query_id'] != row['query_id'] or x['pred'] != y
                                      for x, row, y in zip(saved, rows, pred['actual_fallback'])):
        raise ValueError('Saved selective results differ from independent reconstruction.')
    pred['always_Y'] = [['Y'] * len(r['items']) for r in rows]
    pred['always_N'] = [['N'] * len(r['items']) for r in rows]
    pred['lexical'] = []
    for row in rows:
        words = set(re.findall(r'\w+', row['query'].casefold()))
        pred['lexical'].append(['Y' if words and words.issubset(set(re.findall(r'\w+', x['text'].casefold()))) else 'N'
                                for x in row['items']])
    common = [i for i in range(len(rows)) if all(pred[k][i] is not None for k in
              ('original', 'stable_reverse', 'replay', 'pointwise', 'actual_fallback', 'full_trace_fallback'))]
    from run_experiment import parse_spec
    specs = lines(a.run / 'specifications_responses.jsonl')
    if len(specs) != len(rows):
        raise ValueError('Specification count differs.')
    result = {'protocol': PROTOCOL, 'model': summary['model'], 'selection_inputs_sha256': selection_sha,
              'attempted_pools': len(rows), 'candidate_count': sum(len(r['items']) for r in rows),
              'completion': {k: sum(v is not None for v in values) for k, values in pred.items()},
              'common_complete_pools': len(common), 'common_query_ids': [rows[i]['query_id'] for i in common],
              'specification_fallback_pools': sum(parse_spec(x['final']) is None for x in specs),
              'selective': {'disagreeing_candidates': len(need), 'actual_initial_calls': summary['modes']['selective_pointwise']['initial_calls'],
                            'actual_retry_calls': summary['modes']['selective_pointwise']['retry_calls'],
                            'actual_complete_pools': summary['selective_pipeline']['complete_pools'],
                            'actual_total_calls_with_spec_and_two_keyed_passes': summary['selective_pipeline']['initial_calls'] + summary['selective_pipeline']['retry_calls'],
                            'actual_logical_tokens': summary['selective_pipeline']['logical_tokens'],
                            'offline_batch_wall_seconds': summary['selective_pipeline']['offline_batch_wall_seconds'],
                            'matched_selective_and_full_pointwise': matched, 'changed_selective_vs_full_pointwise': changed},
              'resamples': 4000, 'seed': 20271202, 'quality_common': {}, 'comparisons': {},
              'primary_upper_endpoint_below_zero': None}
    if common:
        draw = np.random.default_rng(20271202).integers(0, len(common), size=(4000, len(common)))
        result['quality_common'] = {k: metrics(rows, v, common) for k, v in pred.items()}
        for right in ('actual_fallback', 'full_trace_fallback', 'pointwise', 'lexical', 'always_N', 'always_Y'):
            result['comparisons'][right + '_minus_original'] = paired(rows, pred, common, 'original', right, draw)
        result['primary_upper_endpoint_below_zero'] = result['comparisons']['actual_fallback_minus_original']['ratio_mae_difference_ci'][1] < 0
        for mode in ('stable_reverse', 'replay'):
            result['keyed_' + mode + '_changed_candidates'] = sum(sum(x != y for x, y in zip(pred['original'][i], pred[mode][i])) for i in common)
        result['common_candidates'] = sum(len(rows[i]['items']) for i in common)
    a.out.mkdir(parents=True, exist_ok=False)
    (a.out / 'summary.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf8')
    print('Analyzed', len(common), 'common-complete pools; primary MAE difference:',
          result['comparisons'].get('actual_fallback_minus_original', {}).get('ratio_mae_difference'))


if __name__ == '__main__':
    main()
