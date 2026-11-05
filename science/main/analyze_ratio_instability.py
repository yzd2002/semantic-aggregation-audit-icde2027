#!/usr/bin/env python3
"""Compute label-free ratio floors from saved WANDS holdout output chunks."""
import argparse
import collections
import json
import math
import re
from fractions import Fraction
from pathlib import Path

from ratio_disagreement_bounds import macro_ratio_mae_floor, pool_ratio_error_floor


def lines(path):
    with path.open(encoding='utf8') as f:
        return [json.loads(line) for line in f]


def vector(record, n):
    result = [None] * n
    for chunk in record['raw']:
        text, indices = chunk['final'], chunk['original_indices_in_prompt_order']
        if not isinstance(text, str) or not re.fullmatch('[YN]{8}', text):
            return None
        if chunk['size'] != 8 or len(indices) != 8:
            raise ValueError('Invalid saved chunk size or mapping.')
        for index, value in zip(indices, text):
            if type(index) is not int or not 0 <= index < n or result[index] is not None:
                raise ValueError('Invalid or duplicate candidate mapping.')
            result[index] = value
    if None in result:
        return None
    if record['pred'] != result:
        raise ValueError('Saved parsed labels disagree with raw chunks.')
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--selection', type=Path, required=True)
    p.add_argument('--prior', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        raise FileExistsError('Choose a new output directory.')
    rows = lines(a.selection / 'inputs.jsonl')
    left = lines(a.prior / 'replay_outputs.jsonl')
    right = lines(a.prior / 'stable_reverse_outputs.jsonl')
    prior = json.loads((a.prior / 'paired_analysis.json').read_text())
    if len(rows) != len(left) or len(rows) != len(right):
        raise ValueError('Selection and output counts differ.')
    records, complete = [], {'replay': 0, 'stable_reverse': 0}
    for row, aa, bb in zip(rows, left, right):
        qid, n = row['query_id'], len(row['items'])
        if aa['query_id'] != qid or bb['query_id'] != qid or n != 32:
            raise ValueError('Unexpected query alignment or pool size.')
        x, y = vector(aa, n), vector(bb, n)
        complete['replay'] += x is not None
        complete['stable_reverse'] += y is not None
        if x is None or y is None:
            continue
        pa, pb = x.count('Y'), y.count('Y')
        records.append({'query_id': qid, 'n': n, 'replay_positives': pa,
            'reverse_positives': pb, 'positive_count_gap': abs(pa - pb),
            'candidate_disagreements': sum(u != v for u, v in zip(x, y)),
            'pool_worst_ratio_error_floor': str(pool_ratio_error_floor(n, pa, pb))})
    ids = [r['query_id'] for r in records]
    if ids != prior['common_query_ids']:
        raise ValueError('Jointly complete queries differ from previous analysis.')
    if not records:
        raise ValueError('No jointly complete pools.')
    q, n = len(records), 32
    total_gap = sum(r['positive_count_gap'] for r in records)
    differences = sum(r['candidate_disagreements'] for r in records)
    comparison = prior['comparisons']['reverse_stable_ids']
    assert comparison['candidate_count'] == q * n and comparison['changed_candidate_count'] == differences
    assert comparison['pools_with_changed_ratio'] == sum(r['positive_count_gap'] > 0 for r in records)
    assert math.isclose(comparison['mean_absolute_pool_ratio_change'], total_gap / (q * n), abs_tol=1e-12)
    pairs = [(r['replay_positives'], r['reverse_positives']) for r in records]
    floor = macro_ratio_mae_floor(n, pairs)
    mean_pool_floor = sum((pool_ratio_error_floor(n, x, y) for x, y in pairs), Fraction()) / q
    summary = {'protocol': 'PROTOCOL_ratio_instability_bounds.md', 'attempted_pools': len(rows),
        'completion': complete, 'common_complete_pools': q, 'candidates_per_pool': n,
        'candidate_disagreements': differences, 'pools_with_changed_ratio': sum(r['positive_count_gap'] > 0 for r in records),
        'positive_count_gap_total': total_gap,
        'positive_count_gap_histogram': dict(sorted(collections.Counter(str(r['positive_count_gap']) for r in records).items(), key=lambda x: int(x[0]))),
        'mean_absolute_ratio_change': str(Fraction(total_gap, q * n)),
        'sharp_worse_macro_mae_floor': str(floor), 'sharp_worse_macro_mae_floor_float': float(floor),
        'mean_of_per_pool_worse_error_floors': str(mean_pool_floor),
        'interpretation': 'Fixed common-complete, previously observed pools; reference-free deterministic floors, not confidence intervals.'}
    a.out.mkdir(parents=True, exist_ok=False)
    with (a.out / 'per_pool.jsonl').open('w', encoding='utf8') as f:
        for row in records:
            f.write(json.dumps(row) + '\n')
    (a.out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print('Pools:', q, 'candidate disagreements:', differences, 'nonzero ratio gaps:', summary['pools_with_changed_ratio'])
    print('Total positive-count gap:', total_gap, 'sharp worse macro-MAE floor:', floor, '=', float(floor))


if __name__ == '__main__':
    main()
