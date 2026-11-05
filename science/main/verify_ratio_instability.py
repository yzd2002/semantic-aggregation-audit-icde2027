#!/usr/bin/env python3
"""Independently verify saved ratio floors and construct an attaining witness."""
import argparse
import collections
import json
import re
from fractions import Fraction
from pathlib import Path


def read(path):
    return [json.loads(line) for line in path.read_text(encoding='utf8').splitlines()]


def decode(row, n):
    pairs = []
    for part in row['raw']:
        value = part['final']
        if not isinstance(value, str) or re.fullmatch('[YN]{8}', value) is None:
            return None
        assert part['size'] == 8 and len(part['original_indices_in_prompt_order']) == 8
        pairs.extend(zip(part['original_indices_in_prompt_order'], value))
    positions = [p for p, _ in pairs]
    assert len(positions) == n and set(positions) == set(range(n))
    decoded = [v for _, v in sorted(pairs)]
    assert decoded == row['pred']
    return decoded


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--selection', type=Path, required=True)
    p.add_argument('--prior', type=Path, required=True)
    p.add_argument('--run', type=Path, required=True)
    a = p.parse_args()
    rows = read(a.selection / 'inputs.jsonl')
    left, right = read(a.prior / 'replay_outputs.jsonl'), read(a.prior / 'stable_reverse_outputs.jsonl')
    prior = json.loads((a.prior / 'paired_analysis.json').read_text())
    recorded = read(a.run / 'per_pool.jsonl')
    report = json.loads((a.run / 'summary.json').read_text())
    assert len(rows) == len(left) == len(right) == report['attempted_pools']
    rebuilt, complete = [], collections.Counter()
    for source, l, r in zip(rows, left, right):
        qid = source['query_id']
        n = len(source['items'])
        assert n == 32 and qid == l['query_id'] == r['query_id']
        x, y = decode(l, n), decode(r, n)
        complete['replay'] += x is not None
        complete['stable_reverse'] += y is not None
        if x is None or y is None:
            continue
        ca, cb = x.count('Y'), y.count('Y')
        delta = abs(ca - cb)
        rebuilt.append({'query_id': qid, 'n': n, 'replay_positives': ca,
            'reverse_positives': cb, 'positive_count_gap': delta,
            'candidate_disagreements': sum(a != b for a, b in zip(x, y)),
            'pool_worst_ratio_error_floor': str(Fraction((delta + 1) // 2, n))})
    assert rebuilt == recorded and [x['query_id'] for x in rebuilt] == prior['common_query_ids']
    assert dict(complete) == report['completion']
    q, n = len(rebuilt), 32
    assert q == report['common_complete_pools'] and n == report['candidates_per_pool']
    d = sum(x['positive_count_gap'] for x in rebuilt)
    c = sum(x['candidate_disagreements'] for x in rebuilt)
    assert d == report['positive_count_gap_total'] and c == report['candidate_disagreements']
    assert sum(x['positive_count_gap'] > 0 for x in rebuilt) == report['pools_with_changed_ratio']
    histogram = dict(collections.Counter(str(x['positive_count_gap']) for x in rebuilt))
    assert histogram == report['positive_count_gap_histogram']
    assert report['mean_absolute_ratio_change'] == str(Fraction(d, q * n))
    floor = Fraction((d + 1) // 2, q * n)
    assert report['sharp_worse_macro_mae_floor'] == str(floor)
    assert abs(report['sharp_worse_macro_mae_floor_float'] - float(floor)) < 1e-12
    assert report['mean_of_per_pool_worse_error_floors'] == str(
        sum((Fraction((x['positive_count_gap'] + 1) // 2, n) for x in rebuilt), Fraction()) / q)
    remaining = (d + 1) // 2
    left_errors = right_errors = 0
    for x in rebuilt:
        aa, bb = x['replay_positives'], x['reverse_positives']
        used = min(remaining, abs(aa - bb))
        reference = aa + used if aa <= bb else aa - used
        assert 0 <= reference <= n
        remaining -= used
        left_errors += abs(aa - reference)
        right_errors += abs(bb - reference)
    assert remaining == 0 and left_errors + right_errors == d
    assert Fraction(max(left_errors, right_errors), q * n) == floor
    assert prior['comparisons']['reverse_stable_ids']['changed_candidate_count'] == c
    print('Independently reconstructed all pool counts, ratio gaps and exact floors.')
    print('Attainable synthetic reference-count witness:', left_errors, right_errors,
          'error-count totals over', q, 'pools; worse macro-MAE floor:', floor)


if __name__ == '__main__':
    main()
