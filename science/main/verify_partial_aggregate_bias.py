#!/usr/bin/env python3
"""Reconstruct the archived partial-aggregate diagnostic without its analysis code."""
import argparse
import hashlib
import json
import re
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
METHOD = 'two_stage_compact_fallback'
DATASETS = {'esci_250': 'run_full_pools_250', 'wands_160': 'run_wands_160'}


def read_lines(path):
    return [json.loads(x) for x in path.read_text(encoding='utf8').splitlines()]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def near(a, b):
    return a is None and b is None or a is not None and b is not None and abs(a - b) < 1e-12


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', type=Path, required=True)
    args = ap.parse_args()
    summary = json.loads((args.run / 'summary.json').read_text())
    details = read_lines(args.run / 'details.jsonl')
    assert summary['protocol'] == 'PROTOCOL_partial_aggregate_bias.md' and summary['method'] == METHOD
    recorded = {(x['dataset'], x['query_id']): x for x in details}
    assert len(recorded) == len(details)
    for label, dirname in DATASETS.items():
        src = HERE / dirname
        rows, outputs = read_lines(src / 'inputs.jsonl'), read_lines(src / 'outputs.jsonl')
        saved = summary['datasets'][label]
        assert saved['source'] == dirname and saved['input_sha256'] == digest(src / 'inputs.jsonl')
        assert saved['output_sha256'] == digest(src / 'outputs.jsonl')
        assert saved['prior_failure_details_sha256'] == digest(src / 'failure_bounds_details.jsonl')
        old = {x['query_id']: x for x in read_lines(src / 'failure_bounds_details.jsonl') if x['method'] == METHOD}
        rebuilt = []
        assert len(rows) == len(outputs)
        for row, out in zip(rows, outputs):
            assert row['query_id'] == out['query_id']
            plan = out['plans'][METHOD]
            if plan['pred'] is not None:
                continue
            n = len(row['items']); labels = {}; visited = set()
            for chunk in plan['raw']:
                indices = list(range(chunk['start'], chunk['start'] + chunk['size']))
                assert indices and min(indices) >= 0 and max(indices) < n and not visited.intersection(indices)
                visited.update(indices)
                output = chunk['final']
                if isinstance(output, str) and re.fullmatch('[YN]{' + str(chunk['size']) + '}', output):
                    labels.update(zip(indices, output))
            assert visited == set(range(n))
            gold = [it['gold'] for it in row['items']]
            assert all(x in ('Y', 'N') for x in gold)
            m = len(labels); p = sum(x == 'Y' for x in labels.values())
            ga = sum(gold[i] == 'Y' for i in labels)
            gm = sum(gold[i] == 'Y' for i in range(n) if i not in labels)
            wrong = sum(v != gold[i] for i, v in labels.items())
            reference = Fraction(ga + gm, n)
            zero = Fraction(p, n)
            available = Fraction(p, m) if m else None
            composition = Fraction(ga, m) - reference if m else None
            discrepancy = Fraction(p - ga, m) if m else None
            error = available - reference if m else None
            if m:
                assert error == composition + discrepancy
            values = {'n': n, 'm': m, 'p': p, 'gold_accepted_positive': ga, 'gold_missing_positive': gm,
                      'accepted_mistakes': wrong, 'reference_ratio': float(reference), 'coverage': m / n,
                      'zero_fill_ratio': float(zero), 'zero_fill_error': float(zero - reference),
                      'available_ratio': None if available is None else float(available),
                      'available_error': None if error is None else float(error),
                      'observed_composition_term': None if composition is None else float(composition),
                      'accepted_discrepancy_term': None if discrepancy is None else float(discrepancy)}
            rec = recorded.pop((label, row['query_id']))
            assert set(rec) == set(values) | {'dataset', 'query_id'}
            assert all(near(rec[k], v) for k, v in values.items())
            prev = old.pop(row['query_id'])
            assert (n, m, p, wrong, ga + gm) == (prev['n'], prev['accepted'], prev['accepted_positive'],
                                                  prev['accepted_mistakes'], prev['reference_positive'])
            rebuilt.append(values)
        assert not old
        avail = [x for x in rebuilt if x['m']]
        avg = lambda xs: sum(xs) / len(xs) if xs else None
        expected = {'attempted_pools': len(rows), 'failed_pools': len(rebuilt),
                    'available_only_defined_pools': len(avail),
                    'accepted_candidates': sum(x['m'] for x in rebuilt),
                    'unresolved_candidates': sum(x['n'] - x['m'] for x in rebuilt),
                    'mean_query_coverage': avg([x['coverage'] for x in rebuilt]),
                    'available_only_mean_signed_error': avg([x['available_error'] for x in avail]),
                    'available_only_mae': avg([abs(x['available_error']) for x in avail]),
                    'observed_composition_mean': avg([x['observed_composition_term'] for x in avail]),
                    'accepted_discrepancy_mean': avg([x['accepted_discrepancy_term'] for x in avail]),
                    'zero_fill_mean_signed_error': avg([x['zero_fill_error'] for x in rebuilt]),
                    'zero_fill_mae': avg([abs(x['zero_fill_error']) for x in rebuilt]),
                    'reference_positives': sum(x['gold_accepted_positive'] + x['gold_missing_positive'] for x in rebuilt)}
        assert all(near(saved[k], v) for k, v in expected.items())
        print(label, 'failed pools:', len(rebuilt), 'available-only defined:', len(avail), 'verified')
    assert not recorded
    print('Raw chunk associations, prior failure counts, exact decompositions and summaries independently verified.')


if __name__ == '__main__':
    main()
