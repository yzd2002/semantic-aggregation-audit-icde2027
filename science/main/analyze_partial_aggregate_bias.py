#!/usr/bin/env python3
"""Post-hoc decomposition of saved, incomplete compact-fallback aggregates."""
import argparse
import hashlib
import json
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
METHOD = 'two_stage_compact_fallback'
RUNS = {'esci_250': 'run_full_pools_250', 'wands_160': 'run_wands_160'}


def read_lines(path):
    with path.open(encoding='utf8') as f:
        return [json.loads(line) for line in f]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def measure(gold, accepted):
    if not gold or len(gold) != len(accepted) or any(x not in ('Y', 'N') for x in gold):
        raise ValueError('Invalid reference vector')
    if any(x not in ('Y', 'N', None) for x in accepted):
        raise ValueError('Invalid accepted vector')
    n = len(gold)
    selected = [i for i, x in enumerate(accepted) if x is not None]
    m = len(selected)
    p = sum(accepted[i] == 'Y' for i in selected)
    ga = sum(gold[i] == 'Y' for i in selected)
    gm = sum(gold[i] == 'Y' and accepted[i] is None for i in range(n))
    wrong = sum(accepted[i] != gold[i] for i in selected)
    ref = (ga + gm) / n
    zero = p / n
    result = {'n': n, 'm': m, 'p': p, 'gold_accepted_positive': ga, 'gold_missing_positive': gm,
              'accepted_mistakes': wrong, 'reference_ratio': ref, 'coverage': m / n,
              'zero_fill_ratio': zero, 'zero_fill_error': zero - ref,
              'available_ratio': None, 'available_error': None,
              'observed_composition_term': None, 'accepted_discrepancy_term': None}
    if m:
        result.update(available_ratio=p / m, available_error=p / m - ref,
                      observed_composition_term=ga / m - ref, accepted_discrepancy_term=(p - ga) / m)
    return result


def reconstruct(row, entry):
    n = len(row['items'])
    accepted = [None] * n
    covered = set()
    for chunk in entry['raw']:
        start, size, text = chunk['start'], chunk['size'], chunk['final']
        if type(start) is not int or type(size) is not int or size <= 0 or start < 0 or start + size > n:
            raise ValueError('Invalid chunk range')
        positions = set(range(start, start + size))
        if covered & positions:
            raise ValueError('Overlapping chunks')
        covered.update(positions)
        if isinstance(text, str) and len(text) == size and set(text) <= {'Y', 'N'}:
            accepted[start:start + size] = list(text)
    if covered != set(range(n)):
        raise ValueError('Incomplete attempted chunk coverage')
    return accepted


def summarize(attempted, records):
    available = [x for x in records if x['m'] > 0]
    def avg(xs):
        values = list(xs)
        return statistics.fmean(values) if values else None
    return {'attempted_pools': attempted, 'failed_pools': len(records),
            'available_only_defined_pools': len(available),
            'accepted_candidates': sum(x['m'] for x in records),
            'unresolved_candidates': sum(x['n'] - x['m'] for x in records),
            'mean_query_coverage': avg(x['coverage'] for x in records),
            'available_only_mean_signed_error': avg(x['available_error'] for x in available),
            'available_only_mae': avg(abs(x['available_error']) for x in available),
            'observed_composition_mean': avg(x['observed_composition_term'] for x in available),
            'accepted_discrepancy_mean': avg(x['accepted_discrepancy_term'] for x in available),
            'zero_fill_mean_signed_error': avg(x['zero_fill_error'] for x in records),
            'zero_fill_mae': avg(abs(x['zero_fill_error']) for x in records),
            'reference_positives': sum(x['gold_accepted_positive'] + x['gold_missing_positive'] for x in records)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    results, details = {}, []
    for label, name in RUNS.items():
        run = HERE / name
        inputs, outputs = read_lines(run / 'inputs.jsonl'), read_lines(run / 'outputs.jsonl')
        prior = {x['query_id']: x for x in read_lines(run / 'failure_bounds_details.jsonl') if x['method'] == METHOD}
        assert len(inputs) == len(outputs) and len({x['query_id'] for x in inputs}) == len(inputs)
        rows = []
        for row, out in zip(inputs, outputs):
            assert row['query_id'] == out['query_id']
            entry = out['plans'][METHOD]
            if entry['pred'] is not None:
                continue
            accepted = reconstruct(row, entry)
            record = {'dataset': label, 'query_id': row['query_id'],
                      **measure([x['gold'] for x in row['items']], accepted)}
            assert record['m'] < record['n'] and row['query_id'] in prior
            old = prior[row['query_id']]
            assert (record['n'], record['m'], record['p'], record['accepted_mistakes'],
                    record['gold_accepted_positive'] + record['gold_missing_positive']) == (
                       old['n'], old['accepted'], old['accepted_positive'], old['accepted_mistakes'], old['reference_positive'])
            if record['m']:
                assert abs(record['available_error'] - record['observed_composition_term'] - record['accepted_discrepancy_term']) < 1e-12
            rows.append(record)
        assert len(rows) == len(prior)
        results[label] = {'source': name, 'input_sha256': sha(run / 'inputs.jsonl'),
                          'output_sha256': sha(run / 'outputs.jsonl'),
                          'prior_failure_details_sha256': sha(run / 'failure_bounds_details.jsonl'),
                          **summarize(len(inputs), rows)}
        details.extend(rows)
    (args.out / 'summary.json').write_text(json.dumps({'protocol': 'PROTOCOL_partial_aggregate_bias.md',
        'method': METHOD, 'datasets': results}, indent=2, allow_nan=False) + '\n')
    with (args.out / 'details.jsonl').open('w', encoding='utf8') as f:
        for x in details:
            f.write(json.dumps(x, ensure_ascii=False, allow_nan=False) + '\n')
    for label, x in results.items():
        print(label, 'failed', x['failed_pools'], 'available-only defined', x['available_only_defined_pools'],
              'available-only MAE', x['available_only_mae'], 'zero-fill MAE', x['zero_fill_mae'])


if __name__ == '__main__':
    main()
