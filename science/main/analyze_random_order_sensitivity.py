#!/usr/bin/env python3
"""Summarize paired saved order-sensitivity outputs on common-complete pools."""
import argparse
import collections
import json
from fractions import Fraction
from pathlib import Path

from run_validation import parse_compact

MODES = ('baseline_a', 'perm_1', 'baseline_b', 'perm_2', 'perm_3', 'baseline_c')
PAIRS = (('perm_1', 'baseline_a'), ('perm_2', 'baseline_b'), ('perm_3', 'baseline_c'),
         ('baseline_b', 'baseline_a'), ('baseline_c', 'baseline_a'))


def lines(path):
    with path.open(encoding='utf8') as f:
        return [json.loads(line) for line in f]


def quality(rows, predictions, indices):
    counts = collections.Counter()
    errors = []
    for i in indices:
        gold = [x['gold'] for x in rows[i]['items']]
        pred = predictions[i]
        counts.update(zip(gold, pred))
        errors.append(Fraction(abs(gold.count('Y') - pred.count('Y')), len(gold)))
    tp, tn, fp, fn = counts['Y', 'Y'], counts['N', 'N'], counts['N', 'Y'], counts['Y', 'N']
    return {'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
        'f1': 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.,
        'balanced_accuracy': (tp / (tp + fn) + tn / (tn + fp)) / 2 if tp + fn and tn + fp else None,
        'ratio_mae': float(sum(errors, Fraction()) / len(errors))}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--selection', type=Path, required=True)
    p.add_argument('--orders', type=Path, required=True)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        raise FileExistsError('Choose a new analysis output directory.')
    rows, mappings = lines(a.selection / 'inputs.jsonl'), lines(a.orders / 'mappings.jsonl')
    pre = json.loads((a.run / 'preflight.json').read_text())
    controls = json.loads((a.run / 'controls.json').read_text())
    assert len(rows) == len(mappings) == 160 and list(pre['mode_order']) == list(MODES)
    assert controls['unconstrained_invalid'] and controls['guided_valid']
    preds, completion = {}, {}
    for mode in MODES:
        records = lines(a.run / (mode + '_records.jsonl'))
        if len(records) != 640:
            raise ValueError('An output mode is partial; preserve it and do not analyze as complete.')
        mode_preds = []
        retries = valid_chunks = 0
        for qi, (row, mapping) in enumerate(zip(rows, mappings)):
            if row['query_id'] != mapping['query_id'] or len(row['items']) != 32:
                raise ValueError('Selection and mapping do not align.')
            vec = [None] * 32
            for ci, chunk in enumerate(mapping['chunks']):
                record = records[qi * 4 + ci]
                start = ci * 8
                order = chunk['orders'][mode] if mode.startswith('perm_') else list(range(8))
                expected = [start + j for j in order]
                if (record['query_id'], record['start'], record['size'], record['indices']) != (row['query_id'], start, 8, expected):
                    raise ValueError('Output mapping differs from fixed selection.')
                labels = parse_compact(record['final'], 8)
                retries += record['retry_finish_reason'] is not None
                if labels is None:
                    continue
                valid_chunks += 1
                for j, value in zip(expected, labels):
                    if vec[j] is not None:
                        raise ValueError('Duplicate candidate position.')
                    vec[j] = value
            mode_preds.append(None if None in vec else vec)
        preds[mode] = mode_preds
        completion[mode] = {'complete_pools': sum(x is not None for x in mode_preds),
                            'valid_chunks': valid_chunks, 'retry_calls': retries}
    common = [i for i in range(160) if all(preds[m][i] is not None for m in MODES)]
    if not common:
        raise ValueError('No common-complete pools.')
    comparison = {}
    per_pool = [{'query_id': rows[i]['query_id'], 'comparisons': {}} for i in common]
    for target, reference in PAIRS:
        changed = ratio = 0
        changed_pools = 0
        for k, i in enumerate(common):
            x, y = preds[target][i], preds[reference][i]
            d = sum(u != v for u, v in zip(x, y))
            delta = abs(x.count('Y') - y.count('Y'))
            changed += d
            ratio += delta
            changed_pools += d > 0
            per_pool[k]['comparisons'][target + '_vs_' + reference] = {
                'changed_candidates': d, 'absolute_positive_count_change': delta}
        comparison[target + '_vs_' + reference] = {'changed_candidates': changed,
            'comparable_candidates': len(common) * 32, 'change_fraction': changed / (len(common) * 32),
            'changed_pools': changed_pools, 'mean_absolute_ratio_change': ratio / (len(common) * 32)}
    summary = {'protocol': 'PROTOCOL_random_order_sensitivity.md', 'model': pre['model'],
        'attempted_pools': len(rows), 'completion': completion, 'common_complete_pools': len(common),
        'common_query_ids': [rows[i]['query_id'] for i in common], 'comparisons': comparison,
        'quality_common': {m: quality(rows, preds[m], common) for m in MODES},
        'interpretation': 'Post-hoc exposed sample; paired descriptive contrasts, not causal or population inference.'}
    a.out.mkdir(parents=True, exist_ok=False)
    (a.out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    with (a.out / 'per_pool.jsonl').open('w', encoding='utf8') as f:
        for row in per_pool:
            f.write(json.dumps(row) + '\n')
    print('Model:', pre['model'], 'common-complete pools:', len(common))
    for name, value in comparison.items():
        print(name, 'changed:', value['changed_candidates'], '/', value['comparable_candidates'],
              'mean ratio change:', round(value['mean_absolute_ratio_change'], 4))


if __name__ == '__main__':
    main()
