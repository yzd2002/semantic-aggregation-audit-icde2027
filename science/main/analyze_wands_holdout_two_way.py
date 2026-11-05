#!/usr/bin/env python3
"""Pigeonhole-style query/product resampling for holdout order disagreements."""
import argparse, collections, json
from pathlib import Path
import numpy as np
from run_compact_prompt_ablation import lines

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--selection', type=Path, required=True)
    a = ap.parse_args()
    rows = lines(a.selection / 'inputs.jsonl')
    analysis = json.loads((a.run / 'paired_analysis.json').read_text())
    orig = lines(a.run / 'original_outputs.jsonl')
    replay = lines(a.run / 'replay_outputs.jsonl')
    reverse = lines(a.run / 'stable_reverse_outputs.jsonl')
    assert len(rows) == len(orig) == len(replay) == len(reverse)
    assert all(r['query_id'] == x['query_id'] == y['query_id'] == z['query_id'] for r, x, y, z in zip(rows, orig, replay, reverse))
    idx = [i for i, x, y, z in zip(range(len(rows)), orig, replay, reverse) if x['pred'] is not None and y['pred'] is not None and z['pred'] is not None]
    assert len(idx) == analysis['common_complete_pools'] and [rows[i]['query_id'] for i in idx] == analysis['common_query_ids']
    if not idx:raise ValueError('No common-complete pools for two-way sensitivity.')
    products = sorted({int(it['product_id']) for i in idx for it in rows[i]['items']})
    product_index = {p: i for i, p in enumerate(products)}
    q_index, p_index, disagreement, product_counts = [], [], [], collections.Counter()
    for k, i in enumerate(idx):
        left, right = replay[i]['pred'], reverse[i]['pred']
        assert len(left) == len(right) == len(rows[i]['items'])
        for item, x, y in zip(rows[i]['items'], left, right):
            pid = int(item['product_id']);q_index.append(k);p_index.append(product_index[pid]);disagreement.append(x != y);product_counts[pid] += 1
    q_index = np.array(q_index, dtype=int);p_index = np.array(p_index, dtype=int);z = np.array(disagreement, dtype=np.int64)
    assert int(z.sum()) == analysis['comparisons']['reverse_stable_ids']['changed_candidate_count']
    rng = np.random.default_rng(20271124);rates = []
    for _ in range(4000):
        qw = np.bincount(rng.integers(0, len(idx), size=len(idx)), minlength=len(idx))
        pw = np.bincount(rng.integers(0, len(products), size=len(products)), minlength=len(products))
        weight = qw[q_index] * pw[p_index]
        denominator = int(weight.sum())
        if denominator == 0:raise RuntimeError('A two-way bootstrap replicate has zero total weight.')
        rates.append(float(np.dot(weight, z) / denominator))
    out = {'protocol': 'PROTOCOL_wands_holdout_two_way.md', 'model': analysis['model'], 'resamples': 4000,
           'seed': 20271124, 'common_pools': len(idx), 'candidates': len(z), 'distinct_products': len(products),
           'products_appearing_multiple_times': sum(v > 1 for v in product_counts.values()),
           'disagreement_count': int(z.sum()), 'disagreement_fraction': float(z.mean()),
           'query_only_ci': analysis['comparisons']['reverse_stable_ids']['candidate_label_change_fraction_ci'],
           'query_product_ci': [float(v) for v in np.quantile(rates, [.025, .975])],
           'interpretation': 'Secondary crossed-factor resampling sensitivity; not exact population coverage or the primary criterion.'}
    (a.run / 'two_way_analysis.json').write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps(out, indent=2))
if __name__ == '__main__':main()
