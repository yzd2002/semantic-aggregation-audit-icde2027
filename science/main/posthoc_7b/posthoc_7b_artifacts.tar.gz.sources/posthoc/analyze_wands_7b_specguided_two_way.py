#!/usr/bin/env python3
"""Compute crossed query–product sensitivity for actual selective fallback."""
import argparse
import json
from pathlib import Path

import numpy as np
from analyze_order_fallback_validation import keyed_vectors
from order_fallback import combine
from run_order_fallback_validation import parse_one
from select_selective_execution import lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--selection', type=Path, required=True)
    a = ap.parse_args()
    rows = lines(a.selection / 'inputs.jsonl')
    previous = json.loads((a.run / 'paired_analysis.json').read_text())
    keyed = {name: keyed_vectors(rows, lines(a.run / (name + '_responses.jsonl')))
             for name in ('original', 'replay', 'stable_reverse')}
    full_records = lines(a.run / 'pointwise_responses.jsonl')
    full_partial = [[parse_one(x['final']) for x in full_records[i*32:(i+1)*32]] for i in range(len(rows))]
    full = [x if None not in x else None for x in full_partial]
    selected = {(x['query_id'], x['start']): parse_one(x['final']) for x in lines(a.run / 'selective_pointwise_responses.jsonl')}
    sparse = [[selected.get((row['query_id'], i)) for i in range(32)] for row in rows]
    actual = [combine(x, y, z) for x, y, z in zip(keyed['original'], keyed['stable_reverse'], sparse)]
    virtual = [combine(x, y, z) for x, y, z in zip(keyed['original'], keyed['stable_reverse'], full_partial)]
    common = [i for i in range(len(rows)) if all(x is not None for x in
              (keyed['original'][i], keyed['replay'][i], keyed['stable_reverse'][i], full[i], actual[i], virtual[i]))]
    if not common:
        raise ValueError('No common-complete pools for crossed resampling.')
    assert [rows[i]['query_id'] for i in common] == previous['common_query_ids']
    product_ids = sorted({int(item['product_id']) for i in common for item in rows[i]['items']})
    product_index = {pid: j for j, pid in enumerate(product_ids)}
    query_index = np.repeat(np.arange(len(common)), 32)
    product_vector = np.array([product_index[int(x['product_id'])] for i in common for x in rows[i]['items']])
    reference = np.array([x['gold'] == 'Y' for i in common for x in rows[i]['items']], dtype=float)
    original = np.array([x == 'Y' for i in common for x in keyed['original'][i]], dtype=float)
    fallback = np.array([x == 'Y' for i in common for x in actual[i]], dtype=float)
    rng = np.random.default_rng(20271203); contrasts = []
    for _ in range(4000):
        qcount = np.bincount(rng.integers(0, len(common), len(common)), minlength=len(common))
        pcount = np.bincount(rng.integers(0, len(product_ids), len(product_ids)), minlength=len(product_ids))
        weights = pcount[product_vector]
        den = np.bincount(query_index, weights=weights, minlength=len(common))
        if np.any((qcount > 0) & (den == 0)):
            raise RuntimeError('A selected query has zero product weight in a bootstrap draw.')
        safe = np.where(den == 0, 1, den)
        ref = np.bincount(query_index, weights=weights*reference, minlength=len(common))/safe
        orig = np.bincount(query_index, weights=weights*original, minlength=len(common))/safe
        fb = np.bincount(query_index, weights=weights*fallback, minlength=len(common))/safe
        contrasts.append(float(np.dot(qcount, np.abs(fb-ref)-np.abs(orig-ref))/qcount.sum()))
    result = {'protocol': 'PROTOCOL_7b_specguided_posthoc.md', 'selection_inputs_sha256': previous['selection_inputs_sha256'],
              'common_complete_pools': len(common), 'distinct_product_ids': len(product_ids),
              'repeated_product_ids': int(sum(x > 1 for x in np.bincount(product_vector))),
              'resamples': 4000, 'seed': 20271203,
              'actual_fallback_minus_original_ratio_mae_query_product_ci':
                  [float(x) for x in np.quantile(contrasts, [.025, .975])]}
    (a.run / 'two_way_analysis.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
