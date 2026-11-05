#!/usr/bin/env python3
"""Independently recompute the crossed query–product resampling sensitivity."""
import argparse
import json
from pathlib import Path

import numpy as np
from verify_order_fallback_validation import keyed, lines, point


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--selection', type=Path, required=True)
    a = ap.parse_args()
    rows = lines(a.selection / 'inputs.jsonl')
    prior = json.loads((a.run / 'paired_analysis.json').read_text())
    observed = json.loads((a.run / 'two_way_analysis.json').read_text())
    modes = {m: lines(a.run / (m + '_responses.jsonl')) for m in ('original', 'replay', 'stable_reverse', 'pointwise')}
    original = []; combined = []; gold = []; product = []; queries = []
    for q, row in enumerate(rows):
        vectors = {}
        for mode in ('original', 'replay', 'stable_reverse'):
            block = []
            for record in modes[mode][q*4:(q+1)*4]:
                result = keyed(record['final'])
                block.extend(result if result is not None else [None]*8)
            vectors[mode] = block
        h = [point(x['final']) for x in modes['pointwise'][q*32:(q+1)*32]]
        if any(None in v for v in vectors.values()) or None in h:
            continue
        out = [x if x == y else z for x, y, z in zip(vectors['original'], vectors['stable_reverse'], h)]
        original.append([x == 'Y' for x in vectors['original']]); combined.append([x == 'Y' for x in out])
        gold.append([x['gold'] == 'Y' for x in row['items']]); product.append([int(x['product_id']) for x in row['items']])
        queries.append(row['query_id'])
    assert queries == prior['common_query_ids']
    distinct = sorted({x for row in product for x in row})
    index = {p: i for i, p in enumerate(distinct)}
    mapping = np.array([[index[x] for x in row] for row in product], dtype=int)
    orig = np.asarray(original, dtype=float); fallback = np.asarray(combined, dtype=float); ref = np.asarray(gold, dtype=float)
    rng = np.random.default_rng(20271128); contrasts = []
    for _ in range(4000):
        qcount = np.bincount(rng.integers(0, len(queries), len(queries)), minlength=len(queries))
        pcount = np.bincount(rng.integers(0, len(distinct), len(distinct)), minlength=len(distinct))
        weights = pcount[mapping]; denom = weights.sum(axis=1)
        assert not np.any((qcount > 0) & (denom == 0))
        safe = np.where(denom == 0, 1, denom)
        g = (weights*ref).sum(axis=1)/safe
        x = (weights*orig).sum(axis=1)/safe
        y = (weights*fallback).sum(axis=1)/safe
        contrasts.append(float(np.dot(qcount, np.abs(y-g)-np.abs(x-g))/qcount.sum()))
    ci = np.quantile(contrasts, [.025, .975])
    assert observed['protocol'] == 'PROTOCOL_order_fallback_two_way.md'
    assert observed['selection_inputs_sha256'] == prior['selection_inputs_sha256']
    assert observed['common_complete_pools'] == len(queries) and observed['distinct_product_ids'] == len(distinct)
    assert observed['repeated_product_ids'] == int(sum(v > 1 for v in np.bincount(mapping.ravel())))
    assert observed['resamples'] == 4000 and observed['seed'] == 20271128
    assert np.allclose(ci, observed['pointwise_fallback_minus_original_ratio_mae_query_product_ci'], rtol=0, atol=1e-12)
    print('Crossed query–product resampling independently verified on',len(queries),'common queries and',len(distinct),'distinct products.')


if __name__ == '__main__':
    main()
