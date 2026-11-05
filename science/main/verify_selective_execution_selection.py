#!/usr/bin/env python3
"""Independently reconstruct the new WANDS sample from pinned source tables."""
import argparse
import collections
import hashlib
import json
import random
from pathlib import Path

import pandas as pd
import run_wands

HERE = Path(__file__).resolve().parent


def lines(path):
    with path.open(encoding='utf8') as f:
        return [json.loads(s) for s in f]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--wands-dir', type=Path, default=Path('/home/gpu/intent2s_wands_official_2026'))
    ap.add_argument('--selection', type=Path, required=True)
    a = ap.parse_args()
    for name, expected in run_wands.EXPECTED.items():
        assert run_wands.sha256(a.wands_dir / name) == expected
    q = pd.read_csv(a.wands_dir / 'query.csv', sep='\t', keep_default_na=False, dtype={'query_id': 'int64', 'query': str})
    l = pd.read_csv(a.wands_dir / 'label.csv', sep='\t', keep_default_na=False,
                    dtype={'id': 'int64', 'query_id': 'int64', 'product_id': 'int64', 'label': str})
    p = pd.read_csv(a.wands_dir / 'product.csv', sep='\t', keep_default_na=False,
                    usecols=['product_id', 'product_name', 'product_class', 'product_features', 'product_description'],
                    dtype={'product_id': 'int64', 'product_name': str, 'product_class': str,
                           'product_features': str, 'product_description': str})
    assert not q.query_id.duplicated().any() and not l.id.duplicated().any() and not p.product_id.duplicated().any()
    assert l.query_id.isin(q.query_id).all() and l.product_id.isin(p.product_id).all()
    assert l.label.isin(['Exact', 'Partial', 'Irrelevant']).all()
    grade_counts = l.groupby(['query_id', 'product_id']).label.nunique()
    conflicts = set(grade_counts[grade_counts > 1].index)
    records = l.sort_values('id').drop_duplicates(['query_id', 'product_id'], keep='first')
    records = records[~pd.MultiIndex.from_frame(records[['query_id', 'product_id']]).isin(conflicts)]
    previous = (lines(HERE / 'run_wands_160/inputs.jsonl') + lines(HERE / 'wands_holdout_selection/inputs.jsonl')
                + lines(HERE / 'wands_fallback_validation_selection/inputs.jsonl'))
    old_products = {int(it['product_id']) for row in previous for it in row['items']}
    old_queries = {row['query'].strip().casefold() for row in previous}
    for name in ('run_test_160', 'run_validation_500', 'run_full_pools_250'):
        old_queries.update(row['query'].strip().casefold() for row in lines(HERE / name / 'inputs.jsonl'))
    records = records[~records.product_id.isin(old_products)]
    allowed = q[q['query'].str.strip().ne('') & ~q['query'].str.strip().str.casefold().isin(old_queries)]
    assert allowed['query'].str.strip().str.casefold().is_unique
    counts = records.groupby('query_id').size().to_dict()
    eligible = sorted(int(x) for x in allowed.query_id if counts.get(x, 0) >= 32)
    assert len(eligible) == 52
    selected = eligible
    saved = lines(a.selection / 'inputs.jsonl')
    meta = json.loads((a.selection / 'selection.json').read_text())
    assert meta['protocol'] == 'PROTOCOL_selective_execution.md' and meta['seed'] == 20271129
    assert len(saved) == 52 and [int(row['query_id']) for row in saved] == selected
    assert meta['eligible_query_ids_after_product_exclusion'] == len(eligible)
    assert meta['prior_wands_sampled_product_ids_excluded'] == len(old_products)
    assert meta['conflicting_query_product_groups_removed'] == len(conflicts)
    by_query = q.set_index('query_id')['query'].to_dict()
    by_product = p.set_index('product_id').to_dict('index')
    for row in saved:
        qid = int(row['query_id'])
        assert row['query'] == by_query[qid] and row['query'].strip().casefold() not in old_queries
        entries = records[records.query_id == qid].sort_values(['id', 'product_id']).to_dict('records')
        seed = int.from_bytes(hashlib.sha256(('20271129\0' + str(qid)).encode()).digest()[:8], 'big')
        rng = random.Random(seed); chosen = rng.sample(entries, 32); rng.shuffle(chosen)
        assert len(row['items']) == 32
        for i, (item, entry) in enumerate(zip(row['items'], chosen), 1):
            pid = int(entry['product_id']); product = by_product[pid]
            text = (f"Name: {product['product_name'][:140]}\nClass: {product['product_class'][:60]}\n"
                    f"Features: {product['product_features'][:210]}\nDescription: {product['product_description'][:140]}")[:600]
            assert item == {'id': i, 'text': text, 'grade': entry['label'],
                            'gold': 'Y' if entry['label'] == 'Exact' else 'N',
                            'example_id': str(entry['id']), 'product_id': str(pid), 'join_missing': False}
            assert pid not in old_products
    counts = collections.Counter(it['grade'] for row in saved for it in row['items'])
    assert dict(counts) == meta['grade_counts'] and meta['candidate_count'] == 52*32
    assert meta['prior_wands_sampled_product_overlap'] == meta['prior_query_text_overlap'] == 0
    print('Pinned source and complete query/product-disjoint sample independently reconstructed.')
    print('Queries:', len(saved), 'candidates:', 52*32, 'eligible:', len(eligible), 'excluded product IDs:', len(old_products))


if __name__ == '__main__':
    main()
