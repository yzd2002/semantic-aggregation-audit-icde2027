#!/usr/bin/env python3
"""Select WANDS queries and products disjoint from two prior sampled workloads."""
import argparse
import collections
import hashlib
import json
import random
from pathlib import Path

import pandas as pd
import run_wands

HERE = Path(__file__).resolve().parent
SEED = 20271129
N = 52
PROTOCOL = 'PROTOCOL_selective_execution.md'


def lines(path):
    with path.open(encoding='utf8') as f:
        return [json.loads(line) for line in f]


def sample(source):
    for name, expected in run_wands.EXPECTED.items():
        if run_wands.sha256(source / name) != expected:
            raise ValueError('Unexpected source hash: ' + name)
    queries = pd.read_csv(source / 'query.csv', sep='\t', keep_default_na=False,
                          dtype={'query_id': 'int64', 'query': str})
    labels = pd.read_csv(source / 'label.csv', sep='\t', keep_default_na=False,
                         dtype={'id': 'int64', 'query_id': 'int64', 'product_id': 'int64', 'label': str})
    products = pd.read_csv(source / 'product.csv', sep='\t', keep_default_na=False,
                           usecols=['product_id', 'product_name', 'product_class', 'product_features', 'product_description'],
                           dtype={'product_id': 'int64', 'product_name': str, 'product_class': str,
                                  'product_features': str, 'product_description': str})
    if queries.query_id.duplicated().any() or labels.id.duplicated().any() or products.product_id.duplicated().any():
        raise ValueError('Nonunique source keys.')
    if not labels.query_id.isin(queries.query_id).all() or not labels.product_id.isin(products.product_id).all():
        raise ValueError('Unmatched source foreign keys.')
    if not labels.label.isin(['Exact', 'Partial', 'Irrelevant']).all():
        raise ValueError('Unexpected grade.')
    grades = labels.groupby(['query_id', 'product_id']).label.nunique()
    conflicts = grades[grades > 1].index
    unique = (labels.sort_values(['query_id', 'product_id', 'id'])
              .drop_duplicates(['query_id', 'product_id'], keep='first')
              .set_index(['query_id', 'product_id']).drop(index=conflicts).reset_index())
    old = (lines(HERE / 'run_wands_160/inputs.jsonl') + lines(HERE / 'wands_holdout_selection/inputs.jsonl')
           + lines(HERE / 'wands_fallback_validation_selection/inputs.jsonl'))
    old_products = {int(item['product_id']) for row in old for item in row['items']}
    old_queries = {row['query'].strip().casefold() for row in old}
    for name in ('run_test_160', 'run_validation_500', 'run_full_pools_250'):
        old_queries.update(row['query'].strip().casefold() for row in lines(HERE / name / 'inputs.jsonl'))
    normalized = queries['query'].str.strip().str.casefold()
    allowed = queries[normalized.ne('') & ~normalized.isin(old_queries)]
    if not allowed['query'].str.strip().str.casefold().is_unique:
        raise ValueError('Remaining normalized query strings are not unique.')
    remaining = unique[~unique.product_id.isin(old_products)]
    counts = remaining.groupby('query_id').size()
    eligible = sorted(int(qid) for qid in allowed.query_id if counts.get(qid, 0) >= 32)
    if len(eligible) != N:
        raise ValueError(f'Expected {N} eligible query IDs, found {len(eligible)}.')
    selected = eligible
    by_query = queries.set_index('query_id')['query'].to_dict()
    by_product = products.set_index('product_id').to_dict('index')
    groups = {int(qid): group.sort_values(['id', 'product_id'])
              for qid, group in remaining[remaining.query_id.isin(selected)].groupby('query_id')}
    rows = []
    for qid in selected:
        entries = groups[qid].to_dict('records')
        seed = int.from_bytes(hashlib.sha256((str(SEED) + '\0' + str(qid)).encode()).digest()[:8], 'big')
        rng = random.Random(seed)
        chosen = rng.sample(entries, 32)
        rng.shuffle(chosen)
        items = []
        for i, entry in enumerate(chosen, 1):
            pid = int(entry['product_id'])
            items.append({'id': i, 'text': run_wands.make_text(by_product[pid]), 'grade': entry['label'],
                          'gold': 'Y' if entry['label'] == 'Exact' else 'N', 'example_id': str(entry['id']),
                          'product_id': str(pid), 'join_missing': False})
        rows.append({'query_id': str(qid), 'query': str(by_query[qid]), 'items': items})
    new_products = {int(it['product_id']) for row in rows for it in row['items']}
    new_queries = {row['query'].strip().casefold() for row in rows}
    assert len(new_queries) == N and not new_queries & old_queries and not new_products & old_products
    metadata = {'protocol': PROTOCOL, 'source': 'Wayfair WANDS', 'revision': '3b74dcf4ba29ab8ff3e6a50b5b09fc627cb882b5',
                'seed': SEED, 'source_hashes': run_wands.EXPECTED, 'conflicting_query_product_groups_removed': len(conflicts),
                'prior_wands_sampled_product_ids_excluded': len(old_products), 'remaining_query_ids': len(allowed),
                'eligible_query_ids_after_product_exclusion': len(eligible), 'sample_size': len(rows),
                'candidate_count': sum(len(row['items']) for row in rows), 'candidates_per_pool': 32,
                'unique_sampled_product_ids': len(new_products), 'prior_wands_sampled_product_overlap': 0,
                'prior_query_text_overlap': 0, 'grade_counts': dict(collections.Counter(x['grade'] for r in rows for x in r['items'])),
                'label_mapping': 'Exact=Y; Partial/Irrelevant=N', 'system_instruction': run_wands.SYSTEM}
    return rows, metadata


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--wands-dir', type=Path, default=Path('/home/gpu/intent2s_wands_official_2026'))
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    rows, metadata = sample(a.wands_dir)
    a.out.mkdir(parents=True, exist_ok=False)
    with (a.out / 'inputs.jsonl').open('x', encoding='utf8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    (a.out / 'selection.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == '__main__':
    main()
