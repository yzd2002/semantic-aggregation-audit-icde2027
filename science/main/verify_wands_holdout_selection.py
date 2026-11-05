#!/usr/bin/env python3
"""Independently rebuild the frozen WANDS holdout selection from source tables."""
import argparse, collections, hashlib, json, random
from pathlib import Path
import pandas as pd
import run_wands

HERE = Path(__file__).resolve().parent
def lines(path):
    with path.open(encoding='utf8') as f:return [json.loads(s) for s in f]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--wands-dir', type=Path, default=Path('/home/gpu/intent2s_wands_official_2026'))
    ap.add_argument('--selection', type=Path, default=HERE / 'wands_holdout_selection')
    a = ap.parse_args()
    for name, expected in run_wands.EXPECTED.items():
        assert run_wands.sha256(a.wands_dir / name) == expected
    queries = pd.read_csv(a.wands_dir / 'query.csv', sep='\t', keep_default_na=False,
                          dtype={'query_id': 'int64', 'query': str})
    judgments = pd.read_csv(a.wands_dir / 'label.csv', sep='\t', keep_default_na=False,
                            dtype={'id': 'int64', 'query_id': 'int64', 'product_id': 'int64', 'label': str})
    products = pd.read_csv(a.wands_dir / 'product.csv', sep='\t', keep_default_na=False,
                           usecols=['product_id', 'product_name', 'product_class', 'product_features', 'product_description'],
                           dtype={'product_id': 'int64', 'product_name': str, 'product_class': str,
                                  'product_features': str, 'product_description': str})
    assert not queries.query_id.duplicated().any() and not judgments.id.duplicated().any() and not products.product_id.duplicated().any()
    assert judgments.query_id.isin(queries.query_id).all() and judgments.product_id.isin(products.product_id).all()
    grades_per_pair = judgments.groupby(['query_id', 'product_id']).label.nunique()
    conflicts = set(grades_per_pair[grades_per_pair > 1].index)
    unique = judgments.sort_values('id').drop_duplicates(['query_id', 'product_id'], keep='first')
    unique = unique[~pd.MultiIndex.from_frame(unique[['query_id', 'product_id']]).isin(conflicts)]
    previous_wands = lines(HERE / 'run_wands_160/inputs.jsonl')
    old_products = {int(it['product_id']) for row in previous_wands for it in row['items']}
    old_queries = {row['query'].strip().casefold() for row in previous_wands}
    for name in ('run_test_160', 'run_validation_500', 'run_full_pools_250'):
        old_queries.update(row['query'].strip().casefold() for row in lines(HERE / name / 'inputs.jsonl'))
    unique = unique[~unique.product_id.isin(old_products)]
    allowed = queries[queries['query'].str.strip().ne('') & ~queries['query'].str.strip().str.casefold().isin(old_queries)]
    assert allowed['query'].str.strip().str.casefold().is_unique
    counts = unique.groupby('query_id').size().to_dict()
    eligible = sorted(int(x) for x in allowed.query_id if counts.get(x, 0) >= 32)
    expected_qids = random.Random(20271122).sample(eligible, 160)
    saved = lines(a.selection / 'inputs.jsonl')
    metadata = json.loads((a.selection / 'selection.json').read_text())
    assert metadata['protocol'] == 'PROTOCOL_wands_holdout.md' and metadata['seed'] == 20271122
    assert len(saved) == 160 and [int(x['query_id']) for x in saved] == expected_qids
    assert metadata['eligible_query_ids_after_product_exclusion'] == len(eligible)
    assert metadata['prior_wands_sampled_product_ids_excluded'] == len(old_products) and metadata['conflicting_query_product_groups_removed'] == len(conflicts)
    by_query = queries.set_index('query_id')['query'].to_dict()
    by_product = products.set_index('product_id').to_dict('index')
    for row in saved:
        qid = int(row['query_id'])
        assert row['query'] == by_query[qid] and row['query'].strip().casefold() not in old_queries
        entries = unique[unique.query_id == qid].sort_values(['id', 'product_id']).to_dict('records')
        seed_bytes = hashlib.sha256(('20271122\0' + str(qid)).encode('utf8')).digest()[:8]
        rng = random.Random(int.from_bytes(seed_bytes, 'big'))
        expected = rng.sample(entries, 32)
        rng.shuffle(expected)
        assert len(row['items']) == 32
        for i, (it, source) in enumerate(zip(row['items'], expected), 1):
            pid = int(source['product_id']);p = by_product[pid]
            text = (f"Name: {p['product_name'][:140]}\nClass: {p['product_class'][:60]}\n"
                    f"Features: {p['product_features'][:210]}\nDescription: {p['product_description'][:140]}")[:600]
            assert it == {'id': i, 'text': text, 'grade': source['label'],
                          'gold': 'Y' if source['label'] == 'Exact' else 'N',
                          'example_id': str(source['id']), 'product_id': str(pid), 'join_missing': False}
            assert pid not in old_products
    counts = collections.Counter(it['grade'] for row in saved for it in row['items'])
    assert dict(counts) == metadata['grade_counts'] and metadata['candidate_count'] == 5120
    assert metadata['prior_wands_sampled_product_overlap'] == metadata['prior_query_text_overlap'] == 0
    print('Pinned source, exact query and product sampling, text serialization, labels and disjointness independently verified.')
    print('Queries:',len(saved),'candidates:',sum(len(r['items']) for r in saved),'eligible queries:',len(eligible),'excluded sampled product IDs:',len(old_products))
if __name__ == '__main__':
    main()
