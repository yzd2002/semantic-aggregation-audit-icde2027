#!/usr/bin/env python3
"""Freeze a query- and previously sampled product-disjoint WANDS workload."""
import argparse, collections, hashlib, json, random
from pathlib import Path
import pandas as pd
import run_wands

HERE = Path(__file__).resolve().parent
SEED = 20271122

def load_lines(path):
    with path.open(encoding='utf8') as f:
        return [json.loads(line) for line in f]

def sample(wands_dir):
    wands_dir = Path(wands_dir)
    for name, expected in run_wands.EXPECTED.items():
        if run_wands.sha256(wands_dir / name) != expected:
            raise ValueError('Unexpected WANDS source hash: ' + name)
    q = pd.read_csv(wands_dir / 'query.csv', sep='\t', keep_default_na=False,
                    dtype={'query_id': 'int64', 'query': str})
    l = pd.read_csv(wands_dir / 'label.csv', sep='\t', keep_default_na=False,
                    dtype={'id': 'int64', 'query_id': 'int64', 'product_id': 'int64', 'label': str})
    p = pd.read_csv(wands_dir / 'product.csv', sep='\t', keep_default_na=False,
                    usecols=['product_id', 'product_name', 'product_class', 'product_features', 'product_description'],
                    dtype={'product_id': 'int64', 'product_name': str, 'product_class': str,
                           'product_features': str, 'product_description': str})
    if q.query_id.duplicated().any() or p.product_id.duplicated().any() or l.id.duplicated().any():
        raise ValueError('Source identifiers must be unique.')
    if not l.query_id.isin(q.query_id).all() or not l.product_id.isin(p.product_id).all():
        raise ValueError('Unmatched source foreign key.')
    if not l.label.isin(['Exact', 'Partial', 'Irrelevant']).all():
        raise ValueError('Unexpected source label.')
    grouped = l.groupby(['query_id', 'product_id'], sort=False).label.nunique()
    conflicts = grouped[grouped > 1].index
    unique = (l.sort_values(['query_id', 'product_id', 'id'])
              .drop_duplicates(['query_id', 'product_id'], keep='first')
              .set_index(['query_id', 'product_id']).drop(index=conflicts).reset_index())
    prior_wands = load_lines(HERE / 'run_wands_160/inputs.jsonl')
    assert len(prior_wands) == 160 and len({r['query_id'] for r in prior_wands}) == 160
    excluded_products = {int(item['product_id']) for row in prior_wands for item in row['items']}
    prior_wands_text = {r['query'].strip().casefold() for r in prior_wands}
    prior_esci_text = set()
    for name in ('run_test_160', 'run_validation_500', 'run_full_pools_250'):
        prior_esci_text.update(r['query'].strip().casefold() for r in load_lines(HERE / name / 'inputs.jsonl'))
    normalized = q['query'].str.strip().str.casefold()
    allowed = q[normalized.ne('') & ~normalized.isin(prior_wands_text | prior_esci_text)].copy()
    remaining_text = allowed['query'].str.strip().str.casefold()
    if remaining_text.duplicated().any():
        raise ValueError('Remaining normalized query strings are not unique.')
    filtered = unique[~unique.product_id.isin(excluded_products)]
    counts = filtered.groupby('query_id').size()
    eligible = sorted(int(v) for v in allowed.query_id if counts.get(v, 0) >= 32)
    if len(eligible) < 160:
        raise ValueError(f'Only {len(eligible)} query IDs remain eligible.')
    chosen = random.Random(SEED).sample(eligible, 160)
    queries = q.set_index('query_id')
    products = p.set_index('product_id').to_dict('index')
    groups = {int(qid): grp.sort_values(['id', 'product_id'])
              for qid, grp in filtered[filtered.query_id.isin(chosen)].groupby('query_id')}
    rows = []
    for qid in chosen:
        entries = groups[qid].to_dict('records')
        query_seed = int.from_bytes(hashlib.sha256((str(SEED) + '\0' + str(qid)).encode()).digest()[:8], 'big')
        rng = random.Random(query_seed)
        picked = rng.sample(entries, 32)
        rng.shuffle(picked)
        items = []
        for i, e in enumerate(picked, 1):
            product_id = int(e['product_id'])
            items.append({'id': i, 'text': run_wands.make_text(products[product_id]), 'grade': e['label'],
                          'gold': 'Y' if e['label'] == 'Exact' else 'N', 'example_id': str(e['id']),
                          'product_id': str(product_id), 'join_missing': False})
        rows.append({'query_id': str(qid), 'query': str(queries.at[qid, 'query']), 'items': items})
    heldout_queries = {r['query'].strip().casefold() for r in rows}
    heldout_products = {int(it['product_id']) for r in rows for it in r['items']}
    assert len(heldout_queries) == 160 and not heldout_queries & (prior_wands_text | prior_esci_text)
    assert not heldout_products & excluded_products
    metadata = {'protocol': 'PROTOCOL_wands_holdout.md', 'source': 'Wayfair WANDS',
                'revision': '3b74dcf4ba29ab8ff3e6a50b5b09fc627cb882b5', 'seed': SEED,
                'source_hashes': run_wands.EXPECTED, 'source_query_count': len(q), 'source_label_rows': len(l),
                'source_product_count': len(p), 'conflicting_query_product_groups_removed': len(conflicts),
                'prior_wands_sampled_product_ids_excluded': len(excluded_products), 'remaining_query_ids': len(allowed),
                'eligible_query_ids_after_product_exclusion': len(eligible), 'sample_size': len(rows),
                'candidate_count': sum(len(r['items']) for r in rows), 'candidates_per_pool': 32,
                'unique_sampled_product_ids': len(heldout_products), 'prior_wands_sampled_product_overlap': 0,
                'prior_query_text_overlap': 0, 'grade_counts': dict(collections.Counter(it['grade'] for r in rows for it in r['items'])),
                'label_mapping': 'Exact=Y; Partial/Irrelevant=N', 'system_instruction': run_wands.SYSTEM}
    return rows, metadata

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--wands-dir', type=Path, default=Path('/home/gpu/intent2s_wands_official_2026'))
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    rows, metadata = sample(a.wands_dir)
    with (a.out / 'inputs.jsonl').open('x', encoding='utf8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    (a.out / 'selection.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps(metadata, indent=2), flush=True)
if __name__ == '__main__':
    main()
