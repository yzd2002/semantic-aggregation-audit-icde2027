#!/usr/bin/env python3
"""Select previously unused ESCI pools under the frozen follow-up protocol."""
import argparse
import hashlib
import json
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROTOCOL = 'PROTOCOL_esci_selective_holdout.md'
SEED = 20271201
N = 250
EXAMPLES_SHA = '4a735b693b4a424a6fc67f5be6e4c811495c488bbf66d02a602d308b2744263a'
PRODUCTS_SHA = '25124442d064d64b26f74082d6fa09438d679efc0c183cf28d19064a2b65a265'
FIELDS = ('product_title', 'product_description', 'product_bullet_point', 'product_brand', 'product_color')
PRIOR = ('run_test_160', 'run_validation_500', 'run_full_pools_250')
TRAIN = 'run_supervised_baseline_1000/train_manifest.jsonl'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def lines(path):
    with path.open(encoding='utf8') as f:
        return [json.loads(line) for line in f]


def normalize(s):
    return str(s).strip().casefold()


def sample(examples, products):
    if sha(examples) != EXAMPLES_SHA or sha(products) != PRODUCTS_SHA:
        raise ValueError('Pinned source Parquet hash differs.')
    prior_queries, prior_products = set(), set()
    for folder in PRIOR:
        for row in lines(HERE / folder / 'inputs.jsonl'):
            prior_queries.add(normalize(row['query']))
            prior_products.update(str(x['product_id']) for x in row['items'] if 'product_id' in x)
    for row in lines(HERE / TRAIN):
        prior_queries.add(normalize(row['query']))
        prior_products.add(str(row['product_id']))

    import pandas as pd
    import pyarrow.dataset as ds
    ex = pd.read_parquet(examples, columns=['example_id', 'query', 'query_id', 'product_id',
        'product_locale', 'esci_label', 'small_version', 'split'],
        filters=[('product_locale', '=', 'us'), ('small_version', '=', 1), ('split', '=', 'test')])
    if ex['query_id'].isna().any() or ex['example_id'].isna().any() or ex['example_id'].duplicated().any():
        raise ValueError('Ambiguous or missing source query/example keys.')
    bad_ids = set(ex.loc[ex['query'].isna() | ex['product_id'].isna(), 'query_id'])
    ex = ex[~ex['query_id'].isin(bad_ids)].copy()
    ex['norm'] = ex['query'].astype(str).str.strip().str.casefold()
    grouped = ex.groupby('query_id', sort=False)
    meta = grouped.agg(norm=('norm', 'first'), norms=('norm', 'nunique'),
        has_prior_product=('product_id', lambda x: any(str(v) in prior_products for v in x)))
    meta = meta[(meta['norms'] == 1) & (meta['norm'] != '') &
                ~meta['norm'].isin(prior_queries) & ~meta['has_prior_product']]
    if len(meta) != 3963 or meta['norm'].nunique() != len(meta):
        raise ValueError('Eligible count or query-string uniqueness differs from feasibility check.')
    eligible = sorted(int(x) for x in meta.index.tolist())
    chosen = random.Random(SEED).sample(eligible, N)
    selected = ex[ex['query_id'].isin(chosen)].copy()
    if not selected['esci_label'].isin(['E', 'S', 'C', 'I']).all():
        raise ValueError('Unsupported grade in a selected pool; do not replace queries.')
    product_ids = sorted(set(str(x) for x in selected['product_id']))
    dataset = ds.dataset(products, format='parquet')
    product_rows = dataset.to_table(columns=['product_id', 'product_locale', *FIELDS],
        filter=(ds.field('product_locale') == 'us') & ds.field('product_id').isin(product_ids)).to_pandas()
    if product_rows.duplicated(['product_id', 'product_locale']).any():
        raise ValueError('Ambiguous product key.')
    joined = selected.merge(product_rows, on=['product_id', 'product_locale'], how='left',
                            indicator=True, validate='many_to_one')
    if not (joined['_merge'] == 'both').all():
        raise ValueError('Missing product join; do not replace queries.')
    rows = []
    for qid in chosen:
        part = joined[joined['query_id'] == qid].sort_values('example_id')
        if part.empty or part['norm'].nunique() != 1:
            raise ValueError('Selected pool is missing or has ambiguous query text.')
        items = []
        for _, item in part.iterrows():
            text = ' '.join(item[k] for k in FIELDS if isinstance(item[k], str) and item[k])[:600]
            if not text:
                raise ValueError('Empty product text; do not replace queries.')
            grade = str(item['esci_label'])
            items.append({'example_id': str(item['example_id']), 'product_id': str(item['product_id']),
                          'text': text, 'grade': grade, 'gold': 'Y' if grade in ('E', 'S') else 'N'})
        seed = int.from_bytes(hashlib.sha256((str(SEED) + '\0' + str(qid)).encode()).digest()[:8], 'big')
        random.Random(seed).shuffle(items)
        for i, item in enumerate(items, 1):
            item['id'] = i
        rows.append({'query_id': str(qid), 'query': str(part['query'].iloc[0]), 'items': items})
    info = {'protocol': PROTOCOL, 'seed': SEED, 'eligible_query_ids': len(eligible),
            'selected_queries': len(rows), 'candidates': sum(len(r['items']) for r in rows),
            'minimum_pool_size': min(len(r['items']) for r in rows),
            'maximum_pool_size': max(len(r['items']) for r in rows),
            'distinct_product_ids': len({x['product_id'] for r in rows for x in r['items']}),
            'excluded_prior_normalized_queries': len(prior_queries), 'excluded_prior_product_ids': len(prior_products),
            'examples_sha256': EXAMPLES_SHA, 'products_sha256': PRODUCTS_SHA}
    return rows, info


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--examples', type=Path, required=True)
    ap.add_argument('--products', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    if a.out.exists():
        raise FileExistsError('Use a new output directory.')
    rows, info = sample(a.examples, a.products)
    a.out.mkdir(parents=True, exist_ok=False)
    payload = ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows)
    (a.out / 'inputs.jsonl').write_text(payload, encoding='utf8')
    info['inputs_sha256'] = sha(a.out / 'inputs.jsonl')
    (a.out / 'selection.json').write_text(json.dumps(info, indent=2) + '\n', encoding='utf8')
    print('Selected', len(rows), 'previously unused query pools;', info['candidates'],
          'candidates. Labels were not summarized.')


if __name__ == '__main__':
    main()
