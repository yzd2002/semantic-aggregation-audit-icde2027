#!/usr/bin/env python3
"""Independently rebuild the frozen new-query ESCI selection from source."""
import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXAMPLES_SHA = '4a735b693b4a424a6fc67f5be6e4c811495c488bbf66d02a602d308b2744263a'
PRODUCTS_SHA = '25124442d064d64b26f74082d6fa09438d679efc0c183cf28d19064a2b65a265'
FIELDS = ('product_title', 'product_description', 'product_bullet_point', 'product_brand', 'product_color')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    with path.open(encoding='utf8') as f:
        return [json.loads(line) for line in f]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--examples', type=Path, required=True)
    p.add_argument('--products', type=Path, required=True)
    p.add_argument('--selection', type=Path, required=True)
    a = p.parse_args()
    if sha(a.examples) != EXAMPLES_SHA or sha(a.products) != PRODUCTS_SHA:
        raise ValueError('Official source hash mismatch.')
    old_queries, old_products = set(), set()
    for folder in ('run_test_160', 'run_validation_500', 'run_full_pools_250'):
        for row in read(HERE / folder / 'inputs.jsonl'):
            old_queries.add(row['query'].strip().casefold())
            for item in row['items']:
                if 'product_id' in item:
                    old_products.add(str(item['product_id']))
    for row in read(HERE / 'run_supervised_baseline_1000/train_manifest.jsonl'):
        old_queries.add(row['query'].strip().casefold())
        old_products.add(str(row['product_id']))

    import pyarrow.dataset as ds
    source = ds.dataset(a.examples, format='parquet')
    table = source.to_table(columns=['example_id', 'query_id', 'query', 'product_id', 'esci_label'],
        filter=(ds.field('product_locale') == 'us') & (ds.field('small_version') == 1) & (ds.field('split') == 'test'))
    grouped = defaultdict(list)
    seen = set()
    for row in table.to_pylist():
        if row['query_id'] is None or row['example_id'] is None or row['example_id'] in seen:
            raise ValueError('Ambiguous source keys.')
        seen.add(row['example_id'])
        grouped[int(row['query_id'])].append(row)
    eligible = []
    norms = set()
    for qid, group in grouped.items():
        if any(x['query'] is None or x['product_id'] is None for x in group):
            continue
        keys = {str(x['query']).strip().casefold() for x in group}
        if len(keys) != 1:
            continue
        key = next(iter(keys))
        if not key or key in old_queries or any(str(x['product_id']) in old_products for x in group):
            continue
        if key in norms:
            raise ValueError('Normalized eligible queries are not unique.')
        norms.add(key)
        eligible.append(qid)
    eligible.sort()
    if len(eligible) != 3963:
        raise ValueError('Eligible count differs from frozen protocol.')
    chosen = random.Random(20271201).sample(eligible, 250)
    observed = read(a.selection / 'inputs.jsonl')
    meta = json.loads((a.selection / 'selection.json').read_text(encoding='utf8'))
    if meta['inputs_sha256'] != sha(a.selection / 'inputs.jsonl') or len(observed) != 250:
        raise ValueError('Selection file hash or length differs.')
    pids = {str(x['product_id']) for q in chosen for x in grouped[q]}
    products = ds.dataset(a.products, format='parquet').to_table(
        columns=['product_id', 'product_locale', *FIELDS],
        filter=(ds.field('product_locale') == 'us') & ds.field('product_id').isin(sorted(pids))).to_pylist()
    product_map = {}
    for row in products:
        pid = str(row['product_id'])
        if pid in product_map:
            raise ValueError('Duplicate source product ID.')
        product_map[pid] = row
    if set(product_map) != pids:
        raise ValueError('Source product joins incomplete.')
    count = 0
    for expected_qid, result in zip(chosen, observed):
        group = sorted(grouped[expected_qid], key=lambda x: x['example_id'])
        if result['query_id'] != str(expected_qid) or result['query'] != group[0]['query']:
            raise ValueError('Selected query or order differs.')
        indices = list(range(len(group)))
        seed = int.from_bytes(hashlib.sha256(('20271201\0' + str(expected_qid)).encode()).digest()[:8], 'big')
        random.Random(seed).shuffle(indices)
        if len(result['items']) != len(indices):
            raise ValueError('A selected pool does not contain all labeled examples.')
        for pos, (index, item) in enumerate(zip(indices, result['items']), 1):
            source_row = group[index]
            grade = source_row['esci_label']
            if grade not in ('E', 'S', 'C', 'I'):
                raise ValueError('Unexpected selected grade.')
            product = product_map[str(source_row['product_id'])]
            text = ' '.join(product[k] for k in FIELDS if isinstance(product[k], str) and product[k])[:600]
            expected = {'id': pos, 'example_id': str(source_row['example_id']),
                        'product_id': str(source_row['product_id']), 'text': text, 'grade': grade,
                        'gold': 'Y' if grade in ('E', 'S') else 'N'}
            if not text or item != expected:
                raise ValueError('Source example, text, label or position differs.')
            count += 1
    if meta['candidates'] != count or meta['distinct_product_ids'] != len(pids):
        raise ValueError('Selection summary differs.')
    print('Independently reconstructed', len(observed), 'queries and', count,
          'source examples, joins, labels, order and previous-sample exclusions.')


if __name__ == '__main__':
    main()
