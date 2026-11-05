#!/usr/bin/env python3
"""Fix three seed-derived within-chunk orderings for the saved WANDS holdout."""
import argparse
import hashlib
import json
import random
from pathlib import Path

PREFIX = 'ICDE2027-random-order-v1'


def mapping(qid, start, replicate):
    seed_text = f'{PREFIX}\0{qid}\0{start}\0{replicate}'
    seed = int.from_bytes(hashlib.sha256(seed_text.encode('utf8')).digest()[:8], 'big')
    positions = list(range(8))
    random.Random(seed).shuffle(positions)
    return positions


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--selection', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        raise FileExistsError('Choose a new output directory.')
    source = a.selection / 'inputs.jsonl'
    rows = [json.loads(line) for line in source.read_text(encoding='utf8').splitlines()]
    if len(rows) != 160 or any(len(r['items']) != 32 for r in rows):
        raise ValueError('Unexpected holdout selection.')
    records, identities, repeated = [], 0, 0
    for row in rows:
        chunks = []
        for start in range(0, 32, 8):
            orders = [mapping(row['query_id'], start, i) for i in (1, 2, 3)]
            identities += sum(x == list(range(8)) for x in orders)
            repeated += 3 - len({tuple(x) for x in orders})
            chunks.append({'start': start, 'orders': {f'perm_{i}': x for i, x in enumerate(orders, 1)}})
        records.append({'query_id': row['query_id'], 'chunks': chunks})
    a.out.mkdir(parents=True, exist_ok=False)
    data = a.out / 'mappings.jsonl'
    with data.open('w', encoding='utf8') as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    summary = {'protocol': 'PROTOCOL_random_order_sensitivity.md', 'queries': len(records),
        'chunks': len(records) * 4, 'permutations_per_chunk': 3,
        'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'mappings_sha256': hashlib.sha256(data.read_bytes()).hexdigest(),
        'identity_orders': identities, 'duplicate_orders_within_chunk': repeated,
        'seed_prefix': PREFIX, 'python_rng': 'random.Random.shuffle'}
    (a.out / 'selection.json').write_text(json.dumps(summary, indent=2) + '\n')
    print('Fixed', len(records), 'queries,', summary['chunks'], 'chunks and',
          summary['chunks'] * 3, 'seeded orders.')


if __name__ == '__main__':
    main()
