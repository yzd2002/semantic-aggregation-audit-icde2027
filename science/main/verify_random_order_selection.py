#!/usr/bin/env python3
"""Independently regenerate each saved random-order mapping."""
import argparse
import hashlib
import json
import random
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--selection', type=Path, required=True)
    p.add_argument('--orders', type=Path, required=True)
    a = p.parse_args()
    source, data = a.selection / 'inputs.jsonl', a.orders / 'mappings.jsonl'
    rows = [json.loads(x) for x in source.read_text(encoding='utf8').splitlines()]
    saved = [json.loads(x) for x in data.read_text(encoding='utf8').splitlines()]
    summary = json.loads((a.orders / 'selection.json').read_text())
    assert len(rows) == len(saved) == summary['queries'] == 160
    assert hashlib.sha256(source.read_bytes()).hexdigest() == summary['source_sha256']
    assert hashlib.sha256(data.read_bytes()).hexdigest() == summary['mappings_sha256']
    identity = repeated = chunks = 0
    for row, rec in zip(rows, saved):
        assert row['query_id'] == rec['query_id'] and len(row['items']) == 32 and len(rec['chunks']) == 4
        for chunk, start in zip(rec['chunks'], range(0, 32, 8)):
            assert chunk['start'] == start
            generated = []
            for rep in (1, 2, 3):
                value = f"ICDE2027-random-order-v1\0{row['query_id']}\0{start}\0{rep}".encode('utf8')
                rng = random.Random(int.from_bytes(hashlib.sha256(value).digest()[:8], 'big'))
                positions = list(range(8))
                for i in range(7, 0, -1):
                    j = rng.randrange(i + 1)
                    positions[i], positions[j] = positions[j], positions[i]
                assert positions == chunk['orders'][f'perm_{rep}']
                generated.append(tuple(positions))
                identity += positions == list(range(8))
            repeated += 3 - len(set(generated))
            chunks += 1
    assert chunks == summary['chunks'] == 640 and summary['permutations_per_chunk'] == 3
    assert identity == summary['identity_orders'] and repeated == summary['duplicate_orders_within_chunk']
    print('Independently regenerated', chunks * 3, 'order mappings for', len(rows), 'queries.')


if __name__ == '__main__':
    main()
