#!/usr/bin/env python3
"""Check chunk-local ID mapping and sharpness under coupled duplicate references."""
import itertools
import json

from run_experiment import parse_batch
from run_full_pools import chunks

# IDs restart within each chunk; the chunk start is required to recover pool positions.
row = {'query': 'example', 'items': [{'id': i+1, 'text': 'same product' if i in (0, 8) else f'product {i}',
                                     'gold': 'Y' if i % 2 else 'N'} for i in range(11)]}
parts = chunks(row)
assert [(x['start'], len(x['items'])) for x in parts] == [(0, 8), (8, 3)]
assert [[x['id'] for x in part['items']] for part in parts] == [list(range(1, 9)), list(range(1, 4))]
actual = [None] * len(row['items'])
for part in parts:
    labels = [x['gold'] for x in part['items']]
    answer = json.dumps([{'id': i+1, 'label': y} for i, y in reversed(list(enumerate(labels)))])
    parsed = parse_batch(answer, len(labels))
    assert parsed == labels
    for local, label in enumerate(parsed, 1):
        absolute = part['start'] + local - 1
        assert actual[absolute] is None
        actual[absolute] = label
assert actual == [x['gold'] for x in row['items']]
assert parts[0]['items'][0]['id'] == parts[1]['items'][0]['id'] == 1
assert parse_batch('[{"id":1,"label":"Y"},{"id":1,"label":"N"}]', 2) is None
assert parse_batch('[{"id":1,"label":"Y"}]', 2) is None

# Conditional interval remains valid under coupled duplicates, but need not be sharp.
# n=2, one accepted Y, no accepted-label error: unrestricted references attain
# counts 1 and 2; identical-reference duplicates only allow count 2.
all_refs = list(itertools.product((0, 1), repeat=2))
admissible = [r for r in all_refs if r[0] == 1]
coupled = [r for r in admissible if r[0] == r[1]]
assert {sum(r) for r in admissible} == {1, 2}
assert {sum(r) for r in coupled} == {2}

# Disagreeing all-Y and all-N outputs have the sharp 1/2 maximum-error floor
# over unrestricted references, but a same-reference duplicate constraint
# raises the minimum attainable maximum error to 1.
a, b = (1, 1), (0, 0)
def worst(ref):
    return max(sum(x != y for x, y in zip(a, ref)), sum(x != y for x, y in zip(b, ref))) / 2
assert min(worst(r) for r in all_refs) == .5
assert min(worst(r) for r in all_refs if r[0] == r[1]) == 1.
print('Chunk-local IDs, pool mapping and duplicate-coupling sharpness counterexamples passed.')
