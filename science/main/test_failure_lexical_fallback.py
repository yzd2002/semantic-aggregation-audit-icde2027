#!/usr/bin/env python3
"""Check fallback source selection, complete outputs and strict chunk boundaries."""
from copy import deepcopy

from analyze_failure_lexical_fallback import lexical, measure, policies, recover

row = {'query': 'red mug', 'items': [{'text': x, 'gold': 'Y'} for x in
       ('red mug', 'red plate', 'red mug cup', 'blue mug')]}
plan = {'pred': None, 'raw': [{'start': 0, 'size': 2, 'final': 'NN'},
                              {'start': 2, 'size': 2, 'final': 'Y?'}]}
x = policies(row, plan)
assert x['failed'] and x['recovered'] == ['N', 'N', None, None]
assert x['lexical'] == ['Y', 'N', 'Y', 'N']
assert x['whole'] == ['Y', 'N', 'Y', 'N'] and x['partial'] == ['N', 'N', 'Y', 'N']
altered = deepcopy(row)
for item in altered['items']: item['gold'] = 'N'
assert lexical(altered) == lexical(row) and policies(altered, plan) == x
complete = {'pred': ['N', 'N', 'Y', 'N'], 'raw': [{'start': 0, 'size': 2, 'final': 'NN'},
                                                 {'start': 2, 'size': 2, 'final': 'YN'}]}
y = policies(row, complete)
assert not y['failed'] and y['whole'] == y['partial'] == complete['pred']
for bad in ([{'start': 0, 'size': 2, 'final': 'YN'}, {'start': 1, 'size': 3, 'final': 'YNN'}],
            [{'start': 0, 'size': 2, 'final': 'YN'}],
            [{'start': 0, 'size': 5, 'final': 'YNNNN'}]):
    try: recover(4, bad)
    except ValueError: pass
    else: raise AssertionError('invalid chunk partition accepted')
assert recover(4, [{'start': 0, 'size': 4, 'final': 'YNNN.'}]) == [None] * 4
try: policies(row, {**complete, 'pred': ['Y'] * 4})
except ValueError: pass
else: raise AssertionError('inconsistent saved complete prediction accepted')
m, errors = measure([['Y', 'N', 'Y', 'N']], [['Y', 'N', 'N', 'N']])
assert (m['tp'], m['tn'], m['fp'], m['fn']) == (1, 2, 0, 1) and errors.tolist() == [.25]
print('Partial and whole fallback, label isolation, complete-pool consistency and chunk boundaries passed.')
