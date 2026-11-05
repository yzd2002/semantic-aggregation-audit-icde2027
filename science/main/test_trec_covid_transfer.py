#!/usr/bin/env python3
"""Test prompt isolation, identifier reversal, strict parsing and partial fallbacks."""
from copy import deepcopy

from analyze_trec_covid_transfer import lexical, metrics
from order_fallback import combine
from run_keyed_order_audit import keyed_regex, parse_keyed
from run_trec_covid_transfer import keyed_prompt, point_prompt

row = {'topic_id': 1, 'query': 'virus origin', 'question': 'Where did it originate?',
       'narrative': 'Evidence about the origin.',
       'items': [{'doc_id': str(i), 'title': 'virus origin' if i % 2 == 0 else 'unrelated',
                  'abstract': 'A document.', 'grade': 2 if i < 4 else 0,
                  'gold': 'Y' if i < 4 else 'N'} for i in range(8)]}
forward, forward_ids = keyed_prompt(row, 0)
backward, backward_ids = keyed_prompt(row, 0, True)
assert forward_ids == list(range(1, 9)) and backward_ids == list(range(8, 0, -1))
assert forward != backward and '"id": 1' in forward
altered = deepcopy(row)
for item in altered['items']:
    item['doc_id'] = 'secret-target-id'
    item['grade'] = 12345
    item['gold'] = 'secret-target-label'
assert keyed_prompt(altered, 0)[0] == forward and point_prompt(altered, 0) == point_prompt(row, 0)
valid = '{' + ','.join('"%d":"%s"' % (i, 'Y' if i % 2 else 'N') for i in range(1, 9)) + '}'
assert parse_keyed(valid, list(range(1, 9))) == {i: 'Y' if i % 2 else 'N' for i in range(1, 9)}
assert parse_keyed(' ' + valid, list(range(1, 9))) is None
assert '[YN]' in keyed_regex(list(range(1, 9)))
assert combine(['Y', 'N'], ['Y', 'Y'], [None, 'N']) == ['Y', 'N']
assert combine(['Y', 'N'], ['Y', 'Y'], [None, None]) is None
assert lexical(row) == ['Y', 'N'] * 4
m = metrics([row], [['Y'] * 8])
assert m['tp'] == 4 and m['fp'] == 4 and m['ratio_mae'] == .5
print('Prompt target isolation, stable reversal, strict parsing, fallback and metrics tests passed.')
