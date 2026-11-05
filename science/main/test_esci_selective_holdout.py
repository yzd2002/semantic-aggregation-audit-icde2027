#!/usr/bin/env python3
"""Exercise variable-sized chunks, selective routing and label isolation."""
import copy
import json

from analyze_esci_selective_holdout import rebuild
from order_fallback import combine
from run_esci_selective_holdout import point_prompt
from run_experiment import p_spec
from run_full_pools import chunks
from run_keyed_order_audit import keyed_prompt, parse_keyed


def keyed(labels):
    return '{' + ','.join(json.dumps(str(i)) + ':' + json.dumps(x) for i, x in enumerate(labels, 1)) + '}'


def event(q, start, size, text, prompt_hash='same'):
    return {'query_id': q, 'start': start, 'size': size, 'final': text, 'prompt_sha256': prompt_hash}


row = {'query_id': 'test', 'query': 'blue item', 'items': [
    {'id': i+1, 'text': 'blue item ' + str(i), 'grade': 'E', 'gold': 'Y'} for i in range(10)]}
altered = copy.deepcopy(row)
for item in altered['items']:
    item['grade'] = 'DO-NOT-INCLUDE-GRADE'
    item['gold'] = 'N'
spec = {'product_type': 'item', 'required': ['blue'], 'preferred': []}
assert p_spec(row) == p_spec(altered)
for left, right in zip(chunks(row), chunks(altered)):
    for reverse in (False, True):
        l = {**left, 'items': list(reversed(left['items'])) if reverse else left['items']}
        r = {**right, 'items': list(reversed(right['items'])) if reverse else right['items']}
        text = keyed_prompt(l, spec)
        assert text == keyed_prompt(r, spec) and 'DO-NOT-INCLUDE-GRADE' not in text
for a, b in zip(row['items'], altered['items']):
    assert point_prompt(row, a, spec) == point_prompt(altered, b, spec)

events = {'original': [event('test', 0, 8, keyed('YYYYYYYY')), event('test', 8, 2, keyed('YY'))],
          'stable_reverse': [event('test', 0, 8, keyed('YNYYYYYY')), event('test', 8, 2, keyed('NY'))],
          'replay': [event('test', 0, 8, keyed('YYYYYYYY')), event('test', 8, 2, keyed('YY'))],
          'selective_pointwise': [event('test', 1, 1, 'N'), event('test', 8, 1, 'Y')],
          'pointwise': [event('test', i, 1, 'N' if i == 1 else 'Y') for i in range(10)]}
pred, need, matched, changed = rebuild([row], events)
assert need == [('test', 1), ('test', 8)] and matched == 2 and changed == 0
assert pred['actual_fallback'][0] == ['Y', 'N'] + ['Y']*8
assert pred['pointwise'][0] == pred['actual_fallback'][0]
assert combine(['Y'], ['N'], [None]) is None
assert parse_keyed('{"1":"Y","1":"N"}', [1]) is None
assert parse_keyed('{"1":"Y","2":"N"}', [1]) is None
bad = copy.deepcopy(events)
bad['stable_reverse'][1]['final'] = '{"1":"Y"}'
bad['selective_pointwise'] = bad['selective_pointwise'][:1]
pred, need, _, _ = rebuild([row], bad)
assert need == [('test', 1)] and pred['stable_reverse'][0] is None and pred['actual_fallback'][0] is None
try:
    bad['selective_pointwise'] = []
    rebuild([row], bad)
    raise AssertionError('Missing actual disagreement request was accepted.')
except ValueError:
    pass
print('Variable-size chunks, stable IDs, label isolation, fallback, failures and missing-request rejection passed.')
