#!/usr/bin/env python3
"""Check keyed parsing and the distinction between the two compact mappings."""
import json
from analyze_keyed_order_audit import counterfactual_numeric_id_order
from run_keyed_order_audit import keyed_prompt, keyed_regex, parse_keyed

ids = list(range(1, 9))
answer = '{' + ','.join('"%d":"%s"' % (i, 'Y' if i % 2 else 'N') for i in ids) + '}'
assert parse_keyed(answer, ids) == {i: ('Y' if i % 2 else 'N') for i in ids}
for invalid in [answer[:-1], answer.replace('"1":"Y"', '"1":"X"'),
                answer.replace('"2":"N"', '"1":"N"'), answer.replace(',', ', ', 1), 'BANANA']:
    assert parse_keyed(invalid, ids) is None
try:
    keyed_regex([1, 1, 2])
    raise AssertionError('Duplicate IDs should fail')
except ValueError:
    pass
part = {'query': 'test', 'items': [{'id': i, 'text': 'product %d' % i} for i in ids]}
reverse = {**part, 'items': list(reversed(part['items']))}
assert keyed_prompt(part, None) != keyed_prompt(reverse, None)
assert 'Use the given ids, not positions' in keyed_prompt(reverse, None)
record = {'raw': [{'start': 0, 'size': 8, 'original_indices_in_prompt_order': list(range(7, -1, -1)),
                   'final': 'YNNNNNNN'}]}
assert counterfactual_numeric_id_order(record, 8) == list('YNNNNNNN')
record['raw'][0]['original_indices_in_prompt_order'] = list(range(8))
try:
    counterfactual_numeric_id_order(record, 8)
    raise AssertionError('Nonreversed prior mapping should fail')
except ValueError:
    pass
assert json.loads(answer)['1'] == 'Y'
print('Keyed format, parser rejection, stable IDs and counterfactual mapping tests passed.')
