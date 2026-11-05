#!/usr/bin/env python3
"""Synthetic edge cases for partial-aggregate decomposition and abstention."""
from analyze_partial_aggregate_bias import measure, reconstruct, summarize

same = measure(['Y', 'N', 'Y', 'N'], ['Y', 'N', None, None])
assert same['n'] == 4 and same['m'] == 2 and same['reference_ratio'] == .5
assert same['available_error'] == 0 and same['zero_fill_error'] == -.25
cancel = measure(['Y', 'Y', 'N', 'N'], ['Y', 'N', None, None])
assert cancel['available_error'] == 0 and cancel['observed_composition_term'] == .5
assert cancel['accepted_discrepancy_term'] == -.5 and cancel['zero_fill_error'] == -.25
missing = measure(['Y', 'N'], [None, None])
assert missing['available_ratio'] is None and missing['zero_fill_error'] == -.5
assert summarize(1, [missing])['available_only_mae'] is None
row = {'items': [None] * 4}
entry = {'raw': [{'start': 0, 'size': 2, 'final': 'YN'}, {'start': 2, 'size': 2, 'final': 'Y?'}]}
assert reconstruct(row, entry) == ['Y', 'N', None, None]
entry['raw'][1]['start'] = 1
try:
    reconstruct(row, entry)
except ValueError:
    pass
else:
    raise AssertionError('Overlapping chunks must fail validation')
print('Synthetic empty, cancellation, zero-fill and chunk-boundary cases passed.')
