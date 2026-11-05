#!/usr/bin/env python3
"""Check paired query aggregation against a small hand-calculated example."""
import numpy as np
from analyze_order_fallback_validation import paired

rows = [{'items': [{'gold': 'Y'}, {'gold': 'N'}]},
        {'items': [{'gold': 'Y'}, {'gold': 'Y'}]}]
preds = {'base': [['Y', 'Y'], ['N', 'Y']], 'fallback': [['Y', 'N'], ['Y', 'Y']]}
draw = np.array([[0, 1], [0, 0], [1, 1]])
result = paired(rows, preds, [0, 1], 'base', 'fallback', draw)
assert abs(result['ratio_mae_difference'] + .5) < 1e-12
assert abs(result['f1_difference'] - (1/3)) < 1e-12
assert result['ratio_mae_difference_ci'] == [-.5, -.5]
assert result['left'] == 'base' and result['right'] == 'fallback'
print('Paired query-macro MAE and pooled F1 agree with a hand-calculated example.')
