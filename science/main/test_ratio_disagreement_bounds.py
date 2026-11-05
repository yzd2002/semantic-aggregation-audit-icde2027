#!/usr/bin/env python3
"""Exhaustively check single-pool and equal-size macro ratio-error floors."""
import itertools
from fractions import Fraction

from ratio_disagreement_bounds import (attaining_reference_counts,
                                       macro_ratio_mae_floor,
                                       pool_ratio_error_floor)

for n in range(1, 9):
    for a, b in itertools.product(range(n + 1), repeat=2):
        errors = [Fraction(max(abs(a - s), abs(b - s)), n) for s in range(n + 1)]
        assert min(errors) == pool_ratio_error_floor(n, a, b)

cases = 0
for n, qmax in ((1, 3), (2, 3), (3, 3), (4, 2)):
    for q in range(1, qmax + 1):
        for flat in itertools.product(range(n + 1), repeat=2 * q):
            pairs = list(zip(flat[::2], flat[1::2]))
            expected = min(max(sum(abs(a - s) for (a, _), s in zip(pairs, refs)),
                               sum(abs(b - s) for (_, b), s in zip(pairs, refs)))
                           for refs in itertools.product(range(n + 1), repeat=q))
            floor = macro_ratio_mae_floor(n, pairs)
            assert floor == Fraction(expected, q * n)
            witness = attaining_reference_counts(n, pairs)
            observed = max(sum(abs(a - s) for (a, _), s in zip(pairs, witness)),
                           sum(abs(b - s) for (_, b), s in zip(pairs, witness)))
            assert Fraction(observed, q * n) == floor
            cases += 1

for call in (lambda: pool_ratio_error_floor(0, 0, 0),
             lambda: pool_ratio_error_floor(3, True, 0),
             lambda: pool_ratio_error_floor(3, 4, 0),
             lambda: macro_ratio_mae_floor(3, []),
             lambda: macro_ratio_mae_floor(3, [(0, -1)])):
    try:
        call()
    except ValueError:
        pass
    else:
        raise AssertionError('Invalid bound input was accepted.')
print('Single-pool floors and', cases, 'equal-size multi-pool cases exhaustively verified, including attainable witnesses.')
