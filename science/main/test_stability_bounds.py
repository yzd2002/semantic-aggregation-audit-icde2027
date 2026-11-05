#!/usr/bin/env python3
"""Exhaustively check the disagreement floor for small binary pools."""
from itertools import product
from stability_bounds import stability_error_floor

for n in range(1, 7):
    vectors = list(product('YN', repeat=n))
    for left in vectors:
        for right in vectors:
            d = sum(a != b for a, b in zip(left, right))
            result = stability_error_floor(left, right)
            assert result['disagreements'] == d and result['n'] == n
            assert result['disagreement_rate'] == d / n and result['average_error_lower_bound'] == d / (2 * n)
            assert result['maximum_error_lower_bound'] == ((d + 1) // 2) / n
            best_maximum = n + 1
            for reference in vectors:
                el = sum(a != s for a, s in zip(left, reference))
                er = sum(b != s for b, s in zip(right, reference))
                assert el + er >= d and max(el, er) >= (d + 1) // 2
                best_maximum = min(best_maximum, max(el, er))
            assert best_maximum == (d + 1) // 2
            # The sum bound is attained by choosing reference equal to left.
            assert sum(b != s for b, s in zip(right, left)) == d
for left, right in [([], []), (['Y'], []), (['?'], ['Y'])]:
    try:
        stability_error_floor(left, right)
    except ValueError:
        pass
    else:
        raise AssertionError('Invalid labels or lengths must be rejected.')
print('Sharp sum and maximum disagreement error floors verified exhaustively for all binary vectors with n<=6.')
