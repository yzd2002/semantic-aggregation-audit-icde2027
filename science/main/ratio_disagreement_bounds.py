#!/usr/bin/env python3
"""Sharp deterministic ratio-error floors for two fixed binary outputs."""
from fractions import Fraction


def _validate_count(value, n):
    if type(value) is not int or not 0 <= value <= n:
        raise ValueError('Positive counts must be integers between zero and pool size.')


def _validate_n(n):
    if type(n) is not int or n <= 0:
        raise ValueError('Pool size must be a positive integer.')


def pool_ratio_error_floor(n, left_positives, right_positives):
    """Sharp floor on the worse absolute ratio error in one fixed pool."""
    _validate_n(n)
    _validate_count(left_positives, n)
    _validate_count(right_positives, n)
    delta = abs(left_positives - right_positives)
    return Fraction((delta + 1) // 2, n)


def macro_ratio_mae_floor(n, count_pairs):
    """Sharp floor on the worse query-macro MAE for equally sized pools."""
    _validate_n(n)
    pairs = list(count_pairs)
    if not pairs:
        raise ValueError('At least one equal-sized pool is required.')
    for a, b in pairs:
        _validate_count(a, n)
        _validate_count(b, n)
    total_gap = sum(abs(a - b) for a, b in pairs)
    return Fraction((total_gap + 1) // 2, n * len(pairs))


def attaining_reference_counts(n, count_pairs):
    """Construct one unconstrained reference-count witness for the macro floor."""
    pairs = list(count_pairs)
    macro_ratio_mae_floor(n, pairs)
    remaining = (sum(abs(a - b) for a, b in pairs) + 1) // 2
    references = []
    for a, b in pairs:
        allocated = min(remaining, abs(a - b))
        references.append(a + allocated if a <= b else a - allocated)
        remaining -= allocated
    assert remaining == 0
    return references
