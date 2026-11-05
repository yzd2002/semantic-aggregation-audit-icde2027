#!/usr/bin/env python3
"""Sharp finite-pool ratio bounds for duplicate groups with a shared reference."""
from fractions import Fraction


def coupled_ratio_bounds(groups, error_budget):
    """Return sharp lower/upper Fractions, or None if the assumptions conflict.

    Each (accepted_Y, accepted_N, missing) triple describes one nonempty group
    whose occurrences share a binary reference label for this fixed query.
    The total number of accepted-label errors is at most error_budget.
    """
    if not isinstance(groups, (list, tuple)) or not groups:
        raise ValueError('groups must be a nonempty sequence')
    checked = []
    for group in groups:
        if not isinstance(group, (list, tuple)) or len(group) != 3:
            raise ValueError('each group must contain three counts')
        if any(type(x) is not int for x in group):
            raise TypeError('group counts must be integers, not booleans')
        if min(group) < 0 or sum(group) == 0:
            raise ValueError('group counts must be nonnegative with positive total')
        checked.append(tuple(group))
    if type(error_budget) is not int:
        raise TypeError('error budget must be an integer, not a boolean')
    n = sum(sum(g) for g in checked)
    m = sum(a + b for a, b, _ in checked)
    if not 0 <= error_budget <= m:
        raise ValueError('error budget outside 0..number of accepted labels')

    # A state stores the smallest/largest reference-positive count attainable
    # at exactly e accepted-label errors after processing the current groups.
    lower = [None] * (error_budget + 1)
    upper = [None] * (error_budget + 1)
    lower[0] = upper[0] = 0
    for accepted_y, accepted_n, missing in checked:
        group_size = accepted_y + accepted_n + missing
        next_lower = [None] * (error_budget + 1)
        next_upper = [None] * (error_budget + 1)
        for errors in range(error_budget + 1):
            if lower[errors] is None:
                continue
            for extra_errors, extra_positives in ((accepted_y, 0), (accepted_n, group_size)):
                new_errors = errors + extra_errors
                if new_errors > error_budget:
                    continue
                new_lower = lower[errors] + extra_positives
                new_upper = upper[errors] + extra_positives
                old_lower, old_upper = next_lower[new_errors], next_upper[new_errors]
                next_lower[new_errors] = new_lower if old_lower is None else min(old_lower, new_lower)
                next_upper[new_errors] = new_upper if old_upper is None else max(old_upper, new_upper)
        lower, upper = next_lower, next_upper
    if all(x is None for x in lower):
        return None
    return (Fraction(min(x for x in lower if x is not None), n),
            Fraction(max(x for x in upper if x is not None), n))
