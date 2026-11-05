#!/usr/bin/env python3
"""Deterministic disagreement floor for two aligned binary label vectors."""
def stability_error_floor(left, right):
    """Return candidate disagreement and its max/average error lower bound.

    The bound holds only relative to the same fixed binary reference for both
    label vectors. It says nothing about aggregate ratio error.
    """
    if not left or len(left) != len(right):
        raise ValueError('Require equally sized, nonempty label vectors.')
    if any(x not in ('Y', 'N') for x in left) or any(x not in ('Y', 'N') for x in right):
        raise ValueError('Every label must be Y or N.')
    n = len(left)
    disagreements = sum(a != b for a, b in zip(left, right))
    return {'n': n, 'disagreements': disagreements, 'disagreement_rate': disagreements / n,
            'error_sum_lower_bound': disagreements / n, 'average_error_lower_bound': disagreements / (2 * n),
            'maximum_error_lower_bound': ((disagreements + 1) // 2) / n}
