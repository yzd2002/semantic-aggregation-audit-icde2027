#!/usr/bin/env python3
"""Compare dynamic-program bounds against independent occurrence-level enumeration."""
from fractions import Fraction
from itertools import product
from coupled_ratio_bounds import coupled_ratio_bounds


def partitions(n):
    # Restricted-growth encodings enumerate every set partition exactly once.
    def rec(values, largest):
        if len(values) == n:
            yield tuple(values)
        else:
            for group in range(largest + 2):
                yield from rec(values + [group], max(largest, group))
    yield from rec([0], 0)


def enumerate_case(partition, observations):
    g = 1 + max(partition)
    groups = [[0, 0, 0] for _ in range(g)]
    for group, obs in zip(partition, observations):
        # -1 denotes missing; labels 0/1 are N/Y.
        groups[group][2 if obs == -1 else 1 - obs] += 1
    reference_counts = []
    for labels in product((0, 1), repeat=g):
        truth = [labels[group] for group in partition]
        err = sum(obs != value for obs, value in zip(observations, truth) if obs != -1)
        reference_counts.append((err, sum(truth)))
    return groups, reference_counts


cases = infeasible = narrowed = gaps = 0
for n in range(1, 7):
    for partition in partitions(n):
        for observations in product((-1, 0, 1), repeat=n):
            groups, outcomes = enumerate_case(partition, observations)
            m = sum(x != -1 for x in observations)
            p = sum(x == 1 for x in observations)
            for k in range(m + 1):
                feasible_counts = {positives for errors, positives in outcomes if errors <= k}
                expected = None if not feasible_counts else (Fraction(min(feasible_counts), n), Fraction(max(feasible_counts), n))
                actual = coupled_ratio_bounds(groups, k)
                assert actual == expected, (partition, observations, k, groups, expected, actual)
                cases += 1
                minimum_required_errors = sum(min(a, b) for a, b, _ in groups)
                assert (actual is None) == (minimum_required_errors > k)
                if actual is None:
                    infeasible += 1
                else:
                    unrestricted = (Fraction(max(0, p-k), n), Fraction(n-m+min(m, p+k), n))
                    assert unrestricted[0] <= actual[0] <= actual[1] <= unrestricted[1]
                    if len(groups) == n:
                        assert actual == unrestricted
                    if actual != unrestricted:
                        narrowed += 1
                    if len(feasible_counts) != max(feasible_counts) - min(feasible_counts) + 1:
                        gaps += 1

assert coupled_ratio_bounds([(1, 0, 1)], 0) == (Fraction(1), Fraction(1))
assert coupled_ratio_bounds([(1, 1, 0)], 0) is None
assert coupled_ratio_bounds([(0, 0, 2)], 0) == (Fraction(0), Fraction(1))
for groups, k in [([], 0), ([(0, 0, 0)], 0), ([(1, 0, 0)], -1), ([(1, 0, 0)], 2)]:
    try:
        coupled_ratio_bounds(groups, k)
    except ValueError:
        pass
    else:
        raise AssertionError((groups, k))
for groups, k in [([(True, 0, 0)], 0), ([(1, 0, 0)], False)]:
    try:
        coupled_ratio_bounds(groups, k)
    except TypeError:
        pass
    else:
        raise AssertionError((groups, k))
print(f'PASS: {cases:,} partition/observation/budget cases, '
      f'{infeasible:,} infeasible, {narrowed:,} stricter than unrestricted, '
      f'{gaps:,} with unattainable interior counts (n <= 6).')
