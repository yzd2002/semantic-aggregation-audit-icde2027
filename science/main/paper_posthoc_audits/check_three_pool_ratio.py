#!/usr/bin/env python3
"""Independent finite-pool check via integer feasibility, not the manuscript's analysis script."""
from itertools import product
from math import lcm

pools = [(n, a, b) for n in range(1, 5) for a in range(n + 1) for b in range(n + 1)]
cases = unequal = nonsharp = equal_exact = 0
for Q in (1, 2, 3):
    for group in product(pools, repeat=Q):
        L = lcm(*(n for n, _, _ in group))
        states = [1]
        total = 0
        for n, a, b in group:
            weight, delta = L // n, abs(a - b)
            total += weight * delta
            reachable = 0
            for v in range(delta + 1):
                reachable |= states[-1] << (v * weight)
            states.append(reachable)
        feasible = [x for x in range(total + 1) if (states[-1] >> x) & 1]
        x = min(feasible, key=lambda v: max(v, total - v))
        worst = max(x, total - x)
        assert 2 * worst >= total
        assert 2 * worst - total <= L // min(n for n, _, _ in group)
        # Recover a concrete reference count in every pool.
        remainder = x
        for j in range(Q - 1, -1, -1):
            n, a, b = group[j]
            weight, delta = L // n, abs(a - b)
            choices = [v for v in range(delta + 1)
                       if remainder >= v * weight and (states[j] >> (remainder - v * weight)) & 1]
            assert choices
            v = choices[0]
            truth = a + (v if b >= a else -v)
            assert 0 <= truth <= n
            assert abs(a - truth) == v and abs(b - truth) == delta - v
            remainder -= v * weight
        assert remainder == 0
        if len({n for n, _, _ in group}) == 1:
            n = group[0][0]
            assert worst == ((sum(abs(a-b) for _, a, b in group) + 1) // 2) * (L // n)
            equal_exact += 1
        else:
            unequal += 1
        if 2 * worst > total:
            nonsharp += 1
        cases += 1
print(f'PASS: {cases:,} ordered one-to-three-pool cases, {unequal:,} with unequal sizes; '
      f'equal-size integer floor exact in {equal_exact:,}; '
      f'continuous floor not attainable in {nonsharp:,}; all reference witnesses recovered (n <= 4).')
