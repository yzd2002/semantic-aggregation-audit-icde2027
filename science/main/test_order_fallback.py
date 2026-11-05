#!/usr/bin/env python3
"""Exhaustively test cached-pair symmetry and failure behavior for small pools."""
from itertools import product
from order_fallback import canonicalize, combine

for n in range(1, 6):
    binary = list(product('YN', repeat=n))
    for first in binary:
        for second in binary:
            for fallback in binary:
                a, b, h = list(first), list(second), list(fallback)
                actual = combine(a, b, h)
                assert actual == combine(b, a, h)
                assert actual == [x if x == y else z for x, y, z in zip(a, b, h)]
                for i in range(n):
                    partial = h[:]; partial[i] = None
                    assert (combine(a, b, partial) is None) == (a[i] != b[i])
    items = [{'id': i, 'text': str(i)} for i in range(1, n + 1)]
    assert canonicalize(items) == canonicalize(list(reversed(items)))
assert combine(None, ['Y'], ['N']) is None
assert combine(['Y'], None, ['N']) is None
try:
    canonicalize([{'id': 1}, {'id': 1}])
    raise AssertionError('Duplicate IDs should fail.')
except ValueError:
    pass
print('Cached pair symmetry, fallback failures and stable-ID canonicalization exhaustively checked for n<=5.')
