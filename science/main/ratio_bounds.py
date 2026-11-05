#!/usr/bin/env python3
"""Sharp finite-pool ratio bounds under an explicit accepted-label error budget."""
def ratio_bounds(n,m,p,k=0):
    """Return (lower, upper) assuming at most k of m accepted binary labels are wrong.

    n is the total pool size, m the accepted-label count, and p the accepted Y count.
    Missing labels are unconstrained. A valid k must be justified independently;
    this function does not estimate semantic error.
    """
    if any(type(x) is not int for x in (n,m,p,k)):
        raise TypeError('n, m, p, and k must be integers')
    if n<=0 or not (0<=m<=n) or not (0<=p<=m) or not (0<=k<=m):
        raise ValueError('Require n>0, 0<=m<=n, 0<=p<=m and 0<=k<=m')
    return max(0,p-k)/n,(min(m,p+k)+n-m)/n
