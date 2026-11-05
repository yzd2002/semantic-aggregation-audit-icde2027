#!/usr/bin/env python3
"""Exhaustively check sharpness against all true labels on small finite pools."""
from itertools import product
from ratio_bounds import ratio_bounds
for n in range(1,7):
    for m in range(n+1):
        for observed in product((0,1),repeat=m):
            p=sum(observed)
            for k in range(m+1):
                feasible=[]
                for truth in product((0,1),repeat=n):
                    if sum(x!=y for x,y in zip(observed,truth[:m]))<=k:feasible.append(sum(truth)/n)
                low,high=ratio_bounds(n,m,p,k)
                assert low==min(feasible) and high==max(feasible),(n,m,observed,k)
                assert all(low<=x<=high for x in feasible)
for args in [(0,0,0,0),(3,4,2,0),(3,2,3,0),(3,2,1,3),(-1,0,0,0)]:
    try:ratio_bounds(*args)
    except ValueError:pass
    else:raise AssertionError(args)
try:ratio_bounds(True,0,0,0)
except TypeError:pass
else:raise AssertionError('Boolean dimensions must be rejected')
print('Sharp completion bounds verified for all finite cases with n<=6.')
