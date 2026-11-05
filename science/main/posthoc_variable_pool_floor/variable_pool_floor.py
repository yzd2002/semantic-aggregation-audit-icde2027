#!/usr/bin/env python3
"""Deterministic ratio-MAE floor and constructive witness for variable pools."""
from fractions import Fraction


def evaluate(pools):
    """Return exact bounds and a feasible reference-count witness.

    Each pool is (size, first_positive_count, second_positive_count).
    """
    if not pools:raise ValueError('At least one pool is required')
    for row in pools:
        if len(row)!=3 or any(type(x) is not int for x in row):raise ValueError('Counts must be integers')
        n,a,b=row
        if n<=0 or min(a,b)<0 or max(a,b)>n:raise ValueError('Invalid pool counts')
    q=len(pools)
    total=sum((Fraction(abs(a-b),n) for n,a,b in pools),Fraction())
    floor=total/(2*q)
    target=total/2
    steps=sorted(((Fraction(1,n),i) for i,(n,a,b) in enumerate(pools) for _ in range(abs(a-b))),
                 key=lambda x:x[0])
    selected=[0]*q;running=Fraction()
    for weight,i in steps:
        if running+weight>=target:
            if abs(running+weight-target)<abs(running-target):
                selected[i]+=1;running+=weight
            break
        selected[i]+=1;running+=weight
    reference=[a+(1 if b>=a else -1)*s for (n,a,b),s in zip(pools,selected)]
    err_a=sum((Fraction(abs(a-r),n) for (n,a,b),r in zip(pools,reference)),Fraction())/q
    err_b=sum((Fraction(abs(b-r),n) for (n,a,b),r in zip(pools,reference)),Fraction())/q
    witness=max(err_a,err_b)
    slack=Fraction(1,2*q*min(n for n,a,b in pools))
    assert floor<=witness<=floor+slack
    return {'floor':floor,'witness_mae':witness,'witness_reference_counts':reference,
            'witness_mae_first':err_a,'witness_mae_second':err_b,'guaranteed_witness_slack':slack,
            'weighted_count_gap_sum':total}
