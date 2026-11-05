#!/usr/bin/env python3
"""Exhaust all binary gold/prediction pairs through size 6; no model or dataset."""
from itertools import product
from fractions import Fraction
from error_cancellation import summarize,keyed
total=0
for n in range(1,7):
    for gold in product('YN',repeat=n):
        for pred in product('YN',repeat=n):
            row={'items':[{'gold':g} for g in gold]};s=summarize([row],[list(pred)])
            e=sum(x!=y for x,y in zip(gold,pred));d=pred.count('Y')-gold.count('Y')
            assert s['candidate_error_macro']==float(Fraction(e,n))
            assert s['ratio_mae']==float(Fraction(abs(d),n))
            assert s['cancelled_error_macro']==float(Fraction(e-abs(d),n));total+=1
s=summarize([{'items':[{'gold':g} for g in 'YYNN']}],[list('NNYY')]);assert s['ratio_mae']==0 and s['candidate_error_macro']==1 and s['exact_ratio_with_candidate_errors']==1
assert keyed('{"1":"Y","1":"N"}',2) is None and keyed(' {"1":"Y"}',1) is None
print('Exhaustive identity checks passed for',total,'binary vector pairs, including fully wrong labels with an exact ratio.')
