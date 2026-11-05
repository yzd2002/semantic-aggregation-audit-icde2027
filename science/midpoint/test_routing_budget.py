#!/usr/bin/env python3
"""Check the exact random-subset formula against independent enumeration."""
from itertools import combinations,product
from fractions import Fraction
from collections import Counter
from routing_budget import exact_random_mae,pool_stat

cases=0
for n in range(1,8):
    for plus in range(n+1):
        for minus in range(n-plus+1):
            zero=n-plus-minus;deltas=[1]*plus+[-1]*minus+[0]*zero
            for k in range(n+1):
                dist=Counter(sum(deltas[i] for i in selected) for selected in combinations(range(n),k))
                for delta in range(-n,n+1):
                    expected=Fraction(sum(count*abs(delta+change) for change,count in dist.items()),n*sum(dist.values()))
                    assert exact_random_mae(plus,minus,zero,k,delta)==expected
                    cases+=1
candidate_cases=0
for n in range(1,5):
    for flat in product('YN',repeat=3*n):
        a=list(flat[:n]);h=list(flat[n:2*n]);g=list(flat[2*n:]);row={'query_id':'test','items':[{'gold':x} for x in g]}
        for k in range(n+1):
            b=[('N' if a[i]=='Y' else 'Y') if i<k else a[i] for i in range(n)]
            s,_=pool_stat(row,a,b,h);errors=[]
            for selected in combinations(range(n),k):
                keep=set(selected);p=[h[i] if i in keep else a[i] for i in range(n)]
                errors.append(Fraction(sum(x!=y for x,y in zip(p,g)),n))
            expected=sum(errors,Fraction(0))/len(errors)
            assert Fraction(*map(int,s['expected_random_candidate_error_exact']))==expected
            candidate_cases+=1
print('PASS:',cases,'exact ratio-expectation cases and',candidate_cases,'candidate-expectation cases against exhaustive uniform subsets.')
