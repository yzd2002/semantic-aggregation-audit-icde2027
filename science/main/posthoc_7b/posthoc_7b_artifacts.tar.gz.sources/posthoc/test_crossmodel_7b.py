#!/usr/bin/env python3
"""Exercise paired ratio comparisons and failure-intersection semantics."""
from crossmodel_7b import NEEDED,paired
rows=[{'query_id':1,'items':[{'gold':'Y'},{'gold':'N'}]},
      {'query_id':2,'items':[{'gold':'Y'},{'gold':'Y'},{'gold':'N'}]},
      {'query_id':3,'items':[{'gold':'N'}]}]
def make(v):return {k:[list(x) if x is not None else None for x in v] for k in NEEDED}
old=make([['Y','Y'],['Y','N','N'],['N']]);new=make([['Y','N'],['Y','Y','N'],['Y']])
new['replay'][2]=None
out=paired(rows,old,new,resamples=100,seed=12)
assert out['common_query_ids']==[1,2] and out['common_pools']==2
assert abs(out['all_y_mae']-(.5+1/3)/2)<1e-12
for mode in ['original','actual_fallback','pointwise']:
 r=out['comparisons'][mode];assert abs(r['older_4b_mae']-5/12)<1e-12
 assert r['new_7b_mae']==0 and abs(r['difference_7b_minus_4b']+5/12)<1e-12
 assert abs(r['older_signed_ratio_bias']-1/12)<1e-12 and r['new_signed_ratio_bias']==0
 assert r['paired_query_95_percentile_ci'][1]<0
for k in NEEDED:new[k]=[None]*3
empty=paired(rows,old,new,resamples=20,seed=1)
assert empty['common_pools']==0 and empty['comparisons']=={} and empty['all_y_mae'] is None
print('Paired ratio deltas, common-plan intersection and no-common-pool case passed.')
