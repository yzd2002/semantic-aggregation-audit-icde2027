#!/usr/bin/env python3
import numpy as np
from analyze_product_weight_sensitivity import run
rows=[{'query_id':'1','items':[{'id':1,'product_id':'a','gold':'Y'},{'id':2,'product_id':'b','gold':'N'}]},
      {'query_id':'2','items':[{'id':1,'product_id':'a','gold':'Y'},{'id':2,'product_id':'c','gold':'N'}]}]
old=[{'query_id':'1','pred':['N','N']},{'query_id':'2','pred':['Y','Y']}]
new=[{'query_id':'1','pred':['Y','N']},{'query_id':'2','pred':['N','Y']}]
comparison={'common_query_ids':['1','2'],'comparison':{'old_unguided_mae':0.5,'new_schema_guided_mae':0.0}}
r=run(rows,old,new,comparison,123,20)
assert r['pools']==2 and r['candidates']==4 and r['distinct_products']==3 and r['repeated_products']==1
assert r['difference_new_minus_old']==-0.5 and len(r['reweighted_differences'])==20
assert all(np.isfinite(r['reweighted_differences'])) and r['sensitivity_range_95'][0]<=r['sensitivity_range_95'][1]
try:run(rows,old,new,{'common_query_ids':['2','1'],'comparison':comparison['comparison']},123,20)
except ValueError:pass
else:raise AssertionError('Unexpected acceptance of reordered queries')
print('Fixture tests: unweighted ratios, repeated product, finite weighting and ID order passed.')
