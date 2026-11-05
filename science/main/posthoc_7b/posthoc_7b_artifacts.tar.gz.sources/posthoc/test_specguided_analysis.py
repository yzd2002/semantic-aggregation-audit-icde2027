#!/usr/bin/env python3
"""Exercise schema control parsing, conditional alignment and paired MAE."""
from compare_spec_guided import compare
from verify_specguided_control import valid
valid_json='{"product_type":"shoe","required":[],"preferred":["red"]}'
assert valid(valid_json)
assert not valid('```json\n'+valid_json+'\n```')
assert not valid('{"product_type":"shoe","product_type":"hat","required":[],"preferred":[]}')
assert not valid('{"product_type":" ","required":[],"preferred":[]}')
assert not valid('{"product_type":"shoe","required":[],"preferred":[],"extra":1}')
rows=[{'query_id':1,'items':[{'gold':'Y'},{'gold':'N'}]}, {'query_id':2,'items':[{'gold':'Y'},{'gold':'Y'},{'gold':'N'}]}, {'query_id':3,'items':[{'gold':'N'}]}]
old=[{'query_id':1,'pred':['Y','Y']},{'query_id':2,'pred':['Y','N','N']},{'query_id':3,'pred':['N']}]
new=[{'query_id':1,'pred':['Y','N']},{'query_id':2,'pred':['Y','Y','N']},{'query_id':3,'pred':None}]
x=compare(rows,old,new,[1,2,3],[1,2],resamples=100,seed=12)
assert x['common_query_ids']==[1,2] and x['common_pools']==2
q=x['comparison'];assert abs(q['old_unguided_mae']-5/12)<1e-12 and q['new_schema_guided_mae']==0
assert abs(q['difference_new_minus_old']+5/12)<1e-12 and q['paired_query_95_percentile_ci'][1]<0
assert compare(rows,old,new,[3],[1,2],resamples=5)['comparison'] is None
try:compare(rows,old,new,[1,2,3],[1,2,3],resamples=5)
except ValueError:pass
else:raise AssertionError('Invalid common-pool prediction accepted')
print('Specification control parser, common-pool intersection and paired MAE tests passed.')
