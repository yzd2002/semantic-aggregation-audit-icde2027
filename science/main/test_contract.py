"""Tests for the failure boundaries of the output contract, not model accuracy."""
from run_experiment import parse_batch,parse_spec
from run_validation import parse_compact
from run_full_pools import chunks
assert parse_batch('[{"id":2,"label":"N"},{"id":1,"label":"Y"}]',2)==['Y','N']
for s in ('[{"id":1,"label":"Y"}]',
          '[{"id":1,"label":"Y"},{"id":1,"label":"N"}]',
          '[{"id":1,"label":"Y"},{"id":3,"label":"N"}]',
          '[{"id":true,"label":"Y"},{"id":2,"label":"N"}]',
          '[{"id":1,"label":"Y","label":"N"},{"id":2,"label":"N"}]',
          '[{"id":1,"label":"Y"},{"id":2,"label":"?"}]'):
    assert parse_batch(s,2) is None,s
assert parse_spec('{"product_type":"shoe","required":[],"preferred":[]}')
assert parse_spec('{"product_type":"shoe","required":"red","preferred":[]}') is None
assert parse_spec('{"product_type":"","required":[],"preferred":[]}') is None
assert parse_compact('YNNY',4)==['Y','N','N','Y']
for s in ('YNN','YNNYX','Y NN','YYN?',''):
    assert parse_compact(s,4) is None,s
pool={'query':'test','items':[{'id':i,'text':str(i)} for i in range(1,18)]}
parts=chunks(pool)
assert [len(p['items']) for p in parts]==[8,8,1]
assert [p['start'] for p in parts]==[0,8,16]
assert all([it['id'] for it in p['items']]==list(range(1,len(p['items'])+1)) for p in parts)
assert [it['text'] for p in parts for it in p['items']]==[str(i) for i in range(1,18)]
print('All contract tests passed.')
