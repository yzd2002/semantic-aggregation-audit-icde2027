#!/usr/bin/env python3
"""Check ID mapping, prompt isolation, and invalid-output rejection."""
import json
from acord_prompts import IDS,SYSTEM,RETRY_SUFFIX,keyed_prompt,point_prompt,keyed_regex,parse_keyed,parse_point

row={'query':'Find a clause matching the stated terms','query_id':'SECRET_QUERY_IDENTIFIER',
     'items':[{'doc_id':'SECRET_DOCUMENT_IDENTIFIER_'+str(i),'text':'candidate clause '+str(i),
               'gold':'SECRET_REFERENCE_LABEL','score':'SECRET_REFERENCE_SCORE','source_chars':123} for i in range(32)]}
for start in (0,8,16,24):
    front,ids=keyed_prompt(row,start);back,reversed_ids=keyed_prompt(row,start,True)
    assert ids==IDS and reversed_ids==IDS[::-1]
    def contents(prompt):
        body=prompt.split('Clauses in order:\n',1)[1].split('\nReturn only a JSON object',1)[0]
        return {x['id']:x['clause'] for x in map(json.loads,body.split('\n'))}
    assert contents(front)==contents(back)=={i+1:'candidate clause '+str(start+i) for i in range(8)}
    for secret in ('SECRET_QUERY_IDENTIFIER','SECRET_DOCUMENT_IDENTIFIER','SECRET_REFERENCE_LABEL','SECRET_REFERENCE_SCORE'):
        assert secret not in front and secret not in back
for i in range(32):
    p=point_prompt(row,i)
    assert 'candidate clause '+str(i) in p and 'SECRET_' not in p
example='{'+','.join(json.dumps(str(i))+':"'+('Y' if i%2 else 'N')+'"' for i in IDS)+'}'
assert parse_keyed(example)=={i:('Y' if i%2 else 'N') for i in IDS}
bad=[example[:-1],example.replace('"1":"Y"','"1":"N","1":"Y"'),
     example.replace('"1":"Y"','"9":"Y"'),example.replace('"1":"Y"','"1":"?"'),
     ' '+example,example.replace('"1":"Y"','"1":"Y", "extra":"N"')]
assert all(parse_keyed(x) is None for x in bad)
assert parse_point('Y')=='Y' and parse_point('N')=='N' and all(parse_point(x) is None for x in (' Y','N\n','yes',''))
assert SYSTEM and RETRY_SUFFIX and keyed_regex()
print('ACORD prompt field isolation, stable-ID reversal and strict parser controls passed.')

long_row={'query':'test','items':[{'text':'a'*2400} for _ in range(32)]}
keyed_prompt(long_row,0)
long_row['items'][0]['text']='a'*2401
try:keyed_prompt(long_row,0)
except ValueError:pass
else:raise AssertionError('A prefix above 2400 characters was accepted')
print('The 2400-character boundary is enforced.')
