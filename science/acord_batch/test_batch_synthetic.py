#!/usr/bin/env python3
"""Prompt and parser fixture with invented data; never invokes a model."""
import json
import acord_prompts
from batch_prompts import prompt,parse

row={'pool_id':'SYNTHETIC','query':'Find the synthetic signal',
     'items':[{'text':'synthetic clause '+str(i),'gold':'SYNTHETIC_GOLD_DO_NOT_LEAK'} for i in range(32)]}
for size in (4,8):
    for start in range(0,32,size):
        for reverse in (False,True):
            text,ids=prompt(row,start,size,reverse)
            assert 'SYNTHETIC_GOLD_DO_NOT_LEAK' not in text
            body=text.split('Clauses in order:\n',1)[1].split('\nReturn only',1)[0]
            clauses=[json.loads(x) for x in body.splitlines()]
            expected=[{'id':i+1,'clause':'synthetic clause '+str(start+i)} for i in range(size)]
            if reverse:expected.reverse()
            assert clauses==expected and ids==[x['id'] for x in expected]
            if size==8:assert (text,ids)==acord_prompts.keyed_prompt(row,start,reverse)
    good='{'+','.join(json.dumps(str(i))+':"'+('Y' if i%2 else 'N')+'"' for i in range(1,size+1))+'}'
    assert parse(good,size)==['Y' if i%2 else 'N' for i in range(1,size+1)]
    assert parse(' '+good,size) is None and parse(good+'\n',size) is None
    assert parse(good.replace('"1":','"2":',1),size) is None
    assert parse(good.replace('"Y"','"M"',1),size) is None
print('Synthetic prompt, stable-ID reversal, old eight-item equivalence and strict parser checks passed; no model was run.')
