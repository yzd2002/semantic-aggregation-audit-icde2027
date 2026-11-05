#!/usr/bin/env python3
"""Test schema constraints and preservation of guarded scheduling."""
import ast,jsonschema
from pathlib import Path
from spec_guidance_schema import SPEC_SCHEMA
from verify_specguided_control import valid
ok={'product_type':'shoe','required':['red'],'preferred':[]}
jsonschema.validate(ok,SPEC_SCHEMA)
for bad in [dict(ok,product_type=''),dict(ok,required=[1]),dict(ok,extra=1),{'required':[],'preferred':[]}]:
 try:jsonschema.validate(bad,SPEC_SCHEMA)
 except jsonschema.ValidationError:pass
 else:raise AssertionError('Invalid schema object accepted')
# The schema permits a whitespace-only type; the independent strict parser rejects it.
blank=dict(ok,product_type=' ');jsonschema.validate(blank,SPEC_SCHEMA)
import json
assert not valid(json.dumps(blank))
w=Path(__file__).resolve().parent
def body(name):
 tree=ast.parse((w/name).read_text())
 matches=[x for x in ast.walk(tree) if isinstance(x,ast.FunctionDef) and x.name=='guarded_generate']
 assert len(matches)==1
 return ast.dump(matches[0],include_attributes=False)
assert body('run_wands_7b_posthoc.py')==body('run_wands_7b_specguided.py')
print('Specification schema rejects invalid structures; strict whitespace check and guarded batching verified.')
