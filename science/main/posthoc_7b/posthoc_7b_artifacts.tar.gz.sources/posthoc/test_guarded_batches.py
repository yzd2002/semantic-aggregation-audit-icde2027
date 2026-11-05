#!/usr/bin/env python3
"""Check guarded fixed-size scheduling without using a GPU or a language model."""
import ast
from pathlib import Path

for name in ('run_wands_7b_posthoc.py','run_trec_7b_posthoc.py'):
    source=Path(__file__).resolve().parent/name
    tree=ast.parse(source.read_text(encoding='utf8'))
    funcs=[x for x in ast.walk(tree) if isinstance(x,ast.FunctionDef) and x.name=='guarded_generate']
    assert len(funcs)==1
    calls=[];checks=[]
    class Fake:
        def generate(self,prompts,params,use_tqdm=False):
            calls.append((list(prompts),params,use_tqdm))
            return list(prompts)
    namespace={'llm':Fake(),'assert_own_gpu':lambda:checks.append('check')}
    isolated=ast.fix_missing_locations(ast.Module(body=funcs,type_ignores=[]))
    exec(compile(isolated,str(source),'exec'),namespace)
    generate=namespace['guarded_generate']
    assert generate([],object())==[] and not calls and not checks
    prompts=list(range(129));params=list(range(200,329))
    assert generate(prompts,params)==prompts
    assert [len(c[0]) for c in calls]==[64,64,1] and [len(c[1]) for c in calls]==[64,64,1]
    assert [x for _,p,_ in calls for x in p]==params and len(checks)==3
    calls.clear();checks.clear();token=object()
    assert generate(list(range(65)),token)==list(range(65))
    assert [len(c[0]) for c in calls]==[64,1] and all(c[1] is token for c in calls) and len(checks)==2
    calls.clear();checks.clear()
    def deny_after_first():
        checks.append('check')
        if len(checks)>1:raise RuntimeError('foreign GPU process')
    namespace['assert_own_gpu']=deny_after_first
    try:generate(list(range(65)),token)
    except RuntimeError:pass
    else:raise AssertionError('A foreign GPU process did not stop scheduling')
    assert len(calls)==1 and len(checks)==2
print('Guarded scheduling: empty, ordered batches, parameter slicing and foreign-process abort passed for both runners.')
