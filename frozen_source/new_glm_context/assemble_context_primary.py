#!/usr/bin/env python3
"""Collect exactly the six fresh DBLP/BoolQ cases; do not invent SNLI results."""
import argparse
from common import *
from global_primary.primary_family import NOMINAL_CONFIDENCE


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--components',type=Path,nargs=3,required=True)
    parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    cases=[]
    for path in args.components:
        data=json.loads(path.read_text());cases.extend(data['cases'])
    expected={(task,model) for task in TASKS for model in MODELS}
    actual={(r['task'],r['model']) for r in cases}
    if len(cases)!=6 or actual!=expected:raise ValueError('Exactly all six DBLP/BoolQ model cases are required')
    if any(r['mode']!='primary_mixed' or r['confidence']!=NOMINAL_CONFIDENCE or r['primary_family_size']!=9 for r in cases):
        raise ValueError('A context component differs from the frozen nine-case primary definition')
    order={(task,model):i for i,(task,model) in enumerate((t,m) for t in TASKS for m in MODELS)}
    write_json(args.out,{'component':'Six fresh DBLP/BoolQ cases; SNLI contributes three separately observed cases',
        'complete_nine_case_suite':False,'cases':sorted(cases,key=lambda r:order[r['task'],r['model']]),
        'source_component_sha256':{str(p):sha(p) for p in args.components},
        'final_nine_case_assembly':'Root must supply the real three SNLI cases to global_primary.primary_family.assemble_suite',
        'legacy_ci98_75_flags_are_not_authoritative':True})


if __name__=='__main__':main()
