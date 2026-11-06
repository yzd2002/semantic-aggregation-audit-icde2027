#!/usr/bin/env python3
"""Collect exactly three real SNLI components; six other cases stay external."""
import argparse
from common import *
from global_primary.primary_family import NOMINAL_CONFIDENCE


def assemble(paths):
    cases=[]
    for path in paths:
        data=json.loads(Path(path).read_text())
        if data.get('synthetic_cpu_fixture_not_model_evidence') is not False:
            raise ValueError('Only actual model-run components may enter real primary assembly')
        if len(data['cases'])!=1 or data['cases'][0]['model']!=data['model']:
            raise ValueError('Each model component must contain its one actual SNLI case')
        cases.extend(data['cases'])
    if len(cases)!=3 or {(r['task'],r['model']) for r in cases}!={('snli',m) for m in MODELS}:
        raise ValueError('Exactly all three real SNLI/model cases are required, including unavailable cases')
    for case in cases:
        if case['mode']!='primary_mixed' or case['scope']!='full-first' or case['confidence']!=NOMINAL_CONFIDENCE or case['primary_family_size']!=9:
            raise ValueError('An SNLI component differs from the frozen nine-case primary definition')
        if case.get('synthetic_cpu_fixture_not_model_evidence') is not False:raise ValueError('Synthetic case is not model evidence')
    return {'component':'Three SNLI cases; six fresh DBLP/BoolQ cases must be separately observed',
        'complete_nine_case_suite':False,'cases':sorted(cases,key=lambda r:MODELS.index(r['model'])),
        'source_component_sha256':{str(p):sha(p) for p in paths},'other_six_cases_not_fabricated':True,
        'final_nine_case_assembly':'Root supplies all nine real cases to global_primary.primary_family.assemble_suite',
        'legacy_ci98_75_flags_are_not_authoritative':True}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--components',type=Path,nargs=3,required=True)
    parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    write_json(args.out,assemble(args.components))


if __name__=='__main__':main()
