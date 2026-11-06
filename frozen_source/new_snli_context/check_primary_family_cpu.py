#!/usr/bin/env python3
"""CPU definition checks for the exact copied global-primary module."""
import itertools
import numpy as np
from common import *
from global_primary.primary_family import *
from analyze import count_condition,count_pair


def main():
    assert TASKS==('dblp_acm','boolq','snli') and PRIMARY_CASES==tuple((t,m) for t in TASKS for m in MODELS)
    assert NOMINAL_CONFIDENCE==179/180 and BOOTSTRAP_DRAWS==4000
    assert (LOWER_QUANTILE,UPPER_QUANTILE)==(1/360,359/360)
    for bits in itertools.product((0,1),repeat=4):
        a1,a2,b1,b2=bits
        expected=((a1!=b1)+(a2!=b2)-(a1!=a2)-(b1!=b2))/2
        assert excess_vector(*[[b] for b in bits]).tolist()==[expected]
    records=[]
    for model in MODELS:
        for values in ([],[-1]*128,[0]*128,[1]*128,[-1,0,1]*42+[0,0]):
            result=primary_interval(values,task='snli',model=model)
            assert result==primary_interval(values,task='snli',model=model)
            assert result['positive_primary_evidence']==(bool(values) and values==[1]*128)
            if not values:
                assert result['available'] is False and result['mean_excess'] is None and result['interval'] is None
            records.append({'model':model,'n':len(values),'available':result['available'],'passed':True})
    assert count_condition([0]*127,[0]*128,128)['status']=='abstain'
    assert count_pair([0]*128,[0]*127,[0]*128,128)['status']=='abstain'
    assert count_condition([0]*31,[0]*32,32)['status']=='abstain'
    rejected=False
    try:assemble_suite([primary_interval([],task='snli',model=m) for m in MODELS])
    except ValueError:rejected=True
    assert rejected,'Three SNLI cases cannot masquerade as the full nine-case suite'
    for name,want in json.loads((ROOT/'global_primary/AMENDMENT_SHA256.json').read_text()).items():assert sha(ROOT/'global_primary'/name)==want
    write_json(ROOT/'verification/PRIMARY_FAMILY_CPU_CHECK.json',{'status':'PASS','checks':records,
        'exhaustive_16_binary_excess_patterns':True,'only_three_SNLI_cannot_assemble_nine':True,
        'COUNT_fixed_128_and_32_missing_target_abstention':True,'all_test_values_synthetic_not_model_evidence':True,
        'module_sha256':sha(ROOT/'global_primary/primary_family.py'),'real_model_judgments':0,'GPU_execution':False})
    print(compact({'status':'PASS','definition_checks':len(records),'real_model_judgments':0}),flush=True)


if __name__=='__main__':main()
