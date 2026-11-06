#!/usr/bin/env python3
"""Pre-inference arithmetic, parser and interrupted-run checks with synthetic labels."""
import itertools
import numpy as np
from common import *
from analyze import paired_vectors,count_pair,count_condition,analyze
from secondary import reference_transport,main as secondary_analysis

out=ROOT/'cpu_checks';out.mkdir(exist_ok=False)
truth=[]
for a,b,c,d in itertools.product((0,1),repeat=4):
 v=paired_vectors([a],[b],[c],[d],[0])
 expected=((a!=c)+(b!=d)-(a!=b)-(c!=d))/2
 assert v['excess'][0]==expected
 assert v['near_minus_random_error'][0]==v['right_to_wrong'][0]-v['wrong_to_right'][0]
 truth.append({'near':[a,b],'random':[c,d],'excess':float(v['excess'][0])})
assert set(r['excess'] for r in truth)=={-1.,0.,1.}
assert paired_vectors([0],[1],[0],[1],[0])['excess'][0]==-1
cancel=count_pair([0,1],[1,0],[1,0],2)
assert cancel['changed_target_count']==2 and cancel['absolute_ratio_change']==0 and cancel['wrong_to_right']==2
assert count_condition([1],[1,0],2)['status']=='abstain'
rows=[{'task':'boolq','target_id':str(i),'selected_ordinal':i,'gold':1} for i in range(3)]
labels={}
for r in rows:
 for s,c,j in [('singleton','none',1),('singleton','none',2),('full-first','random',1),('full-first','random',2)]:labels['boolq',r['target_id'],s,c,j]=0
for target,near in [('0',(1,1)),('1',(None,1)),('2',(0,0))]:
 for j,value in enumerate(near,1):labels['boolq',target,'full-first','near',j]=value
transport=reference_transport(rows,labels)
assert transport['four_call_pass_n']==3 and transport['passing_both_near_calls_valid_n']==2
assert transport['passing_not_evaluable_n']==1 and transport['any_flip_count']==transport['stable_flip_count']==1
assert transport['any_flip_fraction']==.5
empty=reference_transport(rows,{})
assert empty['any_flip_fraction'] is None and empty['reference_gold_error_fraction_on_all_passing'] is None
parser_checks()
assert not (ROOT/'run_qwen').exists() and not (ROOT/'run_smol').exists(), 'This pre-inference check must precede scientific runs.'
failure=analyze(ROOT,'qwen',out/'no_run_main')
failure_secondary=secondary_analysis(ROOT,'qwen',out/'no_run_secondary')
assert failure['completion']['scientific_submitted']==0
for task in TASKS:
 assert failure['tasks'][task]['common12_n']==0
 assert all(r['status']=='abstain' for r in failure['tasks'][task]['fixed_relation_conditions'])
 assert len(failure['tasks'][task]['condition_quality'])==14
 assert failure_secondary['tasks'][task]['reference_transport']['any_flip_fraction'] is None
write_json(out/'CPU_CHECKS.json',{'status':'PASS','truth_table':truth,'negative_excess_retained':True,
 'global_count_cancellation_example':cancel,'invalid_near_reference_probe_example':transport,
 'empty_reference_probe_NA':True,'parser_checks':parser_checks(),'no_run_main_failure_ledger':'PASS',
 'no_run_secondary_failure_ledger':'PASS','synthetic_arithmetic_only_no_model_judgments':True})
print('CPU arithmetic, parser, reference-probe denominator and empty/interrupted-run checks PASS')
