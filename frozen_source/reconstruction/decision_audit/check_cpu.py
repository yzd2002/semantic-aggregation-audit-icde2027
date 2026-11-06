#!/usr/bin/env python3
"""CPU contract tests for exact selectors and finite-relation reporting."""
from copy import deepcopy
from fractions import Fraction
import ast
from common import *
from selector import select_policies,TOLERANCES
from analyze_test import disagreement,gold_metrics,evaluation,empirical_indicator,arithmetic_break_even,compare_policies

def run_checks():
 checks=[]
 def check(name,condition):
  if not condition:raise AssertionError(name)
  checks.append(name)
 def vectors(value=0):
  return {'singleton':{1:[0]*128,2:[0]*128},**{config_id(k,g,t):{r:[value]*128 for r in repeats('dev',t)} for k in KS for g in GROUPS for t in TYPES}}
 for label,tau in TOLERANCES:
  for observations,selector in ((2,'Input'),(4,'Repeat')):
   denominator=128*observations;limit=tau.numerator*denominator//tau.denominator
   for errors,want in ((limit,True),(limit+1,False)):
    v=vectors(1);identifier=config_id(8,'random','C')
    v[identifier]={r:([1]*errors+[0]*(128-errors) if r==1 else [0]*128) for r in range(1,5)}
    ledger=select_policies(v,tau=tau)[selector]['eligibility_ledger']
    entry=next(x for x in ledger if x['configuration_id']==identifier)
    check(f'{label}/{selector}/{errors}',entry['eligible']==want and entry['disagreements']==errors and entry['fixed_denominator']==denominator)
  limit=tau.numerator*128//tau.denominator
  for errors,want in ((limit,True),(limit+1,False)):
   check(f'{label}/test/{errors}',empirical_indicator(disagreement([1]*errors+[0]*(128-errors),[0]*128),tau)==want)
 v=vectors();s=select_policies(v)
 check('call_group_type_ties',all(p['chosen']['configuration_id']=='k8/random/C' for p in s.values()))
 for r in (1,2):v['k8/random/C'][r][0]=1
 check('exact_loss_precedes_group_tie',select_policies(v)['Output']['chosen']['configuration_id']=='k8/random/O')
 v=vectors();v['singleton'][1][0]=None
 check('invalid_reference_fallback',all(p['chosen']['reason']=='reference_calibration_unavailable' for p in select_policies(v).values()))
 v=vectors(1);v['k8/random/O'][1][0]=None
 entry=next(x for x in select_policies(v)['Output']['eligibility_ledger'] if x['configuration_id']=='k8/random/O')
 check('invalid_candidate_does_not_reduce_denominator',not entry['eligible'] and entry['disagreements'] is None and entry['fixed_denominator']==256)
 check('no_eligible_fallback',all(p['chosen']['configuration_id']=='singleton' for p in select_policies(vectors(1)).values()))
 gold=[1]*64+[0]*64;pred=gold.copy();pred[0]=0;pred[-1]=1
 m=gold_metrics(pred,gold);transport=evaluation(pred,gold,gold)['gold_and_count_transport_from_reference']
 check('count_aggregates_before_absolute',m['error']==2/128 and m['absolute_ratio_error']==0 and m['opposing_gold_error_pairs']==1)
 check('candidate_change_is_not_ratio_change',transport['candidate_flips']==2 and transport['absolute_ratio_change_from_reference']==0 and transport['opposing_flip_pairs']==1)
 check('wrong_to_right_sign',evaluation(gold,pred,gold)['gold_and_count_transport_from_reference']['execution_minus_reference_gold_error']==-2/128)
 broken=pred.copy();broken[0]=None;m=gold_metrics(broken,gold)
 check('incomplete_count_abstains',m['predicted_positive_count'] is None and m['absolute_ratio_error'] is None and m['accuracy'] is None and m['valid_outcomes']==127)
 check('incomplete_fidelity_abstains',disagreement(broken,gold)['numerator'] is None and disagreement(broken,gold)['fixed_denominator']==128)
 check('reference_has_no_self_fidelity',evaluation(gold,gold,gold,True)['singleton_fidelity'] is None)
 check('independent_P_uses_common_S',evaluation(pred,gold,gold)['singleton_fidelity']['numerator']==2)
 for k,equal,strict in ((1,None,None),(2,18,19),(4,12,13),(8,11,11)):
  result=arithmetic_break_even(k);check(f'break_even/{k}',result['equal_or_lower_call_cost_relations']==equal and result['strictly_lower_call_cost_relations']==strict)
 def policy(errors,calls,config):
  return {'primary_operating_point':True,'tolerance':'0.05','logical_choice':{'configuration_id':config},
   'repeats':{str(r):{'singleton_fidelity':{'complete':True,'numerator':errors[r-1],'value':errors[r-1]/128},
    'execution_cost':{'planned_calls':calls},'empirical_tolerance_satisfied':errors[r-1]<=6} for r in (1,2)}}
 a=policy([0,0],128,'singleton');b=policy([1,1],16,'k8/random/O')
 c=compare_policies(a,b)
 check('lower_calls_with_error_is_not_dominance',not c['observed_output_dominance'] and c['fewer_calls_and_both_replays_within_tolerance'])
 check('both_replays_required_for_dominance',not compare_policies(policy([1,1],16,'A'),policy([0,2],16,'B'))['observed_output_dominance'])
 check('strict_improvement_at_equal_calls',compare_policies(policy([2,2],16,'A'),policy([1,2],16,'B'))['observed_output_dominance'])
 check('identical_plan_is_not_strict_dominance',not compare_policies(a,a)['observed_output_dominance'])
 check('strict_parser',all(r['valid']==r['expected_valid'] for r in parser_checks()))
 for path in ROOT.glob('*.py'):ast.parse(path.read_text())
 check('all_python_syntax',True)
 return {'status':'PASS','checks':len(checks),'names':checks,'gpu_used':False,'scientific_outputs_used':False,
  'tested_source_hashes':{p.name:sha(p) for p in ROOT.glob('*.py') if p.name!='freeze.py'}}

if __name__=='__main__':
 result=run_checks();write_json(ROOT/'CPU_CONTRACT_CHECKS.json',result);print(compact({'status':result['status'],'checks':result['checks']}))
