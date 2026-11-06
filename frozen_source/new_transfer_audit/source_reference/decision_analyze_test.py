#!/usr/bin/env python3
"""Frozen finite-relation evaluation. No fitting, confidence intervals, or test oracle."""
import argparse,math
from fractions import Fraction
from common import *
from selector import TOLERANCES
from trace_records import load_trace
from test_gate import verify_test_release

def complete(values,n=128):
 return len(values)==n and all(type(v) is int and v in (0,1) for v in values)

def disagreement(pred,reference,n=128):
 okay=complete(pred,n) and complete(reference,n)
 errors=sum(a!=b for a,b in zip(pred,reference)) if okay else None
 return {'complete':okay,'numerator':errors,'fixed_denominator':n,'value':errors/n if okay else None}

def gold_metrics(pred,gold,n=128):
 assert complete(gold,n) and len(pred)==n
 counts={name:0 for name in ('tp','tn','fp','fn')}
 for p,g in zip(pred,gold):
  if type(p) is int and p in (0,1):counts[('tn','fp','fn','tp')[2*g+p]]+=1
 okay=complete(pred,n);valid=sum(counts.values());positive=sum(gold)
 tp,tn,fp,fn=(counts[k] for k in ('tp','tn','fp','fn'))
 predicted=sum(pred) if okay else None
 return {'complete':okay,'fixed_denominator':n,'valid_outcomes':valid,'invalid_or_missing_outcomes':n-valid,
  'confusion_counts_on_valid_outcomes':counts,'gold_positive_count':positive,
  'predicted_positive_count':predicted,'predicted_positive_fraction':predicted/n if okay else None,
  'accuracy':(tp+tn)/n if okay else None,'error':(fp+fn)/n if okay else None,
  'balanced_accuracy':(tp/positive+tn/(n-positive))/2 if okay and 0<positive<n else None,
  'f1':(2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.0) if okay else None,
  'signed_count_bias':predicted-positive if okay else None,
  'absolute_ratio_error':abs(predicted-positive)/n if okay else None,
  'opposing_gold_error_pairs':min(fp,fn) if okay else None,
  'canceled_gold_error_mass':2*min(fp,fn)/n if okay else None,
  'constant_prediction':('Y' if predicted==n else 'N' if predicted==0 else None) if okay else None,
  'count_status':'computed_on_all128' if okay else 'abstain_incomplete_relation'}

def evaluation(pred,reference,gold,reference_only=False):
 fidelity=None if reference_only else disagreement(pred,reference)
 paired=complete(pred) and complete(reference)
 transport=None
 if paired and not reference_only:
  wr=sum(p==g and s!=g for p,s,g in zip(pred,reference,gold))
  rw=sum(p!=g and s==g for p,s,g in zip(pred,reference,gold))
  transport={'reference_wrong_execution_right':wr,'reference_right_execution_wrong':rw,
   'execution_minus_reference_gold_error':(rw-wr)/128,
   'signed_count_change_from_reference':sum(pred)-sum(reference),
   'absolute_ratio_change_from_reference':abs(sum(pred)-sum(reference))/128,
   'candidate_flips':fidelity['numerator'],
   'opposing_flip_pairs':(fidelity['numerator']-abs(sum(pred)-sum(reference)))//2}
 return {'gold':gold_metrics(pred,gold),'singleton_fidelity':fidelity,
  'fidelity_reference':'not_applicable_reference_output' if reference_only else 'fresh_S_same_repeat',
  'gold_and_count_transport_from_reference':transport}

def empirical_indicator(fidelity,tau):
 return None if fidelity is None or not fidelity['complete'] else tau.denominator*fidelity['numerator']<=tau.numerator*fidelity['fixed_denominator']

def arithmetic_break_even(k):
 if k==1:return {'equal_or_lower_call_cost_relations':None,'strictly_lower_call_cost_relations':None}
 saving=128-128//k
 return {'equal_or_lower_call_cost_relations':math.ceil(Fraction(1152,saving)),
  'strictly_lower_call_cost_relations':1152//saving+1,
  'assumption':'same cost and fidelity on future relations; arithmetic only, not measured transfer'}

def policy_result(policy,table,costs):
 logical=policy['chosen']['configuration_id']
 executed='singleton_execution' if logical=='singleton' else logical
 tau=Fraction(policy['tolerance_numerator'],policy['tolerance_denominator'])
 calibration=policy['observed_standalone_calibration_cost'];rows={}
 for repeat in ('1','2'):
  result=table[executed]['repeats'][repeat];execution=costs['executions'][executed][repeat]
  rows[repeat]={**result,'empirical_tolerance_satisfied':empirical_indicator(result['singleton_fidelity'],tau),
   'execution_cost':execution,'standalone_calibration_plus_execution':{
    'planned_calls':1152+execution['planned_calls'],
    'returned_calls':calibration['returned_calls']+execution['returned_calls'],
    'planned_requested_items':3328+execution['planned_requested_items'],
    'actual_input_tokens':calibration['actual_input_tokens']+execution['actual_input_tokens'],
    'actual_output_tokens':calibration['actual_output_tokens']+execution['actual_output_tokens'],
    'actual_total_tokens':calibration['actual_total_tokens']+execution['actual_total_tokens']}}
 return {'selector':policy['selector'],'tolerance':policy['tolerance'],'primary_operating_point':policy['primary_operating_point'],
  'logical_choice':policy['chosen'],'executed_configuration_id':executed,'repeats':rows,
  'observed_standalone_calibration_cost':calibration,'call_break_even_arithmetic':arithmetic_break_even(policy['chosen']['k'])}

def compare_policies(input_policy,output_policy):
 rows={}
 for repeat in ('1','2'):
  a=input_policy['repeats'][repeat];b=output_policy['repeats'][repeat]
  fa=a['singleton_fidelity'];fb=b['singleton_fidelity'];okay=fa['complete'] and fb['complete']
  rows[repeat]={'complete':okay,'output_minus_input_disagreements':fb['numerator']-fa['numerator'] if okay else None,
   'fixed_denominator':128,'output_minus_input_disagreement_rate':fb['value']-fa['value'] if okay else None,
   'output_minus_input_execution_calls':b['execution_cost']['planned_calls']-a['execution_cost']['planned_calls'],
   'input_within_tolerance':a['empirical_tolerance_satisfied'],'output_within_tolerance':b['empirical_tolerance_satisfied']}
 okay=all(r['complete'] for r in rows.values())
 nonworse=okay and all(r['output_minus_input_disagreements']<=0 and r['output_minus_input_execution_calls']<=0 for r in rows.values())
 strict=okay and any(r['output_minus_input_disagreements']<0 or r['output_minus_input_execution_calls']<0 for r in rows.values())
 return {'direction':'Output minus Input','primary':input_policy['primary_operating_point'],'tolerance':input_policy['tolerance'],
  'input_choice':input_policy['logical_choice']['configuration_id'],'output_choice':output_policy['logical_choice']['configuration_id'],
  'same_locked_choice':input_policy['logical_choice']['configuration_id']==output_policy['logical_choice']['configuration_id'],
  'repeats':rows,'mean_disagreement_rate_difference':sum(r['output_minus_input_disagreements'] for r in rows.values())/256 if okay else None,
  'execution_call_difference':rows['1']['output_minus_input_execution_calls'],
  'observed_output_dominance':nonworse and strict if okay else None,
  'fewer_calls_and_both_replays_within_tolerance':all(r['output_minus_input_execution_calls']<0 and r['output_within_tolerance'] is True for r in rows.values()) if okay else None,
  'inference_scope':'exact fixed-relation census; no independent-item sampling inference'}

def analyze(bundle,model,out):
 bundle=Path(bundle);out=Path(out)
 release=verify_test_release(bundle)
 run=bundle/f'run_test_{model}'
 trace=load_trace(bundle,'test',model,require_complete=(run/'summary.json').is_file())
 policies=json.loads((bundle/'POLICY_LOCK.json').read_text())['payload']['policies']
 goldrows=read_lines(bundle/'inputs/gold.jsonl');relations=read_lines(bundle/'inputs/relations.jsonl')
 relation={(r['task'],r['phase'],r['occurrence_id']):r for r in relations}
 tasks={};prediction_rows=[]
 for task in TASKS:
  rawgold=sorted([r for r in goldrows if r['task']==task and r['phase']=='test'],key=lambda r:r['occurrence_id'])
  assert [r['occurrence_id'] for r in rawgold]==list(range(1,129))
  assert all(r['source_id']==relation[(task,'test',r['occurrence_id'])]['source_id'] for r in rawgold)
  gold=[r['gold'] for r in rawgold];vectors=trace['vectors'][task];table={};costs=trace['ledger']['by_task'][task]
  for identifier,reps in vectors.items():
   table[identifier]={'metadata':trace['metadata'][task][identifier],'repeats':{}}
   for repeat in (1,2):
    pred=reps[repeat];ref=vectors['singleton'][repeat]
    table[identifier]['repeats'][str(repeat)]={**evaluation(pred,ref,gold,identifier=='singleton'),
     'execution_cost':costs['executions'][identifier][str(repeat)]}
    for occurrence,(p,s,g) in enumerate(zip(pred,ref,gold),1):
     record=relation[(task,'test',occurrence)]
     prediction_rows.append({'model':model,'task':task,'phase':'test','configuration_id':identifier,'repeat':repeat,
      'occurrence_id':occurrence,'source_id':record['source_id'],'reserve_ordinal':record['reserve_ordinal'],
      'prediction':p,'reference_S_same_repeat':s,'gold':g})
  chosen={};contrasts={}
  for tolerance,_ in TOLERANCES:
   chosen[tolerance]={}
   for selector in ('Input','Output','Repeat'):
    matches=[p for p in policies if (p['model'],p['task'],p['tolerance'],p['selector'])==(model,task,tolerance,selector)]
    assert len(matches)==1
    chosen[tolerance][selector]=policy_result(matches[0],table,costs)
   contrasts[tolerance]=compare_policies(chosen[tolerance]['Input'],chosen[tolerance]['Output'])
  tasks[task]={'fixed_relation_size':128,'gold_positive_count':sum(gold),'all_configurations':table,
   'locked_policy_evaluation':chosen,'output_minus_input':contrasts,
   'singleton_reference_replay_diagnostic':disagreement(vectors['singleton'][1],vectors['singleton'][2]),
   'fixed_baselines':{'singleton':'singleton_execution','random_k8_C':config_id(8,'random','C'),'diversity_k8_C':config_id(8,'diversity','C')},
   'zero_call_constant_gold_references':{'Y':gold_metrics([1]*128,gold),'N':gold_metrics([0]*128,gold)}}
 result={'schema_version':1,'model':model,'policy_count_for_model':18,'release':release,'tasks':tasks,
  'test_research_union_cost':trace['ledger'],'test_control_evidence':trace['control_evidence'],
  'test_raw_sha256':trace['raw_sha256'],'test_submissions_sha256':trace['submissions_sha256'],'test_summary_sha256':trace['summary_sha256'],
  'gold_file_sha256':sha(bundle/'inputs/gold.jsonl'),'scientific_freeze_sha256':sha(bundle/'PREINFERENCE_FROZEN_SHA256.json'),
  'accounting_note':'The 36 policies and three tolerances reuse raw research calls. Do not sum their standalone calibration costs as executed research cost. Controls are separate.',
  'comparison_note':'Every batched/P execution r uses the same S_r reference. Singleton choices execute P_r; S has no self-fidelity score.',
  'scope':'Finite heldout relations with two replays; no confidence intervals, population test, test oracle, SOTA claim or one-query end-to-end savings.'}
 out.mkdir(exist_ok=False)
 write_json(out/'analysis.json',result);write_lines(out/'per_target.jsonl',prediction_rows)
 write_json(out/'ANALYSIS_SHA256.json',{'files':{name:sha(out/name) for name in ('analysis.json','per_target.jsonl')},
  'producer_sha256':sha(bundle/'analyze_test.py'),'policy_lock_sha256':release['policy_lock_sha256']})
 print(compact({'model':model,'analysis':str(out/'analysis.json'),'complete':trace['ledger']['phase_complete'],'policies':18}),flush=True)
 return result

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,default=ROOT);p.add_argument('--model',choices=MODELS,required=True);p.add_argument('--out',type=Path,required=True)
 a=p.parse_args();analyze(a.bundle,a.model,a.out)
