#!/usr/bin/env python3
"""Pre-inference supplement A/B, separate from the four-family primary analysis."""
import argparse
import numpy as np
from common import *
from analyze import quality,estimate,paired_vectors,count_condition,count_pair,maybe_lines

CONDITIONS=[(s,c,j) for s in SCOPES for c in CONTEXTS for j in (1,2)]+[('singleton','none',j) for j in (1,2)]

def load_trace(bundle,model,mode):
 suffix='serial_' if mode=='serial' else ''
 schedule=read_lines(bundle/'prepared'/f'{suffix}schedule.jsonl');jobs={j['request_id']:j for j in schedule}
 run=bundle/f'run_{model}'/('serial' if mode=='serial' else '')
 rows=maybe_lines(run/'responses.jsonl');blocks=maybe_lines(run/'block_submissions.jsonl')
 submitted=[rid for block in blocks for rid in block['request_ids']]
 if len(set(submitted))!=len(submitted) or any(rid not in jobs for rid in submitted):raise ValueError('Duplicate/unknown submitted request')
 for block in blocks:
  expected=[j['request_id'] for j in schedule if j['block']==block['block']]
  if block['request_ids']!=expected or (mode=='serial' and len(expected)!=1):raise ValueError('Submission block mismatch')
 labels={};returned=set();invalid=[]
 for r in rows:
  rid=r['request_id']
  if rid not in jobs or rid in returned or rid not in submitted:raise ValueError('Unmapped/repeated/unsubmitted returned request')
  returned.add(rid);j=jobs[rid]
  for key in ('task','target_id','scope','context','repeat','ordinal','block','selected_ordinal'):
   if r[key]!=j[key]:raise ValueError('Response identity mismatch')
  p=parse(r['raw'],output_ids(j['scope']));v=int(p[1]=='Y') if p is not None else None
  if (p is not None)!=r['strict_valid'] or v!=r['target_label']:raise ValueError('Saved parse disagrees')
  labels[j['task'],j['target_id'],j['scope'],j['context'],j['repeat']]=v
  if v is None:invalid.append(rid)
 return labels,{'scheduled':len(schedule),'submitted':len(submitted),'returned':len(rows),'valid':len(rows)-len(invalid),
  'invalid_request_ids':invalid,'submitted_unreturned_ids':sorted(set(submitted)-returned),
  'unsubmitted_ids':[j['request_id'] for j in schedule if j['request_id'] not in submitted],
  'observed_tokens':{k:sum(r['actual_'+k] for r in rows) for k in ('input_tokens','output_tokens')}}

def label(labels,row,s,c,j):return labels.get((row['task'],row['target_id'],s,c,j))
def complete(rows,labels,conditions=CONDITIONS[:12]):
 return [r for r in rows if all(label(labels,r,*condition) is not None for condition in conditions)]
def scope_vectors(rows,labels,scope):
 arrays=[[label(labels,r,scope,c,j) for r in rows] for c,j in (('near',1),('near',2),('random',1),('random',2))]
 return paired_vectors(*arrays,[r['gold'] for r in rows])
def stable_changes(rows,labels,scope):
 valid=complete(rows,labels,[(scope,c,j) for c in CONTEXTS for j in (1,2)])
 stable=[]
 for r in valid:
  a,b,c,d=[label(labels,r,scope,context,j) for context,j in (('near',1),('near',2),('random',1),('random',2))]
  if a==b and c==d and a!=c:stable.append(r['target_id'])
 return {'four_call_valid_n':len(valid),'stable_change_count':len(stable),
  'stable_change_fraction':len(stable)/len(valid) if valid else None,'stable_change_target_ids':stable}

def reference_transport(rows,labels):
 details=[]
 for r in rows:
  refs=[label(labels,r,s,c,j) for s,c,j in [('singleton','none',1),('singleton','none',2),('full-first','random',1),('full-first','random',2)]]
  ref_valid=all(v is not None for v in refs);passed=ref_valid and len(set(refs))==1
  near=[label(labels,r,'full-first','near',j) for j in (1,2)]
  d={'target_id':r['target_id'],'selected_ordinal':r['selected_ordinal'],'reference_calls_valid':ref_valid,'passes':passed,
   'reference_labels':refs,'near_labels':near,'valid_near_calls':sum(v is not None for v in near),'gold':r['gold']}
  if passed:
   ref=refs[0];d['reference_label']=ref;d['reference_is_gold_error']=ref!=r['gold']
   if all(v is not None for v in near):
    d['any_flip']=any(v!=ref for v in near);d['stable_flip']=all(v!=ref for v in near)
    d['wrong_to_right_count_over_two_near_calls']=sum(ref!=r['gold'] and v==r['gold'] for v in near)
    d['right_to_wrong_count_over_two_near_calls']=sum(ref==r['gold'] and v!=r['gold'] for v in near)
  details.append(d)
 passing=[d for d in details if d['passes']];evaluable=[d for d in passing if d['valid_near_calls']==2]
 return {'selected_n':len(rows),'four_reference_calls_valid_n':sum(d['reference_calls_valid'] for d in details),
  'four_call_pass_n':len(passing),'passing_both_near_calls_valid_n':len(evaluable),
  'passing_invalid_or_unreturned_near_call_count':sum(2-d['valid_near_calls'] for d in passing),
  'passing_near_validity_counts':{str(k):sum(d['valid_near_calls']==k for d in passing) for k in (0,1,2)},
  'passing_not_evaluable_n':len(passing)-len(evaluable),
  'any_flip_count':sum(d['any_flip'] for d in evaluable),'stable_flip_count':sum(d['stable_flip'] for d in evaluable),
  'any_flip_fraction':sum(d['any_flip'] for d in evaluable)/len(evaluable) if evaluable else None,
  'stable_flip_fraction':sum(d['stable_flip'] for d in evaluable)/len(evaluable) if evaluable else None,
  'reference_gold_errors_on_all_passing':sum(d['reference_is_gold_error'] for d in passing),
  'reference_gold_error_fraction_on_all_passing':sum(d['reference_is_gold_error'] for d in passing)/len(passing) if passing else None,
  'reference_gold_errors_on_near_evaluable':sum(d['reference_is_gold_error'] for d in evaluable),
  'wrong_to_right_count_over_two_near_calls':sum(d['wrong_to_right_count_over_two_near_calls'] for d in evaluable),
  'right_to_wrong_count_over_two_near_calls':sum(d['right_to_wrong_count_over_two_near_calls'] for d in evaluable),
  'new_calls_added':0,'reference_calls_already_in_main':4*len(rows),'near_diagnostic_calls_already_in_main':2*len(rows),
  'details':details}

def subset_statistics(rows,labels):
 common12=complete(rows,labels);common14=complete(rows,labels,CONDITIONS);n=len(rows);gold=[r['gold'] for r in rows]
 draws=np.random.default_rng(SEEDS['bootstrap']).integers(0,len(common12),size=(4000,len(common12))) if common12 else None
 result={'selected_subset_n':n,'selected_subset_target_ids':[r['target_id'] for r in rows],
  'common12_n':len(common12),'common14_n':len(common14),'common12_target_ids':[r['target_id'] for r in common12],
  'common14_target_ids':[r['target_id'] for r in common14],'scopes':{},'scope_contrasts':{},
  'constants':{'all_N':quality([0]*n,gold),'all_Y':quality([1]*n,gold)},'condition_quality':[],
  'fixed_relation_conditions':[],'fixed_relation_comparisons':[],'stable_changes_own_four_call_valid':{}}
 vectors={}
 for scope in SCOPES:
  v=scope_vectors(common12,labels,scope);vectors[scope]=v
  result['scopes'][scope]={k:estimate(v[k],draws) for k in ('replacement','replay','excess','near_minus_random_error')}
  result['scopes'][scope]['mean_wrong_to_right_count_over_repeats']=float(v['wrong_to_right'].sum())
  result['scopes'][scope]['mean_right_to_wrong_count_over_repeats']=float(v['right_to_wrong'].sum())
  result['scopes'][scope]['stable_changes_common12']=stable_changes(common12,labels,scope)
  result['stable_changes_own_four_call_valid'][scope]=stable_changes(rows,labels,scope)
 for scope in ('target-only','full-last'):
  result['scope_contrasts'][scope+' minus full-first']=estimate(vectors[scope]['excess']-vectors['full-first']['excess'],draws)
 valid_predictions={}
 for scope,context,repeat in CONDITIONS:
  valid=[r for r in rows if label(labels,r,scope,context,repeat) is not None]
  p=[label(labels,r,scope,context,repeat) for r in valid];valid_predictions[scope,context,repeat]=p
  comparison=common14 if scope=='singleton' else common12
  result['condition_quality'].append({'scope':scope,'context':context,'repeat':repeat,
   'own_valid':quality(p,[r['gold'] for r in valid]),'common_set':'common14' if scope=='singleton' else 'common12',
   'common_quality':quality([label(labels,r,scope,context,repeat) for r in comparison],[r['gold'] for r in comparison])})
  result['fixed_relation_conditions'].append({'scope':scope,'context':context,'repeat':repeat,**count_condition(p,gold,n)})
 for scope in SCOPES:
  for name,c0,j0,c1,j1 in [('replacement_repeat1','random',1,'near',1),('replacement_repeat2','random',2,'near',2),
    ('replay_near','near',1,'near',2),('replay_random','random',1,'random',2)]:
   result['fixed_relation_comparisons'].append({'scope':scope,'comparison':name,
    **count_pair(valid_predictions[scope,c0,j0],valid_predictions[scope,c1,j1],gold,n)})
 return result

def paired_modes(rows,mixed,serial):
 common=complete(complete(rows,mixed),serial);gold=[r['gold'] for r in common]
 draws=np.random.default_rng(SEEDS['bootstrap']).integers(0,len(common),size=(4000,len(common))) if common else None
 result={'direction':'serial minus mixed','common12_both_modes_n':len(common),'common12_both_modes_target_ids':[r['target_id'] for r in common],
  'scopes':{},'paired_scope_contrast_differences':{},'paired_condition_error_differences':[]}
 differences={}
 for scope in SCOPES:
  a=scope_vectors(common,mixed,scope);b=scope_vectors(common,serial,scope)
  differences[scope]=b['excess']-a['excess']
  result['scopes'][scope]={k:estimate(b[k]-a[k],draws) for k in ('replacement','replay','excess','near_minus_random_error')}
 for scope in ('target-only','full-last'):
  result['paired_scope_contrast_differences'][scope+' minus full-first']=estimate(differences[scope]-differences['full-first'],draws)
 for scope,c,j in CONDITIONS[:12]:
  delta=np.array([float(label(serial,r,scope,c,j)!=r['gold'])-float(label(mixed,r,scope,c,j)!=r['gold']) for r in common])
  result['paired_condition_error_differences'].append({'scope':scope,'context':c,'repeat':j,**estimate(delta,draws)})
 return result

def main(bundle,model,out):
 bundle=Path(bundle);out=Path(out);out.mkdir(parents=True,exist_ok=False)
 targets=read_lines(bundle/'selection/targets.jsonl')
 mixed,mixed_ledger=load_trace(bundle,model,'mixed');serial,serial_ledger=load_trace(bundle,model,'serial')
 result={'model':model,'status':'pre-inference secondary supplement; no primary family claim',
  'subset_rule':'selected ordinals 0..31 per task','serial_follows_mixed':True,
  'completion':{'mixed':mixed_ledger,'serial':serial_ledger},'bootstrap':{'seed':SEEDS['bootstrap'],'draws':4000,
   'shared_indices':'same selected target rows across both modes for paired comparisons','ci_quantiles':[.025,.975],'quantile_method':'linear'},'tasks':{}}
 for task in TASKS:
  rows=sorted([r for r in targets if r['task']==task],key=lambda r:r['selected_ordinal']);subset=[r for r in rows if r['selected_ordinal']<32]
  result['tasks'][task]={'reference_transport':reference_transport(rows,mixed),
   'mixed_subset':subset_statistics(subset,mixed),'serial_subset':subset_statistics(subset,serial),
   'paired_serial_minus_mixed':paired_modes(subset,mixed,serial)}
 write_json(out/'secondary.json',result)
 print(compact({'model':model,'mixed_submitted':mixed_ledger['submitted'],'serial_submitted':serial_ledger['submitted'],
  'tasks':{t:{'probe_pass':r['reference_transport']['four_call_pass_n'],'near_evaluable':r['reference_transport']['passing_both_near_calls_valid_n'],
   'paired_modes_n':r['paired_serial_minus_mixed']['common12_both_modes_n']} for t,r in result['tasks'].items()}}),flush=True)
 return result

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,default=ROOT);p.add_argument('--model',choices=MODELS,required=True);p.add_argument('--out',type=Path,required=True)
 a=p.parse_args();main(a.bundle,a.model,a.out)
