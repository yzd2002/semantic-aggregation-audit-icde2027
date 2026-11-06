#!/usr/bin/env python3
"""SNLI raw-trace analysis; inherited estimands, standard library and NumPy, no model imports."""
import argparse,collections
import numpy as np
from common import *

def maybe_lines(path):return read_lines(path) if path.exists() else []
def quality(pred,gold):
 pred=np.asarray(pred,dtype=np.int8);gold=np.asarray(gold,dtype=np.int8);n=len(gold)
 tp=int(np.sum((pred==1)&(gold==1)));fp=int(np.sum((pred==1)&(gold==0)))
 tn=int(np.sum((pred==0)&(gold==0)));fn=int(np.sum((pred==0)&(gold==1)))
 recall=tp/(tp+fn) if tp+fn else None;specificity=tn/(tn+fp) if tn+fp else None
 return {'n':n,'tp':tp,'fp':fp,'tn':tn,'fn':fn,'errors':fp+fn,
  'error_rate':(fp+fn)/n if n else None,'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.0 if n else None,
  'positive_recall':recall,'negative_recall':specificity,
  'balanced_accuracy':(recall+specificity)/2 if recall is not None and specificity is not None else None,
  'predicted_positive_count':tp+fp,'reference_positive_count':tp+fn,'predicted_positive_fraction':(tp+fp)/n if n else None,
  'reference_error_cancellation_count':fp+fn-abs(fp-fn)}

def estimate(values,indices,primary=False):
 values=np.asarray(values,dtype=np.float64)
 if not len(values):return {'n':0,'mean':None,'ci95':None,**({'ci98_75':None,'positive_primary_evidence':False} if primary else {})}
 draws=values[indices].mean(axis=1)
 result={'n':len(values),'mean':float(values.mean()),'ci95':np.quantile(draws,[.025,.975],method='linear').tolist()}
 if primary:
  result['ci98_75']=np.quantile(draws,[.00625,.99375],method='linear').tolist()
  result['positive_primary_evidence']=result['ci98_75'][0]>0
 return result

def paired_vectors(a1,a2,b1,b2,gold):
 a1,a2,b1,b2,gold=[np.asarray(v,dtype=np.int8) for v in (a1,a2,b1,b2,gold)]
 replacement=((a1!=b1).astype(float)+(a2!=b2).astype(float))/2
 replay=((a1!=a2).astype(float)+(b1!=b2).astype(float))/2
 error=((a1!=gold).astype(float)+(a2!=gold).astype(float)-(b1!=gold).astype(float)-(b2!=gold).astype(float))/2
 improvement=(((b1!=gold)&(a1==gold)).astype(float)+((b2!=gold)&(a2==gold)).astype(float))/2
 worsening=(((b1==gold)&(a1!=gold)).astype(float)+((b2==gold)&(a2!=gold)).astype(float))/2
 assert np.array_equal(error,worsening-improvement)
 return {'replacement':replacement,'replay':replay,'excess':replacement-replay,
  'near_minus_random_error':error,'wrong_to_right':improvement,'right_to_wrong':worsening}

def count_condition(pred,gold,n_selected):
 if n_selected==0 or len(pred)!=n_selected:
  return {'status':'abstain','fixed_denominator':n_selected,'valid_targets':len(pred),'reason':'no_selected_targets' if n_selected==0 else 'not_all_selected_targets_valid'}
 count=int(sum(pred));reference=int(sum(gold));q=quality(pred,gold)
 return {'status':'complete','fixed_denominator':n_selected,'valid_targets':n_selected,
  'predicted_positive_count':count,'reference_positive_count':reference,'ratio':count/n_selected,
  'signed_count_error':count-reference,'absolute_count_error':abs(count-reference),'absolute_ratio_error':abs(count-reference)/n_selected,
  'false_positive_count':q['fp'],'false_negative_count':q['fn'],
  'candidate_reference_errors':q['errors'],'reference_error_cancellation_count':q['reference_error_cancellation_count']}

def count_pair(before,after,gold,n_selected):
 if n_selected==0 or len(before)!=n_selected or len(after)!=n_selected:
  return {'status':'abstain','fixed_denominator':n_selected,'reason':'at_least_one_fixed_relation_condition_incomplete'}
 before=np.asarray(before,dtype=np.int8);after=np.asarray(after,dtype=np.int8);gold=np.asarray(gold,dtype=np.int8)
 flips=int(np.sum(before!=after));ntoy=int(np.sum((before==0)&(after==1)));yton=int(np.sum((before==1)&(after==0)))
 signed=int(after.sum()-before.sum());wrong_right=int(np.sum((before!=gold)&(after==gold)));right_wrong=int(np.sum((before==gold)&(after!=gold)))
 assert signed==ntoy-yton and wrong_right+right_wrong==flips
 return {'status':'complete','fixed_denominator':n_selected,'before_positive_count':int(before.sum()),'after_positive_count':int(after.sum()),
  'changed_target_count':flips,'n_to_y':ntoy,'y_to_n':yton,'signed_count_change':signed,'absolute_count_change':abs(signed),
  'signed_ratio_change':signed/n_selected,'absolute_ratio_change':abs(signed)/n_selected,
  'reference_free_cancelled_flip_count':flips-abs(signed),'reference_free_cancelled_flip_fraction':(flips-abs(signed))/flips if flips else None,
  'wrong_to_right':wrong_right,'right_to_wrong':right_wrong,'signed_candidate_reference_error_change':right_wrong-wrong_right,
  'correctness_change_cancellation_count':flips-abs(right_wrong-wrong_right),
  'before_absolute_ratio_error':abs(int(before.sum()-gold.sum()))/n_selected,
  'after_absolute_ratio_error':abs(int(after.sum()-gold.sum()))/n_selected}

def analyze(bundle,model,out):
 bundle=Path(bundle);out=Path(out);out.mkdir(parents=True,exist_ok=False)
 targets=read_lines(bundle/'selection/targets.jsonl');schedule=read_lines(bundle/'prepared/schedule.jsonl')
 jobs={j['request_id']:j for j in schedule};run=bundle/f'run_{model}'
 raw=maybe_lines(run/'responses.jsonl');diagnostics=maybe_lines(run/'diagnostics.jsonl')
 submitted_blocks=maybe_lines(run/'block_submissions.jsonl');diag_submissions=maybe_lines(run/'diagnostic_submissions.jsonl')
 submitted_list=[x for block in submitted_blocks for x in block['request_ids']]
 if len(set(submitted_list))!=len(submitted_list) or any(x not in jobs for x in submitted_list):raise ValueError('Duplicate/unknown scientific submission ID')
 for block in submitted_blocks:
  expected=[j['request_id'] for j in schedule if j['block']==block['block']]
  if block['request_ids']!=expected:raise ValueError('Submitted block differs from frozen schedule')
 submitted=set(submitted_list)
 responses={};lookup={};invalid=[]
 for row in raw:
  rid=row['request_id']
  if rid not in jobs or rid in responses:raise ValueError('Unknown/duplicate scientific response '+rid)
  job=jobs[rid]
  for key in ('task','target_id','selected_ordinal','scope','context','repeat','ordinal','block'):
   if row[key]!=job[key]:raise ValueError('Response mapping differs at '+rid+'/'+key)
  values=parse(row['raw'],output_ids(job['scope']))
  label=int(values[1]=='Y') if values is not None else None
  if row.get('strict_valid')!=(values is not None) or row.get('target_label')!=label:raise ValueError('Saved parser disagrees with raw '+rid)
  responses[rid]={**row,'label':label}
  lookup[job['task'],job['target_id'],job['scope'],job['context'],job['repeat']]=label
  if label is None:invalid.append(rid)
 if any(rid not in submitted for rid in responses):raise ValueError('Returned scientific job lacks submission record')
 diag_scheduled=[rid for b in diag_submissions for rid in b['request_ids']]
 if len(set(diag_scheduled))!=len(diag_scheduled):raise ValueError('Repeated diagnostic submission ID')
 result={'model':model,'bootstrap':{'seed':SEEDS['bootstrap'],'draws':4000,'generator':'NumPy default_rng, reset per task/model',
  'order':'selected_ordinal ascending in common12','shared_indices':'all scopes, two paired scope contrasts and signed error differences',
  'quantile_method':'linear','ci95_quantiles':[.025,.975],'primary_ci98_75_quantiles':[.00625,.99375],
  'positive_primary_evidence_rule':'full-first 98.75% lower endpoint > 0',
  'new_family_models':list(MODELS),'nominal_alpha_per_primary':0.0125,'bonferroni_sum_bound':0.0375,
  'unified_old_and_new_FWER_claim':False,'scope':'approximate descriptive target bootstrap on selected targets'},
  'completion':{'scientific_scheduled':len(schedule),'scientific_submitted':len(submitted),'scientific_returned':len(raw),
   'scientific_valid':len(raw)-len(invalid),'scientific_invalid':len(invalid),'invalid_request_ids':invalid,
   'submitted_unreturned_request_ids':sorted(submitted-set(responses)),
   'unsubmitted_request_ids':[j['request_id'] for j in schedule if j['request_id'] not in submitted],
   'diagnostic_submitted':len(diag_scheduled),'diagnostic_returned':len(diagnostics),
   'diagnostic_submitted_unreturned_ids':sorted(set(diag_scheduled)-{d['request_id'] for d in diagnostics})},
  'costs':{'scientific_observed_tokens':{k:sum(r['actual_'+k] for r in raw) for k in ('input_tokens','output_tokens')},
   'diagnostic_observed_tokens':{k:sum(r['actual_'+k] for r in diagnostics) for k in ('input_tokens','output_tokens')},
   'all_submitted_calls':len(submitted)+len(diag_scheduled),'all_returned_calls':len(raw)+len(diagnostics),
   'all_observed_tokens':{k:sum(r['actual_'+k] for r in raw+diagnostics) for k in ('input_tokens','output_tokens')}},
  'input_hashes':{'targets':sha(bundle/'selection/targets.jsonl'),'schedule':sha(bundle/'prepared/schedule.jsonl'),
   'responses':sha(run/'responses.jsonl') if (run/'responses.jsonl').exists() else None},'tasks':{}}
 per_target=[]
 conditions=[(s,c,j) for s in SCOPES for c in CONTEXTS for j in (1,2)]+[('singleton','none',j) for j in (1,2)]
 for task in TASKS:
  rows=sorted([r for r in targets if r['task']==task],key=lambda r:r['selected_ordinal']);n=len(rows)
  def value(r,s,c,j):return lookup.get((task,r['target_id'],s,c,j))
  common12=[r for r in rows if all(value(r,s,c,j) is not None for s,c,j in conditions[:12])]
  common14=[r for r in common12 if all(value(r,'singleton','none',j) is not None for j in (1,2))]
  gold=[r['gold'] for r in rows];cg=[r['gold'] for r in common12]
  indices=np.random.default_rng(SEEDS['bootstrap']).integers(0,len(common12),size=(4000,len(common12))) if common12 else None
  task_result={'selected_targets':n,'reference_positive_count':sum(gold),'common12_n':len(common12),'common14_n':len(common14),
   'common12_target_ids':[r['target_id'] for r in common12],'common14_target_ids':[r['target_id'] for r in common14],
   'constants_fixed_targets':{'all_N':quality([0]*n,gold),'all_Y':quality([1]*n,gold)},
   'condition_quality':[],'scopes':{},'scope_contrasts':{},'fixed_relation_conditions':[],'fixed_relation_comparisons':[],
   'fixed_relation_ratio_summaries':{},'singleton_comparisons':[]}
  valid_predictions={}
  for s,c,j in conditions:
   valid_rows=[r for r in rows if value(r,s,c,j) is not None]
   preds=[value(r,s,c,j) for r in valid_rows];g=[r['gold'] for r in valid_rows]
   valid_predictions[s,c,j]=preds
   common=common14 if s=='singleton' else common12
   task_result['condition_quality'].append({'scope':s,'context':c,'repeat':j,'selected_denominator':n,
    'own_valid':quality(preds,g),'common_set':'common14' if s=='singleton' else 'common12',
    'common_quality':quality([value(r,s,c,j) for r in common],[r['gold'] for r in common])})
   aggregate=count_condition(preds,gold,n)
   aggregate.update(scope=s,context=c,repeat=j,missing_target_ids=[r['target_id'] for r in rows if value(r,s,c,j) is None])
   task_result['fixed_relation_conditions'].append(aggregate)
  vectors={}
  for scope in SCOPES:
   arrays=[[value(r,scope,c,j) for r in common12] for c,j in (('near',1),('near',2),('random',1),('random',2))]
   v=paired_vectors(*arrays,cg);vectors[scope]=v
   task_result['scopes'][scope]={'replacement':estimate(v['replacement'],indices),'replay':estimate(v['replay'],indices),
    'excess':estimate(v['excess'],indices,scope=='full-first'),'near_minus_random_error':estimate(v['near_minus_random_error'],indices),
    'mean_wrong_to_right_count_over_repeats':float(v['wrong_to_right'].sum()),
    'mean_right_to_wrong_count_over_repeats':float(v['right_to_wrong'].sum()),
    'per_repeat_correctness_changes':[]}
   for repeat in (1,2):
    pair=count_pair([value(r,scope,'random',repeat) for r in common12],[value(r,scope,'near',repeat) for r in common12],cg,len(common12))
    task_result['scopes'][scope]['per_repeat_correctness_changes'].append({'repeat':repeat,**pair})
   for comparison,from_c,from_j,to_c,to_j in [('replacement_repeat1','random',1,'near',1),('replacement_repeat2','random',2,'near',2),
    ('replay_near','near',1,'near',2),('replay_random','random',1,'random',2)]:
    pair=count_pair(valid_predictions[scope,from_c,from_j],valid_predictions[scope,to_c,to_j],gold,n)
    task_result['fixed_relation_comparisons'].append({'scope':scope,'comparison':comparison,'from_context':from_c,'from_repeat':from_j,'to_context':to_c,'to_repeat':to_j,**pair})
   entries=[r for r in task_result['fixed_relation_comparisons'] if r['scope']==scope]
   if all(r['status']=='complete' for r in entries):
    replacement=sum(r['absolute_ratio_change'] for r in entries[:2])/2;replay=sum(r['absolute_ratio_change'] for r in entries[2:])/2
    task_result['fixed_relation_ratio_summaries'][scope]={'status':'complete','fixed_denominator':n,
     'mean_absolute_replacement_ratio_change':replacement,'mean_absolute_replay_ratio_change':replay,
     'excess_absolute_ratio_change_descriptive':replacement-replay}
   else:task_result['fixed_relation_ratio_summaries'][scope]={'status':'abstain','fixed_denominator':n}
  for scope in ('target-only','full-last'):
   task_result['scope_contrasts'][scope+' minus full-first']=estimate(vectors[scope]['excess']-vectors['full-first']['excess'],indices)
  for i,r in enumerate(common12):
   per_target.append({'task':task,'target_id':r['target_id'],'selected_ordinal':r['selected_ordinal'],'gold':r['gold'],
    'scopes':{s:{k:float(v[i]) for k,v in vectors[s].items()} for s in SCOPES}})
  for scope in SCOPES:
   for context in CONTEXTS:
    for repeat in (1,2):
     pair=count_pair([value(r,'singleton','none',repeat) for r in common14],[value(r,scope,context,repeat) for r in common14],
      [r['gold'] for r in common14],len(common14))
     task_result['singleton_comparisons'].append({'scope':scope,'context':context,'repeat':repeat,'set':'common14','n':len(common14),**pair})
  task_result['singleton_replay_common14']=count_pair([value(r,'singleton','none',1) for r in common14],[value(r,'singleton','none',2) for r in common14],
   [r['gold'] for r in common14],len(common14))
  task_result['singleton_replay_fixed_relation']=count_pair(valid_predictions['singleton','none',1],valid_predictions['singleton','none',2],gold,n)
  result['tasks'][task]=task_result
 write_json(out/'analysis.json',result);write_lines(out/'per_target.jsonl',per_target)
 print(compact({'model':model,'scheduled':len(schedule),'submitted':len(submitted),'returned':len(raw),
  'tasks':{t:{'selected':r['selected_targets'],'common12':r['common12_n'],'common14':r['common14_n'],
   'full_first_excess':r['scopes']['full-first']['excess']} for t,r in result['tasks'].items()}}),flush=True)
 return result

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,default=ROOT);p.add_argument('--model',choices=MODELS,required=True);p.add_argument('--out',type=Path,required=True)
 a=p.parse_args();analyze(a.bundle,a.model,a.out)
