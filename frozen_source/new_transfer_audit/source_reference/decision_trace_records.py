"""Strict trace reconstruction and exact role-cost ledgers; no gold access."""
from collections import defaultdict
from common import *

def optional_lines(path):return read_lines(path) if path.exists() else []
def role_member(job,selector):
 if job['execution_type']=='S':return True
 if selector=='Repeat':return job['execution_type']=='C'
 return job['repeat']<=2 and job['execution_type'] in (('C','I') if selector=='Input' else ('C','O'))
def role_cost(jobs,returned):
 present=[returned[j['request_id']] for j in jobs if j['request_id'] in returned]
 return {'planned_calls':len(jobs),'returned_calls':len(present),'planned_requested_items':sum(len(j['occurrence_ids']) for j in jobs),
  'actual_input_tokens':sum(r['actual_input_tokens'] for r in present),'actual_output_tokens':sum(r['actual_output_tokens'] for r in present),
  'actual_total_tokens':sum(r['actual_input_tokens']+r['actual_output_tokens'] for r in present),
  'request_ids_sha256':digest(compact([j['request_id'] for j in jobs]))}

def validate_complete_evidence(bundle,run,phase,model,jobs,submissions,returned,invalid,diagnostics,diag_sub):
 names=('START.json','DATA_START.json','negative_controls.json','synthetic_smoke.json','diagnostics.jsonl','diagnostic_submissions.jsonl','blocks.jsonl','summary.json')
 if not all((run/name).is_file() for name in names):raise ValueError('Complete run lacks controls or execution provenance')
 start=json.loads((run/'START.json').read_text());data=json.loads((run/'DATA_START.json').read_text())
 negative=json.loads((run/'negative_controls.json').read_text());smoke=json.loads((run/'synthetic_smoke.json').read_text())
 summary=json.loads((run/'summary.json').read_text())
 for row in (start,data,summary):
  if row['phase']!=phase or row['model']!=model:raise ValueError('Provenance phase/model mismatch')
 if start['planned_scientific_jobs']!=len(jobs) or data['planned_requests']!=len(jobs):raise ValueError('Planned scientific count mismatch')
 if start['scientific_freeze_sha256']!=sha(bundle/'PREINFERENCE_FROZEN_SHA256.json') or start['request_sha256']!=sha(bundle/'prepared'/f'{model}_{phase}_requests.jsonl'):
  raise ValueError('Run does not bind frozen inputs')
 if data['negative_control_sha256']!=sha(run/'negative_controls.json') or data['synthetic_smoke_sha256']!=sha(run/'synthetic_smoke.json'):
  raise ValueError('DATA_START control hashes differ')
 if negative['valid'] is not True or smoke['valid'] is not True:raise ValueError('Pre-scientific controls did not pass')
 expected_groups=[('ordinary_banana',['diagnostic/ordinary_banana']),('guided_banana',[f'diagnostic/guided_banana/{i}' for i in range(5)]),
  ('synthetic_format_smoke',[f'diagnostic/smoke/{task}/k{k}/{kind}' for task in TASKS for k,kind in [(2,'C'),(2,'I'),(2,'O'),(4,'C'),(4,'I'),(4,'O'),(8,'C')]])]
 if [(b['name'],b['request_ids']) for b in diag_sub]!=expected_groups:raise ValueError('Diagnostic submission identities/order differ')
 expected_ids=[rid for name,ids in expected_groups for rid in ids]
 if len(diagnostics)!=20 or [r['request_id'] for r in diagnostics]!=expected_ids:raise ValueError('Diagnostic return identities/order differ')
 ids=[1,8,9,10,11,64,127,128]
 expected_outputs=[[],[128],[9,10,127,128],[128,127,10,9],ids,list(reversed(ids))]
 expected_outputs += [list(reversed(ids[-k:])) if kind=='O' else ids[-k:] for task in TASKS for k,kind in [(2,'C'),(2,'I'),(2,'O'),(4,'C'),(4,'I'),(4,'O'),(8,'C')]]
 if [r['output_ids'] for r in diagnostics]!=expected_outputs:raise ValueError('Diagnostic output-key contracts differ')
 if [negative['ordinary']]+negative['guided']+smoke['records']!=diagnostics:raise ValueError('Control reports differ from diagnostic raw records')
 for r in diagnostics:
  if r['diagnostic'] is not True or r['actual_output_tokens']!=len(r['output_token_ids']) or r['prompt_sha256']!=digest(r['wrapped_prompt']):raise ValueError('Diagnostic raw/token provenance mismatch')
  parsed=parse(r['raw'],r['output_ids']) if r['output_ids'] else None
  saved={int(k):v for k,v in r['labels'].items()} if r['labels'] is not None else None
  if parsed!=saved or r['strict_valid']!=(parsed is not None):raise ValueError('Diagnostic strict parse differs')
  if r['output_ids'] and (r['grammar']!=grammar(r['output_ids']) or r['grammar_sha256']!=digest(r['grammar']) or parsed is None):raise ValueError('Guided diagnostic failed')
 ordinary=negative['ordinary'];guided=negative['guided']
 if 'BANANA' not in ordinary['raw'] or any(parse(ordinary['raw'],r['output_ids']) is not None for r in guided):raise ValueError('Negative-control contrast failed')
 if negative['parser_checks']!=parser_checks() or smoke['semantic_labels_not_used_as_gate'] is not True:raise ValueError('Control contract differs')
 times=[start['started_unix']]+[b['submitted_unix'] for b in diag_sub]+[data['time_unix']]+[b['submitted_unix'] for b in submissions]+[summary['completed_unix']]
 if times!=sorted(times):raise ValueError('Controls did not precede scientific schedule')
 blocks=read_lines(run/'blocks.jsonl')
 if [(b['block'],b['requests']) for b in blocks]!=[(b['block'],len(b['request_ids'])) for b in submissions]:raise ValueError('Completed block ledger differs')
 scientific={k:sum(r['actual_'+k] for r in returned.values()) for k in ('input_tokens','output_tokens')}
 diag={k:sum(r['actual_'+k] for r in diagnostics) for k in ('input_tokens','output_tokens')}
 if (summary['status']!='complete' or summary['scientific_calls']!=len(jobs) or summary['valid_scientific_calls']!=len(jobs)-len(invalid)
  or summary['invalid_scientific_calls']!=len(invalid) or summary['diagnostic_calls']!=20 or summary['all_calls']!=len(jobs)+20
  or summary['scientific_observed_tokens']!=scientific or summary['diagnostic_observed_tokens']!=diag
  or summary['all_observed_tokens']!={k:scientific[k]+diag[k] for k in scientific}
  or summary['response_sha256']!=sha(run/'responses.jsonl') or summary['block_sha256']!=sha(run/'blocks.jsonl')):
  raise ValueError('Complete summary differs from raw execution evidence')
 return {name:sha(run/name) for name in names}

def load_trace(bundle,phase,model,require_complete=False):
 bundle=Path(bundle)
 schedule=read_lines(bundle/'prepared'/f'{phase}_schedule.jsonl')
 prepared=read_lines(bundle/'prepared'/f'{model}_{phase}_requests.jsonl')
 assert len(prepared)==len(schedule)
 jobs={j['request_id']:j for j in prepared}
 assert len(jobs)==len(schedule)
 for schedule_job in schedule:
  actual=jobs[schedule_job['request_id']]
  assert all(actual[k]==v for k,v in schedule_job.items())
 run=bundle/f'run_{phase}_{model}'
 submissions=optional_lines(run/'block_submissions.jsonl');submitted_list=[i for b in submissions for i in b['request_ids']]
 if [b['block'] for b in submissions]!=list(range(len(submissions))):raise ValueError('Submitted blocks are not the frozen schedule prefix')
 if len(set(submitted_list))!=len(submitted_list) or any(i not in jobs for i in submitted_list):raise ValueError('Unknown or duplicated submitted scientific request')
 for block in submissions:
  expected=[j['request_id'] for j in schedule if j['block']==block['block']]
  first=block['block']*64
  if not expected or block['request_ids']!=expected or block['first_ordinal']!=first or block['last_ordinal']!=first+len(expected)-1:
   raise ValueError('Submitted block differs from frozen membership or ordinal boundaries')
 submitted=set(submitted_list);raw=optional_lines(run/'responses.jsonl');returned={};vectors={task:{} for task in TASKS}
 metadata={task:{} for task in TASKS}
 for j in schedule:
  vectors[j['task']].setdefault(j['configuration_id'],{}).setdefault(j['repeat'],[None]*128)
  metadata[j['task']][j['configuration_id']]={k:j[k] for k in ('k','grouping','execution_type')}
 assigned=set();invalid=[]
 for r in raw:
  rid=r['request_id']
  if rid not in jobs or rid in returned or rid not in submitted:raise ValueError('Unknown, repeated, or unsubmitted returned scientific request')
  j=jobs[rid]
  for k in ('phase','task','configuration_id','k','grouping','execution_type','repeat','batch_index','occurrence_ids','ordinal','block','output_ids','requested_ids',
   'prompt_sha256','grammar_sha256','id_map_sha256'):
   if r[k]!=j[k]:raise ValueError('Raw response metadata mismatch: '+rid+'/'+k)
  if r['actual_input_tokens']!=len(j['prompt_token_ids']) or r['input_token_ids_sha256']!=digest(compact(j['prompt_token_ids'])):
   raise ValueError('Actual prompt token record differs')
  if r['actual_output_tokens']!=len(r['output_token_ids']):raise ValueError('Actual output token count differs')
  if r['diagnostic'] is not False:raise ValueError('Diagnostic record stored as scientific output')
  parsed=parse(r['raw'],j['output_ids'])
  saved={int(k):v for k,v in r['labels'].items()} if r['labels'] is not None else None
  if parsed!=saved or (parsed is not None)!=r['strict_valid']:raise ValueError('Saved parse disagrees with full untrimmed response')
  returned[rid]=r
  if parsed is None:invalid.append(rid)
  for occurrence in j['occurrence_ids']:
   key=j['task'],j['configuration_id'],j['repeat'],occurrence
   if key in assigned:raise ValueError('Tuple assigned twice in the same relation execution')
   assigned.add(key);vectors[j['task']][j['configuration_id']][j['repeat']][occurrence-1]=parsed[occurrence] if parsed is not None else None
 complete=len(returned)==len(jobs) and submitted==set(jobs)
 if require_complete and not complete:raise ValueError('Development union did not completely return; refusing to lock policies')
 diagnostics=optional_lines(run/'diagnostics.jsonl');diag_sub=optional_lines(run/'diagnostic_submissions.jsonl')
 diagnostic_submitted=[rid for b in diag_sub for rid in b['request_ids']]
 if len(set(diagnostic_submitted))!=len(diagnostic_submitted):raise ValueError('Diagnostic call repeated')
 control_evidence=validate_complete_evidence(bundle,run,phase,model,jobs,submissions,returned,invalid,diagnostics,diag_sub) if require_complete else None
 ledger={'planned_calls':len(jobs),'submitted_calls':len(submitted),'returned_calls':len(returned),'valid_calls':len(returned)-len(invalid),
  'invalid_request_ids':invalid,'submitted_unreturned_ids':sorted(submitted-set(returned)),
  'unsubmitted_ids':[j['request_id'] for j in schedule if j['request_id'] not in submitted],
  'diagnostic_submitted_calls':len(diagnostic_submitted),'diagnostic_returned_calls':len(diagnostics),
  'scientific_tokens':role_cost(list(jobs.values()),returned),
  'diagnostic_tokens':{'input':sum(r['actual_input_tokens'] for r in diagnostics),'output':sum(r['actual_output_tokens'] for r in diagnostics)},
  'phase_complete':complete,'by_task':{}}
 for task in TASKS:
  taskjobs=[j for j in prepared if j['task']==task]
  info={'research_union':role_cost(taskjobs,returned),'executions':{}}
  for identifier,reps in vectors[task].items():
   info['executions'][identifier]={str(repeat):role_cost([j for j in taskjobs if j['configuration_id']==identifier and j['repeat']==repeat],returned) for repeat in reps}
  if phase=='dev':
   info['selector_calibration']={s:role_cost([j for j in taskjobs if role_member(j,s)],returned) for s in ('Input','Output','Repeat')}
   for r in info['selector_calibration'].values():
    assert r['planned_calls']==1152 and r['planned_requested_items']==3328
  ledger['by_task'][task]=info
 return {'vectors':vectors,'metadata':metadata,'ledger':ledger,'control_evidence':control_evidence,'raw_sha256':sha(run/'responses.jsonl') if (run/'responses.jsonl').exists() else None,
  'submissions_sha256':sha(run/'block_submissions.jsonl') if (run/'block_submissions.jsonl').exists() else None,
  'summary_sha256':sha(run/'summary.json') if (run/'summary.json').exists() else None}
