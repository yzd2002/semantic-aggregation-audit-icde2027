"""Independent nontrivial synthetic raw-ledger fixtures; never model judgments."""
import os,sys,pathlib,json,hashlib,copy,contextlib,io,itertools,time,types
os.environ['CUDA_VISIBLE_DEVICES']='';sys.dont_write_bytecode=True
P=pathlib.Path('/home/gpu/IntentSat2S_revision_20261106/new_glm_context');O=pathlib.Path(__file__).resolve().parent/os.environ.get('REVIEW_OUTPUT_SUBDIR','analysis_v3_candidate_checks')
O.mkdir(exist_ok=False)
sys.path.insert(0,str(P));import common as c
import numpy as np
from analyze import analyze,paired_vectors,count_condition,count_pair
from analyze_fresh_replication import analyze_all,load_unique,RAW_DIRECTORIES
from secondary import load_trace
from generation_receipts import GenerationReceiptLedger
SOURCE_START={n:c.sha(P/n) for n in ('analyze_fresh_replication.py','generation_receipts.py','analyze.py','secondary.py','common.py')}
F=O/'ANALYSIS_SYNTHETIC_NOT_MODEL_RUN';F.mkdir(exist_ok=False)
for name in ('prepared','selection','historical_context_arm'):(F/name).symlink_to(P/name,target_is_directory=True)
def save(p,rs):p.write_text(''.join(json.dumps(r,ensure_ascii=False,separators=(',',':'))+'\n' for r in rs))
def rd(p):return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
def derivedlabel(j):
 n=j.get('selected_ordinal',0);repeat=j.get('repeat',1)
 if n%4==0:return repeat-1
 if n%4==1:return int(j.get('context')=='random')
 if n%4==2:return 0
 return int(hashlib.sha256(j['prompt_sha256'].encode()).hexdigest()[0],16)%2
jobs_by={};all_effectives={};returned_by={};all_results={}
with contextlib.redirect_stdout((O/'analysis_fixture_console.log').open('w')):
 for model in c.MODELS:
  root=F/f'run_{model}';root.mkdir();env=json.loads((P/'prepared'/f'REQUEST_ENVELOPE_{model}.json').read_text());cfg=json.loads((P/'prepared'/f'runtime_configuration_{model}.json').read_text());effective=[];counter=0;modes={};byresponse={}
  def row(job,diag=False):
   nonlocal_placeholder=None
   global counter
   engine=f'INDEPENDENT_CPU_FIXTURE/{model}/{counter}';counter+=1;label=derivedlabel(job)
   raw='BANANA' if not job['output_ids'] else json.dumps({str(i):('Y' if (label if i==1 else i%2) else 'N') for i in job['output_ids']},separators=(',',':'))
   valid=bool(job['output_ids']);target=label if 1 in job['output_ids'] else None
   keys=('request_id','task','target_id','selected_ordinal','scope','context','repeat','ordinal','block','requested_ids','output_ids','prompt_sha256','grammar_sha256','id_map_sha256','source_main_request_id','source_historical_request_id','execution_mode')
   r={k:job[k] for k in keys if k in job};r.update(engine_request_id=engine,raw=raw,raw_sha256=c.digest(raw),strict_valid=valid,target_label=target,actual_input_tokens=len(job['prompt_token_ids']),actual_output_tokens=1,output_token_ids=[100],input_token_ids_sha256=c.digest(c.compact(job['prompt_token_ids'])),diagnostic=diag,synthetic_cpu_fixture_not_a_model_observation=True)
   effective.append(dict(engine_request_id=engine,prompt_token_ids_sha256=r['input_token_ids_sha256'],actual_processed_controls=dict(n=1,temperature=0,top_p=1,top_k=0,max_tokens=128,all_stop_token_ids=cfg['stop_token_ids'],primary_eos_token_id=cfg['primary_eos_token_id'],guided_backend='xgrammar' if job.get('grammar') else None,guided_disable_fallback=True if job.get('grammar') else None,grammar_sha256=job.get('grammar_sha256'))))
   return r
  for mode,e in env['modes'].items():
   jobs=rd(P/e['file']);modes[mode]=jobs;dest=root/RAW_DIRECTORIES[mode];dest.mkdir(exist_ok=True);rows=[row(j) for j in jobs];byresponse[mode]=rows
   save(dest/'responses.jsonl',rows);save(dest/'block_submissions.jsonl',[dict(block=b,request_ids=[j['request_id'] for j in jobs if j['block']==b]) for b in sorted({j['block'] for j in jobs})])
  dg=rd(P/'prepared'/f'{model}_diagnostic_requests.jsonl');diagrows=[row(j,True) for j in dg];save(root/'diagnostics.jsonl',diagrows);save(root/'diagnostic_submissions.jsonl',[dict(request_ids=[j['request_id'] for j in dg])]);save(root/'effective_requests.jsonl',effective)
  # Exercise the actual durable-receipt and verified-binding code at full budget.
  ledger=GenerationReceiptLedger(root,model);current_jobs=[];current_rows=[];effective_lookup={r['engine_request_id']:r for r in effective}
  class FakeReceiptEngine:
   def generate(self,prompts,parameters,**kw):
    values=[]
    for job,rawrow in zip(current_jobs,current_rows):
     req=types.SimpleNamespace(request_id=rawrow['engine_request_id'],prompt_token_ids=job['prompt_token_ids'])
     ledger.note_processed(req,{'grammar_sha256':job.get('grammar_sha256')})
     values.append(types.SimpleNamespace(request_id=req.request_id,prompt_token_ids=req.prompt_token_ids,outputs=[types.SimpleNamespace(text=rawrow['raw'],token_ids=rawrow['output_token_ids'],finish_reason='stop',stop_reason=None,index=0)]))
    return values
  engine=FakeReceiptEngine()
  batches=[]
  for group in ('ordinary_banana','guided_banana','synthetic_format_smoke'):
   indices=[i for i,j in enumerate(dg) if j['diagnostic_group']==group];batches.append(('diagnostic/'+group,[dg[i] for i in indices],[diagrows[i] for i in indices],True))
  for mode,jobs in modes.items():
   for block in sorted({j['block'] for j in jobs}):
    indices=[i for i,j in enumerate(jobs) if j['block']==block];batches.append((mode,[jobs[i] for i in indices],[byresponse[mode][i] for i in indices],False))
  for mode,current_jobs,current_rows,isdiag in batches:
   values,receipt=ledger.generate(engine,current_jobs,[],mode,diagnostic=isdiag)
   for position,rawrow in enumerate(current_rows):
    rawrow.update(receipt,raw_return_position=position);effective_lookup[rawrow['engine_request_id']]['generation_call_id']=receipt['generation_call_id']
  for mode,rows in byresponse.items():save(root/RAW_DIRECTORIES[mode]/'responses.jsonl',rows)
  save(root/'diagnostics.jsonl',diagrows);save(root/'effective_requests.jsonl',effective)
  jobs_by[model]=modes;all_effectives[model]=effective;returned_by[model]=byresponse
  result=analyze_all(F,model,O/f'analysis_complete_{model}');cost=result['costs'];assert cost['unique_scientific_submitted']==cost['unique_scientific_returned']==4750;assert cost['unique_all_submitted']==cost['unique_all_returned']==4770 and cost['engine_processed_requests']==4770
  assert cost['derived_view_rows_add_no_calls'] and cost['actual_generation_return_accounting']['observed_returned_requests']==4770;all_results[model]=result
  # Independently calculate raw target bits, e_t, bootstrap CIs, total-relation COUNTs.
  lookup={(j['task'],j['target_id'],j['scope'],j['context'],j['repeat']):derivedlabel(j) for j in modes['mixed']}
  targetrows=rd(P/'selection/targets.jsonl')
  for task in c.TASKS:
   ts=[r for r in targetrows if r['task']==task];taskout=result['primary']['tasks'][task];assert taskout['common12_n']==taskout['common14_n']==128
   draws=np.random.default_rng(2026110204).integers(0,128,size=(4000,128))
   for scope in c.SCOPES:
    values=[]
    for t in ts:
     a1,a2,b1,b2=[lookup[task,t['target_id'],scope,ct,j] for ct,j in [('near',1),('near',2),('random',1),('random',2)]]
     values.append(((a1!=b1)+(a2!=b2)-(a1!=a2)-(b1!=b2))/2)
    values=np.array(values);out=taskout['scopes'][scope]['excess'];assert out['mean']==float(values.mean());assert np.allclose(out['ci95'],np.quantile(values[draws].mean(axis=1),[.025,.975],method='linear'),rtol=0,atol=0)
    if scope=='full-first':assert np.allclose(out['ci98_75'],np.quantile(values[draws].mean(axis=1),[.00625,.99375],method='linear'),rtol=0,atol=0)
    assert min(values)==-1, 'negative individual effects must survive'
   for cond in taskout['fixed_relation_conditions']:
    pred=[lookup[task,t['target_id'],cond['scope'],cond['context'],cond['repeat']] for t in ts];assert cond['status']=='complete' and cond['fixed_denominator']==128 and cond['predicted_positive_count']==sum(pred) and cond['ratio']==sum(pred)/128
   # Every historical logical response must equal the uniquely mapped raw target bit.
   view=O/f'analysis_complete_{model}/derived_historical_view';links=rd(P/'prepared'/f'HISTORICAL_SHARED_CALL_MAP_{model}.jsonl');observations={md:{r['request_id']:r for r in rs} for md,rs in byresponse.items()}
   for md,subdir in [('mixed',''),('serial','serial')]:
    projected={r['request_id']:r for r in rd(view/f'run_{model}'/subdir/'responses.jsonl')}
    for link in links:
     if link['historical_mode']!=md:continue
     raw=observations[link['raw_mode']][link['raw_request_id']];assert projected[link['historical_request_id']]['raw']==raw['raw'] and projected[link['historical_request_id']]['target_label']==raw['target_label']
# Exhaustive binary arithmetic tests independent of any synthesized ledger pattern.
for a1,a2,b1,b2,gold in itertools.product((0,1),repeat=5):
 out=paired_vectors([a1],[a2],[b1],[b2],[gold]);expected=((a1!=b1)+(a2!=b2)-(a1!=a2)-(b1!=b2))/2
 assert out['excess'][0]==expected
 assert out['near_minus_random_error'][0]==((a1!=gold)+(a2!=gold)-(b1!=gold)-(b2!=gold))/2
 assert out['wrong_to_right'][0]==(((b1!=gold) and (a1==gold))+((b2!=gold) and (a2==gold)))/2
 assert out['right_to_wrong'][0]==(((b1==gold) and (a1!=gold))+((b2==gold) and (a2!=gold)))/2
q=count_pair([0,1],[1,0],[0,0],2);assert q['changed_target_count']==2 and q['absolute_count_change']==0 and q['reference_free_cancelled_flip_count']==2
assert count_condition([],[],0)['status']=='abstain' and count_pair([],[],[],0)['status']=='abstain'
# All valid JSON assignments + adversarial format variants.
parser_cases=0
for ids in ([1],[1,2,3,4],[2,3,4,1],[9,10,11],[11]):
 for bits in itertools.product('YN',repeat=len(ids)):
  raw=json.dumps(dict(zip(map(str,ids),bits)),separators=(',',':'));assert c.parse(raw,ids)==dict(zip(ids,bits));parser_cases+=1
  for bad in (' '+raw,raw+'\n',raw+' ',raw.replace(':',': ',1),raw.replace('Y','yes').replace('N','no')):assert c.parse(bad,ids) is None;parser_cases+=1
for raw in ('{"1":"Y","1":"N"}','{"1":"Y","2":"N"}','{"01":"Y"}','{"1":"y"}','{"1":true}','{"1":"N"}\x00'):
 assert c.parse(raw,[1]) is None;parser_cases+=1
# A missing shared near raw reduces main/historical common sets but NEVER fixed relation denominator.
model='glm';run=F/'run_glm';base=returned_by[model]
def analyse_variant(name):
 with contextlib.redirect_stdout((O/'analysis_fixture_console.log').open('a')):return analyze_all(F,model,O/name)
def reset(mode):save(run/RAW_DIRECTORIES[mode]/'responses.jsonl',base[mode])
changed={ (r['task'],r['target_id']) for r in rd(P/'selection/targets.jsonl') if r['glm_primary_random_context_redrawn']}
missing=next(r for r in base['mixed'] if r['task']=='boolq' and (r['task'],r['target_id']) in changed and r['selected_ordinal']<32 and r['scope']=='full-first' and r['context']=='near' and r['repeat']==1)
save(run/'responses.jsonl',[r for r in base['mixed'] if r['request_id']!=missing['request_id']]);result=analyse_variant('analysis_missing_shared_near')
assert result['costs']['unique_scientific_submitted']==4750 and result['costs']['unique_scientific_returned']==4749
for arm in ('primary','historical'):
 t=result[arm]['tasks']['boolq'];assert t['common12_n']==127 and t['selected_targets']==128
 cond=next(r for r in t['fixed_relation_conditions'] if (r['scope'],r['context'],r['repeat'])==('full-first','near',1));assert cond['status']=='abstain' and cond['fixed_denominator']==128 and cond['valid_targets']==127
 assert t['fixed_relation_ratio_summaries']['full-first']['status']=='abstain'
reset('mixed')
# Missing exclusive historical serial row must not remove the primary serial row.
missing2=next(r for r in base['historical-extra-serial'] if r['task']=='boolq' and r['scope']=='target-only')
save(run/'historical_extra_serial/responses.jsonl',[r for r in base['historical-extra-serial'] if r['request_id']!=missing2['request_id']]);result=analyse_variant('analysis_missing_historical_serial')
assert result['costs']['unique_scientific_submitted']==4750 and result['costs']['unique_scientific_returned']==4749
histserial=json.loads((O/'analysis_missing_historical_serial/historical_secondary/secondary.json').read_text())['tasks']['boolq']['serial_subset'];primaryserial=json.loads((O/'analysis_missing_historical_serial/primary_secondary/secondary.json').read_text())['tasks']['boolq']['serial_subset']
assert histserial['selected_subset_n']==32 and histserial['common12_n']==31 and primaryserial['common12_n']==32
cond=next(r for r in histserial['fixed_relation_conditions'] if (r['scope'],r['context'],r['repeat'])==(missing2['scope'],missing2['context'],missing2['repeat']));assert cond['status']=='abstain' and cond['fixed_denominator']==32
reset('historical-extra-serial')
# Missing singleton leaves common12 intact and reduces common14 only.
missing3=next(r for r in base['mixed'] if r['task']=='boolq' and r['scope']=='singleton')
save(run/'responses.jsonl',[r for r in base['mixed'] if r['request_id']!=missing3['request_id']]);result=analyse_variant('analysis_missing_singleton');t=result['primary']['tasks']['boolq'];assert t['common12_n']==128 and t['common14_n']==127 and t['singleton_replay_fixed_relation']['status']=='abstain';reset('mixed')
# Wrong prompt binding and guidance fallback must fail closed.
negative=[]
corrupt=copy.deepcopy(base['mixed']);corrupt[0]['prompt_sha256']='f'*64;save(run/'responses.jsonl',corrupt)
try:load_unique(F,'glm');raise AssertionError('corrupt prompt accepted')
except ValueError as e:negative.append({'case':'corrupt_prompt_sha','error':str(e)})
reset('mixed');eff=copy.deepcopy(all_effectives['glm']);eff[0]['actual_processed_controls']['guided_disable_fallback']=False;save(run/'effective_requests.jsonl',eff)
try:load_unique(F,'glm');raise AssertionError('fallback accepted')
except ValueError as e:negative.append({'case':'guided_fallback_enabled','error':str(e)})
save(run/'effective_requests.jsonl',all_effectives['glm'])
assert SOURCE_START=={n:c.sha(P/n) for n in SOURCE_START},'reviewed analysis sources changed during execution'
summary={'reviewed_sources':SOURCE_START,'full_budget_actual_durable_ledger_binding_exercised':True,'CPU_only':True,'fixture_is_not_model_evidence':True,'all_three_models_complete_raw_ledgers_checked':True,'per_model_unique_science':4750,'per_model_diagnostic':20,'per_model_historical_logical_references_not_new_calls':4480,'independent_all_scopes_effect_and_shared_bootstrap_recomputed':True,'negative_per_target_excess_retained':True,'truth_table_combinations':32,'parser_positive_negative_cases':parser_cases,'COUNT_cancellation_separated_from_target_flips':True,'missing_shared_near_MAIN_HISTORY_fixed_denominator_128_abstains':True,'missing_exclusive_history_serial_fixed_denominator_32_abstains_primary_unaffected':True,'missing_singleton_common12_128_common14_127':True,'negative_rejections':negative,'script_sha256':c.sha(__file__),'analysis_source_sha256':c.sha(P/'analyze_fresh_replication.py'),'created_unix':time.time()}
(O/'independent_analysis_v3_results.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary))
