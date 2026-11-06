import os,sys,pathlib,json,hashlib,copy,contextlib,itertools,importlib.util,runpy,numpy as np
os.environ['CUDA_VISIBLE_DEVICES']='';sys.dont_write_bytecode=True
P=pathlib.Path('/home/gpu/IntentSat2S_revision_20261106/new_glm_context');O=pathlib.Path(__file__).resolve().parent/os.environ.get('REVIEW_OUTPUT_SUBDIR','global_wrapper_candidate_checks');O.mkdir(exist_ok=False)
S=pathlib.Path(__file__).resolve().parent/'analysis_v3_candidate_checks/ANALYSIS_SYNTHETIC_NOT_MODEL_RUN'
sys.path.insert(0,str(P));import common as c
from analyze_fresh_replication import analyze_all,load_unique
files=('analyze_fresh_replication.py','assemble_context_primary.py','generation_receipts.py','global_primary/primary_family.py','global_primary/GLOBAL_PRIMARY_AMENDMENT.md');start={f:c.sha(P/f) for f in files}
assert start['global_primary/primary_family.py']=='9661951458cc592be20f92411c23f40a6b7a55afbb3d60c284393c1a5c581ab2'
assert start['global_primary/GLOBAL_PRIMARY_AMENDMENT.md']=='cb0dbba13ffe8f2b2fc5b8efc3e4989f2a9496c4e93792ce1ea9dd4138ce32b4'
def rd(p):return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
def save(p,rows):p.write_text(''.join(c.compact(r)+'\n' for r in rows))
components=[];allcases=[]
with contextlib.redirect_stdout((O/'console.log').open('w')):
 for model in c.MODELS:
  result=analyze_all(S,model,O/f'complete_{model}');cost=result['costs'];assert cost['unique_all_returned']==4770 and cost['actual_generation_return_accounting']['observed_returned_requests']==4770
  assert cost['all_observed_tokens']['input_tokens']==json.loads((P/'prepared'/f'REQUEST_ENVELOPE_{model}.json').read_text())['all_input_tokens']
  path=O/f'complete_{model}/global_primary_component.json';component=json.loads(path.read_text());assert len(component['cases'])==2 and not component['complete_nine_case_suite'] and component['legacy_ci98_75_flags_are_not_authoritative']
  labels={}
  for row in rd(S/f'run_{model}/responses.jsonl'):
   parsed=json.loads(row['raw']);labels[row['task'],row['target_id'],row['scope'],row['context'],row['repeat']]=int(parsed['1']=='Y')
  for case in component['cases']:
   task=case['task'];rows=sorted([r for r in rd(P/'selection/targets.jsonl') if r['task']==task],key=lambda r:r['selected_ordinal']);assert case['common12_target_ids']==[r['target_id'] for r in rows]
   values=[]
   for r in rows:
    a1,a2,b1,b2=[labels[task,r['target_id'],'full-first',ctx,j] for ctx,j in [('near',1),('near',2),('random',1),('random',2)]];values.append(((a1!=b1)+(a2!=b2)-(a1!=a2)-(b1!=b2))/2)
   x=np.array(values);seed=int.from_bytes(hashlib.sha256(f'2026110613|{task}|{model}'.encode()).digest()[:8],'big');draws=x[np.random.default_rng(seed).integers(0,128,(4000,128))].mean(axis=1);interval=np.quantile(draws,[1/360,359/360],method='linear').tolist()
   assert case['mean_excess']==float(x.mean()) and case['interval']==interval and case['bootstrap_seed']==seed and case['positive_primary_evidence']==(interval[0]>0)
   assert case['source_per_target_sha256']==c.sha(O/f'complete_{model}/primary_main/per_target.jsonl') and case['source_primary_analysis_sha256']==c.sha(O/f'complete_{model}/primary_main/analysis.json')
   assert case['planned_target_n']==case['common12_complete_n']==128 and case['excluded_n']==0 and case['excluded_target_ids']==[]
   allcases.append(case)
  components.append(path)
 # Six-case assembler includes exactly context data and cannot pass a duplicate model file.
 sys.argv=['assemble_context_primary.py','--components',*map(str,components),'--out',str(O/'six_context_cases.json')];runpy.run_path(str(P/'assemble_context_primary.py'),run_name='__main__')
 assembled=json.loads((O/'six_context_cases.json').read_text());assert len(assembled['cases'])==6 and not assembled['complete_nine_case_suite'] and all(r['task']!='snli' for r in assembled['cases'])
 duplicate_rejected=False
 try:
  sys.argv=['assemble_context_primary.py','--components',str(components[0]),str(components[0]),str(components[2]),'--out',str(O/'MUST_NOT_EXIST.json')];runpy.run_path(str(P/'assemble_context_primary.py'),run_name='__main__')
 except ValueError:duplicate_rejected=True
 assert duplicate_rejected and not (O/'MUST_NOT_EXIST.json').exists()
 # No real/canonical outputs yields honest NA primary components and COUNT abstention.
 empty=O/'EMPTY_NOT_MODEL_RUN';empty.mkdir()
 for name in ('prepared','selection','historical_context_arm'):(empty/name).symlink_to(P/name,target_is_directory=True)
 result=analyze_all(empty,'glm',O/'empty_results');ec=json.loads((O/'empty_results/global_primary_component.json').read_text());assert result['costs']['unique_all_returned']==0
 for case in ec['cases']:assert not case['available'] and case['common12_complete_n']==0 and case['excluded_n']==128 and case['interval'] is None and not case['positive_primary_evidence']
 for task in c.TASKS:
  assert all(r['status']=='abstain' and r['fixed_denominator']==128 for r in result['primary']['tasks'][task]['fixed_relation_conditions'])
# Corrupt only our own synthetic raw ledger and restore byte-for-byte in finally.
run=S/'run_glm';rawpath=run/'responses.jsonl';original=rawpath.read_bytes();baseline=rd(rawpath);tests=[]
try:
 for name in ('duplicate_engine_id_same_prompt','wrong_return_position','changed_raw_label','wrong_generation_call'):
  rows=copy.deepcopy(baseline);i=next(i for i,r in enumerate(rows) if r['scope']=='full-first' and r['context']=='near' and r['repeat']==1)
  r=rows[i];j=next(j for j,q in enumerate(rows) if q['task']==r['task'] and q['target_id']==r['target_id'] and q['scope']==r['scope'] and q['context']==r['context'] and q['repeat']==2)
  assert r['input_token_ids_sha256']==rows[j]['input_token_ids_sha256'] and r['grammar_sha256']==rows[j]['grammar_sha256']
  if name=='duplicate_engine_id_same_prompt':rows[j]['engine_request_id']=r['engine_request_id']
  if name=='wrong_return_position':r['raw_return_position']=(r['raw_return_position']+1)%64
  if name=='changed_raw_label':
   values=json.loads(r['raw']);values['1']='N' if values['1']=='Y' else 'Y';r['raw']=c.compact(values);r['raw_sha256']=c.digest(r['raw']);r['target_label']=int(values['1']=='Y')
  if name=='wrong_generation_call':r['generation_call_id']=rows[j]['generation_call_id']
  save(rawpath,rows);message=None
  try:load_unique(S,'glm')
  except ValueError as e:message=str(e)
  assert message is not None,name;tests.append({'case':name,'rejection':message})
finally:rawpath.write_bytes(original)
assert rawpath.read_bytes()==original
# Missing a singleton historical-serial raw-return receipt keeps total token cost unknown.
rawpath2=run/'historical_extra_serial/responses.jsonl';receiptpath=run/'raw_return_receipts.jsonl';savedraw=rawpath2.read_bytes();savedreceipt=receiptpath.read_bytes()
try:
 rows=rd(rawpath2);missing=rows[-1];save(rawpath2,rows[:-1]);save(receiptpath,[r for r in rd(receiptpath) if r['generation_call_id']!=missing['generation_call_id']])
 obs,sub,paths,cost=load_unique(S,'glm');assert cost['unique_scientific_submitted']==4750 and cost['unique_scientific_returned']==4749 and cost['all_observed_tokens']=={'input_tokens':None,'output_tokens':None} and cost['costs_are_unknown_rather_than_zero_when_any_invocation_is_unresolved']
finally:rawpath2.write_bytes(savedraw);receiptpath.write_bytes(savedreceipt)
assert start=={f:c.sha(P/f) for f in files},'source drift while reviewing integration'
summary={'CPU_only':True,'source_sha256':start,'full_three_model_durable_receipt_costs_and_six_context_primary_components_verified':True,'all_common12_e_vectors_rebuilt_directly_from_raw_JSON':True,'new_seeds_and_179_180_intervals_independently_recomputed':True,'source_per_target_and_analysis_hash_binding_verified':True,'six_case_assembler_preserves_SNLI_missingness':True,'duplicate_component_rejected':duplicate_rejected,'empty_all_128_targets_unavailable_and_COUNT_abstains':True,'canonical_integrity_negative_tests':tests,'unreturned_raw_receipt_costs_unknown_not_zero':True,'synthetic_source_files_restored_exactly':True,'script_sha256':c.sha(__file__)}
(O/'global_wrapper_results.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary))
