"""Read actual durable transfer returns and recompute all finite-panel mathematics.

No producer modules are imported. Two pure gold/fidelity oracles are AST-isolated
from the independently certified baseline95 reviewer, not the producer.
"""
from pathlib import Path
from fractions import Fraction
import argparse,ast,collections,hashlib,json,math,os,re,shutil,statistics,sys,time,traceback
os.environ['CUDA_VISIBLE_DEVICES']=''
try:os.nice(10)
except (AttributeError,OSError):pass
R=Path(os.environ.get('I2S_TRANSFER_ROOT','/home/gpu/IntentSat2S_revision_20261106/new_transfer_audit'))
B=Path(os.environ.get('I2S_BASELINE_REVIEW_ROOT','/home/gpu/IntentSat2S_revision_20261106/independent_transfer_preflight_review'))
FINAL='83ca3beb7488e4fd1c5998d5d6f38a6003286ad183dbf61c179a40d46672d38a'
def compact(x):return json.dumps(x,ensure_ascii=False,separators=(',',':'))
def digest(x):return hashlib.sha256(x.encode()).hexdigest()
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def load(p):return json.loads(p.read_text())
def stream(p):
 with p.open() as f:
  for t in f:
   if t.strip():yield json.loads(t)
def write(p,x):
 with p.open('x') as f:json.dump(x,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
def valid_int(x):return type(x) is int and x>=0
def parse(raw,ids):
 if not ids:return None
 assert len(set(ids))==len(ids) and all(type(i) is int and i>0 for i in ids)
 rule=r'\{'+','.join('"'+str(i)+'":"[YN]"' for i in ids)+r'\}'
 if not isinstance(raw,str) or re.fullmatch(rule,raw) is None:return None
 pairs=json.loads(raw,object_pairs_hook=list)
 assert [int(k) for k,v in pairs]==ids
 return {int(k):int(v=='Y') for k,v in pairs}
# Extract only already independently reviewed functions. Top-level imports and test
# drivers in that reviewer script are deliberately never executed.
helper=B/'runner_and_math_review.py';helper_manifest=load(B/'BASELINE95_EVIDENCE_SHA256.json')
assert sha(helper)==helper_manifest['files']['runner_and_math_review.py']
helper_ast=ast.parse(helper.read_text());pure=[n for n in helper_ast.body if isinstance(n,ast.FunctionDef) and n.name in {'own_gold','own_fidelity'}]
assert len(pure)==2
space={'collections':collections};exec(compile(ast.Module(body=pure,type_ignores=[]),'<independent_baseline95_pure_oracles>','exec'),space)
gold_metric=space['own_gold'];fidelity=space['own_fidelity']
METRICS=('error','accuracy','balanced_accuracy','f1','absolute_ratio_error','predicted_positive_fraction','signed_count_bias','canceled_gold_error_mass')
def avg(values):return None if not values or any(x is None for x in values) else statistics.fmean(values)
def summary(replays,tau=None):
 f=[replays[str(i)]['singleton_fidelity'] for i in (1,2)];complete=all(x is not None and x['complete'] for x in f)
 flags={k:None for k in ('MEAN','BOTH')}
 if complete and tau is not None:
  exact=[Fraction(x['numerator'],128) for x in f]
  flags={'MEAN':sum(exact)/2<=tau,'BOTH':all(x<=tau for x in exact)}
 means={k:avg([replays[str(i)]['gold'][k] for i in (1,2)]) for k in METRICS}
 means['mean_confusion_counts']={k:avg([replays[str(i)]['gold']['confusion_counts_on_valid_outcomes'][k] if replays[str(i)]['gold']['complete'] else None for i in (1,2)]) for k in ('tp','tn','fp','fn')}
 return {'mean_S_disagreement':avg([x['value'] if x is not None else None for x in f]),'within_tolerance_MEAN':flags['MEAN'],'within_tolerance_BOTH':flags['BOTH'],'public_gold_mean_over_two_replays':means,'replays_are_execution_diagnostics_not_independent_calibrations':True}
def aggregate(relations):
 ss=[v['summary'] for v in relations.values()]
 out={'relation_count_fixed_denominator':len(ss),'mean_S_disagreement_over_relations':avg([v['mean_S_disagreement'] for v in ss]),'public_gold_relation_macro_mean':{k:avg([v['public_gold_mean_over_two_replays'][k] for v in ss]) for k in METRICS},'public_gold_relation_macro_confusion_counts':{k:avg([v['public_gold_mean_over_two_replays']['mean_confusion_counts'][k] for v in ss]) for k in ('tp','tn','fp','fn')}}
 for kind in ('MEAN','BOTH'):
  f=[v['within_tolerance_'+kind] for v in ss];passed=f.count(True);out['tolerance_'+kind]={'pass':passed,'fail':f.count(False),'unknown':f.count(None),'fixed_relation_denominator':len(ss),'pass_fraction':None if None in f else passed/len(ss)}
 return out

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--model',choices=['smol','qwen'],required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();out=a.out;out.mkdir(parents=True,exist_ok=False);model=a.model;run=R/f'run_{model}';analysis=R/f'analysis_{model}/analysis.json';started=time.time();shutil.copy2(Path(__file__),out/'AUDIT_SOURCE_SNAPSHOT.py')
 paths=['PREINFERENCE_FROZEN_SHA256.json','POLICY_LOCK.json','inputs/relations.jsonl','inputs/gold.jsonl','prepared/runtime_observation_contract.json','runtime_observer.py','verify_effective_runtime.py','analyze_transfer.py',f'prepared/{model}_requests.jsonl',f'prepared/{model}_schedule.jsonl',f'prepared/{model}_controls.jsonl']
 paths += [str(p.relative_to(R)) for p in sorted(run.glob('*')) if p.is_file()]
 paths += [str(analysis.relative_to(R))]
 sources={n:sha(R/n) for n in paths};assert sources['PREINFERENCE_FROZEN_SHA256.json']==FINAL
 frozen=load(R/'PREINFERENCE_FROZEN_SHA256.json');assert all(sources[n]==frozen['files'][n] for n in paths if n in frozen['files'])
 assert not (run/'FAILURE.json').exists();got=load(analysis);start=load(run/'START.json');done=load(run/'summary.json');contract=load(R/'prepared/runtime_observation_contract.json');native=load(run/'EFFECTIVE_NATIVE_RUNTIME.json');expected_native=contract['models'][model]
 assert done['status']=='complete' and got['model']==model and start['model']==model and start['freeze_sha256']==FINAL
 assert got['freeze_sha256']==FINAL and got['policy_lock_sha256']==sources['POLICY_LOCK.json']
 assert start['request_sha256']==sources[f'prepared/{model}_requests.jsonl'] and start['runtime_observation_contract_sha256']==sources['prepared/runtime_observation_contract.json']
 assert contract['runtime_observer_sha256']==sources['runtime_observer.py']
 worker=native['worker_identity'];assert len(worker)==1 and worker[0]['model_class']==expected_native['native_class'] and worker[0]['model_module']==expected_native['native_module'] and worker[0]['parameter_dtypes']==['torch.bfloat16'] and not worker[0]['remote_model_modules_loaded'] and worker[0]['parameters']>0
 assert native['native_hf_config_class']==expected_native['native_hf_config_class'] and native['dtype']=='torch.bfloat16' and native['model_impl_requested'] in ['auto','vllm']
 for k,v in {'max_model_len':4096,'max_num_seqs':16,'tensor_parallel_size':1,'enforce_eager':True,'prefix_caching':False,'gpu_memory_utilization':.7,'guided_decoding_backend':'xgrammar','guided_decoding_disable_fallback':True}.items():assert native[k]==v
 jobs=list(stream(R/f'prepared/{model}_schedule.jsonl'));controls=list(stream(R/f'prepared/{model}_controls.jsonl'));byjob={j['request_id']:j for j in jobs};assert len(byjob)==len(jobs)
 assert len(controls)==20
 jobgroups=[('diagnostic/'+name,None,[j for j in controls if j['name']==name]) for name in ('ordinary_banana','guided_banana','synthetic_format_smoke')]
 jobgroups += [('science',k//64,jobs[k:k+64]) for k in range(0,len(jobs),64)]
 planned={j['request_id']:j for _,_,batch in jobgroups for j in batch};assert len(planned)==len(jobs)+20
 prepared={}
 for ix,j in enumerate(stream(R/f'prepared/{model}_requests.jsonl')):
  schedule=jobs[ix];assert all(j[k]==v for k,v in schedule.items()) and j['ordinal']==ix
  ids=sorted(j['occurrence_ids'],reverse=j['execution_type']=='O');assert ids==j['output_ids']
  assert j['prompt_sha256']==digest(j['wrapped_prompt']) and j['grammar_sha256']==digest(j['grammar']) and j['prompt_tokens_preflight']==len(j['prompt_token_ids'])
  prepared[j['request_id']]={'ids':ids,'prompt_digest':digest(compact(j['prompt_token_ids'])),'length':len(j['prompt_token_ids']),'grammar_digest':digest(j['grammar']),'id_map_sha256':j['id_map_sha256'],'prompt_sha256':j['prompt_sha256']}
 assert len(prepared)==len(jobs)
 for j in controls:prepared[j['request_id']]={'ids':j['output_ids'],'prompt_digest':digest(compact(j['prompt_token_ids'])),'length':len(j['prompt_token_ids']),'grammar_digest':None if j.get('guided') is False else digest(j['grammar']),'id_map_sha256':j.get('id_map_sha256'),'prompt_sha256':j['prompt_sha256']}
 receipts=done['effective_runtime_observation_receipts'];names={'EFFECTIVE_NATIVE_RUNTIME.json','effective_requests.jsonl','effective_batch_submissions.jsonl','effective_return_bindings.jsonl','effective_binding_validation.jsonl'};assert set(receipts['files'])==names
 for name,h in receipts['files'].items():assert sha(run/name)==h
 assert receipts['bound_science_and_control_requests']==len(planned) and receipts['active_batch_unbound'] is False
 effective=list(stream(run/'effective_requests.jsonl'));eids=set();effective_map={}
 assert [e['candidate_logical_request_id'] for e in effective]==[j['request_id'] for _,_,batch in jobgroups for j in batch]
 for e in effective:
  rid=e['candidate_logical_request_id'];p=prepared[rid];c=e['effective_controls'];assert e['engine_request_id'] not in eids;eids.add(e['engine_request_id']);effective_map[rid]=e
  assert e['prompt_token_ids_sha256']==p['prompt_digest']
  for k,v in {'n':1,'temperature':0,'top_p':1,'top_k':0,'min_p':0,'max_tokens':128,'ignore_eos':False,'repetition_penalty':1,'presence_penalty':0,'frequency_penalty':0,'seed':None,'skip_special_tokens':True,'spaces_between_special_tokens':True,'stop':[],'sampling_type':'GREEDY','primary_eos_token_id':expected_native['primary_eos_token_id'],'all_stop_token_ids':expected_native['all_stop_token_ids'],'stop_token_ids':[i for i in expected_native['all_stop_token_ids'] if i!=expected_native['primary_eos_token_id']],'grammar_sha256':p['grammar_digest']}.items():assert c[k]==v,(rid,k)
  assert c['guided_backend']==('xgrammar' if p['grammar_digest'] is not None else None) and c['guided_disable_fallback']==(True if p['grammar_digest'] is not None else None)
 bindings=list(stream(run/'effective_return_bindings.jsonl'));validations=list(stream(run/'effective_binding_validation.jsonl'));submissions=list(stream(run/'effective_batch_submissions.jsonl'));assert len(bindings)==len(validations)==len(submissions)==len(jobgroups)
 raw={};journal_counts=collections.Counter();journal_tokens=collections.Counter();seen_engines=set();byte_offset=0
 with (run/'engine_returns.jsonl').open('rb') as f:
  for index,(stage,block,batch) in enumerate(jobgroups):
   segment=f.readline();assert segment;entry=json.loads(segment);b=bindings[index];v=validations[index];s=submissions[index];rids=[j['request_id'] for j in batch]
   assert entry['stage']==b['stage']==v['stage']==s['stage']==stage and entry['block']==b['block']==v['block']==s['block']==block
   assert entry['expected_batch_request_ids']==s['logical_request_ids']==v['logical_request_ids']==rids and v['validated'] is True
   assert b['journal_byte_offset']==byte_offset and b['journal_byte_length']==len(segment) and b['journal_segment_sha256']==hashlib.sha256(segment).hexdigest();byte_offset+=len(segment)
   assert entry['journal_written_before_count_identity_or_parser_validation'] is True and entry['actual_return_count']==len(entry['actual_returns'])==len(batch)
   assert len(b['bindings'])==len(batch)
   for position,(j,n,bind) in enumerate(zip(batch,entry['actual_returns'],b['bindings'])):
    rid=j['request_id'];p=prepared[rid];e=effective_map[rid];assert rid not in raw and n['engine_request_id'] not in seen_engines;seen_engines.add(n['engine_request_id'])
    assert n['return_position']==position==e['position'];assert bind['logical_request_id']==rid and bind['stage']==stage and bind['block']==block
    assert bind['engine_request_id']==bind['processed_engine_request_id']==n['engine_request_id']==e['engine_request_id']
    ph=digest(compact(n['actual_prompt_token_ids']));assert ph==p['prompt_digest']==bind['returned_prompt_token_ids_sha256']==bind['processed_prompt_token_ids_sha256'] and bind['grammar_sha256']==p['grammar_digest']
    assert n['actual_input_tokens']==len(n['actual_prompt_token_ids'])==p['length'] and all(valid_int(i) for i in n['actual_prompt_token_ids'])
    assert len(n['outputs'])==1;answer=n['outputs'][0];assert answer['output_position']==0 and isinstance(answer['raw'],str) and isinstance(answer['output_token_ids'],list) and all(valid_int(i) for i in answer['output_token_ids'])
    assert n['actual_output_tokens']==len(answer['output_token_ids'])<=128
    raw[rid]={'engine_id':n['engine_request_id'],'input_hash':ph,'input_tokens':n['actual_input_tokens'],'output_tokens':n['actual_output_tokens'],'answer':answer,'parsed':parse(answer['raw'],p['ids']),'stage':stage}
    kind='science' if stage=='science' else 'control';journal_counts[kind]+=1;journal_tokens[kind+'_input']+=n['actual_input_tokens'];journal_tokens[kind+'_output']+=n['actual_output_tokens']
  assert f.read()==b''
 assert set(raw)==set(planned) and len(seen_engines)==len(planned)
 canonical={};invalid=[]
 for diagnostic,name in [(False,'responses.jsonl'),(True,'diagnostics.jsonl')]:
  expected_ids=[j['request_id'] for j in (controls if diagnostic else jobs)];seen=[]
  for c in stream(run/name):
   rid=c['request_id'];assert rid not in canonical;seen.append(rid);j=planned[rid];n=raw[rid];p=prepared[rid]
   assert c['engine_request_id']==n['engine_id'] and c['input_token_ids_sha256']==n['input_hash'] and c['actual_input_tokens']==n['input_tokens'] and c['actual_output_tokens']==n['output_tokens']
   for k in ('raw','output_token_ids','finish_reason','stop_reason'):assert c[k]==n['answer'][k]
   assert c['labels']==({str(k):v for k,v in n['parsed'].items()} if n['parsed'] is not None else None) and c['strict_valid'] is (n['parsed'] is not None)
   assert c['diagnostic'] is diagnostic and all(c[k]==v for k,v in j.items() if k in c)
   if not diagnostic:
    assert all(c[k]==v for k,v in j.items()) and c['id_map_sha256']==p['id_map_sha256'] and c['prompt_sha256']==p['prompt_sha256'] and c['grammar_sha256']==p['grammar_digest']
    if n['parsed'] is None:invalid.append(rid)
   canonical[rid]=c
  assert seen==expected_ids
 assert not invalid
 # Counterfactual parse alone never determines the row IDs: frozen occurrence IDs
 # and exact native prompt identity above define where predictions are placed.
 vectors=collections.defaultdict(lambda:collections.defaultdict(lambda:collections.defaultdict(lambda:[None]*128)));cost_index=collections.defaultdict(list)
 gold={};source_ids={}
 for j in stream(R/'inputs/relations.jsonl'):source_ids[j['relation_id'],j['occurrence_id']]=j['source_id']
 for j in stream(R/'inputs/gold.jsonl'):
  assert source_ids[j['relation_id'],j['occurrence_id']]==j['source_id'];gold.setdefault(j['relation_id'],[None]*128)[j['occurrence_id']-1]=j['gold']
 assert len(gold)==23 and all(len(g)==128 and all(type(x) is int and x in (0,1) for x in g) for g in gold.values())
 for j in jobs:
  key=(j['relation_id'],j['configuration_id'],j['repeat']);cost_index[key].append(j['request_id']);vec=vectors[key[0]][key[1]][key[2]]
  for i in j['occurrence_ids']:assert vec[i-1] is None;vec[i-1]=raw[j['request_id']]['parsed'][i]
 def cost(ids,items=None):
  records=[raw[i] for i in ids];n=len(ids);ni=sum(r['input_tokens'] for r in records);no=sum(r['output_tokens'] for r in records)
  return {'planned_calls':n,'submitted_calls':n,'returned_calls':n,'planned_requested_items':sum(len(byjob[i]['occurrence_ids']) for i in ids) if items is None else items,'actual_input_tokens':ni,'observed_input_token_lower_bound':ni,'unknown_input_token_call_count':0,'actual_output_tokens':no,'observed_output_token_lower_bound':no,'unknown_output_token_call_count':0,'actual_total_tokens':ni+no,'observed_total_token_lower_bound':ni+no,'token_cost_complete':True}
 compared=collections.Counter();mismatches=[]
 def compare(actual,want,path):
  compared['all_nodes']+=1
  if isinstance(want,dict):
   if not isinstance(actual,dict) or set(actual)!=set(want):mismatches.append({'path':path,'error':'dict keys differ','actual_keys':list(actual) if isinstance(actual,dict) else None,'expected_keys':list(want)});return
   for k,v in want.items():compare(actual[k],v,path+'/'+str(k))
  elif isinstance(want,list):
   if not isinstance(actual,list) or len(actual)!=len(want):mismatches.append({'path':path,'error':'list lengths differ'});return
   for i,(x,y) in enumerate(zip(actual,want)):compare(x,y,path+'/'+str(i))
  elif type(want) is float:
   compared['numeric_fields']+=1
   if type(actual) not in (float,int) or not math.isclose(actual,want,rel_tol=1e-12,abs_tol=1e-12):mismatches.append({'path':path,'actual':actual,'expected':want})
  else:
   if type(want) is int:compared['numeric_fields']+=1
   if type(actual)!=type(want) or actual!=want:mismatches.append({'path':path,'actual':actual,'expected':want})
 def evaluate(pred,ref,g,reference=False):
  gm=gold_metric(pred,g);sf=None if reference else fidelity(pred,ref);tr=None
  if not reference and gm['complete'] and sf['complete']:
   repairs=sum(y==z and x!=z for y,x,z in zip(pred,ref,g));damage=sum(y!=z and x==z for y,x,z in zip(pred,ref,g));delta=sum(pred)-sum(ref);flips=sf['numerator']
   tr={'reference_wrong_execution_right':repairs,'reference_right_execution_wrong':damage,'execution_minus_reference_gold_error':Fraction(damage-repairs,128).__float__(),'signed_count_change_from_reference':delta,'absolute_ratio_change_from_reference':abs(delta)/128,'candidate_flips':flips,'opposing_flip_pairs':(flips-abs(delta))//2}
  return {'gold':gm,'singleton_fidelity':sf,'fidelity_reference':'not_applicable_reference_output' if reference else 'fresh_S_same_repeat','gold_and_count_transport_from_reference':tr}
 evaluated={};per_target=[]
 for rel in sorted(vectors):
  evaluated[rel]={}
  for cfg in sorted(vectors[rel]):
   reps={}
   for repeat in (1,2):
    pred=vectors[rel][cfg][repeat];ref=vectors[rel]['singleton'][repeat];assert all(type(p) is int and p in (0,1) for p in pred)
    ev=evaluate(pred,ref,gold[rel],cfg=='singleton');ev['execution_cost']=cost(cost_index[rel,cfg,repeat]);reps[str(repeat)]=ev
    for i,(p,s,g) in enumerate(zip(pred,ref,gold[rel]),1):per_target.append({'model':model,'relation_id':rel,'configuration_id':cfg,'repeat':repeat,'occurrence_id':i,'source_id':source_ids[rel,i],'prediction':p,'S_reference':s,'gold':g})
   evaluated[rel][cfg]={'replays':reps,'summary':summary(reps)}
 compare(got['all_executed_configurations'],evaluated,'all_executed_configurations')
 lock=load(R/'POLICY_LOCK.json');policies=[p for p in lock['payload']['policies'] if p['model']==model];assert len(policies)==18
 def execution_ids(rels,cfg,repeat):return [rid for rel in rels for rid in cost_index[rel,cfg,repeat]]
 def prefix(relations,cfg,cal=None):
  result=[]
  for m in range(1,len(relations)+1):
   rr=relations[:m];rep={}
   for r in (1,2):
    ex=cost(execution_ids(rr,cfg,r));pb=cost(execution_ids(rr,'singleton_execution',r));row={'independent_P_baseline':pb}
    if cal is None:row.update(execution_cost=ex,planned_calls_difference_vs_P=ex['planned_calls']-pb['planned_calls'],observed_tokens_difference_vs_P=ex['actual_total_tokens']-pb['actual_total_tokens'])
    else:
     total={k:cal[k]+ex[k] for k in ['planned_calls','returned_calls','planned_requested_items','actual_input_tokens','actual_output_tokens','actual_total_tokens']};total.update(unknown_execution_input_token_call_count=0,unknown_execution_output_token_call_count=0,calibration_counted_once=True)
     row.update(execution=ex,old_calibration_once_plus_execution=total,planned_calls_difference_vs_P=total['planned_calls']-pb['planned_calls'],calls_amortized=total['planned_calls']<=pb['planned_calls'],observed_tokens_difference_vs_P=total['actual_total_tokens']-pb['actual_total_tokens'],tokens_amortized=total['actual_total_tokens']<=pb['actual_total_tokens'])
    rep[str(r)]=row
   result.append({'prefix_relations':m,**({'calibration_calls':0} if cal is None else {'relation_ids':rr}),'replays':rep})
  return result
 results={};policy_views=[]
 for p in policies:
  pid='/'.join(p[k] for k in ('model','task','tolerance','selector'));cfg=p['chosen']['configuration_id'];cfg='singleton_execution' if cfg=='singleton' else cfg;rels=sorted(r for r in vectors if r.startswith(p['task']+'/'));tau=Fraction(p['tolerance']);assert tau==Fraction(p['tolerance_numerator'],p['tolerance_denominator']);rr={}
  for rel in rels:
   reps={}
   for repeat in (1,2):
    ev=dict(evaluated[rel][cfg]['replays'][str(repeat)]);f=ev['singleton_fidelity'];ev['empirical_tolerance_satisfied']=None if not f['complete'] else Fraction(f['numerator'],128)<=tau;reps[str(repeat)]=ev
   rr[rel]={'replays':reps,'summary':summary(reps,tau)}
  ag=aggregate(rr);px=prefix(rels,cfg,p['observed_standalone_calibration_cost']);gp=got['all_policies'][pid];compare(gp['policy'],p,'policy/'+pid);compare(gp['executed_configuration_id'],cfg,'executed/'+pid);compare(gp['relations'],rr,'relations/'+pid);compare(gp['aggregate'],ag,'aggregate/'+pid);compare(gp['all_prefix_costs'],px,'prefix/'+pid)
  k=p['chosen']['k'];be=gp['call_break_even_arithmetic']
  if k==1:assert be['equal_or_lower_call_cost_relations'] is None
  else:assert be['equal_or_lower_call_cost_relations']==math.ceil(Fraction(1152,128-128//k))
  assert gp['calibration_counted_once_per_prefix'] is True and gp['new_calibration_calls']==0
  results[pid]={'configuration':cfg,'relations':rr,'aggregate':ag,'prefix':px}
  policy_views.append({'policy':pid,'configuration':cfg,'relations':len(rels),'aggregate':ag,'final_prefix_costs':px[-1],'calibration_cost':p['observed_standalone_calibration_cost'],'identical_plan_is_shared_observation_not_new_study':True})
 fixed_views={}
 for task in ('dblp_acm','boolq'):
  rels=sorted(r for r in vectors if r.startswith(task+'/'));fixed_views[task]={}
  for label,cfg in [('P_baseline','singleton_execution'),('random_k8_C','k8/random/C'),('diversity_k8_C','k8/diversity/C'),('S_reference','singleton')]:
   rr={r:evaluated[r][cfg] for r in rels};f={'calibration_calls':0,'configuration_id':cfg,'relations':rr,'aggregate':aggregate(rr),'all_prefix_execution_costs':prefix(rels,cfg)};compare(got['fixed_zero_calibration_references'][task][label],f,'fixed/'+task+'/'+label);fixed_views[task][label]={'aggregate':f['aggregate'],'final_prefix':f['all_prefix_execution_costs'][-1]}
  constants={r:{'Y':gold_metric([1]*128,gold[r]),'N':gold_metric([0]*128,gold[r])} for r in rels};compare(got['fixed_zero_calibration_references'][task]['zero_call_gold_constants'],constants,'constants/'+task)
 contrasts={}
 for task in ('dblp_acm','boolq'):
  for tol in ('0.05','0.10','0.20'):
   aa=results[f'{model}/{task}/{tol}/Input'];bb=results[f'{model}/{task}/{tol}/Output'];rows={}
   for rel in aa['relations']:
    rows[rel]={}
    for repeat in ('1','2'):
     av=aa['relations'][rel]['replays'][repeat];bv=bb['relations'][rel]['replays'][repeat];rows[rel][repeat]={'output_minus_input_S_disagreement':bv['singleton_fidelity']['value']-av['singleton_fidelity']['value'],'output_minus_input_execution_calls':bv['execution_cost']['planned_calls']-av['execution_cost']['planned_calls']}
   contrasts[task+'/'+tol]={'same_locked_plan':aa['configuration']==bb['configuration'],'per_relation_replay':rows}
 compare(got['output_minus_input'],contrasts,'output_minus_input')
 union=got['physical_execution_union_cost'];compare(union['science_union'],cost(list(byjob)),'physical_science_union');compare(union['controls'],cost([j['request_id'] for j in controls],0),'physical_controls');assert union['phase_complete'] is True and not union['invalid_request_ids'] and not union['missing_request_ids'] and union['failure'] is None
 for field,want in [('scientific_calls',len(jobs)),('valid_scientific_calls',len(jobs)),('invalid_scientific_calls',0),('diagnostic_calls',20),('all_calls',len(planned))]:compare(done[field],want,'summary/'+field)
 for field,group in [('scientific_observed_tokens','science'),('diagnostic_observed_tokens','control')]:compare(done[field],{'input_tokens':journal_tokens[group+'_input'],'output_tokens':journal_tokens[group+'_output']},'summary/'+field)
 compare(done['all_observed_tokens'],{'input_tokens':journal_tokens['science_input']+journal_tokens['control_input'],'output_tokens':journal_tokens['science_output']+journal_tokens['control_output']},'summary/all_tokens')
 assert done['response_sha256']==sha(run/'responses.jsonl') and done['engine_return_journal_sha256']==sha(run/'engine_returns.jsonl') and done['blocks_sha256']==sha(run/'blocks.jsonl')
 compare(got['original_development_union_cost_from_immutable_lock'],lock['payload']['development_research_costs'][model],'original_locked_development_cost')
 # Full per-target producer output is checked independently of summary objects.
 producer_rows={(x['relation_id'],x['configuration_id'],x['repeat'],x['occurrence_id']):x for x in stream(R/f'analysis_{model}/per_target.jsonl')};assert len(producer_rows)==len(per_target)
 for x in per_target:
  y=producer_rows[x['relation_id'],x['configuration_id'],x['repeat'],x['occurrence_id']];compare({k:y[k] for k in ['model','source_id','prediction']},{k:x[k] for k in ['model','source_id','prediction']},'per_target_identity');compare(y['common_S_reference'],x['S_reference'],'per_target_S');compare(y['public_gold'],x['gold'],'per_target_gold')
 write(out/'SOURCE_SHA256_BEFORE.json',sources)
 write(out/'INDEPENDENT_RECOMPUTED_CONFIGURATIONS.json',evaluated);write(out/'INDEPENDENT_POLICY_AND_PREFIX_VIEWS.json',policy_views);write(out/'INDEPENDENT_FIXED_REFERENCE_VIEWS.json',fixed_views)
 with (out/'INDEPENDENT_PER_TARGET.jsonl').open('x') as f:
  for x in per_target:f.write(compact(x)+'\n')
 # Useful, explicitly descriptive same-relation whole-plan evidence for the paper.
 examples=[]
 for task in ('dblp_acm','boolq'):
  for rel in sorted(r for r in vectors if r.startswith(task+'/')):
   for cfg in ('k8/random/C','k8/diversity/C'):
    for repeat in (1,2):
     pp=vectors[rel]['singleton_execution'][repeat];bb=vectors[rel][cfg][repeat];changes=sum(p!=b for p,b in zip(pp,bb));delta=sum(bb)-sum(pp)
     examples.append({'relation':rel,'configuration':cfg,'repeat':repeat,'target_n':128,'gold_count':sum(gold[rel]),'P_count':sum(pp),'batch_count':sum(bb),'changed_rows':changes,'N_to_Y':sum(p==0 and b==1 for p,b in zip(pp,bb)),'Y_to_N':sum(p==1 and b==0 for p,b in zip(pp,bb)),'count_delta':delta,'P_gold':gold_metric(pp,gold[rel]),'batch_gold':gold_metric(bb,gold[rel]),'P_cost':cost(cost_index[rel,'singleton_execution',repeat]),'batch_cost':cost(cost_index[rel,cfg,repeat])})
 write(out/'ALL_WHOLE_RELATION_COUNT_EXAMPLES.json',examples)
 source_after={n:sha(R/n) for n in paths};assert source_after==sources
 write(out/'SOURCE_SHA256_AFTER.json',source_after);write(out/'MATHEMATICAL_COMPARISON_MISMATCHES.json',mismatches)
 result={'status':'PASS' if not mismatches else 'FAIL','model':model,'freeze_sha256':FINAL,'science_requests':len(jobs),'controls':20,'all_native_engine_ID_prompt_grammar_and_canonical_payloads_independently_bound':True,'native_journal_fsynced_before_binding_by_exact_reviewed_runner_source':True,'all_effective_controls_replayed':True,'all_valid_scientific_outputs':len(jobs)-len(invalid),'fixed_relations':len(vectors),'configuration_replay_target_predictions':len(per_target),'policies_recomputed':18,'all_prefix_policy_and_zero_calibration_reference_costs_recomputed':True,'calibration_cost_source':'Original baseline95 independent DEV source/hash/36policy audit inherited by immutable POLICY_LOCK hash; new costs derive independently from native return token-array lengths.','baseline95_certificate_sha256':sha(B/'BASELINE95_CPU_REVIEW_CERTIFICATE.json'),'pure_independent_helper_sha256':sha(helper),'only_AST_pure_independent_oracles_executed':['own_gold','own_fidelity'],'producer_modules_imported':False,'mathematical_comparison_fields':dict(compared),'comparison_mismatches':len(mismatches),'unknown_token_calls':0,'tokens':dict(journal_tokens),'source_hashes_stable':True,'original_failure_and_destructor_logs_retained_without_altering_outputs':True,'GPU_or_model_execution_by_reviewer':False,'scientific_tree_modified':False,'tokenizer_decoding_reexecuted':False,'interpretation':'Finite relations and execution replays; same physical plan shared by selectors is not independent replication. Raw/count/token completeness is not deployment speed or paper acceptance.','elapsed_CPU_seconds':time.time()-started,'created_unix':time.time()}
 write(out/'RESULTS_CERTIFICATE.json',result);print(json.dumps(result,indent=2))
 if mismatches:raise SystemExit(1)
if __name__=='__main__':main()
