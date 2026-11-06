#!/usr/bin/env python3
"""Offline integrity/controls replay for actual transfer runtime receipts."""
import argparse,hashlib,json,re
from pathlib import Path
def compact(x):return json.dumps(x,ensure_ascii=False,separators=(',',':'))
def digest(x):return hashlib.sha256(x.encode()).hexdigest()
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def load(p):return json.loads(p.read_text())
def rows(p):
 with p.open() as f:return [json.loads(x) for x in f if x.strip()]
def require(condition,message):
 if not condition:raise ValueError(message)
def integer(value,want,message):
 require(type(value) is int and value==want,message)
def token_list(value,message):
 require(isinstance(value,list) and all(type(x) is int and x>=0 for x in value),message)
 return value
def strict_labels(raw,ids):
 if not ids:return None
 require(len(ids)==len(set(ids)) and all(type(x) is int and x>0 for x in ids),'Invalid frozen output IDs')
 rule=r'\{'+','.join(re.escape('"'+str(i)+'":')+'"[YN]"' for i in ids)+r'\}'
 if not isinstance(raw,str) or re.fullmatch(rule,raw) is None:return None
 pairs=json.loads(raw,object_pairs_hook=list)
 if [int(k) for k,v in pairs]!=ids or len({k for k,v in pairs})!=len(ids):return None
 return {k:int(v=='Y') for k,v in pairs}
def verify(root,model):
 root=Path(root);out=root/f'run_{model}';contract=load(root/'prepared/runtime_observation_contract.json');expected=contract['models'][model]
 start=load(out/'START.json');summary=load(out/'summary.json');actual=load(out/'EFFECTIVE_NATIVE_RUNTIME.json')
 assert start['runtime_observation_contract_sha256']==sha(root/'prepared/runtime_observation_contract.json')
 assert contract['runtime_observer_sha256']==sha(root/'runtime_observer.py')
 receipt=summary['effective_runtime_observation_receipts']
 require(summary['status']=='complete' and summary['model']==start['model']==model,'Run is not a complete matching model')
 for field,name in [('response_sha256','responses.jsonl'),('engine_return_journal_sha256','engine_returns.jsonl'),('blocks_sha256','blocks.jsonl')]:
  require(summary[field]==sha(out/name),'Summary file hash differs: '+field)
 expected_receipt_files={'EFFECTIVE_NATIVE_RUNTIME.json','effective_requests.jsonl','effective_batch_submissions.jsonl','effective_return_bindings.jsonl','effective_binding_validation.jsonl'}
 require(set(receipt['files'])==expected_receipt_files,'Missing or extra effective receipt hash inventory')
 assert receipt['active_batch_unbound'] is False and receipt['model']==model
 for name,want in receipt['files'].items():assert sha(out/name)==want
 native=actual['worker_identity'];assert len(native)==1
 assert native[0]['model_class']==expected['native_class'] and native[0]['model_module']==expected['native_module']
 assert native[0]['parameter_dtypes']==['torch.bfloat16'] and native[0]['parameters']>0 and not native[0]['remote_model_modules_loaded']
 assert actual['native_hf_config_class']==expected['native_hf_config_class'] and actual['model_impl_requested'] in ('auto','vllm')
 assert actual['dtype']=='torch.bfloat16' and actual['max_model_len']==4096 and actual['max_num_seqs']==16 and actual['tensor_parallel_size']==1
 assert actual['enforce_eager'] is True and actual['prefix_caching'] is False and actual['gpu_memory_utilization']==.70
 assert actual['primary_eos_token_id']==expected['primary_eos_token_id'] and actual['guided_decoding_backend']=='xgrammar' and actual['guided_decoding_disable_fallback'] is True
 assert actual['root_trusted_callable_sha256']==contract['worker_identity_source_sha256']
 planned={};controls=rows(root/'prepared'/f'{model}_controls.jsonl');science=rows(root/'prepared'/f'{model}_requests.jsonl');science_count=len(science)
 require(start['request_sha256']==sha(root/'prepared'/f'{model}_requests.jsonl'),'START/frozen scientific request hash differs')
 integer(start['planned_scientific_calls'],science_count,'START planned science count differs')
 group_names=('ordinary_banana','guided_banana','synthetic_format_smoke')
 require(len(controls)==20 and all(j['name'] in group_names for j in controls),'Frozen diagnostic plan differs')
 groups=[[j for j in controls if j['name']==name] for name in group_names]
 require(list(map(len,groups))==[1,5,14],'Frozen diagnostic group sizes differ')
 expected_batches=[('diagnostic/'+name,None,group) for name,group in zip(group_names,groups)]
 for offset in range(0,science_count,64):
  batch=science[offset:offset+64]
  for index,j in enumerate(batch,offset):
   integer(j['ordinal'],index,'Frozen science ordinal differs');integer(j['block'],offset//64,'Frozen science block differs')
  expected_batches.append(('science',offset//64,batch))
 for stage,block,batch in expected_batches:
  for j in batch:
   rid=j['request_id'];require(rid not in planned,'Repeated frozen request ID')
   tokens=token_list(j['prompt_token_ids'],'Frozen prompt tokens are malformed')
   require(j['prompt_sha256']==digest(j['wrapped_prompt']),'Frozen prompt text/hash differs')
   if j.get('grammar_sha256') is not None:require(j['grammar_sha256']==digest(j['grammar']),'Frozen grammar/hash differs')
   planned[rid]={'job':j,'prompt_sha256':digest(compact(tokens)),'grammar_sha256':digest(j['grammar']) if j.get('guided') is not False else None,'stage':stage,'block':block}
 effective=rows(out/'effective_requests.jsonl');byengine={};bylogical={}
 for r in effective:
  eid=r['engine_request_id'];rid=r['candidate_logical_request_id'];assert eid not in byengine and rid not in bylogical
  p=planned[rid];c=r['effective_controls'];assert r['prompt_token_ids_sha256']==p['prompt_sha256'] and r['stage']==p['stage'] and r['block']==p['block']
  assert (c['n'],c['temperature'],c['top_p'],c['top_k'],c['min_p'],c['max_tokens'])==(1,0,1,0,0,128)
  assert c['sampling_type']=='GREEDY' and c['ignore_eos'] is False and (c['repetition_penalty'],c['presence_penalty'],c['frequency_penalty'],c['seed'])==(1,0,0,None)
  assert c['all_stop_token_ids']==expected['all_stop_token_ids'] and c['primary_eos_token_id']==expected['primary_eos_token_id']
  require(c['stop']==[],'Unplanned custom stop strings')
  require(c['stop_token_ids']==sorted(i for i in expected['all_stop_token_ids'] if i!=expected['primary_eos_token_id']),'Unexpected non-primary EOS stop token IDs')
  require(c['skip_special_tokens'] is True and c['spaces_between_special_tokens'] is True,'Output formatting controls changed')
  assert c['grammar_sha256']==p['grammar_sha256']
  if p['grammar_sha256'] is not None:assert c['guided_backend']=='xgrammar' and c['guided_disable_fallback'] is True
  else:assert c['guided_backend'] is None and c['guided_disable_fallback'] is None
  byengine[eid]=r;bylogical[rid]=r
 assert set(bylogical)==set(planned) and receipt['bound_science_and_control_requests']==len(planned)
 batches=rows(out/'effective_return_bindings.jsonl');validated=rows(out/'effective_binding_validation.jsonl');submitted=rows(out/'effective_batch_submissions.jsonl');seen=set();offset=0;native_bylogical={}
 require(len(batches)==len(validated)==len(submitted)==len(expected_batches),'Submitted/returned batch count differs from complete frozen plan')
 require([r['candidate_logical_request_id'] for r in effective]==[j['request_id'] for stage,block,jobs in expected_batches for j in jobs],'Actual processed order differs from complete frozen plan')
 with (out/'engine_returns.jsonl').open('rb') as f:
  for b,v,s,(stage,block,jobs) in zip(batches,validated,submitted,expected_batches):
   assert b['stage']==v['stage']==s['stage'] and b['block']==v['block']==s['block'] and v['validated'] is True
   require((b['stage'],b['block'])==(stage,block),'Batch stage/order differs from frozen source plan')
   assert b['journal_byte_offset']==offset;segment=f.read(b['journal_byte_length']);assert len(segment)==b['journal_byte_length'] and hashlib.sha256(segment).hexdigest()==b['journal_segment_sha256'];offset+=len(segment)
   journal=json.loads(segment);assert journal['journal_written_before_count_identity_or_parser_validation'] is True
   assert journal['stage']==b['stage'] and journal['block']==b['block']
   ids=[r['logical_request_id'] for r in b['bindings']];assert ids==journal['expected_batch_request_ids']==v['logical_request_ids']==s['logical_request_ids']
   require(ids==[j['request_id'] for j in jobs],'Batch IDs/order differ from frozen source plan')
   assert journal['actual_return_count']==len(journal['actual_returns'])==len(ids)
   for position,(rid,binding,native_return) in enumerate(zip(ids,b['bindings'],journal['actual_returns'])):
    assert rid not in seen;seen.add(rid);er=bylogical[rid]
    integer(native_return['return_position'],position,'Native return position differs')
    integer(er['position'],position,'Processed position differs')
    require(binding['stage']==stage and binding['block']==block,'Individual return binding stage/block differs')
    assert binding['engine_request_id']==binding['processed_engine_request_id']==er['engine_request_id']==native_return['engine_request_id']
    assert binding['returned_prompt_token_ids_sha256']==binding['processed_prompt_token_ids_sha256']==er['prompt_token_ids_sha256']==digest(compact(native_return['actual_prompt_token_ids']))
    assert binding['grammar_sha256']==er['effective_controls']['grammar_sha256']
    prompt=token_list(native_return['actual_prompt_token_ids'],'Unknown native input token cost cannot be zero')
    require(prompt==planned[rid]['job']['prompt_token_ids'],'Native prompt differs from exact frozen token IDs')
    integer(native_return['actual_input_tokens'],len(prompt),'Native input token count differs or is unknown')
    outputs=native_return['outputs'];require(isinstance(outputs,list) and len(outputs)==1,'Complete canonical run requires frozen n=1 native completion count')
    completion=outputs[0];integer(completion['output_position'],0,'Native completion position differs')
    output_tokens=token_list(completion['output_token_ids'],'Unknown native output token cost cannot be zero')
    require(len(output_tokens)<=128,'Native output exceeds fixed cap')
    require(isinstance(completion['raw'],str),'Native raw completion is unknown or not a string')
    integer(native_return['actual_output_tokens'],len(output_tokens),'Native output token count differs or is unknown')
    native_bylogical[rid]=native_return
  assert f.read()==b''
 assert seen==set(planned)
 science_rows=rows(out/'responses.jsonl');diagnostic_rows=rows(out/'diagnostics.jsonl');canonical=science_rows+diagnostic_rows;seen=set()
 require([r['request_id'] for r in science_rows]==[j['request_id'] for j in science],'Canonical science order/count differs from frozen plan')
 require([r['request_id'] for r in diagnostic_rows]==[j['request_id'] for group in groups for j in group],'Canonical diagnostic order/count differs from frozen plan')
 totals={name:{'input_tokens':0,'output_tokens':0} for name in ('science','diagnostic')};valid_science=0
 omitted={'system','unwrapped_user','wrapped_prompt','prompt_token_ids','grammar','input_ids_in_order','id_map','prompt_tokens_preflight','canonical_output_token_lengths_allY_allN'}
 for r in canonical:
  rid=r['request_id'];assert rid not in seen;seen.add(rid);er=bylogical[rid]
  assert r['engine_request_id']==er['engine_request_id'] and r['input_token_ids_sha256']==er['prompt_token_ids_sha256']
  p=planned[rid];j=p['job'];native_return=native_bylogical[rid];completion=native_return['outputs'][0]
  for key,value in j.items():
   if key not in omitted:require(r[key]==value,'Canonical/frozen metadata differs: '+rid+'/'+key)
  diagnostic=p['stage']!='science';require(r['diagnostic'] is diagnostic,'Canonical diagnostic/science status differs')
  if diagnostic:require(r['wrapped_prompt']==j['wrapped_prompt'] and r['grammar']==j.get('grammar'),'Canonical diagnostic input differs')
  for key in ('raw','output_token_ids','finish_reason','stop_reason'):
   require(r[key]==completion[key],'Canonical/native completion differs: '+rid+'/'+key)
  if 'raw_sha256' in r:require(r['raw_sha256']==digest(completion['raw']),'Canonical raw hash differs')
  token_list(r['output_token_ids'],'Malformed canonical output token IDs')
  integer(r['actual_input_tokens'],len(native_return['actual_prompt_token_ids']),'Canonical actual input cost differs')
  integer(r['actual_output_tokens'],len(completion['output_token_ids']),'Canonical actual output cost differs')
  labels=strict_labels(completion['raw'],j['output_ids'])
  require(r['strict_valid'] is (labels is not None),'Canonical strict validity differs from exact raw parse')
  require(r['labels']==labels,'Canonical labels differ from exact raw parse')
  if labels is not None:
   require(list(r['labels'])==list(labels) and all(type(value) is int for value in r['labels'].values()),'Canonical labels/order/types differ')
  if not diagnostic:valid_science+=labels is not None
  group='diagnostic' if diagnostic else 'science'
  for key in ('input_tokens','output_tokens'):totals[group][key]+=r['actual_'+key]
 assert seen==set(planned) and summary['scientific_calls']==science_count and summary['diagnostic_calls']==20
 for key,value in {'scientific_calls':science_count,'valid_scientific_calls':valid_science,'invalid_scientific_calls':science_count-valid_science,'diagnostic_calls':len(controls),'all_calls':len(planned)}.items():
  integer(summary[key],value,'Summary count/status differs: '+key)
 for field,want in [('scientific_observed_tokens',totals['science']),('diagnostic_observed_tokens',totals['diagnostic']),('all_observed_tokens',{k:totals['science'][k]+totals['diagnostic'][k] for k in totals['science']})]:
  require(set(summary[field])==set(want),'Summary token fields differ: '+field)
  for key,value in want.items():integer(summary[field][key],value,'Summary actual token total differs or is unknown: '+field+'/'+key)
 diag_submissions=rows(out/'diagnostic_submissions.jsonl');science_submissions=rows(out/'block_submissions.jsonl');blocks=rows(out/'blocks.jsonl')
 require(len(diag_submissions)==3 and len(science_submissions)==len(blocks)==len(expected_batches)-3,'Canonical submission/block completion count differs')
 for submitted_group,name,group in zip(diag_submissions,group_names,groups):
  require(submitted_group['name']==name and submitted_group['request_ids']==[j['request_id'] for j in group],'Diagnostic submission IDs/order differ')
 for index,(submission,block_record,(stage,block,jobs)) in enumerate(zip(science_submissions,blocks,expected_batches[3:])):
  require(submission['request_ids']==[j['request_id'] for j in jobs],'Science submission IDs/order differ')
  integer(submission['block'],index,'Science submitted block differs');integer(block_record['block'],index,'Completed block differs');integer(block_record['requests'],len(jobs),'Completed block request count differs')
  integer(submission['first_ordinal'],index*64,'First submitted ordinal differs');integer(submission['last_ordinal'],index*64+len(jobs)-1,'Last submitted ordinal differs')
 control=load(out/'CONTROL_RESULT.json')
 ordinary=diagnostic_rows[0];good='BANANA' in ordinary['raw'] and ordinary['strict_valid'] is False and all(r['strict_valid'] for r in diagnostic_rows[1:])
 require(control['diagnostic_sha256']==sha(out/'diagnostics.jsonl'),'Diagnostic result hash differs')
 require(good and control['status']=='PASS' and control['diagnostic_calls']==20 and control['semantic_labels_not_a_gate'] is True,'Diagnostic status/count differs from raw format controls')
 require(load(out/'DATA_START.json')['control_sha256']==sha(out/'CONTROL_RESULT.json'),'Scientific start/control-result hash differs')
 return {'status':'PASS','model':model,'scientific_requests':science_count,'control_requests':20,'all_actual_processed_requests':len(effective),
  'actual_native_worker_and_BF16_verified':True,'effective_greedy_EOS_maxcap_and_guidance_verified':True,
  'all_engineIDs_prompt_hashes_and_grammars_bound_after_durable_returns':True,
  'all_canonical_raw_completions_tokens_finish_stop_labels_and_costs_match_native_journal':True,
  'source_planned_ID_order_and_all_batch_counts_verified':True,'summary_hashes_counts_tokens_and_invalid_status_recomputed':True,
  'unknown_native_usage_cannot_be_certified_as_zero_cost':True,'stop_strings_nonprimary_EOS_and_output_format_controls_verified':True,
  'runtime_contract_sha256':sha(root/'prepared/runtime_observation_contract.json'),'effective_receipt_files':receipt['files']}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parent);p.add_argument('--model',choices=('qwen','smol'),required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();r=verify(a.root,a.model)
 with a.out.open('x') as f:json.dump(r,f,indent=2);f.write('\n')
 print(compact({'status':r['status'],'science_requests':r['scientific_requests'],'control_requests':r['control_requests']}))
