#!/usr/bin/env python3
"""Offline integrity/controls replay for actual transfer runtime receipts."""
import argparse,hashlib,json
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
def verify(root,model):
 root=Path(root);out=root/f'run_{model}';contract=load(root/'prepared/runtime_observation_contract.json');expected=contract['models'][model]
 start=load(out/'START.json');summary=load(out/'summary.json');actual=load(out/'EFFECTIVE_NATIVE_RUNTIME.json')
 assert start['runtime_observation_contract_sha256']==sha(root/'prepared/runtime_observation_contract.json')
 assert contract['runtime_observer_sha256']==sha(root/'runtime_observer.py')
 receipt=summary['effective_runtime_observation_receipts']
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
 planned={};science_count=0
 for label in ('controls','requests'):
  with (root/'prepared'/f'{model}_{label}.jsonl').open() as stream:
   for line in stream:
    j=json.loads(line);rid=j['request_id'];assert rid not in planned
    planned[rid]={'prompt_sha256':digest(compact(j['prompt_token_ids'])),'grammar_sha256':digest(j['grammar']) if j.get('guided') is not False else None,
     'stage':'science' if label=='requests' else 'diagnostic/'+j['name'],'block':j['block'] if label=='requests' else None}
    science_count+=label=='requests'
 effective=rows(out/'effective_requests.jsonl');byengine={};bylogical={}
 for r in effective:
  eid=r['engine_request_id'];rid=r['candidate_logical_request_id'];assert eid not in byengine and rid not in bylogical
  p=planned[rid];c=r['effective_controls'];assert r['prompt_token_ids_sha256']==p['prompt_sha256'] and r['stage']==p['stage'] and r['block']==p['block']
  assert (c['n'],c['temperature'],c['top_p'],c['top_k'],c['min_p'],c['max_tokens'])==(1,0,1,0,0,128)
  assert c['sampling_type']=='GREEDY' and c['ignore_eos'] is False and (c['repetition_penalty'],c['presence_penalty'],c['frequency_penalty'],c['seed'])==(1,0,0,None)
  assert c['all_stop_token_ids']==expected['all_stop_token_ids'] and c['primary_eos_token_id']==expected['primary_eos_token_id']
  assert c['grammar_sha256']==p['grammar_sha256']
  if p['grammar_sha256'] is not None:assert c['guided_backend']=='xgrammar' and c['guided_disable_fallback'] is True
  else:assert c['guided_backend'] is None and c['guided_disable_fallback'] is None
  byengine[eid]=r;bylogical[rid]=r
 assert set(bylogical)==set(planned) and receipt['bound_science_and_control_requests']==len(planned)
 batches=rows(out/'effective_return_bindings.jsonl');validated=rows(out/'effective_binding_validation.jsonl');submitted=rows(out/'effective_batch_submissions.jsonl');seen=set();offset=0
 assert len(batches)==len(validated)==len(submitted)
 with (out/'engine_returns.jsonl').open('rb') as f:
  for b,v,s in zip(batches,validated,submitted):
   assert b['stage']==v['stage']==s['stage'] and b['block']==v['block']==s['block'] and v['validated'] is True
   assert b['journal_byte_offset']==offset;segment=f.read(b['journal_byte_length']);assert len(segment)==b['journal_byte_length'] and hashlib.sha256(segment).hexdigest()==b['journal_segment_sha256'];offset+=len(segment)
   journal=json.loads(segment);assert journal['journal_written_before_count_identity_or_parser_validation'] is True
   assert journal['stage']==b['stage'] and journal['block']==b['block']
   ids=[r['logical_request_id'] for r in b['bindings']];assert ids==journal['expected_batch_request_ids']==v['logical_request_ids']==s['logical_request_ids']
   assert journal['actual_return_count']==len(journal['actual_returns'])==len(ids)
   for rid,binding,native_return in zip(ids,b['bindings'],journal['actual_returns']):
    assert rid not in seen;seen.add(rid);er=bylogical[rid]
    assert binding['engine_request_id']==binding['processed_engine_request_id']==er['engine_request_id']==native_return['engine_request_id']
    assert binding['returned_prompt_token_ids_sha256']==binding['processed_prompt_token_ids_sha256']==er['prompt_token_ids_sha256']==digest(compact(native_return['actual_prompt_token_ids']))
    assert binding['grammar_sha256']==er['effective_controls']['grammar_sha256']
  assert f.read()==b''
 assert seen==set(planned)
 canonical=rows(out/'responses.jsonl')+rows(out/'diagnostics.jsonl');seen=set()
 for r in canonical:
  rid=r['request_id'];assert rid not in seen;seen.add(rid);er=bylogical[rid]
  assert r['engine_request_id']==er['engine_request_id'] and r['input_token_ids_sha256']==er['prompt_token_ids_sha256']
 assert seen==set(planned) and summary['scientific_calls']==science_count and summary['diagnostic_calls']==20
 return {'status':'PASS','model':model,'scientific_requests':science_count,'control_requests':20,'all_actual_processed_requests':len(effective),
  'actual_native_worker_and_BF16_verified':True,'effective_greedy_EOS_maxcap_and_guidance_verified':True,
  'all_engineIDs_prompt_hashes_and_grammars_bound_after_durable_returns':True,
  'runtime_contract_sha256':sha(root/'prepared/runtime_observation_contract.json'),'effective_receipt_files':receipt['files']}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parent);p.add_argument('--model',choices=('qwen','smol'),required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();r=verify(a.root,a.model)
 with a.out.open('x') as f:json.dump(r,f,indent=2);f.write('\n')
 print(compact({'status':r['status'],'science_requests':r['scientific_requests'],'control_requests':r['control_requests']}))
