#!/usr/bin/env python3
"""Mechanical CPU handoff checks; this is not the later independent review."""
from common import *
cpu_environment()
import collections,time
import pyarrow.parquet as pq

def main():
 start=time.time()
 prep=ROOT/'prepared';selection=ROOT/'selection'
 targets=read_lines(selection/'targets.jsonl');reserves=read_lines(selection/'reserves.jsonl');bank=read_lines(selection/'background_bank.jsonl')
 assert len(targets)==128 and len(reserves)==256 and len(bank)==10000
 assert [r['selected_ordinal'] for r in targets]==list(range(128))
 assert all(r['task']=='snli' and r['target']['source_split']=='test' for r in targets)
 source_rows={s:pq.read_table(ROOT/'sources'/f'{s}.parquet').to_pylist() for s in ('test','train')}
 def check_record(r):
  source=source_rows[r['source_split']][r['source_index']]
  assert r['source_id']==f'{r["source_split"]}:{r["source_index"]:06d}'
  assert r['features']=={k:source[k] for k in SOURCE_FIELDS}
  assert source['label'] in (0,1,2)
  assert digest(compact(r['features']))==r['feature_sha256']
  assert all(0<n<=512 for n in r['token_lengths'].values())
 for r in bank:check_record(r);assert r['source_split']=='train'
 for entry in reserves:check_record(entry['record'])
 reserved=set()
 for entry in reserves:
  ids=identities(entry['record']);assert not ids&reserved;reserved|=ids
 for r in bank:assert not identities(r)&reserved
 assert len({normalized_feature_fingerprint(r) for r in bank})==10000
 assert [r['source_id'] for r in bank]==[r['source_id'] for r in sorted(bank,key=lambda r:(hash_order('bank',r['source_id']),r['source_id']))]
 bank_by_id={r['source_id']:r for r in bank}
 for r in targets:
  assert r['target']==reserves[r['reserve_ordinal']]['record']
  source=source_rows['test'][r['target']['source_index']]
  assert r['original_gold_class']==source['label'] and r['gold']==int(source['label']==0)
  all_near=set()
  for c in CONTEXTS:
   seen=set()
   for item in r['contexts'][c]:
    assert item==bank_by_id[item['source_id']]
    ids=identities(item);assert not seen&ids;seen|=ids
   if c=='near':all_near=seen
   else:assert not all_near&seen
  assert all(abs(n)<=24 for per_model in r['length_audit']['near_minus_random'].values() for n in per_model.values())
 tokenizers=load_tokenizers();ledger=read_lines(prep/'planned_call_ledger.jsonl');ledger_lookup={x['call_id']:x for x in ledger}
 assert len(ledger)==len(ledger_lookup)==6780
 target_lookup={r['target_id']:r for r in targets};costs=json.loads((prep/'PLANNED_COSTS.json').read_text())
 summary={}
 for model,tok in tokenizers.items():
  # Fresh native tokenization of every retained bank/reserve feature, with no truncation.
  records=bank+[x['record'] for x in reserves]
  for start_index in range(0,len(records),512):
   batch=records[start_index:start_index+512]
   fresh=tok([feature_string(r) for r in batch],add_special_tokens=False,truncation=False,padding=False)['input_ids']
   assert [len(x) for x in fresh]==[r['token_lengths'][model] for r in batch]
  jobs=read_lines(prep/f'{model}_main_requests.jsonl');serial=read_lines(prep/f'{model}_serial_requests.jsonl');controls=read_lines(prep/f'{model}_diagnostic_requests.jsonl')
  assert len(jobs)==1792 and len(serial)==448 and len(controls)==20
  main_lookup={j['request_id']:j for j in jobs}
  look={(j['target_id'],j['scope'],j['context'],j['repeat']):j for j in jobs}
  for phase,rows in [('main',jobs),('serial',serial),('diagnostic',controls)]:
   for j in rows:
    assert tok.encode(j['wrapped_prompt'],add_special_tokens=False,truncation=False)==j['prompt_token_ids']
    assert len(j['prompt_token_ids'])==j['prompt_tokens_preflight']
    assert len(j['prompt_token_ids'])+128<=4096
    assert digest(j['wrapped_prompt'])==j['prompt_sha256']
    if j['grammar'] is not None:assert j['grammar']==grammar(j['output_ids']) and digest(j['grammar'])==j['grammar_sha256']
    plan=ledger_lookup[model+'/'+j['request_id']]
    assert plan['phase']==phase and plan['planned_input_tokens']==len(j['prompt_token_ids']) and plan['requested_YN_label_slots']==len(j['output_ids'])
    assert plan['state']=='not_submitted' and plan['raw_output'] is None
    if phase=='diagnostic':continue
    r=target_lookup[j['target_id']]
    expected=[r['target']]+(r['contexts'][j['context']] if j['scope']!='singleton' else [])
    lines=[compact({'id':str(i+1),'features':rec['features']}) for i,rec in enumerate(expected)]
    user='Records:\n'+'\n'.join(lines)+'\nRequested IDs: '+compact(requested_ids(j['scope']))+'\nReturn one Y or N value for each requested ID.'
    assert j['unwrapped_user']==user
    wrapped=tok.apply_chat_template([{'role':'system','content':SYSTEMS['snli']},{'role':'user','content':user}],tokenize=False,add_generation_prompt=True)
    assert wrapped==j['wrapped_prompt']
    a,b=j['target_input_span_chars'];assert wrapped[a:b]==lines[0]
    ab,bb=j['target_input_span_bytes'];assert wrapped.encode()[ab:bb]==lines[0].encode()
    assert j['id_map']['1']['source_id']==r['target_id']
    if phase=='serial':
     original=main_lookup[j['source_main_request_id']]
     for key in ['wrapped_prompt','prompt_token_ids','grammar','output_ids','requested_ids','id_map','prompt_sha256','grammar_sha256']:
      assert j[key]==original[key]
     assert j['selected_ordinal']<32 and j['block']==j['ordinal']
   contract=costs['models'][model][phase]
   assert contract['calls']==len(rows)
   assert contract['planned_input_tokens']==sum(len(j['prompt_token_ids']) for j in rows)
   assert contract['requested_YN_label_slots']==sum(len(j['output_ids']) for j in rows)
  for r in targets:
   prefixes=[];token_prefixes=[]
   for c in CONTEXTS:
    ff=look[r['target_id'],'full-first',c,1];fl=look[r['target_id'],'full-last',c,1]
    assert ff['wrapped_prompt']==fl['wrapped_prompt'] and ff['prompt_token_ids']==fl['prompt_token_ids']
    assert ff['output_ids']==[1,2,3,4] and fl['output_ids']==[2,3,4,1]
    for s in SCOPES:
     p=look[r['target_id'],s,c,1];prefixes.append(p['target_input_prefix_sha256']);token_prefixes.append(p['target_prefix_token_ids_sha256'])
     assert p['prompt_token_ids']==look[r['target_id'],s,c,2]['prompt_token_ids']
   assert len(set(prefixes))==len(set(token_prefixes))==1
  summary[model]={'main':1792,'serial':448,'diagnostic':20,'native_tokenization':'PASS',
                  'untruncated_source_fields':'PASS','prefix_and_scope_and_serial_checks':'PASS'}
 alpha=json.loads((prep/'ALPHA_PLAN.json').read_text())
 assert alpha['confidence']==0.9875 and alpha['bonferroni_sum_bound']==0.0375 and alpha['old_and_new_tasks_unified_FWER_claim'] is False
 assert json.loads((selection/'GOLD_PERTURBATION_CHECK.json').read_text())['status']=='PASS'
 status=json.loads((prep/'PREINFERENCE_STATUS.json').read_text())
 assert status['status']=='NO_GPU_GO' and status['scientific_calls_attempted']==status['diagnostic_calls_attempted']==status['models_loaded']==0
 assert not any(ROOT.glob('run_*'))
 result={'status':'PASS','review_type':'author mechanical CPU checks; independent review pending','models':summary,
  'reserve':256,'targets':128,'background_bank':10000,'planned_calls':6780,'new_model_judgments':0,
  'seconds':time.time()-start,'cuda_visible_devices':os.environ['CUDA_VISIBLE_DEVICES']}
 write_json(prep/'AUTHOR_CPU_VERIFICATION.json',result)
 print(compact(result),flush=True)

if __name__=='__main__':main()
