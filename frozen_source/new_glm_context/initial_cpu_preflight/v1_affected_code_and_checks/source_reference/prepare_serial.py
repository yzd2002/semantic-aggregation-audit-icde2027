#!/usr/bin/env python3
"""Append-only serial secondary schedule; existing scientific prompts stay intact."""
import random,time
from common import *
prep=ROOT/'prepared'
main=read_lines(prep/'schedule.jsonl')
subset=[r for r in main if r['selected_ordinal']<32]
scope_order={s:i for i,s in enumerate((*SCOPES,'singleton'))}
subset.sort(key=lambda r:(TASKS.index(r['task']),r['selected_ordinal'],scope_order[r['scope']],('near','random','none').index(r['context']),r['repeat']))
random.Random(2026110205).shuffle(subset)
schedule=[]
for ordinal,row in enumerate(subset):
 schedule.append({**row,'source_main_request_id':row['request_id'],'request_id':'serial/'+row['request_id'],
  'ordinal':ordinal,'block':ordinal,'execution_mode':'serial'})
write_lines(prep/'serial_schedule.jsonl',schedule)
write_json(prep/'serial_block_membership.json',[[r['request_id']] for r in schedule])
receipt={'created_unix':time.time(),'seed':2026110205,'selected_ordinal_rule':'0 <= selected_ordinal < 32 within each task',
 'requests_per_model':len(schedule),'same_loaded_engine_after_mixed':True,'requests_per_generate_call':1,'models':{}}
for model in MODELS:
 originals={r['request_id']:r for r in read_lines(prep/f'{model}_requests.jsonl')}
 rows=[{**originals[j['source_main_request_id']],**j} for j in schedule]
 for row in rows:
  old=originals[row['source_main_request_id']]
  for key in ('system','unwrapped_user','wrapped_prompt','prompt_token_ids','requested_ids','output_ids','grammar','prompt_sha256','grammar_sha256','id_map','id_map_sha256'):
   assert row[key]==old[key]
 write_lines(prep/f'{model}_serial_requests.jsonl',rows)
 receipt['models'][model]={'main_requests_sha256':sha(prep/f'{model}_requests.jsonl'),
  'serial_requests_sha256':sha(prep/f'{model}_serial_requests.jsonl'),'exact_main_prompt_grammar_token_identity':'PASS'}
write_json(prep/'SERIAL_PREFLIGHT.json',receipt)
print(compact(receipt))
