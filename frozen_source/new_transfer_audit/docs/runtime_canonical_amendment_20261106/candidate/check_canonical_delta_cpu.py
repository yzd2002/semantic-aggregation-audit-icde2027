#!/usr/bin/env python3
"""Author CPU self-checks of this repair using freshly observed synthetic runs.

Mutations are isolated copies, never edits of producer or independent evidence.
The old verifier is run only to reproduce the five reported false accepts.
"""
from pathlib import Path
import os,sys,json,copy,shutil,importlib.util,hashlib
os.environ['CUDA_VISIBLE_DEVICES']='';sys.dont_write_bytecode=True
C=Path(__file__).resolve().parent
def load(p):return json.loads(p.read_text())
def lines(p):return [json.loads(x) for x in p.read_text().splitlines() if x]
def compact(v):return json.dumps(v,ensure_ascii=False,separators=(',',':'))
def write(p,v):p.write_text(json.dumps(v,indent=2)+'\n')
def rows(p,v):p.write_text(''.join(compact(x)+'\n' for x in v))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def module(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
NEW=module('candidate_verifier',C/'verify_effective_runtime.py');OLD=module('baseline_verifier',C/'verify_effective_runtime.baseline.py')
def refresh(root):
 out=root/'run_qwen';s=load(out/'summary.json')
 s['response_sha256']=sha(out/'responses.jsonl');s['engine_return_journal_sha256']=sha(out/'engine_returns.jsonl');s['blocks_sha256']=sha(out/'blocks.jsonl')
 for name in s['effective_runtime_observation_receipts']['files']:s['effective_runtime_observation_receipts']['files'][name]=sha(out/name)
 write(out/'summary.json',s)
 c=load(out/'CONTROL_RESULT.json');c['diagnostic_sha256']=sha(out/'diagnostics.jsonl');write(out/'CONTROL_RESULT.json',c)
 d=load(out/'DATA_START.json');d['control_sha256']=sha(out/'CONTROL_RESULT.json');write(out/'DATA_START.json',d)
def refresh_journal_bindings(root):
 out=root/'run_qwen';events=lines(out/'engine_returns.jsonl');bindings=lines(out/'effective_return_bindings.jsonl');offset=0
 for event,b in zip(events,bindings):
  segment=(compact(event)+'\n').encode();b.update(journal_byte_offset=offset,journal_byte_length=len(segment),journal_segment_sha256=hashlib.sha256(segment).hexdigest());offset+=len(segment)
 rows(out/'effective_return_bindings.jsonl',bindings)
def accepted(m,root):
 try:return True,m.verify(root,'qwen')
 except Exception as e:return False,type(e).__name__+': '+str(e)
def main():
 out=C/'CANONICAL_DELTA_SELFTEST';out.mkdir(exist_ok=False);base=C/'CPU_ACTUAL_RUNNER_FIXTURES/qwen_none';results=[]
 for name,model in [('qwen_none','qwen'),('smol_none','smol'),('qwen_invalid_raw','qwen')]:
  root=C/'CPU_ACTUAL_RUNNER_FIXTURES'/name;result=NEW.verify(root,model)
  results.append({'case':'fresh_'+name,'expected_accepted':True,'accepted':True,'result':result})
 faults=['canonical_raw','canonical_labels','canonical_output_tokens','canonical_finish_reason','canonical_cost','canonical_strict','canonical_stop_reason','canonical_raw_hash','canonical_diagnostic_raw','canonical_diagnostic_labels','canonical_diagnostic_tokens','canonical_diagnostic_finish','canonical_diagnostic_cost',
  'summary_response_hash','summary_journal_hash','summary_blocks_hash','summary_science_count','summary_valid_count','summary_invalid_count','summary_diagnostic_count','summary_all_count',
  'summary_science_input','summary_science_output','summary_diagnostic_input','summary_diagnostic_output','summary_all_input','summary_all_output','summary_status','missing_receipt_hash',
  'unknown_journal_input_zero_canonical','unknown_journal_output_zero_canonical','native_multiple_completions','native_return_count','native_planned_ID_order','canonical_order','block_count','block_submission_ID',
  'effective_stop_string','effective_stop_tokens','effective_skip_special','effective_spaces']
 for fault in faults:
  root=out/fault;shutil.copytree(base,root,symlinks=True);run=root/'run_qwen';p=run/'responses.jsonl';rr=lines(p);diag=lines(run/'diagnostics.jsonl')
  if fault=='canonical_raw':rr[0]['raw']=rr[0]['raw'].replace('"Y"','"N"');rr[0]['labels']={k:0 for k in rr[0]['labels']}
  elif fault=='canonical_labels':rr[0]['labels']={k:0 for k in rr[0]['labels']}
  elif fault=='canonical_output_tokens':rr[0]['output_token_ids']=[999,998,997];rr[0]['actual_output_tokens']=3
  elif fault=='canonical_finish_reason':rr[0]['finish_reason']='length'
  elif fault=='canonical_cost':rr[0]['actual_input_tokens']=0;rr[0]['actual_output_tokens']=0
  elif fault=='canonical_strict':rr[0]['strict_valid']=False;rr[0]['labels']=None
  elif fault=='canonical_stop_reason':rr[0]['stop_reason']='CPU_UNPLANNED_STOP'
  elif fault=='canonical_raw_hash':rr[0]['raw_sha256']='0'*64
  elif fault=='canonical_diagnostic_raw':diag[1]['raw']=diag[1]['raw'].replace('"Y"','"N"')
  elif fault=='canonical_diagnostic_labels':diag[1]['labels']={k:0 for k in diag[1]['labels']}
  elif fault=='canonical_diagnostic_tokens':diag[1]['output_token_ids']=[999,998]
  elif fault=='canonical_diagnostic_finish':diag[1]['finish_reason']='length'
  elif fault=='canonical_diagnostic_cost':diag[1]['actual_output_tokens']=0
  elif fault=='canonical_order':rr.reverse()
  elif fault.startswith('unknown_journal_') or fault.startswith('native_'):
   jp=run/'engine_returns.jsonl';jj=lines(jp);last=jj[-1];native=last['actual_returns'][0]
   if fault=='unknown_journal_input_zero_canonical':native['actual_prompt_token_ids']=None;native['actual_input_tokens']=None;rr[0]['actual_input_tokens']=0
   elif fault=='unknown_journal_output_zero_canonical':native['outputs'][0]['output_token_ids']=None;native['actual_output_tokens']=None;rr[0]['output_token_ids']=[];rr[0]['actual_output_tokens']=0
   elif fault=='native_multiple_completions':native['outputs'].append(copy.deepcopy(native['outputs'][0]));native['outputs'][1]['output_position']=1;native['actual_output_tokens']*=2
   elif fault=='native_return_count':last['actual_return_count']+=1
   elif fault=='native_planned_ID_order':last['expected_batch_request_ids'].reverse()
   rows(jp,jj);refresh_journal_bindings(root)
  elif fault=='block_count':bp=run/'blocks.jsonl';bb=lines(bp);bb[0]['requests']+=1;rows(bp,bb)
  elif fault=='block_submission_ID':bp=run/'block_submissions.jsonl';bb=lines(bp);bb[0]['request_ids'].reverse();rows(bp,bb)
  elif fault.startswith('effective_'):
   ep=run/'effective_requests.jsonl';ee=lines(ep);controls=ee[-1]['effective_controls']
   if fault=='effective_stop_string':controls['stop']=['CPU_UNPLANNED_STOP']
   if fault=='effective_stop_tokens':controls['stop_token_ids']=[999]
   if fault=='effective_skip_special':controls['skip_special_tokens']=False
   if fault=='effective_spaces':controls['spaces_between_special_tokens']=False
   rows(ep,ee)
  rows(p,rr);rows(run/'diagnostics.jsonl',diag)
  # Refresh file hashes so field-level binding, parsing and totals checks must
  # reject tampering even when the corrupted canonical hash was updated.
  refresh(root);s=load(run/'summary.json')
  if fault.startswith('summary_'):
   targets={'summary_response_hash':('response_sha256','0'*64),'summary_journal_hash':('engine_return_journal_sha256','0'*64),'summary_blocks_hash':('blocks_sha256','0'*64),'summary_science_count':('scientific_calls',4),'summary_valid_count':('valid_scientific_calls',2),'summary_invalid_count':('invalid_scientific_calls',1),'summary_diagnostic_count':('diagnostic_calls',19),'summary_all_count':('all_calls',22),'summary_status':('status','failed')}
   if fault in targets:k,v=targets[fault];s[k]=v
   else:
    stage,kind=fault.split('_')[1:];field={'science':'scientific_observed_tokens','diagnostic':'diagnostic_observed_tokens','all':'all_observed_tokens'}[stage];s[field][kind+'_tokens']+=1
  if fault=='missing_receipt_hash':s['effective_runtime_observation_receipts']['files'].pop('effective_requests.jsonl')
  write(run/'summary.json',s)
  okay,error=accepted(NEW,root);assert not okay,(fault,'candidate accepted tamper')
  record={'case':fault,'expected_accepted':False,'accepted':okay,'rejected_by':error,'status':'PASS'}
  if fault in faults[:5]:
   oldokay,oldresult=accepted(OLD,root);assert oldokay,(fault,'baseline defect not reproduced',oldresult);record['baseline_false_accept_reproduced']=True
  results.append(record)
 report={'status':'PASS','author_selfcheck_not_independent_review':True,'candidate_verifier_sha256':sha(C/'verify_effective_runtime.py'),'baseline_verifier_sha256':sha(C/'verify_effective_runtime.baseline.py'),'clean_cases':3,'tamper_rejections':len(faults),'all_mutated_hashes_refreshed_before_field_checks_except_explicit_hash_faults':True,'original_five_false_accepts_reproduced_and_fixed':True,'old_controls_not_fabricated_or_backfilled':True,'GPU_execution':False,'real_model_judgments':0,'cases':results}
 write(C/'CANONICAL_DELTA_SELFTEST_RESULTS.json',report);print(compact({'status':'PASS','clean_cases':3,'tamper_rejections':len(faults)}))
if __name__=='__main__':main()
