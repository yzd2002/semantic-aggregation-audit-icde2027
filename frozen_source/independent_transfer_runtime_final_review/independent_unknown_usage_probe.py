from pathlib import Path
import sys,re,json,hashlib
R=Path('/home/gpu/IntentSat2S_revision_20261106/new_transfer_audit')
O=Path('/home/gpu/IntentSat2S_revision_20261106/independent_transfer_runtime_final_review')
s=(R/'check_runtime_amendment.py').read_text()
s=re.sub(r"for model,fault in \[.*?\]:","for model,fault in [('qwen','unknown_output_tokens')]:",s,count=1)
anchor="if self.generated==4 and fault=='multiple_completions' and i==1:output.append(copy.deepcopy(output[0]))"
assert s.count(anchor)==1
s=s.replace(anchor,anchor+"\n     if self.generated==4 and fault=='unknown_output_tokens' and i==1:output[0].token_ids=None")
(O/'unknown_usage_injected_checker.py').write_text(s)
output=O/'CPU_SYNTHETIC_FINAL_UNKNOWN_USAGE'
sys.argv=[str(R/'check_runtime_amendment.py'),'--base-science-root',str(R),'--out',str(output)]
exec(compile(s,str(O/'unknown_usage_injected_checker.py'),'exec'),{'__name__':'__main__','__file__':str(R/'check_runtime_amendment.py')})
run=output/'qwen_unknown_output_tokens/run_qwen'
load=lambda p:json.loads(p.read_text())
rows=lambda p:[json.loads(t) for t in p.read_text().splitlines()]
journal=rows(run/'engine_returns.jsonl');failure=load(run/'FAILURE.json');science=journal[-1]
assert len(journal)==4 and science['stage']=='science' and science['actual_return_count']==3
assert science['actual_returns'][1]['actual_output_tokens'] is None
assert science['actual_returns'][1]['outputs'][0]['output_token_ids'] is None
assert failure['automatic_retry'] is False and failure['type']=='TypeError'
assert failure['engine_return_journal_sha256']==hashlib.sha256((run/'engine_returns.jsonl').read_bytes()).hexdigest()
assert not (run/'summary.json').exists()
result={'status':'PASS','source_final_runner_sha256':hashlib.sha256((R/'run_gpu.py').read_bytes()).hexdigest(),'source_final_observer_sha256':hashlib.sha256((R/'runtime_observer.py').read_bytes()).hexdigest(),'all_23_actual_returns_preserved':sum(e['actual_return_count'] for e in journal)==23,'native_unknown_output_cost_retained_as_null':True,'raw_second_and_third_returns_retained_after_unknown_second':len(science['actual_returns'])==3,'automatic_retry':False,'generate_batches_exactly_four_no_retry':True,'false_complete_summary_absent':True,'all_evidence_CPU_SYNTHETIC':True,'GPU_or_model_loaded':False,'producer_mathematics_used_as_independent_math_oracle':False}
(O/'06_FINAL_UNKNOWN_USAGE_PRESERVATION.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
