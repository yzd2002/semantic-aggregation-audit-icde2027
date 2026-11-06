#!/usr/bin/env python3
"""CPU fake-engine tests execute the actual new runner inside isolated temp fixtures."""
import contextlib,io,os,re,sys,tempfile,types
from unittest.mock import patch
from common import *
import run_gpu as runner
import analyze_transfer as analyzer

def run():
    fullcontrols=read_lines(ROOT/'prepared/qwen_controls.jsonl')
    # These are input fixtures only. All fake outputs are deleted with the isolated fixture.
    science=read_lines(ROOT/'prepared/qwen_requests.jsonl')[:3]
    schedule=read_lines(ROOT/'prepared/qwen_schedule.jsonl')[:3]
    results=[]
    for stage in ('science','diagnostic'):
        for fault in ('response_count_mismatch','prompt_token_mismatch','invalid_raw'):
            with tempfile.TemporaryDirectory(prefix='CPU_SYNTHETIC_RUNNER_ONLY_',dir=ROOT/'tmp') as tmp:
                fixture=Path(tmp);(fixture/'prepared').mkdir();(fixture/'runtime_override').mkdir()
                write_json(fixture/'model_locations.json',{'qwen':str(fixture/'synthetic_model_path')})
                write_json(fixture/'prepared/MODEL_FILE_SHA256.json',{'qwen':{'files':{}}})
                write_json(fixture/'prepared/runtime_configuration.json',{'packages':{},'python':__import__('platform').python_version(),
                    'python_executable':sys.executable,'FLASHINFER_WORKSPACE_BASE':str(fixture/'synthetic_cache')})
                write_lines(fixture/'prepared/qwen_requests.jsonl',science);write_lines(fixture/'prepared/qwen_controls.jsonl',fullcontrols)
                write_lines(fixture/'prepared/qwen_schedule.jsonl',schedule)
                class Sampling:
                    def __init__(self,**kwargs):self.__dict__.update(kwargs)
                class Guided:
                    def __init__(self,**kwargs):self.__dict__.update(kwargs)
                instances=[]
                class Engine:
                    def __init__(self,**kwargs):self.calls=0;self.returned=[];instances.append(self)
                    def generate(self,prompts,parameters,use_tqdm=False):
                        self.calls+=1;this=[];target=(stage=='science' and self.calls==4) or (stage=='diagnostic' and self.calls==3)
                        for ix,(p,param) in enumerate(zip(prompts,parameters)):
                            guide=param.guided_decoding
                            if guide is None:raw='BANANA'
                            else:
                                ids=re.findall(r'"(\d+)":',guide.regex);raw=compact({i:'Y' for i in ids})
                            if target and fault=='invalid_raw' and ix==1:raw='CPU_SYNTHETIC_INVALID_RAW_WITH_TRAILING_NEWLINE\n'
                            prompt=list(p['prompt_token_ids'])
                            if target and fault=='prompt_token_mismatch' and ix==1:prompt=prompt+[987654321]
                            answer=types.SimpleNamespace(text=raw,token_ids=[1001,1002],finish_reason='stop',stop_reason=None)
                            this.append(types.SimpleNamespace(prompt_token_ids=prompt,outputs=[answer],request_id=f'CPU_fixture_{self.calls}_{ix}'))
                        if target and fault=='response_count_mismatch':this=this[:-1]
                        self.returned.append(this);return this
                fakevllm=types.ModuleType('vllm');fakevllm.LLM=Engine;fakevllm.SamplingParams=Sampling
                fakesampling=types.ModuleType('vllm.sampling_params');fakesampling.GuidedDecodingParams=Guided
                fakecompat=types.ModuleType('single_rank_nccl_compat');fakecompat.ACTIVE=True;fakecompat.__file__=str(fixture/'runtime_override/single_rank_nccl_compat.py')
                realsha=runner.sha
                def fixture_sha(path):
                    return 'CPU_SYNTHETIC_FIXTURE_NO_FREEZE_FILE_CREATED' if Path(path).name=='PREINFERENCE_FROZEN_SHA256.json' else realsha(path)
                failed=None
                with patch.object(runner,'ROOT',fixture),patch.object(runner,'verify',lambda *args:{'CPU_synthetic_fixture':True}),\
                    patch.object(runner,'processes',lambda *args:[]),patch.object(runner,'owned',lambda *args:[]),patch.object(runner,'sha',fixture_sha),\
                    patch.object(analyzer,'analyze',lambda *args:None),patch.dict(sys.modules,{'vllm':fakevllm,'vllm.sampling_params':fakesampling,'single_rank_nccl_compat':fakecompat}),\
                    patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':'CPU_fixture','FLASHINFER_WORKSPACE_BASE':str(fixture/'synthetic_cache')}),contextlib.redirect_stdout(io.StringIO()):
                    try:runner.main('qwen','CPU_fixture')
                    except BaseException as exc:failed=type(exc).__name__+': '+str(exc)
                out=fixture/'run_qwen';journals=read_lines(out/'engine_returns.jsonl');target_index=3 if stage=='science' else 2
                journal=journals[target_index];actual=instances[0].returned[target_index]
                assert journal['actual_return_count']==len(actual)
                assert len(journal['actual_returns'])==len(actual)
                for saved,response in zip(journal['actual_returns'],actual):
                    assert saved['engine_request_id']==response.request_id
                    assert saved['actual_prompt_token_ids']==response.prompt_token_ids
                    assert saved['outputs'][0]['raw']==response.outputs[0].text
                    assert saved['outputs'][0]['output_token_ids']==response.outputs[0].token_ids
                should_fail=not(stage=='science' and fault=='invalid_raw')
                assert (failed is not None)==should_fail and (out/'FAILURE.json').exists()==should_fail
                raw=read_lines(out/'responses.jsonl') if (out/'responses.jsonl').exists() else []
                if stage=='diagnostic':assert not raw
                if stage=='science' and fault=='invalid_raw':assert len(raw)==3 and sum(not r['strict_valid'] for r in raw)==1
                with patch.object(analyzer,'ROOT',fixture),patch.object(analyzer,'sha',fixture_sha):
                    _,_,_,_,ledger=analyzer.load_trace('qwen')
                native=ledger['all_native_returns_science_journal' if stage=='science' else 'all_native_returns_control_journal']
                assert native['observed_input_token_sum_from_actual_returns']>0 and native['observed_output_token_sum_from_actual_returns']>0
                if fault in ('response_count_mismatch','prompt_token_mismatch'):
                    role=ledger['science_union' if stage=='science' else 'controls']
                    assert role['actual_total_tokens'] is None and role['unknown_input_token_call_count']>0
                results.append({'stage':stage,'fault':fault,'status':'PASS','exception':failed,
                    'expected_batch_request_ids':journal['expected_batch_request_ids'],
                    'actual_returns':len(actual),'journal_preserved_returns':len(journal['actual_returns']),
                    'all_returned_raw_prompt_and_output_tokens_engine_ids_preserved_before_validation':True,
                    'saved_validated_scientific_rows':len(raw),'failure_receipt_exists':(out/'FAILURE.json').exists(),
                    'analysis_native_return_token_observations':native,
                    'automatic_retry':False,'GPU_execution':False,'model_loaded':False,'fixture_deleted_after_check':True})
    return {'status':'PASS','test_count':len(results),'used_actual_runner_source_sha256':sha(ROOT/'run_gpu.py'),
        'used_actual_return_journal_source_sha256':sha(ROOT/'return_journal.py'),
        'used_actual_analyzer_source_sha256':sha(ROOT/'analyze_transfer.py'),'CPU_fake_engine_only':True,'cases':results}

if __name__=='__main__':
    result=run();write_json(ROOT/'verification/RUNNER_RETURN_PRESERVATION.json',result)
    print(compact({'status':result['status'],'test_count':result['test_count']}))
