#!/usr/bin/env python3
"""CPU fake-engine execution of the actual v3 runner under failure cases.

This creates explicitly synthetic attempts in a caller-specified fresh output
directory. It does not import a real vLLM/torch runtime or access any GPU.
"""
import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
import types

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
sys.dont_write_bytecode=True
from common import compact,digest,grammar,parse as strict_parse,read_lines,sha,write_json,write_lines
from generation_receipts import return_accounting
from analyze_fresh_replication import load_unique

S=types.SimpleNamespace


class Guided:
    def __init__(self,**values):self.__dict__.update(values)


class Sample:
    def __init__(self,**values):
        self.__dict__.update(n=1,temperature=0,top_k=0,top_p=1,min_p=0,max_tokens=128,ignore_eos=False,
            repetition_penalty=1,presence_penalty=0,frequency_penalty=0,seed=None,skip_special_tokens=True,
            spaces_between_special_tokens=True)
        self.__dict__.update(values)
        self.all_stop_token_ids=set(self.stop_token_ids);self.sampling_type=S(name='GREEDY')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    output=args.out.resolve();output.mkdir(parents=True,exist_ok=False)
    scenarios=('success','science_short','science_extra','science_middle_prompt','science_reordered',
        'science_unknown_id','science_duplicate_id','science_multiple_completions','science_invalid_raw',
        'science_parse_exception','science_generate_exception','science_empty','science_none',
        'diagnostic_short','diagnostic_middle_prompt','diagnostic_reordered','diagnostic_extra')
    results=[]
    for fault in scenarios:
        root=output/fault;root.mkdir()
        for name in ('prepared','runtime_override','model'):(root/name).mkdir()
        write_json(root/'model/config.json',{'SYNTHETIC_CPU_FIXTURE':True})
        cfg=json.loads((ROOT/'prepared/runtime_configuration_glm.json').read_text())
        cfg.update(packages={},python='.'.join(map(str,sys.version_info[:3])),python_executable=sys.executable,
            vllm_source_sha256={},llm_kwargs={})
        write_json(root/'prepared/runtime_configuration_glm.json',cfg)
        write_json(root/'prepared/MODEL_FILE_SHA256.json',{'glm':{'path':str(root/'model'),
            'files':{'config.json':sha(root/'model/config.json')}}})
        write_json(root/'PREINFERENCE_FROZEN_SHA256.json',{'files':{},'SYNTHETIC_CPU_FIXTURE':True})
        write_json(root/'GPU_GO_glm.json',{'authorized_by':'root','device':0,'model':'glm',
            'freeze_sha256':sha(root/'PREINFERENCE_FROZEN_SHA256.json'),'SYNTHETIC_NOT_REAL_AUTHORIZATION':True})
        def job(rid,index,ids=None):
            ids=[1] if ids is None else ids
            row={'request_id':rid,'ordinal':index,'block':0,'task':'boolq','target_id':f'fixture-{index}',
                'selected_ordinal':index,'scope':'target-only','context':'near','repeat':1,'requested_ids':ids,
                'output_ids':ids,'prompt_token_ids':[42+index],'prompt_sha256':'SYNTHETIC',
                'id_map_sha256':'SYNTHETIC','wrapped_prompt':'SYNTHETIC CPU FIXTURE'}
            if ids:row.update(grammar=grammar(ids),grammar_sha256=digest(grammar(ids)))
            return row
        diagnostics=[{**job('diagnostic/ordinary',10,[]),'diagnostic_group':'ordinary_banana','guided':False}]
        diagnostics += [{**job(f'diagnostic/guided/{i}',20+i),'diagnostic_group':'guided_banana','guided':True} for i in range(3)]
        diagnostics += [{**job(f'diagnostic/smoke/{i}',30+i),'diagnostic_group':'synthetic_format_smoke','guided':True} for i in range(3)]
        science=[job(f'science/{i}',i) for i in range(3)]
        # Repeat-like identical prompt/grammar calls must still have distinct
        # engine IDs and verified raw receipt bindings in reanalysis.
        science[1]['prompt_token_ids']=list(science[0]['prompt_token_ids'])
        write_lines(root/'prepared/glm_diagnostic_requests.jsonl',diagnostics)
        write_lines(root/'prepared/science.jsonl',science)
        write_json(root/'prepared/REQUEST_ENVELOPE_glm.json',{'unique_all_calls':10,'execution_order':['mixed'],
            'modes':{'mixed':{'file':'prepared/science.jsonl','requests_per_submission':64}}})
        instances=[]
        class FakeLLM:
            def __init__(self,**values):
                instances.append(self);self.calls=0;self.returned_batches=[]
                mc=S(hf_config=S(),model_impl='vllm',dtype='torch.bfloat16',max_model_len=4096,enforce_eager=True)
                config=S(scheduler_config=S(max_num_seqs=16),parallel_config=S(tensor_parallel_size=1),
                    cache_config=S(enable_prefix_caching=False,gpu_memory_utilization=.7),
                    decoding_config=S(backend='xgrammar',disable_fallback=True))
                self.llm_engine=S(model_config=mc,vllm_config=config,
                    processor=S(generation_config_fields={},process_inputs=lambda request:(None,request)),
                    engine_core=S(shutdown=lambda:None))
            def collective_rpc(self,*values):
                return [{'model_class':cfg['expected_native']['class'],'model_module':cfg['expected_native']['module'],
                    'parameter_dtypes':['torch.bfloat16'],'remote_model_modules_loaded':[]}]
            def get_tokenizer(self):return S(eos_token_id=cfg['primary_eos_token_id'])
            def generate(self,prompts,params,**values):
                self.calls+=1;responses=[]
                for i,(prompt,parameter) in enumerate(zip(prompts,params)):
                    eid=f'synthetic-engine/{self.calls}/{i}'
                    self.llm_engine.processor.process_inputs(S(request_id=eid,prompt_token_ids=prompt['prompt_token_ids'],
                        eos_token_id=cfg['primary_eos_token_id'],sampling_params=parameter))
                    if parameter.guided_decoding:
                        ids=re.findall(r'"(\d+)"',parameter.guided_decoding.regex)
                        raw=compact({key:'Y' for key in ids})
                    else:raw='BANANA'
                    responses.append(S(request_id=eid,prompt_token_ids=list(prompt['prompt_token_ids']),
                        outputs=[S(index=0,text=raw,token_ids=[100+i],finish_reason='stop',stop_reason=None)]))
                target=(self.calls==2 and fault.startswith('diagnostic_')) or (self.calls==4 and fault.startswith('science_'))
                if target:
                    if fault.endswith('_short'):responses=responses[:-1]
                    elif fault.endswith('_extra'):
                        extra=copy.deepcopy(responses[-1]);extra.request_id='unbound-extra-engine-id'
                        extra.outputs[0].text='  UNTRIMMED EXTRA\n';extra.outputs[0].token_ids=[801,802]
                        responses.append(extra)
                    elif fault.endswith('_middle_prompt'):responses[1].prompt_token_ids=[999,998]
                    elif fault.endswith('_reordered'):responses=list(reversed(responses))
                    elif fault.endswith('_unknown_id'):responses[1].request_id='unknown-engine-id'
                    elif fault.endswith('_duplicate_id'):responses[1].request_id=responses[0].request_id
                    elif fault.endswith('_multiple_completions'):
                        responses[1].outputs.append(S(index=1,text=' extra completion \n',token_ids=[901,902,903],finish_reason='length',stop_reason=None))
                    elif fault.endswith('_invalid_raw'):responses[1].outputs[0].text=' {"1":"Y"}\n'
                    elif fault.endswith('_parse_exception'):responses[1].outputs[0].text='SIMULATED_COLLECT_PARSE_EXCEPTION'
                    elif fault.endswith('_generate_exception'):raise RuntimeError('Synthetic engine exception after request processing; completion cost unknown')
                    elif fault.endswith('_empty'):responses=[]
                    elif fault.endswith('_none'):
                        self.returned_batches.append(None);return None
                self.returned_batches.append(copy.deepcopy(responses))
                return responses
        fake=types.ModuleType('vllm');fake.LLM=FakeLLM;fake.SamplingParams=Sample
        fake.__spec__=importlib.util.spec_from_file_location('vllm',root/'fake_vllm/__init__.py')
        fakeparams=types.ModuleType('vllm.sampling_params');fakeparams.GuidedDecodingParams=Guided
        adapter=types.ModuleType('single_rank_nccl_compat');adapter.ACTIVE=True
        adapter.__file__=str(root/'runtime_override/single_rank_nccl_compat.py')
        for name,value in [('vllm',fake),('vllm.sampling_params',fakeparams),('single_rank_nccl_compat',adapter)]:sys.modules[name]=value
        spec=importlib.util.spec_from_file_location('actual_runner_under_test',ROOT/'run_gpu.py')
        runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
        runner.ROOT=root;runner.gpu_processes=lambda:[];runner.check_owned=lambda:[]
        def parse_with_injected_error(raw,ids):
            if raw=='SIMULATED_COLLECT_PARSE_EXCEPTION':raise ValueError('Synthetic collector parser exception')
            return strict_parse(raw,ids)
        runner.parse=parse_with_injected_error
        sys.argv=['run_gpu.py','--model','glm','--device','0'];os.environ['CUDA_VISIBLE_DEVICES']='0'
        caught=None
        try:runner.main()
        except BaseException as error:caught=type(error).__name__+': '+str(error)
        finally:os.environ['CUDA_VISIBLE_DEVICES']=''
        llm=instances[0];run=root/'run_glm'
        receipts=read_lines(run/'raw_return_receipts.jsonl')
        captured=[r for event in receipts for r in event['actual_returns']]
        expected=[r for batch in llm.returned_batches if batch is not None for r in batch]
        assert len(captured)==len(expected),(fault,'returned request lost')
        for actual,original in zip(captured,expected):
            assert actual['engine_request_id']==original.request_id
            assert actual['actual_prompt_token_ids']==original.prompt_token_ids
            assert actual['actual_prompt_token_ids_sha256']==digest(compact(original.prompt_token_ids))
            assert 'UNVERIFIED' in actual['identity_validation_status']
            assert len(actual['outputs'])==len(original.outputs)
            for a,b in zip(actual['outputs'],original.outputs):
                assert a['raw']==b.text and a['output_token_ids']==b.token_ids
        accounting=return_accounting(run)
        success=fault in ('success','science_invalid_raw')
        assert (caught is None)==success,(fault,caught)
        assert llm.calls==(2 if fault.startswith('diagnostic_') else 4),'Automatic retry occurred'
        canonical=read_lines(run/'responses.jsonl') if (run/'responses.jsonl').exists() else []
        if fault=='science_invalid_raw':assert len(canonical)==3 and not canonical[1]['strict_valid']
        if fault=='science_parse_exception':
            assert len(canonical)==1 and accounting['observed_returned_requests']==10
            assert accounting['complete_total_input_tokens'] is not None
        elif not success:
            assert accounting['unresolved_invocations']
            assert accounting['complete_total_input_tokens'] is None and accounting['complete_total_output_tokens'] is None
        if fault=='science_generate_exception':assert accounting['failed_generate_events']==1
        if fault.endswith('_extra'):
            assert receipts[-1]['actual_returns'][-1]['candidate_expected_request_id'] is None
        if not success:assert (run/'FAILURE.json').exists()
        if success:
            _,_,_,checked_costs=load_unique(root,'glm')
            assert checked_costs['actual_generation_return_accounting']['observed_returned_requests']==10
            path=run/'responses.jsonl';saved=path.read_bytes()
            for mutation in ('reuse_engine_id','wrong_return_position','raw_label_changed','wrong_call_id'):
                damaged=copy.deepcopy(canonical)
                if mutation=='reuse_engine_id':damaged[1]['engine_request_id']=damaged[0]['engine_request_id']
                elif mutation=='wrong_return_position':damaged[1]['raw_return_position']=damaged[0]['raw_return_position']
                elif mutation=='raw_label_changed':
                    damaged[0]['raw']='{"1":"N"}';damaged[0]['raw_sha256']=digest(damaged[0]['raw'])
                    damaged[0]['target_label']=0;damaged[0]['parsed']={'1':'N'}
                else:damaged[1]['generation_call_id']=damaged[0]['generation_call_id']+'/WRONG'
                path.write_text(''.join(compact(row)+'\n' for row in damaged))
                rejected=False
                try:load_unique(root,'glm')
                except ValueError:rejected=True
                assert rejected,('Reanalysis accepted corrupted authoritative binding',mutation)
                path.write_bytes(saved)
        result={'scenario':fault,'status':'PASS','exception':caught,'actual_return_objects':len(expected),
            'all_actual_returns_and_all_completions_preserved':True,'untrimmed_text_and_full_token_ids_exact':True,
            'candidate_positions_explicitly_unverified_until_validation':True,'generate_calls':llm.calls,
            'no_automatic_retry':True,'unknown_compute_cost_not_zero':True,'GPU_execution':False,
            'actual_runner_sha256':sha(ROOT/'run_gpu.py'),'receipt_module_sha256':sha(ROOT/'generation_receipts.py')}
        results.append(result);print(fault,'PASS',flush=True)
    write_json(output/'RESULTS.json',{'status':'PASS','scenarios':results,'actual_runner_executed_with_fake_engine':True,
        'real_model_judgments':0,'original_prepared_files_modified':False,
        'analysis_wrapper_sha256':sha(ROOT/'analyze_fresh_replication.py'),
        'reanalysis_rejects_duplicate_engine_id_with_identical_repeat_prompts':True,
        'reanalysis_rejects_wrong_raw_position_text_or_call_binding':True,
        'test_scope':'failure retention, accounting and verified raw binding only'})


if __name__=='__main__':main()
