#!/usr/bin/env python3
"""Execute every original SNLI request through the actual runner with a fake engine.

All outputs live under an explicitly synthetic CPU-only fixture. No torch,
real vLLM, model weights, CUDA call or real GPU authorization is used here.
"""
import argparse
import contextlib
import copy
import importlib.util
import io
import os
import re
import shutil
import sys
import types
from common import *
from runtime_contract import load_diagnostic_jobs
from analyze_snli_runtime import analyze_all,load_unique
from assemble_snli_primary import assemble
from global_primary.primary_family import primary_interval

S=types.SimpleNamespace


class Guided:
    def __init__(self,**values):self.__dict__.update(values)


class Sample:
    def __init__(self,**values):
        self.__dict__.update(n=1,temperature=0,top_k=0,top_p=1,min_p=0,max_tokens=128,ignore_eos=False,
            repetition_penalty=1,presence_penalty=0,frequency_penalty=0,seed=None,skip_special_tokens=True,
            spaces_between_special_tokens=True)
        self.__dict__.update(values);self.all_stop_token_ids=set(self.stop_token_ids);self.sampling_type=S(name='GREEDY')


def target_label(job):
    index=job['selected_ordinal'];repeat=job['repeat'];context=job['context'];scope=job['scope']
    if scope=='singleton':return (index+repeat)%2
    if scope=='target-only':return index%2
    if scope=='full-last':return (index+(context=='near'))%2
    group=index%4
    if group==0:return int(context=='near')
    if group==1:return int(context=='near' and repeat==2)
    if group==2:return int((context=='near' and repeat==2) or (context=='random' and repeat==1))
    return int(repeat==2)


def execute_fixture(output,model,invalid_target=False):
    root=output/(model+('_one_invalid' if invalid_target else '_complete'));root.mkdir()
    for directory in ('prepared','selection','runtime_override','model','global_primary'):(root/directory).mkdir()
    write_json(root/'SYNTHETIC_CPU_FIXTURE.json',{'real_model_judgments':0,'GPU_execution':False,
        'all_outputs_are_synthetic_not_model_evidence':True,'fake_engine_runs_actual_runner':True})
    for name in ('schedule.jsonl','serial_schedule.jsonl',f'{model}_main_requests.jsonl',f'{model}_serial_requests.jsonl',
            f'{model}_diagnostic_requests.jsonl',f'REQUEST_ENVELOPE_{model}.json'):
        shutil.copyfile(ROOT/'prepared'/name,root/'prepared'/name)
    shutil.copyfile(ROOT/'selection/targets.jsonl',root/'selection/targets.jsonl')
    for path in (ROOT/'global_primary').iterdir():shutil.copyfile(path,root/'global_primary'/path.name)
    write_json(root/'model/config.json',{'SYNTHETIC_CPU_FIXTURE':True})
    cfg=json.loads((ROOT/'prepared'/f'runtime_configuration_{model}.json').read_text())
    cfg.update(packages={},python='.'.join(map(str,sys.version_info[:3])),python_executable=sys.executable,
        vllm_source_sha256={},llm_kwargs={})
    write_json(root/'prepared'/f'runtime_configuration_{model}.json',cfg)
    write_json(root/'prepared/MODEL_FILE_SHA256.json',{model:{'path':str(root/'model'),
        'files':{'config.json':sha(root/'model/config.json')}}})
    write_json(root/'PREINFERENCE_FROZEN_SHA256.json',{'SYNTHETIC_CPU_FIXTURE':True,
        'files':{str(p.relative_to(root)):sha(p) for name in ('prepared','selection','global_primary') for p in (root/name).iterdir()}})
    write_json(root/f'GPU_GO_{model}.json',{'authorized_by':'root','device':0,'model':model,
        'freeze_sha256':sha(root/'PREINFERENCE_FROZEN_SHA256.json'),'SYNTHETIC_NOT_REAL_AUTHORIZATION':True})
    diagnostics=load_diagnostic_jobs(root,model)
    batches=[[j for j in diagnostics if j['diagnostic_group']==name]
        for name in ('ordinary_banana','guided_banana','synthetic_format_smoke')]
    for kind,width in [('main',64),('serial',1)]:
        jobs=read_lines(root/'prepared'/f'{model}_{kind}_requests.jsonl')
        batches.extend(jobs[offset:offset+width] for offset in range(0,len(jobs),width))
    expected_science_labels={j['request_id']:target_label(j) for batch in batches[3:] for j in batch}
    instances=[]
    class FakeLLM:
        def __init__(self,**values):
            self.calls=0;self.shutdowns=0;instances.append(self)
            mc=S(hf_config=S(),model_impl='vllm',dtype='torch.bfloat16',max_model_len=4096,enforce_eager=True)
            config=S(scheduler_config=S(max_num_seqs=16),parallel_config=S(tensor_parallel_size=1),
                cache_config=S(enable_prefix_caching=False,gpu_memory_utilization=.7),
                decoding_config=S(backend='xgrammar',disable_fallback=True))
            def shutdown():self.shutdowns+=1
            self.llm_engine=S(model_config=mc,vllm_config=config,
                processor=S(generation_config_fields={},process_inputs=lambda request:(None,request)),engine_core=S(shutdown=shutdown))
        def collective_rpc(self,*values):
            return [{'model_class':cfg['expected_native']['class'],'model_module':cfg['expected_native']['module'],
                'parameter_dtypes':['torch.bfloat16'],'remote_model_modules_loaded':[]}]
        def get_tokenizer(self):return S(eos_token_id=cfg['primary_eos_token_id'])
        def generate(self,prompts,params,**values):
            jobs=batches[self.calls];self.calls+=1;responses=[]
            assert len(prompts)==len(params)==len(jobs)
            for index,(prompt,parameter,job) in enumerate(zip(prompts,params,jobs)):
                assert prompt['prompt_token_ids']==job['prompt_token_ids']
                engine_id=f'SYNTHETIC/{model}/{self.calls}/{index}'
                self.llm_engine.processor.process_inputs(S(request_id=engine_id,prompt_token_ids=prompt['prompt_token_ids'],
                    eos_token_id=cfg['primary_eos_token_id'],sampling_params=parameter))
                if parameter.guided_decoding:
                    ids=re.findall(r'"(\d+)"',parameter.guided_decoding.regex)
                    target=expected_science_labels.get(job['request_id'],1)
                    raw=compact({key:('Y' if target else 'N') if key=='1' else ('Y' if int(key)%2 else 'N') for key in ids})
                else:raw='BANANA'
                if invalid_target and job.get('execution_mode')=='mixed' and job.get('selected_ordinal')==0 and job.get('scope')=='full-first' and job.get('context')=='near' and job.get('repeat')==1:
                    raw=' {"1":"Y"}\n'
                responses.append(S(request_id=engine_id,prompt_token_ids=list(prompt['prompt_token_ids']),
                    outputs=[S(index=0,text=raw,token_ids=[701+index,801+index],finish_reason='stop',stop_reason=cfg['primary_eos_token_id'])]))
            return responses
    fake=types.ModuleType('vllm');fake.LLM=FakeLLM;fake.SamplingParams=Sample
    fake.__spec__=importlib.util.spec_from_file_location('vllm',root/'fake_vllm/__init__.py')
    fakeparams=types.ModuleType('vllm.sampling_params');fakeparams.GuidedDecodingParams=Guided
    adapter=types.ModuleType('single_rank_nccl_compat');adapter.ACTIVE=True;adapter.__file__=str(root/'runtime_override/single_rank_nccl_compat.py')
    for name,value in [('vllm',fake),('vllm.sampling_params',fakeparams),('single_rank_nccl_compat',adapter)]:sys.modules[name]=value
    spec=importlib.util.spec_from_file_location('actual_snli_runner_under_cpu_test',ROOT/'run_gpu.py')
    runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
    runner.ROOT=root;runner.gpu_processes=lambda:[];runner.check_owned=lambda:[]
    sys.argv=['run_gpu.py','--model',model,'--device','0'];os.environ['CUDA_VISIBLE_DEVICES']='0'
    os.environ.update(cfg['native_process_environment'])
    # Every potential GPU entry is replaced before the actual runner is called.
    # Captured log text stays inside the synthetic fixture.
    with (root/'CPU_FAKE_RUNNER.log').open('w') as stream,contextlib.redirect_stdout(stream):
        try:runner.main()
        finally:os.environ['CUDA_VISIBLE_DEVICES']=''
    result=analyze_all(root,model,root/'ANALYSIS_SYNTHETIC_NOT_MODEL_EVIDENCE')
    run=root/f'run_{model}';totals=json.loads((run/'ALL_WORK_COMPLETE.json').read_text())
    assert totals['unique_scientific_calls']==2240 and totals['diagnostic_calls']==20 and totals['all_calls']==2260
    assert instances[0].calls==3+28+448 and instances[0].shutdowns==1
    assert result['costs']['unique_all_returned']==2260 and result['costs']['all_observed_tokens']['output_tokens']==4520
    assert result['costs']['all_observed_tokens']['input_tokens']==sum(len(j['prompt_token_ids']) for b in batches for j in b)
    assert not result['costs']['costs_unknown_rather_than_zero_when_any_attempt_is_unresolved']
    component=result['component'];assert component['synthetic_cpu_fixture_not_model_evidence'] is True
    rows=sorted(read_lines(root/'selection/targets.jsonl'),key=lambda r:r['selected_ordinal'])
    expected_excess=[1 if r['selected_ordinal']%4==0 else -1 if r['selected_ordinal']%4==3 else 0
        for r in rows if not invalid_target or r['selected_ordinal']!=0]
    reference=primary_interval(expected_excess,task='snli',model=model)
    for key in reference:assert component[key]==reference[key],key
    main=result['primary']['tasks']['snli'];secondary=result['secondary']['tasks']['snli']
    assert main['common12_n']==(127 if invalid_target else 128)
    for condition in main['fixed_relation_conditions']:
        assert condition['fixed_denominator']==128
        affected=invalid_target and condition['scope']=='full-first' and condition['context']=='near' and condition['repeat']==1
        assert condition['status']==('abstain' if affected else 'complete')
    affected_comparisons=0
    for comparison in main['fixed_relation_comparisons']:
        assert comparison['fixed_denominator']==128
        if invalid_target and comparison['scope']=='full-first' and comparison['comparison'] in ('replacement_repeat1','replay_near'):
            assert comparison['status']=='abstain';affected_comparisons+=1
        else:assert comparison['status']=='complete'
    assert affected_comparisons==(2 if invalid_target else 0)
    for condition in secondary['serial_subset']['fixed_relation_conditions']:
        assert condition['fixed_denominator']==32 and condition['status']=='complete'
    return {'model':model,'one_invalid_scientific_target':invalid_target,'status':'PASS','science_calls':2240,'diagnostic_calls':20,
        'common12_n':main['common12_n'],'COUNT_denominator_128_and_serial_32_verified':True,
        'original_request_bytes_used':True,'native_controls_and_durable_chain_verified':True,
        'authoritative_primary_matches_independent_synthetic_e_vector':True,
        'actual_runner_sha256':sha(ROOT/'run_gpu.py'),'analysis_sha256':sha(ROOT/'analyze_snli_runtime.py'),
        'raw_return_receipts_sha256':sha(run/'raw_return_receipts.jsonl'),
        'generation_validation_sha256':sha(run/'generation_validation.jsonl'),
        'canonical_mixed_sha256':sha(run/'responses.jsonl'),'canonical_serial_sha256':sha(run/'serial/responses.jsonl'),
        'synthetic_component_path':str(root/'ANALYSIS_SYNTHETIC_NOT_MODEL_EVIDENCE/global_primary_component.json'),
        'real_model_judgments':0,'GPU_execution':False}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    assert 'torch' not in sys.modules and 'vllm' not in sys.modules,'Run this CPU check in a fresh interpreter'
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=False)
    results=[]
    for model in MODELS:
        results.append(execute_fixture(out,model));print(model,'full 2260-call fake chain PASS',flush=True)
    results.append(execute_fixture(out,'qwen',invalid_target=True))
    print('one invalid target, fixed COUNT abstention PASS',flush=True)
    rejected=False
    try:assemble([Path(r['synthetic_component_path']) for r in results[:3]])
    except ValueError:rejected=True
    assert rejected,'Synthetic components were accepted as real model evidence'
    assert 'torch' not in sys.modules,'CPU fake checks imported a real GPU runtime'
    write_json(out/'RESULTS.json',{'status':'PASS','cases':results,'all_outputs_synthetic_not_model_evidence':True,
        'synthetic_components_rejected_by_real_assembly':True,'real_model_judgments':0,'GPU_execution':False,
        'real_torch_imported':False,'original_input_files_modified':False})


if __name__=='__main__':main()
