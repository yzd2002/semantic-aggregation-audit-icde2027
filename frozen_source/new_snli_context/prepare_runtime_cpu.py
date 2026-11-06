#!/usr/bin/env python3
"""Observe native runtime/sampling on CPU; never load model parameters or CUDA."""
import argparse
import ast
import collections
import importlib.metadata as metadata
import importlib.util
import os
import platform
import subprocess
import sys
import time
from common import *
from runtime_contract import cache_environment,load_diagnostic_jobs,DIAGNOSTIC_GROUPS

BASELINE_SHA256='e0c61d331338e544939ac28d3c656a61eee79659ab8a098d0f6bdef92fa93832'


def check_baseline():
    manifest=ROOT/'CPU_REVIEW_CANDIDATE_SHA256.json'
    assert sha(manifest)==BASELINE_SHA256,'Original CPU candidate was changed'
    data=json.loads(manifest.read_text());assert len(data['files'])==105
    for name,want in data['files'].items():
        assert sha(ROOT/name)==want,'Original CPU candidate file changed: '+name
    return {'status':'PASS','candidate_sha256':BASELINE_SHA256,'file_count':105,
        'original_files_byte_identical':True,'original_analysis_functions_unchanged':True}


def prepare_model(model,torch,SamplingParams,GuidedDecodingParams,get_config):
    model_entry=json.loads((ROOT/'prepared/MODEL_FILE_SHA256.json').read_text())[model]
    info={'qwen':{'architecture':'Qwen3ForCausalLM','class':'Qwen3ForCausalLM',
                'module':'vllm.model_executor.models.qwen3','primary_eos':151645,'stops':[151643,151645]},
          'smol':{'architecture':'LlamaForCausalLM','class':'LlamaForCausalLM',
                'module':'vllm.model_executor.models.llama','primary_eos':2,'stops':[2]},
          'glm':{'architecture':'ChatGLMModel','class':'ChatGLMForCausalLM',
                'module':'vllm.model_executor.models.chatglm','primary_eos':151329,'stops':[151329,151336,151338]}}[model]
    info={'path':model_entry['path'],**info};model_path=Path(info['path'])
    cfg=get_config(str(model_path),trust_remote_code=False)
    assert cfg.architectures==[info['architecture']]
    if model=='glm':assert type(cfg).__module__=='vllm.transformers_utils.configs.chatglm'
    installed=Path(importlib.util.find_spec('vllm').origin).parent
    registry=ast.parse((installed/'model_executor/models/registry.py').read_text())
    mappings={}
    for node in registry.body:
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='_TEXT_GENERATION_MODELS' for t in node.targets):
            mappings=ast.literal_eval(node.value)
    assert mappings[info['architecture']][1]==info['class']
    sample=SamplingParams(temperature=0,top_k=0,top_p=1,max_tokens=128,stop_token_ids=info['stops'],
        guided_decoding=GuidedDecodingParams(regex=grammar([1,2,3,4]),backend='xgrammar',disable_fallback=True))
    sample.update_from_generation_config(json.loads((model_path/'generation_config.json').read_text()),info['primary_eos'])
    effective={k:getattr(sample,k) for k in ('n','temperature','top_k','top_p','min_p','max_tokens','ignore_eos',
        'repetition_penalty','presence_penalty','frequency_penalty','seed','skip_special_tokens','spaces_between_special_tokens')}
    effective.update(stop_token_ids=sorted(sample.stop_token_ids),all_stop_token_ids=sorted(sample.all_stop_token_ids),
        sampling_type=sample.sampling_type.name,guided_backend=sample.guided_decoding.backend,
        guided_disable_fallback=sample.guided_decoding.disable_fallback)
    assert (sample.temperature,sample.top_p,sample.top_k,sample.max_tokens,sample.n)==(0,1,0,128,1)
    assert sorted(sample.all_stop_token_ids)==info['stops'] and not sample.ignore_eos
    assert sample.sampling_type.name=='GREEDY'
    package_names=tuple(dict.fromkeys((*PACKAGES,'safetensors','flashinfer-python','psutil','msgspec')))
    packages={name:metadata.version(name) for name in package_names}
    pip_check=subprocess.run([sys.executable,'-m','pip','check'],capture_output=True,text=True)
    (ROOT/'verification'/f'pip_check_runtime_{model}.log').write_text(pip_check.stdout+pip_check.stderr)
    llm_kwargs={'model':str(model_path),'tokenizer':str(model_path),'trust_remote_code':model=='glm',
        'model_impl':'vllm','dtype':'bfloat16','tensor_parallel_size':1,'max_model_len':4096,
        'gpu_memory_utilization':.70,'max_num_seqs':16,'enforce_eager':True,'enable_prefix_caching':False,
        'disable_log_stats':True,'guided_decoding_backend':'xgrammar','guided_decoding_disable_fallback':True,
        'generation_config':'auto'}
    source_paths=('model_executor/models/registry.py','model_executor/models/chatglm.py','model_executor/models/qwen3.py',
        'model_executor/models/llama.py','transformers_utils/config.py','transformers_utils/configs/chatglm.py',
        'transformers_utils/tokenizer.py','config/__init__.py','sampling_params.py','v1/engine/processor.py',
        'distributed/parallel_state.py','envs.py','utils/__init__.py','v1/serial_utils.py','v1/engine/core_client.py','v1/engine/core.py',
        'executor/uniproc_executor.py')
    result={'created_unix':time.time(),'observation_type':'CPU native config and normalized sampling only; actual worker must be recorded after root GO',
        'python':platform.python_version(),'python_executable':sys.executable,'packages':packages,
        'pip_check_exit':pip_check.returncode,'environment_status':'Inherited runtime; recorded conflicts; no installed package edits',
        'native_config_class':type(cfg).__module__+'.'+type(cfg).__name__,'config_loaded_without_trusting_remote_model_code':True,
        'native_registry_mapping':list(mappings[info['architecture']]),'expected_native':info,'llm_kwargs':llm_kwargs,
        'requested_sampling':{'temperature':0,'top_k':0,'top_p':1,'max_tokens':128,'stop_token_ids':info['stops']},
        'cpu_effective_sampling':effective,'primary_eos_token_id':info['primary_eos'],'stop_token_ids':info['stops'],
        'trust_remote_code_scope':'Only pinned local GLM tokenizer; native config/model; remote modeling/configuration modules forbidden',
        'vllm_source_sha256':{name:sha(installed/name) for name in source_paths},
        'native_process_environment':{'VLLM_RPC_BASE_PATH':str(ROOT/'rpc'/model[0]),'VLLM_ALLOW_INSECURE_SERIALIZATION':'1'},
        'serialization_scope':'Generic pickle/cloudpickle fallback in this process and its own local vLLM children, not a function allowlist. Actual protocol sends only frozen trusted worker_identity; no external serialized input is read.',
        'block_requests':64,'serial_requests_per_generate':1,'bootstrap_draws':4000,'seeds':SEEDS,
        'cuda_initialized_after_cpu_checks':torch.cuda.is_initialized(),
        'full_model_file_check_before_inference':'Runner rehashes every file in original prepared/MODEL_FILE_SHA256.json before model import/load'}
    assert not result['cuda_initialized_after_cpu_checks']
    write_json(ROOT/'prepared'/f'runtime_configuration_{model}.json',result)
    modes={};all_ids=[]
    for mode,kind,width in [('mixed','main',64),('serial','serial',1)]:
        relative=f'prepared/{model}_{kind}_requests.jsonl';jobs=read_lines(ROOT/relative)
        expected_n=1792 if mode=='mixed' else 448;assert len(jobs)==expected_n
        assert len({j['request_id'] for j in jobs})==expected_n
        for offset in range(0,len(jobs),width):
            assert len({j['block'] for j in jobs[offset:offset+width]})==1
        assert all(j['execution_mode']==mode for j in jobs)
        modes[mode]={'file':relative,'sha256':sha(ROOT/relative),'planned_calls':len(jobs),
            'requests_per_submission':width,'planned_input_tokens':sum(len(j['prompt_token_ids']) for j in jobs)}
        all_ids.extend(j['request_id'] for j in jobs)
    diagnostics=load_diagnostic_jobs(ROOT,model)
    groups={name:[j['request_id'] for j in diagnostics if j['diagnostic_group']==name] for name in DIAGNOSTIC_GROUPS}
    assert list(map(len,groups.values()))==[1,5,14]
    all_ids.extend(j['request_id'] for j in diagnostics);assert len(all_ids)==len(set(all_ids))==2260
    envelope={'model':model,'execution_order':['mixed','serial'],'modes':modes,
        'diagnostic_file':f'prepared/{model}_diagnostic_requests.jsonl',
        'diagnostic_sha256':sha(ROOT/'prepared'/f'{model}_diagnostic_requests.jsonl'),
        'diagnostic_groups':groups,'diagnostic_execution_order':list(DIAGNOSTIC_GROUPS),
        'unique_scientific_calls':2240,'unique_diagnostic_calls':20,'unique_all_calls':2260,
        'same_loaded_engine_for_diagnostics_and_science':True,'automatic_retry':False,
        'original_prepared_bytes_modified':False,'diagnostic_derived_audit_metadata_only':['diagnostic_group','grammar_sha256']}
    write_json(ROOT/'prepared'/f'REQUEST_ENVELOPE_{model}.json',envelope)
    return {'model':model,'planned_scientific_calls':2240,'planned_diagnostic_calls':20,
        'native_config_class':result['native_config_class'],'stops':info['stops'],'cuda_initialized':False,
        'configuration_sha256':sha(ROOT/'prepared'/f'runtime_configuration_{model}.json')}


def main():
    baseline=check_baseline()
    assert not any((ROOT/f'run_{m}').exists() or (ROOT/f'GPU_GO_{m}.json').exists() for m in MODELS)
    os.environ.update(cache_environment(ROOT,'cpu_runtime_preparation'))
    os.environ['CUDA_VISIBLE_DEVICES']=''
    import torch
    from vllm import SamplingParams
    from vllm.sampling_params import GuidedDecodingParams
    from vllm.transformers_utils.config import get_config
    models=[prepare_model(m,torch,SamplingParams,GuidedDecodingParams,get_config) for m in MODELS]
    assert not torch.cuda.is_initialized()
    check_baseline()
    write_json(ROOT/'prepared/REQUEST_ENVELOPE.json',{'models':models,'all_unique_scientific_calls':6720,
        'all_unique_diagnostic_calls':60,'all_unique_calls':6780,'new_model_judgments':0})
    write_json(ROOT/'verification/CPU_RUNTIME_PREPARATION.json',{'status':'PASS','baseline':baseline,'models':models,
        'CUDA_VISIBLE_DEVICES':'','torch_cuda_initialized':False,'new_model_judgments':0,'system_packages_modified':False})
    print(compact({'status':'PASS','models':models,'calls':6780,'new_model_judgments':0}),flush=True)


if __name__=='__main__':main()
