#!/usr/bin/env python3
"""Current CPU observations of native configuration, sampling and resource bounds."""
import ast
import argparse
import difflib
import importlib.metadata as metadata
import importlib.util
import os
import platform
import subprocess
import sys
import time
from common import *


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--model',choices=MODELS,default='glm');model=parser.parse_args().model
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'This inspection must be CPU-only'
    import torch
    from vllm import SamplingParams
    from vllm.sampling_params import GuidedDecodingParams
    from vllm.transformers_utils.config import get_config
    info={'glm':{'path':'/smoothcloud/models/glm-4-9b-chat','architecture':'ChatGLMModel','class':'ChatGLMForCausalLM',
                    'module':'vllm.model_executor.models.chatglm','primary_eos':151329,'stops':[151329,151336,151338]},
          'qwen':{'path':'/root/models/Qwen/Qwen3-4B-Instruct-2507','architecture':'Qwen3ForCausalLM','class':'Qwen3ForCausalLM',
                    'module':'vllm.model_executor.models.qwen3','primary_eos':151645,'stops':[151643,151645]},
          'smol':{'path':'/home/gpu/intent2s_models/SmolLM2-1.7B-Instruct','architecture':'LlamaForCausalLM','class':'LlamaForCausalLM',
                    'module':'vllm.model_executor.models.llama','primary_eos':2,'stops':[2]}}[model]
    model_path=info['path']
    cfg = get_config(model_path, trust_remote_code=False)
    if model=='glm':
        assert type(cfg).__name__=='ChatGLMConfig' and type(cfg).__module__=='vllm.transformers_utils.configs.chatglm'
    assert cfg.architectures==[info['architecture']]
    installed = Path(importlib.util.find_spec('vllm').origin).parent
    registry = ast.parse((installed/'model_executor/models/registry.py').read_text())
    mappings = {}
    for node in registry.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == '_TEXT_GENERATION_MODELS' for t in node.targets):
            mappings = ast.literal_eval(node.value)
    assert mappings[info['architecture']][1]==info['class']
    generation=json.loads((Path(model_path)/'generation_config.json').read_text())
    sample = SamplingParams(temperature=0, top_k=0, top_p=1, max_tokens=128,
        stop_token_ids=info['stops'],
        guided_decoding=GuidedDecodingParams(regex=grammar([1,2,3,4]), backend='xgrammar', disable_fallback=True))
    sample.update_from_generation_config(generation,info['primary_eos'])
    effective = {k: getattr(sample,k) for k in ('n','temperature','top_k','top_p','min_p','max_tokens','ignore_eos',
        'repetition_penalty','presence_penalty','frequency_penalty','seed','skip_special_tokens','spaces_between_special_tokens')}
    effective.update(stop_token_ids=sorted(sample.stop_token_ids), all_stop_token_ids=sorted(sample.all_stop_token_ids),
        sampling_type=sample.sampling_type.name, guided_backend=sample.guided_decoding.backend,
        guided_disable_fallback=sample.guided_decoding.disable_fallback)
    assert effective['sampling_type'] == 'GREEDY'
    assert (sample.temperature,sample.top_p,sample.top_k,sample.max_tokens) == (0,1,0,128)
    assert sorted(sample.all_stop_token_ids)==info['stops']
    packages = {p: metadata.version(p) for p in (*PACKAGES,'tiktoken','safetensors','flashinfer-python','psutil','msgspec')}
    check = subprocess.run([sys.executable,'-m','pip','check'],capture_output=True,text=True)
    (ROOT/'verification'/f'pip_check_current_{model}.log').write_text(check.stdout+check.stderr)
    llm_kwargs = {'model': model_path, 'tokenizer': model_path, 'trust_remote_code': model=='glm',
        'model_impl': 'vllm', 'dtype': 'bfloat16', 'tensor_parallel_size': 1, 'max_model_len': 4096,
        'gpu_memory_utilization': 0.70, 'max_num_seqs': 16, 'enforce_eager': True,
        'enable_prefix_caching': False, 'disable_log_stats': True,
        'guided_decoding_backend': 'xgrammar', 'guided_decoding_disable_fallback': True,
        'generation_config': 'auto'}
    requested_sample={'temperature':0,'top_k':0,'top_p':1,'max_tokens':128,'stop_token_ids':info['stops']}
    paths = ('model_executor/models/registry.py','model_executor/models/chatglm.py','transformers_utils/config.py',
        'transformers_utils/configs/chatglm.py','transformers_utils/tokenizer.py','config/__init__.py','sampling_params.py',
        'v1/engine/processor.py','distributed/parallel_state.py')
    configuration = {'created_unix': time.time(),
        'observation_type':'CPU-only observed native config and SamplingParams normalization; actual loaded-worker and per-request controls must still be recorded after root GPU GO',
        'python':platform.python_version(),'python_executable':sys.executable,'packages':packages,
        'pip_check_exit':check.returncode,'environment_status':'Inherited machine runtime and recorded dependency conflicts, no package edits',
        'native_config_class':type(cfg).__module__+'.'+type(cfg).__name__,
        'config_loaded_without_trusting_remote_model_code':True,
        'native_registry_mapping':list(mappings[info['architecture']]),'expected_native':info,
        'llm_kwargs':llm_kwargs,'requested_sampling':requested_sample,'cpu_effective_sampling':effective,
        'trust_remote_code_scope':'Needed only for the exact pinned local tokenizer; native config and native vLLM model are used; remote modeling/configuration modules are forbidden at actual runtime.',
        'vllm_source_sha256':{name:sha(installed/name) for name in paths},
        'primary_eos_token_id':info['primary_eos'],'stop_token_ids':info['stops'],
        'block_requests':64,'serial_requests_per_generate':1,'bootstrap_draws':4000,'seeds':SEEDS,
        'cuda_initialized_after_cpu_checks':torch.cuda.is_initialized()}
    assert configuration['cuda_initialized_after_cpu_checks'] is False
    write_json(ROOT/'prepared'/f'runtime_configuration_{model}.json',configuration)
    if model=='glm':write_json(ROOT/'prepared/runtime_configuration.json',configuration)
    original=(ROOT/'source_reference/original_single_rank_nccl_compat.py').read_text()
    actual=(ROOT/'runtime_override/single_rank_nccl_compat.py').read_text()
    diff=''.join(difflib.unified_diff(original.splitlines(True),actual.splitlines(True),
        fromfile='archived/original_single_rank_nccl_compat.py',tofile='runtime_override/single_rank_nccl_compat.py'))
    (ROOT/'verification/NCCL_ADAPTER_PATH_DIFF.patch').write_text(diff)
    assert actual==original.replace("ROOT=Path('/dev/shm/intentsat-review-20261102/controlled_order/runtime_override')",'ROOT=Path(__file__).resolve().parent')
    ast.parse(actual)
    write_json(ROOT/'verification/NCCL_ADAPTER_STATIC_REVIEW.json', {'created_unix':time.time(),'status':'PASS_STATIC_ONLY',
        'archive':'working_package/03_submission/review_audit_supplement.zip',
        'archive_entry':'review_audit/controlled_order/runtime_override/single_rank_nccl_compat.py',
        'original_sha256':sha(ROOT/'source_reference/original_single_rank_nccl_compat.py'),
        'adapted_sha256':sha(ROOT/'runtime_override/single_rank_nccl_compat.py'),
        'only_change':'Event-log root from missing historical /dev/shm location to adapter __file__.parent inside this persistent directory',
        'scope':'Only new_group(ranks=[0], backend=gloo) inside an initialized rank-0/world-size-1 NCCL world is redirected to a real singleton NCCL group.',
        'guards':'Inspected collective/point-to-point operations on replaced groups raise rather than fabricate data; real barrier/group metadata/destruction remain permitted.',
        'static_vllm_evidence':'parallel_state.py creates device NCCL and CPU Gloo groups, bypasses common data collectives at world_size==1; source hash frozen.',
        'limitations':'This is a scoped static inspection of the used installed rank-one paths, not a universal guarantee about every communication API. Actual adapter events and runtime failures must be retained.',
        'no_system_dependency_changes':True,'no_gpu_execution_in_this_review':True})
    print(compact({'native_config_class':configuration['native_config_class'],'native_mapping':configuration['native_registry_mapping'],
                   'effective_sampling':effective,'pip_check_exit':check.returncode,'cuda_initialized':False}),flush=True)


if __name__=='__main__':
    main()
