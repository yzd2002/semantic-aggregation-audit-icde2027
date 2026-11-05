#!/usr/bin/env python3
"""Run a small synthetic guided-decoding check under vLLM's declared dependencies."""
import argparse
import importlib.metadata as md
import json
import os
import subprocess
import time
from pathlib import Path

import psutil
from run_compact_prompt_ablation import prompt_without_example
from run_keyed_order_audit import keyed_prompt, keyed_regex, parse_keyed
from run_validation import parse_compact
import run_wands


def gpu_processes():
    out = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True)
    return {int(x.strip()) for x in out.splitlines() if x.strip().isdigit()}


def assert_own_gpu():
    process = psutil.Process(os.getpid())
    own = {process.pid} | {p.pid for p in process.children(recursive=True)}
    used = gpu_processes()
    if used and not used <= own:
        raise RuntimeError('Another GPU process appeared; stopping this synthetic check.')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    if os.environ.get('VLLM_USE_V1') != '1' or Path(args.model).name != 'Qwen3-4B-Instruct-2507':
        raise ValueError('This check requires V1 and the recorded Qwen model directory.')
    packages = ('vllm', 'torch', 'transformers', 'numpy', 'xgrammar', 'outlines-core', 'setuptools')
    versions = {name: md.version(name) for name in packages}
    expected = {'vllm': '0.10.2', 'torch': '2.8.0+cu128', 'transformers': '4.57.1',
                'numpy': '2.2.6', 'xgrammar': '0.1.23', 'outlines-core': '0.2.11',
                'setuptools': '79.0.1'}
    if versions != expected:
        raise RuntimeError(f'Unexpected package versions: {versions}')
    if gpu_processes():
        raise RuntimeError('GPU is already in use; no model was loaded.')
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / 'setup.json').write_text(json.dumps({'protocol': 'PROTOCOL_vllm_declared_deps_smoke.md',
        'model_directory_name': Path(args.model).name, 'versions': versions, 'engine': 'V1',
        'dtype': 'bfloat16', 'max_model_len': 4096, 'max_num_seqs': 16,
        'gpu_memory_utilization': .70, 'enforce_eager': True, 'prefix_caching': True}, indent=2) + '\n')

    import torch
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import GuidedDecodingParams
    llm = LLM(model=args.model, tokenizer=args.model, trust_remote_code=False, dtype='bfloat16',
              max_model_len=4096, gpu_memory_utilization=.70, max_num_seqs=16, enforce_eager=True,
              enable_prefix_caching=True, disable_log_stats=True)
    tokenizer = llm.get_tokenizer()
    assert_own_gpu()
    def wrap(system, user):
        return tokenizer.apply_chat_template([{'role': 'system', 'content': system},
                                               {'role': 'user', 'content': user}],
                                              tokenize=False, add_generation_prompt=True)
    records = []
    def run(name, system, prompt, cap, regex=None):
        assert_own_gpu()
        params = SamplingParams(temperature=0, max_tokens=cap,
                                guided_decoding=GuidedDecodingParams(regex=regex) if regex else None)
        start = time.perf_counter()
        response = llm.generate([wrap(system, prompt)], params, use_tqdm=False)[0]
        output = response.outputs[0]
        item = {'case': name, 'system': system, 'prompt': prompt, 'regex': regex,
                'output': output.text.strip(), 'finish_reason': output.finish_reason,
                'prompt_tokens': len(response.prompt_token_ids), 'output_tokens': len(output.token_ids),
                'elapsed_seconds': time.perf_counter() - start}
        records.append(item)
        (args.out / 'records.json').write_text(json.dumps(records, indent=2, ensure_ascii=False) + '\n')
        return item['output']

    system = 'Follow the user instruction exactly.'
    control = 'Reply with only the word BANANA.'
    ids = list(range(1, 9))
    ordinary = run('control_unconstrained', system, control, 128)
    compact = run('control_compact', system, control, 32, '[YN]{4}')
    keyed = run('control_keyed', system, control, 128, keyed_regex(ids))
    controls = {'unconstrained_invalid_compact': parse_compact(ordinary, 4) is None,
                'unconstrained_invalid_keyed': parse_keyed(ordinary, ids) is None,
                'compact_valid': parse_compact(compact, 4) is not None,
                'keyed_valid': parse_keyed(keyed, ids) is not None}
    (args.out / 'controls.json').write_text(json.dumps(controls, indent=2) + '\n')
    if not all(controls.values()):
        raise RuntimeError('Negative control failed; no synthetic product prompts were run.')

    items = ['Red ceramic mug with a handle.', 'Blue ceramic mug.', 'Red plastic mug.',
             'Red ceramic plate.', 'Red ceramic coffee mug.', 'Set of red ceramic bowls.',
             'Red mug lid only.', 'Small red ceramic cup.']
    part = {'query': 'red ceramic mug', 'items': [{'id': i + 1, 'text': x} for i, x in enumerate(items)]}
    compact_out = run('synthetic_compact', run_wands.SYSTEM, prompt_without_example(part, None), 32, '[YN]{8}')
    keyed_out = run('synthetic_keyed', run_wands.SYSTEM, keyed_prompt(part, None), 128, keyed_regex(ids))
    checks = {'compact_valid': parse_compact(compact_out, 8) is not None,
              'keyed_valid': parse_keyed(keyed_out, ids) is not None}
    (args.out / 'summary.json').write_text(json.dumps({'controls': controls, 'synthetic': checks,
        'gpu': torch.cuda.get_device_name(0), 'cuda': torch.version.cuda,
        'interpretation': 'Synthetic formatting check only; not a benchmark or full environment reproduction.'}, indent=2) + '\n')
    if not all(checks.values()):
        raise RuntimeError('At least one synthetic output failed its grammar.')
    print('Both guided negative controls and both synthetic output parsers passed.')


if __name__ == '__main__':
    main()
