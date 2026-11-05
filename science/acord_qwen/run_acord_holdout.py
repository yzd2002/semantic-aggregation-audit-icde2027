#!/usr/bin/env python3
"""Execute the frozen ACORD keyed, selective and pointwise transfer check."""
import argparse
import collections
import hashlib
import importlib.metadata as metadata
import json
import os
import subprocess
import time
from pathlib import Path

import psutil
from order_fallback import combine
from acord_prompts import SYSTEM, RETRY_SUFFIX, IDS, keyed_prompt, point_prompt
from acord_prompts import keyed_regex as _keyed_regex, parse_keyed as _parse_keyed, parse_point as parse_one

PROTOCOL = 'PROTOCOL_acord_holdout.md'


def keyed_regex(ids):
    if ids != IDS: raise ValueError('Expected chunk-local IDs 1--8')
    return _keyed_regex()


def parse_keyed(text, ids):
    if ids != IDS: raise ValueError('Expected chunk-local IDs 1--8')
    return _parse_keyed(text)


def sha(text):
    return hashlib.sha256(text.encode('utf8')).hexdigest()


def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def memory_guard(initial=False):
    stat={k:int(v) for k,v in (line.split() for line in Path('/sys/fs/cgroup/memory.stat').read_text().splitlines())}
    available=int(Path('/sys/fs/cgroup/memory.max').read_text())-int(Path('/sys/fs/cgroup/memory.current').read_text())
    available+=stat.get('active_file',0)+stat.get('inactive_file',0)
    reserve=(9 if initial else 1.5)*1024**3
    if available < reserve:raise RuntimeError('Memory reserve too low; no other processes will be stopped')
    if initial:
        st=os.statvfs('/dev/shm')
        if st.f_bavail*st.f_frsize < 10*1024**3:raise RuntimeError('Temporary storage reserve below 10 GiB')
        st=os.statvfs('/')
        if st.f_bavail*st.f_frsize < 32*1024**2:raise RuntimeError('Root filesystem reserve below 32 MiB')


def gpu_processes():
    output = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True)
    return {int(x.strip()) for x in output.splitlines() if x.strip().isdigit()}


def assert_own_gpu():
    me = psutil.Process(os.getpid())
    ours = {me.pid} | {x.pid for x in me.children(recursive=True)}
    if gpu_processes() - ours:
        raise RuntimeError('Another GPU process appeared; stop this run without affecting it.')


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf8')


def write_rows(path, rows):
    path.write_text(''.join(json.dumps(x, ensure_ascii=False) + '\n' for x in rows), encoding='utf8')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', type=Path, required=True)
    ap.add_argument('--selection', type=Path, required=True)
    ap.add_argument('--preflight', type=Path, required=True)
    ap.add_argument('--protocol', type=Path, required=True)
    ap.add_argument('--amendment', type=Path, required=True)
    ap.add_argument('--decision', type=Path, required=True)
    ap.add_argument('--weights-manifest', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    if os.environ.get('VLLM_USE_V1') != '1' or a.model.name != 'Qwen3-4B-Instruct-2507':
        raise ValueError('The protocol requires the recorded Qwen model and vLLM V1')
    packages = ('vllm', 'torch', 'transformers', 'numpy', 'xgrammar', 'outlines-core', 'setuptools')
    versions = {name: metadata.version(name) for name in packages}
    expected = {'vllm': '0.10.2', 'torch': '2.8.0+cu128', 'transformers': '4.57.1',
                'numpy': '2.2.6', 'xgrammar': '0.1.23', 'outlines-core': '0.2.11', 'setuptools': '79.0.1'}
    if versions != expected:
        raise RuntimeError('Dependency overlay differs from the specified configuration: ' + str(versions))
    with (a.selection / 'inputs.jsonl').open(encoding='utf8') as f:
        rows = [json.loads(x) for x in f if x.strip()]
    selection = json.loads((a.selection / 'selection.json').read_text(encoding='utf8'))
    selection_sha = hashlib.sha256((a.selection / 'inputs.jsonl').read_bytes()).hexdigest()
    preflight = json.loads((a.preflight / 'summary.json').read_text(encoding='utf8'))
    decision = json.loads(a.decision.read_text(encoding='utf8'))
    selected_variant = decision['variants'].get('2400', {})
    if (selection['protocol'] != PROTOCOL or selection['inputs_sha256'] != selection_sha
            or len(rows) != 2 * selection['eligible_queries'] or not rows
            or any(len(row['items']) != 32 for row in rows)
            or file_sha(a.protocol) != selection['protocol_sha256']
            or selection.get('prefix_length') != 2400
            or decision.get('selected_cap_unicode_code_points') != 2400
            or file_sha(a.amendment) != selection.get('prefix_amendment_sha256')
            or file_sha(a.amendment) != decision.get('prefix_search_amendment_sha256')
            or decision.get('base_600_inputs_sha256') != selection.get('base_inputs_sha256')
            or decision.get('source_sha256') != selection.get('source_sha256')
            or selected_variant.get('selection_sha256') != selection_sha
            or selected_variant.get('preflight_summary_sha256') != file_sha(a.preflight / 'summary.json')
            or selected_variant.get('prompt_code_sha256') != preflight['prompt_code_sha256']
            or selected_variant.get('overflow_pool_count') != 0
            or preflight['selection_sha256'] != selection_sha or preflight['overflow_pool_count'] != 0
            or preflight['prompt_code_sha256'] != file_sha(Path(__file__).with_name('acord_prompts.py'))
            or preflight['prompt_records_sha256'] != file_sha(a.preflight / 'prompts.jsonl')):
        raise ValueError('Unexpected selected inputs')
    for name,expected_sha in preflight['tokenizer_assets_sha256'].items():
        if file_sha(a.model/name) != expected_sha:raise RuntimeError('Tokenizer asset changed: '+name)
    memory_guard(initial=True)
    if gpu_processes():
        raise RuntimeError('GPU in use; no model loaded')
    expected_weights={}
    for line in a.weights_manifest.read_text(encoding='utf8').splitlines():
        expected,path=line.split(None,1)
        name=Path(path.strip()).name
        if name in expected_weights:raise ValueError('Duplicate weight manifest entry')
        expected_weights[name]=expected
    if set(expected_weights)!={f'model-0000{i}-of-00003.safetensors' for i in (1,2,3)}:
        raise ValueError('Unexpected weight manifest')
    for name,expected in expected_weights.items():
        memory_guard()
        if file_sha(a.model/name)!=expected:raise RuntimeError('Model weight SHA mismatch: '+name)
    a.out.mkdir(parents=True, exist_ok=False)
    summary = {'protocol': PROTOCOL, 'selection_inputs_sha256': selection_sha,
               'model': a.model.name, 'versions': versions, 'engine': 'V1',
               'runtime': {'dtype': 'bfloat16', 'max_model_len': 4096, 'max_num_seqs': 16,
                           'gpu_memory_utilization': .70, 'enforce_eager': True, 'prefix_caching': True},
               'system': SYSTEM, 'retry_suffix': RETRY_SUFFIX,
               'protocol_sha256': file_sha(a.protocol), 'preflight_sha256': file_sha(a.preflight/'summary.json'),
               'prefix_amendment_sha256': file_sha(a.amendment), 'prefix_decision_sha256': file_sha(a.decision),
               'prefix_length_unicode_code_points': 2400,
               'weights_manifest_sha256': file_sha(a.weights_manifest), 'weights_sha256': expected_weights,
               'runner_sha256': file_sha(Path(__file__)), 'prompt_code_sha256': file_sha(Path(__file__).with_name('acord_prompts.py')),
               'mode_order': ['original', 'stable_reverse', 'selective_pointwise', 'replay', 'pointwise'],
               'modes': {}}
    dump(a.out / 'summary.json', summary)
    import torch
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import GuidedDecodingParams
    llm = LLM(model=str(a.model), tokenizer=str(a.model), trust_remote_code=False, dtype='bfloat16',
              max_model_len=4096, gpu_memory_utilization=.70, max_num_seqs=16, enforce_eager=True,
              enable_prefix_caching=True, disable_log_stats=True)
    tokenizer = llm.get_tokenizer()
    summary['runtime']['gpu'] = torch.cuda.get_device_name(0)
    assert_own_gpu()

    def wrap(system, text, tokenize=False):
        return tokenizer.apply_chat_template([{'role': 'system', 'content': system},
             {'role': 'user', 'content': text}], tokenize=tokenize, add_generation_prompt=True)

    keyed_params = SamplingParams(temperature=0, max_tokens=128,
                                  guided_decoding=GuidedDecodingParams(regex=keyed_regex(IDS)))
    point_params = SamplingParams(temperature=0, max_tokens=4,
                                  guided_decoding=GuidedDecodingParams(regex='[YN]'))
    negative = wrap('Follow the user instruction exactly.', 'Reply with only the word BANANA.')
    controls = {}
    for name, params in [('ordinary', SamplingParams(temperature=0, max_tokens=128)),
                         ('keyed', keyed_params), ('pointwise', point_params)]:
        assert_own_gpu()
        result = llm.generate([negative], params, use_tqdm=False)[0]
        o = result.outputs[0]
        controls[name] = {'text': o.text, 'finish_reason': o.finish_reason,
                          'input_tokens': len(result.prompt_token_ids), 'output_tokens': len(o.token_ids)}
    controls['checks'] = {'ordinary_invalid_keyed': parse_keyed(controls['ordinary']['text'], IDS) is None,
                          'ordinary_invalid_point': parse_one(controls['ordinary']['text']) is None,
                          'keyed_valid': parse_keyed(controls['keyed']['text'], IDS) is not None,
                          'point_valid': parse_one(controls['pointwise']['text']) is not None}
    dump(a.out / 'guidance_sanity.json', controls)
    if not all(controls['checks'].values()):
        raise RuntimeError('Negative control failed; no candidate inference was started')

    def execute(name, requests, params, parser, cap):
        worst = max((len(wrap(SYSTEM, r['prompt'] + RETRY_SUFFIX, tokenize=True)) + cap
                     for r in requests), default=0)
        if worst > 4096:
            raise RuntimeError(name + ' prompt and cap exceed context: ' + str(worst))
        memory_guard()
        start_time = time.perf_counter()
        assert_own_gpu()
        first = llm.generate([wrap(SYSTEM, x['prompt']) for x in requests], params, use_tqdm=False) if requests else []
        bad = [i for i, x in enumerate(first) if parser(x.outputs[0].text) is None]
        assert_own_gpu()
        retry = llm.generate([wrap(SYSTEM, requests[i]['prompt'] + RETRY_SUFFIX) for i in bad],
                             params, use_tqdm=False) if bad else []
        retries = dict(zip(bad, retry))
        records = []
        for i, (request, response) in enumerate(zip(requests, first)):
            output = response.outputs[0]
            retried = retries.get(i)
            last = retried.outputs[0] if retried else output
            records.append({'pool_id': request['pool_id'], 'start': request['start'],
                'size': request['size'], 'presented_ids': request.get('presented_ids'),
                'prompt_sha256': sha(request['prompt']), 'wrapped_sha256': sha(wrap(SYSTEM, request['prompt'])),
                'first': output.text, 'final': last.text, 'first_finish_reason': output.finish_reason,
                'retry_finish_reason': retried.outputs[0].finish_reason if retried else None,
                'first_input_tokens': len(response.prompt_token_ids), 'first_output_tokens': len(output.token_ids),
                'retry_input_tokens': len(retried.prompt_token_ids) if retried else 0,
                'retry_output_tokens': len(retried.outputs[0].token_ids) if retried else 0})
        write_rows(a.out / (name + '_responses.jsonl'), records)
        stats = {'initial_calls': len(requests), 'retry_calls': len(bad),
                 'valid_first': sum(parser(x['first']) is not None for x in records),
                 'valid_final': sum(parser(x['final']) is not None for x in records),
                 'first_finish_reasons': dict(collections.Counter(x['first_finish_reason'] for x in records)),
                 'logical_tokens': {kind: sum(x['first_' + kind + '_tokens'] + x['retry_' + kind + '_tokens']
                                              for x in records) for kind in ('input', 'output')},
                 'max_prompt_with_retry_plus_cap': worst,
                 'offline_generation_seconds': time.perf_counter() - start_time}
        summary['modes'][name] = stats
        dump(a.out / 'summary.json', summary)
        print(name, json.dumps(stats), flush=True)
        return records

    pipeline_start = time.perf_counter()
    requests = {'original': [], 'stable_reverse': [], 'replay': []}
    for row in rows:
        for start in range(0, 32, 8):
            for name, rev in [('original', False), ('stable_reverse', True), ('replay', False)]:
                prompt, presented = keyed_prompt(row, start, rev)
                requests[name].append({'pool_id': row['pool_id'], 'start': start, 'size': 8,
                                       'presented_ids': presented, 'prompt': prompt})
    original = execute('original', requests['original'], keyed_params, lambda x: parse_keyed(x, IDS), 128)
    reverse = execute('stable_reverse', requests['stable_reverse'], keyed_params, lambda x: parse_keyed(x, IDS), 128)
    selected_requests = []
    for q, row in enumerate(rows):
        for chunk in range(4):
            x = parse_keyed(original[q * 4 + chunk]['final'], IDS)
            y = parse_keyed(reverse[q * 4 + chunk]['final'], IDS)
            if x is None or y is None:
                continue
            for local in IDS:
                index = chunk * 8 + local - 1
                if x[local] != y[local]:
                    selected_requests.append({'pool_id': row['pool_id'], 'start': index, 'size': 1,
                                              'prompt': point_prompt(row, index)})
    selected = execute('selective_pointwise', selected_requests, point_params, parse_one, 4)
    selected_by_key = {(r['pool_id'], r['start']): parse_one(r['final']) for r in selected}
    if len(selected_by_key) != len(selected):
        raise ValueError('duplicate selective request')
    outputs = []
    for q, row in enumerate(rows):
        pair = []
        for records in (original, reverse):
            vector = [None] * 32
            for chunk in range(4):
                labels = parse_keyed(records[q * 4 + chunk]['final'], IDS)
                if labels is not None:
                    for local, value in labels.items():
                        vector[chunk * 8 + local - 1] = value
            pair.append(vector if None not in vector else None)
        fallback = [selected_by_key.get((row['pool_id'], i)) for i in range(32)]
        outputs.append({'pool_id': row['pool_id'], 'pred': combine(pair[0], pair[1], fallback)})
    write_rows(a.out / 'selective_outputs.jsonl', outputs)
    plans = [summary['modes'][x] for x in ('original', 'stable_reverse', 'selective_pointwise')]
    summary['selective_pipeline'] = {'complete_pools': sum(x['pred'] is not None for x in outputs),
        'initial_calls': sum(x['initial_calls'] for x in plans), 'retry_calls': sum(x['retry_calls'] for x in plans),
        'logical_tokens': {kind: sum(x['logical_tokens'][kind] for x in plans) for kind in ('input', 'output')},
        'offline_batch_wall_seconds': time.perf_counter() - pipeline_start}
    dump(a.out / 'summary.json', summary)
    print('selective_pipeline', json.dumps(summary['selective_pipeline']), flush=True)
    execute('replay', requests['replay'], keyed_params, lambda x: parse_keyed(x, IDS), 128)
    full_requests = [{'pool_id': row['pool_id'], 'start': i, 'size': 1, 'prompt': point_prompt(row, i)}
                     for row in rows for i in range(32)]
    execute('pointwise', full_requests, point_params, parse_one, 4)
    summary['completed'] = True
    dump(a.out / 'summary.json', summary)
    print('Completed ACORD transfer execution.', flush=True)


if __name__ == '__main__':
    main()
