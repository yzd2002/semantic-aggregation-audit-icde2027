#!/usr/bin/env python3
"""Run the secondary SmolLM2 actual selective plan on the existing WANDS queries."""
import argparse
import collections
import hashlib
import json
import os
import time
from pathlib import Path

import run_experiment as base
import run_full_pools as harness
import run_wands
from run_keyed_order_audit import keyed_prompt, keyed_regex, parse_keyed
from select_selective_execution import N, PROTOCOL as SELECTION_PROTOCOL, lines
PROTOCOL = 'PROTOCOL_selective_model_sensitivity.md'

SUFFIX = '\nYour previous answer had the wrong format; follow the output instructions exactly.'


def digest(text):
    return hashlib.sha256(text.encode('utf8')).hexdigest()


def point_prompt(row, item, spec):
    text = 'Query: ' + row['query'] + '\n'
    if spec is not None:
        text += 'Query specification: ' + json.dumps(spec, ensure_ascii=False) + '\n'
    return text + 'Product: ' + item['text'] + '\nAnswer ONLY Y or N.'


def parse_one(text):
    return text if text in ('Y', 'N') else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--selection', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    if os.environ.get('VLLM_USE_V1') != '1' or os.path.basename(a.model) != 'SmolLM2-1.7B-Instruct':
        raise ValueError('This protocol requires the recorded SmolLM2 model and vLLM V1.')
    rows = lines(a.selection / 'inputs.jsonl')
    selection = json.loads((a.selection / 'selection.json').read_text())
    if (selection['protocol'] != SELECTION_PROTOCOL or len(rows) != N
            or any(len(row['items']) != 32 for row in rows)):
        raise ValueError('Unexpected selection.')
    selection_sha = hashlib.sha256((a.selection / 'inputs.jsonl').read_bytes()).hexdigest()
    a.out.mkdir(parents=True, exist_ok=False)
    import torch
    import transformers
    import vllm
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import GuidedDecodingParams

    llm = LLM(model=a.model, tokenizer=a.model, trust_remote_code=False, dtype='bfloat16',
              max_model_len=4096, gpu_memory_utilization=.70, max_num_seqs=16, enforce_eager=True,
              enable_prefix_caching=True, disable_log_stats=True)
    tok = llm.get_tokenizer()
    def wrap(system, text, tokenize=False):
        return tok.apply_chat_template([{'role': 'system', 'content': system}, {'role': 'user', 'content': text}],
                                       tokenize=tokenize, add_generation_prompt=True)
    keyed_params = SamplingParams(temperature=0, max_tokens=128,
        guided_decoding=GuidedDecodingParams(regex=keyed_regex(list(range(1, 9)))))
    point_params = SamplingParams(temperature=0, max_tokens=4,
        guided_decoding=GuidedDecodingParams(regex='[YN]'))
    control = wrap('Follow the user instruction exactly.', 'Reply with only the word BANANA.')
    ordinary = llm.generate([control], SamplingParams(temperature=0, max_tokens=128), use_tqdm=False)[0].outputs[0].text.strip()
    json_control = llm.generate([control], keyed_params, use_tqdm=False)[0].outputs[0].text.strip()
    point_control = llm.generate([control], point_params, use_tqdm=False)[0].outputs[0].text.strip()
    ids = list(range(1, 9))
    sanity = {'unconstrained': ordinary, 'json_guided': json_control, 'point_guided': point_control,
              'unconstrained_valid_json': parse_keyed(ordinary, ids) is not None,
              'unconstrained_valid_point': parse_one(ordinary) is not None,
              'guided_json_valid': parse_keyed(json_control, ids) is not None,
              'guided_point_valid': parse_one(point_control) is not None}
    (a.out / 'guidance_sanity.json').write_text(json.dumps(sanity, indent=2) + '\n')
    if (sanity['unconstrained_valid_json'] or sanity['unconstrained_valid_point']
            or not sanity['guided_json_valid'] or not sanity['guided_point_valid']):
        raise RuntimeError('Constraint negative control failed; no candidate evaluation was run.')

    system = run_wands.SYSTEM
    pipeline_start = time.perf_counter()
    spec_prompts = [base.p_spec(row) for row in rows]
    max_spec = max(len(wrap(system, text + '\nReturn valid JSON only.', tokenize=True)) + 160 for text in spec_prompts)
    if max_spec > 4096:
        raise RuntimeError('Specification prompt exceeds context.')
    t = time.perf_counter()
    first_results = llm.generate([wrap(system, text) for text in spec_prompts],
                                 SamplingParams(temperature=0, max_tokens=160), use_tqdm=False)
    first = [r.outputs[0].text.strip() for r in first_results]
    final = first[:]
    bad = [i for i, x in enumerate(first) if base.parse_spec(x) is None]
    retry = llm.generate([wrap(system, spec_prompts[i] + '\nReturn valid JSON only.') for i in bad],
                         SamplingParams(temperature=0, max_tokens=160), use_tqdm=False) if bad else []
    for i, result in zip(bad, retry):
        final[i] = result.outputs[0].text.strip()
    specs = [base.parse_spec(x) for x in final]
    spec_records = [{'query_id': row['query_id'], 'prompt_sha256': digest(prompt), 'first': x, 'final': y,
                     'parsed': spec, 'fallback': spec is None} for row, prompt, x, y, spec in zip(rows, spec_prompts, first, final, specs)]
    with (a.out / 'specifications.jsonl').open('w', encoding='utf8') as f:
        for row in spec_records:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    spec_tokens = {'input': sum(len(r.prompt_token_ids) for r in first_results + retry),
                   'output': sum(len(r.outputs[0].token_ids) for r in first_results + retry)}
    summary = {'protocol': PROTOCOL, 'model': os.path.basename(a.model), 'selection_inputs_sha256': selection_sha,
               'runtime': {'torch': torch.__version__, 'transformers': transformers.__version__, 'vllm': vllm.__version__,
                           'gpu': torch.cuda.get_device_name(0), 'dtype': 'bfloat16', 'max_model_len': 4096,
                           'engine': 'V1 (VLLM_USE_V1=1)', 'guided_backend': 'vLLM automatic selection'},
               'specifications': {'initial_calls': len(rows), 'retry_calls': len(bad), 'fallback_pools': sum(x is None for x in specs),
                                  'logical_tokens': spec_tokens, 'max_prompt_with_retry_plus_cap': max_spec,
                                  'offline_generation_seconds': time.perf_counter() - t},
               'mode_order': ['original', 'stable_reverse', 'selective_pointwise', 'replay', 'pointwise'], 'modes': {}}
    (a.out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')

    def execute(mode, requests, params, parse, cap):
        max_len = max((len(wrap(system, r['text'] + SUFFIX, tokenize=True)) + cap for r in requests), default=0)
        if max_len > 4096:
            raise RuntimeError('Candidate prompt exceeds context.')
        t = time.perf_counter()
        first_out = llm.generate([wrap(system, r['text']) for r in requests], params, use_tqdm=False) if requests else []
        first = [r.outputs[0].text.strip() for r in first_out]
        final = first[:]
        failures = [i for i, text in enumerate(first) if parse(text) is None]
        retry_out = llm.generate([wrap(system, requests[i]['text'] + SUFFIX) for i in failures], params, use_tqdm=False) if failures else []
        retries = {i: r for i, r in zip(failures, retry_out)}
        for i, result in retries.items():
            final[i] = result.outputs[0].text.strip()
        records = []
        for i, request in enumerate(requests):
            original = first_out[i]; retried = retries.get(i)
            records.append({'query_id': request['query_id'], 'start': request['start'], 'size': request['size'],
                            'presented_ids': request.get('presented_ids'), 'prompt_sha256': digest(request['text']),
                            'first': first[i], 'final': final[i], 'first_finish_reason': original.outputs[0].finish_reason,
                            'retry_finish_reason': retried.outputs[0].finish_reason if retried else None,
                            'first_input_tokens': len(original.prompt_token_ids), 'first_output_tokens': len(original.outputs[0].token_ids),
                            'retry_input_tokens': len(retried.prompt_token_ids) if retried else 0,
                            'retry_output_tokens': len(retried.outputs[0].token_ids) if retried else 0})
        with (a.out / (mode + '_responses.jsonl')).open('w', encoding='utf8') as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + '\n')
        stats = {'initial_calls': len(first), 'retry_calls': len(failures),
                 'valid_first': sum(parse(x) is not None for x in first),
                 'valid_final': sum(parse(x) is not None for x in final),
                 'finish_reasons': dict(collections.Counter(r['first_finish_reason'] for r in records)),
                 'logical_tokens': {'input': sum(r['first_input_tokens'] + r['retry_input_tokens'] for r in records),
                                    'output': sum(r['first_output_tokens'] + r['retry_output_tokens'] for r in records)},
                 'max_prompt_with_retry_plus_cap': max_len, 'offline_generation_seconds': time.perf_counter() - t}
        summary['modes'][mode] = stats
        (a.out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        print(mode, json.dumps(stats), flush=True)
        return records

    requests = {mode: [] for mode in ('original', 'stable_reverse', 'replay')}
    for row, spec in zip(rows, specs):
        for part in harness.chunks(row):
            if [x['id'] for x in part['items']] != ids:
                raise ValueError('Unexpected local product IDs.')
            reverse = {**part, 'items': [dict(x) for x in reversed(part['items'])]}
            for mode, current in [('original', part), ('stable_reverse', reverse), ('replay', part)]:
                requests[mode].append({'query_id': row['query_id'], 'start': part['start'], 'size': 8,
                                       'presented_ids': [x['id'] for x in current['items']], 'text': keyed_prompt(current, spec)})
    original = execute('original', requests['original'], keyed_params, lambda s: parse_keyed(s, ids), 128)
    reverse = execute('stable_reverse', requests['stable_reverse'], keyed_params, lambda s: parse_keyed(s, ids), 128)
    selective_requests = []
    for q, (row, spec) in enumerate(zip(rows, specs)):
        for c in range(4):
            x = parse_keyed(original[q*4+c]['final'], ids)
            y = parse_keyed(reverse[q*4+c]['final'], ids)
            if x is None or y is None:
                continue
            for local in ids:
                index = c*8 + local - 1
                if x[local] != y[local]:
                    selective_requests.append({'query_id': row['query_id'], 'start': index, 'size': 1,
                                               'text': point_prompt(row, row['items'][index], spec)})
    selected = execute('selective_pointwise', selective_requests, point_params, parse_one, 4)
    selected_by_key = {(r['query_id'], r['start']): parse_one(r['final']) for r in selected}
    if len(selected_by_key) != len(selected):
        raise ValueError('Duplicate selective request.')
    from order_fallback import combine
    out = []
    for q, row in enumerate(rows):
        pair = []
        for records in (original, reverse):
            vector = [None]*32
            for c in range(4):
                labels = parse_keyed(records[q*4+c]['final'], ids)
                if labels is not None:
                    for local, label in labels.items():
                        vector[c*8+local-1] = label
            pair.append(vector if None not in vector else None)
        h = [selected_by_key.get((row['query_id'], i)) for i in range(32)]
        out.append({'query_id': row['query_id'], 'pred': combine(pair[0], pair[1], h)})
    with (a.out / 'selective_outputs.jsonl').open('w', encoding='utf8') as f:
        for rec in out:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')
    plans = [summary['modes'][m] for m in ('original', 'stable_reverse', 'selective_pointwise')]
    sp = summary['specifications']
    summary['selective_pipeline'] = {'complete_pools': sum(r['pred'] is not None for r in out),
        'initial_calls': sp['initial_calls'] + sum(x['initial_calls'] for x in plans),
        'retry_calls': sp['retry_calls'] + sum(x['retry_calls'] for x in plans),
        'logical_tokens': {key: sp['logical_tokens'][key] + sum(x['logical_tokens'][key] for x in plans)
                           for key in ('input', 'output')},
        'offline_batch_wall_seconds': time.perf_counter() - pipeline_start}
    (a.out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print('selective_pipeline', json.dumps(summary['selective_pipeline']), flush=True)
    execute('replay', requests['replay'], keyed_params, lambda s: parse_keyed(s, ids), 128)
    all_points = [{'query_id': row['query_id'], 'start': i, 'size': 1, 'text': point_prompt(row, item, spec)}
                  for row, spec in zip(rows, specs) for i, item in enumerate(row['items'])]
    execute('pointwise', all_points, point_params, parse_one, 4)
    print('Completed actual selective validation.', flush=True)


if __name__ == '__main__':
    main()
