#!/usr/bin/env python3
"""Run the frozen variable-size ESCI selective-fallback evaluation."""
import argparse
import collections
import hashlib
import importlib.metadata as metadata
import json
import os
import time
from pathlib import Path

import run_experiment as base
import run_full_pools as harness
from order_fallback import combine
from run_keyed_order_audit import keyed_prompt, keyed_regex, parse_keyed
from run_offline_wands_replay import assert_own_gpu, gpu_processes
from spec_guidance_schema import SPEC_SCHEMA
from select_esci_selective_holdout import PROTOCOL

RETRY = '\nYour previous answer had the wrong format; follow the output instructions exactly.'
SPEC_RETRY = '\nReturn valid JSON only.'
MODES = ('original', 'stable_reverse', 'selective_pointwise', 'replay', 'pointwise')


def sha_text(s):
    return hashlib.sha256(s.encode('utf8')).hexdigest()


def sha_file(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def lines(path):
    with path.open(encoding='utf8') as f:
        return [json.loads(x) for x in f]


def point_prompt(row, item, spec):
    text = 'Query: ' + row['query'] + '\n'
    if spec is not None:
        text += 'Query specification: ' + json.dumps(spec, ensure_ascii=False) + '\n'
    return text + 'Product: ' + item['text'] + '\nAnswer ONLY Y or N.'


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', type=Path, required=True)
    ap.add_argument('--selection', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    if os.environ.get('VLLM_USE_V1') != '1' or a.model.name != 'Qwen2.5-7B-Instruct':
        raise ValueError('Use V1 and the recorded post-hoc Qwen2.5 model.')
    expected = {'vllm': '0.10.2', 'torch': '2.8.0+cu128', 'transformers': '4.57.1',
                'numpy': '2.2.6', 'xgrammar': '0.1.23', 'outlines-core': '0.2.11', 'setuptools': '79.0.1'}
    versions = {name: metadata.version(name) for name in expected}
    if versions != expected or a.out.exists() or gpu_processes():
        raise RuntimeError('Unexpected environment, output exists or GPU is already in use.')
    rows = lines(a.selection / 'inputs.jsonl')
    selection = json.loads((a.selection / 'selection.json').read_text(encoding='utf8'))
    if (selection['protocol'] != PROTOCOL or len(rows) != 250 or
            selection['inputs_sha256'] != sha_file(a.selection / 'inputs.jsonl') or
            any(not r['items'] or [x['id'] for x in r['items']] != list(range(1, len(r['items'])+1)) for r in rows)):
        raise ValueError('Selection or candidate positions differ from protocol.')
    a.out.mkdir(parents=True, exist_ok=False)

    import torch
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import GuidedDecodingParams
    llm = LLM(model=str(a.model), tokenizer=str(a.model), trust_remote_code=False, dtype='bfloat16',
              max_model_len=4096, gpu_memory_utilization=.70, max_num_seqs=16, enforce_eager=True,
              enable_prefix_caching=True, disable_log_stats=True)
    tok = llm.get_tokenizer()

    def wrap(text, tokenized=False, system=base.SYSTEM):
        return tok.apply_chat_template([{'role': 'system', 'content': system}, {'role': 'user', 'content': text}],
                                       tokenize=tokenized, add_generation_prompt=True)

    keyed_params = {n: SamplingParams(temperature=0, max_tokens=128,
        guided_decoding=GuidedDecodingParams(regex=keyed_regex(list(range(1, n+1))))) for n in range(1, 9)}
    point_params = SamplingParams(temperature=0, max_tokens=4, guided_decoding=GuidedDecodingParams(regex='[YN]'))
    spec_params = SamplingParams(temperature=0, max_tokens=160, guided_decoding=GuidedDecodingParams(json=SPEC_SCHEMA))
    assert_own_gpu()
    control = wrap('Reply with only the word BANANA.', system='Follow the user instruction exactly.')
    unconstrained = llm.generate([control], SamplingParams(temperature=0, max_tokens=128), use_tqdm=False)[0].outputs[0].text.strip()
    guided_json = llm.generate([control], keyed_params[8], use_tqdm=False)[0].outputs[0].text.strip()
    guided_one = llm.generate([control], point_params, use_tqdm=False)[0].outputs[0].text.strip()
    guided_spec = llm.generate([control], spec_params, use_tqdm=False)[0].outputs[0].text.strip()
    sanity = {'unconstrained': unconstrained, 'guided_keyed': guided_json, 'guided_point': guided_one,
              'guided_spec': guided_spec, 'unconstrained_valid_spec': base.parse_spec(unconstrained) is not None,
              'guided_spec_valid': base.parse_spec(guided_spec) is not None,
              'unconstrained_valid_keyed': parse_keyed(unconstrained, list(range(1, 9))) is not None,
              'unconstrained_valid_point': unconstrained in ('Y', 'N'),
              'guided_keyed_valid': parse_keyed(guided_json, list(range(1, 9))) is not None,
              'guided_point_valid': guided_one in ('Y', 'N')}
    (a.out / 'guidance_sanity.json').write_text(json.dumps(sanity, indent=2) + '\n', encoding='utf8')
    if (sanity['unconstrained_valid_keyed'] or sanity['unconstrained_valid_point'] or
            not sanity['guided_keyed_valid'] or not sanity['guided_point_valid'] or
            sanity['unconstrained_valid_spec'] or not sanity['guided_spec_valid']):
        raise RuntimeError('Grammar negative control failed; no candidate evaluation was run.')
    summary = {'protocol': PROTOCOL, 'posthoc_protocol': 'PROTOCOL_7b_specguided_posthoc.md', 'model': a.model.name, 'selection_inputs_sha256': sha_file(a.selection / 'inputs.jsonl'),
               'versions': versions, 'gpu': torch.cuda.get_device_name(0), 'dtype': 'bfloat16',
               'max_model_len': 4096, 'batch_size': 64, 'mode_order': list(MODES), 'modes': {}}
    (a.out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf8')

    def parse(request, text):
        if request['kind'] == 'spec':
            return base.parse_spec(text)
        if request['kind'] == 'keyed':
            return parse_keyed(text, list(range(1, request['size'] + 1)))
        if request['kind'] == 'point':
            return text if text in ('Y', 'N') else None
        raise ValueError('Unknown request kind.')

    def params(request):
        return spec_params if request['kind'] == 'spec' else keyed_params[request['size']] if request['kind'] == 'keyed' else point_params

    def execute(mode, requests):
        start_time = time.perf_counter()
        records = []
        path = a.out / (mode + '_responses.jsonl')
        with path.open('x', encoding='utf8') as saved:
            for start in range(0, len(requests), 64):
                batch = requests[start:start+64]
                wrapped, limits = [], []
                for req in batch:
                    suffix = SPEC_RETRY if req['kind'] == 'spec' else RETRY
                    cap = 160 if req['kind'] == 'spec' else 128 if req['kind'] == 'keyed' else 4
                    length = len(wrap(req['text'] + suffix, True)) + cap
                    if length > 4096:
                        raise RuntimeError('Worst-case request exceeds context.')
                    limits.append(length)
                    wrapped.append(wrap(req['text']))
                assert_own_gpu()
                first_out = llm.generate(wrapped, [params(r) for r in batch], use_tqdm=False)
                first = [r.outputs[0].text.strip() for r in first_out]
                bad = [i for i, (req, text) in enumerate(zip(batch, first)) if parse(req, text) is None]
                retry_texts = [batch[i]['text'] + (SPEC_RETRY if batch[i]['kind'] == 'spec' else RETRY) for i in bad]
                assert_own_gpu()
                retry_out = llm.generate([wrap(x) for x in retry_texts], [params(batch[i]) for i in bad],
                                         use_tqdm=False) if bad else []
                retries = dict(zip(bad, retry_out))
                for i, req in enumerate(batch):
                    original, retry = first_out[i], retries.get(i)
                    final = retry.outputs[0].text.strip() if retry else first[i]
                    rec = {'query_id': req['query_id'], 'start': req['start'], 'size': req['size'],
                           'presented_ids': req.get('presented_ids'), 'prompt_sha256': sha_text(req['text']),
                           'wrapped_sha256': sha_text(wrapped[i]), 'first': first[i], 'final': final,
                           'first_finish_reason': original.outputs[0].finish_reason,
                           'retry_finish_reason': retry.outputs[0].finish_reason if retry else None,
                           'first_input_tokens': len(original.prompt_token_ids), 'first_output_tokens': len(original.outputs[0].token_ids),
                           'retry_input_tokens': len(retry.prompt_token_ids) if retry else 0,
                           'retry_output_tokens': len(retry.outputs[0].token_ids) if retry else 0,
                           'retry_prompt_plus_cap': limits[i]}
                    records.append(rec)
                    saved.write(json.dumps(rec, ensure_ascii=False) + '\n')
                saved.flush()
                print(mode, min(start + len(batch), len(requests)), '/', len(requests), flush=True)
        stats = {'initial_calls': len(records), 'retry_calls': sum(x['retry_finish_reason'] is not None for x in records),
                 'valid_first': sum(parse(r, x['first']) is not None for r, x in zip(requests, records)),
                 'valid_final': sum(parse(r, x['final']) is not None for r, x in zip(requests, records)),
                 'finish_reasons': dict(collections.Counter(x['first_finish_reason'] for x in records)),
                 'logical_tokens': {'input': sum(x['first_input_tokens'] + x['retry_input_tokens'] for x in records),
                                    'output': sum(x['first_output_tokens'] + x['retry_output_tokens'] for x in records)},
                 'max_retry_prompt_plus_cap': max((x['retry_prompt_plus_cap'] for x in records), default=0),
                 'offline_batch_seconds': time.perf_counter() - start_time}
        summary['modes'][mode] = stats
        (a.out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf8')
        return records

    pipeline_start = time.perf_counter()
    spec_requests = [{'query_id': r['query_id'], 'start': 0, 'size': len(r['items']),
                      'kind': 'spec', 'text': base.p_spec(r)} for r in rows]
    spec_events = execute('specifications', spec_requests)
    specs = [base.parse_spec(x['final']) for x in spec_events]
    requests = {mode: [] for mode in ('original', 'stable_reverse', 'replay')}
    for row, spec in zip(rows, specs):
        for part in harness.chunks(row):
            rev = {**part, 'items': list(reversed(part['items']))}
            for mode, current in (('original', part), ('stable_reverse', rev), ('replay', part)):
                requests[mode].append({'query_id': row['query_id'], 'start': part['start'], 'size': len(part['items']),
                    'presented_ids': [x['id'] for x in current['items']], 'kind': 'keyed', 'text': keyed_prompt(current, spec)})
    original = execute('original', requests['original'])
    reverse = execute('stable_reverse', requests['stable_reverse'])
    keyed_pairs = {}
    for req, a_event, b_event in zip(requests['original'], original, reverse):
        ids = list(range(1, req['size'] + 1))
        keyed_pairs[(req['query_id'], req['start'])] = (parse_keyed(a_event['final'], ids), parse_keyed(b_event['final'], ids))
    selected_requests = []
    for row, spec in zip(rows, specs):
        for part in harness.chunks(row):
            first, second = keyed_pairs[(row['query_id'], part['start'])]
            if first is None or second is None:
                continue
            for item in part['items']:
                if first[item['id']] != second[item['id']]:
                    index = part['start'] + item['id'] - 1
                    selected_requests.append({'query_id': row['query_id'], 'start': index, 'size': 1,
                        'kind': 'point', 'text': point_prompt(row, row['items'][index], spec)})
    selected = execute('selective_pointwise', selected_requests)
    selected_by_key = {(r['query_id'], r['start']): e['final'] if e['final'] in ('Y', 'N') else None
                       for r, e in zip(selected_requests, selected)}
    if len(selected_by_key) != len(selected):
        raise ValueError('Duplicate selective item request.')
    outputs = []
    for row in rows:
        vectors = [[None] * len(row['items']) for _ in range(2)]
        for part in harness.chunks(row):
            paired = keyed_pairs[(row['query_id'], part['start'])]
            for which, labels in enumerate(paired):
                if labels is not None:
                    for local, label in labels.items():
                        vectors[which][part['start'] + local - 1] = label
        full = [v if None not in v else None for v in vectors]
        fallback = [selected_by_key.get((row['query_id'], i)) for i in range(len(row['items']))]
        outputs.append({'query_id': row['query_id'], 'pred': combine(full[0], full[1], fallback)})
    with (a.out / 'selective_outputs.jsonl').open('x', encoding='utf8') as f:
        for row in outputs:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    included = [summary['modes'][m] for m in ('specifications', 'original', 'stable_reverse', 'selective_pointwise')]
    summary['selective_pipeline'] = {'complete_pools': sum(r['pred'] is not None for r in outputs),
        'initial_calls': sum(x['initial_calls'] for x in included), 'retry_calls': sum(x['retry_calls'] for x in included),
        'logical_tokens': {k: sum(x['logical_tokens'][k] for x in included) for k in ('input', 'output')},
        'offline_batch_wall_seconds': time.perf_counter() - pipeline_start}
    (a.out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf8')
    execute('replay', requests['replay'])
    points = [{'query_id': row['query_id'], 'start': i, 'size': 1, 'kind': 'point',
               'text': point_prompt(row, item, spec)} for row, spec in zip(rows, specs) for i, item in enumerate(row['items'])]
    execute('pointwise', points)
    summary['completed'] = True
    (a.out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf8')
    print('Completed all fixed stages; selective complete pools:', summary['selective_pipeline']['complete_pools'], flush=True)


if __name__ == '__main__':
    main()
