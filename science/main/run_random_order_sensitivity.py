#!/usr/bin/env python3
"""Run fixed interleaved random-order controls on saved WANDS holdout inputs."""
import argparse
import hashlib
import json
import os
from importlib import metadata
from pathlib import Path

from run_compact_prompt_ablation import prompt_without_example
from run_full_pools import chunks
from run_offline_wands_replay import assert_own_gpu, digest, gpu_processes, lines
from run_validation import parse_compact
import run_wands

MODES = ('baseline_a', 'perm_1', 'baseline_b', 'perm_2', 'perm_3', 'baseline_c')
SUFFIX = '\nYour previous answer had the wrong format; follow the output instructions exactly.'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--selection', type=Path, required=True)
    p.add_argument('--orders', type=Path, required=True)
    p.add_argument('--prior', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if os.environ.get('VLLM_USE_V1') != '1' or a.model.name not in ('Qwen3-4B-Instruct-2507', 'SmolLM2-1.7B-Instruct'):
        raise ValueError('Use V1 and one of the two recorded models.')
    expected = {'vllm': '0.10.2', 'torch': '2.8.0+cu128', 'transformers': '4.57.1',
                'numpy': '2.2.6', 'xgrammar': '0.1.23', 'outlines-core': '0.2.11',
                'setuptools': '79.0.1'}
    versions = {name: metadata.version(name) for name in expected}
    if versions != expected or a.out.exists() or gpu_processes():
        raise RuntimeError('Unexpected versions, existing output, or GPU in use.')
    rows = lines(a.selection / 'inputs.jsonl')
    mappings = lines(a.orders / 'mappings.jsonl')
    specs = lines(a.prior / 'specifications.jsonl')
    old = lines(a.prior / 'original_outputs.jsonl')
    if not (len(rows) == len(mappings) == len(specs) == len(old) == 160):
        raise ValueError('Unexpected input length.')
    orders_meta = json.loads((a.orders / 'selection.json').read_text())
    if hashlib.sha256((a.orders / 'mappings.jsonl').read_bytes()).hexdigest() != orders_meta['mappings_sha256']:
        raise ValueError('Order mapping hash mismatch.')
    inputs = {m: [] for m in MODES}
    for row, mapping, spec, prior in zip(rows, mappings, specs, old):
        qid = row['query_id']
        if qid != mapping['query_id'] or qid != spec['query_id'] or qid != prior['query_id'] or len(row['items']) != 32:
            raise ValueError('Query alignment failure.')
        for index, (part, order) in enumerate(zip(chunks(row), mapping['chunks'])):
            start = part['start']
            if order['start'] != start or len(part['items']) != 8:
                raise ValueError('Unexpected chunk.')
            original = prompt_without_example(part, spec['parsed'])
            previous = prior['raw'][index]
            if (previous['start'] != start or previous['size'] != 8 or
                previous['original_indices_in_prompt_order'] != list(range(start, start + 8)) or
                previous['condition_prompt_sha256'] != digest(original)):
                raise ValueError('Saved baseline prompt or mapping differs.')
            for mode in MODES:
                positions = order['orders'][mode] if mode.startswith('perm_') else list(range(8))
                if sorted(positions) != list(range(8)):
                    raise ValueError('Invalid permutation.')
                ordered = {**part, 'items': [part['items'][i] for i in positions]}
                text = prompt_without_example(ordered, spec['parsed'])
                inputs[mode].append({'query_id': qid, 'start': start, 'size': 8,
                    'indices': [start + i for i in positions], 'text': text,
                    'original_prompt_sha256': digest(original)})
    if any(len(inputs[m]) != 640 for m in MODES):
        raise ValueError('Unexpected number of chunks.')
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(a.model), local_files_only=True, trust_remote_code=False)
    def wrap(tokenizer, text, tokenize=False):
        return tokenizer.apply_chat_template([{'role': 'system', 'content': run_wands.SYSTEM},
            {'role': 'user', 'content': text}], tokenize=tokenize, add_generation_prompt=True)
    max_context = max(len(wrap(tok, x['text'] + SUFFIX, True)) + 32
                      for mode in ('baseline_a', 'perm_1', 'perm_2', 'perm_3') for x in inputs[mode])
    if max_context > 4096:
        raise RuntimeError('A retry prompt exceeds the context limit.')
    a.out.mkdir(parents=True, exist_ok=False)
    (a.out / 'preflight.json').write_text(json.dumps({'protocol': 'PROTOCOL_random_order_sensitivity.md',
        'model': a.model.name, 'versions': versions, 'queries': len(rows), 'chunks_per_mode': 640,
        'mode_order': MODES, 'batch_size': 64, 'max_retry_prompt_plus_cap': max_context,
        'selection_sha256': hashlib.sha256((a.selection / 'inputs.jsonl').read_bytes()).hexdigest(),
        'mappings_sha256': orders_meta['mappings_sha256']}, indent=2) + '\n')
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import GuidedDecodingParams
    llm = LLM(model=str(a.model), tokenizer=str(a.model), trust_remote_code=False, dtype='bfloat16',
              max_model_len=4096, gpu_memory_utilization=.70, max_num_seqs=16, enforce_eager=True,
              enable_prefix_caching=True, disable_log_stats=True)
    actual_tok = llm.get_tokenizer()
    assert_own_gpu()
    for mode in ('baseline_a', 'perm_1', 'perm_2', 'perm_3'):
        for x in inputs[mode]:
            if wrap(tok, x['text']) != wrap(actual_tok, x['text']):
                raise RuntimeError('Loaded tokenizer changed a preflight prompt.')
    def guided():
        return SamplingParams(temperature=0, max_tokens=32,
                              guided_decoding=GuidedDecodingParams(regex='[YN]{8}'))
    control = actual_tok.apply_chat_template([{'role': 'system', 'content': 'Follow the user instruction exactly.'},
        {'role': 'user', 'content': 'Reply with only the word BANANA.'}], tokenize=False, add_generation_prompt=True)
    assert_own_gpu()
    plain = llm.generate([control], SamplingParams(temperature=0, max_tokens=32), use_tqdm=False)[0].outputs[0].text.strip()
    assert_own_gpu()
    constrained = llm.generate([control], guided(), use_tqdm=False)[0].outputs[0].text.strip()
    controls = {'unconstrained': plain, 'guided': constrained,
                'unconstrained_invalid': parse_compact(plain, 8) is None,
                'guided_valid': parse_compact(constrained, 8) is not None}
    (a.out / 'controls.json').write_text(json.dumps(controls, indent=2) + '\n')
    if not controls['unconstrained_invalid'] or not controls['guided_valid']:
        raise RuntimeError('Guided-decoding negative control failed.')
    for mode in MODES:
        group = inputs[mode]
        with (a.out / (mode + '_records.jsonl')).open('x', encoding='utf8') as f:
            for start in range(0, len(group), 64):
                batch = group[start:start + 64]
                assert_own_gpu()
                generated = llm.generate([wrap(actual_tok, x['text']) for x in batch],
                                         [guided() for _ in batch], use_tqdm=False)
                if len(generated) != len(batch):
                    raise RuntimeError('Generation returned an unexpected number of responses.')
                first = [r.outputs[0].text.strip() for r in generated]
                invalid = [i for i, value in enumerate(first) if parse_compact(value, 8) is None]
                retry_results = []
                if invalid:
                    assert_own_gpu()
                    retry_results = llm.generate([wrap(actual_tok, batch[i]['text'] + SUFFIX) for i in invalid],
                                                 [guided() for _ in invalid], use_tqdm=False)
                if len(retry_results) != len(invalid):
                    raise RuntimeError('Retry returned an unexpected number of responses.')
                retry_by_index = dict(zip(invalid, retry_results))
                for i, (x, result) in enumerate(zip(batch, generated)):
                    retry = retry_by_index.get(i)
                    final = retry.outputs[0].text.strip() if retry else first[i]
                    record = {k: x[k] for k in ('query_id', 'start', 'size', 'indices', 'original_prompt_sha256')} | {
                        'prompt_sha256': digest(x['text']), 'wrapped_sha256': digest(wrap(actual_tok, x['text'])),
                        'first': first[i], 'final': final, 'first_finish_reason': result.outputs[0].finish_reason,
                        'retry_finish_reason': retry.outputs[0].finish_reason if retry else None,
                        'first_input_tokens': len(result.prompt_token_ids), 'first_output_tokens': len(result.outputs[0].token_ids),
                        'retry_input_tokens': len(retry.prompt_token_ids) if retry else 0,
                        'retry_output_tokens': len(retry.outputs[0].token_ids) if retry else 0}
                    f.write(json.dumps(record, ensure_ascii=False) + '\n')
                f.flush()
                print(mode, 'completed', min(start + 64, len(group)), '/', len(group), flush=True)
    print('Completed all fixed random-order conditions.', flush=True)


if __name__ == '__main__':
    main()
