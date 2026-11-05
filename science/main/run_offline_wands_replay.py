#!/usr/bin/env python3
"""Replay a fixed prefix of observed WANDS prompts in an isolated environment."""
import argparse
import collections
import hashlib
import json
import os
import subprocess
from importlib import metadata
from pathlib import Path

import psutil
from run_compact_prompt_ablation import prompt_without_example
from run_full_pools import chunks
from run_validation import parse_compact
import run_wands

MODES = ('original', 'replay', 'stable_reverse')
SUFFIX = '\nYour previous answer had the wrong format; follow the output instructions exactly.'


def lines(path):
    with path.open(encoding='utf8') as f:
        return [json.loads(line) for line in f]


def digest(text):
    return hashlib.sha256(text.encode('utf8')).hexdigest()


def gpu_processes():
    text = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                                    '--format=csv,noheader'], text=True)
    return {int(x.strip()) for x in text.splitlines() if x.strip().isdigit()}


def assert_own_gpu():
    proc = psutil.Process(os.getpid())
    own = {proc.pid} | {p.pid for p in proc.children(recursive=True)}
    if not gpu_processes() <= own:
        raise RuntimeError('Another GPU process appeared; stopping this replay.')


def compare(current, baseline):
    before = {(r['query_id'], r['start']): r for r in baseline}
    assert len(before) == len(baseline) == len(current)
    changed, comparable, equal_chunks, valid_chunks, changed_queries = 0, 0, 0, 0, set()
    for r in current:
        b = before[(r['query_id'], r['start'])]
        assert r['size'] == b['size'] and r['indices'] == b['indices']
        x, y = parse_compact(r['final'], r['size']), parse_compact(b['final'], b['size'])
        if x is None or y is None:
            continue
        valid_chunks += 1
        equal_chunks += r['final'] == b['final']
        n = sum(a != z for a, z in zip(x, y))
        changed += n
        comparable += len(x)
        if n:
            changed_queries.add(r['query_id'])
    return {'comparable_chunks': valid_chunks, 'identical_final_chunks': equal_chunks,
            'comparable_candidates': comparable, 'changed_candidates': changed,
            'changed_queries': len(changed_queries)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--selection', type=Path, required=True)
    p.add_argument('--prior', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if os.environ.get('VLLM_USE_V1') != '1' or a.model.name != 'Qwen3-4B-Instruct-2507':
        raise ValueError('Use V1 and the recorded Qwen model.')
    expected = {'vllm': '0.10.2', 'torch': '2.8.0+cu128', 'transformers': '4.57.1',
                'numpy': '2.2.6', 'xgrammar': '0.1.23', 'outlines-core': '0.2.11',
                'setuptools': '79.0.1'}
    versions = {name: metadata.version(name) for name in expected}
    if versions != expected:
        raise RuntimeError(f'Unexpected package versions: {versions}')
    if a.out.exists() or gpu_processes():
        raise RuntimeError('Choose a new output directory and wait for an idle GPU.')
    rows = lines(a.selection / 'inputs.jsonl')[:16]
    specs = lines(a.prior / 'specifications.jsonl')[:16]
    old = {m: lines(a.prior / (m + '_outputs.jsonl'))[:16] for m in MODES}
    if len(rows) != 16 or any(len(row['items']) != 32 for row in rows):
        raise ValueError('Unexpected saved 16-query prefix.')
    inputs, baseline = {m: [] for m in MODES}, {m: [] for m in MODES}
    for i, (row, spec) in enumerate(zip(rows, specs)):
        if row['query_id'] != spec['query_id']:
            raise ValueError('Saved specifications do not match selection.')
        for m in MODES:
            if old[m][i]['query_id'] != row['query_id'] or len(old[m][i]['raw']) != 4:
                raise ValueError('Saved output does not match selection.')
        for index, part in enumerate(chunks(row)):
            start, size = part['start'], len(part['items'])
            assert size == 8
            rev = {**part, 'items': list(reversed(part['items']))}
            text_original = prompt_without_example(part, spec['parsed'])
            text_reverse = prompt_without_example(rev, spec['parsed'])
            for mode, text, positions in [('original', text_original, list(range(start, start + size))),
                                          ('replay', text_original, list(range(start, start + size))),
                                          ('stable_reverse', text_reverse, list(range(start + size - 1, start - 1, -1)))]:
                previous = old[mode][i]['raw'][index]
                if (previous['start'] != start or previous['size'] != size or
                    previous['original_indices_in_prompt_order'] != positions or
                    previous['original_prompt_sha256'] != digest(text_original) or
                    previous['condition_prompt_sha256'] != digest(text)):
                    raise ValueError('Prior prompt hash or candidate mapping mismatch.')
                inputs[mode].append({'query_id': row['query_id'], 'start': start, 'size': size,
                                     'indices': positions, 'text': text})
                baseline[mode].append({'query_id': row['query_id'], 'start': start, 'size': size,
                                       'indices': positions, 'final': previous['final']})

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(a.model), local_files_only=True, trust_remote_code=False)
    def wrap(tok, text, tokenize=False):
        return tok.apply_chat_template([{'role': 'system', 'content': run_wands.SYSTEM},
                                        {'role': 'user', 'content': text}], tokenize=tokenize,
                                       add_generation_prompt=True)
    preflight_max = max(len(wrap(tokenizer, x['text'] + SUFFIX, True)) + 32
                        for m in MODES for x in inputs[m])
    if preflight_max > 4096:
        raise RuntimeError('Prompt and retry cap exceed the context limit.')
    a.out.mkdir(parents=True, exist_ok=False)
    (a.out / 'preflight.json').write_text(json.dumps({'queries': len(rows), 'chunks_per_mode': 64, 'versions': versions,
        'max_retry_prompt_plus_cap': preflight_max, 'selection_sha256': hashlib.sha256(
            (a.selection / 'inputs.jsonl').read_bytes()).hexdigest()}, indent=2) + '\n')

    from vllm import LLM, SamplingParams
    from vllm.sampling_params import GuidedDecodingParams
    llm = LLM(model=str(a.model), tokenizer=str(a.model), trust_remote_code=False, dtype='bfloat16',
              max_model_len=4096, gpu_memory_utilization=.70, max_num_seqs=16, enforce_eager=True,
              enable_prefix_caching=True, disable_log_stats=True)
    tok = llm.get_tokenizer()
    assert_own_gpu()
    for mode in MODES:
        for x in inputs[mode]:
            if wrap(tok, x['text']) != wrap(tokenizer, x['text']):
                raise RuntimeError('Loaded tokenizer changed the preflight prompt.')
    def guided():
        return SamplingParams(temperature=0, max_tokens=32,
                              guided_decoding=GuidedDecodingParams(regex='[YN]{8}'))
    control = tok.apply_chat_template([{'role': 'system', 'content': 'Follow the user instruction exactly.'},
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

    all_records = {}
    for mode in MODES:
        group = inputs[mode]
        assert_own_gpu()
        result = llm.generate([wrap(tok, x['text']) for x in group],
                              [guided() for _ in group], use_tqdm=False)
        first = [r.outputs[0].text.strip() for r in result]
        bad = [i for i, text in enumerate(first) if parse_compact(text, 8) is None]
        retries = []
        if bad:
            assert_own_gpu()
            retries = llm.generate([wrap(tok, group[i]['text'] + SUFFIX) for i in bad],
                                   [guided() for _ in bad], use_tqdm=False)
        retry_by_index = dict(zip(bad, retries))
        records = []
        for i, (x, r) in enumerate(zip(group, result)):
            retry = retry_by_index.get(i)
            final = retry.outputs[0].text.strip() if retry else first[i]
            records.append({k: x[k] for k in ('query_id', 'start', 'size', 'indices')} | {
                'prompt_sha256': digest(x['text']), 'wrapped_sha256': digest(wrap(tok, x['text'])),
                'first': first[i], 'final': final, 'first_finish_reason': r.outputs[0].finish_reason,
                'retry_finish_reason': retry.outputs[0].finish_reason if retry else None,
                'first_input_tokens': len(r.prompt_token_ids), 'first_output_tokens': len(r.outputs[0].token_ids),
                'retry_input_tokens': len(retry.prompt_token_ids) if retry else 0,
                'retry_output_tokens': len(retry.outputs[0].token_ids) if retry else 0})
        with (a.out / (mode + '_records.jsonl')).open('w', encoding='utf8') as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
        all_records[mode] = records
        print(mode, 'completed', len(records), 'chunks; retries:', len(bad), flush=True)
    report = {'protocol': 'PROTOCOL_offline_wands_replay.md', 'attempted_queries': len(rows),
              'attempted_chunks_per_mode': 64, 'comparisons_to_prior': {}, 'completion': {}}
    for mode in MODES:
        records = all_records[mode]
        pool_valid = collections.defaultdict(list)
        for r in records:
            pool_valid[r['query_id']].append(parse_compact(r['final'], r['size']) is not None)
        report['completion'][mode] = {'complete_pools': sum(len(v) == 4 and all(v) for v in pool_valid.values()),
            'valid_chunks': sum(parse_compact(r['final'], r['size']) is not None for r in records),
            'retry_calls': sum(r['retry_finish_reason'] is not None for r in records)}
        report['comparisons_to_prior'][mode] = compare(records, baseline[mode])
    report['current_replay_vs_original'] = compare(all_records['replay'], all_records['original'])
    (a.out / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    print('Completed isolated WANDS prompt replay.', flush=True)


if __name__ == '__main__':
    main()
