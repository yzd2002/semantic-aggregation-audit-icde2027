#!/usr/bin/env python3
"""Independently reconstruct a saved WANDS isolated-environment replay."""
import argparse
import collections
import hashlib
import json
import re
from pathlib import Path

MODES = ('original', 'replay', 'stable_reverse')
SYSTEM = ("You judge whether a product fully matches a shopper's query. "
          "Mark Y only if it matches the requested product type and every stated essential modifier; otherwise mark N. "
          "Use only the supplied information. Do not assume unknown product properties.")
SUFFIX = '\nYour previous answer had the wrong format; follow the output instructions exactly.'


def lines(path):
    with path.open(encoding='utf8') as f:
        return [json.loads(line) for line in f]


def sha(text):
    return hashlib.sha256(text.encode('utf8')).hexdigest()


def parse(text):
    return list(text) if isinstance(text, str) and re.fullmatch('[YN]{8}', text) else None


def stats(current, previous):
    by_key = {(r['query_id'], r['start']): r for r in previous}
    assert len(by_key) == len(previous) == len(current)
    chunks = candidates = changed = equal = 0
    queries = set()
    for row in current:
        before = by_key[(row['query_id'], row['start'])]
        assert row['indices'] == before['indices'] and row['size'] == before['size'] == 8
        a, b = parse(row['final']), parse(before['final'])
        if a is None or b is None:
            continue
        chunks += 1
        candidates += 8
        equal += row['final'] == before['final']
        count = sum(x != y for x, y in zip(a, b))
        changed += count
        if count:
            queries.add(row['query_id'])
    return {'comparable_chunks': chunks, 'identical_final_chunks': equal,
            'comparable_candidates': candidates, 'changed_candidates': changed,
            'changed_queries': len(queries)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--selection', type=Path, required=True)
    p.add_argument('--prior', type=Path, required=True)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--model', type=Path)
    a = p.parse_args()
    rows = lines(a.selection / 'inputs.jsonl')[:16]
    specs = lines(a.prior / 'specifications.jsonl')[:16]
    old = {m: lines(a.prior / (m + '_outputs.jsonl'))[:16] for m in MODES}
    out = {m: lines(a.run / (m + '_records.jsonl')) for m in MODES}
    pre = json.loads((a.run / 'preflight.json').read_text())
    ctrl = json.loads((a.run / 'controls.json').read_text())
    summary = json.loads((a.run / 'summary.json').read_text())
    assert pre['queries'] == len(rows) == len(specs) == 16 and pre['chunks_per_mode'] == 64
    assert pre['selection_sha256'] == hashlib.sha256((a.selection / 'inputs.jsonl').read_bytes()).hexdigest()
    assert pre['max_retry_prompt_plus_cap'] <= 4096
    assert pre['versions'] == {'vllm': '0.10.2', 'torch': '2.8.0+cu128', 'transformers': '4.57.1',
                               'numpy': '2.2.6', 'xgrammar': '0.1.23', 'outlines-core': '0.2.11',
                               'setuptools': '79.0.1'}
    assert ctrl['unconstrained_invalid'] == (parse(ctrl['unconstrained']) is None) is True
    assert ctrl['guided_valid'] == (parse(ctrl['guided']) is not None) is True
    assert summary['protocol'] == 'PROTOCOL_offline_wands_replay.md'
    assert summary['attempted_queries'] == 16 and summary['attempted_chunks_per_mode'] == 64
    tokenizer = None
    if a.model:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(a.model), local_files_only=True, trust_remote_code=False)
    max_context = 0
    baseline = {m: [] for m in MODES}
    for qi, (row, spec) in enumerate(zip(rows, specs)):
        assert row['query_id'] == spec['query_id'] and len(row['items']) == 32
        for mode in MODES:
            assert old[mode][qi]['query_id'] == row['query_id'] and len(old[mode][qi]['raw']) == 4
        for chunk in range(4):
            start = chunk * 8
            items = [{'id': j + 1, 'text': it['text']}
                     for j, it in enumerate(row['items'][start:start + 8])]
            def prompt(product_list):
                value = 'Query: ' + row['query'] + '\n'
                if spec['parsed'] is not None:
                    value += 'Query specification: ' + json.dumps(spec['parsed'], ensure_ascii=False) + '\n'
                value += 'Products in order:\n' + '\n'.join(json.dumps(it, ensure_ascii=False) for it in product_list)
                return value + ('\nReturn exactly 8 characters, one Y or N per product in the listed order. '
                                'No spaces, punctuation, or explanations.')
            original, reverse = prompt(items), prompt(list(reversed(items)))
            for mode, text, positions in [('original', original, list(range(start, start + 8))),
                                          ('replay', original, list(range(start, start + 8))),
                                          ('stable_reverse', reverse, list(range(start + 7, start - 1, -1)))]:
                prior = old[mode][qi]['raw'][chunk]
                record = out[mode][qi * 4 + chunk]
                assert (prior['start'], prior['size'], prior['original_indices_in_prompt_order']) == (start, 8, positions)
                assert prior['original_prompt_sha256'] == sha(original) and prior['condition_prompt_sha256'] == sha(text)
                assert (record['query_id'], record['start'], record['size'], record['indices']) == (row['query_id'], start, 8, positions)
                assert record['prompt_sha256'] == sha(text)
                invalid = parse(record['first']) is None
                assert (record['retry_finish_reason'] is not None) == invalid
                if invalid:
                    assert record['retry_input_tokens'] > 0 and record['retry_output_tokens'] > 0
                else:
                    assert record['first'] == record['final'] and record['retry_input_tokens'] == record['retry_output_tokens'] == 0
                assert record['first_input_tokens'] > 0 and record['first_output_tokens'] > 0
                if tokenizer:
                    def wrap(value, tokenize):
                        return tokenizer.apply_chat_template([{'role': 'system', 'content': SYSTEM},
                            {'role': 'user', 'content': value}], tokenize=tokenize, add_generation_prompt=True)
                    assert sha(wrap(text, False)) == record['wrapped_sha256']
                    assert len(wrap(text, True)) == record['first_input_tokens']
                    max_context = max(max_context, len(wrap(text + SUFFIX, True)) + 32)
                    if invalid:
                        assert len(wrap(text + SUFFIX, True)) == record['retry_input_tokens']
                baseline[mode].append({'query_id': row['query_id'], 'start': start, 'size': 8,
                                       'indices': positions, 'final': prior['final']})
    if tokenizer:
        assert max_context == pre['max_retry_prompt_plus_cap']
    for mode in MODES:
        assert len(out[mode]) == 64
        by_query = collections.defaultdict(list)
        for r in out[mode]:
            by_query[r['query_id']].append(parse(r['final']) is not None)
        completion = {'complete_pools': sum(len(v) == 4 and all(v) for v in by_query.values()),
                      'valid_chunks': sum(parse(r['final']) is not None for r in out[mode]),
                      'retry_calls': sum(r['retry_finish_reason'] is not None for r in out[mode])}
        assert completion == summary['completion'][mode]
        assert stats(out[mode], baseline[mode]) == summary['comparisons_to_prior'][mode]
    assert stats(out['replay'], out['original']) == summary['current_replay_vs_original']
    print('Independently reconstructed all prompts, candidate mappings, retries, completions and replay comparisons.')
    print('Wrapped prompts and token lengths checked:', bool(tokenizer))


if __name__ == '__main__':
    main()
