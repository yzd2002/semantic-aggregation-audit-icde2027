#!/usr/bin/env python3
"""Independently reconstruct random-order prompts, labels and paired summaries."""
import argparse
import collections
import hashlib
import json
import re
from fractions import Fraction
from pathlib import Path

MODES = ('baseline_a', 'perm_1', 'baseline_b', 'perm_2', 'perm_3', 'baseline_c')
PAIRS = (('perm_1', 'baseline_a'), ('perm_2', 'baseline_b'), ('perm_3', 'baseline_c'),
         ('baseline_b', 'baseline_a'), ('baseline_c', 'baseline_a'))
SYSTEM = ("You judge whether a product fully matches a shopper's query. "
          "Mark Y only if it matches the requested product type and every stated essential modifier; otherwise mark N. "
          "Use only the supplied information. Do not assume unknown product properties.")
SUFFIX = '\nYour previous answer had the wrong format; follow the output instructions exactly.'


def read(path):
    return [json.loads(x) for x in path.read_text(encoding='utf8').splitlines()]


def sha(text):
    return hashlib.sha256(text.encode('utf8')).hexdigest()


def parse(value):
    return list(value) if isinstance(value, str) and re.fullmatch('[YN]{8}', value) else None


def close(a, b):
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(close(a[k], b[k]) for k in a)
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(a - b) < 1e-12
    return a == b


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--selection', type=Path, required=True)
    p.add_argument('--orders', type=Path, required=True)
    p.add_argument('--prior', type=Path, required=True)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--analysis', type=Path, required=True)
    p.add_argument('--model', type=Path)
    a = p.parse_args()
    rows, maps = read(a.selection / 'inputs.jsonl'), read(a.orders / 'mappings.jsonl')
    specs, prior = read(a.prior / 'specifications.jsonl'), read(a.prior / 'original_outputs.jsonl')
    saved = {m: read(a.run / (m + '_records.jsonl')) for m in MODES}
    pre = json.loads((a.run / 'preflight.json').read_text())
    ctrl = json.loads((a.run / 'controls.json').read_text())
    report = json.loads((a.analysis / 'summary.json').read_text())
    per_pool = read(a.analysis / 'per_pool.jsonl')
    assert len(rows) == len(maps) == len(specs) == len(prior) == 160
    assert all(len(saved[m]) == 640 for m in MODES)
    assert pre['protocol'] == report['protocol'] == 'PROTOCOL_random_order_sensitivity.md'
    assert pre['model'] == report['model'] and pre['queries'] == 160 and pre['chunks_per_mode'] == 640
    assert pre['mode_order'] == list(MODES) and pre['batch_size'] == 64
    assert pre['versions'] == {'vllm': '0.10.2', 'torch': '2.8.0+cu128', 'transformers': '4.57.1',
                               'numpy': '2.2.6', 'xgrammar': '0.1.23', 'outlines-core': '0.2.11',
                               'setuptools': '79.0.1'}
    assert pre['max_retry_prompt_plus_cap'] <= 4096
    assert pre['selection_sha256'] == hashlib.sha256((a.selection / 'inputs.jsonl').read_bytes()).hexdigest()
    assert pre['mappings_sha256'] == hashlib.sha256((a.orders / 'mappings.jsonl').read_bytes()).hexdigest()
    assert ctrl['unconstrained_invalid'] == (parse(ctrl['unconstrained']) is None) is True
    assert ctrl['guided_valid'] == (parse(ctrl['guided']) is not None) is True
    tokenizer = None
    if a.model:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(a.model), local_files_only=True, trust_remote_code=False)
    decoded = {m: [] for m in MODES}
    completion = {m: {'complete_pools': 0, 'valid_chunks': 0, 'retry_calls': 0} for m in MODES}
    max_context = 0
    for qi, (row, mapping, spec, old) in enumerate(zip(rows, maps, specs, prior)):
        qid = row['query_id']
        assert qid == mapping['query_id'] == spec['query_id'] == old['query_id'] and len(row['items']) == 32
        vectors = {m: [None] * 32 for m in MODES}
        for ci, start in enumerate(range(0, 32, 8)):
            order = mapping['chunks'][ci]
            assert order['start'] == start
            items = [{'id': j + 1, 'text': x['text']} for j, x in enumerate(row['items'][start:start + 8])]
            def prompt(product_list):
                text = 'Query: ' + row['query'] + '\n'
                if spec['parsed'] is not None:
                    text += 'Query specification: ' + json.dumps(spec['parsed'], ensure_ascii=False) + '\n'
                text += 'Products in order:\n' + '\n'.join(json.dumps(x, ensure_ascii=False) for x in product_list)
                return text + ('\nReturn exactly 8 characters, one Y or N per product in the listed order. '
                               'No spaces, punctuation, or explanations.')
            original = prompt(items)
            prior_chunk = old['raw'][ci]
            assert prior_chunk['start'] == start and prior_chunk['size'] == 8
            assert prior_chunk['original_indices_in_prompt_order'] == list(range(start, start + 8))
            assert prior_chunk['condition_prompt_sha256'] == sha(original)
            for mode in MODES:
                positions = order['orders'][mode] if mode.startswith('perm_') else list(range(8))
                assert sorted(positions) == list(range(8))
                text = prompt([items[j] for j in positions])
                record = saved[mode][qi * 4 + ci]
                expected_indices = [start + j for j in positions]
                assert (record['query_id'], record['start'], record['size'], record['indices']) == (qid, start, 8, expected_indices)
                assert record['original_prompt_sha256'] == sha(original) and record['prompt_sha256'] == sha(text)
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
                labels = parse(record['final'])
                completion[mode]['retry_calls'] += invalid
                completion[mode]['valid_chunks'] += labels is not None
                if labels is not None:
                    for index, label in zip(expected_indices, labels):
                        assert vectors[mode][index] is None
                        vectors[mode][index] = label
        for mode in MODES:
            valid = None not in vectors[mode]
            completion[mode]['complete_pools'] += valid
            decoded[mode].append(vectors[mode] if valid else None)
    if tokenizer:
        assert max_context == pre['max_retry_prompt_plus_cap']
    assert completion == report['completion']
    common = [i for i in range(160) if all(decoded[m][i] is not None for m in MODES)]
    assert len(common) == report['common_complete_pools'] > 0
    assert [rows[i]['query_id'] for i in common] == report['common_query_ids']
    for mode in MODES:
        counts, errors = collections.Counter(), []
        for i in common:
            reference = [it['gold'] for it in rows[i]['items']]
            output = decoded[mode][i]
            counts.update(zip(reference, output))
            errors.append(Fraction(abs(reference.count('Y') - output.count('Y')), 32))
        tp, tn, fp, fn = counts['Y', 'Y'], counts['N', 'N'], counts['N', 'Y'], counts['Y', 'N']
        expected = {'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
            'f1': 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.,
            'balanced_accuracy': (tp / (tp + fn) + tn / (tn + fp)) / 2 if tp + fn and tn + fp else None,
            'ratio_mae': float(sum(errors, Fraction()) / len(errors))}
        assert close(expected, report['quality_common'][mode])
    expected_rows = [{'query_id': rows[i]['query_id'], 'comparisons': {}} for i in common]
    for target, base in PAIRS:
        changed = gap = changed_pools = 0
        for j, i in enumerate(common):
            x, y = decoded[target][i], decoded[base][i]
            d, delta = sum(u != v for u, v in zip(x, y)), abs(x.count('Y') - y.count('Y'))
            changed += d
            gap += delta
            changed_pools += d > 0
            expected_rows[j]['comparisons'][target + '_vs_' + base] = {'changed_candidates': d,
                                                                       'absolute_positive_count_change': delta}
        expected = {'changed_candidates': changed, 'comparable_candidates': len(common) * 32,
            'change_fraction': changed / (len(common) * 32), 'changed_pools': changed_pools,
            'mean_absolute_ratio_change': gap / (len(common) * 32)}
        assert close(expected, report['comparisons'][target + '_vs_' + base])
    assert expected_rows == per_pool
    print('Independently reconstructed prompts, mappings, retries, completions, quality and paired order comparisons.')
    print('Wrapped prompts and token lengths checked:', bool(tokenizer))


if __name__ == '__main__':
    main()
