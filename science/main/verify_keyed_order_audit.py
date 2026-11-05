#!/usr/bin/env python3
"""Independently rebuild keyed prompts, parsing, associations and paired statistics."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import run_experiment as base
import run_wands


def records(path):
    with path.open(encoding='utf8') as f:
        return [json.loads(line) for line in f]


def sha(text):
    return hashlib.sha256(text.encode('utf8')).hexdigest()


def read_answer(text):
    try:
        obj = json.loads(text)
    except (ValueError, TypeError):
        return None
    if (not isinstance(obj, dict) or list(obj) != [str(i) for i in range(1, 9)]
            or any(v not in ('Y', 'N') for v in obj.values())
            or json.dumps(obj, separators=(',', ':')) != text):
        return None
    return [obj[str(i)] for i in range(1, 9)]


def prompt(row, chunk, spec, reverse):
    body = 'Query: ' + row['query'] + '\n'
    if spec is not None:
        body += 'Query specification: ' + json.dumps(spec, ensure_ascii=False) + '\n'
    items = [{'id': local + 1, 'text': item['text']} for local, item in enumerate(chunk)]
    if reverse:
        items.reverse()
    body += 'Products in order:\n' + '\n'.join(json.dumps(x, ensure_ascii=False) for x in items)
    return body + ('\nReturn only a JSON object mapping each given product id to Y or N for that product. '
                   'Use the given ids, not positions; include every id once, in increasing numeric order. '
                   'Do not add spaces or explanations.')


def close(a, b):
    if isinstance(a, list):
        return len(a) == len(b) and all(close(x, y) for x, y in zip(a, b))
    if a is None or b is None:
        return a is None and b is None
    return abs(a - b) < 1e-11


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--prior', type=Path, required=True)
    ap.add_argument('--selection', type=Path, required=True)
    ap.add_argument('--model', help='Optional local tokenizer path for an independent context audit')
    a = ap.parse_args()
    summary = json.loads((a.run / 'summary.json').read_text())
    analysis = json.loads((a.run / 'paired_analysis.json').read_text())
    sanity = json.loads((a.run / 'guidance_sanity.json').read_text())
    rows = records(a.selection / 'inputs.jsonl')
    specs = records(a.prior / 'specifications.jsonl')
    source_sha = hashlib.sha256((a.selection / 'inputs.jsonl').read_bytes()).hexdigest()
    spec_sha = hashlib.sha256((a.prior / 'specifications.jsonl').read_bytes()).hexdigest()
    assert summary['protocol'] == analysis['protocol'] == 'PROTOCOL_keyed_order_audit.md'
    assert summary['selection_inputs_sha256'] == analysis['selection_inputs_sha256'] == source_sha
    assert summary['prior_specifications_sha256'] == analysis['prior_specifications_sha256'] == spec_sha
    assert read_answer(sanity['unconstrained']) is None and read_answer(sanity['guided_keyed']) is not None
    assert len(rows) == len(specs) == 160
    modes = ['original', 'replay', 'stable_reverse']
    assert summary['mode_order'] == modes
    keyed = {mode: records(a.run / (mode + '_outputs.jsonl')) for mode in modes}
    compact = {mode: records(a.prior / (mode + '_outputs.jsonl')) for mode in modes}
    tokenizer = None
    if a.model:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(a.model, trust_remote_code=False, local_files_only=True)
    max_context = 0
    for mode in modes:
        assert len(keyed[mode]) == len(rows)
        first_valid = final_valid = retry_count = 0
        finish = Counter()
        for row, spec, saved in zip(rows, specs, keyed[mode]):
            assert row['query_id'] == spec['query_id'] == saved['query_id']
            assert spec['parsed'] == base.parse_spec(spec['final'])
            assert spec['fallback'] == (spec['parsed'] is None)
            assert len(row['items']) == 32 and len(saved['raw']) == 4
            predictions = []
            for j, raw in enumerate(saved['raw']):
                start = 8 * j
                chunk = row['items'][start:start + 8]
                normal = prompt(row, chunk, spec['parsed'], False)
                rendered = prompt(row, chunk, spec['parsed'], mode == 'stable_reverse')
                assert raw['start'] == start and raw['size'] == 8
                assert raw['presented_ids'] == (list(range(8, 0, -1)) if mode == 'stable_reverse' else list(range(1, 9)))
                assert raw['original_prompt_sha256'] == sha(normal) and raw['condition_prompt_sha256'] == sha(rendered)
                first, final = read_answer(raw['first']), read_answer(raw['final'])
                first_valid += first is not None
                final_valid += final is not None
                finish[raw['first_finish_reason']] += 1
                if first is None:
                    retry_count += 1
                    assert raw['retry_finish_reason'] is not None
                else:
                    assert raw['first'] == raw['final'] and raw['retry_finish_reason'] is None
                predictions.extend(final if final is not None else [None] * 8)
                if tokenizer:
                    suffix = '\nYour previous answer had the wrong format; follow the output instructions exactly.'
                    tokens = tokenizer.apply_chat_template([{'role': 'system', 'content': run_wands.SYSTEM},
                                                            {'role': 'user', 'content': rendered + suffix}],
                                                           tokenize=True, add_generation_prompt=True)
                    max_context = max(max_context, len(tokens) + 128)
            assert saved['pred'] == (predictions if None not in predictions else None)
        stats = summary['modes'][mode]
        assert stats['chunks'] == 640 and stats['attempted_pools'] == 160
        assert stats['valid_first_chunks'] == first_valid and stats['valid_final_chunks'] == final_valid
        assert stats['retry_chunk_calls'] == retry_count and stats['finish_reasons'] == dict(finish)
        assert stats['complete_pools'] == sum(r['pred'] is not None for r in keyed[mode])
    if tokenizer:
        assert max_context == summary['max_prompt_with_retry_plus_cap'] and max_context <= 4096
    pred = {f'keyed_{m}': [r['pred'] for r in keyed[m]] for m in modes}
    pred.update({f'compact_{m}': [r['pred'] for r in compact[m]] for m in modes})
    alternate = []
    for row, rec in zip(rows, compact['stable_reverse']):
        assert row['query_id'] == rec['query_id']
        vector = [None] * 32
        for raw in rec['raw']:
            start = raw['start']
            assert raw['size'] == 8 and raw['original_indices_in_prompt_order'] == list(range(start + 7, start - 1, -1))
            s = raw['final']
            if len(s) == 8 and set(s) <= {'Y', 'N'}:
                vector[start:start + 8] = list(s)
        alternate.append(vector if None not in vector else None)
    pred['compact_reverse_hypothetical_id_order'] = alternate
    common = [i for i in range(len(rows)) if all(v[i] is not None for v in pred.values())]
    assert len(common) == analysis['common_complete_pools']
    assert [rows[i]['query_id'] for i in common] == analysis['common_query_ids']
    for k, v in pred.items():
        assert sum(x is not None for x in v) == analysis['completion'][k]
        tp = fp = tn = fn = 0
        distances = []
        for i in common:
            reference = [x['gold'] for x in rows[i]['items']]
            p = v[i]
            distances.append(abs(p.count('Y') - reference.count('Y')) / len(p))
            for y, g in zip(p, reference):
                tp += y == g == 'Y'; tn += y == g == 'N'
                fp += y == 'Y' and g == 'N'; fn += y == 'N' and g == 'Y'
        actual = {'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn, 'f1': 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
                  'balanced_accuracy': (tp / (tp + fn) + tn / (tn + fp)) / 2 if tp + fn and tn + fp else None,
                  'ratio_mae': sum(distances) / len(distances) if distances else None}
        assert all(close(value, analysis['quality_common'][k][name]) for name, value in actual.items())
    draw = np.random.default_rng(20271125).integers(0, len(common), size=(4000, len(common)))
    for name, data in analysis['comparisons'].items():
        left, right = pred[data['left_condition']], pred[data['right_condition']]
        points = np.asarray([(sum(x != y for x, y in zip(left[i], right[i])), len(left[i]),
                              abs(left[i].count('Y') - right[i].count('Y')) / len(left[i])) for i in common], dtype=float)
        fractions = points[draw, 0].sum(axis=1) / points[draw, 1].sum(axis=1)
        changes = points[draw, 2].mean(axis=1)
        assert data['candidate_count'] == int(points[:, 1].sum())
        assert data['changed_candidate_count'] == int(points[:, 0].sum())
        assert close(data['candidate_label_change_fraction'], points[:, 0].sum() / points[:, 1].sum())
        assert close(data['candidate_label_change_fraction_ci'], np.quantile(fractions, [.025, .975]).tolist())
        assert close(data['mean_absolute_pool_ratio_change'], points[:, 2].mean())
        assert close(data['mean_absolute_pool_ratio_change_ci'], np.quantile(changes, [.025, .975]).tolist())
        assert data['pools_with_any_label_change'] == int((points[:, 0] > 0).sum())
        assert data['pools_with_changed_ratio'] == int((points[:, 2] > 0).sum())
        assert close(data['fixed_reference_max_candidate_error_floor'], points[:, 0].sum() / (2 * points[:, 1].sum()))
    print('Keyed prompts, product associations, parsing, completion, quality and paired bootstrap verified.')
    print('Common pools:', len(common), 'model:', summary['model'], 'context with retry and cap:', max_context if tokenizer else 'not checked')


if __name__ == '__main__':
    main()
