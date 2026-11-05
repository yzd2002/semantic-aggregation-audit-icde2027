#!/usr/bin/env python3
"""Independently verify prompts, associations, fallback policy and paired results."""
import argparse
import collections
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import run_experiment as base
import run_wands


def lines(path):
    with path.open(encoding='utf8') as f:
        return [json.loads(s) for s in f]


def digest(text):
    return hashlib.sha256(text.encode('utf8')).hexdigest()


def keyed(text):
    try:
        x = json.loads(text)
    except (ValueError, TypeError):
        return None
    if (not isinstance(x, dict) or list(x) != [str(i) for i in range(1, 9)]
            or any(v not in ('Y', 'N') for v in x.values())
            or json.dumps(x, separators=(',', ':')) != text):
        return None
    return [x[str(i)] for i in range(1, 9)]


def point(text):
    return text if text in ('Y', 'N') else None


def key_prompt(row, items, spec, reverse):
    text = 'Query: ' + row['query'] + '\n'
    if spec is not None:
        text += 'Query specification: ' + json.dumps(spec, ensure_ascii=False) + '\n'
    group = [{'id': i+1, 'text': item['text']} for i, item in enumerate(items)]
    if reverse:
        group.reverse()
    text += 'Products in order:\n' + '\n'.join(json.dumps(x, ensure_ascii=False) for x in group)
    return text + ('\nReturn only a JSON object mapping each given product id to Y or N for that product. '
                   'Use the given ids, not positions; include every id once, in increasing numeric order. '
                   'Do not add spaces or explanations.')


def single_prompt(row, item, spec):
    text = 'Query: ' + row['query'] + '\n'
    if spec is not None:
        text += 'Query specification: ' + json.dumps(spec, ensure_ascii=False) + '\n'
    return text + 'Product: ' + item['text'] + '\nAnswer ONLY Y or N.'


def close(a, b):
    if isinstance(a, list):
        return len(a) == len(b) and all(close(x, y) for x, y in zip(a, b))
    if a is None or b is None:
        return a is None and b is None
    return abs(a-b) < 1e-11


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--selection', type=Path, required=True)
    ap.add_argument('--model', help='Optional local tokenizer path for independent context/input-token checks')
    a = ap.parse_args()
    rows = lines(a.selection / 'inputs.jsonl')
    metadata = json.loads((a.selection / 'selection.json').read_text())
    summary = json.loads((a.run / 'summary.json').read_text())
    analysis = json.loads((a.run / 'paired_analysis.json').read_text())
    sanity = json.loads((a.run / 'guidance_sanity.json').read_text())
    sha = hashlib.sha256((a.selection / 'inputs.jsonl').read_bytes()).hexdigest()
    assert len(rows) == 96 and all(len(r['items']) == 32 for r in rows)
    assert metadata['protocol'] == summary['protocol'] == analysis['protocol'] == 'PROTOCOL_order_fallback_validation.md'
    assert summary['selection_inputs_sha256'] == analysis['selection_inputs_sha256'] == sha
    assert summary['model'] == analysis['model'] == 'Qwen3-4B-Instruct-2507'
    assert keyed(sanity['unconstrained']) is None and point(sanity['unconstrained']) is None
    assert keyed(sanity['json_guided']) is not None and point(sanity['point_guided']) is not None
    assert not sanity['unconstrained_valid_json'] and not sanity['unconstrained_valid_point']
    assert sanity['guided_json_valid'] and sanity['guided_point_valid']
    modes = ['original', 'replay', 'stable_reverse', 'pointwise', 'pointwise_replay_disagreements']
    assert summary['mode_order'] == modes and summary['runtime']['engine'] == 'V1 (VLLM_USE_V1=1)'
    records = {m: lines(a.run / (m + '_responses.jsonl')) for m in modes}
    specs = lines(a.run / 'specifications.jsonl')
    assert len(specs) == len(rows)
    first_bad = final_bad = 0
    for row, rec in zip(rows, specs):
        assert rec['query_id'] == row['query_id'] and rec['prompt_sha256'] == digest(base.p_spec(row))
        first, final = base.parse_spec(rec['first']), base.parse_spec(rec['final'])
        first_bad += first is None; final_bad += final is None
        if first is not None:
            assert rec['first'] == rec['final']
        assert rec['parsed'] == final and rec['fallback'] == (final is None)
    assert summary['specifications']['initial_calls'] == 96
    assert summary['specifications']['retry_calls'] == first_bad and summary['specifications']['fallback_pools'] == final_bad
    tok = None
    if a.model:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=False, local_files_only=True)
    def tokens(text):
        return len(tok.apply_chat_template([{'role': 'system', 'content': run_wands.SYSTEM},
                                            {'role': 'user', 'content': text}], tokenize=True, add_generation_prompt=True))
    suffix = '\nYour previous answer had the wrong format; follow the output instructions exactly.'
    if tok:
        max_spec = max(tokens(base.p_spec(r) + '\nReturn valid JSON only.') + 160 for r in rows)
        input_spec = sum(tokens(base.p_spec(row)) + (tokens(base.p_spec(row) + '\nReturn valid JSON only.')
                         if base.parse_spec(rec['first']) is None else 0) for row, rec in zip(rows, specs))
        assert max_spec == summary['specifications']['max_prompt_with_retry_plus_cap'] <= 4096
        assert input_spec == summary['specifications']['logical_tokens']['input']
    pred = {}
    expected_point_prompts = {}
    for mode in modes[:4]:
        saved = records[mode]
        per_query = 4 if mode != 'pointwise' else 32
        assert len(saved) == len(rows) * per_query
        vectors = []
        max_context = 0
        for q, (row, spec) in enumerate(zip(rows, specs)):
            vector = []
            for j in range(per_query):
                rec = saved[q*per_query+j]
                start = j*8 if mode != 'pointwise' else j
                assert rec['query_id'] == row['query_id'] and rec['start'] == start
                if mode != 'pointwise':
                    assert rec['size'] == 8
                    expected = list(range(8, 0, -1)) if mode == 'stable_reverse' else list(range(1, 9))
                    assert rec['presented_ids'] == expected
                    prompt = key_prompt(row, row['items'][start:start+8], spec['parsed'], mode == 'stable_reverse')
                    parser, cap = keyed, 128
                else:
                    assert rec['size'] == 1 and rec['presented_ids'] is None
                    prompt = single_prompt(row, row['items'][start], spec['parsed'])
                    expected_point_prompts[(row['query_id'], start)] = prompt
                    parser, cap = point, 4
                assert rec['prompt_sha256'] == digest(prompt)
                first, final = parser(rec['first']), parser(rec['final'])
                if first is None:
                    assert rec['retry_finish_reason'] is not None
                else:
                    assert rec['first'] == rec['final'] and rec['retry_finish_reason'] is None
                vector.extend((final if final is not None else [None]*8) if mode != 'pointwise' else [final])
                if tok:
                    assert rec['first_input_tokens'] == tokens(prompt)
                    assert rec['retry_input_tokens'] == (tokens(prompt + suffix) if first is None else 0)
                    max_context = max(max_context, tokens(prompt + suffix) + cap)
            vectors.append(vector if None not in vector else None)
        pred[mode] = vectors
        stats = summary['modes'][mode]
        parser = point if mode == 'pointwise' else keyed
        assert stats['initial_calls'] == len(saved) and stats['retry_calls'] == sum(parser(r['first']) is None for r in saved)
        assert stats['valid_first'] == sum(parser(r['first']) is not None for r in saved)
        assert stats['valid_final'] == sum(parser(r['final']) is not None for r in saved)
        assert stats['finish_reasons'] == dict(collections.Counter(r['first_finish_reason'] for r in saved))
        for key in ('input', 'output'):
            assert stats['logical_tokens'][key] == sum(r['first_'+key+'_tokens'] + r['retry_'+key+'_tokens'] for r in saved)
        if tok:
            assert stats['max_prompt_with_retry_plus_cap'] == max_context <= 4096
    point_records = records['pointwise']
    partial = [[point(x['final']) for x in point_records[q*32:(q+1)*32]] for q in range(len(rows))]
    need = []
    for q, row in enumerate(rows):
        for c in range(4):
            x = keyed(records['original'][q*4+c]['final']); y = keyed(records['stable_reverse'][q*4+c]['final'])
            if x is not None and y is not None:
                need.extend((q, c*8+i) for i in range(8) if x[i] != y[i])
    replay = records['pointwise_replay_disagreements']
    expected_replays = [(rows[q]['query_id'], i) for q, i in need if partial[q][i] is not None]
    assert [(x['query_id'], x['start']) for x in replay] == expected_replays
    matched = changed = 0
    max_replay = 0
    for rec in replay:
        key = (rec['query_id'], rec['start']); prompt = expected_point_prompts[key]
        assert rec['prompt_sha256'] == digest(prompt) and rec['size'] == 1 and rec['presented_ids'] is None
        first, final = point(rec['first']), point(rec['final'])
        if first is None:assert rec['retry_finish_reason'] is not None
        else:assert rec['first'] == rec['final'] and rec['retry_finish_reason'] is None
        if final is not None:
            matched += 1
            initial = next(x for x in point_records if x['query_id'] == key[0] and x['start'] == key[1])
            changed += initial['final'] != final
        if tok:
            assert rec['first_input_tokens'] == tokens(prompt)
            assert rec['retry_input_tokens'] == (tokens(prompt + suffix) if first is None else 0)
            max_replay = max(max_replay, tokens(prompt + suffix) + 4)
    stats = summary['modes']['pointwise_replay_disagreements']
    assert stats['initial_calls'] == len(replay) and stats['retry_calls'] == sum(point(x['first']) is None for x in replay)
    assert stats['valid_first'] == sum(point(x['first']) is not None for x in replay)
    assert stats['valid_final'] == sum(point(x['final']) is not None for x in replay)
    for key in ('input', 'output'):
        assert stats['logical_tokens'][key] == sum(x['first_'+key+'_tokens'] + x['retry_'+key+'_tokens'] for x in replay)
    if tok:
        assert stats['max_prompt_with_retry_plus_cap'] == max_replay <= 4096
    lexical = []
    for row in rows:
        words = set(re.findall(r'\w+', row['query'].casefold()))
        lexical.append(['Y' if words and words <= set(re.findall(r'\w+', x['text'].casefold())) else 'N' for x in row['items']])
    pred['always_Y'] = [['Y']*32 for row in rows]
    pred['always_N'] = [['N']*32 for row in rows]
    pred['lexical'] = lexical
    def fallback(first, second, itemwise):
        if first is None or second is None:
            return None
        result = [x if x == y else h for x, y, h in zip(first, second, itemwise)]
        return result if None not in result else None
    pred['pointwise_fallback'] = [fallback(x, y, h) for x, y, h in zip(pred['original'], pred['stable_reverse'], partial)]
    pred['lexical_fallback'] = [fallback(x, y, h) for x, y, h in zip(pred['original'], pred['stable_reverse'], lexical)]
    common = [i for i in range(96) if all(pred[k][i] is not None for k in ('original', 'replay', 'stable_reverse', 'pointwise'))]
    assert analysis['attempted_pools'] == 96 and analysis['candidate_count'] == 3072
    assert analysis['resamples'] == 4000 and analysis['seed'] == 20271127
    assert analysis['common_complete_pools'] == len(common) and analysis['common_query_ids'] == [rows[i]['query_id'] for i in common]
    assert analysis['completion'] == {k: sum(x is not None for x in v) for k, v in pred.items()}
    assert analysis['grade_counts'] == dict(collections.Counter(x['grade'] for row in rows for x in row['items']))
    assert analysis['specification_fallback_pools'] == final_bad
    for k, v in pred.items():
        tp = fp = tn = fn = 0; errors = []
        for i in common:
            gold = [x['gold'] for x in rows[i]['items']]; y = v[i]
            errors.append(abs(y.count('Y')-gold.count('Y'))/len(gold))
            for p, g in zip(y, gold):
                tp += p == g == 'Y'; tn += p == g == 'N'; fp += p == 'Y' and g == 'N'; fn += p == 'N' and g == 'Y'
        vals = {'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn, 'f1': 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,
                'balanced_accuracy': (tp/(tp+fn)+tn/(tn+fp))/2 if tp+fn and tn+fp else None,
                'ratio_mae': sum(errors)/len(errors) if errors else None}
        assert all(close(x, analysis['quality_common'][k][name]) for name, x in vals.items())
    draw = np.random.default_rng(20271127).integers(0, len(common), size=(4000, len(common)))
    for name, data in analysis['comparisons'].items():
        left, right = data['left'], data['right']; per_query = []
        for i in common:
            gold = [x['gold'] for x in rows[i]['items']]
            def stat(y):
                tp = sum(p == g == 'Y' for p, g in zip(y, gold))
                fp = sum(p == 'Y' and g == 'N' for p, g in zip(y, gold))
                fn = sum(p == 'N' and g == 'Y' for p, g in zip(y, gold))
                return tp, fp, fn, abs(y.count('Y') - gold.count('Y'))/len(gold)
            per_query.append(stat(pred[left][i])+stat(pred[right][i]))
        arr = np.asarray(per_query, dtype=float)[draw]; totals = arr.sum(axis=1)
        def f1(j):
            numerator = 2*totals[:, j]; denominator = numerator+totals[:, j+1]+totals[:, j+2]
            return np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator != 0)
        fdelta = f1(4)-f1(0); mdelta = arr[:, :, 7].mean(axis=1)-arr[:, :, 3].mean(axis=1)
        assert close(data['f1_difference'], analysis['quality_common'][right]['f1']-analysis['quality_common'][left]['f1'])
        assert close(data['ratio_mae_difference'], analysis['quality_common'][right]['ratio_mae']-analysis['quality_common'][left]['ratio_mae'])
        assert close(data['f1_difference_ci'], np.quantile(fdelta, [.025, .975]).tolist())
        assert close(data['ratio_mae_difference_ci'], np.quantile(mdelta, [.025, .975]).tolist())
    d = analysis['disagreement']
    assert d['candidates'] == len(need) and d['valid_pointwise_replays'] == matched and d['pointwise_replay_changed'] == changed
    req = [point_records[q*32+i] for q, i in need]
    retries = sum(x['retry_finish_reason'] is not None for x in req)
    assert d['virtual_fallback_calls'] == len(req) and d['virtual_fallback_retries'] == retries
    spec = summary['specifications']; keyed_stats = [summary['modes'][x] for x in ('original', 'stable_reverse')]
    calls = spec['initial_calls']+spec['retry_calls']+sum(x['initial_calls']+x['retry_calls'] for x in keyed_stats)+len(req)+retries
    assert d['virtual_total_calls_with_spec_and_two_keyed_passes'] == calls
    for key in ('input', 'output'):
        v = spec['logical_tokens'][key]+sum(x['logical_tokens'][key] for x in keyed_stats)
        v += sum(x['first_'+key+'_tokens']+x['retry_'+key+'_tokens'] for x in req)
        assert d['virtual_logical_tokens'][key] == v
    assert analysis['keyed_replay_changed_candidates'] == sum(sum(a != b for a,b in zip(pred['original'][i], pred['replay'][i])) for i in common)
    primary = analysis['comparisons']['pointwise_fallback_minus_original']['ratio_mae_difference_ci'][1] < 0
    assert analysis['primary_upper_endpoint_below_zero'] == primary
    print('Prompt hashes, constrained outputs, keyed mapping, fallback, replay, quality and paired bootstrap independently verified.')
    print('Common pools:',len(common),'disagreeing candidates:',len(need),'tokenizer context audit:',bool(tok))


if __name__ == '__main__':
    main()
