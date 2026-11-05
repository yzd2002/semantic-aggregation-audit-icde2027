#!/usr/bin/env python3
"""Independently audit the new ESCI prompts, routes, responses and primary analysis."""
import argparse
import collections
import hashlib
import json
import math
import re
from fractions import Fraction
from pathlib import Path

SYSTEM = ("You judge whether each product can satisfy a shopper's query. "
          "Mark Y for an exact match or a functional substitute, and N for a mere accessory, "
          "an unrelated product, or a product lacking a stated essential requirement. "
          "Use only the supplied information. Do not assume unknown product properties.")
RETRY = '\nYour previous answer had the wrong format; follow the output instructions exactly.'
SPEC_RETRY = '\nReturn valid JSON only.'
PROTOCOL = 'PROTOCOL_esci_selective_holdout.md'


def lines(path):
    with path.open(encoding='utf8') as f:
        return [json.loads(x) for x in f]


def sha(text):
    return hashlib.sha256(text.encode('utf8')).hexdigest()


def nodups(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate key')
        result[key] = value
    return result


def specparse(text):
    try:
        x = json.loads(text, object_pairs_hook=nodups)
        if (not isinstance(x, dict) or set(x) != {'product_type', 'required', 'preferred'} or
                not isinstance(x['product_type'], str) or not x['product_type'].strip() or
                any(not isinstance(x[k], list) or any(not isinstance(y, str) for y in x[k])
                    for k in ('required', 'preferred'))):
            return None
        return x
    except (ValueError, TypeError):
        return None


def keyedparse(text, n):
    pattern = r'\{' + ','.join('"' + str(i) + '":"[YN]"' for i in range(1, n+1)) + r'\}'
    if re.fullmatch(pattern, text) is None:
        return None
    x = json.loads(text, object_pairs_hook=nodups)
    return [x[str(i)] for i in range(1, n+1)]


def one(text):
    return text if text in ('Y', 'N') else None


def specprompt(row):
    return ('Describe only requirements explicitly supported by this shopping query. '
            'Return ONLY JSON with exactly these keys: "product_type" (a nonempty string), '
            '"required" (an array of strings), and "preferred" (an array of strings). '
            'Do not invent unstated attributes. Empty arrays are allowed.\nQuery: ' + row['query'])


def prefix(row, spec):
    result = 'Query: ' + row['query'] + '\n'
    if spec is not None:
        result += 'Query specification: ' + json.dumps(spec, ensure_ascii=False) + '\n'
    return result


def keyprompt(row, start, spec, reverse):
    items = [{'id': i+1, 'text': item['text']} for i, item in enumerate(row['items'][start:start+8])]
    if reverse:
        items.reverse()
    text = prefix(row, spec) + 'Products in order:\n' + '\n'.join(json.dumps(x, ensure_ascii=False) for x in items)
    text += ('\nReturn only a JSON object mapping each given product id to Y or N for that product. '
             'Use the given ids, not positions; include every id once, in increasing numeric order. '
             'Do not add spaces or explanations.')
    return text, [x['id'] for x in items]


def pointprompt(row, index, spec):
    return prefix(row, spec) + 'Product: ' + row['items'][index]['text'] + '\nAnswer ONLY Y or N.'


def quality(rows, vectors, indices):
    counts = collections.Counter()
    ratios = []
    for i in indices:
        ref = [x['gold'] for x in rows[i]['items']]
        out = vectors[i]
        if len(out) != len(ref):
            raise ValueError('Candidate alignment mismatch.')
        counts.update(zip(ref, out))
        ratios.append(Fraction(abs(ref.count('Y') - out.count('Y')), len(ref)))
    tp, fp, tn, fn = counts['Y', 'Y'], counts['N', 'Y'], counts['N', 'N'], counts['Y', 'N']
    return {'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
            'f1': 2*tp / (2*tp+fp+fn) if 2*tp+fp+fn else 0.,
            'balanced_accuracy': (tp/(tp+fn)+tn/(tn+fp))/2 if tp+fn and tn+fp else None,
            'ratio_mae': float(sum(ratios, Fraction())/len(ratios)) if ratios else None}


def close(x, y):
    if isinstance(x, dict):
        return isinstance(y, dict) and x.keys() == y.keys() and all(close(x[k], y[k]) for k in x)
    if isinstance(x, (list, tuple)):
        return isinstance(y, (list, tuple)) and len(x) == len(y) and all(close(a, b) for a, b in zip(x, y))
    if isinstance(x, (float, int)) and not isinstance(x, bool):
        return isinstance(y, (float, int)) and math.isclose(x, y, rel_tol=1e-10, abs_tol=1e-10)
    return x == y


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--selection', type=Path, required=True)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--analysis', type=Path, required=True)
    p.add_argument('--model', type=Path)
    a = p.parse_args()
    rows = lines(a.selection / 'inputs.jsonl')
    summary = json.loads((a.run / 'summary.json').read_text(encoding='utf8'))
    analysis = json.loads((a.analysis / 'summary.json').read_text(encoding='utf8'))
    digest = hashlib.sha256((a.selection / 'inputs.jsonl').read_bytes()).hexdigest()
    if (len(rows) != 250 or summary['protocol'] != PROTOCOL or not summary.get('completed') or
            summary['selection_inputs_sha256'] != digest or analysis['selection_inputs_sha256'] != digest or
            summary['model'] != analysis['model']):
        raise ValueError('Protocol, sample, completion or model mismatch.')
    sanity = json.loads((a.run / 'guidance_sanity.json').read_text(encoding='utf8'))
    if (sanity['unconstrained_valid_keyed'] or sanity['unconstrained_valid_point'] or
            not sanity['guided_keyed_valid'] or not sanity['guided_point_valid'] or
            keyedparse(sanity['unconstrained'], 8) is not None or one(sanity['unconstrained']) is not None or
            keyedparse(sanity['guided_keyed'], 8) is None or one(sanity['guided_point']) is None):
        raise ValueError('Constraint negative control mismatch.')
    tokenizer = None
    if a.model:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(a.model), local_files_only=True, trust_remote_code=False)

    def wrap(text, tokenized):
        return tokenizer.apply_chat_template([{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': text}],
                                             tokenize=tokenized, add_generation_prompt=True)

    def check(record, qid, start, n, ids, prompt, kind):
        if (record['query_id'], record['start'], record['size'], record['presented_ids'], record['prompt_sha256']) != (qid, start, n, ids, sha(prompt)):
            raise ValueError('Request association, order, IDs or prompt hash differs.')
        parser = specparse if kind == 'spec' else (lambda s: keyedparse(s, n)) if kind == 'keyed' else one
        suffix = SPEC_RETRY if kind == 'spec' else RETRY
        cap = 160 if kind == 'spec' else 128 if kind == 'keyed' else 4
        first, final = parser(record['first']), parser(record['final'])
        retried = first is None
        if (record['retry_finish_reason'] is not None) != retried:
            raise ValueError('Retry was not triggered by invalid first response.')
        if not retried and (record['first'] != record['final'] or record['retry_input_tokens'] or record['retry_output_tokens']):
            raise ValueError('Unexpected unrecorded retry.')
        if record['first_input_tokens'] <= 0 or min(record['first_output_tokens'], record['retry_input_tokens'], record['retry_output_tokens']) < 0:
            raise ValueError('Invalid token count.')
        if retried and record['retry_input_tokens'] <= 0:
            raise ValueError('Missing retry token count.')
        if record['retry_prompt_plus_cap'] > 4096:
            raise ValueError('Recorded context bound exceeds limit.')
        if tokenizer:
            if (sha(wrap(prompt, False)) != record['wrapped_sha256'] or
                    len(wrap(prompt, True)) != record['first_input_tokens'] or
                    len(wrap(prompt + suffix, True)) + cap != record['retry_prompt_plus_cap'] or
                    (retried and len(wrap(prompt + suffix, True)) != record['retry_input_tokens'])):
                raise ValueError('Wrapped prompt hash, token count or context bound differs.')
        return first, final

    modes = ('specifications', 'original', 'stable_reverse', 'selective_pointwise', 'replay', 'pointwise')
    records = {m: lines(a.run / (m + '_responses.jsonl')) for m in modes}
    specs = []
    if len(records['specifications']) != len(rows):
        raise ValueError('Specification count differs.')
    for row, rec in zip(rows, records['specifications']):
        _, parsed = check(rec, row['query_id'], 0, len(row['items']), None, specprompt(row), 'spec')
        specs.append(parsed)
    vectors = {m: [] for m in ('original', 'stable_reverse', 'replay')}
    chunk_parsed = {m: {} for m in vectors}
    offsets = {m: 0 for m in vectors}
    for row, spec in zip(rows, specs):
        current = {m: [None] * len(row['items']) for m in vectors}
        for start in range(0, len(row['items']), 8):
            n = min(8, len(row['items']) - start)
            for mode in vectors:
                rec = records[mode][offsets[mode]]
                offsets[mode] += 1
                prompt, ids = keyprompt(row, start, spec, mode == 'stable_reverse')
                _, parsed = check(rec, row['query_id'], start, n, ids, prompt, 'keyed')
                chunk_parsed[mode][(row['query_id'], start)] = parsed
                if parsed is not None:
                    current[mode][start:start+n] = parsed
        for mode in vectors:
            vectors[mode].append(current[mode] if None not in current[mode] else None)
    if any(offsets[m] != len(records[m]) for m in offsets):
        raise ValueError('Extra keyed request records.')
    need = []
    full_wanted = []
    for row, spec in zip(rows, specs):
        for start in range(0, len(row['items']), 8):
            x = chunk_parsed['original'][(row['query_id'], start)]
            y = chunk_parsed['stable_reverse'][(row['query_id'], start)]
            if x is not None and y is not None:
                need.extend((row, start + i, spec) for i, (u, v) in enumerate(zip(x, y)) if u != v)
        full_wanted.extend((row, i, spec) for i in range(len(row['items'])))
    for mode, expected in (('selective_pointwise', need), ('pointwise', full_wanted)):
        if len(records[mode]) != len(expected):
            raise ValueError('Pointwise request count or routing differs.')
        for rec, (row, i, spec) in zip(records[mode], expected):
            check(rec, row['query_id'], i, 1, None, pointprompt(row, i, spec), 'point')
    selected = {(x['query_id'], x['start']): one(x['final']) for x in records['selective_pointwise']}
    full = {(x['query_id'], x['start']): one(x['final']) for x in records['pointwise']}
    vectors['pointwise'], vectors['actual_fallback'], vectors['full_trace_fallback'] = [], [], []
    for index, row in enumerate(rows):
        q = row['query_id']
        h = [full[(q, i)] for i in range(len(row['items']))]
        s = [selected.get((q, i)) for i in range(len(row['items']))]
        first, reverse = vectors['original'][index], vectors['stable_reverse'][index]
        def combine(fallback):
            if first is None or reverse is None:
                return None
            result = [x if x == y else z for x, y, z in zip(first, reverse, fallback)]
            return result if None not in result else None
        vectors['pointwise'].append(h if None not in h else None)
        vectors['actual_fallback'].append(combine(s))
        vectors['full_trace_fallback'].append(combine(h))
    saved = lines(a.run / 'selective_outputs.jsonl')
    if len(saved) != len(rows) or any(x != {'query_id': row['query_id'], 'pred': v}
                                      for x, row, v in zip(saved, rows, vectors['actual_fallback'])):
        raise ValueError('Saved actual fallback differs.')
    for mode in modes:
        rec = records[mode]
        observed = summary['modes'][mode]
        expected = {'initial_calls': len(rec), 'retry_calls': sum(x['retry_finish_reason'] is not None for x in rec),
                    'logical_tokens': {'input': sum(x['first_input_tokens'] + x['retry_input_tokens'] for x in rec),
                                       'output': sum(x['first_output_tokens'] + x['retry_output_tokens'] for x in rec)},
                    'max_retry_prompt_plus_cap': max((x['retry_prompt_plus_cap'] for x in rec), default=0)}
        if any(not close(v, observed[k]) for k, v in expected.items()):
            raise ValueError('Recorded request statistics differ for ' + mode)
    pipeline_modes = ('specifications', 'original', 'stable_reverse', 'selective_pointwise')
    for field in ('initial_calls', 'retry_calls'):
        if summary['selective_pipeline'][field] != sum(summary['modes'][m][field] for m in pipeline_modes):
            raise ValueError('Selective cost includes the wrong stages.')
    for field in ('input', 'output'):
        if summary['selective_pipeline']['logical_tokens'][field] != sum(summary['modes'][m]['logical_tokens'][field] for m in pipeline_modes):
            raise ValueError('Selective token total differs.')
    if summary['selective_pipeline']['complete_pools'] != sum(x is not None for x in vectors['actual_fallback']):
        raise ValueError('Selective completion differs.')
    vectors['always_Y'] = [['Y']*len(r['items']) for r in rows]
    vectors['always_N'] = [['N']*len(r['items']) for r in rows]
    vectors['lexical'] = []
    for row in rows:
        words = set(re.findall(r'\w+', row['query'].casefold()))
        vectors['lexical'].append(['Y' if words and words.issubset(set(re.findall(r'\w+', x['text'].casefold()))) else 'N' for x in row['items']])
    common = [i for i in range(len(rows)) if all(vectors[m][i] is not None for m in
              ('original', 'stable_reverse', 'replay', 'pointwise', 'actual_fallback', 'full_trace_fallback'))]
    if (analysis['common_query_ids'] != [rows[i]['query_id'] for i in common] or
            analysis['completion'] != {k: sum(x is not None for x in v) for k, v in vectors.items()} or
            analysis['specification_fallback_pools'] != sum(x is None for x in specs) or
            analysis['selective']['disagreeing_candidates'] != len(need)):
        raise ValueError('Completion or route analysis differs.')
    if common:
        for name, vec in vectors.items():
            if not close(quality(rows, vec, common), analysis['quality_common'][name]):
                raise ValueError('Quality mismatch for ' + name)
        import numpy as np
        differences = []
        for i in common:
            gold = sum(x['gold'] == 'Y' for x in rows[i]['items'])
            n = len(rows[i]['items'])
            old = abs(vectors['original'][i].count('Y') - gold) / n
            new = abs(vectors['actual_fallback'][i].count('Y') - gold) / n
            differences.append(new - old)
        draw = np.random.default_rng(20271202).integers(0, len(common), size=(4000, len(common)))
        interval = np.quantile(np.array(differences)[draw].mean(axis=1), [.025, .975]).tolist()
        primary = analysis['comparisons']['actual_fallback_minus_original']
        if (not close(float(np.mean(differences)), primary['ratio_mae_difference']) or
                not close(interval, primary['ratio_mae_difference_ci'])):
            raise ValueError('Primary paired query bootstrap differs.')
    print('Independently verified all prompts, IDs, retries, selective routes, completions, costs, quality and primary bootstrap.')
    print('Wrapped prompts and token counts checked:', bool(tokenizer))


if __name__ == '__main__':
    main()
