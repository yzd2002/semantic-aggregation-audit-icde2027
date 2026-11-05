#!/usr/bin/env python3
"""Independently reconstruct TREC-COVID prompts, decisions and reported metrics."""
import argparse
import collections
import hashlib
import json
import re
from fractions import Fraction
from pathlib import Path

import numpy as np

SYSTEM = ('Assess scientific document relevance to the stated search topic using only the supplied '
          'title and abstract. Mark Y only if the document is fully responsive and can answer the '
          'question on its own; mark N for partial, unrelated or insufficient information. '
          'Use the narrative as context. Give only the requested labels.')
SUFFIX = '\nYour previous answer had the wrong format; follow the output instructions exactly.'


def read(p):
    return [json.loads(s) for s in p.read_text(encoding='utf8').splitlines()]


def sha(s):
    return hashlib.sha256(s.encode('utf8')).hexdigest()


def parse_json(s):
    if re.fullmatch(r'\{"1":"[YN]","2":"[YN]","3":"[YN]","4":"[YN]",'
                    r'"5":"[YN]","6":"[YN]","7":"[YN]","8":"[YN]"\}', s) is None:
        return None
    obj = json.loads(s)
    return [obj[str(i)] for i in range(1, 9)]


def parse_single(s):
    return s if s in ('Y', 'N') else None


def prefix(r):
    return 'Search query: ' + r['query'] + '\nQuestion: ' + r['question'] + '\nNarrative: ' + r['narrative'] + '\n'


def document(r, i):
    x = r['items'][i]
    return {'title': x['title'], 'abstract': x['abstract']}


def chunk_prompt(r, start, reverse):
    items = [{'id': i + 1, **document(r, start + i)} for i in range(8)]
    if reverse: items.reverse()
    p = prefix(r) + 'Documents in order:\n' + '\n'.join(json.dumps(x, ensure_ascii=False) for x in items)
    p += ('\nReturn only a JSON object mapping each given document id to Y or N for that document. '
          'Use the given ids, not positions; include every id once, in increasing numeric order. '
          'Do not add spaces or explanations.')
    return p, [x['id'] for x in items]


def single_prompt(r, i):
    return prefix(r) + 'Document: ' + json.dumps(document(r, i), ensure_ascii=False) + '\nAnswer ONLY Y or N.'


def idx(records):
    out = {(r['topic_id'], r['start']): r for r in records}
    assert len(out) == len(records)
    return out


def near(a, b):
    if isinstance(a, dict): return a.keys() == b.keys() and all(near(a[k], b[k]) for k in a)
    if isinstance(a, list): return len(a) == len(b) and all(near(x, y) for x, y in zip(a, b))
    if a is None or b is None: return a is None and b is None
    if isinstance(a, (int, float)) and isinstance(b, (int, float)): return abs(a - b) < 1e-12
    return a == b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--selection', type=Path, required=True)
    ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--analysis', type=Path, required=True)
    ap.add_argument('--model', type=Path, help='optional local tokenizer for independent token checks')
    a = ap.parse_args()
    rows = read(a.selection / 'inputs.jsonl')
    rmap = {r['topic_id']: r for r in rows}
    assert len(rmap) == len(rows)
    summary = json.loads((a.run / 'summary.json').read_text())
    analysis = json.loads(a.analysis.read_text())
    assert summary['system'] == SYSTEM and summary['retry_suffix'] == SUFFIX
    assert summary['selection_inputs_sha256'] == hashlib.sha256((a.selection / 'inputs.jsonl').read_bytes()).hexdigest()
    controls = json.loads((a.run / 'guidance_sanity.json').read_text())
    assert parse_json(controls['ordinary']['text']) is None
    assert parse_single(controls['ordinary']['text']) is None
    assert parse_json(controls['keyed']['text']) is not None
    assert parse_single(controls['pointwise']['text']) is not None
    assert all(controls['checks'].values())
    tokenizer = None
    if a.model:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(a.model), local_files_only=True, trust_remote_code=False)
    raw = {mode: read(a.run / (mode + '_responses.jsonl')) for mode in
           ('original', 'stable_reverse', 'selective_pointwise', 'replay', 'pointwise')}
    records = {k: idx(v) for k, v in raw.items()}
    expected_chunks = {(r['topic_id'], start) for r in rows for start in range(0, 32, 8)}
    expected_points = {(r['topic_id'], i) for r in rows for i in range(32)}
    for mode in ('original', 'stable_reverse', 'replay'): assert set(records[mode]) == expected_chunks
    assert set(records['pointwise']) == expected_points
    for mode, recs in raw.items():
        keyed = mode in ('original', 'stable_reverse', 'replay')
        parse = parse_json if keyed else parse_single
        for x in recs:
            r = rmap[x['topic_id']]
            if keyed:
                p, present = chunk_prompt(r, x['start'], mode == 'stable_reverse')
                assert x['size'] == 8 and x['presented_ids'] == present
            else:
                p = single_prompt(r, x['start'])
                assert x['size'] == 1 and x['presented_ids'] is None
            assert sha(p) == x['prompt_sha256']
            invalid = parse(x['first']) is None
            assert (x['retry_finish_reason'] is not None) == invalid
            if not invalid:
                assert x['first'] == x['final'] and x['retry_input_tokens'] == x['retry_output_tokens'] == 0
            else:
                assert x['retry_input_tokens'] > 0
            if tokenizer:
                def wrap(text, tokenize=False):
                    return tokenizer.apply_chat_template([{'role': 'system', 'content': SYSTEM},
                        {'role': 'user', 'content': text}], tokenize=tokenize, add_generation_prompt=True)
                assert sha(wrap(p)) == x['wrapped_sha256']
                assert len(wrap(p, True)) == x['first_input_tokens']
                if invalid: assert len(wrap(p + SUFFIX, True)) == x['retry_input_tokens']
        stats = summary['modes'][mode]
        assert stats['initial_calls'] == len(recs)
        assert stats['retry_calls'] == sum(parse(x['first']) is None for x in recs)
        assert stats['valid_first'] == sum(parse(x['first']) is not None for x in recs)
        assert stats['valid_final'] == sum(parse(x['final']) is not None for x in recs)
        assert stats['first_finish_reasons'] == dict(collections.Counter(x['first_finish_reason'] for x in recs))
        for kind in ('input', 'output'):
            assert stats['logical_tokens'][kind] == sum(x['first_' + kind + '_tokens'] +
                x['retry_' + kind + '_tokens'] for x in recs)
        assert stats['max_prompt_with_retry_plus_cap'] <= 4096
    expected_selected = set()
    rebuilt = {}
    for r in rows:
        tid = r['topic_id']; v = {}
        for mode in ('original', 'stable_reverse', 'replay'):
            chunks = [parse_json(records[mode][(tid, s)]['final']) for s in range(0, 32, 8)]
            v[mode] = None if any(c is None for c in chunks) else sum(chunks, [])
        for start in range(0, 32, 8):
            orig = parse_json(records['original'][(tid, start)]['final'])
            rev = parse_json(records['stable_reverse'][(tid, start)]['final'])
            if orig is not None and rev is not None:
                expected_selected.update((tid, start + i) for i in range(8) if orig[i] != rev[i])
        points = [parse_single(records['pointwise'][(tid, i)]['final']) for i in range(32)]
        v['pointwise'] = points if None not in points else None
        if v['original'] is None or v['stable_reverse'] is None:
            v['selective'] = None
        else:
            out = []
            for i, (x, y) in enumerate(zip(v['original'], v['stable_reverse'])):
                out.append(x if x == y else parse_single(records['selective_pointwise'][(tid, i)]['final']))
            v['selective'] = out if None not in out else None
        rebuilt[tid] = v
    assert set(records['selective_pointwise']) == expected_selected
    saved = read(a.run / 'selective_outputs.jsonl')
    assert saved == [{'topic_id': r['topic_id'], 'pred': rebuilt[r['topic_id']]['selective']} for r in rows]
    plan = summary['selective_pipeline']
    assert plan['complete_pools'] == sum(v['selective'] is not None for v in rebuilt.values())
    for key in ('initial_calls', 'retry_calls'):
        assert plan[key] == sum(summary['modes'][m][key] for m in ('original', 'stable_reverse', 'selective_pointwise'))
    for kind in ('input', 'output'):
        assert plan['logical_tokens'][kind] == sum(summary['modes'][m]['logical_tokens'][kind]
            for m in ('original', 'stable_reverse', 'selective_pointwise'))
    names = ('original', 'stable_reverse', 'replay', 'pointwise', 'selective')
    completion = {name: sum(v[name] is not None for v in rebuilt.values()) for name in names}
    assert analysis['topics_attempted'] == len(rows) and analysis['completion'] == completion
    common = [r for r in rows if all(rebuilt[r['topic_id']][name] is not None for name in names)]
    assert analysis['common_topic_ids'] == [r['topic_id'] for r in common]
    assert analysis['common_complete_topics'] == len(common) and common
    prediction = {name: [rebuilt[r['topic_id']][name] for r in common] for name in names}
    prediction['all_negative'] = [['N'] * 32 for _ in common]
    prediction['all_positive'] = [['Y'] * 32 for _ in common]
    prediction['all_query_words'] = []
    for r in common:
        terms = re.findall('[a-z0-9]+', r['query'].lower())
        prediction['all_query_words'].append(['Y' if all(w in re.findall('[a-z0-9]+',
            (x['title'] + ' ' + x['abstract']).lower()) for w in terms) else 'N' for x in r['items']])
    for name, all_predictions in prediction.items():
        pairs = [(i['gold'], p) for r, preds in zip(common, all_predictions) for i, p in zip(r['items'], preds)]
        tp = sum(g == 'Y' and p == 'Y' for g, p in pairs)
        tn = sum(g == 'N' and p == 'N' for g, p in pairs)
        fp = sum(g == 'N' and p == 'Y' for g, p in pairs)
        fn = sum(g == 'Y' and p == 'N' for g, p in pairs)
        maes = [abs(Fraction(sum(p == 'Y' for p in preds) - sum(i['gold'] == 'Y' for i in r['items']), 32))
                for r, preds in zip(common, all_predictions)]
        m = {'queries': len(common), 'candidates': len(pairs), 'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
             'f1': 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
             'balanced_accuracy': (tp / (tp + fn) + tn / (tn + fp)) / 2 if tp + fn and tn + fp else None,
             'ratio_mae': float(sum(maes) / len(maes))}
        assert near(analysis['metrics'][name], m)
    deltas = []
    for r, orig, sel in zip(common, prediction['original'], prediction['selective']):
        g = sum(x['gold'] == 'Y' for x in r['items'])
        deltas.append(float(Fraction(abs(sel.count('Y') - g) - abs(orig.count('Y') - g), 32)))
    rng = np.random.default_rng(20271032); values = np.array(deltas)
    boots = values[rng.integers(0, len(values), size=(4000, len(values)))].mean(axis=1)
    assert near(analysis['primary_difference'], {'selective_minus_original_ratio_mae': float(values.mean()),
        'query_bootstrap_95': [float(x) for x in np.percentile(boots, [2.5, 97.5])], 'draws': 4000})
    for mode in ('stable_reverse', 'replay'):
        changes = sum(x != y for orig, alt in zip(prediction['original'], prediction[mode]) for x, y in zip(orig, alt))
        assert near(analysis[mode + '_changes'], {'changed': changes, 'total': len(common) * 32,
                                                  'fraction': changes / (len(common) * 32)})
    changes = total = 0
    for r, orig, rev, full in zip(common, prediction['original'], prediction['stable_reverse'], prediction['pointwise']):
        for i, (x, y, z) in enumerate(zip(orig, rev, full)):
            if x != y:
                total += 1
                changes += records['selective_pointwise'][(r['topic_id'], i)]['final'] != z
    assert analysis['selective_vs_later_full_pointwise'] == {'matched_disagreements': total, 'label_changes': changes}

    docs = sorted({x['doc_id'] for r in common for x in r['items']}); dindex = {x: i for i, x in enumerate(docs)}
    dids = np.array([[dindex[x['doc_id']] for x in r['items']] for r in common])
    gold = np.array([[int(x['gold'] == 'Y') for x in r['items']] for r in common])
    orig = np.array([[int(x == 'Y') for x in p] for p in prediction['original']])
    sel = np.array([[int(x == 'Y') for x in p] for p in prediction['selective']])
    rng = np.random.default_rng(20271033); crossed = []
    for _ in range(4000):
        qsample = rng.integers(0, len(common), len(common))
        dsample = rng.integers(0, len(docs), len(docs))
        counts = np.bincount(dsample, minlength=len(docs)); diffs = []
        for q in qsample:
            weight = counts[dids[q]]; n = int(weight.sum()); assert n > 0
            g, o, s = (int(np.dot(v[q], weight)) for v in (gold, orig, sel))
            diffs.append((abs(s - g) - abs(o - g)) / n)
        crossed.append(float(np.mean(diffs)))
    assert near(analysis['crossed_sensitivity'], {'distinct_documents': len(docs), 'draws': 4000,
        'interval_95': [float(x) for x in np.percentile(crossed, [2.5, 97.5])]})
    print('Prompts, response associations, retries, selective routing, counts, metrics and both resampling analyses independently verified.')
    if tokenizer: print('Wrapped prompt hashes and all input-token lengths independently verified.')


if __name__ == '__main__':
    main()
