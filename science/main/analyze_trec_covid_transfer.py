#!/usr/bin/env python3
"""Compute the fixed TREC-COVID transfer outcomes from saved model responses."""
import argparse
import collections
import json
import re
from pathlib import Path

import numpy as np
from run_keyed_order_audit import parse_keyed

IDS = list(range(1, 9))


def lines(path):
    return [json.loads(x) for x in path.read_text(encoding='utf8').splitlines()]


def keyed_vector(topic, records):
    values = []
    for start in range(0, 32, 8):
        parsed = parse_keyed(records[(topic, start)]['final'], IDS)
        if parsed is None:
            return None
        values.extend(parsed[i] for i in IDS)
    return values


def point_vector(topic, records):
    values = [records[(topic, i)]['final'] for i in range(32)]
    return values if set(values) <= {'Y', 'N'} else None


def index(records):
    d = {(x['topic_id'], x['start']): x for x in records}
    if len(d) != len(records):
        raise ValueError('duplicate response key')
    return d


def lexical(row):
    words = re.findall(r'[a-z0-9]+', row['query'].lower())
    if not words:
        raise ValueError('empty lexical query')
    return ['Y' if all(w in set(re.findall(r'[a-z0-9]+', (x['title'] + ' ' + x['abstract']).lower()))
                        for w in words) else 'N' for x in row['items']]


def metrics(rows, predictions):
    tp = tn = fp = fn = 0
    errors = []
    for row, pred in zip(rows, predictions):
        gold = [x['gold'] for x in row['items']]
        if pred is None or len(pred) != len(gold):
            raise ValueError('metrics require complete aligned pools')
        for g, p in zip(gold, pred):
            if g == 'Y' and p == 'Y': tp += 1
            elif g == 'N' and p == 'N': tn += 1
            elif g == 'N' and p == 'Y': fp += 1
            elif g == 'Y' and p == 'N': fn += 1
            else: raise ValueError('nonbinary label')
        errors.append(abs(pred.count('Y') / len(pred) - gold.count('Y') / len(gold)))
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    bal = (tp / (tp + fn) + tn / (tn + fp)) / 2 if tp + fn and tn + fp else None
    return {'queries': len(rows), 'candidates': tp + tn + fp + fn,
            'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn, 'f1': f1,
            'balanced_accuracy': bal, 'ratio_mae': float(np.mean(errors))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--selection', type=Path, required=True)
    ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    rows = lines(a.selection / 'inputs.jsonl')
    mode_names = ('original', 'stable_reverse', 'replay', 'pointwise', 'selective_pointwise')
    records = {name: index(lines(a.run / (name + '_responses.jsonl'))) for name in mode_names}
    selected = {x['topic_id']: x['pred'] for x in lines(a.run / 'selective_outputs.jsonl')}
    if len(selected) != len(rows):
        raise ValueError('unexpected selective output count')
    vectors = {}
    for row in rows:
        tid = row['topic_id']
        original = keyed_vector(tid, records['original'])
        reverse = keyed_vector(tid, records['stable_reverse'])
        replay = keyed_vector(tid, records['replay'])
        full = point_vector(tid, records['pointwise'])
        sent = selected[tid]
        if sent is not None and (len(sent) != 32 or set(sent) - {'Y', 'N'}):
            raise ValueError('invalid saved selective vector')
        vectors[tid] = {'original': original, 'stable_reverse': reverse, 'replay': replay,
                        'pointwise': full, 'selective': sent}
    completion = {name: sum(v[name] is not None for v in vectors.values())
                  for name in ('original', 'stable_reverse', 'replay', 'pointwise', 'selective')}
    common = [r for r in rows if all(x is not None for x in vectors[r['topic_id']].values())]
    if not common:
        raise RuntimeError('no common-complete pools: no quality comparison is defined')
    preds = {name: [vectors[r['topic_id']][name] for r in common]
             for name in ('original', 'stable_reverse', 'replay', 'pointwise', 'selective')}
    preds['all_negative'] = [['N'] * 32 for _ in common]
    preds['all_positive'] = [['Y'] * 32 for _ in common]
    preds['all_query_words'] = [lexical(r) for r in common]
    result = {'topics_attempted': len(rows), 'completion': completion,
              'common_complete_topics': len(common), 'common_topic_ids': [r['topic_id'] for r in common],
              'metrics': {name: metrics(common, pred) for name, pred in preds.items()}}
    errors = []
    for row, orig, sel in zip(common, preds['original'], preds['selective']):
        g = sum(x['gold'] == 'Y' for x in row['items']) / 32
        errors.append(abs(sel.count('Y') / 32 - g) - abs(orig.count('Y') / 32 - g))
    rng = np.random.default_rng(20271032)
    err = np.array(errors, dtype=float)
    draws = err[rng.integers(0, len(err), size=(4000, len(err)))].mean(axis=1)
    result['primary_difference'] = {'selective_minus_original_ratio_mae': float(err.mean()),
        'query_bootstrap_95': [float(x) for x in np.percentile(draws, [2.5, 97.5])], 'draws': 4000}
    for other in ('stable_reverse', 'replay'):
        changed = sum(x != y for orig, alt in zip(preds['original'], preds[other]) for x, y in zip(orig, alt))
        result[other + '_changes'] = {'changed': changed, 'total': len(common) * 32,
                                       'fraction': changed / (len(common) * 32)}
    match_count = match_changes = 0
    for row, orig, rev, full in zip(common, preds['original'], preds['stable_reverse'], preds['pointwise']):
        for i, (x, y, f) in enumerate(zip(orig, rev, full)):
            if x != y:
                actual = records['selective_pointwise'][(row['topic_id'], i)]['final']
                match_count += 1
                match_changes += actual != f
    result['selective_vs_later_full_pointwise'] = {'matched_disagreements': match_count,
                                                   'label_changes': match_changes}

    documents = sorted({i['doc_id'] for row in common for i in row['items']})
    doc_index = {doc: i for i, doc in enumerate(documents)}
    doc_ids = np.array([[doc_index[x['doc_id']] for x in row['items']] for row in common])
    gold = np.array([[x['gold'] == 'Y' for x in row['items']] for row in common], dtype=float)
    orig = np.array([[x == 'Y' for x in v] for v in preds['original']], dtype=float)
    sel = np.array([[x == 'Y' for x in v] for v in preds['selective']], dtype=float)
    rng = np.random.default_rng(20271033)
    crossed = []
    for _ in range(4000):
        qdraw = rng.integers(0, len(common), len(common))
        ddraw = rng.integers(0, len(documents), len(documents))
        weights = np.bincount(ddraw, minlength=len(documents))[doc_ids]
        denominator = weights.sum(axis=1)
        if np.any(denominator[qdraw] == 0):
            raise RuntimeError('zero document weight for a sampled topic')
        valid = denominator > 0
        gold_ratio = np.zeros(len(common)); orig_ratio = gold_ratio.copy(); sel_ratio = gold_ratio.copy()
        gold_ratio[valid] = (gold * weights).sum(axis=1)[valid] / denominator[valid]
        orig_ratio[valid] = (orig * weights).sum(axis=1)[valid] / denominator[valid]
        sel_ratio[valid] = (sel * weights).sum(axis=1)[valid] / denominator[valid]
        delta = np.abs(sel_ratio - gold_ratio) - np.abs(orig_ratio - gold_ratio)
        crossed.append(float(np.mean(delta[qdraw])))
    result['crossed_sensitivity'] = {'distinct_documents': len(documents), 'draws': 4000,
        'interval_95': [float(x) for x in np.percentile(crossed, [2.5, 97.5])]}
    result['limitations'] = ('Strict fully relevant binary reference on sampled judged documents; '
        'truncated abstracts, public model/data, repeated documents and domain-adapted prompts. '
        'Crossed sensitivity reweights observed pools and is not exact fixed-pool coverage.')
    a.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    print('Common-complete topics:', len(common))
    print('Primary selective minus original ratio MAE:', result['primary_difference'])
    print('Crossed sensitivity:', result['crossed_sensitivity'])
    print('Original, selective, full pointwise, lexical MAE:',
          [result['metrics'][x]['ratio_mae'] for x in ('original', 'selective', 'pointwise', 'all_query_words')])


if __name__ == '__main__':
    main()
