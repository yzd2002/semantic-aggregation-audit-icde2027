#!/usr/bin/env python3
"""Compare trace-derived lexical fallbacks on all attempted product pools."""
import argparse
import collections
import hashlib
import json
import re
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PROTOCOL = 'PROTOCOL_failure_lexical_fallback.md'
RUNS = {
    'esci_qwen': ('run_full_pools_250', 250, 20271041),
    'wands_qwen': ('run_wands_160', 160, 20271042),
    'esci_smollm': ('run_full_pools_smollm_250', 250, 20271043),
    'wands_smollm': ('run_wands_smollm_160', 160, 20271044),
}
METHOD = 'two_stage_compact_fallback'


def read(path):
    return [json.loads(x) for x in path.read_text(encoding='utf8').splitlines()]


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for x in iter(lambda: f.read(1024 * 1024), b''): h.update(x)
    return h.hexdigest()


def lexical(row):
    terms = set(re.findall(r'\w+', row['query'].casefold()))
    return ['Y' if terms and terms.issubset(set(re.findall(r'\w+', item['text'].casefold()))) else 'N'
            for item in row['items']]


def recover(size, raw):
    if type(size) is not int or size <= 0 or not isinstance(raw, list):
        raise ValueError('invalid pool or chunk list')
    result = [None] * size
    seen = set()
    for chunk in raw:
        start, length, text = chunk['start'], chunk['size'], chunk['final']
        if type(start) is not int or type(length) is not int or length <= 0 or start < 0 or start + length > size:
            raise ValueError('invalid chunk position')
        positions = set(range(start, start + length))
        if seen & positions: raise ValueError('overlapping chunks')
        seen.update(positions)
        if isinstance(text, str) and len(text) == length and set(text) <= {'Y', 'N'}:
            result[start:start + length] = list(text)
    if seen != set(range(size)): raise ValueError('chunks do not cover attempted pool')
    return result


def policies(row, plan):
    n = len(row['items'])
    partial = recover(n, plan['raw'])
    saved = plan['pred']
    complete = None not in partial
    if (saved is not None) != complete or complete and saved != partial:
        raise ValueError('saved compact prediction differs from strict final-chunk reconstruction')
    lex = lexical(row)
    return {'failed': not complete, 'recovered': partial, 'lexical': lex,
            'whole': list(saved) if complete else lex[:],
            'partial': [v if v is not None else lex[i] for i, v in enumerate(partial)]}


def measure(golds, predictions):
    tp = tn = fp = fn = 0
    errors = []
    for gold, pred in zip(golds, predictions):
        if not gold or len(gold) != len(pred) or set(gold + pred) - {'Y', 'N'}:
            raise ValueError('invalid labels')
        tp += sum(g == 'Y' and p == 'Y' for g, p in zip(gold, pred))
        tn += sum(g == 'N' and p == 'N' for g, p in zip(gold, pred))
        fp += sum(g == 'N' and p == 'Y' for g, p in zip(gold, pred))
        fn += sum(g == 'Y' and p == 'N' for g, p in zip(gold, pred))
        errors.append(abs(pred.count('Y') / len(pred) - gold.count('Y') / len(gold)))
    return {'candidates': tp + tn + fp + fn, 'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
            'f1': 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.,
            'balanced_accuracy': (tp / (tp + fn) + tn / (tn + fp)) / 2 if tp + fn and tn + fp else None,
            'ratio_mae': float(np.mean(errors))}, np.array(errors, dtype=float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    result = {'protocol': PROTOCOL, 'analysis_is_post_hoc': True, 'datasets': {}}
    detail = []
    for name, (directory, expected, seed) in RUNS.items():
        src = HERE / directory
        rows, outputs = read(src / 'inputs.jsonl'), read(src / 'outputs.jsonl')
        saved_summary = json.loads((src / 'summary.json').read_text())
        if len(rows) != len(outputs) or len(rows) != expected:
            raise ValueError('unexpected number of pools')
        golds, predictions = [], {k: [] for k in ('pointwise', 'lexical', 'whole', 'partial', 'all_yes', 'all_no')}
        source_counts = {'whole': collections.Counter(), 'partial': collections.Counter()}
        failures = 0
        for row, out in zip(rows, outputs):
            if row['query_id'] != out['query_id'] or row['query'] != out['query']:
                raise ValueError('query association mismatch')
            gold = [x['gold'] for x in row['items']]
            n = len(gold)
            point = out['plans']['pointwise']['pred']
            raw_point = [x['final'] if x['final'] in ('Y', 'N') else None
                         for x in out['plans']['pointwise']['raw']]
            if point is None or point != raw_point or len(point) != n:
                raise ValueError('incomplete or inconsistent pointwise baseline')
            p = policies(row, out['plans'][METHOD]); failures += p['failed']
            accepted = sum(x is not None for x in p['recovered'])
            source_counts['whole']['lexical'] += n if p['failed'] else 0
            source_counts['whole']['model'] += 0 if p['failed'] else n
            source_counts['partial']['model'] += accepted
            source_counts['partial']['lexical'] += n - accepted
            choices = {'pointwise': point, 'lexical': p['lexical'], 'whole': p['whole'], 'partial': p['partial'],
                       'all_yes': ['Y'] * n, 'all_no': ['N'] * n}
            golds.append(gold)
            for k, v in choices.items(): predictions[k].append(v)
            detail.append({'dataset': name, 'query_id': row['query_id'], 'n': n, 'failed': p['failed'],
                           'recovered': p['recovered'], 'lexical': p['lexical'], 'whole': p['whole'],
                           'partial': p['partial'], 'pointwise': point})
        if expected - failures != saved_summary['plans'][METHOD]['complete_pools']:
            raise ValueError('saved compact completion count mismatch')
        stats, errors = {}, {}
        for k, v in predictions.items():stats[k], errors[k] = measure(golds, v)
        draws = np.random.default_rng(seed).integers(0, expected, size=(4000, expected))
        comparisons = {}
        for left, right in (('whole', 'pointwise'), ('partial', 'pointwise'), ('partial', 'whole')):
            delta = errors[left] - errors[right]
            boots = delta[draws].mean(axis=1)
            comparisons[left + '_minus_' + right] = {'ratio_mae_difference': float(delta.mean()),
                'query_bootstrap_95': [float(x) for x in np.percentile(boots, [2.5, 97.5])]}
        result['datasets'][name] = {'source': directory, 'source_sha256': {
                k: sha(src / k) for k in ('inputs.jsonl', 'outputs.jsonl', 'summary.json')},
            'attempted_pools': expected, 'compact_complete_pools': expected - failures,
            'failed_pools': failures, 'source_counts': {k: dict(v) for k, v in source_counts.items()},
            'metrics': stats, 'comparisons': comparisons, 'bootstrap_seed': seed, 'bootstrap_draws': 4000}
        print(name, 'attempted', expected, 'failures', failures,
              'ratio MAE (pointwise, lexical, whole, partial):',
              [round(stats[k]['ratio_mae'], 4) for k in ('pointwise', 'lexical', 'whole', 'partial')])
    (a.out / 'details.jsonl').write_text(''.join(json.dumps(x, ensure_ascii=False) + '\n' for x in detail), encoding='utf8')
    (a.out / 'summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf8')


if __name__ == '__main__':
    main()
