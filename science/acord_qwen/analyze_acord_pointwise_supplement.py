#!/usr/bin/env python3
"""Compare a preplanned pointwise diagnostic on pools complete for all plans.

Written during inference after detecting an omission in the original analyzer;
it does not change the pre-inference runner or its primary analysis.
"""
import argparse, hashlib, json, math, random, re
from pathlib import Path


def lines(path):
    with path.open(encoding='utf8') as f:
        return [json.loads(line) for line in f if line.strip()]


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def keyed(text):
    grammar = r'\{' + ','.join('"' + str(i) + '":"[YN]"' for i in range(1, 9)) + r'\}'
    if not isinstance(text, str) or re.fullmatch(grammar, text) is None:
        return None
    try:
        obj = json.loads(text, object_pairs_hook=lambda pairs: pairs)
    except (TypeError, ValueError):
        return None
    if (not isinstance(obj, list) or len(obj) != 8
            or any(not isinstance(pair, (list, tuple)) or len(pair) != 2 for pair in obj)):
        return None
    keys = [k for k, _ in obj]
    if len(set(keys)) != 8 or set(keys) != {str(i) for i in range(1, 9)}:
        return None
    result = dict(obj)
    if any(result[str(i)] not in ('Y', 'N') for i in range(1, 9)):
        return None
    return [result[str(i)] for i in range(1, 9)]


def score(golds, predictions):
    tp = fp = fn = tn = 0
    errors = []
    for gold, pred in zip(golds, predictions):
        assert len(gold) == len(pred) == 32
        errors.append(abs(gold.count('Y') - pred.count('Y')) / 32)
        for g, p in zip(gold, pred):
            tp += g == p == 'Y'; tn += g == p == 'N'
            fp += g == 'N' and p == 'Y'; fn += g == 'Y' and p == 'N'
    denom = 2 * tp + fp + fn
    return {'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
            'f1': 2 * tp / denom if denom else 0.0,
            'balanced_accuracy': (tp / (tp + fn) + tn / (tn + fp)) / 2 if tp + fn and tn + fp else None,
            'ratio_mae': sum(errors) / len(errors)}


def percentile(values, q):
    a = sorted(values); index = (len(a) - 1) * q
    lo = math.floor(index); hi = math.ceil(index)
    return a[lo] + (index - lo) * (a[hi] - a[lo])


def main(selection, run, out):
    if out.exists():
        raise FileExistsError('Choose a new output file')
    rows = lines(selection / 'inputs.jsonl')
    meta = json.loads((selection / 'selection.json').read_text())
    summary = json.loads((run / 'summary.json').read_text())
    assert summary['completed'] and sha(selection / 'inputs.jsonl') == meta['inputs_sha256'] == summary['selection_inputs_sha256']
    groups = {mode: lines(run / (mode + '_responses.jsonl')) for mode in ('original', 'stable_reverse', 'replay', 'pointwise')}
    index = {}
    for mode, records in groups.items():
        index[mode] = {(r['pool_id'], r['start']): r for r in records}
        assert len(index[mode]) == len(records)
    selective = {r['pool_id']: r['pred'] for r in lines(run / 'selective_outputs.jsonl')}
    assert len(selective) == len(rows)
    result = {'scope': 'Preplanned full pointwise diagnostic; independent supplementary analysis prepared during inference.',
              'selection_sha256': sha(selection / 'inputs.jsonl'), 'run_summary_sha256': sha(run / 'summary.json'),
              'script_sha256': sha(Path(__file__)), 'resamples': 4000, 'scenarios': {}}
    for scenario, seed in (('case_control', 20261103), ('uniform', 20261104)):
        chosen = [r for r in rows if r['scenario'] == scenario]
        completed = []
        reversal_changes = replay_changes = replay_denominator = reversal_denominator = 0
        for row in chosen:
            pool = row['pool_id']
            vectors = {}
            for mode in ('original', 'stable_reverse', 'replay'):
                chunks = [keyed(index[mode][pool, start]['final']) for start in (0, 8, 16, 24)]
                vectors[mode] = sum(chunks, []) if all(x is not None for x in chunks) else None
            points = [index['pointwise'][pool, j]['final'] for j in range(32)]
            vectors['pointwise'] = points if all(x in ('Y', 'N') for x in points) else None
            vectors['selective'] = selective[pool]
            if vectors['selective'] is not None:
                assert len(vectors['selective']) == 32 and all(x in ('Y', 'N') for x in vectors['selective'])
            if vectors['original'] is not None and vectors['stable_reverse'] is not None:
                reversal_changes += sum(a != b for a, b in zip(vectors['original'], vectors['stable_reverse']))
                reversal_denominator += 32
            if vectors['original'] is not None and vectors['replay'] is not None:
                replay_changes += sum(a != b for a, b in zip(vectors['original'], vectors['replay']))
                replay_denominator += 32
            if all(vectors[name] is not None for name in ('original', 'stable_reverse', 'selective', 'pointwise')):
                completed.append((row, vectors))
        if not completed:
            result['scenarios'][scenario] = {'attempted': len(chosen), 'joint_complete': 0}
            continue
        gold = [[item['gold'] for item in row['items']] for row, _ in completed]
        plans = {name: [v[name] for _, v in completed] for name in ('original', 'stable_reverse', 'selective', 'pointwise')}
        plans['always_N'] = [['N'] * 32 for _ in completed]
        metrics = {name: score(gold, values) for name, values in plans.items()}
        delta = [abs(g.count('Y') - s.count('Y')) / 32 - abs(g.count('Y') - p.count('Y')) / 32
                 for g, s, p in zip(gold, plans['selective'], plans['pointwise'])]
        rng = random.Random(seed); n = len(delta)
        sampled = [sum(delta[rng.randrange(n)] for _ in range(n)) / n for _ in range(4000)]
        result['scenarios'][scenario] = {'attempted': len(chosen), 'joint_complete': len(completed),
            'metrics': metrics, 'selective_minus_pointwise_ratio_mae': sum(delta) / n,
            'query_bootstrap_95': [percentile(sampled, .025), percentile(sampled, .975)],
            'query_bootstrap_seed': seed,
            'stable_reversal_changes': reversal_changes, 'stable_reversal_denominator': reversal_denominator,
            'identical_replay_changes': replay_changes, 'identical_replay_denominator': replay_denominator,
            'warning': 'Unadjusted descriptive paired interval; no population or deployment guarantee.'}
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n', encoding='utf8')
    print('Full-pointwise supplement complete; compared only jointly complete pools. No inference rerun.')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--selection', type=Path, required=True)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    main(a.selection, a.run, a.out)
