#!/usr/bin/env python3
"""Independently check the pointwise supplement against raw saved responses."""
import argparse, json, math, random
from pathlib import Path
from acord_prompts import parse_keyed


def read(path):
    with path.open(encoding='utf8') as f:
        return [json.loads(line) for line in f if line.strip()]


def close(a, b):
    return math.isclose(a, b, rel_tol=0, abs_tol=1e-12)


def main(selection, run, primary_path, supplement_path):
    rows = read(selection / 'inputs.jsonl')
    primary = json.loads(primary_path.read_text())
    supplement = json.loads(supplement_path.read_text())
    keyed = {}
    for mode in ('original', 'stable_reverse', 'replay'):
        keyed[mode] = {(x['pool_id'], x['start']): parse_keyed(x['final']) for x in read(run / (mode + '_responses.jsonl'))}
    point = {(x['pool_id'], x['start']): x['final'] for x in read(run / 'pointwise_responses.jsonl')}
    selected = {x['pool_id']: x['pred'] for x in read(run / 'selective_outputs.jsonl')}
    for scenario in ('case_control', 'uniform'):
        sample = [r for r in rows if r['scenario'] == scenario]
        supplemental = supplement['scenarios'][scenario]; main_result = primary['scenarios'][scenario]
        valid = []
        replay_changes = reverse_changes = replay_n = reverse_n = 0
        for row in sample:
            name = row['pool_id']; vectors = {}
            for mode in keyed:
                chunks = [keyed[mode][name, n] for n in (0, 8, 16, 24)]
                vectors[mode] = [chunk[j] for chunk in chunks for j in range(1, 9)] if all(x is not None for x in chunks) else None
            full = [point[name, i] for i in range(32)]
            vectors['pointwise'] = full if all(x in ('Y', 'N') for x in full) else None
            vectors['selective'] = selected[name]
            if vectors['original'] is not None and vectors['stable_reverse'] is not None:
                reverse_changes += sum(x != y for x, y in zip(vectors['original'], vectors['stable_reverse']));reverse_n += 32
            if vectors['original'] is not None and vectors['replay'] is not None:
                replay_changes += sum(x != y for x, y in zip(vectors['original'], vectors['replay']));replay_n += 32
            if all(vectors[x] is not None for x in ('original', 'stable_reverse', 'selective', 'pointwise')):
                valid.append((row, vectors))
        assert len(sample) == supplemental['attempted'] and len(valid) == supplemental['joint_complete']
        assert (reverse_changes, reverse_n, replay_changes, replay_n) == (
            supplemental['stable_reversal_changes'], supplemental['stable_reversal_denominator'],
            supplemental['identical_replay_changes'], supplemental['identical_replay_denominator'])
        errors = {k: [] for k in ('original', 'stable_reverse', 'selective', 'pointwise', 'always_N')}
        confusion = {k: [0, 0, 0, 0] for k in errors}
        for row, vec in valid:
            gold = [x['gold'] for x in row['items']]
            predictions = {'original': vec['original'], 'stable_reverse': vec['stable_reverse'],
                           'selective': vec['selective'], 'pointwise': vec['pointwise'], 'always_N': ['N'] * 32}
            for mode, output in predictions.items():
                errors[mode].append(abs(sum(x == 'Y' for x in gold) - sum(x == 'Y' for x in output)) / 32)
                for g, p in zip(gold, output):
                    idx = 0 if (g, p) == ('Y', 'Y') else 1 if (g, p) == ('N', 'Y') else 2 if (g, p) == ('Y', 'N') else 3
                    confusion[mode][idx] += 1
        names = {'original': 'original', 'stable_reverse': 'stable_reverse', 'selective': 'selective',
                 'pointwise': 'pointwise', 'always_N': 'always_N'}
        for mode, label in names.items():
            m = supplemental['metrics'][label];tp, fp, fn, tn = confusion[mode]
            assert [m[k] for k in ('tp', 'fp', 'fn', 'tn')] == confusion[mode]
            assert close(m['ratio_mae'], sum(errors[mode]) / len(valid))
            assert close(m['f1'], 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0)
            if tp + fn and tn + fp:
                assert close(m['balanced_accuracy'], .5 * (tp / (tp + fn) + tn / (tn + fp)))
        if len(valid) == main_result['common_complete_pools']:
            for mode in ('original', 'selective', 'always_N'):
                assert close(supplemental['metrics'][mode]['ratio_mae'], main_result['metrics'][mode]['ratio_mae'])
        delta = [a - b for a, b in zip(errors['selective'], errors['pointwise'])]
        rng = random.Random(supplemental['query_bootstrap_seed']);n = len(delta)
        boot = sorted(sum(delta[rng.randrange(n)] for _ in range(n)) / n for _ in range(4000))
        def quantile(q):
            i = (len(boot) - 1) * q;l = math.floor(i);h = math.ceil(i)
            return boot[l] + (i - l) * (boot[h] - boot[l])
        assert close(sum(delta) / n, supplemental['selective_minus_pointwise_ratio_mae'])
        assert all(close(a, b) for a, b in zip([quantile(.025), quantile(.975)], supplemental['query_bootstrap_95']))
        print(scenario, 'verified', len(valid), 'jointly complete pools and full-pointwise quality')
    print('Independent supplementary verification passed; no inference was rerun.')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--selection', type=Path, required=True);p.add_argument('--run', type=Path, required=True)
    p.add_argument('--primary', type=Path, required=True);p.add_argument('--supplement', type=Path, required=True)
    a = p.parse_args();main(a.selection, a.run, a.primary, a.supplement)
