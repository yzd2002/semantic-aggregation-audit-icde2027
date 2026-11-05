#!/usr/bin/env python3
"""Compare replay, stable-ID reversal and the saved renumbered reversal."""
import argparse, json
from pathlib import Path
import numpy as np
from run_compact_prompt_ablation import lines, metrics

COMPARISONS = [('identical_prompt_replay', 'original', 'replay'),
               ('reverse_stable_ids', 'replay', 'stable_reverse'),
               ('reassign_ids_on_reversal', 'stable_reverse', 'renumbered_reverse'),
               ('reverse_reassigned_ids', 'original', 'renumbered_reverse')]

def ci(x):
    return [float(v) for v in np.quantile(x, [.025, .975])]

def paired(rows, left, right, idx, draw):
    records = []
    for i in idx:
        x, y = left[i], right[i]
        assert len(x) == len(y) == len(rows[i]['items']) and set(x + y) <= {'Y', 'N'}
        changed = sum(a != b for a, b in zip(x, y))
        records.append((changed, len(x), abs(x.count('Y') - y.count('Y')) / len(x), changed > 0,
                        x.count('Y') != y.count('Y')))
    arr = np.array(records, dtype=float)
    label_rate = arr[draw, 0].sum(axis=1) / arr[draw, 1].sum(axis=1)
    ratio_delta = arr[draw, 2].mean(axis=1)
    return {'candidate_count': int(arr[:, 1].sum()), 'changed_candidate_count': int(arr[:, 0].sum()),
            'candidate_label_change_fraction': float(arr[:, 0].sum() / arr[:, 1].sum()),
            'pools_with_any_label_change': int(arr[:, 3].sum()), 'pools_with_changed_ratio': int(arr[:, 4].sum()),
            'mean_absolute_pool_ratio_change': float(arr[:, 2].mean()),
            'candidate_label_change_fraction_ci': ci(label_rate),
            'mean_absolute_pool_ratio_change_ci': ci(ratio_delta)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--baseline', type=Path, required=True)
    ap.add_argument('--renumbered', type=Path, required=True)
    ap.add_argument('--esci-run', type=Path, required=True)
    ap.add_argument('--wands-run', type=Path, required=True)
    a = ap.parse_args()
    summary = json.loads((a.run / 'summary.json').read_text())
    sanity = json.loads((a.run / 'guidance_sanity.json').read_text())
    assert summary['protocol'] == 'PROTOCOL_order_controls.md'
    assert summary['baseline'] == a.baseline.name and summary['renumbered'] == a.renumbered.name
    assert summary['mode_order'] == ['replay', 'stable_reverse'] and summary['runtime']['engine'] == 'V1 (VLLM_USE_V1=1)'
    assert not sanity['unconstrained_valid'] and sanity['guided_valid']
    out = {'protocol': summary['protocol'], 'resamples': 4000, 'seed': 20271121,
           'description': 'Post-hoc four-condition comparison, conditional on common completion; one replay does not establish deterministic stability.', 'datasets': {}}
    for name, source in [('esci', a.esci_run), ('wands', a.wands_run)]:
        rows = lines(source / 'inputs.jsonl')
        outputs = {'original': lines(a.baseline / (name + '_outputs.jsonl')),
                   'replay': lines(a.run / ('replay_' + name + '_outputs.jsonl')),
                   'stable_reverse': lines(a.run / ('stable_reverse_' + name + '_outputs.jsonl')),
                   'renumbered_reverse': lines(a.renumbered / (name + '_outputs.jsonl'))}
        assert all(len(x) == len(rows) for x in outputs.values())
        assert all(all(row['query_id'] == out['query_id'] for row, out in zip(rows, val)) for val in outputs.values())
        preds = {k: [x['pred'] for x in vals] for k, vals in outputs.items()}
        idx = [i for i in range(len(rows)) if all(v[i] is not None for v in preds.values())]
        d = {'attempted_pools': len(rows), 'completion': {k: sum(v is not None for v in vals) for k, vals in preds.items()},
             'common_complete_pools': len(idx), 'common_query_ids': [rows[i]['query_id'] for i in idx],
             'quality_common': {}, 'comparisons': {}}
        if idx:
            rng = np.random.default_rng(20271121)
            draw = rng.integers(0, len(idx), size=(4000, len(idx)))
            for key, vals in preds.items():
                d['quality_common'][key] = metrics(rows, vals, idx)
            for key, left, right in COMPARISONS:
                d['comparisons'][key] = {'left_condition': left, 'right_condition': right,
                                          **paired(rows, preds[left], preds[right], idx, draw)}
        out['datasets'][name] = d
    (a.run / 'paired_analysis.json').write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps(out, indent=2))
if __name__ == '__main__':
    main()
