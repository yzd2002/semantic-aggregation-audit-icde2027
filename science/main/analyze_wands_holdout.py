#!/usr/bin/env python3
"""Evaluate the frozen holdout's primary order contrast and reference baselines."""
import argparse, hashlib, json, re
from pathlib import Path
import numpy as np
from run_compact_prompt_ablation import lines, metrics

def ci(values):
    return [float(x) for x in np.quantile(values, [.025, .975])]

def comparison(rows, left, right, idx, draw):
    points = []
    for i in idx:
        a, b = left[i], right[i]
        assert len(a) == len(b) == len(rows[i]['items']) and set(a+b) <= {'Y', 'N'}
        changed = sum(x != y for x, y in zip(a, b))
        points.append((changed, len(a), abs(a.count('Y') - b.count('Y')) / len(a), changed > 0,
                       a.count('Y') != b.count('Y')))
    p = np.array(points, dtype=float)
    rates = p[draw, 0].sum(axis=1) / p[draw, 1].sum(axis=1)
    ratio_shift = p[draw, 2].mean(axis=1)
    return {'candidate_count': int(p[:, 1].sum()), 'changed_candidate_count': int(p[:, 0].sum()),
            'candidate_label_change_fraction': float(p[:, 0].sum() / p[:, 1].sum()),
            'candidate_label_change_fraction_ci': ci(rates), 'pools_with_any_label_change': int(p[:, 3].sum()),
            'pools_with_changed_ratio': int(p[:, 4].sum()), 'mean_absolute_pool_ratio_change': float(p[:, 2].mean()),
            'mean_absolute_pool_ratio_change_ci': ci(ratio_shift),
            'fixed_reference_max_candidate_error_floor': float(p[:, 0].sum() / (2 * p[:, 1].sum()))}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--selection', type=Path, required=True)
    a = ap.parse_args()
    summary = json.loads((a.run / 'summary.json').read_text())
    source = json.loads((a.selection / 'selection.json').read_text())
    sanity = json.loads((a.run / 'guidance_sanity.json').read_text())
    rows = lines(a.selection / 'inputs.jsonl')
    source_sha = hashlib.sha256((a.selection / 'inputs.jsonl').read_bytes()).hexdigest()
    assert summary['protocol'] == source['protocol'] == 'PROTOCOL_wands_holdout.md'
    assert summary['selection_inputs_sha256'] == source_sha and summary['runtime']['engine'] == 'V1 (VLLM_USE_V1=1)'
    assert summary['mode_order'] == ['original', 'replay', 'stable_reverse']
    assert not sanity['unconstrained_valid'] and sanity['guided_valid'] and len(rows) == 160
    outputs = {mode: lines(a.run / (mode + '_outputs.jsonl')) for mode in summary['mode_order']}
    assert all(len(v) == len(rows) for v in outputs.values())
    assert all(all(r['query_id'] == o['query_id'] for r, o in zip(rows, output)) for output in outputs.values())
    preds = {k: [o['pred'] for o in v] for k, v in outputs.items()}
    preds['always_Y'] = [['Y'] * len(row['items']) for row in rows]
    preds['always_N'] = [['N'] * len(row['items']) for row in rows]
    preds['lexical_all_terms'] = []
    for row in rows:
        query_terms = set(re.findall(r'\w+', row['query'].casefold()))
        preds['lexical_all_terms'].append(['Y' if query_terms and query_terms.issubset(set(re.findall(r'\w+', it['text'].casefold()))) else 'N'
                                           for it in row['items']])
    idx = [i for i in range(len(rows)) if all(preds[k][i] is not None for k in summary['mode_order'])]
    out = {'protocol': summary['protocol'], 'model': summary['model'], 'selection_inputs_sha256': source_sha,
           'attempted_pools': len(rows), 'grade_counts': source['grade_counts'],
           'specification_fallback_pools': summary['specifications']['fallback_pools'],
           'completion': {k: sum(x is not None for x in preds[k]) for k in summary['mode_order']},
           'common_complete_pools': len(idx), 'common_query_ids': [rows[i]['query_id'] for i in idx],
           'resamples': 4000, 'seed': 20271123, 'primary_threshold': .10,
           'primary_model': summary['model'] == 'Qwen3-4B-Instruct-2507', 'quality_common': {}, 'comparisons': {}}
    if idx:
        rng = np.random.default_rng(20271123)
        draw = rng.integers(0, len(idx), size=(4000, len(idx)))
        for k, v in preds.items():out['quality_common'][k] = metrics(rows, v, idx)
        for label, left, right in [('identical_prompt_replay', 'original', 'replay'),
                                   ('reverse_stable_ids', 'replay', 'stable_reverse')]:
            out['comparisons'][label] = {'left_condition': left, 'right_condition': right,
                                         **comparison(rows, preds[left], preds[right], idx, draw)}
        out['primary_criterion_met'] = (out['comparisons']['reverse_stable_ids']['candidate_label_change_fraction_ci'][0] > .10
                                        if out['primary_model'] else None)
    else:
        out['primary_criterion_met'] = None
    (a.run / 'paired_analysis.json').write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps(out, indent=2), flush=True)
if __name__ == '__main__':
    main()
