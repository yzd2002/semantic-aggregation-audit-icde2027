#!/usr/bin/env python3
"""Paired candidate-aligned analysis of within-chunk reversal."""
import argparse, json
from pathlib import Path
import numpy as np
from run_compact_prompt_ablation import lines, metrics

def ci(x):
    return [float(v) for v in np.quantile(x, [.025, .975])]

def records(rows, before, after, indices):
    answer = []
    for i in indices:
        gold = [it['gold'] for it in rows[i]['items']]
        old, new = before[i], after[i]
        assert len(gold) == len(old) == len(new) and set(old + new) <= {'Y', 'N'}
        def confusion(pred):
            return (sum(g == 'Y' and y == 'Y' for g, y in zip(gold, pred)),
                    sum(g == 'N' and y == 'Y' for g, y in zip(gold, pred)),
                    sum(g == 'Y' and y == 'N' for g, y in zip(gold, pred)),
                    sum(g == 'N' and y == 'N' for g, y in zip(gold, pred)),
                    abs(pred.count('Y') / len(pred) - gold.count('Y') / len(gold)))
        changed = sum(a != b for a, b in zip(old, new))
        answer.append((*confusion(old), *confusion(new), changed, len(gold),
                       abs(old.count('Y') - new.count('Y')) / len(gold),
                       old.count('Y') != new.count('Y'), changed > 0,
                       sum(a == 'Y' and b == 'N' for a, b in zip(old, new)),
                       sum(a == 'N' and b == 'Y' for a, b in zip(old, new))))
    return np.array(answer, dtype=float)

def paired_bootstrap(rec):
    rng = np.random.default_rng(20271119)
    draw = rng.integers(0, len(rec), size=(4000, len(rec)))
    def f1(c):
        totals = c[draw, :4].sum(axis=1)
        den = 2 * totals[:, 0] + totals[:, 1] + totals[:, 2]
        return np.divide(2 * totals[:, 0], den, out=np.zeros_like(den), where=den != 0)
    changed = rec[draw, 10].sum(axis=1) / rec[draw, 11].sum(axis=1)
    return {'f1_reversed_minus_original_ci': ci(f1(rec[:, 5:10]) - f1(rec[:, :5])),
            'mae_reversed_minus_original_ci': ci(rec[draw, 9].mean(axis=1) - rec[draw, 4].mean(axis=1)),
            'candidate_label_change_fraction_ci': ci(changed),
            'mean_absolute_pool_ratio_change_ci': ci(rec[draw, 12].mean(axis=1)),
            'resamples': 4000, 'seed': 20271119}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--baseline', type=Path, required=True)
    ap.add_argument('--esci-run', type=Path, required=True)
    ap.add_argument('--wands-run', type=Path, required=True)
    a = ap.parse_args()
    summary = json.loads((a.run / 'summary.json').read_text())
    sanity = json.loads((a.run / 'guidance_sanity.json').read_text())
    assert summary['protocol'] == 'PROTOCOL_order_sensitivity.md'
    assert summary['runtime']['engine'] == 'V1 (VLLM_USE_V1=1)' and summary['baseline'] == a.baseline.name
    assert not sanity['unconstrained_valid'] and sanity['guided_valid']
    result = {'protocol': summary['protocol'], 'description': 'Post-hoc within-chunk reversal, candidate-aligned; common-complete quality is conditional on success.', 'datasets': {}}
    for name, source in [('esci', a.esci_run), ('wands', a.wands_run)]:
        rows = lines(source / 'inputs.jsonl')
        old = lines(a.baseline / (name + '_outputs.jsonl'))
        new = lines(a.run / (name + '_outputs.jsonl'))
        assert len(rows) == len(old) == len(new) and all(r['query_id'] == b['query_id'] == z['query_id'] for r, b, z in zip(rows, old, new))
        before, after = [x['pred'] for x in old], [x['pred'] for x in new]
        idx = [i for i, (x, y) in enumerate(zip(before, after)) if x is not None and y is not None]
        d = {'attempted_pools': len(rows), 'original_complete_pools': sum(x is not None for x in before),
             'reversed_complete_pools': sum(x is not None for x in after), 'common_complete_pools': len(idx),
             'common_query_ids': [rows[i]['query_id'] for i in idx]}
        if idx:
            rec = records(rows, before, after, idx)
            d.update({'common_candidate_count': int(rec[:, 11].sum()), 'changed_candidate_count': int(rec[:, 10].sum()),
                      'candidate_label_change_fraction': float(rec[:, 10].sum() / rec[:, 11].sum()),
                      'pools_with_any_label_change': int(rec[:, 14].sum()), 'pools_with_changed_ratio': int(rec[:, 13].sum()),
                      'pools_changed_labels_same_ratio': int((rec[:, 14] - rec[:, 13]).sum()),
                      'yes_to_no_changes': int(rec[:, 15].sum()), 'no_to_yes_changes': int(rec[:, 16].sum()),
                      'mean_absolute_pool_ratio_change': float(rec[:, 12].mean()),
                      'median_absolute_pool_ratio_change': float(np.quantile(rec[:, 12], .5)),
                      'p95_absolute_pool_ratio_change': float(np.quantile(rec[:, 12], .95)),
                      'max_absolute_pool_ratio_change': float(rec[:, 12].max()),
                      'original_quality_common': metrics(rows, before, idx), 'reversed_quality_common': metrics(rows, after, idx),
                      'paired_bootstrap': paired_bootstrap(rec)})
            accepted = wrong = contained = 0
            widths = []
            for i in idx:
                gold = [it['gold'] for it in rows[i]['items']]
                agreed = [(g, x) for g, x, y in zip(gold, before[i], after[i]) if x == y]
                m, n = len(agreed), len(gold)
                p = sum(x == 'Y' for _, x in agreed)
                reference = gold.count('Y')
                accepted += m
                wrong += sum(g != x for g, x in agreed)
                contained += p <= reference <= p + n - m
                widths.append((n - m) / n)
            d['agreement_only_diagnostic'] = {
                'accepted_candidates': accepted, 'abstained_candidates': int(rec[:, 11].sum()) - accepted,
                'accepted_reference_disagreements': wrong,
                'accepted_reference_error_rate': wrong / accepted if accepted else None,
                'reference_ratio_inside_k0_count': contained, 'attempted_pools': len(idx),
                'mean_k0_interval_width': float(np.mean(widths)),
                'interpretation': 'k=0 conditions on all accepted model labels being correct; reference inclusion can also occur through error cancellation.'}
        result['datasets'][name] = d
    (a.run / 'paired_analysis.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()
