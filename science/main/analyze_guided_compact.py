#!/usr/bin/env python3
"""Analyze constrained vs example-removed unconstrained saved outputs."""
import argparse, json
from pathlib import Path
import numpy as np
from run_compact_prompt_ablation import lines, metrics

def pool_records(rows, preds, idx):
    records = []
    for i in idx:
        gold = [it['gold'] for it in rows[i]['items']]
        p = preds[i]
        assert len(p) == len(gold) and set(p) <= {'Y', 'N'}
        tp = sum(g == 'Y' and y == 'Y' for g, y in zip(gold, p))
        fp = sum(g == 'N' and y == 'Y' for g, y in zip(gold, p))
        fn = sum(g == 'Y' and y == 'N' for g, y in zip(gold, p))
        tn = sum(g == 'N' and y == 'N' for g, y in zip(gold, p))
        mae = abs(p.count('Y') / len(p) - gold.count('Y') / len(gold))
        records.append((tp, fp, fn, tn, mae))
    return np.array(records, dtype=float)

def paired_ci(a, b):
    assert len(a) == len(b) > 0
    rng = np.random.default_rng(20271117)
    draw = rng.integers(0, len(a), size=(4000, len(a)))
    def boot(x):
        counts = x[draw, :4].sum(axis=1)
        den = 2 * counts[:, 0] + counts[:, 1] + counts[:, 2]
        f1 = np.divide(2 * counts[:, 0], den, out=np.zeros_like(den), where=den != 0)
        return f1, x[draw, 4].mean(axis=1)
    af, am = boot(a)
    bf, bm = boot(b)
    return {'f1_guided_minus_unconstrained_ci': [float(x) for x in np.quantile(af - bf, [.025, .975])],
            'mae_guided_minus_unconstrained_ci': [float(x) for x in np.quantile(am - bm, [.025, .975])],
            'resamples': 4000, 'seed': 20271117}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--guided-run', type=Path, required=True)
    ap.add_argument('--unconstrained-run', type=Path, required=True)
    ap.add_argument('--esci-run', type=Path, required=True)
    ap.add_argument('--wands-run', type=Path, required=True)
    a = ap.parse_args()
    run_summary = json.loads((a.guided_run / 'summary.json').read_text())
    sanity = json.loads((a.guided_run / 'guidance_sanity.json').read_text())
    assert run_summary['runtime']['engine'] == 'V1 (VLLM_USE_V1=1)'
    assert not sanity['unconstrained_valid'] and sanity['guided_valid'], 'Guidance negative control failed.'
    out = {'protocol': 'PROTOCOL_guided_compact.md', 'description': 'Post-hoc: quality is conditional on completed pools, with paired inference only on the shared completed subset.', 'datasets': {}}
    for name, source in [('esci', a.esci_run), ('wands', a.wands_run)]:
        rows = lines(source / 'inputs.jsonl')
        previous = lines(source / 'outputs.jsonl')
        guided = lines(a.guided_run / (name + '_outputs.jsonl'))
        un = lines(a.unconstrained_run / (name + '_outputs.jsonl'))
        assert len(rows) == len(previous) == len(guided) == len(un)
        assert all(len({r['query_id'], p['query_id'], g['query_id'], u['query_id']}) == 1 for r, p, g, u in zip(rows, previous, guided, un))
        preds = {'guided': [x['pred'] for x in guided], 'unconstrained': [x['pred'] for x in un],
                 'pointwise': [x['plans']['pointwise']['pred'] for x in previous],
                 'always_Y': [['Y'] * len(r['items']) for r in rows], 'always_N': [['N'] * len(r['items']) for r in rows]}
        idx = {k: [i for i, p in enumerate(vals) if p is not None] for k, vals in preds.items()}
        shared = sorted(set(idx['guided']) & set(idx['unconstrained']))
        assert all(i in idx['pointwise'] for i in shared)
        d = {'attempted_pools': len(rows), 'guided_complete_pools': len(idx['guided']),
             'unconstrained_complete_pools': len(idx['unconstrained']), 'common_complete_pools': len(shared),
             'quality_own_complete': {}, 'quality_common_complete': {}}
        for k in ('guided', 'unconstrained'):
            if idx[k]:
                d['quality_own_complete'][k] = metrics(rows, preds[k], idx[k])
        if shared:
            for k, val in preds.items():
                d['quality_common_complete'][k] = metrics(rows, val, shared)
            d['paired_bootstrap'] = paired_ci(pool_records(rows, preds['guided'], shared), pool_records(rows, preds['unconstrained'], shared))
        if len(idx['guided']) == len(rows):
            d['quality_all_attempted'] = {k: metrics(rows, preds[k], idx['guided']) for k in ('guided', 'pointwise', 'always_Y', 'always_N')}
        out['datasets'][name] = d
    (a.guided_run / 'paired_analysis.json').write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps(out, indent=2))
if __name__ == '__main__':
    main()
