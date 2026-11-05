#!/usr/bin/env python3
"""Recompute query/product bootstrap weights and the reported percentile interval."""
import argparse, collections, json
from pathlib import Path
import numpy as np
from run_compact_prompt_ablation import lines

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--selection', type=Path, required=True)
    a = ap.parse_args()
    rows = lines(a.selection / 'inputs.jsonl')
    conditions = {x: lines(a.run / (x + '_outputs.jsonl')) for x in ('original', 'replay', 'stable_reverse')}
    primary = json.loads((a.run / 'paired_analysis.json').read_text())
    saved = json.loads((a.run / 'two_way_analysis.json').read_text())
    assert saved['protocol'] == 'PROTOCOL_wands_holdout_two_way.md' and saved['seed'] == 20271124 and saved['resamples'] == 4000
    common = [i for i, row in enumerate(rows) if all(conditions[k][i]['pred'] is not None for k in conditions)]
    assert [rows[i]['query_id'] for i in common] == primary['common_query_ids'] and saved['common_pools'] == len(common)
    pids = sorted({int(it['product_id']) for i in common for it in rows[i]['items']})
    lookup = {pid: i for i, pid in enumerate(pids)}
    per_query = [];counts = collections.Counter();total_changes = total_candidates = 0
    for i in common:
        arow,brow = conditions['replay'][i],conditions['stable_reverse'][i]
        assert rows[i]['query_id'] == arow['query_id'] == brow['query_id']
        entries = []
        for item, x, y in zip(rows[i]['items'], arow['pred'], brow['pred']):
            pid = int(item['product_id']);changed = int(x != y);entries.append((lookup[pid], changed));counts[pid] += 1
            total_changes += changed;total_candidates += 1
        per_query.append(entries)
    assert saved['distinct_products'] == len(pids) and saved['products_appearing_multiple_times'] == sum(v>1 for v in counts.values())
    assert saved['disagreement_count'] == total_changes and saved['candidates'] == total_candidates
    assert abs(saved['disagreement_fraction'] - total_changes/total_candidates) < 1e-12
    assert saved['query_only_ci'] == primary['comparisons']['reverse_stable_ids']['candidate_label_change_fraction_ci']
    rng = np.random.default_rng(20271124);replicates = []
    for _ in range(4000):
        qdraw = np.bincount(rng.integers(0,len(common),size=len(common)),minlength=len(common))
        pdraw = np.bincount(rng.integers(0,len(pids),size=len(pids)),minlength=len(pids))
        num = den = 0
        for qi, q in enumerate(per_query):
            if qdraw[qi]:
                for pi, changed in q:
                    w = int(qdraw[qi]) * int(pdraw[pi]);den += w;num += w * changed
        if den == 0:raise RuntimeError('Zero-weight two-way resample.')
        replicates.append(num/den)
    assert np.allclose(np.quantile(replicates,[.025,.975]),saved['query_product_ci'],rtol=0,atol=1e-12)
    print('Secondary query–product bootstrap independently verified; products:',len(pids),'repeated products:',saved['products_appearing_multiple_times'])
if __name__ == '__main__':main()
