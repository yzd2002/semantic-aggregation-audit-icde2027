#!/usr/bin/env python3
"""Describe keyed ordering and counterfactual compact parsing on exposed pools."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from analyze_wands_holdout import comparison
from run_compact_prompt_ablation import lines, metrics
from run_keyed_order_audit import PROTOCOL, parse_keyed
from run_validation import parse_compact


def counterfactual_numeric_id_order(record, count):
    """Interpret a reversed compact string by ascending ID, contrary to its prompt."""
    out = [None] * count
    for raw in record['raw']:
        start, n = raw['start'], raw['size']
        if raw['original_indices_in_prompt_order'] != list(range(start + n - 1, start - 1, -1)):
            raise ValueError('Prior reversed mapping does not match the frozen protocol.')
        labels = parse_compact(raw['final'], n)
        if labels is None:
            return None
        out[start:start + n] = labels
    return None if None in out else out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--prior', type=Path, required=True)
    ap.add_argument('--selection', type=Path, required=True)
    a = ap.parse_args()
    summary = json.loads((a.run / 'summary.json').read_text())
    prior_summary = json.loads((a.prior / 'summary.json').read_text())
    sanity = json.loads((a.run / 'guidance_sanity.json').read_text())
    rows = lines(a.selection / 'inputs.jsonl')
    source_sha = hashlib.sha256((a.selection / 'inputs.jsonl').read_bytes()).hexdigest()
    spec_sha = hashlib.sha256((a.prior / 'specifications.jsonl').read_bytes()).hexdigest()
    if (summary['protocol'] != PROTOCOL or summary['selection_inputs_sha256'] != source_sha
            or prior_summary['selection_inputs_sha256'] != source_sha
            or summary['prior_specifications_sha256'] != spec_sha or summary['model'] != prior_summary['model']
            or summary['mode_order'] != ['original', 'replay', 'stable_reverse']
            or sanity['unconstrained_valid'] or not sanity['guided_valid']):
        raise ValueError('Unexpected protocol, selection, specifications or negative control.')
    assert parse_keyed(sanity['guided_keyed'], list(range(1, 9))) is not None
    keys = summary['mode_order']
    keyed = {k: lines(a.run / (k + '_outputs.jsonl')) for k in keys}
    compact = {k: lines(a.prior / (k + '_outputs.jsonl')) for k in keys}
    for group in (keyed, compact):
        for k, records in group.items():
            if len(records) != len(rows) or any(r['query_id'] != v['query_id'] for r, v in zip(rows, records)):
                raise ValueError('Query ordering mismatch in ' + k)
    pred = {f'keyed_{k}': [r['pred'] for r in keyed[k]] for k in keys}
    pred.update({f'compact_{k}': [r['pred'] for r in compact[k]] for k in keys})
    pred['compact_reverse_hypothetical_id_order'] = [
        counterfactual_numeric_id_order(r, len(row['items'])) for row, r in zip(rows, compact['stable_reverse'])]
    common = [i for i in range(len(rows)) if all(v[i] is not None for v in pred.values())]
    result = {'protocol': PROTOCOL, 'model': summary['model'], 'selection_inputs_sha256': source_sha,
              'prior_specifications_sha256': spec_sha, 'attempted_pools': len(rows),
              'completion': {k: sum(x is not None for x in v) for k, v in pred.items()},
              'common_complete_pools': len(common), 'common_query_ids': [rows[i]['query_id'] for i in common],
              'resamples': 4000, 'seed': 20271125, 'quality_common': {}, 'comparisons': {}}
    if common:
        draw = np.random.default_rng(20271125).integers(0, len(common), size=(4000, len(common)))
        result['quality_common'] = {k: metrics(rows, v, common) for k, v in pred.items()}
        for label, left, right in [
            ('keyed_identical_replay', 'keyed_original', 'keyed_replay'),
            ('keyed_stable_reverse', 'keyed_replay', 'keyed_stable_reverse'),
            ('compact_instructed_list_order', 'compact_replay', 'compact_stable_reverse'),
            ('compact_counterfactual_id_order', 'compact_replay', 'compact_reverse_hypothetical_id_order'),
            ('keyed_vs_compact_original', 'compact_original', 'keyed_original'),
            ('keyed_vs_compact_stable_reverse', 'compact_stable_reverse', 'keyed_stable_reverse'),
        ]:
            result['comparisons'][label] = {'left_condition': left, 'right_condition': right,
                                            **comparison(rows, pred[left], pred[right], common, draw)}
    (a.run / 'paired_analysis.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
