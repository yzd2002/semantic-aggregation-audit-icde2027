#!/usr/bin/env python3
"""Compare the saved supervised reference with each new-pool model run."""
import argparse
import json
from pathlib import Path

import numpy as np
from analyze_esci_selective_holdout import rebuild, lines
from analyze_order_fallback_validation import paired
from run_compact_prompt_ablation import metrics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--selection', type=Path, required=True)
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--qwen-run', type=Path, required=True)
    p.add_argument('--qwen-analysis', type=Path, required=True)
    p.add_argument('--smollm-run', type=Path, required=True)
    p.add_argument('--smollm-analysis', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        raise FileExistsError('Use a new analysis path.')
    rows = lines(a.selection/'inputs.jsonl')
    predicted = lines(a.reference/'predictions.jsonl')
    if len(rows) != len(predicted) or any(x['query_id'] != y['query_id'] or len(x['items']) != len(y['pred'])
                                          for x, y in zip(rows, predicted)):
        raise ValueError('Reference and new pools differ.')
    supervised = [x['pred'] for x in predicted]
    if any(any(label not in ('Y', 'N') for label in row) for row in supervised):
        raise ValueError('Invalid supervised labels.')
    result = {'protocol': 'PROTOCOL_esci_selective_supervised_reference.md',
              'all_250_supervised': metrics(rows, supervised, list(range(len(rows)))), 'models': {}}
    for model, run, analysis in (('qwen', a.qwen_run, a.qwen_analysis), ('smollm', a.smollm_run, a.smollm_analysis)):
        events = {mode: lines(run/(mode+'_responses.jsonl')) for mode in
                  ('original', 'stable_reverse', 'selective_pointwise', 'replay', 'pointwise')}
        pred, _, _, _ = rebuild(rows, events)
        report = json.loads((analysis/'summary.json').read_text())
        common_ids = set(report['common_query_ids'])
        common = [i for i, row in enumerate(rows) if row['query_id'] in common_ids]
        if [rows[i]['query_id'] for i in common] != report['common_query_ids']:
            raise ValueError('Common set differs from model analysis.')
        pred['supervised'] = supervised
        section = {'common_complete_pools': len(common), 'supervised_quality_common': metrics(rows, supervised, common),
                   'secondary_comparisons': {}}
        if common:
            draws = np.random.default_rng(20271202).integers(0, len(common), size=(4000, len(common)))
            for left in ('original', 'actual_fallback', 'pointwise'):
                section['secondary_comparisons']['supervised_minus_'+left] = paired(rows, pred, common, left, 'supervised', draws)
        result['models'][model] = section
    a.out.mkdir(parents=True, exist_ok=False)
    (a.out/'summary.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf8')
    print('Saved secondary supervised comparisons for both model-specific common subsets.')


if __name__ == '__main__':
    main()
