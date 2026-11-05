#!/usr/bin/env python3
"""Confirm why the recorded V0 guided-decoding attempts are invalid."""
from pathlib import Path
from run_compact_prompt_ablation import lines
from run_validation import parse_compact

HERE = Path(__file__).resolve().parent
for model in ('qwen', 'smollm'):
    baseline = HERE / ('run_compact_prompt_ablation_qwen_v0' if model == 'qwen' else 'run_compact_prompt_ablation_smollm')
    attempt = HERE / ('run_guided_compact_' + model)
    for dataset in ('esci', 'wands'):
        b = lines(baseline / (dataset + '_outputs.jsonl'))
        g = lines(attempt / (dataset + '_outputs.jsonl'))
        assert len(b) == len(g)
        assert all(x['query_id'] == y['query_id'] for x, y in zip(b, g))
        old = [c for row in b for c in row['raw']]
        new = [c for row in g for c in row['raw']]
        assert len(old) == len(new)
        assert all(x['start'] == y['start'] and x['size'] == y['size'] and x['first'] == y['first'] and x['final'] == y['final'] for x, y in zip(old, new))
        invalid = sum(parse_compact(c['final'], c['size']) is None for c in new)
        assert invalid > 0
        print(model, dataset, 'identical chunks', len(old), 'invalid final chunks', invalid)
print('Both V0 attempts are invalid as guided-decoding comparisons.')
