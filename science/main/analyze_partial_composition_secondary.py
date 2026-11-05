#!/usr/bin/env python3
"""Post-inspection descriptive spread of the observed-subset composition term."""
import argparse
import hashlib
import json
import statistics
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('--run', type=Path, required=True)
ap.add_argument('--out', type=Path, required=True)
args = ap.parse_args()
source = args.run / 'details.jsonl'
records = [json.loads(x) for x in source.read_text().splitlines()]
datasets = {}
for name in ('esci_250', 'wands_160'):
    values = [x['observed_composition_term'] for x in records if x['dataset'] == name and x['m'] > 0]
    datasets[name] = {'defined_pools': len(values),
                      'mean_absolute_composition_term': statistics.fmean(abs(x) for x in values) if values else None,
                      'min_composition_term': min(values) if values else None,
                      'max_composition_term': max(values) if values else None,
                      'nonzero_pools': sum(abs(x) > 1e-12 for x in values)}
result = {'description': 'Post-inspection descriptive spread; not a pre-specified test or a causal effect.',
          'details_sha256': hashlib.sha256(source.read_bytes()).hexdigest(), 'datasets': datasets}
with args.out.open('x', encoding='utf8') as f:
    json.dump(result, f, indent=2, allow_nan=False)
    f.write('\n')
for name, value in datasets.items():
    print(name, 'mean absolute composition term:', value['mean_absolute_composition_term'])
