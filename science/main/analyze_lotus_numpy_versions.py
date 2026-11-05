#!/usr/bin/env python3
"""Post-hoc NumPy sensitivity of the recorded LOTUS helper-score calculation."""
import argparse
import json
from pathlib import Path
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--run', type=Path, required=True)
parser.add_argument('--out', type=Path, required=True)
args = parser.parse_args()
raw = []
with (args.run / 'model_requests.jsonl').open(encoding='utf8') as f:
    for line in f:
        event = json.loads(line)
        if event['role'] != 'helper':
            continue
        tokens = [item['token'] for item in event['logprobs']]
        value = None
        for item in event['logprobs']:
            top = {x['token']: np.exp(x['logprob']) for x in item['top_logprobs']}
            if 'True' in top and 'False' in top:
                value = float(top['True'] / (top['True'] + top['False']))
                break
        raw.append(value if value is not None else float('True' in tokens))
assert len(raw) == 1664
quantiles = np.percentile(raw, np.linspace(0, 100, 51))
scores = np.clip((np.digitize(raw, quantiles) - 1) / 50, 0, 1)
result = {'numpy_version': np.__version__, 'raw': raw,
          'quantiles': quantiles.tolist(), 'scores': scores.tolist()}
with args.out.open('x', encoding='utf8') as f:
    json.dump(result, f, indent=2, allow_nan=False)
    f.write('\n')
print('Recorded helper-score recomputation with NumPy', np.__version__, 'at', args.out)
