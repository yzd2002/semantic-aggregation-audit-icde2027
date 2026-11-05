#!/usr/bin/env python3
"""Post-hoc audit of where LOTUS obtained the helper's True/False scores."""
import collections
import json
import sys
from pathlib import Path

run = Path(__file__).resolve().parent / 'run_lotus_wands_cascade'
summary = json.loads((run / 'summary.json').read_text())
generated = collections.Counter()
positions = collections.Counter()
count = 0
with (run / 'model_requests.jsonl').open(encoding='utf8') as f:
    for line in f:
        e = json.loads(line)
        if e['role'] != 'helper':
            continue
        count += 1
        generated[e['logprobs'][0]['token']] += 1
        position = next((i for i, token in enumerate(e['logprobs'])
                         if {'True', 'False'} <= {x['token'] for x in token['top_logprobs']}), None)
        positions[str(position)] += 1
assert count == summary['candidates']
assert dict(positions) == summary['proxy_diagnostics']['first_both_label_position_counts']
audit = {'method': 'post-hoc recorded helper token-position audit', 'helper_responses': count,
         'first_generated_tokens': dict(generated), 'first_both_label_positions': dict(positions)}
path = run / 'proxy_position_audit.json'
if sys.argv[1:] == ['--verify-only']:
    assert json.loads(path.read_text()) == audit
elif not sys.argv[1:]:
    path.write_text(json.dumps(audit, indent=2) + '\n')
else:
    raise SystemExit('Usage: analyze_lotus_proxy_position.py [--verify-only]')
print(json.dumps(audit, indent=2))
