#!/usr/bin/env python3
"""Exercise the saved-trace analyzer and independent verifier without GPU inference."""
import copy
import hashlib
import json
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

from order_fallback import combine
from run_esci_selective_holdout import point_prompt
from run_experiment import p_spec
from run_full_pools import chunks
from run_keyed_order_audit import keyed_prompt

root = Path(__file__).resolve().parent
temp = Path(tempfile.mkdtemp(prefix='intentsat-esci-selective-test-', dir='/dev/shm'))
selection, run, analysis = temp/'selection', temp/'run', temp/'analysis'
selection.mkdir(); run.mkdir()
rows = []
for q in range(250):
    n = 3 + q % 10
    items = [{'id': i+1, 'text': 'item ' + str(i), 'grade': 'E' if i % 2 else 'I',
              'gold': 'Y' if i % 2 else 'N'} for i in range(n)]
    rows.append({'query_id': str(q), 'query': 'synthetic ' + str(q), 'items': items})
def write_rows(path, values):
    path.write_text(''.join(json.dumps(x, ensure_ascii=False) + '\n' for x in values), encoding='utf8')
write_rows(selection/'inputs.jsonl', rows)
digest = hashlib.sha256((selection/'inputs.jsonl').read_bytes()).hexdigest()
(selection/'selection.json').write_text(json.dumps({'protocol': 'PROTOCOL_esci_selective_holdout.md',
                                                     'inputs_sha256': digest}) + '\n')
spec = {'product_type': 'item', 'required': [], 'preferred': []}
spec_text = json.dumps(spec, ensure_ascii=False)
def keytext(labels):
    return '{' + ','.join(json.dumps(str(i)) + ':' + json.dumps(x) for i, x in enumerate(labels, 1)) + '}'
def record(qid, start, size, ids, text, answer, invalid=False):
    return {'query_id': qid, 'start': start, 'size': size, 'presented_ids': ids,
            'prompt_sha256': hashlib.sha256(text.encode()).hexdigest(), 'wrapped_sha256': '0'*64,
            'first': 'invalid' if invalid else answer, 'final': answer, 'first_finish_reason': 'stop',
            'retry_finish_reason': 'stop' if invalid else None, 'first_input_tokens': 1, 'first_output_tokens': 1,
            'retry_input_tokens': int(invalid), 'retry_output_tokens': int(invalid), 'retry_prompt_plus_cap': 512}
events = {x: [] for x in ('specifications', 'original', 'stable_reverse', 'selective_pointwise', 'replay', 'pointwise')}
saved = []
for qi, row in enumerate(rows):
    qid = row['query_id']
    events['specifications'].append(record(qid, 0, len(row['items']), None, p_spec(row), spec_text))
    first, reverse, fallback = [], [], [None]*len(row['items'])
    for part in chunks(row):
        n, start = len(part['items']), part['start']
        a = ['Y' if (start+i) % 2 else 'N' for i in range(n)]
        b = a.copy(); b[0] = 'N' if b[0] == 'Y' else 'Y'
        # One genuinely failed chunk exercises pool abstention and exclusion.
        invalid = qi == 0 and start == 0
        for mode, labels, rev in (('original', a, False), ('stable_reverse', b, True), ('replay', a, False)):
            current = {**part, 'items': list(reversed(part['items'])) if rev else part['items']}
            answer = 'invalid' if invalid and mode == 'stable_reverse' else keytext(labels)
            events[mode].append(record(qid, start, n, [x['id'] for x in current['items']],
                keyed_prompt(current, spec), answer, invalid=invalid and mode == 'stable_reverse'))
        first.extend(a); reverse.extend(b if not invalid else [None]*n)
        if not invalid:
            fallback[start] = a[0]
            events['selective_pointwise'].append(record(qid, start, 1, None,
                point_prompt(row, row['items'][start], spec), a[0]))
    for i, item in enumerate(row['items']):
        answer = 'Y' if i % 2 else 'N'
        events['pointwise'].append(record(qid, i, 1, None, point_prompt(row, item, spec), answer))
    saved.append({'query_id': qid, 'pred': combine(first, reverse if None not in reverse else None, fallback)})
for name, records in events.items():
    write_rows(run/(name+'_responses.jsonl'), records)
write_rows(run/'selective_outputs.jsonl', saved)
sanity = {'unconstrained': 'BANANA', 'guided_keyed': keytext('YYYYYYYY'), 'guided_point': 'Y',
          'unconstrained_valid_keyed': False, 'unconstrained_valid_point': False,
          'guided_keyed_valid': True, 'guided_point_valid': True}
(run/'guidance_sanity.json').write_text(json.dumps(sanity) + '\n')
mode_stats = {}
for mode, recs in events.items():
    retry = sum(x['retry_finish_reason'] is not None for x in recs)
    mode_stats[mode] = {'initial_calls': len(recs), 'retry_calls': retry,
        'logical_tokens': {'input': len(recs)+retry, 'output': len(recs)+retry},
        'max_retry_prompt_plus_cap': max((x['retry_prompt_plus_cap'] for x in recs), default=0)}
parts = ('specifications', 'original', 'stable_reverse', 'selective_pointwise')
pipeline = {'complete_pools': sum(x['pred'] is not None for x in saved),
            'initial_calls': sum(mode_stats[x]['initial_calls'] for x in parts),
            'retry_calls': sum(mode_stats[x]['retry_calls'] for x in parts),
            'logical_tokens': {k: sum(mode_stats[x]['logical_tokens'][k] for x in parts) for k in ('input', 'output')},
            'offline_batch_wall_seconds': 1.0}
summary = {'protocol': 'PROTOCOL_esci_selective_holdout.md', 'model': 'synthetic',
           'selection_inputs_sha256': digest, 'completed': True,
           'mode_order': ['original', 'stable_reverse', 'selective_pointwise', 'replay', 'pointwise'],
           'modes': mode_stats, 'selective_pipeline': pipeline}
(run/'summary.json').write_text(json.dumps(summary) + '\n')
analyze = [sys.executable, str(root/'analyze_esci_selective_holdout.py'), '--selection', str(selection),
           '--run', str(run), '--out', str(analysis)]
verify = [sys.executable, str(root/'verify_esci_selective_holdout.py'), '--selection', str(selection),
          '--run', str(run), '--analysis', str(analysis)]
for command in (analyze, verify):
    subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
report = json.loads((analysis/'summary.json').read_text())
assert report['common_complete_pools'] == 249 and report['completion']['actual_fallback'] == 249
tampered = copy.deepcopy(events['selective_pointwise'])
tampered[0]['prompt_sha256'] = '0'*64
write_rows(run/'selective_pointwise_responses.jsonl', tampered)
if subprocess.run(verify, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True).returncode == 0:
    raise AssertionError('Independent verifier accepted a modified prompt association.')
print('Synthetic 250-pool analysis, incomplete-pool handling, independent reconstruction and prompt-tamper rejection passed.')
