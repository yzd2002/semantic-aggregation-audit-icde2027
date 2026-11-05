#!/usr/bin/env python3
"""Exercise order mappings, prompt isolation, analysis and tamper detection."""
import copy
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from run_compact_prompt_ablation import prompt_without_example
from run_full_pools import chunks
from run_random_order_sensitivity import MODES, digest

root = Path(__file__).resolve().parent
rows = [json.loads(x) for x in (root / 'wands_holdout_selection/inputs.jsonl').read_text().splitlines()]
maps = [json.loads(x) for x in (root / 'random_order_selection/mappings.jsonl').read_text().splitlines()]
assert len(rows) == len(maps) == 160
for previous in ('run_wands_holdout_qwen_v1', 'run_wands_holdout_smollm_v1'):
    specs = [json.loads(x) for x in (root / previous / 'specifications.jsonl').read_text().splitlines()]
    old = [json.loads(x) for x in (root / previous / 'original_outputs.jsonl').read_text().splitlines()]
    for row, spec, prior in zip(rows, specs, old):
        assert row['query_id'] == spec['query_id'] == prior['query_id']
        altered = copy.deepcopy(row)
        for item in altered['items']:
            item['gold'] = 'N' if item['gold'] == 'Y' else 'Y'
            item['grade'] = 'TEST-GRADE-MUST-NOT-APPEAR'
        for i, (part, changed) in enumerate(zip(chunks(row), chunks(altered))):
            text = prompt_without_example(part, spec['parsed'])
            assert text == prompt_without_example(changed, spec['parsed'])
            assert 'TEST-GRADE-MUST-NOT-APPEAR' not in text
            assert digest(text) == prior['raw'][i]['condition_prompt_sha256']

tmp = Path(tempfile.mkdtemp(prefix='intentsat-random-order-test-', dir='/dev/shm'))
run, analysis = tmp / 'run', tmp / 'analysis'
run.mkdir()
versions = {'vllm': '0.10.2', 'torch': '2.8.0+cu128', 'transformers': '4.57.1',
            'numpy': '2.2.6', 'xgrammar': '0.1.23', 'outlines-core': '0.2.11', 'setuptools': '79.0.1'}
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
(run / 'preflight.json').write_text(json.dumps({'protocol': 'PROTOCOL_random_order_sensitivity.md',
    'model': 'Qwen3-4B-Instruct-2507', 'versions': versions, 'queries': 160, 'chunks_per_mode': 640,
    'mode_order': MODES, 'batch_size': 64, 'max_retry_prompt_plus_cap': 1,
    'selection_sha256': sha(root / 'wands_holdout_selection/inputs.jsonl'),
    'mappings_sha256': sha(root / 'random_order_selection/mappings.jsonl')}))
(run / 'controls.json').write_text(json.dumps({'unconstrained': 'BANANA', 'guided': 'YYYYYYYY',
    'unconstrained_invalid': True, 'guided_valid': True}))
specs = [json.loads(x) for x in (root / 'run_wands_holdout_qwen_v1/specifications.jsonl').read_text().splitlines()]
for mode in MODES:
    with (run / (mode + '_records.jsonl')).open('w', encoding='utf8') as f:
        for qi, (row, mapping, spec) in enumerate(zip(rows, maps, specs)):
            for ci, (part, order) in enumerate(zip(chunks(row), mapping['chunks'])):
                positions = order['orders'][mode] if mode.startswith('perm_') else list(range(8))
                selected = {**part, 'items': [part['items'][j] for j in positions]}
                text, original = prompt_without_example(selected, spec['parsed']), prompt_without_example(part, spec['parsed'])
                labels = ['Y' if (part['start'] + j) % 2 == 0 else 'N' for j in positions]
                if mode == 'perm_1' or (mode == 'baseline_b' and qi == 0 and ci == 0):
                    labels[0] = 'N' if labels[0] == 'Y' else 'Y'
                value = ''.join(labels)
                record = {'query_id': row['query_id'], 'start': part['start'], 'size': 8,
                    'indices': [part['start'] + j for j in positions], 'original_prompt_sha256': digest(original),
                    'prompt_sha256': digest(text), 'wrapped_sha256': '0' * 64, 'first': value, 'final': value,
                    'first_finish_reason': 'stop', 'retry_finish_reason': None, 'first_input_tokens': 1,
                    'first_output_tokens': 8, 'retry_input_tokens': 0, 'retry_output_tokens': 0}
                f.write(json.dumps(record) + '\n')
env = {**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'}
analyze = [sys.executable, str(root / 'analyze_random_order_sensitivity.py'), '--selection',
    str(root / 'wands_holdout_selection'), '--orders', str(root / 'random_order_selection'),
    '--run', str(run), '--out', str(analysis)]
verify = [sys.executable, str(root / 'verify_random_order_sensitivity.py'), '--selection',
    str(root / 'wands_holdout_selection'), '--orders', str(root / 'random_order_selection'),
    '--prior', str(root / 'run_wands_holdout_qwen_v1'), '--run', str(run), '--analysis', str(analysis)]
subprocess.run(analyze, check=True, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
subprocess.run(verify, check=True, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
summary = json.loads((analysis / 'summary.json').read_text())
assert summary['common_complete_pools'] == 160
assert summary['comparisons']['perm_1_vs_baseline_a']['changed_candidates'] == 640
assert summary['comparisons']['perm_2_vs_baseline_b']['changed_candidates'] == 1
assert summary['comparisons']['perm_3_vs_baseline_c']['changed_candidates'] == 0
assert summary['comparisons']['baseline_b_vs_baseline_a']['changed_candidates'] == 1
path = run / 'perm_1_records.jsonl'
records = [json.loads(x) for x in path.read_text().splitlines()]
records[0]['prompt_sha256'] = '0' * 64
path.write_text(''.join(json.dumps(x) + '\n' for x in records))
assert subprocess.run(verify, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE).returncode != 0
print('Saved prompts for both models, label isolation, synthetic paired outcomes and prompt-tamper rejection passed.')
