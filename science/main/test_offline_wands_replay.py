#!/usr/bin/env python3
"""Check frozen WANDS replay inputs and synthetic comparison edge cases."""
import copy
import json
from pathlib import Path

from run_compact_prompt_ablation import prompt_without_example
from run_full_pools import chunks
from run_offline_wands_replay import MODES, compare, digest, lines
from verify_offline_wands_replay import stats

base = Path(__file__).resolve().parent
rows = lines(base / 'wands_holdout_selection/inputs.jsonl')[:16]
specs = lines(base / 'run_wands_holdout_qwen_v1/specifications.jsonl')[:16]
old = {m: lines(base / 'run_wands_holdout_qwen_v1' / (m + '_outputs.jsonl'))[:16] for m in MODES}
assert len(rows) == len(specs) == 16
checked = 0
for i, (row, spec) in enumerate(zip(rows, specs)):
    assert row['query_id'] == spec['query_id'] and len(row['items']) == 32
    mutated = copy.deepcopy(row)
    for item in mutated['items']:
        item['gold'] = 'N' if item['gold'] == 'Y' else 'Y'
        item['grade'] = 'TEST-GRADE-MUST-NOT-BE-IN-PROMPT'
    for chunk, (part, alt) in enumerate(zip(chunks(row), chunks(mutated))):
        assert prompt_without_example(part, spec['parsed']) == prompt_without_example(alt, spec['parsed'])
        reverse = {**part, 'items': list(reversed(part['items']))}
        original, reversed_text = prompt_without_example(part, spec['parsed']), prompt_without_example(reverse, spec['parsed'])
        start = part['start']
        for mode, text, positions in [('original', original, list(range(start, start + 8))),
                                      ('replay', original, list(range(start, start + 8))),
                                      ('stable_reverse', reversed_text, list(range(start + 7, start - 1, -1)))]:
            r = old[mode][i]['raw'][chunk]
            assert old[mode][i]['query_id'] == row['query_id']
            assert r['start'] == start and r['size'] == 8 and r['original_indices_in_prompt_order'] == positions
            assert r['original_prompt_sha256'] == digest(original) and r['condition_prompt_sha256'] == digest(text)
            assert 'TEST-GRADE-MUST-NOT-BE-IN-PROMPT' not in text
            checked += 1

def record(q, start, value):
    return {'query_id': q, 'start': start, 'size': 8,
            'indices': list(range(start, start + 8)), 'final': value}

before = [record('q1', 0, 'YNNNNNNY'), record('q2', 0, 'YYYYYYYY')]
after = [record('q1', 0, 'YNNYYNNY'), record('q2', 0, 'invalid')]
expected = {'comparable_chunks': 1, 'identical_final_chunks': 0,
            'comparable_candidates': 8, 'changed_candidates': 2, 'changed_queries': 1}
assert compare(after, before) == stats(after, before) == expected
bad = copy.deepcopy(after)
bad[0]['indices'] = list(reversed(bad[0]['indices']))
for method in (compare, stats):
    try:
        method(bad, before)
        raise AssertionError('Candidate mapping corruption was accepted')
    except AssertionError as exc:
        if str(exc) == 'Candidate mapping corruption was accepted':
            raise
print('Checked', checked, 'saved prompt hashes and mappings; gold-label isolation and comparison edge cases passed.')
