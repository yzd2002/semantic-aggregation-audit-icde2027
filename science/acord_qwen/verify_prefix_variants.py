#!/usr/bin/env python3
"""Check fixed-pool prefix variants against source, and choose by recorded rule.

This is a data/tokenizer check only. It does not load model weights or predict.
"""
import argparse, hashlib, json
from pathlib import Path
from zipfile import ZipFile


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def rows(path):
    with path.open(encoding='utf8') as f:
        return [json.loads(line) for line in f if line.strip()]


def main(source, base, root, output):
    metadata = json.loads((base / 'selection.json').read_text())
    assert sha(source) == metadata['source_sha256']
    assert sha(base / 'inputs.jsonl') == metadata['inputs_sha256']
    original = rows(base / 'inputs.jsonl')
    with ZipFile(source) as z:
        paths = [n for n in z.namelist() if n.endswith('/corpus.jsonl')]
        assert len(paths) == 1
        with z.open(paths[0]) as f:
            docs = {}
            for row in map(json.loads, f):
                assert row['_id'] not in docs
                docs[row['_id']] = row['text']

    variants = {1200: ('selection_1200', 'preflight_1200', root / 'cap1200'),
                1800: ('selection_1800_corrected', 'preflight_1800', root / 'cap1800'),
                2400: ('selection_2400_corrected', 'preflight_2400', root / 'cap2400')}
    results = {}
    for cap, (selection_name, preflight_name, code) in variants.items():
        path = root / selection_name
        info = json.loads((path / 'selection.json').read_text())
        current = rows(path / 'inputs.jsonl')
        assert info['prefix_length'] == cap and sha(path / 'inputs.jsonl') == info['inputs_sha256']
        assert info['base_inputs_sha256'] == metadata['inputs_sha256']
        assert len(current) == len(original) == 114
        truncated = {'uniform': 0, 'case_control': 0}
        extended = 0
        for old, new in zip(original, current):
            assert {k: v for k, v in old.items() if k != 'items'} == {k: v for k, v in new.items() if k != 'items'}
            assert len(old['items']) == len(new['items']) == 32
            for before, after in zip(old['items'], new['items']):
                assert {k: v for k, v in before.items() if k != 'text'} == {k: v for k, v in after.items() if k != 'text'}
                full = docs[before['doc_id']]
                assert before['source_chars'] == len(full) and before['text'] == full[:600]
                assert after['text'] == full[:cap]
                truncated[new['scenario']] += len(full) > cap
                extended += before['text'] != after['text']
        assert extended == info['changed_candidate_occurrences']
        for name in truncated:
            assert truncated[name] == info['scenarios'][name]['truncated_occurrences']
        pre = root / preflight_name
        pf = json.loads((pre / 'summary.json').read_text())
        assert pf['selection_sha256'] == info['inputs_sha256']
        assert pf['prompt_code_sha256'] == sha(code / 'acord_prompts.py')
        assert pf['prompt_records_sha256'] == sha(pre / 'prompts.jsonl')
        assert pf['pool_count'] == 114 and pf['prompt_count'] == 4560
        assert max(pf['max_with_retry_and_output_cap'].values()) <= 4096 if pf['overflow_pool_count'] == 0 else True
        results[str(cap)] = {'selection_sha256': info['inputs_sha256'],
                             'preflight_summary_sha256': sha(pre / 'summary.json'),
                             'prompt_code_sha256': pf['prompt_code_sha256'],
                             'max_with_retry_and_output_cap': pf['max_with_retry_and_output_cap'],
                             'overflow_pool_count': pf['overflow_pool_count'],
                             'truncated_occurrences': truncated,
                             'extended_occurrences_from_600': extended}
    qualifying = [cap for cap in variants if results[str(cap)]['overflow_pool_count'] == 0]
    assert qualifying
    decision = {'scope': 'Fixed inputs and tokenizer preflight only; no model inference.',
                'rule': 'Largest cap in 1200, 1800, 2400 with zero overflowing pools.',
                'selected_cap_unicode_code_points': max(qualifying),
                'source_sha256': metadata['source_sha256'],
                'base_600_inputs_sha256': metadata['inputs_sha256'],
                'prefix_search_amendment_sha256': sha(root / 'AMENDMENT_prefix_search.md'),
                'variants': results}
    if output.exists():
        raise FileExistsError('Choose a fresh decision file')
    output.write_text(json.dumps(decision, indent=2) + '\n')
    print('Verified all 3 variants, all 114 fixed pools, source prefixes, identities and labels; selected', max(qualifying))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--base', type=Path, required=True)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    main(a.source, a.base, a.root, a.out)
