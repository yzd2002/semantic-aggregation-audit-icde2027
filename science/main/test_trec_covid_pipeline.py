#!/usr/bin/env python3
"""Exercise the complete saved-trace analysis and independent verifier on fabricated data."""
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from run_trec_covid_transfer import SYSTEM, RETRY_SUFFIX, keyed_prompt, point_prompt, sha


def write_lines(path, rows):
    path.write_text(''.join(json.dumps(x) + '\n' for x in rows))


def main():
    base = Path(tempfile.mkdtemp(prefix='trec-covid-synthetic-', dir='/dev/shm'))
    selection, run = base / 'selection', base / 'run'
    selection.mkdir(); run.mkdir()
    rows = []
    for q in range(1, 4):
        items = [{'doc_id': 'doc' + str(q * 16 + i), 'title': 'alpha ' + str(i),
                  'abstract': 'beta text', 'grade': 2 if i % 3 == 0 else 0,
                  'gold': 'Y' if i % 3 == 0 else 'N'} for i in range(32)]
        rows.append({'topic_id': q, 'query': 'alpha', 'question': 'a question',
                     'narrative': 'some context', 'items': items})
    write_lines(selection / 'inputs.jsonl', rows)
    names = ('original', 'stable_reverse', 'selective_pointwise', 'replay', 'pointwise')
    records = {k: [] for k in names}
    outputs = []
    def rec(row, start, size, presented, prompt, value):
        return {'topic_id': row['topic_id'], 'start': start, 'size': size, 'presented_ids': presented,
                'prompt_sha256': sha(prompt), 'wrapped_sha256': 'not-used-by-this-test',
                'first': value, 'final': value, 'first_finish_reason': 'stop', 'retry_finish_reason': None,
                'first_input_tokens': 1, 'first_output_tokens': 1, 'retry_input_tokens': 0,
                'retry_output_tokens': 0}
    def encode(vals):
        return '{' + ','.join('"%d":"%s"' % (i + 1, value) for i, value in enumerate(vals)) + '}'
    for row in rows:
        out = []
        for start in range(0, 32, 8):
            original = ['Y' if i % 4 == 0 else 'N' for i in range(start, start + 8)]
            reverse = [('N' if x == 'Y' else 'Y') if i % 5 == 0 else x
                       for i, x in zip(range(start, start + 8), original)]
            for name, labels, reversed_order in [('original', original, False), ('replay', original, False),
                                                  ('stable_reverse', reverse, True)]:
                p, presented = keyed_prompt(row, start, reversed_order)
                records[name].append(rec(row, start, 8, presented, p, encode(labels)))
            for i, (x, y) in enumerate(zip(original, reverse), start=start):
                full = 'Y' if i % 3 == 0 else 'N'
                records['pointwise'].append(rec(row, i, 1, None, point_prompt(row, i), full))
                if x != y:
                    records['selective_pointwise'].append(rec(row, i, 1, None, point_prompt(row, i), 'N'))
                    out.append('N')
                else: out.append(x)
        outputs.append({'topic_id': row['topic_id'], 'pred': out})
    for name, rs in records.items(): write_lines(run / (name + '_responses.jsonl'), rs)
    write_lines(run / 'selective_outputs.jsonl', outputs)
    stats = {}
    for name, rs in records.items():
        n = len(rs)
        stats[name] = {'initial_calls': n, 'retry_calls': 0, 'valid_first': n, 'valid_final': n,
                       'first_finish_reasons': {'stop': n}, 'logical_tokens': {'input': n, 'output': n},
                       'max_prompt_with_retry_plus_cap': 100, 'offline_generation_seconds': 0}
    names_used = ('original', 'stable_reverse', 'selective_pointwise')
    pipeline_calls = sum(stats[n]['initial_calls'] for n in names_used)
    summary = {'system': SYSTEM, 'retry_suffix': RETRY_SUFFIX,
               'selection_inputs_sha256': hashlib.sha256((selection / 'inputs.jsonl').read_bytes()).hexdigest(),
               'modes': stats, 'selective_pipeline': {'complete_pools': 3, 'initial_calls': pipeline_calls,
                   'retry_calls': 0, 'logical_tokens': {'input': pipeline_calls, 'output': pipeline_calls}}}
    (run / 'summary.json').write_text(json.dumps(summary))
    keyed = encode(['Y'] * 8)
    (run / 'guidance_sanity.json').write_text(json.dumps({'ordinary': {'text': 'BANANA'},
        'keyed': {'text': keyed}, 'pointwise': {'text': 'Y'}, 'checks': {'ok': True}}))
    directory = Path(__file__).resolve().parent
    analysis = base / 'analysis.json'
    subprocess.run([sys.executable, str(directory / 'analyze_trec_covid_transfer.py'), '--selection', str(selection),
                    '--run', str(run), '--out', str(analysis)], check=True)
    subprocess.run([sys.executable, str(directory / 'verify_trec_covid_transfer.py'), '--selection', str(selection),
                    '--run', str(run), '--analysis', str(analysis)], check=True)
    print('Fabricated end-to-end trace analysis and independent verification passed.')


if __name__ == '__main__':
    main()
