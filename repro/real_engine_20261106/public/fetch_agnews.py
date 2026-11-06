#!/usr/bin/env python3
"""Anonymous pinned AG News acquisition and exact local input reconstruction.

The upstream card declares license unknown. No new license is assigned to
news text or token IDs. Reconstructed text stays in the caller's local tree.
"""
import argparse
import json
from pathlib import Path
import shutil
import urllib.request
import uuid
from portable_common import compact, digest, lines, load, sha, write_json


PLACEHOLDER = '__EXACT_FULL_RECORD_TEXT__'


def reconstruct(plan, source, out):
    import pyarrow.parquet as pq
    if sha(source) != plan['source_sha256']:
        raise ValueError('Pinned source SHA256 differs; original source preserved')
    table = pq.read_table(source)
    if table.column_names != ['text', 'label'] or table.num_rows != 7600:
        raise ValueError('Unexpected fixed source schema/population')
    texts, labels = table['text'].to_pylist(), table['label'].to_pylist()
    if len(set(texts)) != len(texts) or any(not isinstance(t, str) or not t for t in texts):
        raise ValueError('Unexpected malformed/duplicate complete source text')
    ordered = sorted([{'record_id': f'agnews_test_{i}', 'fields': {'text': text}}
                      for i, text in enumerate(texts)],
                     key=lambda r: digest(plan['feature_seed']+'\0'+r['record_id']+'\0'+r['fields']['text']))
    label_map = {f'agnews_test_{i}': int(value) for i, value in enumerate(labels)}
    prepared = out/'prepared/agnews'
    prepared.mkdir(parents=True)
    for name, values in [('records.jsonl', ordered), ('gold.jsonl',
                          [{'record_id': r['record_id'], 'source_class': label_map[r['record_id']]} for r in ordered])]:
        with (prepared/name).open('x') as stream:
            for value in values:
                stream.write(compact(value)+'\n')
    for workload in plan['native_request_templates']:
        target = out/workload['relative_output']
        with target.open('x') as stream:
            for record in ordered:
                row = json.loads(json.dumps(workload['row_template'], ensure_ascii=False))
                row['record_id'] = record['record_id']
                for message in row['messages']:
                    message['content'] = message['content'].replace(PLACEHOLDER, record['fields']['text'])
                row['messages_sha256'] = digest(json.dumps(row['messages'], ensure_ascii=False, sort_keys=True, separators=(',', ':')))
                stream.write((compact(row) if workload['serialization'] == 'compact' else json.dumps(row, ensure_ascii=False))+'\n')
    verified = {}
    for rel, expected in plan['expected_reconstruction_sha256'].items():
        actual = sha(out/rel)
        if actual != expected:
            raise ValueError('Exact reconstructed input differs: ' + rel)
        verified[rel] = actual
    write_json(out/'EXACT_AGNEWS_RECONSTRUCTION.json', {'status': 'PASS', 'full_records': 7600,
        'truncation': False, 'network_used': plan.get('_network_used', False),
        'source_sha256': sha(source), 'exact_files': verified,
        'source_license': 'unknown as declared by publisher; no new license assigned',
        'new_model_inference': False, 'token_ID_reconstruction': 'Not required for mathematical replay; exact model tokenizer assets are separate licensed upstream assets.'})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--from-file', type=Path, help='Use already acquired exact pinned Parquet bytes; preserve the original')
    p.add_argument('--mirror', action='store_true', help='Use the explicitly recorded public mirror URL instead of the publisher host')
    a = p.parse_args()
    plan = load(a.plan)
    a.out.mkdir(parents=True, exist_ok=False)
    source = a.out/'test-00000-of-00001.parquet'
    if a.from_file:
        if sha(a.from_file) != plan['source_sha256']:
            raise ValueError('Existing source differs from pinned hash')
        shutil.copyfile(a.from_file, source)
    else:
        url = plan['recorded_mirror_url'] if a.mirror else plan['publisher_url']
        partial = a.out/('source.attempt-'+uuid.uuid4().hex+'.part')
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        try:
            with opener.open(url, timeout=60) as response, partial.open('xb') as stream:
                shutil.copyfileobj(response, stream, 1 << 20)
        except Exception as error:
            raise RuntimeError('Anonymous source download failed; unique partial retained; type='+type(error).__name__) from None
        if sha(partial) != plan['source_sha256']:
            raise ValueError('Downloaded source hash differs; unique attempt retained')
        partial.rename(source)
        plan['_network_used'] = True
    reconstruct(plan, source, a.out)
    print('PASS: exact pinned full AG News records, labels and native request JSONL reconstructed locally.')


if __name__ == '__main__':
    main()
