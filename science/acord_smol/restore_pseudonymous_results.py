#!/usr/bin/env python3
"""Restore result identifiers using a locally reconstructed ACORD selection.

The released IDs are unsalted SHA-256 digests, not anonymization. Source text
and judgments must be obtained separately. The result files are only written
to a fresh local directory and checked against recorded original hashes.
"""
import argparse, hashlib, json, shutil
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):h.update(b)
    return h.hexdigest()


def pseudonym(text):
    return hashlib.sha256(('IntentSat-ACORD-v1\0' + text).encode('utf8')).hexdigest()


def main(selection, redacted, manifest, out):
    if out.exists() or out.is_symlink():raise FileExistsError('Use a fresh local output directory')
    expected = json.loads(manifest.read_text(encoding='utf8'))
    data = selection / 'inputs.jsonl'
    if sha(data) != expected['selection_sha256']:raise ValueError('Reconstructed selection hash differs')
    with data.open(encoding='utf8') as f:rows = [json.loads(line) for line in f if line.strip()]
    reverse = {pseudonym(row['pool_id']): row['pool_id'] for row in rows}
    if len(reverse) != len(rows) or len(rows) != 114:raise ValueError('Unexpected pool IDs or digest collision')
    out.mkdir(parents=True, exist_ok=False)
    for name, wanted in expected['result_files'].items():
        if Path(name).name != name:raise ValueError('Unexpected result filename')
        source = redacted / name;dest = out / name
        if name.endswith('.jsonl'):
            with source.open(encoding='utf8') as inp, dest.open('w', encoding='utf8', newline='\n') as target:
                for line in inp:
                    row = json.loads(line);pid = row.get('pool_id')
                    if pid not in reverse:raise ValueError('Unknown pseudonymous pool ID')
                    row['pool_id'] = reverse[pid]
                    target.write(json.dumps(row, ensure_ascii=False) + '\n')
        else:
            shutil.copy2(source, dest)
        if sha(dest) != wanted:raise ValueError('Restored result hash mismatch: ' + name)
    print('Restored and byte-verified', len(expected['result_files']), 'result files in a fresh local directory.')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--selection', type=Path, required=True);p.add_argument('--redacted', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True);p.add_argument('--out', type=Path, required=True)
    a = p.parse_args();main(a.selection, a.redacted, a.manifest, a.out)
