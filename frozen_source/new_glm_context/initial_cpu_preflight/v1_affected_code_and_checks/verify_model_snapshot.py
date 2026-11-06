#!/usr/bin/env python3
"""CPU-only, current observation of every repository file at the fixed revision.

No model/tensor/tokenizer imports. Public HTTPS uses an explicit no-proxy opener;
the process and global proxy environment are neither printed nor modified.
"""
import hashlib
import json
import time
import urllib.request
import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODEL = Path('/smoothcloud/models/glm-4-9b-chat')
REPOSITORY = 'THUDM/glm-4-9b-chat'
REVISION = 'bd8234fe5e0c09c48637a92abb0c797cb5fa0e73'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--metadata', type=Path)
    parser.add_argument('--metadata-provenance', type=Path)
    args = parser.parse_args()
    url = f'https://huggingface.co/api/models/{REPOSITORY}/revision/{REVISION}?blobs=true'
    started = time.time()
    if args.metadata:
        payload = args.metadata.read_bytes()
        metadata_url = url
        status = 'See preserved metadata acquisition provenance'
        method = 'Current official pinned Hugging Face API response retrieved by the root task via Firecrawl; copied byte-for-byte; local model hashes independently recomputed here'
        if args.metadata_provenance:
            shutil.copyfile(args.metadata_provenance, ROOT/'provenance'/'HF_METADATA_ACQUISITION_PROVENANCE.json')
    else:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(urllib.request.Request(url, headers={'User-Agent': 'IntentSat-context-integrity-audit/20261106'}), timeout=40) as response:
            payload = response.read()
            metadata_url = response.geturl()
            status = response.status
        method = 'urllib HTTPS with process-local ProxyHandler({}); no authentication; no global proxy change'
    metadata = json.loads(payload)
    assert metadata['sha'] == REVISION, metadata.get('sha')
    official = {r['rfilename']: r for r in metadata['siblings']}
    assert len(official) == 24, sorted(official)
    (ROOT/'provenance'/'HF_FIXED_REVISION_METADATA.json').write_bytes(payload)
    local = {str(p.relative_to(MODEL)): p for p in MODEL.rglob('*') if p.is_file() and '.hfd' not in p.relative_to(MODEL).parts}
    extra = sorted(set(local)-set(official))
    missing = sorted(set(official)-set(local))
    results = []
    for name in sorted(official):
        row = official[name]
        if name not in local:
            results.append({'file': name, 'status': 'MISSING'})
            continue
        path = local[name]
        size = path.stat().st_size
        stat_before = {'size': size, 'mtime_ns': path.stat().st_mtime_ns}
        sha256 = hashlib.sha256()
        blob = hashlib.sha1(f'blob {size}\0'.encode())
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(8 << 20), b''):
                sha256.update(chunk)
                blob.update(chunk)
        stat_after = {'size': path.stat().st_size, 'mtime_ns': path.stat().st_mtime_ns}
        lfs = row.get('lfs')
        observed = sha256.hexdigest() if lfs else blob.hexdigest()
        expected = lfs['sha256'] if lfs else row['blobId']
        result = {'file': name, 'size': size, 'official_size': row['size'],
                  'sha256': sha256.hexdigest(), 'git_blob_sha1': blob.hexdigest(),
                  'comparison': 'LFS SHA256' if lfs else 'Git blob SHA1',
                  'official_hash': expected,
                  'stat_before': stat_before, 'stat_after': stat_after,
                  'status': 'PASS' if observed == expected and size == row['size'] and stat_before == stat_after else 'FAIL'}
        results.append(result)
        print(name, result['status'], size, flush=True)
    receipt = {'created_unix': time.time(), 'observation_started_unix': started,
               'record_type': 'New verification from the current local snapshot and current public fixed-revision metadata; not a recovered historical certificate.',
               'repository_requested': REPOSITORY, 'repository_in_metadata': metadata.get('id'),
               'revision': REVISION, 'public_metadata_url': url, 'resolved_metadata_url': metadata_url,
               'http_status': status, 'network_method': method,
               'metadata_sha256': hashlib.sha256(payload).hexdigest(), 'model_path': str(MODEL),
               'official_files': len(official), 'local_repository_files': len(local),
               'excluded_local_download_bookkeeping': '.hfd directory (not read)',
               'missing': missing, 'extra': extra, 'files': results,
               'status': 'PASS' if not missing and not extra and all(r['status'] == 'PASS' for r in results) else 'FAIL'}
    (ROOT/'provenance'/'MODEL_SNAPSHOT_VERIFICATION.json').write_text(json.dumps(receipt, indent=2)+'\n')
    if receipt['status'] != 'PASS':
        raise SystemExit('Model snapshot mismatch; tokenizer/model execution not authorized by this check.')
    (ROOT/'prepared'/'MODEL_FILE_SHA256.json').write_text(json.dumps({'glm': {'path': str(MODEL), 'revision': REVISION,
        'files': {r['file']: r['sha256'] for r in results}}}, indent=2)+'\n')
    print('PASS: all 24 files match the fixed official revision; new verification only.', flush=True)


if __name__ == '__main__':
    main()
