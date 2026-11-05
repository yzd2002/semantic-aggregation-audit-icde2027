#!/usr/bin/env python3
"""Exercise independent package-copy verification with safe synthetic files."""
import gzip
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

verify = Path(__file__).with_name('verify_offline_isolated_gpu.py')
root = Path(tempfile.mkdtemp(prefix='intentsat-offline-verify-test-', dir='/dev/shm'))
site, saved = root / 'site', root / 'saved'
site.mkdir()
saved.mkdir()
source, target = root / 'source', site / 'sample.txt'
source.write_bytes(b'synthetic package file')
target.write_bytes(source.read_bytes())
graph = root / 'graph.json'
graph.write_text(json.dumps({'sample': {'version': '1'}}))
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
manifest = saved / 'copied_files.jsonl.gz'


def prepare(path):
    with gzip.open(manifest, 'wt', encoding='utf8') as f:
        f.write(json.dumps({'distribution': 'sample', 'source': str(source),
                            'target': path, 'sha256': sha(source)}) + '\n')
    (saved / 'copy_summary.json').write_text(json.dumps({'packages': 1, 'file_records': 1,
        'manifest_sha256': sha(manifest), 'graph_sha256': sha(graph)}))


def check():
    return subprocess.run([sys.executable, str(verify), '--site', str(site),
                           '--copy', str(saved), '--graph', str(graph)],
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE).returncode


prepare('sample.txt')
assert check() == 0
target.write_bytes(b'tampered file')
assert check() != 0
target.write_bytes(source.read_bytes())
prepare('../source')
assert check() != 0
print('Synthetic matching files passed; modified destination and traversal were rejected.')
