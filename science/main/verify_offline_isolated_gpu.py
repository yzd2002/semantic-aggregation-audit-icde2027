#!/usr/bin/env python3
"""Independently hash copied package files in source and isolated destination."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path, PurePosixPath


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--site', type=Path, required=True)
    p.add_argument('--copy', type=Path, required=True)
    p.add_argument('--graph', type=Path, required=True)
    a = p.parse_args()
    site = a.site.resolve()
    summary = json.loads((a.copy / 'copy_summary.json').read_text())
    manifest = a.copy / 'copied_files.jsonl.gz'
    assert digest(manifest) == summary['manifest_sha256']
    assert digest(a.graph) == summary['graph_sha256']
    graph = json.loads(a.graph.read_text())
    assert len(graph) == summary['packages']
    count, seen = 0, set()
    with gzip.open(manifest, 'rt', encoding='utf8') as f:
        for line in f:
            r = json.loads(line)
            relative = PurePosixPath(r['target'])
            assert not relative.is_absolute() and '..' not in relative.parts
            target = site.joinpath(*relative.parts).resolve()
            assert target.is_relative_to(site) and r['distribution'] in graph
            source = Path(r['source'])
            assert source.is_absolute() and source.is_file() and target.is_file()
            assert digest(source) == r['sha256'] == digest(target), r['target']
            seen.add(r['distribution'])
            count += 1
    assert count == summary['file_records'] and seen == set(graph)
    print('Independently verified source and destination hashes for', count,
          'files from', len(seen), 'distributions.')


if __name__ == '__main__':
    main()
