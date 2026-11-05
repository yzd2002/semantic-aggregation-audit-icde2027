#!/usr/bin/env python3
"""Copy a frozen, dependency-checked host package closure into a fresh venv."""
import argparse
import gzip
import hashlib
import json
import os
import shutil
from collections import deque
from importlib import metadata
from pathlib import Path, PurePosixPath

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

ROOTS = ('vllm', 'torch', 'transformers', 'numpy', 'xgrammar', 'outlines-core',
         'setuptools', 'pandas', 'pyarrow', 'zstandard', 'psutil')


def select(overlay_path):
    overlay = {canonicalize_name(d.metadata['Name']): d
               for d in metadata.distributions(path=[str(overlay_path)])}
    queue = deque((Requirement(n), '<root>') for n in ROOTS)
    selected, extras = {}, {}
    while queue:
        req, parent = queue.popleft()
        name = canonicalize_name(req.name)
        if name not in selected:
            try:
                dist = overlay.get(name) or metadata.distribution(name)
            except metadata.PackageNotFoundError as exc:
                raise RuntimeError(f'Missing dependency {name} required by {parent}') from exc
            selected[name], extras[name], new = dist, set(), True
        else:
            dist, new = selected[name], False
        if req.specifier and not req.specifier.contains(dist.version, prereleases=True):
            raise RuntimeError(f'Unsatisfied {parent} dependency {req}; installed {dist.version}')
        added = set(req.extras) - extras[name]
        if not new and not added:
            continue
        extras[name].update(added)
        for raw in dist.requires or []:
            child = Requirement(raw)
            if child.marker and not any(child.marker.evaluate({'extra': x})
                                        for x in ({''} | extras[name])):
                continue
            queue.append((child, name))
    graph = {name: {'version': dist.version, 'extras': sorted(extras[name]),
                    'source': 'overlay' if name in overlay else 'host'}
             for name, dist in sorted(selected.items())}
    return graph, selected


def file_hash(path):
    h = hashlib.sha256()
    with path.open('rb') as src:
        for block in iter(lambda: src.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--overlay', type=Path, required=True)
    ap.add_argument('--graph', type=Path, required=True)
    ap.add_argument('--site', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    if a.out.exists() or not a.site.is_dir():
        raise ValueError('Use a new output directory and an existing fresh venv site directory')
    graph, selected = select(a.overlay)
    if graph != json.loads(a.graph.read_text()):
        raise ValueError('Dependency closure or version changed since the graph was frozen')
    site = a.site.resolve()
    a.out.mkdir(parents=True, exist_ok=False)
    manifest = a.out / 'copied_files.jsonl.gz'
    skipped, count, total, shared = [], 0, 0, 0
    with gzip.open(manifest, 'xt', encoding='utf8') as records:
        for name, dist in sorted(selected.items()):
            base = Path(dist.locate_file('')).resolve()
            if dist.files is None:
                raise ValueError(f'Missing installed file metadata for {name}')
            for file in dist.files:
                relative = PurePosixPath(str(file))
                if relative.is_absolute() or '..' in relative.parts:
                    skipped.append({'distribution': name, 'path': str(relative)})
                    continue
                source = Path(dist.locate_file(file)).resolve()
                if not source.is_relative_to(base):
                    raise ValueError('A package file resolves outside its source site directory')
                if not source.is_file():
                    raise FileNotFoundError(f'Missing {name} file {relative}')
                target = site.joinpath(*relative.parts)
                if not target.resolve().is_relative_to(site):
                    raise ValueError('Destination path escapes the isolated site directory')
                target.parent.mkdir(parents=True, exist_ok=True)
                h = hashlib.sha256()
                if target.exists():
                    if not target.is_file() or file_hash(source) != file_hash(target):
                        raise ValueError(f'Conflicting destination file {relative}')
                    digest = file_hash(source)
                    shared += 1
                else:
                    with source.open('rb') as src, target.open('xb') as dst:
                        for block in iter(lambda: src.read(1024 * 1024), b''):
                            h.update(block)
                            dst.write(block)
                    shutil.copymode(source, target)
                    digest = h.hexdigest()
                    total += source.stat().st_size
                records.write(json.dumps({'distribution': name, 'source': str(source),
                    'target': relative.as_posix(), 'sha256': digest}, ensure_ascii=True) + '\n')
                count += 1
    summary = {'packages': len(selected), 'file_records': count, 'shared_file_records': shared,
               'copied_bytes': total, 'skipped_external_entries': skipped,
               'graph_sha256': file_hash(a.graph), 'manifest_sha256': file_hash(manifest),
               'interpretation': 'Local installed-file copies; not a public-wheel or fresh-machine installation.'}
    (a.out / 'copy_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print('Copied', count, 'file records from', len(selected), 'selected distributions;',
          'bytes:', total, 'external entries omitted:', len(skipped))


if __name__ == '__main__':
    main()
