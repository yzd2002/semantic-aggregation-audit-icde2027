#!/usr/bin/env python3
"""Create modular tar.xz streams split into <=8 MiB public blob files."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import tarfile

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE/'public'))
from portable_common import load, safe_name, sha, write_json

PART_BYTES = 512 << 10
DENIED = {'.venv', 'venv', '__pycache__', '.git', 'cache', 'runtime_cache', 'credentials', 'manuscript', 'submission'}
SECRET = [re.compile(rb'gh[pousr]_[A-Za-z0-9_]{25,}'), re.compile(rb'github_pat_[A-Za-z0-9_]{25,}'),
          re.compile(rb'sk-(?:proj-)?[A-Za-z0-9_-]{25,}'), re.compile(rb'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----')]


def review(bundle):
    files = []
    for p in sorted(bundle.rglob('*')):
        if p.is_symlink():
            raise ValueError('No symlinks may be published')
        if not p.is_file():
            continue
        rel = str(p.relative_to(bundle))
        safe_name(rel)
        if DENIED.intersection(Path(rel).parts) or p.suffix in ('.safetensors', '.gguf', '.pt', '.pth'):
            raise ValueError('Forbidden publication path: ' + rel)
        if p.suffix in ('.json', '.jsonl', '.py', '.txt', '.md'):
            with p.open('rb') as stream:
                tail = b''
                for block in iter(lambda: stream.read(4 << 20), b''):
                    if any(pattern.search(tail+block) for pattern in SECRET):
                        raise ValueError('Potential credential pattern; inspect filename only: ' + rel)
                    tail = block[-256:]
        # Public projection must contain no token arrays or raw input fields.
        if p.name in ('submitted_calls.jsonl', 'raw_return_snapshots.jsonl'):
            text = p.read_text()
            if re.search(r'"(?:prompt_token_ids|output_token_ids)"\s*:', text):
                raise ValueError('Actual token array leaked into compact projection: ' + rel)
        files.append((rel, p))
    return files


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--fixture', action='store_true')
    p.add_argument('--part-bytes', type=int, default=PART_BYTES, help='Maximum decoded public blob bytes; default 512 KiB for conservative GitHub App/Contents transport')
    a = p.parse_args()
    bundle, out = a.bundle.resolve(), a.out.resolve()
    if not 1 <= a.part_bytes <= 8 << 20:
        raise ValueError('Part size must be positive and at most 8 MiB')
    if not out.is_relative_to(HERE):
        raise ValueError('Archive writes must stay in portable_artifact')
    scope = load(bundle/'COMPACT_SCOPE.json')
    fixture = scope['payload_kind'] == 'SYNTHETIC_CPU_FIXTURE_NOT_MODEL_EVIDENCE'
    if fixture != a.fixture:
        raise ValueError('Fixture/actual distinction must remain explicit')
    report = bundle.parent/'relocation_replay/CPU_REPLAY_REPORT.json'
    if not report.is_file() or load(report).get('status') != 'PASS':
        raise ValueError('Actual compact relocation replay must PASS before building publication parts')
    files = review(bundle)
    for rel, expected in load(bundle/'FILE_SHA256.json')['files'].items():
        if sha(bundle/rel) != expected:
            raise ValueError('Compact file drift: ' + rel)
    out.mkdir(parents=True, exist_ok=False)
    private = out/'joined_streams_local_only';private.mkdir()
    parts_dir = out/'parts';parts_dir.mkdir()
    early = out/'EARLY_COMPLETE_MODULES';early.mkdir()
    modules = {}
    for rel, file in files:
        first = Path(rel).parts[0]
        is_run_directory = len(Path(rel).parts) > 1 and (bundle/first).is_dir() and first.startswith(('run_', 'budget_run_'))
        module = first if is_run_directory else 'failed_attempts' if first == 'failed_attempts' else 'foundation'
        modules.setdefault(module, []).append((rel, file))
    entries = []
    for name, selected in modules.items():
        stream_path = private/(name+'.tar.xz')
        with tarfile.open(stream_path, 'w:xz', preset=4, format=tarfile.PAX_FORMAT) as archive:
            for rel, file in selected:
                info = archive.gettarinfo(str(file), arcname=rel)
                info.uid = info.gid = 0;info.uname = info.gname = '';info.mode = 0o644
                with file.open('rb') as content:
                    archive.addfile(info, content)
        pieces = []
        with stream_path.open('rb') as source:
            for ordinal, data in enumerate(iter(lambda: source.read(a.part_bytes), b''), 1):
                part = parts_dir/f'{name}.tar.xz.part{ordinal:04d}'
                with part.open('xb') as sink:
                    sink.write(data)
                if part.stat().st_size > 8 << 20:
                    raise ValueError('Public part exceeds 8 MiB')
                pieces.append({'filename': part.name, 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest(),
                               'url': None, 'git_blob_sha': None})
        entry = {'module': name, 'archive_bytes': stream_path.stat().st_size,
                        'archive_sha256': sha(stream_path), 'uncompressed_bytes': sum(f.stat().st_size for _, f in selected),
                        'files': {rel: sha(f) for rel, f in selected}, 'parts': pieces}
        entries.append(entry)
        write_json(early/(name+'.json'), {
            'status': 'MODULE_CLOSED_CPU_BUNDLE_VERIFIED_UNPUBLISHED',
            'payload_kind': scope['payload_kind'], 'module': name,
            'archive_sha256': entry['archive_sha256'], 'archive_bytes': entry['archive_bytes'],
            'CPU_bundle_replay_report_sha256': sha(report),
            'parts': {'parts/'+part['filename']: {'bytes': part['bytes'], 'sha256': part['sha256']}
                      for part in pieces},
            'publication_performed': False, 'final_one_command_archive_replay_pending': True,
            'scope': 'Only closed, hash-measured module parts. Root may upload content-addressed blobs without updating any branch. The complete index and final safe-expansion CPU receipt follow separately.'})
    write_json(out/'artifact_index.template.json', {'schema_version': 1, 'published': False,
        'payload_kind': scope['payload_kind'], 'actual_publication_urls_verified': False,
        'maximum_part_bytes': a.part_bytes, 'complete_study_configurations': scope['actual_configurations'],
        'separate_budget_followup_policy_jobs': scope.get('budget_followup_policy_jobs', 0),
        'no_credentials_or_PAT_needed': True, 'modules': entries,
        'scope': 'Only files in parts/ plus public tool/source files are publication candidates. joined_streams_local_only is not uploaded.',
        'old_six_science_archives': 'Reuse immutable prior-v2 verified links; no duplicate payload is included.'})
    write_json(out/'PUBLIC_BLOB_WHITELIST.json', {'payload_kind': scope['payload_kind'],
        'files': {str(p.relative_to(out)): {'sha256': sha(p), 'bytes': p.stat().st_size}
                  for p in sorted(parts_dir.iterdir())}, 'all_parts_at_most_8MiB': True})
    print('Created explicitly inventoried public parts; no upload performed.', len(entries), 'modules')


if __name__ == '__main__':
    main()
