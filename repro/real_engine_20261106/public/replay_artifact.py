#!/usr/bin/env python3
"""One-command anonymous acquisition, safe extraction and CPU replay."""
import argparse
import base64
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import urllib.parse
import urllib.request
import uuid
from portable_common import load, safe_name, sha, write_json


def part_file(entry, local, downloads):
    filename = entry['filename']
    if safe_name(filename) != Path(filename).name:
        raise ValueError('Part filename must be a basename')
    target = local/filename if local else downloads/filename
    if target.exists():
        if target.stat().st_size != entry['bytes'] or sha(target) != entry['sha256']:
            raise ValueError('Existing part differs; preserved: ' + filename)
        return target
    if local:
        raise ValueError('Missing local part: ' + filename)
    url = entry.get('url');parsed = urllib.parse.urlparse(url or '')
    if parsed.scheme != 'https' or parsed.username or parsed.password:
        raise ValueError('A verified public anonymous HTTPS part URL is required')
    partial = downloads/(filename+'.attempt-'+uuid.uuid4().hex+'.part')
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    headers = {'User-Agent': 'semantic-count-public-cpu-replay/1.0'}
    if parsed.hostname == 'api.github.com' and '/git/blobs/' in parsed.path:
        headers['Accept'] = 'application/vnd.github.raw+json'
    try:
        with opener.open(urllib.request.Request(url, headers=headers), timeout=60) as source, partial.open('xb') as sink:
            shutil.copyfileobj(source, sink, 1 << 20)
    except Exception as error:
        raise RuntimeError('Anonymous part download failed; unique partial preserved; type='+type(error).__name__) from None
    if partial.stat().st_size != entry['bytes'] or sha(partial) != entry['sha256']:
        # Some public Git-blob API proxies return JSON despite the raw Accept.
        try:
            blob = json.loads(partial.read_text())
            decoded = base64.b64decode(blob['content']) if blob.get('encoding') == 'base64' else None
        except (ValueError, KeyError, UnicodeDecodeError):
            decoded = None
        if decoded is None:
            raise ValueError('Downloaded part mismatch; attempt preserved: ' + filename)
        alternate = downloads/(filename+'.attempt-'+uuid.uuid4().hex+'.decoded')
        with alternate.open('xb') as sink:
            sink.write(decoded)
        partial = alternate
    if partial.stat().st_size != entry['bytes'] or sha(partial) != entry['sha256']:
        raise ValueError('Downloaded/decoded part mismatch; attempts preserved: ' + filename)
    if target.exists():
        raise ValueError('Target appeared during download; successful attempt preserved')
    partial.rename(target)
    return target


def extract(index, local, out):
    out.mkdir(parents=True, exist_ok=False)
    downloads, streams = out/'downloaded_parts', out/'joined_streams'
    downloads.mkdir();streams.mkdir()
    root = out/'expanded'
    root.mkdir()
    plans, seen = [], set()
    for module in index['modules']:
        if not isinstance(module['module'], str) or not module['module'] or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in module['module']):
            raise ValueError('Unsafe module name')
        stream_path = streams/(module['module']+'.tar.xz')
        with stream_path.open('xb') as stream:
            for entry in module['parts']:
                if type(entry['bytes']) is not int or not 0 <= entry['bytes'] <= 8 << 20:
                    raise ValueError('Indexed public part exceeds 8 MiB')
                piece = part_file(entry, local, downloads)
                with piece.open('rb') as source:
                    shutil.copyfileobj(source, stream, 1 << 20)
        if stream_path.stat().st_size != module['archive_bytes'] or sha(stream_path) != module['archive_sha256']:
            raise ValueError('Joined archive hash/length differs')
        with tarfile.open(stream_path, 'r:xz') as archive:
            members = archive.getmembers()
            if len(members) > 200000:
                raise ValueError('Unreasonable archive member count')
            total = 0
            for member in members:
                rel = safe_name(member.name)
                if not member.isfile() or rel in seen or rel not in module['files']:
                    raise ValueError('Uninventoried, linked, special or repeated archive member')
                total += member.size;seen.add(rel)
            if total != module['uncompressed_bytes'] or len(members) != len(module['files']):
                raise ValueError('Archive expanded size/count differs')
            plans.append((stream_path, members, module))
    # All entries preflighted before scientific files are written.
    for stream_path, members, module in plans:
        with tarfile.open(stream_path, 'r:xz') as archive:
            for member in members:
                target = root/member.name
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.extractfile(member) as source, target.open('xb') as sink:
                    shutil.copyfileobj(source, sink, 1 << 20)
                if sha(target) != module['files'][member.name]:
                    raise ValueError('Expanded file hash mismatch: ' + member.name)
    return root


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--index', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--parts-dir', type=Path, help='Local verified parts; no network is used')
    p.add_argument('--extract-only', action='store_true')
    a = p.parse_args()
    if not a.index.is_file():
        p.error('Obtain the exact public artifact index first; an unpublished template certifies no live URLs.')
    index = load(a.index)
    if not a.parts_dir and index.get('published') is not True:
        raise ValueError('Unpublished index cannot initiate public downloads')
    root = extract(index, a.parts_dir, a.out)
    if not a.extract_only:
        command = [sys.executable, '-B', str(root/'_portable_tools/replay_compact.py'), '--bundle', str(root), '--out', str(a.out/'cpu_replay')]
        with (a.out/'CPU_REPLAY.log').open('x') as stream:
            subprocess.run(command, check=True, stdout=stream, stderr=subprocess.STDOUT)
        write_json(a.out/'ONE_COMMAND_RESULT.json', {'status': 'PASS', 'payload_kind': index['payload_kind'],
            'safe_extraction': True, 'CPU_replay': load(a.out/'cpu_replay/CPU_REPLAY_REPORT.json'),
            'public_anonymous_download_performed': a.parts_dir is None})
    print('PASS: public parts verified, safely expanded' + (' and CPU replay matched.' if not a.extract_only else '.'))


if __name__ == '__main__':
    main()
