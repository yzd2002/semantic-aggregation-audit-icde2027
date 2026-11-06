#!/usr/bin/env python3
"""Public, credential-free artifact download, SHA-256 verification and safe extraction."""
import argparse
import hashlib
import http.client
import json
import re
import shutil
import tarfile
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path, PurePosixPath


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 << 20), b''):
            h.update(block)
    return h.hexdigest()


def safe_relative(name):
    p = PurePosixPath(name)
    if not name or p.is_absolute() or '..' in p.parts or '\\' in name:
        raise ValueError('Unsafe archive/index path: ' + name)
    return p


def selected(index, names):
    entries = index['assets']
    if names:
        wanted = set(names)
        entries = [x for x in entries if x['module'] in wanted]
        if {x['module'] for x in entries} != wanted:
            raise ValueError('Requested module is absent from this index')
    if not entries:
        raise ValueError('No assets selected')
    return entries


def check_asset(path, entry):
    if path.stat().st_size != entry['bytes'] or sha(path) != entry['sha256']:
        raise ValueError('Archive byte size or SHA-256 mismatch: ' + path.name)


def download(index, entries, directory, direct=False):
    if not index.get('published'):
        raise ValueError('This is an unpublished index; no public release is asserted')
    directory.mkdir(parents=True, exist_ok=True)
    # Process-local opener only; default TLS validation stays enabled. No
    # authentication, cookie jar, global proxy edits or signed redirect logging.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({})) if direct else urllib.request.build_opener()
    for e in entries:
        filename = e['filename']
        if str(safe_relative(filename)) != Path(filename).name:
            raise ValueError('Archive filename must be a basename')
        url = e.get('url')
        parsed = urllib.parse.urlparse(url or '')
        if parsed.scheme != 'https' or not parsed.hostname or parsed.username or parsed.password:
            raise ValueError('A public credential-free HTTPS URL is required')
        target = directory / filename
        if target.exists():
            check_asset(target, e)
            print('Already verified:', filename, flush=True)
            continue
        fallback = e.get('public_asset_api_url')
        if fallback is None:
            global_api = index.get('public_asset_api_url')
            fallback = global_api.get(filename, global_api.get(e['module'])) if isinstance(global_api, dict) else global_api
        if fallback:
            api = urllib.parse.urlparse(fallback)
            if (api.scheme != 'https' or api.hostname != 'api.github.com' or api.username or api.password or api.query or api.fragment
                    or re.fullmatch(r'/repos/[^/]+/[^/]+/releases/assets/[0-9]+', api.path) is None):
                raise ValueError('Fallback must be an unsigned public GitHub asset API HTTPS URL')
        candidates = [(url, False)] + ([(fallback, True)] if fallback else [])
        for position, (candidate, is_api) in enumerate(candidates):
            temporary = directory / (filename + '.attempt-' + uuid.uuid4().hex + '.part')
            headers = {'User-Agent': 'semantic-count-artifact-public-download/1.0'}
            if is_api:
                headers['Accept'] = 'application/octet-stream'
            try:
                with temporary.open('xb') as f:
                    request = urllib.request.Request(candidate, headers=headers)
                    with opener.open(request, timeout=60) as response:
                        shutil.copyfileobj(response, f, 4 << 20)
                check_asset(temporary, e)
                if target.exists():
                    raise ValueError('Destination appeared during download; verified attempt retained')
                temporary.rename(target)
                break
            except Exception as error:
                code = error.code if isinstance(error, urllib.error.HTTPError) else None
                reason = error.reason if isinstance(error, urllib.error.URLError) else error
                can_fallback = code == 403 or isinstance(reason, http.client.RemoteDisconnected)
                print('Download attempt retained:', temporary.name, 'error type:', type(error).__name__,
                      'HTTP status:', code, flush=True)
                # Never interpolate error URLs or signed redirect query strings.
                if position == 0 and fallback and can_fallback:
                    print('Trying anonymous public asset API:', filename, flush=True)
                    continue
                raise RuntimeError('Anonymous download failed for ' + filename + '; attempt preserved') from None
        print('Downloaded and verified:', filename, flush=True)


def verify_files(root, module):
    manifest = root / '_artifact_manifests' / (module + '.json')
    document = json.loads(manifest.read_text())
    for rel, record in document['files'].items():
        safe_relative(rel)
        p = root / rel
        if not p.is_file() or p.is_symlink() or p.stat().st_size != record['bytes'] or sha(p) != record['sha256']:
            raise ValueError('Expanded file mismatch: ' + rel)
    return len(document['files'])


def extract(entries, directory, out):
    # Preflight every archive and every target before writing. Links and special
    # files are rejected; only explicitly inventoried regular files are shipped.
    plans, seen = [], set()
    for e in entries:
        archive = directory / e['filename']
        check_asset(archive, e)
        with tarfile.open(archive, 'r:xz') as t:
            members = t.getmembers()
            if len(members) > 200000:
                raise ValueError('Unreasonable archive member count')
            expanded = 0
            for m in members:
                rel = str(safe_relative(m.name))
                if not m.isfile() or rel in seen:
                    raise ValueError('Special, linked or duplicate member: ' + rel)
                if m.size < 0:
                    raise ValueError('Negative member size')
                seen.add(rel)
                expanded += m.size
            if expanded != e['uncompressed_bytes']:
                raise ValueError('Unexpected expanded archive size')
            plans.append((archive, members, e))
    out.mkdir(parents=True, exist_ok=False)
    for archive, members, e in plans:
        with tarfile.open(archive, 'r:xz') as t:
            for m in members:
                target = out / m.name
                target.parent.mkdir(parents=True, exist_ok=True)
                with t.extractfile(m) as source, target.open('xb') as sink:
                    shutil.copyfileobj(source, sink, 4 << 20)
        n = verify_files(out, e['module'])
        print('Safely expanded and verified:', e['module'], n, 'files', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['download', 'verify', 'extract'])
    p.add_argument('--index', type=Path, required=True)
    p.add_argument('--archives-dir', type=Path, required=True)
    p.add_argument('--modules', nargs='+')
    p.add_argument('--out', type=Path)
    p.add_argument('--direct', action='store_true', help='Bypass proxy variables for this download process only; retain TLS verification')
    a = p.parse_args()
    if not a.index.is_file():
        p.error('Download the verified asset_index.json attachment from the matching GitHub release first. The immutable scientific tag may contain only asset_index.template.json; its unpublished template cannot certify public downloads.')
    index = json.loads(a.index.read_text())
    entries = selected(index, a.modules)
    if a.action == 'download':
        download(index, entries, a.archives_dir, direct=a.direct)
    elif a.action == 'verify':
        for e in entries:
            check_asset(a.archives_dir / e['filename'], e)
            print('Verified:', e['filename'])
    else:
        if a.out is None:
            p.error('--out is required for extract; it must not already exist')
        extract(entries, a.archives_dir, a.out.resolve())


if __name__ == '__main__':
    main()
