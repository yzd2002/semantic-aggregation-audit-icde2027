#!/usr/bin/env python3
"""Compare locally supplied LOTUS model assets with recorded experiment hashes.

This verifies the files supplied on disk, not the identity of a running server.
"""
import argparse
import hashlib
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
HASH_LINE = re.compile(r'^([0-9a-f]{64})\s+(.+?)\s*$')
HELPER_WEIGHT = re.compile(r'^Expected official model\.safetensors SHA-256:\s*([0-9a-f]{64})\s*$')


def read_hash_lines(path):
    values = {}
    for line in path.read_text(encoding='utf8').splitlines():
        match = HASH_LINE.fullmatch(line)
        if not match:
            continue
        name = Path(match.group(2)).name
        if not name or name in values:
            raise ValueError(f'Invalid or duplicate filename in {path.name}')
        values[name] = match.group(1)
    if not values:
        raise ValueError(f'No file hashes found in {path.name}')
    return values


def expected_hashes(manifest_dir):
    main = read_hash_lines(manifest_dir / 'MODEL_SHA256.txt')
    assets = read_hash_lines(manifest_dir / 'MODEL_QWEN_ASSETS_SHA256.txt')
    if main.keys() & assets.keys():
        raise ValueError('Main-model manifests have duplicate filenames')
    main.update(assets)
    helper_path = manifest_dir / 'MODEL_SENSITIVITY_SHA256.txt'
    helper = read_hash_lines(helper_path)
    matches = [m.group(1) for line in helper_path.read_text(encoding='utf8').splitlines()
               if (m := HELPER_WEIGHT.fullmatch(line))]
    if len(matches) != 1 or 'model.safetensors' in helper:
        raise ValueError('Expected exactly one helper weight hash')
    helper['model.safetensors'] = matches[0]
    return {'main': main, 'helper': helper}


def file_hash(path):
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--main-model', type=Path, required=True)
    p.add_argument('--helper-model', type=Path, required=True)
    p.add_argument('--manifest-dir', type=Path, default=HERE,
                   help='Directory containing the three recorded model hash files')
    args = p.parse_args()
    expected = expected_hashes(args.manifest_dir)
    roots = {'main': args.main_model, 'helper': args.helper_model}
    errors = []
    for role in ('main', 'helper'):
        for name, wanted in sorted(expected[role].items()):
            path = roots[role] / name
            if not path.is_file():
                errors.append(f'{role}/{name}: missing')
            elif file_hash(path) != wanted:
                errors.append(f'{role}/{name}: SHA-256 mismatch')
    if errors:
        print('\n'.join(errors))
        raise SystemExit(1)
    print(f"Recorded hashes verified: {len(expected['main'])} main-model files and "
          f"{len(expected['helper'])} helper-model files.")
    print('This does not check unlisted files or prove which weights a server loaded.')


if __name__ == '__main__':
    main()
