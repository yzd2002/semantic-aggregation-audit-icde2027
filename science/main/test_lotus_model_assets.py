#!/usr/bin/env python3
"""Check success, missing-file and changed-file behavior on synthetic assets."""
import hashlib
import subprocess
import sys
import tempfile
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent / 'verify_lotus_model_assets.py'


def digest(data):
    return hashlib.sha256(data).hexdigest()


with tempfile.TemporaryDirectory(prefix='intentsat-model-hash-test-') as tmp:
    root = Path(tmp)
    for name in ('main', 'helper', 'manifests'):
        (root / name).mkdir()
    main_weight, main_config = b'synthetic weight', b'synthetic configuration'
    helper_weight, helper_config = b'synthetic helper', b'synthetic helper configuration'
    (root / 'main/part.safetensors').write_bytes(main_weight)
    (root / 'main/config.json').write_bytes(main_config)
    (root / 'helper/model.safetensors').write_bytes(helper_weight)
    (root / 'helper/config.json').write_bytes(helper_config)
    (root / 'manifests/MODEL_SHA256.txt').write_text(f'{digest(main_weight)}  /old/path/part.safetensors\n')
    (root / 'manifests/MODEL_QWEN_ASSETS_SHA256.txt').write_text(f'{digest(main_config)}  config.json\n')
    (root / 'manifests/MODEL_SENSITIVITY_SHA256.txt').write_text(
        f'Expected official model.safetensors SHA-256: {digest(helper_weight)}\n'
        f'{digest(helper_config)}  /old/path/config.json\n')
    cmd = [sys.executable, str(SCRIPT), '--main-model', str(root / 'main'),
           '--helper-model', str(root / 'helper'), '--manifest-dir', str(root / 'manifests')]
    good = subprocess.run(cmd, capture_output=True, text=True)
    assert good.returncode == 0 and '2 main-model files and 2 helper-model files' in good.stdout
    (root / 'helper/config.json').write_bytes(b'changed')
    changed = subprocess.run(cmd, capture_output=True, text=True)
    assert changed.returncode != 0 and 'helper/config.json: SHA-256 mismatch' in changed.stdout
    (root / 'main/part.safetensors').unlink()
    missing = subprocess.run(cmd, capture_output=True, text=True)
    assert missing.returncode != 0 and 'main/part.safetensors: missing' in missing.stdout
print('Synthetic model-hash pass, changed-file and missing-file checks succeeded.')
