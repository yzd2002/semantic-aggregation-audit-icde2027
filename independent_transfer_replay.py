#!/usr/bin/env python3
"""Explicit path adapter for the unchanged real-transfer mathematical audit."""
import argparse
import ast
import collections
import hashlib
import importlib.util
import json
import math
import os
import pathlib
import re
import shutil
import statistics
import sys
import time
import traceback
from fractions import Fraction
from pathlib import Path
from unittest.mock import patch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit', type=Path, required=True)
    p.add_argument('--transfer', type=Path, required=True)
    p.add_argument('--model', choices=('qwen', 'smol'), required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    audit, transfer, out = [x.resolve() for x in (a.audit, a.transfer, a.out)]
    source = audit / 'independent_real_transfer_audit.py'
    expected = 'e68d5a7f8bda0c00705105737eb4bf1931ed10897f2947d2f69e7eadaaf7890b'
    if hashlib.sha256(source.read_bytes()).hexdigest() != expected:
        raise ValueError('Real-transfer independent source changed')
    old_root = '/home/gpu/IntentSat2S_revision_20261106/'
    mappings = {old_root + 'new_transfer_audit': transfer,
                old_root + 'independent_transfer_preflight_review': audit / 'helpers'}
    real_path = type(Path())

    def mapped_path(*parts):
        value = real_path(*parts)
        text = str(value)
        for recorded, target in mappings.items():
            if text == recorded:
                return target
            if text.startswith(recorded + '/'):
                return target / text[len(recorded) + 1:]
        return value

    # All dependencies are standard-library modules imported above. The
    # original module obtains its Path constructor while this explicit adapter
    # is active; top-level helper hash checks then use the relocated helpers.
    # The source bytes and all its AST-extracted mathematical functions remain
    # unchanged. pathlib itself is restored immediately after loading.
    spec = importlib.util.spec_from_file_location('frozen_real_transfer_audit', source)
    module = importlib.util.module_from_spec(spec)
    with patch.object(pathlib, 'Path', mapped_path):
        spec.loader.exec_module(module)
    sys.dont_write_bytecode = True
    sys.argv = [str(source), '--model', a.model, '--out', str(out)]
    module.main()
    report = json.loads((out / 'RESULTS_CERTIFICATE.json').read_text())
    if report.get('status') != 'PASS' or report.get('comparison_mismatches') != 0:
        raise ValueError('Actual independent transfer replay did not pass')
    (out / 'PATH_MAPPING.json').write_text(json.dumps({
        'frozen_source_edited': False, 'source_sha256': expected,
        'explicit_constructor_path_mappings': {k: str(v) for k, v in mappings.items()},
        'no_producer_modules_imported': True, 'new_model_inference': False,
        'external_human_review_claimed': False
    }, indent=2) + '\n')


if __name__ == '__main__':
    main()
