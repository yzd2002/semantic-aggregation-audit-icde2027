#!/usr/bin/env python3
"""Explicit path adapter for the immutable SNLI independent audit source.

No frozen file is edited. Original six-case certificate and baseline remain
byte-identical. Its recorded absolute source path is mapped to --context.
"""
import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit', type=Path, required=True)
    p.add_argument('--context', type=Path, required=True)
    p.add_argument('--snli', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    audit, context, snli, out = [x.resolve() for x in (a.audit, a.context, a.snli, a.out)]
    out.mkdir(parents=True, exist_ok=False)
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(audit))
    spec = importlib.util.spec_from_file_location('frozen_snli_independent', audit / 'snli_independent_audit.py')
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    recorded = json.loads((audit / 'RESULTS_CERTIFICATE.json').read_text())['read_only_source_bundle']
    original_path = m.Path

    def mapped_path(value):
        if str(value) == recorded:
            return context
        return original_path(value)

    m.Path = mapped_path
    m.BASE, m.PARENT, m.DEST = snli, audit, out
    # The archived interactive audit's ledger also registered three top-level
    # analysis logs and six report/cost files before main(). A direct standalone
    # call to the frozen main reads 221 rather than those original 230 files.
    # Verify every original ledger entry now and register the actual current
    # bytes; no original hash record or mathematical source is edited.
    archived_hashes = json.loads((audit / 'snli_extension/SNLI_SOURCE_RAW_ANALYSIS_SHA256.json').read_text())
    for rel, expected in archived_hashes.items():
        file = snli / rel
        if not file.is_file() or m.file_hash(file) != expected:
            raise ValueError('Archived SNLI full source ledger differs: ' + rel)
        m.remember(file)
    (out / 'PATH_MAPPING.json').write_text(json.dumps({
        'frozen_source_edited': False,
        'recorded_context_path': recorded,
        'mapped_context_path': str(context),
        'snli_source': str(snli),
        'baseline_evidence': str(audit),
        'full_original_source_ledger_entries_preverified': len(archived_hashes),
        'full_ledger_adapter_reason': 'The original interactive audit registered nine reporting/log files outside standalone main; all original 230 entries are explicitly reverified before recomputation.',
        'mode': 'CPU-only mathematical recomputation, no external human review or fresh GPU inference'
    }, indent=2) + '\n')
    m.main()
    certificate = out / 'SNLI_AND_NINE_CASES_CERTIFICATE.json'
    if not certificate.is_file() or not json.loads(certificate.read_text())['status'].startswith('PASS:'):
        raise ValueError('No actual complete nine-case certificate was produced')


if __name__ == '__main__':
    main()
