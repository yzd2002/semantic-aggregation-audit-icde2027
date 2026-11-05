#!/usr/bin/env python3
"""Relocate and reproduce the frozen consensus-envelope diagnostic (CPU only).

The original analysis and independent verifier are imported without editing.
Only their path globals, and the verifier's historical manifest-path lookup,
are rebound. Every original input/code hash remains mandatory.
"""
import argparse
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import shlex
import shutil
import sys

sys.dont_write_bytecode = True
BUNDLE = Path(__file__).resolve().parent
BASELINE_MANIFEST_SHA256 = '4cedcd027f785b3e5bbad1fdbd56d83743277a9d0f6fa72ab2c748a12718fc0b'
AUDIT_SHA256 = 'c6974e2b86dab9682e17a4b58f5e1887c9fc32d124b16e08d69315384fecdef3'
BASELINE_RESULT_SHA256 = {
    'envelope_per_pool.jsonl': '874a1c8c46a5b9fc3aac9a4e1e910bf88018e34993de8ff55c82667a82c57070',
    'envelope_summary.json': '1f6e4053c3e7e41911d1a8ac93583fbd7bf94ea022226a00245b5584c643a429',
}
CODE_FILES = ('PROTOCOL_consensus_envelope.md', 'analyze_envelope.py', 'verify_envelope.py')
SELECTIONS = ('wands_selective_execution_selection', 'esci_selective_holdout_selection')
RUNS = ('run_wands_selective_execution_qwen_v1', 'run_wands_selective_execution_smollm_v1',
        'run_esci_selective_holdout_qwen', 'run_esci_selective_holdout_smollm')
TRACES = ('original_responses.jsonl', 'stable_reverse_responses.jsonl',
          'selective_pointwise_responses.jsonl', 'selective_outputs.jsonl')
RELATIVE_INPUTS = tuple(f'{p}/inputs.jsonl' for p in SELECTIONS) + tuple(
    f'{p}/{f}' for p in RUNS for f in TRACES)


def fail(message):
    raise RuntimeError(message)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require_hash(path, expected):
    actual = sha256(path)
    if actual != expected:
        fail(f'SHA-256 mismatch: {path}; expected {expected}, observed {actual}')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def normalize_manifest(manifest):
    """Identify a closed set of logical inputs; never read old absolute paths."""
    logical = {}
    for old_path, digest in manifest['files'].items():
        matches = []
        if digest == AUDIT_SHA256:
            matches.append('audit')
        matches += [f'code/{f}' for f in CODE_FILES if old_path.endswith('/' + f)]
        matches += [f'artifact/{f}' for f in RELATIVE_INPUTS if old_path.endswith('/' + f)]
        if len(matches) != 1 or matches[0] in logical:
            fail(f'Unknown, ambiguous, or duplicate manifest source: {old_path}')
        logical[matches[0]] = (old_path, digest)
    wanted = {'audit'} | {f'code/{f}' for f in CODE_FILES} | {
        f'artifact/{f}' for f in RELATIVE_INPUTS}
    if set(logical) != wanted:
        fail(f'Incomplete source manifest; missing {sorted(wanted - set(logical))}')
    return logical


def bind_paths(logical, artifact, audit, code_dir):
    mapping = {}
    for name, (old_path, expected) in logical.items():
        if name == 'audit':
            local = audit
        elif name.startswith('code/'):
            local = code_dir / name.removeprefix('code/')
        else:
            local = artifact / name.removeprefix('artifact/')
        require_hash(local, expected)
        mapping[old_path] = local
    if len(set(mapping.values())) != len(mapping):
        fail('Different frozen sources map to the same local file.')
    return mapping


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def captured_call(function):
    buffer = io.StringIO()
    try:
        with contextlib.redirect_stdout(buffer):
            function()
    finally:
        print(buffer.getvalue(), end='')
    return buffer.getvalue()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact', type=Path, required=True,
                        help='Directory containing the six original selection/run subdirectories.')
    parser.add_argument('--consensus-audit', type=Path, required=True,
                        help='Relocated, unchanged methods/consensus_audit.json.')
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--output', type=Path, help='Fresh output directory, which must not exist.')
    mode.add_argument('--result-dir', type=Path, help='Existing saved result directory; read-only verification.')
    parser.add_argument('--report', type=Path,
                        help='Optional new JSON report file for --result-dir (never overwritten).')
    args = parser.parse_args()
    if sys.version_info < (3, 9):
        fail('Python 3.9 or later is required; standard library only.')
    if sys.flags.optimize:
        fail('Do not use -O/-OO or PYTHONOPTIMIZE: frozen checks require assertions.')
    if args.output and args.report:
        fail('--report is only valid with --result-dir; fresh runs write their own report.')
    artifact = args.artifact.expanduser().resolve(strict=True)
    audit = args.consensus_audit.expanduser().resolve(strict=True)
    target_arg = args.output if args.output is not None else args.result_dir
    target = target_arg.expanduser().resolve()
    if args.output is not None and target.exists():
        fail(f'Fresh output already exists: {target}')
    report_path = args.report.expanduser().resolve() if args.report else None
    if report_path is not None and report_path.exists():
        fail(f'Report already exists: {report_path}')

    # These are the original published diagnostic's hashes, independent of location.
    require_hash(BUNDLE / 'PRECOMPUTATION_SHA256.json', BASELINE_MANIFEST_SHA256)
    baseline = normalize_manifest(read_json(BUNDLE / 'PRECOMPUTATION_SHA256.json'))
    bind_paths(baseline, artifact, audit, BUNDLE)
    for name, expected in BASELINE_RESULT_SHA256.items():
        require_hash(BUNDLE / name, expected)

    analysis_log = None
    if args.output is not None:
        target.mkdir()  # Parent must already exist; never reuse a result directory.
        for name in CODE_FILES:
            shutil.copyfile(BUNDLE / name, target / name)
        analyzer = load_module('_frozen_envelope_analyzer', target / 'analyze_envelope.py')
        analyzer.BASE, analyzer.AUDIT, analyzer.HERE = artifact, audit, target
        analysis_log = captured_call(analyzer.main)
        (target / 'ANALYSIS.log').write_text(analysis_log, encoding='utf8')

    frozen = normalize_manifest(read_json(target / 'PRECOMPUTATION_SHA256.json'))
    if {k: v[1] for k, v in frozen.items()} != {k: v[1] for k, v in baseline.items()}:
        fail('Saved result manifest differs from the original frozen source/code hashes.')
    mapping = bind_paths(frozen, artifact, audit, target)
    verifier = load_module('_frozen_envelope_verifier', target / 'verify_envelope.py')
    verifier.DATA, verifier.LEGACY, verifier.ROOT = artifact, audit, target

    def relocated_path(value):
        # Only string entries in the saved manifest need rebasing. All normal
        # local Path objects (result loading, raw trace loading) stay untouched.
        if isinstance(value, str) and value in mapping:
            return mapping[value]
        return Path(value)

    verifier.Path = relocated_path
    verification_log = captured_call(verifier.verify)

    require_hash(target / 'envelope_per_pool.jsonl', BASELINE_RESULT_SHA256['envelope_per_pool.jsonl'])
    original_summary = read_json(BUNDLE / 'envelope_summary.json')
    reproduced_summary = read_json(target / 'envelope_summary.json')
    original_summary.pop('source_sha256')
    reproduced_summary.pop('source_sha256')
    if reproduced_summary != original_summary:
        fail('Reproduced summary differs from original numeric/interpretive fields.')
    count = sum(1 for line in (target / 'envelope_per_pool.jsonl').read_text(encoding='utf8').splitlines() if line.strip())
    if count != 604:
        fail(f'Expected 604 rows, observed {count}.')
    print('PASS portable reproduction: all 604 rows byte-identical; all summary fields identical except source locations.')

    report = {
        'status': 'PASS', 'mode': 'recompute-and-verify' if args.output is not None else 'verify-saved',
        'python': sys.version, 'command': shlex.join([sys.executable, '-B', str(Path(__file__).resolve()), *sys.argv[1:]]),
        'launcher_sha256': sha256(Path(__file__)),
        'original_manifest_sha256': BASELINE_MANIFEST_SHA256,
        'artifact': str(artifact), 'consensus_audit': str(audit), 'result_dir': str(target),
        'pool_model_instances': count, 'distinct_dataset_query_pools': 302,
        'all_rows_byte_identical': True, 'summary_identical_except_source_locations': True,
        'source_path_mapping': {k: str(v) for k, v in sorted(mapping.items())},
        'output_sha256': {name: sha256(target / name) for name in
                          ('PRECOMPUTATION_SHA256.json', 'envelope_per_pool.jsonl', 'envelope_summary.json')},
        'verification_log': verification_log,
    }
    if args.output is not None:
        (target / 'VERIFY.log').write_text(verification_log, encoding='utf8')
        report_path = target / 'PORTABLE_RUN.json'
    if report_path is not None:
        with report_path.open('x', encoding='utf8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2, allow_nan=False)
            f.write('\n')
        print(f'Report: {report_path}')
    else:
        print(json.dumps(report, ensure_ascii=False, sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    try:
        main()
    except (AssertionError, OSError, RuntimeError, ValueError, KeyError) as exc:
        print(f'FAIL portable envelope: {type(exc).__name__}: {exc}', file=sys.stderr)
        sys.exit(1)
