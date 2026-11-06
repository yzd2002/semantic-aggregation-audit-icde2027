#!/usr/bin/env python3
"""Replay the separately disclosed post-hoc native-vector HG readout on CPU.

The writer's script and finite_audit.py are executed unchanged. Only the U/A
filesystem roots are explicitly mapped. No missing source text is invented.
"""
import argparse
import os
from pathlib import Path
import shutil
import sys
from portable_common import import_file, load, sha, write_json
from replay_compact import check, guard_original_tree


REQUIRED = (
    'manuscript/analyze_native_audit_sensitivity.py',
    'manuscript/analysis/ANALYSIS_COMPLETE.json',
    'manuscript/analysis/main/integrity.json',
    'manuscript/analysis/finite/FIXED_BUDGET_AUDITS.json',
    'manuscript/analysis/native_sensitivity/NATIVE_VECTOR_AUDITS.json',
    'manuscript/analysis/native_sensitivity/NATIVE_VECTOR_AUDITS.csv',
)


def replay(bundle, supplement, out):
    bundle, supplement, out = (Path(p).resolve() for p in (bundle, supplement, out))
    scope = load(bundle/'COMPACT_SCOPE.json')
    check(scope['payload_kind'] == 'ACTUAL_COMPACT_EVIDENCE' and scope['actual_configurations'] == 36,
          'Post-hoc replay requires the complete actual 36-configuration compact bundle')
    out.mkdir(parents=True, exist_ok=False)
    snapshot = out/'source_snapshot'
    copied = {}
    for rel in REQUIRED:
        source, target = supplement/rel, snapshot/rel
        check(source.is_file() and not source.is_symlink(), 'Missing explicit post-hoc source/evidence file: '+rel)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        copied[rel] = sha(target)
        check(copied[rel] == sha(source), 'Post-hoc file changed while snapshotting: '+rel)
    script = snapshot/REQUIRED[0]
    expected = load(snapshot/REQUIRED[4])
    check(expected['status'] == 'POST_HOC_NATIVE_VECTOR_SENSITIVITY_COMPLETE'
          and expected['planned_configurations'] == 36, 'Native-vector author result is not complete')
    check(expected['script_sha256'] == sha(script), 'Author native-vector source hash differs')
    check(expected['source_api_sha256'] == sha(bundle/'novelty_theory/finite_audit.py'),
          'Author HG API differs from the compact original API')
    check(expected['analysis_receipt_sha256'] == sha(snapshot/REQUIRED[1]),
          'Author analysis receipt binding differs')
    reader = out/'reader_view'
    for rel in REQUIRED[1:4]:
        source = snapshot/rel
        target = reader/Path(rel).relative_to('manuscript/analysis')
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    # Every input has now been snapshotted. Actual math may open only these
    # supplied receipts and the compact policy-result/source-gold files.
    guard = guard_original_tree(scope['original_root'], [bundle, out, Path(__file__).resolve().parent])
    sys.path.insert(0, str(bundle/'novelty_theory'))
    author = import_file('unchanged_posthoc_native_vector_author', script)
    author.U, author.M, author.A = bundle, snapshot/'manuscript', reader
    author.main()
    result = reader/'native_sensitivity/NATIVE_VECTOR_AUDITS.json'
    csv = reader/'native_sensitivity/NATIVE_VECTOR_AUDITS.csv'
    check(load(result) == expected, 'Native-vector HG result differs from the original full-input author analysis')
    check(sha(csv) == copied[REQUIRED[5]], 'Native-vector CSV differs from the original author result')
    write_json(out/'POSTHOC_CPU_REPLAY_REPORT.json', {
        'status': 'PASS', 'analysis_role': 'explicit post-hoc native-vector sensitivity',
        'configurations': 36, 'result_rows': len(expected['rows']),
        'matching_canonical_pair_checks': expected['matching_canonical_pair_checks'],
        'author_source_and_HG_math_changed': False,
        'explicit_path_mapping': 'Writer U is the expanded compact bundle; writer A is a fresh receipt/output view.',
        'posthoc_source_snapshot_sha256': copied,
        'recomputed_JSON_sha256': sha(result), 'recomputed_CSV_sha256': sha(csv),
        'original_tree_access_guard': guard,
        'new_GPU_or_model_inference': False, 'source_text_or_token_reconstruction': False,
        'future_reference_or_joint_budget_coverage_claimed': False,
    })
    print('PASS: unchanged post-hoc native-vector HG source recomputed from compact actual policy/gold vectors.')


if __name__ == '__main__':
    os.environ.update(CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle', type=Path, required=True)
    p.add_argument('--supplement', type=Path, required=True,
                   help='Published supplement root with the explicit manuscript/ paths listed in REQUIRED')
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    replay(a.bundle, a.supplement, a.out)
