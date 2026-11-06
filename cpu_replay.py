#!/usr/bin/env python3
"""Replay preserved actual outcomes on CPU into a fresh output tree."""
import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from artifact_cli import sha, verify_files


MODULES = ('context', 'snli', 'transfer', 'decision_dependency', 'independent_audit', 'independent_transfer_audit')
MODELS = ('qwen', 'smol', 'glm')


def run(command, log, env):
    print('CPU replay:', log.name, flush=True)
    with log.open('x') as stream:
        subprocess.run(command, check=True, stdout=stream, stderr=subprocess.STDOUT, env=env)


def compare_tree(original, replay, *, ignore=()):
    checked = {}
    for source in sorted(original.rglob('*')):
        if not source.is_file() or source.name in ignore:
            continue
        rel = source.relative_to(original)
        target = replay / rel
        if not target.is_file() or sha(source) != sha(target):
            raise ValueError('CPU replay result differs: ' + str(rel))
        checked[str(rel)] = sha(target)
    if not checked:
        raise ValueError('Empty result comparison')
    return checked


def context_replay(root, out, env, script_name, study):
    result = {}
    for model in MODELS:
        target = out / model
        run([sys.executable, '-B', str(study / script_name), '--bundle', str(study),
             '--model', model, '--out', str(target)], out / (model + '.log'), env)
        result[model] = compare_tree(study / ('analysis_' + model), target)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle-root', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--modules', nargs='+', choices=MODULES, default=list(MODULES))
    a = p.parse_args()
    root, out = a.bundle_root.resolve(), a.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'PYTHONDONTWRITEBYTECODE': '1',
           'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'USE_TORCH': '0',
           'TOKENIZERS_PARALLELISM': 'false', 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'}
    sys.dont_write_bytecode = True
    report = {'status': 'RUNNING', 'started_unix': time.time(), 'new_inference': False,
              'gpu_used': False, 'python': sys.version, 'modules': {}, 'source_root': str(root),
              'limits': 'Relocated CPU recomputation of archived actual outcomes; no clean-machine GPU rerun or external human assessment is claimed.'}
    for name in ('numpy', 'pyarrow'):
        report[name] = importlib.metadata.version(name)
    try:
        for name in a.modules:
            count = verify_files(root, name)
            directory = out / name
            directory.mkdir()
            if name == 'context':
                comparisons = context_replay(root, directory, env, 'analyze_fresh_replication.py', root / 'new_glm_context')
            elif name == 'snli':
                comparisons = context_replay(root, directory, env, 'analyze_snli_runtime.py', root / 'new_snli_context')
            elif name == 'transfer':
                study = root / 'new_transfer_audit'
                comparisons = {}
                for model in ('qwen', 'smol'):
                    summary = json.loads((study / f'run_{model}/summary.json').read_text())
                    if summary.get('status') != 'complete' or (study / f'run_{model}/FAILURE.json').exists():
                        raise ValueError('Transfer has no actual successful completion: ' + model)
                    run([sys.executable, '-B', str(study / 'verify_effective_runtime.py'), '--root', str(study),
                         '--model', model, '--out', str(directory / (model + '_effective_runtime.json'))],
                        directory / (model + '_effective_runtime.log'), env)
                    run([sys.executable, '-B', str(study / 'analyze_transfer.py'), '--model', model,
                         '--out', str(directory / model)], directory / (model + '.log'), env)
                    comparisons[model] = compare_tree(study / ('analysis_' + model), directory / model)
            elif name == 'decision_dependency':
                run([sys.executable, '-B', str(Path(__file__).parent / 'decision_prepare_cpu.py'),
                     '--bundle', str(root / 'reconstruction/decision_audit'), '--out', str(directory / 'rebuild')],
                    directory / 'dev_lock_and_inputs.log', env)
                comparisons = json.loads((directory / 'rebuild/DECISION_DEPENDENCY_REPLAY.json').read_text())
            elif name == 'independent_transfer_audit':
                audit = root / 'final_independent_review_runtime_referee'
                comparisons = {}
                for model, version in (('qwen', 'qwen_v1'), ('smol', 'smol_v2')):
                    run([sys.executable, '-B', str(Path(__file__).parent / 'independent_transfer_replay.py'),
                         '--audit', str(audit), '--transfer', str(root / 'new_transfer_audit'), '--model', model,
                         '--out', str(directory / model)], directory / (model + '.log'), env)
                    comparisons[model] = compare_tree(audit / 'REAL_TRANSFER' / version, directory / model,
                                                       ignore=('RESULTS_CERTIFICATE.json',))
                    before = json.loads((audit / 'REAL_TRANSFER' / version / 'RESULTS_CERTIFICATE.json').read_text())
                    after = json.loads((directory / model / 'RESULTS_CERTIFICATE.json').read_text())
                    for ephemeral in ('created_unix', 'elapsed_CPU_seconds'):
                        before.pop(ephemeral, None)
                        after.pop(ephemeral, None)
                    if before != after:
                        raise ValueError('Independent transfer certificate scientific fields differ: ' + model)
                    comparisons[model]['certificate_scientific_fields_equal'] = True
            else:
                audit = root / 'independent_fresh_results_audit'
                run([sys.executable, '-B', str(audit / 'independent_audit.py'), '--source', str(root / 'new_glm_context'),
                     '--out', str(directory / 'six')], directory / 'six.log', env)
                stable_six = [f'{prefix}{model}.json' for model in MODELS for prefix in ('RECOMPUTED_', 'RAW_CHAIN_AND_COSTS_')]
                stable_six += ['INDEPENDENT_RAW_BINDINGS.jsonl', 'INDEPENDENT_PER_TARGET.jsonl', 'NINE_FAMILY_PRIMARY_SIX_OBSERVED.json', 'SIX_STARTUP_FAILURES.json', 'SELF_CHECKS.json', 'SOURCE_RAW_ANALYSIS_SHA256.json']
                comparisons = {'six': {}}
                for name6 in stable_six:
                    if sha(audit / name6) != sha(directory / 'six' / name6):
                        raise ValueError('Independent six-case replay differs: ' + name6)
                    comparisons['six'][name6] = sha(directory / 'six' / name6)
                run([sys.executable, '-B', str(Path(__file__).parent / 'independent_snli_replay.py'),
                     '--audit', str(audit), '--context', str(root / 'new_glm_context'), '--snli', str(root / 'new_snli_context'),
                     '--out', str(directory / 'nine')], directory / 'nine.log', env)
                stable_nine = [f'{prefix}{model}.json' for model in MODELS for prefix in ('SNLI_RECOMPUTED_', 'SNLI_RAW_CHAIN_AND_COSTS_')]
                stable_nine += ['NINE_CASE_PRIMARY_INDEPENDENT.json', 'SNLI_INDEPENDENT_PER_TARGET.jsonl', 'SNLI_INDEPENDENT_RAW_BINDINGS.jsonl', 'SNLI_SOURCE_AND_FREEZE.json', 'SNLI_SOURCE_RAW_ANALYSIS_SHA256.json']
                comparisons['nine'] = {}
                for namen in stable_nine:
                    if sha(audit / 'snli_extension' / namen) != sha(directory / 'nine' / namen):
                        raise ValueError('Independent nine-case replay differs: ' + namen)
                    comparisons['nine'][namen] = sha(directory / 'nine' / namen)
            report['modules'][name] = {'status': 'PASS', 'archived_files_verified': count, 'result_comparisons': comparisons}
        report['status'] = 'PASS'
    except BaseException as error:
        report.update(status='FAIL', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        report['finished_unix'] = time.time()
        (out / 'CPU_REPLAY_REPORT.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    print('PASS: all requested archived outcome replays matched.', flush=True)


if __name__ == '__main__':
    main()
