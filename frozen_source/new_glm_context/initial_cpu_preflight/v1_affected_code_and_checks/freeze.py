#!/usr/bin/env python3
"""Create a new current CPU freeze, never a replacement historical certificate."""
import time
from common import *


def main():
    target=ROOT/'PREINFERENCE_FROZEN_SHA256.json'
    if target.exists():raise SystemExit('Freeze already exists. Changes require an explicit append-only reviewed amendment.')
    for model in MODELS:
        assert not (ROOT/f'run_{model}').exists(),'Scientific run directory already exists before the current freeze'
        assert not (ROOT/f'GPU_GO_{model}.json').exists(),'Authorization must follow final CPU freeze review'
    envelope=json.loads((ROOT/'prepared/REQUEST_ENVELOPE.json').read_text())
    assert envelope['all_unique_scientific_calls']==14250 and envelope['all_unique_diagnostic_calls']==60
    assert json.loads((ROOT/'provenance/MODEL_SNAPSHOT_VERIFICATION.json').read_text())['status']=='PASS'
    for path in ('cpu_checks/CPU_CHECKS.json','cpu_checks/fresh_full_chain/CHECKS.json',
                 'verification/COMPLETE_SOURCE_FIELD_REJOIN.json','verification/REMATCH_GOLD_INVERSION_CHECK.json'):
        assert json.loads((ROOT/path).read_text())['status']=='PASS'
    files=set()
    for p in ROOT.iterdir():
        if p.is_file() and (p.suffix in ('.py','.md') or p.name=='LICENSE'):files.add(p)
    for name in ('prepared','selection','sources','source_reference','historical_context_arm','initial_cpu_preflight','provenance'):
        files.update(p for p in (ROOT/name).rglob('*') if p.is_file() and not p.is_symlink())
    files.update(p for p in (ROOT/'runtime_override').glob('*.py'))
    files.update(p for p in (ROOT/'verification').iterdir() if p.is_file())
    files.update(ROOT/path for path in ('cpu_checks/CPU_CHECKS.json','cpu_checks/fresh_full_chain/CHECKS.json'))
    manifest={'created_unix':time.time(),'status':'New CPU freeze before any fresh three-family context judgments; awaiting root static review and explicit per-model GPU GO.',
        'not_a_recreated_historical_certificate':True,'scientific_models':list(MODELS),'unique_scientific_calls':14250,
        'unique_diagnostic_calls':60,'files':{str(p.relative_to(ROOT)):sha(p) for p in sorted(files)},
        'excluded_runtime_mutable_paths':['runtime_cache/','tmp/','run_*/','GPU_GO_*.json','cpu_checks/fresh_full_chain/SYNTHETIC_NOT_MODEL_RUN/','cpu_checks/fresh_full_chain/*_results/'],
        'no_new_model_inference':True}
    write_json(target,manifest)
    print('New freeze',sha(target),'files',len(manifest['files']),flush=True)


if __name__=='__main__':main()
