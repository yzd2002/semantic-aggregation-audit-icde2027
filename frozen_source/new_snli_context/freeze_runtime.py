#!/usr/bin/env python3
"""Append-only CPU candidate/final freeze. Never creates GPU authorization."""
import argparse
import time
from common import *
from prepare_runtime_cpu import check_baseline

CANDIDATE='RUNTIME_REVIEW_CANDIDATE_SHA256.json'
FINAL='PREINFERENCE_FROZEN_SHA256.json'


def current_files():
    base=json.loads((ROOT/'CPU_REVIEW_CANDIDATE_SHA256.json').read_text())
    files={ROOT/name for name in base['files']}
    for p in ROOT.iterdir():
        if p.is_file() and p.name not in (CANDIDATE,FINAL) and not p.name.startswith('GPU_GO_') and not p.name.endswith('.sha256'):
            files.add(p)
    files.add(ROOT/'CPU_REVIEW_CANDIDATE_SHA256.json.sha256')
    for directory in ('prepared','selection','sources','tokenizers','gold_perturbation_sources','source_reference','global_primary'):
        files.update(p for p in (ROOT/directory).rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    files.update((ROOT/'runtime_override').glob('*.py'))
    files.update(p for p in (ROOT/'verification').iterdir() if p.is_file() and not p.name.startswith('INDEPENDENT_RUNTIME_REVIEW'))
    for path in ('cpu_checks/runtime_failure_retention_final/RESULTS.json','cpu_checks/runtime_full_chain_final/RESULTS.json',
            'cpu_checks/runtime_source_recheck_final/RESULTS.json','cpu_checks/runtime_source_recheck_final/SOURCE_CONTRACT_RECHECK.json'):
        files.add(ROOT/path)
    return {str(p.relative_to(ROOT)):sha(p) for p in sorted(files)}


def no_inference():
    for model in MODELS:
        if (ROOT/f'run_{model}').exists() or (ROOT/f'GPU_GO_{model}.json').exists():
            raise ValueError('A real run/GO exists before this CPU freeze')


def validate_candidate(data):
    for name,want in data['files'].items():
        if sha(ROOT/name)!=want:raise ValueError('Review candidate file changed: '+name)
    check_baseline()
    assert sha(ROOT/'generation_receipts.py')=='a58f21a1596759a7903fb23468aa8a72a3f827ad51295f513d04cfe14be6b31d'
    for name,want in json.loads((ROOT/'global_primary/AMENDMENT_SHA256.json').read_text()).items():
        assert sha(ROOT/'global_primary'/name)==want


def main():
    parser=argparse.ArgumentParser();mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--candidate',action='store_true');mode.add_argument('--final',action='store_true')
    parser.add_argument('--independent-review',type=Path);args=parser.parse_args()
    no_inference();check_baseline()
    if args.candidate:
        if (ROOT/CANDIDATE).exists():raise ValueError('Candidate already exists; preserve it and use an explicit new amendment/version')
        for path in ('verification/CPU_RUNTIME_PREPARATION.json','verification/NATIVE_IPC_CPU_CHECK.json',
                'verification/AUTHOR_RUNTIME_CPU_CHECKS.json','verification/SERIALIZATION_REUSE_PROVENANCE.json',
                'cpu_checks/runtime_source_recheck_final/RESULTS.json','cpu_checks/runtime_failure_retention_final/RESULTS.json',
                'cpu_checks/runtime_full_chain_final/RESULTS.json'):
            result=json.loads((ROOT/path).read_text());assert result['status']=='PASS',path
        for path in ('cpu_checks/runtime_failure_retention_final/RESULTS.json','cpu_checks/runtime_full_chain_final/RESULTS.json'):
            assert json.loads((ROOT/path).read_text())['real_model_judgments']==0
        result={'created_unix':time.time(),'status':'CPU_RUNTIME_REVIEW_CANDIDATE_NO_GPU_GO',
            'version':'SNLI native runtime, durable v3 receipts and prospective nine-case primary',
            'original_candidate_sha256':sha(ROOT/'CPU_REVIEW_CANDIDATE_SHA256.json'),
            'unique_scientific_calls':6720,'unique_diagnostic_calls':60,'new_model_judgments':0,
            'legacy_ci98_75_flags_authoritative':False,'files':current_files(),
            'excluded_mutable_paths':['runtime_cache/','rpc/','run_*/','GPU_GO_*.json','runtime_override/events_*.jsonl',
                'cpu_cache/','__pycache__/','cpu_checks/runtime_*/* (only final RESULTS.json is in this inventory)'],
            'independent_review_pending':True,'GPU_GO_created':False}
        validate_candidate(result);write_json(ROOT/CANDIDATE,result)
        (ROOT/(CANDIDATE+'.sha256')).write_text(sha(ROOT/CANDIDATE)+'  '+CANDIDATE+'\n')
        print(compact({'status':result['status'],'manifest_sha256':sha(ROOT/CANDIDATE),'files':len(result['files'])}),flush=True)
        return
    if args.independent_review is None:raise ValueError('Final freeze requires the concrete independent review report')
    if (ROOT/FINAL).exists():raise ValueError('Final freeze already exists; do not replace it')
    candidate=json.loads((ROOT/CANDIDATE).read_text());validate_candidate(candidate)
    review_path=(ROOT/args.independent_review).resolve()
    if review_path.parent!=ROOT/'verification':raise ValueError('Independent report must be in this SNLI verification directory')
    review=json.loads(review_path.read_text())
    assert review['status']=='PASS' and review['candidate_manifest_sha256']==sha(ROOT/CANDIDATE)
    assert review['runner_source_sha256']==sha(ROOT/'run_gpu.py')
    assert review['generation_receipts_source_sha256']==sha(ROOT/'generation_receipts.py')
    assert review['analysis_source_sha256']==sha(ROOT/'analyze_snli_runtime.py')
    assert review['real_model_judgments']==0 and review['GPU_execution'] is False and not review['issues']
    for key in ('baseline_105_preserved','receipt_retention_and_identity','authoritative_primary_and_COUNT',
            'no_gpu_and_no_dependency_changes','native_runtime_and_short_ipc'):
        assert review['checks'][key] is True,key
    files={**candidate['files'],CANDIDATE:sha(ROOT/CANDIDATE),CANDIDATE+'.sha256':sha(ROOT/(CANDIDATE+'.sha256')),
        str(review_path.relative_to(ROOT)):sha(review_path)}
    result={'created_unix':time.time(),'status':'CPU_FROZEN_INDEPENDENT_REVIEW_PASS_AWAITING_ROOT_GPU_GO',
        'version':candidate['version'],'not_a_historical_certificate':True,'files':files,
        'original_cpu_candidate_sha256':sha(ROOT/'CPU_REVIEW_CANDIDATE_SHA256.json'),
        'runtime_review_candidate_sha256':sha(ROOT/CANDIDATE),'independent_review_sha256':sha(review_path),
        'unique_scientific_calls':6720,'unique_diagnostic_calls':60,'new_model_judgments':0,
        'primary_nominal_confidence':179/180,'primary_quantiles':[1/360,359/360],'primary_family_size':9,
        'planned_SNLI_components_only':3,'legacy_ci98_75_flags_authoritative':False,
        'excluded_mutable_paths':candidate['excluded_mutable_paths'],'GPU_GO_created':False}
    write_json(ROOT/FINAL,result)
    (ROOT/(FINAL+'.sha256')).write_text(sha(ROOT/FINAL)+'  '+FINAL+'\n')
    print(compact({'status':result['status'],'manifest_sha256':sha(ROOT/FINAL),'files':len(files),'GPU_GO_created':False}),flush=True)


if __name__=='__main__':main()
