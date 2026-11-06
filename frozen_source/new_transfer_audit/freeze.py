#!/usr/bin/env python3
"""New freeze only: never recreate or backdate the lost old transfer freeze."""
import argparse,time
from common import *

FOLDERS=('source_reference','sources','inputs','prepared','provenance','docs','verification','runtime_override')
EXCLUDED={'INDEPENDENT_REVIEW.json','GPU_RELEASE.json'}

def paths():
    result=list(ROOT.glob('*.py'))+[ROOT/'POLICY_LOCK.json',ROOT/'model_locations.json']
    for folder in FOLDERS:
        result += [p for p in (ROOT/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name not in EXCLUDED]
    return sorted(set(result))

def check():
    payload=json.loads((ROOT/'PREINFERENCE_FROZEN_SHA256.json').read_text())
    for rel,want in payload['files'].items():
        if sha(ROOT/rel)!=want:raise RuntimeError('New replication frozen file differs: '+rel)
    if set(payload['files'])!={str(p.relative_to(ROOT)) for p in paths()}:
        raise RuntimeError('Scientific file inventory differs from freeze')
    return {'status':'PASS','files':len(payload['files']),'freeze_sha256':sha(ROOT/'PREINFERENCE_FROZEN_SHA256.json')}

def create():
    assert not any((ROOT/f'run_{m}').exists() for m in MODELS)
    budget=json.loads((ROOT/'prepared/BUDGET.json').read_text())
    tests=json.loads((ROOT/'verification/SYNTHETIC_TESTS.json').read_text());assert tests['status']=='PASS'
    assert all(sha(ROOT/name)==want for name,want in tests['source_sha256'].items())
    calib=json.loads((ROOT/'verification/CALIBRATION_RECHECK.json').read_text());assert calib['status']=='PASS' and calib['policy_count']==36
    cpu=json.loads((ROOT/'verification/CPU_PREPARED_CHECK.json').read_text());assert cpu['status']=='PASS'
    assert cpu['producer_sha256']==sha(ROOT/'check_prepared.py')
    returns=json.loads((ROOT/'verification/RUNNER_RETURN_PRESERVATION.json').read_text());assert returns['status']=='PASS' and returns['test_count']==6
    assert returns['used_actual_runner_source_sha256']==sha(ROOT/'run_gpu.py')
    assert returns['used_actual_return_journal_source_sha256']==sha(ROOT/'return_journal.py')
    assert returns['used_actual_analyzer_source_sha256']==sha(ROOT/'analyze_transfer.py')
    payload={'created_unix':time.time(),'created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        'new_real_replication':True,'old_transfer_raw_lost':True,'known_prior_aggregate_results':True,
        'policy_lock_sha256':sha(ROOT/'POLICY_LOCK.json'),'budget':budget,
        'GPU_authorized':False,'requires_independent_review_and_fresh_root_GPU_RELEASE':True,
        'files':{str(p.relative_to(ROOT)):sha(p) for p in paths()}}
    write_json(ROOT/'PREINFERENCE_FROZEN_SHA256.json',payload);return check()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--check',action='store_true');a=p.parse_args();print(compact(check() if a.check else create()))
