from pathlib import Path
import sys,json,time,hashlib,shutil
r=Path(__file__).resolve().parents[2];sys.path.insert(0,str(r))
from common import sha,write_json
from freeze import paths,check
old_path=r/'PREINFERENCE_FROZEN_SHA256.json';assert sha(old_path)=='3349e50072791650512c7708ebf1c54a37ed2012b0a72cf785730666b3e2b38c'
old=json.loads(old_path.read_text());allowed={'run_gpu.py','launch_gpu.py','verification/SYNTHETIC_TESTS.json'}
for rel,want in old['files'].items():
 if rel not in allowed:assert sha(r/rel)==want,rel
assert not any((r/f'run_{m}').exists() for m in ['qwen','smol'])
newtests=json.loads((r/'verification/SYNTHETIC_TESTS.json').read_text());assert newtests['status']=='PASS' and newtests['test_count']==37
for name,want in newtests['source_sha256'].items():assert sha(r/name)==want,name
runtime=r/'cpu_runtime_amendment_root/RUNTIME_AMENDMENT_CPU_CHECK.json';rr=json.loads(runtime.read_text());assert rr['status']=='PASS' and rr['test_count']==14
for name,want in rr['source_sha256'].items():assert sha(r/name)==want,name
assert rr['runtime_observation_contract_sha256']==sha(r/'prepared/runtime_observation_contract.json')
shutil.copy2(runtime,r/'verification/RUNTIME_AMENDMENT_ROOT_CPU_CHECK.json')
record={'created_unix':time.time(),'created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'new_real_replication':True,'old_transfer_raw_lost':True,'known_prior_aggregate_results':True,
 'baseline95_freeze_sha256':sha(old_path),'runtime_amendment_candidate_sha256':sha(r/'docs/runtime_amendment_20261106/CANDIDATE_MANIFEST.json'),
 'unchanged_scientific_inputs_policy_sampling_constructors_and_analyzers':True,'runtime_amendment_scope':'Native/effective observation, identity binding, independent caches and short local IPC; all original constructors and target/policy inputs unchanged. Observer I/O can change engine arrival timing, so bitwise equality to uninstrumented executions is not claimed.',
 'policy_lock_sha256':sha(r/'POLICY_LOCK.json'),'budget':json.loads((r/'prepared/BUDGET.json').read_text()),'GPU_authorized':False,'requires_independent_review_and_fresh_root_GPU_RELEASE':True,
 'current_runtime_author_test_count':14,'current_numerical_test_count':37,'old_runner6case_report_is_preserved_baseline_only':True,'files':{str(p.relative_to(r)):sha(p) for p in paths()}}
old_path.rename(r/'docs/runtime_baseline95/PREINFERENCE_FROZEN_original_file.json');record['files']={str(p.relative_to(r)):sha(p) for p in paths()};write_json(old_path,record);print(json.dumps(check()))
