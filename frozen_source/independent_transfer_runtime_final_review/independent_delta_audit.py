from pathlib import Path
import ast,hashlib,json,time,importlib.metadata as md,platform,sys
R=Path('/home/gpu/IntentSat2S_revision_20261106/new_transfer_audit')
O=Path('/home/gpu/IntentSat2S_revision_20261106/independent_transfer_runtime_final_review')
B=Path('/home/gpu/IntentSat2S_revision_20261106/independent_transfer_preflight_review')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def load(p):return json.loads(p.read_text())
def calls(p):
 t=ast.parse(p.read_text());names={'LLM','SamplingParams','GuidedDecodingParams'}
 return {n:[ast.dump(x,include_attributes=False) for x in ast.walk(t) if isinstance(x,ast.Call) and isinstance(x.func,ast.Name) and x.func.id==n] for n in sorted(names)}
f=load(R/'PREINFERENCE_FROZEN_SHA256.json');old=load(R/'docs/runtime_baseline95/PREINFERENCE_FROZEN_SHA256.json')
expected='14088307c668516820aaf9b6215989d9e362d6edfecdbe1be319ac51222dcfaa'
assert sha(R/'PREINFERENCE_FROZEN_SHA256.json')==expected
bad=[n for n,h in f['files'].items() if sha(R/n)!=h];assert not bad
# Build inventory independently without importing the producer's freeze helpers.
paths={p.name for p in R.glob('*.py')}|{'POLICY_LOCK.json','model_locations.json'}
for folder in ('source_reference','sources','inputs','prepared','provenance','docs','verification','runtime_override'):
 paths|={str(p.relative_to(R)) for p in (R/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name not in {'INDEPENDENT_REVIEW.json','GPU_RELEASE.json'}}
assert paths==set(f['files'])
changed=[n for n,h in old['files'].items() if sha(R/n)!=h]
assert set(changed)=={'run_gpu.py','launch_gpu.py','verification/SYNTHETIC_TESTS.json'}
assert len(old['files'])==95 and len(f['files'])==126
assert calls(R/'run_gpu.py')==calls(R/'docs/runtime_baseline95/run_gpu.py')
assert sha(R/'docs/runtime_baseline95/run_gpu.py')==old['files']['run_gpu.py']
assert sha(R/'docs/runtime_baseline95/launch_gpu.py')==old['files']['launch_gpu.py']
assert sha(R/'docs/runtime_baseline95/verification/SYNTHETIC_TESTS.json')==old['files']['verification/SYNTHETIC_TESTS.json']
assert sha(R/'docs/runtime_baseline95/PREINFERENCE_FROZEN_SHA256.json')==f['baseline95_freeze_sha256']==sha(R/'docs/runtime_baseline95/PREINFERENCE_FROZEN_original_file.json')
c=load(B/'BASELINE95_CPU_REVIEW_CERTIFICATE.json');em=B/'BASELINE95_EVIDENCE_SHA256.json';ev=load(em)
assert sha(em)==c['evidence_manifest_sha256'];assert len(ev['files'])==260
bade=[n for n,h in ev['files'].items() if sha(B/n)!=h]
assert set(bade)=={'CPU_SYNTHETIC_FULL_PIPELINE/qwen/PREINFERENCE_FROZEN_SHA256.json','CPU_SYNTHETIC_FULL_PIPELINE/smol/PREINFERENCE_FROZEN_SHA256.json'}
for n in bade:
 assert (B/n).is_symlink() and (B/n).resolve()==R/'PREINFERENCE_FROZEN_SHA256.json'
 assert ev['files'][n]==sha(R/'docs/runtime_baseline95/PREINFERENCE_FROZEN_SHA256.json')
assert c['freeze_sha256']==f['baseline95_freeze_sha256'] and c['verified_policy_count']==36
assert sha(B/'BASELINE95_CPU_REVIEW_CERTIFICATE.json')==sha(R/'docs/runtime_baseline95/BASELINE95_CPU_REVIEW_CERTIFICATE.json')
contract=load(R/'prepared/runtime_observation_contract.json');assert contract['runtime_observer_sha256']==sha(R/'runtime_observer.py')
t=ast.parse((R/'runtime_observer.py').read_text());w=next(n for n in t.body if isinstance(n,ast.FunctionDef) and n.name=='worker_identity')
assert hashlib.sha256(ast.get_source_segment((R/'runtime_observer.py').read_text(),w).encode()).hexdigest()==contract['worker_identity_source_sha256']
installed=Path('/usr/local/lib/python3.12/dist-packages/vllm')
assert all(sha(installed/n)==h for n,h in contract['vllm_source_sha256'].items())
config=load(R/'prepared/runtime_configuration.json');assert config['python']==platform.python_version() and config['python_executable']==sys.executable
assert all(md.version(p)==v for p,v in config['packages'].items())
r={'status':'PASS_STATIC_AND_BASELINE_INHERITANCE','freeze_sha256':expected,'frozen_files_independently_rehashed':len(f['files']),'inventory_exact':True,'baseline95_sha256':f['baseline95_freeze_sha256'],'baseline_changed_files':changed,'baseline_unchanged_files':len(old['files'])-len(changed),'added_frozen_files':sorted(set(f['files'])-set(old['files'])),'constructor_AST_exact_equality':calls(R/'run_gpu.py'),'baseline_certificate_sha256':sha(B/'BASELINE95_CPU_REVIEW_CERTIFICATE.json'),'baseline_evidence_manifest_sha256':sha(em),'baseline_evidence_files_rehashed':len(ev['files'])-len(bade),'baseline_evidence_dynamic_symlinks_rebuilt_against_preserved_baseline':bade,'baseline_independent_math_inherited_equalities':c['full_pipeline_independent_equalities'],'baseline_independent_request_checks_inherited':28800,'vllm_installed_sources_rehashed':len(contract['vllm_source_sha256']),'runtime_contract_sha256':sha(R/'prepared/runtime_observation_contract.json'),'worker_callable_source_hash_rebuilt':True,'actual_environment_package_versions_match':True,'reviewer_GPU_execution_or_model_loading':False,'created_unix':time.time()}
(O/'01_INDEPENDENT_FREEZE_AND_INHERITANCE.json').write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps({k:v for k,v in r.items() if k not in ['constructor_AST_exact_equality','added_frozen_files']},indent=2))
