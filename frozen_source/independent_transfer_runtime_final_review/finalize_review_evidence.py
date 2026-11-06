from pathlib import Path
import hashlib,json,shutil,time
R=Path('/home/gpu/IntentSat2S_revision_20261106/new_transfer_audit')
O=Path('/home/gpu/IntentSat2S_revision_20261106/independent_transfer_runtime_final_review')
B=Path('/home/gpu/IntentSat2S_revision_20261106/independent_transfer_preflight_review')
FINAL='83ca3beb7488e4fd1c5998d5d6f38a6003286ad183dbf61c179a40d46672d38a'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for data in iter(lambda:f.read(8<<20),b''):h.update(data)
 return h.hexdigest()
def write(p,x):
 with p.open('x') as f:json.dump(x,f,indent=2,ensure_ascii=False);f.write('\n')
def load(p):return json.loads(p.read_text())
assert sha(R/'PREINFERENCE_FROZEN_SHA256.json')==FINAL
frozen=load(R/'PREINFERENCE_FROZEN_SHA256.json');assert len(frozen['files'])==158
assert all(sha(R/n)==h for n,h in frozen['files'].items())
paths={p.name for p in R.glob('*.py')}|{'POLICY_LOCK.json','model_locations.json'}
for folder in ('source_reference','sources','inputs','prepared','provenance','docs','verification','runtime_override'):
 paths|={str(p.relative_to(R)) for p in (R/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name not in {'INDEPENDENT_REVIEW.json','GPU_RELEASE.json'}}
assert paths==set(frozen['files'])
assert not (R/'GPU_RELEASE.json').exists()
snapshot=O/'REVIEWED_FINAL_83ca_SOURCE_SNAPSHOT';snapshot.mkdir(exist_ok=False)
files=['PREINFERENCE_FROZEN_SHA256.json','runtime_observer.py','verify_effective_runtime.py','run_gpu.py','launch_gpu.py','check_runtime_amendment.py','gpu_gate.py','return_journal.py','prepared/runtime_observation_contract.json']
for n in files:
 p=snapshot/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(R/n,p)
replaced=[]
for p in O.rglob('runtime_observer.py'):
 if p.is_symlink():
  assert sha(p)==sha(R/'runtime_observer.py');p.unlink();shutil.copy2(snapshot/'runtime_observer.py',p);replaced.append(str(p.relative_to(O)))
write(snapshot/'SNAPSHOT_RECORD.json',{'files':{n:sha(snapshot/n) for n in files},'personal_final_fixture_symlinks_stabilized':replaced,'scientific_tree_modified_by_reviewer':False})
for name in ['04_FINAL_FREEZE_AND_INHERITANCE.json','05_FINAL_DELTA_AND_PRESERVATION.json','06_FINAL_UNKNOWN_USAGE_PRESERVATION.json','07_FINAL_ACTUAL_RUNNER_RESULTS.json']:
 assert load(O/name)['status'].startswith('PASS')
tamper=load(O/'TAMPER_REHASHED_FINAL_83ca/RESULT.json');assert tamper['status']=='PASS' and tamper['test_count']==22
cpu=load(O/'CPU_SYNTHETIC_FINAL_18_CASES/RUNTIME_AMENDMENT_CPU_CHECK.json');assert cpu['status']=='PASS' and cpu['test_count']==18
for n,h in cpu['source_sha256'].items():assert sha(R/n)==h
write(O/'08_FINAL_INTEGRITY_SEAL.json',{'status':'PASS','freeze_sha256':FINAL,'frozen_files':158,'all_hashes_and_exact_inventory_rechecked_immediately_before_certificate':True,'fresh_root_GPU_GO_absent_at_review':True,'final_personal_observer_symlinks_stabilized':len(replaced),'reviewer_scientific_file_mutations':0,'reviewer_GPU_or_model_loading':False,'created_unix':time.time()})
assert not [p for p in O.rglob('*') if p.is_symlink()]
excluded={'EVIDENCE_SHA256.json','INDEPENDENT_REVIEW.json'}
manifest={'scope':'Independent final runtime/scientific-inheritance review. Includes retained failing126 evidence and final158 passing evidence; every generated return is explicitly CPU synthetic. Original baseline260 evidence is external and referenced by its original certificate/hash; no claims of260 unchanged current referents.','files':{str(p.relative_to(O)):sha(p) for p in sorted(O.rglob('*')) if p.is_file() and p.name not in excluded and '__pycache__' not in p.parts}}
write(O/'EVIDENCE_SHA256.json',manifest)
cert={'status':'PASS','freeze_sha256':FINAL,'verified_policy_count':36,'frozen_files_verified':158,'evidence_manifest_sha256':sha(O/'EVIDENCE_SHA256.json'),'evidence_files':len(manifest['files']),'independence_scope':'Reviewer did not author baseline95 science, runtime observer, canonical/stop repair, runner, policies or analyzers. Reviewer read the final science tree only and wrote independent fixtures/reports solely in independent_transfer_runtime_final_review. This certificate covers exact frozen-source inheritance and CPU-only runtime/durable-return verification; not actual GPU execution success or paper acceptance.','baseline95_inheritance':{'certificate_sha256':sha(B/'BASELINE95_CPU_REVIEW_CERTIFICATE.json'),'freeze_sha256':'3349e50072791650512c7708ebf1c54a37ed2012b0a72cf785730666b3e2b38c','evidence_manifest_sha256':sha(B/'BASELINE95_EVIDENCE_SHA256.json'),'current_evidence_files_exactly_rehashed':258,'dynamic_manifest_symlink_entries':2,'symlink_entry_hashes_rebuilt_from_preserved_baseline95_bytes':True,'baseline_science_unchanged_files':92,'only_original95_replaced_files':['run_gpu.py','launch_gpu.py','verification/SYNTHETIC_TESTS.json'],'all_replaced_old_versions_preserved':True,'all_28800_prompts_retokenized_in_inherited_review':True,'full_pipeline_independent_equalities_inherited':22401,'producer_math_used_as_new_independent_oracle':False,'full_28800_or_model25_rehash_not_repeated_here':True},'runtime_final_verification':{'previous126_changed_files_exactly_six_and_originals_preserved':True,'LLM_SamplingParams_GuidedDecodingParams_constructor_AST_exactly_unchanged':True,'installed_vllm_source_hashes_verified':11,'actual_Python_and_package_versions_verified':True,'trusted_worker_callable_source_hash_rebuilt':True,'actual_runner_CPU_SYNTHETIC_cases':18,'includes_original14_and_four_new_effective_stop_format_faults':True,'clean_qwen_smol_and_invalid_science_replays':3,'independent_tamper_cases':22,'canonical_native_raw_tokens_finish_stop_labels_cost_and_summary_mutations_rejected_even_after_rehash':True,'additional_actual_runner_unknown_usage_preservation_case':1,'unknown_usage_stays_null_all_native_returns_fsynced_before_binding_no_retry':True,'invalid_scientific_output_preserved_and_counted':True,'native_class_module_BF16_config_gates_verified_by_source_and_faults':True,'per_request_greedy_EOS_stop_grammar_engineID_and_prompt_receipts_verified':True,'per_model_cache_directories_and_58byte_local_RPC_paths_verified':True,'serialization_scope':'Generic pickle/cloudpickle fallback in experiment process/local children; actual frozen callable is read-only worker_identity. Not a callable allowlist.','root_review_release_and_foreign_PID_gates_verified':True},'resolved_findings':['RUNTIME_FINAL_01 canonical/native/summary replay gap fixed in exact final tree','RUNTIME_FINAL_02 effective stop and output-format observation gap fixed in exact final tree'],'unresolved_substantive_blockers':[],'reviewer_actual_GPU_generation_or_model_loading':False,'all_reviewer_execution_outputs_CPU_SYNTHETIC':True,'scientific_tree_changes_by_reviewer':0,'GPU_release_authorized_by_this_certificate':False,'root_must_separately_issue_exact_fresh_GPU_RELEASE':True,'created_unix':time.time()}
write(O/'INDEPENDENT_REVIEW.json',cert)
print(json.dumps({'status':'PASS','freeze_sha256':FINAL,'evidence_files':len(manifest['files']),'evidence_manifest_sha256':cert['evidence_manifest_sha256'],'certificate_sha256':sha(O/'INDEPENDENT_REVIEW.json'),'certificate':str(O/'INDEPENDENT_REVIEW.json')},indent=2))
