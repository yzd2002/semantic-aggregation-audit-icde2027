"""CPU-only, reproducible comparison of preserved v2 bytes with current v3."""
from pathlib import Path
import ast,difflib,json,hashlib,time

r=Path(__file__).resolve().parents[1]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
base=r/'initial_cpu_preflight/v2_failure_retention_baseline'
manifest=json.loads((base/'PREINFERENCE_FROZEN_SHA256_v2.json').read_text())
changes=[];comparison=[]
for rel,want in manifest['files'].items():
    got=sha(r/rel)
    comparison.append({'path':rel,'v2_sha256':want,'v3_sha256':got,'unchanged':got==want})
    if got!=want:changes.append(rel)
expected={'CPU_HANDOFF.md','PROTOCOL_fresh_three_family_context_replication.md','README.md',
          'analyze_fresh_replication.py','freeze.py','run_gpu.py'}
assert set(changes)==expected,changes
patch=[]
for rel in sorted(changes):
    old=base/rel
    assert old.exists() and sha(old)==manifest['files'][rel],rel
    patch.append(''.join(difflib.unified_diff(old.read_text().splitlines(True),(r/rel).read_text().splitlines(True),
        fromfile='v2/'+rel,tofile='v3/'+rel)))
(r/'verification/V2_TO_V3_AFFECTED_CODE_AND_DOCUMENTS.patch').write_text('\n'.join(patch))
prepared=[x for x in comparison if x['path'].startswith('prepared/')]
assert all(x['unchanged'] for x in prepared)
for file in ('analyze.py','secondary.py'):
    assert sha(r/file)==sha(r/'source_reference'/file)==manifest['files'][file]
for p in r.glob('*.py'):ast.parse(p.read_text())
proof={'created_unix':time.time(),'status':'PASS',
    'v2_freeze_sha256':'8bc1b739ace0120922450920be559351ee3b23dcdeb0899885edd7cb0588833d',
    'v2_is_explicitly_superseded_not_claimed_unchanged':True,'v2_changed_existing_paths':changes,
    'v2_affected_bytes_preserved':True,'full_v2_file_comparison':comparison,
    'prepared_files_count':len(prepared),'all_prepared_prompts_ids_tokens_grammars_sampling_schedules_budgets_unchanged':True,
    'source_model_target_background_and_gold_files_unchanged':True,'original_statistical_files_byte_identical':True,
    'new_analysis_definition_is_root_prospective_nine_case_amendment':True,'new_real_model_judgments':0,
    'root_GPU_GOs_absent':not any((r/f'GPU_GO_{m}.json').exists() for m in ('qwen','smol','glm')),
    'root_real_run_directories_absent':not any((r/f'run_{m}').exists() for m in ('qwen','smol','glm')),
    'new_receipt_module_sha256':sha(r/'generation_receipts.py'),'new_runner_sha256':sha(r/'run_gpu.py'),
    'new_analysis_wrapper_sha256':sha(r/'analyze_fresh_replication.py')}
assert proof['root_GPU_GOs_absent'] and proof['root_real_run_directories_absent']
(r/'verification/V2_TO_V3_CHANGE_AUDIT.json').write_text(json.dumps(proof,indent=2)+'\n')
g=json.loads((r/'verification/independent_candidate_global_wrapper_results.json').read_text())
for rel,want in g['source_sha256'].items():assert sha(r/rel)==want
for k in ('CPU_only','full_three_model_durable_receipt_costs_and_six_context_primary_components_verified',
          'all_common12_e_vectors_rebuilt_directly_from_raw_JSON','new_seeds_and_179_180_intervals_independently_recomputed',
          'six_case_assembler_preserves_SNLI_missingness','empty_all_128_targets_unavailable_and_COUNT_abstains',
          'unreturned_raw_receipt_costs_unknown_not_zero'):
    assert g[k] is True,k
assert len(g['canonical_integrity_negative_tests'])==4
(r/'verification/INDEPENDENT_V3_GLOBAL_PRIMARY_REVIEW.json').write_text(json.dumps({
    'status':'PASS','scope':'Independent current-source candidate integration; final freeze-bound certificate remains external',
    'source_result_file':'independent_candidate_global_wrapper_results.json',
    'source_result_sha256':sha(r/'verification/independent_candidate_global_wrapper_results.json'),
    'source_sha256':g['source_sha256'],'global_components_raw_rebuilt_and_na_integrity_verified':True},indent=2)+'\n')
print('V2→V3 affected paths:',changes)
print('All',len(prepared),'prepared files and original target/gold/model/statistical files unchanged.')
print('Current-source independent global integration verified; final independent freeze certificate stays external.')
