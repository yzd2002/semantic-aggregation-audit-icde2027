"""Finalize the six completed runs after existing independent deep analyses.

This is a bounded read-only supplement for actual runtime/reset/provenance
checks and the requested canonical readout. It never runs inference or edits
scientific artifacts. It refuses to finalize partial or discrepant analyses.
"""
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime, timezone
from importlib import metadata
import argparse
import hashlib
import json
import re
import sys
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import analyze_study as a


def load(path):
    return json.loads(Path(path).read_text())


def norm_counts(row):
    n, gold, prediction = row.get('n'), row.get('gold_positive_count'), row.get('predicted_count')
    error = prediction-gold if prediction is not None and gold is not None else None
    absolute = abs(error) if error is not None else None
    result = dict(row)
    result.update(source_count=gold, source_count_over_N=gold/n if gold is not None and n else None,
        predicted_count_over_N=prediction/n if prediction is not None and n else None,
        count_error_signed_over_N=error/n if error is not None and n else None,
        count_error_absolute_over_N=absolute/n if absolute is not None and n else None,
        count_error_signed_over_G=(error/gold if gold else 0.0 if error == 0 else None) if error is not None else None,
        count_error_absolute_over_G=(absolute/gold if gold else 0.0 if absolute == 0 else None) if absolute is not None else None,
        zero_gold_rule='error/G=0 for exact zero; null for nonzero prediction when G=0',
        relative_count_error_status='unavailable_prediction_or_gold' if error is None else
            'zero_gold_exact' if gold == 0 and prediction == 0 else
            'undefined_zero_gold_nonzero_prediction' if gold == 0 else 'defined')
    return result


def runtime_checks(root, study, budget, audit):
    provenance = load(root/'prepared/MODEL_PROVENANCE.json')
    v3 = load(root/'GLM_NATIVE_TOKENIZER_GUARD_AMENDMENT_V3.json')
    token_hash = provenance['glm']['files']['tokenization_chatglm.py']
    audit.check(token_hash == v3['allowed_tokenizer_sha256'], 'final_pinned_glm_tokenizer_provenance')
    tokenizer_path = Path(study['models']['glm'])/'tokenization_chatglm.py'
    audit.fingerprint(tokenizer_path, token_hash)
    installed = Path(metadata.distribution('vllm').locate_file('vllm'))
    audit.check(metadata.version('vllm') == '0.10.2', 'final_installed_vllm_version')
    freezes = {name: load(root/name) for name in ('PREINFERENCE_FREEZE.json', 'BUDGET_FOLLOWUP_FREEZE.json')}
    for name, frozen in freezes.items():
        audit.fingerprint(root/name)
        for rel, wanted in frozen['files'].items():
            audit.fingerprint(root/rel, wanted)
        for rel, wanted in frozen['installed_vllm_files'].items():
            audit.fingerprint(installed/rel, wanted)
    base, extended = freezes['PREINFERENCE_FREEZE.json'], freezes['BUDGET_FOLLOWUP_FREEZE.json']
    audit.check(len(base['files']) == 117 and len(extended['files']) == 125, 'final_117_and_125_frozen_file_counts')
    audit.check(all(extended['files'].get(k) == v for k, v in base['files'].items()), 'final_all_original_117_unchanged_in_budget_freeze')
    audit.check(extended['base_freeze_sha256'] == a.sha_file(root/'PREINFERENCE_FREEZE.json'), 'final_budget_base_freeze_identity')
    for filename, field in [('native_runtime_glm_v3.py', 'runtime_sha256'),
                            ('run_study_glm_v3.py', 'entrypoint_sha256'), ('launch_glm_v3.py', 'launcher_sha256')]:
        audit.fingerprint(root/filename, v3[field])
    raw_observations = Counter()
    observer_files = sorted((root/'runtime_cache').rglob('actual_cache_resets.jsonl'))
    for path in observer_files:
        audit.fingerprint(path)
        for row in a.jsonl(path, audit, completed=True):
            raw_observations[a.sha_value(row, sorted_keys=True)] += 1
    runtime_rows, reset_rows, request_ledger = [], [], []
    expected_models = {'qwen': ('Qwen3ForCausalLM', 'vllm.model_executor.models.qwen3', 4022468096),
                       'glm': ('ChatGLMForCausalLM', 'vllm.model_executor.models.chatglm', 9399951360)}
    specs = [(model, cache, root/f'run_{model}_cache{cache}', False)
             for model in ('qwen', 'glm') for cache in (0, 1)] + [
             (model, 1, root/f'budget_run_{model}', True) for model in ('qwen', 'glm')]
    for model, cache, directory, followup in specs:
        native = load(directory/'EFFECTIVE_NATIVE_RUNTIME.json')
        workers = native['native_workers']
        audit.check(len(workers) == 1, 'final_actual_worker_count', run=directory.name)
        worker = workers[0]
        audit.check((worker['model_class'], worker['module'], worker['parameters']) == expected_models[model]
                    and worker['dtypes'] == ['torch.bfloat16'] and native['dtype'] == 'torch.bfloat16',
                    'final_actual_native_bf16_parameter_class', run=directory.name)
        audit.check(native['model_path'] == study['models'][model] and native['prefix_cache'] == bool(cache)
                    and native['max_model_len'] == 8192 and native['max_num_seqs'] == 32
                    and native['tensor_parallel_size'] == 1 and native['enforce_eager'] is True,
                    'final_actual_engine_profile', run=directory.name)
        if model == 'glm':
            audit.check(worker.get('remote_model_modules') == [v3['allowed_tokenizer_module']]
                        and worker.get('dynamic_module_source_sha256') == {v3['allowed_tokenizer_module']: token_hash},
                        'final_actual_loaded_glm_exact_tokenizer_only_no_model_or_config_module', run=directory.name)
        else:
            audit.check(worker.get('remote_model_modules') == [], 'final_actual_qwen_no_dynamic_modules', run=directory.name)
        if followup:
            expected = {(job['job_id'], 'NATIVE_CP') for job in budget['jobs_per_model']}
            stages = {job['job_id']+'/'+phase: (job['job_id'], 'NATIVE_CP')
                      for job in budget['jobs_per_model'] for phase in ('calibration', 'execution')}
        else:
            cases = [c for c in study['cases'] if c['prefix_cache'] == bool(cache)]
            expected = {('official_q3', 'native')} | {(c['case_id'], p) for c in cases for p in a.POLICIES}
            stages = {'OFFICIAL_SEMBENCH_Q3': ('official_q3', 'native')}
            for case in cases:
                for policy in a.POLICIES:
                    stages[case['case_id']+'/'+policy] = (case['case_id'], policy)
                stages[case['case_id']+'/CURRENT_CP/execution'] = (case['case_id'], 'CURRENT_CP')
                cp = load(directory/case['case_id']/'CURRENT_CP_RESULT.json')
                if cp['selected_k'] > 1:
                    expected.add((case['case_id'], 'INDEPENDENT_VALIDATION'))
                    stages[case['case_id']+'/INDEPENDENT_VALIDATION'] = (case['case_id'], 'INDEPENDENT_VALIDATION')
        effective_first, effective_identity, effective_guided, stage_counts = {}, {}, {}, Counter()
        for row in a.jsonl(directory/'effective_requests.jsonl', audit, completed=True):
            audit.check(row['stage'] in stages or row['stage'] == 'controls',
                        'final_every_effective_request_has_declared_bill_or_control_stage', run=directory.name,
                        stage=row['stage'])
            effective_identity[row['logical_id']] = row['stage'], row['engine_request_id']
            effective_guided[row['logical_id']] = row['controls']['guided']
            stage_counts[row['stage']] += 1
            key = stages.get(row['stage'])
            if key is not None:
                effective_first[key] = min(effective_first.get(key, float('inf')), row['observed_unix'])
        def purpose(logical):
            stage = effective_identity[logical][0]
            guided = effective_guided[logical]
            if stage == 'controls':
                return 'technical_controls'
            if followup:
                if stage.endswith('/calibration'):
                    return 'budget_calibration_batch_probes' if guided else 'budget_calibration_fresh_native_references'
                if stage.endswith('/execution'):
                    return 'budget_selected_plan_execution'
                return 'UNDECLARED_STAGE'
            key = stages.get(stage)
            if key == ('official_q3','native'):
                return 'official_SemBench_Q3_native_integration'
            if key is None:
                return 'UNDECLARED_STAGE'
            policy = key[1]
            if policy in ('S1','S2'):
                return 'main_full_native_'+policy+'_reference'
            if policy in ('RANDOM8','SIMILAR8'):
                return 'main_fixed_plan_'+policy+'_execution'
            if policy == 'CURRENT_CP':
                if stage.endswith('/execution'):
                    return 'main_Current80_S_selected_plan_execution'
                return 'main_Current80_S_calibration_batch_probes' if guided else 'main_Current80_S_calibration_fresh_native_references'
            if policy == 'INDEPENDENT_VALIDATION':
                return 'research_validation_independent_batch_probes' if guided else 'research_validation_fresh_native_references'
            return 'UNDECLARED_STAGE'
        for batch in a.jsonl(directory/'submissions.jsonl', audit, completed=True):
            for job in batch['jobs']:
                identity = effective_identity.get(job['logical_id'])
                audit.check(identity is not None and identity[0] == batch['stage'],
                            'final_submission_effective_bill_stage_identity', run=directory.name,
                            logical_id=job['logical_id'])
        parent_batches = Counter()
        for batch in a.jsonl(directory/'raw_returns.jsonl', audit, completed=True):
            categories = {purpose(logical) for logical in batch['expected_ids']}
            audit.check(len(categories) == 1, 'final_one_declared_purpose_per_parent_generate_batch', run=directory.name)
            if len(categories) == 1:
                parent_batches[next(iter(categories))] += 1
            for logical, returned in zip(batch['expected_ids'],batch['returns']):
                audit.check(effective_identity.get(logical) == (batch['stage'], returned['engine_request_id']),
                            'final_raw_effective_engine_and_bill_stage_identity', run=directory.name,
                            logical_id=logical)
        resets = list(a.jsonl(directory/'cache_resets.jsonl', audit, completed=True))
        observed = Counter((r['case_id'], r['policy']) for r in resets)
        audit.check(set(observed) == expected and all(n == 1 for n in observed.values()),
                    'final_every_policy_exactly_one_declared_cache_boundary', run=directory.name,
                    expected=len(expected), observed=len(resets))
        for row in resets:
            key = row['case_id'], row['policy']
            observations = row.get('native_reset_observations', [])
            audit.check(row['prefix_cache'] == bool(cache), 'final_policy_cache_mode', run=directory.name, key=key)
            audit.check(key in effective_first and row['time_unix'] <= effective_first[key],
                        'final_actual_cache_boundary_before_effective_inference', run=directory.name, key=key)
            if cache:
                audit.check(len(observations) == 1, 'final_one_native_blockpool_observation', run=directory.name, key=key)
                for observation in observations:
                    audit.check(observation['native_result'] is True and observation['cached_hash_count_after'] == 0
                        and observation['num_gpu_blocks']-observation['num_free_blocks_after'] == 1
                        and observation['pid'] == worker['pid'] and observation['native_class'] == 'vllm.v1.core.block_pool.BlockPool'
                        and observation['result_not_modified'] is True and observation['observed_unix'] <= row['time_unix'],
                        'final_actual_empty_native_pool_and_matching_worker_pid', run=directory.name, key=key)
                    audit.check(raw_observations[a.sha_value(observation, sorted_keys=True)] == 1,
                        'final_policy_receipt_matches_actual_native_observer_log', run=directory.name, key=key)
            else:
                audit.check(observations == [], 'final_cache_off_no_fabricated_reset_success', run=directory.name, key=key)
            reset_rows.append({'run': directory.name, 'case_id': key[0], 'policy': key[1], 'cache_enabled': bool(cache),
                'native_worker_pid': worker['pid'], 'native_empty_pool_verified': bool(cache),
                'cache_off_effective_profile_verified': not bool(cache), 'native_observations': observations})
        off_known, off_unknown = 0, 0
        rows_by_purpose = defaultdict(list)
        for row in a.jsonl(directory/'responses.jsonl', audit, completed=True):
            audit.check(effective_identity.get(row['logical_id']) == (row['stage'],row['engine_request_id']),
                        'final_response_effective_engine_and_bill_stage_identity', run=directory.name,
                        logical_id=row['logical_id'])
            rows_by_purpose[purpose(row['logical_id'])].append(row)
            if not cache:
                cached = row.get('cached_input_tokens')
                if cached is None:
                    off_unknown += 1
                else:
                    off_known += 1
                    audit.check(cached == 0, 'final_actual_cache_off_no_reported_prefix_hits', run=directory.name)
        for category, rows in sorted(rows_by_purpose.items()):
            request_ledger.append({'run':directory.name, 'run_family':'budget_followup' if followup else 'original_main',
                'model':model, 'prefix_cache':bool(cache), 'purpose':category,
                'native_request_unit':'one native vLLM prompt/RequestOutput; not a complete query or parent generate() call',
                'parent_generate_batches':parent_batches[category], **a.usage(rows)})
        runtime_rows.append({'run': directory.name, 'model': model, 'prefix_cache': bool(cache),
            'native_worker': worker, 'policy_boundaries_verified': len(resets),
            'complete_effective_request_stage_ledger':dict(stage_counts),
            'cache_off_known_zero_requests': off_known, 'cache_off_unknown_cache_requests': off_unknown})
    return {'runtime_runs': runtime_rows, 'policy_cache_boundaries': reset_rows,
            'native_request_usage_by_run_and_purpose':request_ledger,
            'raw_native_observer_files': [str(p) for p in observer_files], 'glm_tokenizer_sha256': token_hash,
            'base_frozen_files': len(base['files']), 'budget_frozen_files': len(extended['files'])}


def failure_checks(root, audit):
    v2 = load(root/'RUNTIME_TECHNICAL_AMENDMENT_V2.json')
    v3 = load(root/'GLM_NATIVE_TOKENIZER_GUARD_AMENDMENT_V3.json')
    expected = dict(v2['failures_preserved'])
    expected[Path(v3['previous_failure']).name] = v3['previous_failure_files']
    audit.check(set(expected) == {'qwen_cache1_attempt01', 'glm_cache1_attempt01', 'glm_cache1_attempt02'},
                'final_three_failure_archives_declared')
    rows = []
    for name, files in expected.items():
        directory = root/'failed_attempts'/name
        for filename, wanted in files.items():
            audit.fingerprint(directory/filename, wanted)
        failure, start = load(directory/'FAILURE.json'), load(directory/'START.json')
        responses = list(a.jsonl(directory/'responses.jsonl', audit, completed=True))
        raw_calls = sum(len(b['returns']) for b in a.jsonl(directory/'raw_returns.jsonl', audit, completed=True))
        effective = sum(1 for _ in a.jsonl(directory/'effective_requests.jsonl', audit, completed=True))
        if name.startswith('glm'):
            audit.check(not responses and raw_calls == 0 and effective == 0, 'final_failed_glm_no_generate_records', archive=name)
        else:
            audit.check(len(responses) == raw_calls == effective == 10 and all(r['stage'] == 'controls' for r in responses),
                        'final_failed_qwen_only_ten_controls', archive=name)
        observed = a.usage(responses) if responses else None
        begin = start.get('started_unix', start.get('time_unix'))
        rows.append({'archive': name, 'failure_type': failure['type'], 'error': failure['error'],
            'native_returned_requests': raw_calls, 'scientific_case_requests': sum(r['stage'] != 'controls' for r in responses),
            'observed_usage': observed, 'elapsed_wall_seconds': failure['time_unix']-begin if begin is not None else None,
            'zero_generate_is_not_zero_initialization_cost': True, 'preserved_file_hashes_verified': len(files)})
    return rows


def public_receipt_checks(root, audit):
    """Tie the preserved anonymous, pre-call receipts to their current bytes.

    This checks the original downloaded-byte receipts; it does not represent a
    new network fetch or change their original verification timestamps.
    """
    specs = [
        ('PUBLIC_RUNTIME_AMENDMENT_V2_VERIFIED.json',
         ['run_qwen_cache1', 'run_glm_cache1'],
         'c8eb58c1b2a7eefb99c850491c6c4e9989f43dea'),
        ('PUBLIC_GLM_TOKENIZER_V3_VERIFIED.json', ['run_glm_cache1'],
         'a6303afb741e0640e5d1cdd386bc363a338ce233'),
        ('PUBLIC_BUDGET_FOLLOWUP_VERIFIED.json', ['budget_run_qwen', 'budget_run_glm'],
         'e02d11ee251660e75bd2be0a6f33272a8ea7ce30')]
    rows = []
    for name, runs, commit in specs:
        receipt = load(root/name)
        audit.fingerprint(root/name)
        audit.check(receipt['status'] == 'PASS' and receipt['commit'] == commit,
                    'final_preserved_public_receipt_commit', receipt=name)
        earliest = min(row['observed_unix'] for directory in runs
                       for row in a.jsonl(root/directory/'effective_requests.jsonl', audit, completed=True))
        for item in receipt['files']:
            audit.fingerprint(root/item['file'], item['sha256'])
            audit.check(item['actual_anonymous_download'] is True and item['authentication'] is False
                        and item['verified_unix'] < earliest and '?ref='+commit in item['url'],
                        'final_preserved_anonymous_publication_before_relevant_actual_calls',
                        receipt=name, file=item['file'])
        rows.append({'receipt':name, 'commit':commit, 'verified_files':len(receipt['files']),
            'earliest_relevant_effective_request_unix':earliest,
            'latest_preserved_verification_unix':max(r['verified_unix'] for r in receipt['files']),
            'preserved_pre_call_anonymous_download_receipt_verified':True,
            'new_network_download_performed_by_final_audit':False})
    return rows


def teardown_receipts(root, audit):
    """Preserve completed-output teardown diagnostics separately from failures."""
    schedulers, rows = {}, []
    for model in ('qwen','glm'):
        path = root/f'scheduling_{model}.log'
        audit.fingerprint(path)
        events = list(a.jsonl(path,audit,completed=True))
        audit.check(sum(e.get('kind') == 'all_reviewed_model_stages_completed' for e in events) == 1,
                    'final_scheduler_completed_budget_and_OFF', model=model)
        schedulers[model] = events
    specs = [
        ('qwen','run_qwen_cache1','execution_qwen_cache1_attempt02.log',None),
        ('glm','run_glm_cache1','execution_glm_cache1_attempt03.log',None),
        ('qwen','run_qwen_cache0','execution_qwen_cache0.log','cache_OFF'),
        ('glm','run_glm_cache0','execution_glm_cache0.log','cache_OFF'),
        ('qwen','budget_run_qwen','execution_budget_qwen.log','budget'),
        ('glm','budget_run_glm','execution_budget_glm.log','budget')]
    tz = ZoneInfo('Asia/Shanghai')
    for model,run,filename,stage in specs:
        path = root/filename
        audit.fingerprint(path)
        marker_name = 'BUDGET_ALL_WORK_COMPLETE.json' if stage == 'budget' else 'ALL_WORK_COMPLETE.json'
        marker = load(root/run/marker_name)
        completed = marker['time_unix']
        lines = path.read_text(errors='replace').splitlines()
        diagnostics, anchors = [], []
        for line_number,line in enumerate(lines,1):
            kind = ('NCCL_process_group_teardown' if 'destroy_process_group() was not called' in line else
                'nanobind_leaked_binding_objects' if 'nanobind: leaked' in line else
                'nanobind_reference_counting_diagnostic' if 'reference counting issue in the binding code' in line else
                'engine_core_client_teardown_diagnostic' if 'Engine core proc' in line and 'died unexpectedly' in line else None)
            if kind is None:
                continue
            timestamp = None
            match = re.search(r'\[W(\d{2})(\d{2}) (\d{2}):(\d{2}):(\d{2})\.(\d+) ',line)
            if match:
                month,day,hour,minute,second,fraction = match.groups()
                timestamp = datetime(datetime.fromtimestamp(completed,tz).year,int(month),int(day),int(hour),
                    int(minute),int(second),int((fraction+'000000')[:6]),tzinfo=tz).timestamp()
                anchors.append((line_number,timestamp))
            coarse_lower = None
            coarse = re.search(r'ERROR (\d{2})-(\d{2}) (\d{2}):(\d{2}):(\d{2}) ',line)
            if coarse:
                month,day,hour,minute,second = map(int,coarse.groups())
                coarse_lower = datetime(datetime.fromtimestamp(completed,tz).year,month,day,hour,minute,second,
                                        tzinfo=tz).timestamp()
            preceding = [t for number,t in anchors if number <= line_number]
            diagnostics.append({'line':line_number,'kind':kind,'message':line,
                'precise_observed_unix':timestamp,
                'precise_seconds_relative_to_complete_marker':timestamp-completed if timestamp is not None else None,
                'own_second_precision_log_time_lower_bound':coarse_lower,
                'own_second_precision_log_time_upper_bound':coarse_lower+1 if coarse_lower is not None else None,
                'after_marker_proven_by_preceding_precise_log_anchor':bool(preceding and preceding[-1] > completed),
                'after_marker_proven_by_own_timestamp_or_preceding_precise_anchor':
                    bool((coarse_lower is not None and coarse_lower > completed) or (preceding and preceding[-1] > completed))})
        exits = [e for e in schedulers[model] if e.get('kind') == 'stage_exited' and e.get('stage') == stage] if stage else []
        if stage:
            audit.check(len(exits) == 1 and exits[0]['exit_code'] == 0 and exits[0]['marker_exists'] is True
                        and exits[0]['time_unix'] >= completed,
                        'final_scheduler_actual_zero_exit_after_completed_stage', model=model,stage=stage)
        qwen_off = run == 'run_qwen_cache0'
        if qwen_off:
            audit.check(any(d['kind'] == 'engine_core_client_teardown_diagnostic' and
                            d['after_marker_proven_by_preceding_precise_log_anchor'] for d in diagnostics)
                        and any(d['kind'] == 'nanobind_reference_counting_diagnostic' for d in diagnostics),
                        'final_qwen_OFF_completed_then_retained_teardown_diagnostics')
        rows.append({'run':run,'execution_log':filename,'complete_marker_unix':completed,
            'science_outputs_complete':True,'declared_scientific_failure_file':(root/run/'FAILURE.json').exists(),
            'launcher_exit_code':exits[0]['exit_code'] if exits else None,
            'launcher_exit_evidence':'scheduler stage_exited record' if exits else
                'not separately recorded by this scheduler for original ON; full output marker and request chains verified',
            'diagnostics':diagnostics,
            'interpretation':'Retained teardown/runtime warnings near completed output; separate from startup archives and missing/failed scientific results. Precise relative timing is reported, not assumed for all warnings.'})
    return rows


def joint_flags(original):
    metrics = {(r['model'], r['case_id'], r['policy'], r['contract']): r for r in original['policy_metrics']}
    costs = {(r['model'], r['case_id'], r['policy']): r for r in original['costs'] if r['phase'] == 'total'}
    comparisons = {(r['model'], r['case_id'], r['policy'], r['reference'], r['contract']): r
                   for r in original['comparisons'] if r.get('status') == 'available' and not r.get('comparison_role')}
    rows = []
    for case in original['case_matrix']:
        if not case['primary_matrix']:
            continue
        model, case_id = case['model'], case['case_id']
        counts = {contract: {role: metrics[model, case_id, role, contract] for role in ('CURRENT_CP', 'S1', 'S2')}
                  for contract in ('native', 'strict')}
        cp, s1 = costs[model, case_id, 'CURRENT_CP'], costs[model, case_id, 'S1']
        tuple_row = {**case, 'policy':'CURRENT_CP','display_policy':'Current80-S',
            'native_delivery': counts['native']['CURRENT_CP']['complete'],
            'strict_validity': {role: counts['strict'][role]['complete'] for role in counts['strict']},
            'native_counts': {role: counts['native'][role]['predicted_count'] for role in counts['native']},
            'source_positive_count': counts['native']['CURRENT_CP']['gold_positive_count'],
            'native_source_quality': {role: {field: counts['native'][role][field] for field in
                ('row_error_full','fp_valid','fn_valid','count_error_signed','count_error_absolute_over_n')}
                for role in counts['native']},
            'selected_k_gt_1': case['selected_k'] > 1,
            'complete_policy_fewer_requests_than_native_S1': cp['requests'] < s1['requests'],
            'complete_policy_fewer_raw_tokens_than_native_S1': cp['total_tokens'] < s1['total_tokens'],
            'resource_bill': {'CURRENT_CP_full': cp, 'native_S1': s1},
            'scope': 'descriptive finite primary panel; no joint statistical rejection claim'}
        for contract, flag_name in [('strict', 'frozen_strict_vector_joint_benefit'),
                                    ('native', 'native_semantics_joint_benefit')]:
            losses = {role: comparisons[model, case_id, 'CURRENT_CP', role, contract].get('disagreement_rate_full')
                      for role in ('S1', 'S2')}
            complete = all(counts[contract][role]['complete'] for role in counts[contract])
            tuple_row[contract+'_dF'] = losses
            tuple_row[flag_name] = (case['selected_k'] > 1 and all(v is not None and v <= .05 for v in losses.values())
                and cp['requests'] < s1['requests'] and cp['total_tokens'] < s1['total_tokens']) if complete else None
            tuple_row[flag_name+'_unavailable_reason'] = None if complete else 'one or more required '+contract+' vectors incomplete'
        rows.append(tuple_row)
    return rows


def budget_costs_with_paired_cp80(budget):
    base = {(r['model'],r['workload_id'],r['repeat'],r['phase']):r
            for r in budget['costs'] if r['budget'] == 80}
    rows = []
    for row in budget['costs']:
        reference = base[row['model'],row['workload_id'],row['repeat'],row['phase']]
        expanded = {**row, 'paired_CP80_job_id':reference['job_id'],
            'comparison_scope':'same model/input/rep/phase CP80; planned contexts paired, physical calls independent'}
        for field in ('requests','input_tokens','output_tokens','total_tokens','cached_tokens','uncached_tokens'):
            left, right = reference[field], row[field]
            expanded[field+'_ratio_vs_paired_CP80'] = a.ratio(right,left)
            expanded[field+'_minus_paired_CP80'] = right-left if right is not None and left is not None else None
        uncached = row['uncached_tokens']
        expanded['uncached_input_plus_output_tokens'] = uncached+row['output_tokens'] if uncached is not None else None
        expanded['uncached_input_plus_output_tokens_lower_bound'] = row['uncached_tokens_lower_bound']+row['output_tokens']
        expanded['uncached_input_plus_output_tokens_upper_bound'] = row['uncached_tokens_upper_bound']+row['output_tokens']
        rows.append(expanded)
    return rows


def primary_native_plan_benefits(original):
    metrics = {(r['model'],r['case_id'],r['policy'],r['contract']):r for r in original['policy_metrics']}
    costs = {(r['model'],r['case_id'],r['policy']):r for r in original['costs'] if r['phase'] == 'total'}
    comparisons = {(r['model'],r['case_id'],r['policy'],r['reference']):r
        for r in original['comparisons'] if r['contract'] == 'native' and r.get('status') == 'available'
        and not r.get('comparison_role')}
    rows = []
    for case in original['case_matrix']:
        if not case['primary_matrix']:
            continue
        model, case_id = case['model'],case['case_id']
        for policy in ('RANDOM8','SIMILAR8','CURRENT_CP'):
            k = case['selected_k'] if policy == 'CURRENT_CP' else 8
            measured, singleton = costs[model,case_id,policy],costs[model,case_id,'S1']
            roles = (policy,'S1','S2')
            complete = all(metrics[model,case_id,role,'native']['complete'] for role in roles)
            losses = {role:comparisons[model,case_id,policy,role]['disagreement_rate_full'] for role in ('S1','S2')}
            fewer_calls = measured['requests'] < singleton['requests']
            fewer_raw_tokens = measured['total_tokens'] < singleton['total_tokens']
            row = {key:value for key,value in case.items() if key != 'selected_k'}
            row.update(policy=policy, display_policy='Current80-S' if policy == 'CURRENT_CP' else policy,
                actual_policy_k=k, current80s_selected_k=case['selected_k'],
                full_native_vectors_complete=complete, native_d1_vs_S1=losses['S1'], native_d2_vs_S2=losses['S2'],
                full_requests=measured['requests'], S1_requests=singleton['requests'],
                full_raw_input_plus_output_tokens=measured['total_tokens'],S1_raw_input_plus_output_tokens=singleton['total_tokens'],
                full_requests_ratio_vs_S1=a.ratio(measured['requests'],singleton['requests']),
                full_raw_tokens_ratio_vs_S1=a.ratio(measured['total_tokens'],singleton['total_tokens']),
                fewer_requests_than_S1=fewer_calls, fewer_raw_tokens_than_S1=fewer_raw_tokens,
                native_semantics_joint_benefit=(k>1 and all(v is not None and v <= .05 for v in losses.values())
                    and fewer_calls and fewer_raw_tokens) if complete else None,
                certificate_scope='fixed plan has no selection certificate' if policy != 'CURRENT_CP' else
                    'frozen strict-gated CP80 trial-law selection; this native full-plan evaluation is descriptive',
                interpretation='finite realised-plan observation against both native reference vectors; not a certificate or source-gold correctness claim')
            rows.append(row)
    return rows


def main(root):
    root = Path(root).resolve()
    out = root/'real_engine_analysis/final_actual_audit'
    required = [root/f'run_{m}_cache{c}/ALL_WORK_COMPLETE.json' for m in ('qwen','glm') for c in (0,1)] + [
        root/f'budget_run_{m}/BUDGET_ALL_WORK_COMPLETE.json' for m in ('qwen','glm')]
    if any(not p.exists() for p in required):
        raise RuntimeError('Finalization refused: six completion markers are not all present')
    original, budget, shadow = [load(out/p) for p in ('original/summary.json','budget/summary.json','native_shadow/MATRIX_AUDIT.json')]
    original_integrity, budget_integrity = [load(out/p) for p in ('original/integrity.json','budget/runtime_integrity.json')]
    audit = a.Audit()
    audit.check(original['analysis_status'] == 'complete' and original['completed_configurations'] == 36
                and original_integrity['error_count'] == 0, 'final_actual_original_36_deep_complete')
    audit.check(budget['status'] == 'complete' and budget['completed_policies'] == 16
                and budget_integrity['error_count'] == 0, 'final_actual_budget_16_deep_complete')
    audit.check(shadow['expected_configurations'] == 36 and shadow['pending'] == 0 and len(shadow['candidates']) == 108,
                'final_actual_native_shadow_all_108_slots_accounted')
    for row in original_integrity['runs']+budget_integrity['runs']:
        audit.check(row['deep_integrity']['status'] == 'checked_completed_run', 'final_six_actual_deep_request_chains', path=row.get('path',row.get('directory')))
        audit.check(row['failure'] is None, 'final_no_unreported_failure_in_six_completed_run_directories',
                    path=row.get('path',row.get('directory')))
    study, budget_study = load(root/'prepared/STUDY.json'), load(root/'BUDGET_FOLLOWUP_STUDY.json')
    runtime = runtime_checks(root, study, budget_study, audit)
    failures = failure_checks(root, audit)
    public_receipts = public_receipt_checks(root, audit)
    teardown = teardown_receipts(root, audit)
    flags = joint_flags(original)
    all_plan_flags = primary_native_plan_benefits(original)
    audit.check(len(flags) == 12, 'final_twelve_original_primary_cells')
    audit.check(len(all_plan_flags) == 36 and all(
        next(r for r in all_plan_flags if r['model'] == row['model'] and r['case_id'] == row['case_id']
             and r['policy'] == 'CURRENT_CP')['native_semantics_joint_benefit'] == row['native_semantics_joint_benefit']
        for row in flags), 'final_all_three_primary_plans_and_original_CP_flag_consistent')
    original_metrics = [norm_counts(r) for r in original['policy_metrics']]
    budget_metrics = [norm_counts(r) for r in budget['metrics']]
    audit.check(len(original_metrics) == 360 and len(budget_metrics) == 32
                and all(r['gold_available'] == r['n'] for r in original_metrics+budget_metrics),
                'final_all_native_and_strict_metrics_have_complete_source_labels')
    budget_paired_costs = budget_costs_with_paired_cp80(budget)
    primary_native_s1 = [r for r in original_metrics if r['primary_matrix']
                         and r['contract'] == 'native' and r['policy'] == 'S1']
    most_fallback = max(primary_native_s1, key=lambda r:r['native_fallback_count']/r['n'])
    official = []
    for row in original['official_q3']:
        expanded = dict(row)
        audit.check(row['record_count'] == row['native_gold_available'] == row['strict_gold_available'] == 120,
                    'final_each_actual_official_q3_has_120_source_labels', model=row['model'],
                    prefix_cache=row['prefix_cache'])
        for contract in ('native','strict'):
            expanded[contract+'_count_normalization'] = norm_counts({'n':row[contract+'_n'],
                'gold_positive_count':row[contract+'_gold_positive_count'], 'predicted_count':row[contract+'_predicted_count']})
        official.append(expanded)
    for field in ('requests','input_tokens','output_tokens','total_tokens','cached_tokens_observed_lower_bound',
                  'cached_tokens_unknown_requests','uncached_tokens_lower_bound','uncached_tokens_upper_bound'):
        audit.check(sum(r[field] for r in runtime['native_request_usage_by_run_and_purpose']) ==
            sum(r['observed_research_union_usage'][field] for r in original_integrity['runs']+budget_integrity['runs']),
            'final_disjoint_purpose_ledger_balances_all_six_actual_runs', field=field)
    errors = sum(i['severity'] == 'error' for i in audit.issues)
    supplemental = {'successful_checks':audit.successful_checks, 'error_count':errors, 'issues':audit.issues,
                    'fingerprints':list(audit.fingerprints.values()), **runtime, 'failure_archives':failures,
                    'preserved_public_verification_receipts':public_receipts,'completed_output_teardown_diagnostics':teardown}
    (out/'ACTUAL_RUNTIME_AND_ARCHIVE_VERIFICATION.json').write_text(json.dumps(supplemental,indent=2,ensure_ascii=False)+'\n')
    if errors:
        raise RuntimeError(f'Final actual verification found {errors} discrepancies; no canonical completion file written')
    flag_summary = {name: {'pass':sum(r[name] is True for r in flags), 'false':sum(r[name] is False for r in flags),
                          'unavailable':sum(r[name] is None for r in flags), 'fixed_denominator':12}
                    for name in ('frozen_strict_vector_joint_benefit','native_semantics_joint_benefit')}
    plan_flag_summary = {policy:{'pass':sum(r['native_semantics_joint_benefit'] is True for r in all_plan_flags if r['policy'] == policy),
        'false':sum(r['native_semantics_joint_benefit'] is False for r in all_plan_flags if r['policy'] == policy),
        'unavailable':sum(r['native_semantics_joint_benefit'] is None for r in all_plan_flags if r['policy'] == policy),
        'fixed_denominator':12, 'certificate':'none for fixed plans; descriptive native evaluation for Current80-S'}
        for policy in ('RANDOM8','SIMILAR8','CURRENT_CP')}
    scope_usage = [r['observed_research_union_usage'] for r in original_integrity['runs']+budget_integrity['runs']]
    usage_totals = {key:sum(u[key] for u in scope_usage) for key in ('requests','input_tokens','output_tokens','total_tokens',
        'cached_tokens_unknown_requests','cached_tokens_observed_lower_bound','cached_tokens_upper_bound',
        'uncached_tokens_lower_bound','uncached_tokens_upper_bound')}
    usage_totals['cached_tokens'] = sum(u['cached_tokens'] for u in scope_usage) if all(u['cached_tokens'] is not None for u in scope_usage) else None
    usage_totals['uncached_tokens'] = sum(u['uncached_tokens'] for u in scope_usage) if all(u['uncached_tokens'] is not None for u in scope_usage) else None
    result = {'status':'COMPLETE_VERIFIED_ACTUAL_DATA', 'created_utc':datetime.now(timezone.utc).isoformat(),
        'analysis_scope':'Post-run CPU analysis; no new inference and no changes to frozen scientific artifacts.',
        'completed_run_definition':'All planned scientific outputs and raw request chains complete, with no run FAILURE.json; launcher exit receipts and teardown diagnostics are separate fields.',
        'analysis_code_sha256':{str(path.relative_to(root)):a.sha_file(path) for path in [
            Path(__file__).resolve(),root/'real_engine_analysis/analyze_study.py',
            root/'real_engine_analysis/budget_followup/analyze_budget_followup.py',
            root/'real_engine_analysis/budget_followup/analysis_common.py',
            root/'real_engine_analysis/representativeness_audit/audit_representativeness.py']},
        'completion_marker_sha256':{str(path.relative_to(root)):a.sha_file(path) for path in required},
        'coverage':{'original_configurations':36,'budget_policies':16,'original_candidate_slots':108,
            'observed_original_candidates':sum(r.get('observed_trials') is not None for r in shadow['candidates']),
            'unreached_original_candidate_slots':sum(r.get('observed_trials') is None for r in shadow['candidates']),
            'shadow_same_actual_decision_and_cost':shadow['same_actual_decision_and_cost'],
            'shadow_needs_faithful_followup':shadow['needs_faithful_followup'],'shadow_pending':0,
            'successful_runs':6,'successful_original_main_runs':4,'successful_budget_runs':2,
            'failed_archives':3,'original_main_launcher_attempts_preserved':7,
            'original_five_policy_executions':180,'official_original_query_integration_executions':4,
            'independent_validation_probe_runs':sum(r['selected_k']>1 for r in original['case_matrix'])},
        'verification':{'original_deep_successful_checks':original_integrity['successful_checks'],
            'budget_deep_successful_checks':budget_integrity['successful_checks'],
            'supplemental_successful_checks':audit.successful_checks,'integrity_errors':0},
        'primary_joint_benefit_definitions':{'evaluated_policy':'CURRENT_CP / Current80-S only; other plans are separately reported.',
            'frozen_strict_vector_joint_benefit':'All required strict vectors complete, selected k>1, dF<=.05 to BOTH strict S1/S2, and full calibration+execution requests/raw tokens less than native S1.',
            'native_semantics_joint_benefit':'Complete actual native Boolean vectors, selected k>1, native dF<=.05 to BOTH native S1/S2, and full calibration+execution requests/raw tokens less than native S1.',
            'null_rule':'Missing required vector => unavailable, retained in fixed denominator; both flags are finite-panel descriptive readouts, not competing confirmatory tests.'},
        'primary_joint_benefit_summary':flag_summary,'primary_delivery_fidelity_resource_tuples':flags,
        'primary_native_plan_joint_benefit_summary':plan_flag_summary,
        'primary_native_plan_delivery_fidelity_resource_tuples':all_plan_flags,
        'original_case_matrix':original['case_matrix'],'original_native_and_strict_quality':original_metrics,
        'original_fidelity_comparisons':original['comparisons'],'original_policy_costs':original['costs'],
        'original_cp_decisions':original['cp_decisions'],'original_independent_validation':original['validation'],
        'original_native_shadow':shadow,'official_q3_actual':official,
        'budget_job_matrix':budget['job_matrix'],'budget_native_and_strict_quality':budget_metrics,
        'budget_reference_fidelity':budget['reference_fidelity'],'budget_policy_costs':budget['costs'],
        'budget_policy_costs_vs_paired_CP80_and_original_references':budget_paired_costs,
        'budget_cp_candidates':budget['cp_candidates'],'budget_paired_budgets':budget['paired_budgets'],
        'budget_paired_repeats':budget['paired_repeats'],'budget_paired_candidate_prefixes':budget['paired_candidate_prefixes'],
        'budget_summary':budget['budget_summary'],'exact_budget_operating_characteristics':budget['exact_operating_characteristics'],
        'covered_successful_run_usage':usage_totals,'failure_archives':failures,
        'native_request_usage_by_run_and_purpose':runtime['native_request_usage_by_run_and_purpose'],
        'native_request_unit':'One prompt/native vLLM RequestOutput, not a query or a parent generate() batch. Gold-label oracle reads are outside this native-request ledger.',
        'original_main_run_outcomes':original_integrity['runs'],'budget_run_outcomes':budget_integrity['runs'],
        'preserved_public_verification_receipts':public_receipts,
        'completed_output_teardown_diagnostics':teardown,
        'remaining_review_risks':[
            {'risk':'system scope and reference quality',
             'scope':'Actual LOTUS singleton reference; custom batched operator and paper-defined CP; two models and two domains.',
             'measured_primary_S1_largest_default_fraction':{
                 key:most_fallback[key] for key in ('model','case_id','n','native_fallback_count',
                     'predicted_count','gold_positive_count','row_error_full')},
             'interpretation':'Native fidelity is not source-gold correctness; large default rates limit extrapolation to a well-behaved semantic checker.'},
            {'risk':'incremental research contribution',
             'interpretation':'Prior work already studies batching/context variability and calibration. These auditable fixed-budget costs do not by themselves establish novelty or determine acceptance.'}],
        'runtime_verification_file':'ACTUAL_RUNTIME_AND_ARCHIVE_VERIFICATION.json',
        'confidence_and_history_scope':['Original calibration remains the frozen strict-gated rule; native-semantic readout is separately named.',
            'Budget follow-up targets native parsed Boolean with disclosed default=True; strict format is separate.',
            'delta/3 is within each fixed budget/candidate family, not a joint certificate for a budget winner.',
            'Nominal CP bounds require the intended independent random-batch/oracle trial law; they do not certify a realised full partition, a future singleton run, or source-gold correctness.',
            'Two models, two domains; repetitions/scales/caches/predicates do not add independent domains.',
            'Observed policy elapsed timers start after reset_policy; boundary reset and model loading are outside those timers. Request/token bills include all actual calibration and execution calls.',
            'Covered usage includes six runs with their controls/Q3/validation, plus three separately reported startup archives; it is not all historical project work.',
            'Zero returned generation requests does not mean zero initialization cost.']}
    for filename, rows in [('ORIGINAL_QUALITY_WITH_COUNT_N_AND_G.csv',original_metrics),
                            ('BUDGET_QUALITY_WITH_COUNT_N_AND_G.csv',budget_metrics),
                            ('BUDGET_COSTS_VS_PAIRED_CP80_AND_ORIGINAL_S1_S2.csv',budget_paired_costs),
                            ('PRIMARY_NATIVE_PLAN_JOINT_BENEFIT_ALL_THREE_POLICIES.csv',all_plan_flags),
                            ('NATIVE_REQUEST_USAGE_BY_RUN_AND_PURPOSE.csv',runtime['native_request_usage_by_run_and_purpose']),
                            ('PRIMARY_JOINT_BENEFIT_BOTH_DEFINITIONS.csv',flags)]:
        a.write_csv(out/filename, rows)
    original_k = Counter(row['selected_k'] for row in original['case_matrix'])
    pairs = budget['paired_budgets']
    calls_better = sum(row['full_requests_right_minus_left'] < 0 for row in pairs)
    tokens_better = sum(row['full_total_tokens_right_minus_left'] < 0 for row in pairs)
    lines = ['# 最终实际数据核验', '',
        '36/36 原配置、16/16 预算策略及六条实际请求链完整核验通过；完整性错误为0。108个原候选位置全部交代，未调用候选不伪造结果。', '',
        f"原CURRENT_CP选中批大小分布：{dict(original_k)}。Native shadow同决定/同调用成本 {shadow['same_actual_decision_and_cost']}/36；需要faithful follow-up {shadow['needs_faithful_followup']}/36，pending为0。", '',
        f"Current80-S的固定12格primary：严格向量联合净收益 {flag_summary['frozen_strict_vector_joint_benefit']['pass']}/12（unavailable {flag_summary['frozen_strict_vector_joint_benefit']['unavailable']}）；原生布尔语义口径 {flag_summary['native_semantics_joint_benefit']['pass']}/12（unavailable {flag_summary['native_semantics_joint_benefit']['unavailable']}）。两种定义透明并列，不改原校准门槛。", '',
        '同一固定12格的三策略native净收益：'+ '；'.join(
            ('Current80-S' if policy == 'CURRENT_CP' else policy)+f" {row['pass']}/12（unavailable {row['unavailable']}）"
            for policy,row in plan_flag_summary.items())+'。条件均为k>1、对native S1/S2两者dF≤.05、完整requests及raw tokens都低于S1。RANDOM8/SIMILAR8是无证书固定计划；这些有限面板描述性结果不外推所有计划。', '',
        f'预算配对共8组：320预算比80预算少用完整策略请求的有{calls_better}组，少用完整原始input+output tokens的有{tokens_better}组。均包含各自校准与执行，不给预算赢家联合95%保证。', '',
        '精确CP口径为每个固定预算内3候选分配delta/3：80次完整探针最多允许0次不一致；320次最多允许7次。所有提前停止仍按实际发出chunk的完整调用记账。', '',
        '| 模型 | 工作负载 | 预算 | 两次选k | 平均完整请求 | 平均完整raw input+output tokens | 平均完整uncached input tokens |',
        '|---|---|---:|---|---:|---:|---:|']
    for row in budget['budget_summary']:
        lines.append(f"| {row['model']} | {row['workload_id']} | {row['budget']} | {row.get('selected_sizes_by_repeat')} | {row.get('mean_full_requests')} | {row.get('mean_full_total_tokens')} | {row.get('mean_full_uncached_tokens')} |")
    def show(value):
        if value is None:
            return 'unavailable'
        if type(value) is bool:
            return 'true' if value else 'false'
        return f'{value:.6f}' if isinstance(value, float) else str(value)
    lines += ['', '固定primary 12格的完整交付、参考保真与资源读数：', '',
        '| 模型 | case | k | native COUNT / G | native dF→S1 / S2 | strict完整 CP/S1/S2 | 完整calls / S1 | 完整raw tokens / S1 | strict / native净收益 |',
        '|---|---|---:|---|---|---|---:|---:|---|']
    for row in flags:
        validity = '/'.join(show(row['strict_validity'][role]) for role in ('CURRENT_CP','S1','S2'))
        cost, ref = row['resource_bill']['CURRENT_CP_full'], row['resource_bill']['native_S1']
        lines.append(f"| {row['model']} | {row['case_id']} | {row['selected_k']} | {row['native_counts']['CURRENT_CP']} / {row['source_positive_count']} | {show(row['native_dF']['S1'])} / {show(row['native_dF']['S2'])} | {validity} | {show(cost['requests']/ref['requests'])} | {show(cost['total_tokens']/ref['total_tokens'])} | {show(row['frozen_strict_vector_joint_benefit'])} / {show(row['native_semantics_joint_benefit'])} |")
    lines += ['', '原SemBench Q3原方法在四次主运行中真实执行，完整输入关系均为120行：', '',
        '| 模型 | prefix cache | native COUNT | strict COUNT | 源标签 G | native defaults |',
        '|---|---|---:|---|---:|---:|']
    for row in official:
        lines.append(f"| {row['model']} | {row['prefix_cache']} | {row['native_predicted_count']} | {show(row['strict_predicted_count'])} | {row['native_gold_positive_count']} | {row['native_fallback_count']} |")
    lines += ['', '原生COUNT/源标签FP-FN/行错误与strict状态均保留；COUNT误差按N及G两个分母给出。G=0且预测0时error/G为0，G=0但预测非0时为null并标原因。', '',
        f"四条主运行成功、两条budget运行成功；另保留三份失败启动archive（主launcher共7次保留尝试），不把GLM的零generate说成零初始化工作。六条成功运行合计返回{usage_totals['requests']}个native请求，单位为一个prompt/RequestOutput，绝非完整查询次数；S1/S2、校准refs/probes、执行、研究validation、controls、官方Q3与budget分账见NATIVE_REQUEST_USAGE_BY_RUN_AND_PURPOSE.csv。", '',
        f"实际GLM的动态模块仅包含已固定tokenizer（SHA-256 {runtime['glm_tokenizer_sha256']}）；模型与配置没有动态模块，native类/参数数/BF16已核对。缓存开启策略每次真实BlockPool空缓存/PID记录均匹配原生日志；缓存关闭运行核对实际engine profile与返回cached字段。", '',
        '原始执行日志中的NCCL/nanobind/refcount/EngineCore teardown诊断全部保留并列出时间证据。Qwen OFF在完整标记后出现这些诊断，scheduler记录launcher exit0；其他运行分别记录其实际时间关系及可用exit证据。它们与三份启动失败archive及缺失科学结果分账，不因含ERROR字样而改写已核验结果。', '',
        '仍存的两项研究风险：', '',
        f"1. 系统范围与参考质量：实际使用LOTUS参考算子，批处理为明确自定义物理算子、CP为论文定义复现；两模型、两域不等于完整优化器广泛复现。参考质量也有限：primary S1最高默认率出现在{most_fallback['model']} {most_fallback['case_id']}，为{most_fallback['native_fallback_count']}/{most_fallback['n']}，native COUNT={most_fallback['predicted_count']}、源G={most_fallback['gold_positive_count']}、源标签行错误={most_fallback['row_error_full']:.6f}；高native fidelity不能替代源标签正确性。",
        '2. 新认识的幅度仍需与近邻比较：参考变异、批处理扰动与校准开销已有前例；更大固定预算的功效/完整成本对照及可核验实例能补证据，但不能据可靠性、请求数量或经典统计控制本身预设新颖性与接收结论。', '',
        '所有数值的可引用统一版本见 COMPLETE_VERIFIED_NUMBERS.json；本文件不修改论文或发布材料。']
    (out/'READOUT.md').write_text('\n'.join(lines)+'\n')
    result['snapshot_files'] = {str(path.relative_to(out)):a.sha_file(path) for path in [
        out/'original/summary.json',out/'original/integrity.json',out/'budget/summary.json',
        out/'budget/runtime_integrity.json',out/'native_shadow/MATRIX_AUDIT.json',
        out/'ACTUAL_RUNTIME_AND_ARCHIVE_VERIFICATION.json',out/'READOUT.md',
        out/'ORIGINAL_QUALITY_WITH_COUNT_N_AND_G.csv',out/'BUDGET_QUALITY_WITH_COUNT_N_AND_G.csv',
        out/'BUDGET_COSTS_VS_PAIRED_CP80_AND_ORIGINAL_S1_S2.csv',
        out/'PRIMARY_NATIVE_PLAN_JOINT_BENEFIT_ALL_THREE_POLICIES.csv',
        out/'NATIVE_REQUEST_USAGE_BY_RUN_AND_PURPOSE.csv',
        out/'PRIMARY_JOINT_BENEFIT_BOTH_DEFINITIONS.csv']}
    temporary = out/'COMPLETE_VERIFIED_NUMBERS.json.tmp'
    temporary.write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    temporary.replace(out/'COMPLETE_VERIFIED_NUMBERS.json')
    print(json.dumps({'status':result['status'],'coverage':result['coverage'],
                      'primary_joint_benefit_summary':flag_summary,
                      'primary_native_plan_joint_benefit_summary':plan_flag_summary},indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True)
    main(parser.parse_args().root)
