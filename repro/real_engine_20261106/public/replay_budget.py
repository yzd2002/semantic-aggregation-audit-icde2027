"""Separate native-Boolean CP80/CP320 follow-up mathematics.

This family was prospectively added after an original case was observed. It
does not replace the 36-cell matrix or create additional independent domains.
Same standard CP, classification and token-cost helpers as the main audit.
"""
from collections import defaultdict
import json
from pathlib import Path
from portable_common import import_file, lines, load, sha, write_json


def require(ok, message):
    if not ok:
        raise ValueError(message)


def analyze_budget(root, out, *, compact_mode=False):
    from replay_compact import compact_chain, install_adapters
    root, out = Path(root), Path(out)
    out.mkdir(parents=True, exist_ok=False)
    config = load(root/'BUDGET_FOLLOWUP_STUDY.json')
    base_study = load(root/'prepared/STUDY.json')
    original_root = Path(load(root/'COMPACT_SCOPE.json')['original_root']) if compact_mode else root
    m = import_file('budget_independent_math_helpers', root/'real_engine_analysis/analyze_study.py')
    audit = m.Audit()
    if compact_mode:
        install_adapters(m, root, load(root/'COMPACT_BINDINGS.json'))
    def map_workload(workload):
        result = dict(workload)
        for key in ('records', 'gold', 'native_requests', 'metadata'):
            if key in result:
                result[key] = str(root/Path(result[key]).relative_to(original_root))
        return result
    workloads = {w['workload_id']: {**m.load_workload(map_workload(w), audit), 'config': base_study['config']}
                 for w in config['selected_workloads']}
    result_rows, union = [], {}
    reference_cache = {}
    class CompactBudgetRun:
        def __init__(self, path):
            self.path = path
            self.by_id = {r['logical_id']: r for r in lines(path/'responses.jsonl')}
            self.by_stage = defaultdict(list)
            for row in self.by_id.values():
                self.by_stage[row['stage']].append(row)
            compact_chain(path, synthetic=load(root/'COMPACT_SCOPE.json')['payload_kind'] == 'SYNTHETIC_CPU_FIXTURE_NOT_MODEL_EVIDENCE')
        def rows(self, stage, *, prefix=False):
            return [r for k, rows in self.by_stage.items() if k == stage or prefix and k.startswith(stage+'/') for r in rows]
    for model in config['models']:
        run_path = root/f'budget_run_{model}'
        complete = load(run_path/'BUDGET_ALL_WORK_COMPLETE.json')
        require(complete['status'] == 'complete' and complete['jobs_complete'] == len(config['jobs_per_model']), 'Budget family incomplete')
        if compact_mode:
            run = CompactBudgetRun(run_path)
        else:
            run = m.Run(run_path, audit, deep=False)
            for name in ('responses', 'raw_returns'):
                require(sha(run_path/(name+'.jsonl')) == complete[name+'_sha256'], 'Budget actual raw hash differs')
            run.deep_result = run.deep_check()
        all_rows = list(run.by_id.values())
        m.compare_usage(complete['all_usage'], m.usage(all_rows), audit, f'budget/{model}/union')
        union[model] = m.usage(all_rows)
        union[model]['controls_only'] = m.usage(run.rows('controls'))
        for job in config['jobs_per_model']:
            job_id, n, budget = job['job_id'], job['n'], job['budget']
            key, data = job['sampling_key'], workloads[job['workload_id']]
            folder = run_path/job_id
            marker, result = load(folder/'POLICY_COMPLETE.json'), load(folder/'POLICY_RESULT.json')
            require(sha(folder/'POLICY_RESULT.json') == marker['result_sha256'], 'Budget policy marker differs')
            require(result['job'] == job, 'Budget job identity differs')
            probes = m.expected_probes(base_study['config'], n, key, 'native_budget_probes', budget)
            calibration_rows = run.rows(job_id+'/calibration')
            references = m.check_native_file(folder/'calibration_singletons/native_result.json',
                [data['records'][i]['record_id'] for i in probes], [data['requests'][i]['messages'] for i in probes],
                run, calibration_rows[:budget], audit, job_id+'/native_reference')
            require(result['native_ref_rows'] == budget and result['native_ref_format_valid_rows'] == sum(x is not None for x in references['strict']), 'Native-reference format diagnostic differs')
            selected, candidate_rows, batch_calls = 1, [], 0
            for ordinal, decision in enumerate(result['calibration']):
                k = decision['k']
                require(k == config['candidates'][ordinal], 'Budget candidates not descending fixed sizes')
                groups, slots = m.expected_probe_batches(base_study['config'], n, k, key, 'native_budget_batches', probes)
                calls = []
                for chunk in sorted(folder.glob(f'candidate_{k}_chunk_*/batch_calls.jsonl'), key=lambda p: int(p.parent.name.rsplit('_', 1)[1])):
                    calls.extend(m.batch_file(chunk, data['requests'][:n], run, audit, job_id+'/candidate')['calls'])
                checked = decision['probes']
                require(len(calls) == len(checked) == decision['actually_checked'] <= budget, 'Budget probe billing count differs')
                agreements = []
                for j, (call, scored) in enumerate(zip(calls, checked)):
                    parsed = m.parse_batch(call['raw'], k)
                    label = parsed[slots[j]] if parsed is not None else None
                    native_ref = references['native'][j]
                    agrees = parsed is not None and label == native_ref
                    require(call['indices'] == groups[j] and scored['probe_index'] == probes[j] and scored['probe_slot'] == slots[j], 'Budget random-batch trial law differs')
                    require(scored['logical_id'] == call['logical_id'] and scored['label'] == label
                            and scored['reference_native_label'] == native_ref
                            and scored['reference_strict_label'] == references['strict'][j]
                            and scored['reference_strict_valid'] == (references['strict'][j] is not None)
                            and scored['agreement'] == agrees, 'Budget native-Boolean agreement differs')
                    agreements.append(agrees)
                successes, checked_n = sum(agreements), len(agreements)
                lower = m.cp_lower(successes, budget, config['delta']/len(config['candidates']))
                possible = m.cp_lower(successes+budget-checked_n, budget, config['delta']/len(config['candidates']))
                passed = checked_n == budget and lower >= 1-config['eta']
                require(successes == decision['agreements'] and abs(lower-decision['lower']) < 2e-10 and passed == decision['passed'], 'Budget exact CP decision differs')
                require(checked_n == budget or possible < 1-config['eta'], 'Budget early rejection still could pass')
                earliest, running = None, 0
                for j, value in enumerate(agreements, 1):
                    running += value
                    if m.cp_lower(running+budget-j, budget, config['delta']/len(config['candidates'])) < 1-config['eta']:
                        earliest = j;break
                candidate_rows.append({'k': k, 'budget': budget, 'actually_checked': checked_n,
                    'agreements_native_boolean': successes, 'lower_exact': lower,
                    'possible_final_lower_at_stop': possible, 'passed': passed,
                    'first_impossible_probe': earliest, 'overissued_billed_calls': checked_n-earliest if earliest is not None else 0})
                batch_calls += checked_n
                if passed:
                    selected = k
                    require(ordinal == len(result['calibration'])-1, 'Budget selection continued after first pass')
                    break
            require(selected == result['selected_k'], 'Budget selected size differs')
            require(selected != 1 or len(result['calibration']) == len(config['candidates']), 'Budget fallback omitted a candidate')
            execution_rows = run.rows(job_id+'/execution')
            if selected == 1:
                parsed = m.check_native_file(folder/'execution_singletons/native_result.json',
                    [r['record_id'] for r in data['records'][:n]], [r['messages'] for r in data['requests'][:n]],
                    run, execution_rows, audit, job_id+'/execution')
                native, strict = parsed['native'], parsed['strict']
                fallback = parsed['fallback_count']
            else:
                parsed = m.batch_file(folder/'execution_partition/batch_calls.jsonl', data['requests'][:n], run, audit, job_id+'/execution', n=n)
                require(parsed['groups'] == m.expected_random_partition(base_study['config'], n, selected, key), 'Budget complete deployment partition differs')
                native = strict = parsed['labels'];fallback = 0
            require(result['labels'] == native, 'Budget materialized native output differs')
            native_stats, strict_stats = m.classification(native, data['gold'][:n]), m.classification(strict, data['gold'][:n])
            require(result['native_count'] == native_stats['predicted_count'] and result['native_complete'] == native_stats['complete'], 'Budget native COUNT/completion differs')
            require(result['strict_count'] == strict_stats['predicted_count'] and result['strict_complete'] == strict_stats['complete'], 'Budget strict-format COUNT/completion differs')
            total, cal, execution = m.usage(run.rows(job_id, prefix=True)), m.usage(calibration_rows), m.usage(execution_rows)
            for label, declared, actual in [('full', result['usage'], total), ('calibration', result['calibration_usage'], cal), ('execution', result['execution_usage'], execution)]:
                m.compare_usage(declared, actual, audit, job_id+'/'+label)
            require(cal['requests'] == budget+batch_calls, 'Budget does not charge fresh references plus all issued probe calls')
            require(total['requests'] == cal['requests']+execution['requests'] and total['total_tokens'] == cal['total_tokens']+execution['total_tokens'], 'Budget full bill identity differs')
            comparisons = {}
            ref_key = (model, job['workload_id'])
            if ref_key not in reference_cache:
                base_case = next(c for c in base_study['cases'] if c['workload_id'] == job['workload_id'] and c['n'] == n and c['prefix_cache'])
                base_run = m.Run(root/f'run_{model}_cache1', audit, deep=False)
                reference_cache[ref_key] = {}
                for policy in ('S1', 'S2'):
                    r = m.check_native_file(base_run.path/base_case['case_id']/policy/'native_result.json',
                        [x['record_id'] for x in data['records'][:n]], [x['messages'] for x in data['requests'][:n]],
                        base_run, base_run.rows(base_case['case_id']+'/'+policy), audit, 'evaluation_only/'+policy)
                    reference_cache[ref_key][policy] = (r, m.usage(base_run.rows(base_case['case_id']+'/'+policy)))
            for policy, (reference, cost) in reference_cache[ref_key].items():
                comparisons[policy] = {'native_boolean': m.paired(reference['native'], native, data['gold'][:n]),
                    'strict_format': m.paired(reference['strict'], strict, data['gold'][:n]),
                    'full_requests_ratio': m.ratio(total['requests'], cost['requests']),
                    'full_raw_token_ratio': m.ratio(total['total_tokens'], cost['total_tokens']),
                    'used_for_calibration': False}
            result_rows.append({'model': model, **job, 'family': 'prospective_native_budget_followup',
                'selected_k': selected, 'calibration_candidates': candidate_rows,
                'native_reference_strict_valid_rows': sum(x is not None for x in references['strict']),
                'native_fallback_count_in_execution': fallback, 'native_source_quality': native_stats,
                'strict_source_quality': strict_stats, 'usage': total, 'calibration_usage': cal,
                'execution_usage': execution, 'reference_comparisons': comparisons,
                'elapsed_seconds_observed': result['elapsed_seconds'], 'calibration_seconds_observed': result['calibration_seconds']})
        billed = [row for row in result_rows if row['model'] == model]
        require(sum(row['usage']['requests'] for row in billed)+union[model]['controls_only']['requests'] == union[model]['requests'], 'Budget family physical union does not equal policies plus controls')
    require(audit.successful_checks >= 0 and not any(x['severity'] == 'error' for x in audit.issues), 'Independent budget source/receipt validation failed')
    summary = {'family': 'prospective_native_budget_followup_after_observed_original_case',
        'status': 'PASS', 'jobs': result_rows, 'physical_union_including_controls': union,
        'model_workload_configurations': len(config['models'])*len(config['selected_workloads']),
        'policy_jobs': len(result_rows),
        'scope': 'The real bounded family has four fixed model/domain configurations and 16 physical policy jobs. Fixture counts are explicitly recorded separately. Budgets/repeats are not new independent domains; no selected-budget joint 95% claim.',
        'reference_contract': 'Native LOTUS Boolean, including separately disclosed default behavior; strict format is diagnostic.',
        'GPU_or_new_model_inference': False}
    write_json(out/'BUDGET_RECOMPUTATION.json', summary)
    return summary
