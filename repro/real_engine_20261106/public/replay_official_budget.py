"""Run the unchanged separately authored budget analyzer with explicit I/O adapters."""
from pathlib import Path
import sys
from portable_common import import_file, load, sha, write_json

TABLES = ('job_matrix', 'metrics', 'reference_fidelity', 'costs', 'cp_candidates',
          'calibration_reference_statistics', 'paired_budgets', 'paired_repeats',
          'paired_candidate_prefixes', 'budget_summary', 'exact_operating_characteristics')


def run_official(root, out, *, compact_mode=False):
    from replay_compact import compact_chain, install_adapters
    root, out = Path(root), Path(out)
    folder = root/'real_engine_analysis/budget_followup'
    helper = import_file('analysis_common', folder/'analysis_common.py')
    sys.modules['analysis_common'] = helper
    module = import_file('separately_authored_budget_analysis', folder/'analyze_budget_followup.py')
    excluded = []
    if compact_mode:
        install_adapters(helper, root, load(root/'COMPACT_BINDINGS.json'))
        original = module.BudgetRun
        class PublicBudgetRun(original):
            def __init__(self, path, audit, model, **kwargs):
                kwargs['deep'] = False
                super().__init__(path, audit, model, **kwargs)
                self.deep_result = compact_chain(path, synthetic=load(root/'COMPACT_SCOPE.json')['payload_kind'] == 'SYNTHETIC_CPU_FIXTURE_NOT_MODEL_EVIDENCE')
        module.BudgetRun = PublicBudgetRun
        read_original = helper.read_json
        old_root = Path(load(root/'COMPACT_SCOPE.json')['original_root'])
        projected = set()
        for workload in load(root/'prepared/STUDY.json')['workloads']:
            for key in ('records', 'native_requests', 'metadata'):
                if key in workload:
                    projected.add(str(Path(workload[key]).relative_to(old_root)))
        def scoped_read(path, audit, **kwargs):
            value = read_original(path, audit, **kwargs)
            if Path(path) == root/'BUDGET_FOLLOWUP_FREEZE.json' and value:
                selected = {}
                for rel, expected in value['files'].items():
                    source = root/rel
                    if rel in projected or not source.exists():
                        excluded.append(rel)
                    else:
                        selected[rel] = expected
                # Explicit reader projection only: original freeze bytes stay
                # unchanged and full-private analysis verifies every original
                # file. This read view never certifies omitted text as present.
                value = {**value, 'files': selected}
            return value
        helper.read_json = scoped_read
    summary, integrity = module.analyze(root, root/'BUDGET_FOLLOWUP_STUDY.json', out,
        deep=not compact_mode, relocate_from=load(root/'COMPACT_SCOPE.json')['original_root'] if compact_mode else None)
    if summary['status'] != 'complete' or integrity['error_count']:
        raise ValueError('Separately authored budget analyzer has pending data or integrity errors')
    if compact_mode:
        write_json(out/'EXPLICIT_COMPACT_READER_SCOPE.json', {
            'original_analysis_code_edited': False,
            'original_budget_freeze_file_bytes_changed': False,
            'full_private_freeze_verification_claimed_for_this_compact_run': False,
            'omitted_or_projected_full_input_members': sorted(set(excluded)),
            'scope': 'Unchanged mathematics. Original full-text/token-array file readers are explicitly adapted; full-private and compact scientific tables are separately compared.'})
    return summary, {name: summary[name] for name in TABLES}


def compare_views(own, official, out):
    jobs = {(r['model'], r['job_id']): r for r in own['jobs']}
    checked = 0
    def equal(a, b, name):
        nonlocal checked
        if isinstance(a, float) or isinstance(b, float):
            ok = a is not None and b is not None and abs(a-b) <= 2e-10
        else:
            ok = a == b
        if not ok:
            raise ValueError('Own/separately authored budget analysis differs: '+name)
        checked += 1
    for row in official['job_matrix']:
        ours = jobs[row['model'], row['job_id']]
        equal(ours['selected_k'], row['selected_k'], 'selected_k')
        equal(ours['native_source_quality']['complete'], row['native_complete'], 'native completion')
        equal(ours['strict_source_quality']['complete'], row['strict_complete'], 'strict completion')
    for row in official['metrics']:
        ours = jobs[row['model'], row['job_id']]
        stats = ours['native_source_quality'] if row['contract']=='native' else ours['strict_source_quality']
        for key, value in stats.items():
            equal(value, row[key], 'quality/'+key)
    for row in official['reference_fidelity']:
        ours = jobs[row['model'], row['job_id']]['reference_comparisons'][row['reference']]
        paired = ours['native_boolean'] if row['contract']=='native' else ours['strict_format']
        for key, value in paired.items():
            equal(value, row[key], 'reference fidelity/'+key)
    for row in official['costs']:
        ours = jobs[row['model'], row['job_id']]
        cost = ours['usage'] if row['phase']=='full' else ours[row['phase']+'_usage']
        for key, value in cost.items():
            equal(value, row[key], 'cost/'+row['phase']+'/'+key)
    for row in official['cp_candidates']:
        ours = jobs[row['model'], row['job_id']]
        by_k = {d['k']:d for d in ours['calibration_candidates']}
        if row['status'] == 'not_reached_after_larger_pass':
            equal(row['k'] in by_k, False, 'unreached candidate')
            continue
        candidate = by_k[row['k']]
        for ours_key, author_key in [('budget','budget'),('actually_checked','observed_trials'),
                ('agreements_native_boolean','agreements'),('lower_exact','lower_exact_binomial'),
                ('possible_final_lower_at_stop','best_possible_final_lower_at_stop'),('passed','passed'),
                ('first_impossible_probe','first_impossible_probe'),('overissued_billed_calls','overissued_but_billed_probe_calls')]:
            equal(candidate[ours_key], row[author_key], 'CP/'+ours_key)
    for row in official['calibration_reference_statistics']:
        ours = jobs[row['model'], row['job_id']]
        equal(ours['budget'],row['reference_draws'],'fresh own reference count')
        equal(ours['native_reference_strict_valid_rows'],row['strict_valid_rows'],'strict reference diagnostic')
    write_json(out, {'status':'PASS','compared_fields':checked,'policy_jobs':len(jobs),
        'scope':'Own independent raw/CP/cost replay equals the separately authored analyzer on all shared selection, completion, quality, fidelity, per-phase cost and candidate-bound fields. Author paired/aggregate/theory tables are additionally reproduced unchanged from compact input.',
        'new_model_inference':False})
