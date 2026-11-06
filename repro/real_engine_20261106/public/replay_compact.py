#!/usr/bin/env python3
"""CPU mathematical replay of actual compact native-engine evidence.

The original independent analysis and finite-audit mathematics are imported
unchanged. Explicit adapters replace only full-text/token-array file readers.
No source news text, prompt token arrays, weights, network or GPU are needed.
"""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys
from portable_common import compact, digest, import_file, lines, load, sha, write_json


TABLES = ('case_matrix', 'policy_metrics', 'comparisons', 'costs', 'cp_decisions',
          'validation', 'official_q3', 'failed_attempts')


def check(condition, message):
    if not condition:
        raise ValueError(message)


def guard_original_tree(original_root, allowed_roots):
    """Make a stale absolute source dependency fail even on the producer host."""
    original_root = Path(original_root).resolve()
    allowed_roots = [Path(path).resolve() for path in allowed_roots]
    state = {'enabled': True, 'blocked_original_tree_accesses': 0,
             'scope': 'Original scientific tree opens/listings are forbidden outside the expanded bundle, new replay output and portable tool directory.'}
    def audit(event, args):
        if event not in ('open', 'os.listdir', 'os.scandir') or not args:
            return
        path = args[0]
        if isinstance(path, int) or path is None:
            return
        if isinstance(path, bytes):
            path = os.fsdecode(path)
        try:
            path = Path(path).resolve()
        except (TypeError, OSError, ValueError):
            return
        if path.is_relative_to(original_root) and not any(path.is_relative_to(base) for base in allowed_roots):
            state['blocked_original_tree_accesses'] += 1
            raise ValueError('Stale original-tree dependency is forbidden during portable CPU replay: '+str(path.relative_to(original_root)))
    sys.addaudithook(audit)
    return state


def compact_chain(path, synthetic=False):
    """Bind all publicly retained actual observations without pretending to
    retokenize omitted inputs or rerun an engine on independent hardware.
    """
    path = Path(path)
    budget_family = (path/'BUDGET_ALL_WORK_COMPLETE.json').is_file()
    complete = (path/'ALL_WORK_COMPLETE.json').is_file() or budget_family
    expected_stops = None
    native_identity_checked = False
    if complete and not synthetic:
        runtime = load(path/'EFFECTIVE_NATIVE_RUNTIME.json')
        expected = {'qwen': ('Qwen3ForCausalLM', 'vllm.model_executor.models.qwen3', 4022468096),
                    'glm': ('ChatGLMForCausalLM', 'vllm.model_executor.models.chatglm', 9399951360)}[runtime['model']]
        workers = runtime['native_workers']
        check(len(workers) == 1, 'Unexpected native worker count')
        worker = workers[0]
        check((worker['model_class'], worker['module'], worker['parameters']) == expected, 'Actual native implementation/parameter count differs')
        check(worker['dtypes'] == ['torch.bfloat16'] and runtime['dtype'] == 'torch.bfloat16', 'Native BF16 differs')
        for key, value in {'max_model_len': 8192, 'max_num_seqs': 32, 'tensor_parallel_size': 1, 'enforce_eager': True}.items():
            check(runtime[key] == value, 'Native runtime envelope differs: '+key)
        start = load(path/'START.json')
        if budget_family:
            check(start['freeze_sha256'] == sha(path.parent/'BUDGET_FOLLOWUP_FREEZE.json'), 'Budget run freeze binding differs')
        check(runtime['prefix_cache'] == (True if budget_family else start['prefix_cache']), 'Actual prefix-cache mode differs')
        if runtime['model'] == 'qwen':
            check(worker['remote_model_modules'] == [], 'Unexpected remote Qwen model module')
        else:
            amendment = load(path.parent/'GLM_NATIVE_TOKENIZER_GUARD_AMENDMENT_V3.json')
            tokenizer = amendment['allowed_tokenizer_module']
            check(worker['remote_model_modules'] == [tokenizer], 'Unexpected GLM remote modeling/configuration module')
            check(worker['dynamic_module_source_sha256'] == {tokenizer: amendment['allowed_tokenizer_sha256']}, 'GLM tokenizer source differs')
            if budget_family:
                check(start['freeze_sha256'] == sha(path.parent/'BUDGET_FOLLOWUP_FREEZE.json'), 'Budget run does not bind its prospective follow-up freeze')
                check(sha(path.parent/'native_runtime_glm_v3.py') == amendment['runtime_sha256'], 'GLM native runtime differs from its amendment')
            else:
                check(start['technical_amendment_sha256'] == sha(path.parent/'GLM_NATIVE_TOKENIZER_GUARD_AMENDMENT_V3.json'), 'Actual GLM run does not bind its documented sidecar amendment')
                check(sha(path.parent/start['effective_runtime']) == amendment['runtime_sha256']
                      and sha(path.parent/start['effective_entrypoint']) == amendment['entrypoint_sha256'],
                      'Actual GLM sidecar code bytes differ from the public amendment')
        stops = runtime['generation_config_fields'].get('eos_token_id', [])
        stops = stops if isinstance(stops, list) else [stops]
        expected_stops = sorted(set(stops+[runtime['primary_eos']]))
        resets = list(lines(path/'cache_resets.jsonl'))
        boundaries = set()
        for reset in resets:
            key = (reset['case_id'], reset['policy'])
            check(key not in boundaries, 'Duplicate actual policy cache reset boundary')
            boundaries.add(key)
            check(reset['prefix_cache'] == runtime['prefix_cache'], 'Reset cache mode differs')
            observations = reset['native_reset_observations']
            if runtime['prefix_cache']:
                check(len(observations) == 1, 'No unique native reset receipt')
                item = observations[0]
                check(item['native_result'] is True and item['cached_hash_count_after'] == 0
                      and item['num_gpu_blocks']-item['num_free_blocks_after'] == 1
                      and item['pid'] == worker['pid'] and item['native_class'] == 'vllm.v1.core.block_pool.BlockPool',
                      'Actual native cache-empty boundary was not verified')
            else:
                check(observations == [], 'Cache-OFF boundary unexpectedly has a native reset')
        native_identity_checked = True
    response = {r['logical_id']: r for r in lines(path/'responses.jsonl')} if (path/'responses.jsonl').exists() else {}
    effective = {r['logical_id']: r for r in lines(path/'effective_requests.jsonl')} if (path/'effective_requests.jsonl').exists() else {}
    submitted, engines, actual, native_count = {}, set(), {}, 0
    for batch in lines(path/'submitted_calls.jsonl'):
        for job in batch['jobs']:
            check(job['logical_id'] not in submitted, 'Duplicate submitted logical ID')
            submitted[job['logical_id']] = (batch['stage'], job)
    for batch in lines(path/'raw_return_snapshots.jsonl'):
        check(batch['raw_saved_before_validation'] is True, 'Raw return persistence flag absent')
        check(len(batch['returns']) == len(batch['expected_ids']), 'Incomplete actual return batch')
        for logical, raw in zip(batch['expected_ids'], batch['returns']):
            check(logical in submitted and logical not in actual, 'Unknown/duplicate returned logical ID')
            stage, job = submitted[logical]
            check(raw['engine_request_id'] not in engines, 'Duplicate native engine ID')
            engines.add(raw['engine_request_id'])
            check(len(raw['outputs']) == 1, 'Ambiguous native completion count')
            output = raw['outputs'][0]
            check(raw['prompt_token_count'] == job['prompt_token_count'], 'Prompt token length mismatch')
            check(raw['prompt_token_ids_sha256'] == job['prompt_token_ids_sha256'], 'Prompt token SHA mismatch')
            if logical in response:
                r, e = response[logical], effective[logical]
                check(stage == batch['stage'] == r['stage'] == e['stage'], 'Stage binding mismatch')
                check(r['engine_request_id'] == e['engine_request_id'] == raw['engine_request_id'], 'Engine ID binding mismatch')
                check(r['raw'] == output['raw'] and r['finish_reason'] == output['finish_reason'] and r['stop_reason'] == output['stop_reason'], 'Raw completion mismatch')
                check(r['actual_input_tokens'] == raw['prompt_token_count'], 'Actual input token count mismatch')
                check(r['actual_output_tokens'] == output['output_token_count'], 'Actual output token count mismatch')
                check(r['prompt_token_ids_sha256'] == e['prompt_token_ids_sha256'] == raw['prompt_token_ids_sha256'], 'Three-way prompt hash binding mismatch')
                check(r['messages_sha256'] == job['messages_sha256'], 'Submitted messages hash mismatch')
                check(r['cached_input_tokens'] == raw['num_cached_tokens'], 'Observed cached-token receipt mismatch')
                controls = e['controls']
                for key, value in {'n': 1, 'temperature': 0, 'top_k': 0, 'top_p': 1, 'min_p': 0,
                                   'repetition_penalty': 1, 'presence_penalty': 0, 'frequency_penalty': 0,
                                   'ignore_eos': False, 'sampling_type': 'GREEDY'}.items():
                    check(controls[key] == value, 'Effective native control mismatch: ' + key)
                check(controls['max_tokens'] == job['max_output'], 'Actual output cap mismatch')
                if expected_stops is not None:
                    check(controls['all_stop_token_ids'] == expected_stops, 'Actual EOS stop-token controls differ')
                check(controls['guided'] == (job['grammar'] is not None), 'Guidance mismatch')
                if job['grammar'] is not None:
                    check(controls['grammar_sha256'] == digest(job['grammar']) and controls['guided_backend'] == 'xgrammar' and controls['guided_disable_fallback'] is True, 'Guidance contract mismatch')
            actual[logical] = raw
            native_count += 1
    if complete:
        check(set(response) == set(effective) == set(submitted) == set(actual), 'Completed run has missing or extra requests')
    else:
        check(set(response).issubset(actual), 'Canonical response lacks durable raw binding')
    return {'status': 'PASS', 'scope': 'Compact durable input-hash/length, output payload, native ID and effective-control consistency; no omitted-input token decoding or text reconstruction',
            'returned_native_records': native_count, 'canonical_responses': len(response), 'completed': complete,
            'actual_native_identity_and_cache_boundaries_checked': native_identity_checked}


def install_adapters(m, root, source_manifest):
    original_run = m.Run

    class PublicRun(original_run):
        def __init__(self, path, audit, deep=False):
            super().__init__(path, audit, deep=False)
            self.deep_result = compact_chain(path, synthetic=load(root/'COMPACT_SCOPE.json')['payload_kind'] == 'SYNTHETIC_CPU_FIXTURE_NOT_MODEL_EVIDENCE')

    def load_workload(workload, audit):
        records = list(lines(workload['records']))
        gold_rows = list(lines(workload['gold']))
        requests = list(lines(workload['native_requests']))
        check([r['record_id'] for r in records] == [r['record_id'] for r in gold_rows], 'Gold/occurrence order mismatch')
        check([r['record_id'] for r in requests] == [r['record_id'] for r in records[:len(requests)]], 'Request/occurrence order mismatch')
        gold = [r['source_class'] == workload['gold_positive_class'] if 'gold_positive_class' in workload
                else r.get(workload.get('gold_field', 'gold')) for r in gold_rows]
        for request in requests:
            # Reader metadata only, never a reconstructed or submitted prompt.
            request['messages'] = {'omitted_full_messages_sha256': request['messages_sha256']}
        return {'records': records, 'gold': gold, 'requests': requests}

    def native_file(path, expected_ids, expected_messages, run, stage_rows, audit, context):
        native = load(path)
        rows = native['rows']
        check([r['record_id'] for r in rows] == expected_ids, 'Actual native record order differs')
        labels = [m.native_label(r['raw_output'])[0] for r in rows]
        strict = [m.strict_label(r['raw_output']) for r in rows]
        fallback = sum(m.native_label(r['raw_output'])[1] for r in rows)
        hashes = native['compact_transport_messages_sha256']
        check(len(hashes) == len(rows) == len(stage_rows), 'Native invocation/receipt count mismatch')
        if expected_messages is not None:
            check(hashes == [item['omitted_full_messages_sha256'] for item in expected_messages],
                  'Actual native input hash differs from prepared occurrence/predicate input')
        for row, receipt, message_hash in zip(rows, stage_rows, hashes):
            check(row['raw_output'] == receipt['raw'] and message_hash == receipt['messages_sha256'], 'Native payload/message hash binding differs')
            p = m.strict_label(row['raw_output'])
            check(row['native_label'] == m.native_label(row['raw_output'])[0] and row['strict_label'] == p and row['strict_valid'] == (p is not None), 'Native parser result mismatch')
        check(native['native_count'] == sum(labels), 'Native count differs')
        check(native['strict_count'] == (sum(strict) if all(type(x) is bool for x in strict) else None), 'Strict count differs')
        return {'native': labels, 'strict': strict, 'fallback_count': fallback,
                'noncontract_native_labels': sum(x is None for x in strict), 'raw_rows': rows}

    def batch_file(path, requests, run, audit, context, *, n=None):
        calls = list(lines(path))
        labels, seen = [None]*(n if n is not None else len(requests)), Counter()
        for call in calls:
            group = call['indices']
            check(all(type(i) is int and 0 <= i < len(requests) for i in group), 'Physical partition/probe index range error')
            parsed = m.parse_batch(call['raw'], len(group))
            check(call['labels'] == parsed and call['strict_valid'] == (parsed is not None), 'Actual batch parse mismatch')
            response = run.by_id[call['logical_id']]
            for key in ('raw', 'messages_sha256', 'actual_input_tokens', 'actual_output_tokens', 'cached_input_tokens'):
                check(call[key] == response[key], 'Batch/raw receipt differs: ' + key)
            key = run.path.name+'|'+call['logical_id'] if source_manifest.get('binding_key_format') == 'run_directory|logical_id' else call['logical_id']
            binding = source_manifest['batch_message_bindings'][key]
            check(binding['indices'] == group and binding['expected_messages_sha256'] == call['messages_sha256'], 'Frozen group/message hash binding differs')
            if n is not None:
                for slot, index in enumerate(group):
                    seen[index] += 1
                    labels[index] = parsed[slot] if parsed is not None else None
        if n is not None:
            check(sorted(seen) == list(range(n)) and all(x == 1 for x in seen.values()), 'Incomplete or overlapping physical partition')
        return {'calls': calls, 'labels': labels, 'groups': [c['indices'] for c in calls],
                'invalid_batch_calls': sum(m.parse_batch(c['raw'], len(c['indices'])) is None for c in calls)}

    def official_q3(run, root, model, cache, audit, resolve_path=Path):
        data = load(run.path/'OFFICIAL_Q3_COMPACT_SOURCE.json')
        receipt = load(run.path/'OFFICIAL_Q3_RECEIPT.json')
        native_path = run.path/receipt.get('result_file', 'official_sembench_q3/native_result.json')
        parsed = native_file(native_path, data['record_ids'], None, run,
                             run.rows('OFFICIAL_SEMBENCH_Q3'), audit, 'official_q3')
        statistics = m.classification(parsed['native'], data['gold'])
        strict = m.classification(parsed['strict'], data['gold'])
        measured = m.usage(run.rows('OFFICIAL_SEMBENCH_Q3'))
        m.compare_usage(receipt.get('usage'), measured, audit, 'official_q3')
        return {'model': model, 'prefix_cache': cache, 'query': 'original_sembench_movie_q3_demo',
                'status': 'available', 'record_count': len(data['gold']), 'method_sha256': data['method_sha256'],
                'native_fallback_count': parsed['fallback_count'],
                **{'native_'+k: v for k, v in statistics.items()},
                **{'strict_'+k: v for k, v in strict.items()},
                **{'usage_'+k: v for k, v in measured.items()},
                'scope': 'unchanged original Q3 method on official repository demo; not regenerated full SemBench benchmark'}

    m.Run, m.load_workload = PublicRun, load_workload
    m.check_native_file, m.batch_file, m.official_q3 = native_file, batch_file, official_q3


def normalise_tables(summary):
    result = {name: summary[name] for name in TABLES}
    for row in result['failed_attempts']:
        row['directory'] = 'failed_attempts/' + Path(row['directory']).name
        row.pop('deep_integrity', None)  # Reader-scope evidence differs; actual failure/cost fields do not.
    return result


def replay(bundle, out):
    root, out = Path(bundle).resolve(), Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    scope = load(root/'COMPACT_SCOPE.json')
    check(scope['payload_kind'] in ('ACTUAL_COMPACT_EVIDENCE', 'SYNTHETIC_CPU_FIXTURE_NOT_MODEL_EVIDENCE'), 'Unknown payload type')
    access_guard = guard_original_tree(scope['original_root'], [root, out, Path(__file__).resolve().parent])
    for rel, expected in load(root/'FILE_SHA256.json')['files'].items():
        check(sha(root/rel) == expected, 'Bundle file mismatch: ' + rel)
    m = import_file('unchanged_independent_analyzer', root/'real_engine_analysis/analyze_study.py')
    install_adapters(m, root, load(root/'COMPACT_BINDINGS.json'))
    summary, integrity = m.analyze(root/'prepared/STUDY.json', root, out/'engine',
                                  freeze_file=root/'COMPACT_SELECTED_FILES.json',
                                  relocate_from=scope['original_root'])
    check(integrity['error_count'] == 0 and summary['analysis_status'] == 'complete', 'Compact mathematical analysis is incomplete or inconsistent')
    actual_tables = normalise_tables(summary)
    expected_tables = load(root/'EXPECTED_MATHEMATICS.json')
    check(actual_tables == expected_tables, 'Mathematical/CP/cost table differs from complete private-input analysis')
    sys.path.insert(0, str(root))
    common = import_file('experiment_common', root/'experiment_common.py')
    sys.modules['experiment_common'] = common
    common.ROOT, common.MODELS = root, load(root/'prepared/STUDY.json')['models']
    original_reader = common.read_lines
    old_root = Path(scope['original_root'])
    def mapped_reader(path):
        p = Path(path)
        if p.is_absolute():
            try:
                p = root/p.relative_to(old_root)
            except ValueError:
                pass
        return original_reader(p)
    common.read_lines = mapped_reader
    finite = import_file('unchanged_finite_analyzer', root/'analyze_finite_audits.py')
    finite.main(out/'finite')
    check(load(out/'finite/FIXED_BUDGET_AUDITS.json') == load(root/'EXPECTED_FINITE_AUDITS.json'), 'Finite-population audit differs')
    budget_jobs = 0
    if scope.get('with_budget_followup'):
        from replay_budget import analyze_budget
        budget_summary = analyze_budget(root, out/'budget_followup', compact_mode=True)
        check(budget_summary == load(root/'EXPECTED_BUDGET_AUDITS.json'), 'Separate budget-family mathematics differs')
        budget_jobs = len(budget_summary['jobs'])
        if (root/'EXPECTED_OFFICIAL_BUDGET_TABLES.json').exists():
            from replay_official_budget import run_official, compare_views
            official, tables = run_official(root, out/'separate_author_budget_analysis', compact_mode=True)
            check(tables == load(root/'EXPECTED_OFFICIAL_BUDGET_TABLES.json'), 'Separately authored budget tables differ from full-private-input results')
            compare_views(budget_summary, official, out/'BUDGET_ANALYZER_AGREEMENT.json')
    write_json(out/'CPU_REPLAY_REPORT.json', {'status': 'PASS', 'payload_kind': scope['payload_kind'],
        'configurations': summary['completed_configurations'], 'math_tables_equal': list(TABLES),
        'separate_budget_followup_policy_jobs': budget_jobs,
        'finite_audits_equal': True, 'original_mathematical_source_edited': False,
        'original_tree_access_guard': access_guard,
        'explicit_read_adapters': ['full-text workload reader', 'full-text native/batch message verification reader', 'official Q3 source reader', 'compact native binding reader'],
        'source_input_scope': 'Actual outputs and input identity/length receipts; omitted news text and prompt arrays require optional pinned reconstruction for full-source verification.',
        'source_token_redecoding': False, 'new_GPU_or_model_inference': False,
        'cleanroom_GPU_reproduction_claimed': False, 'original_root_mapping': scope['original_root']})
    print('PASS: compact actual-output mathematics, CP selection, cost and finite-population tables match.')


if __name__ == '__main__':
    os.environ.update(CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    replay(args.bundle, args.out)
