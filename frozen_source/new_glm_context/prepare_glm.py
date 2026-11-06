#!/usr/bin/env python3
"""CPU-only native-tokenizer preparation of an already fixed complete target set.

No sample/neighbor selection is performed, and no gold label is used to form a
request or decide retention. All original targets and schedules are retained.
"""
import collections
import itertools
import time
import argparse
from common import *
from glm_tokenizer import load_tokenizer, prepare
from synthetic_contracts import synthetic_records

ORIGINAL = Path('/home/gpu/IntentSat2S_revision_20261106/reconstruction/context_audit')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',choices=MODELS,default='glm')
    model=parser.parse_args().model
    if model=='glm':
        tokenizer=load_tokenizer()
        make_prompt=prepare
        model_path='/smoothcloud/models/glm-4-9b-chat'
    else:
        from transformers import AutoTokenizer
        model_path={'qwen':'/root/models/Qwen/Qwen3-4B-Instruct-2507','smol':'/home/gpu/intent2s_models/SmolLM2-1.7B-Instruct'}[model]
        tokenizer=AutoTokenizer.from_pretrained(model_path,trust_remote_code=False,local_files_only=True)
        make_prompt=prepared_prompt
    targets = read_lines(ROOT/'selection/targets.jsonl')
    bytarget = {(r['task'], r['target_id']): r for r in targets}
    schedule = read_lines(ROOT/'prepared/schedule.jsonl')
    serial_schedule = read_lines(ROOT/'prepared/serial_schedule.jsonl')
    original_jobs = {r['request_id']: r for r in read_lines(ORIGINAL/'prepared'/f'{model if model != "glm" else "qwen"}_requests.jsonl')}
    jobs = []
    comparison_keys = ('system', 'unwrapped_user', 'requested_ids', 'output_ids', 'grammar', 'grammar_sha256',
                       'target_input_line', 'id_map', 'id_map_sha256', 'fixed_output_prefix_before_target')
    cached = {}
    for job in schedule:
        r = bytarget[job['task'], job['target_id']]
        records = [r['target']] + (r['contexts'][job['context']] if job['scope'] != 'singleton' else [])
        key = job['task'], job['target_id'], job['scope'], job['context']
        if key not in cached:
            cached[key] = make_prompt(tokenizer, job['task'], records, job['scope'])
        p = cached[key]
        mapping = {str(i+1): {'source_id': rec['source_id'], 'source_split': rec['source_split'],
                             'feature_sha256': rec['feature_sha256']} for i, rec in enumerate(records)}
        row = {**job, **p, 'id_map': mapping, 'id_map_sha256': digest(compact(mapping))}
        original = original_jobs[job['request_id']]
        changed_random = r.get('glm_primary_random_context_redrawn', False) and job['context'] == 'random'
        for k in comparison_keys:
            if changed_random and k in ('unwrapped_user', 'id_map', 'id_map_sha256'):
                continue
            assert row[k] == original[k], (job['request_id'], k)
        jobs.append(row)
    write_lines(ROOT/'prepared'/f'{model}_requests.jsonl', jobs)
    byid = {r['request_id']: r for r in jobs}
    serial = [{**byid[j['source_main_request_id']], **j} for j in serial_schedule]
    for job in serial:
        source = byid[job['source_main_request_id']]
        for key in ('system', 'unwrapped_user', 'wrapped_prompt', 'prompt_token_ids', 'requested_ids', 'output_ids',
                    'grammar', 'prompt_sha256', 'grammar_sha256', 'id_map', 'id_map_sha256'):
            assert job[key] == source[key]
    write_lines(ROOT/'prepared'/f'{model}_serial_requests.jsonl', serial)
    lookup = {(j['task'], j['target_id'], j['scope'], j['context'], j['repeat']): j for j in jobs}
    length_rows = []
    feature_lengths = []
    for target in targets:
        k = target['task'], target['target_id']
        prefixes, tokenprefixes, positions, prefix_counts = [], [], [], []
        for context in CONTEXTS:
            first = lookup[(*k, 'full-first', context, 1)]
            last = lookup[(*k, 'full-last', context, 1)]
            assert first['wrapped_prompt'] == last['wrapped_prompt']
            assert first['prompt_token_ids'] == last['prompt_token_ids']
            for scope in SCOPES:
                p = lookup[(*k, scope, context, 1)]
                prefixes.append(p['target_input_prefix_sha256'])
                tokenprefixes.append(p['target_prefix_token_ids_sha256'])
                positions.append(p['target_input_span_bytes'])
                prefix_counts.append(p['target_prefix_token_count'])
        assert len(set(prefixes)) == len(set(tokenprefixes)) == len(set(prefix_counts)) == 1
        assert all(v == positions[0] for v in positions)
        for scope in SCOPES:
            near = lookup[(*k, scope, 'near', 1)]['prompt_tokens_preflight']
            random = lookup[(*k, scope, 'random', 1)]['prompt_tokens_preflight']
            length_rows.append({'task': target['task'], 'target_id': target['target_id'],
                                'selected_ordinal': target['selected_ordinal'], 'scope': scope,
                                'near_prompt_tokens': near, 'random_prompt_tokens': random,
                                'near_minus_random': near-random, 'absolute_difference': abs(near-random),
                                'original_24_token_matching_satisfied': abs(near-random) <= 24,
                                'all_original_targets_retained': True})
        for context, records in [('target', [target['target']]), *target['contexts'].items()]:
            for record in records:
                n = len(tokenizer.encode(feature_string(record), add_special_tokens=False, truncation=False))
                feature_lengths.append({'task': target['task'], 'target_id': target['target_id'],
                    'context': context, 'source_id': record['source_id'], 'feature_tokens_model': n,
                    'original_512_feature_bound_satisfied_in_this_tokenizer': n <= 512})
    write_lines(ROOT/'prepared'/f'{model.upper()}_LENGTH_AUDIT.jsonl', length_rows)
    write_lines(ROOT/'prepared'/f'{model.upper()}_SELECTED_FEATURE_LENGTHS.jsonl', feature_lengths)
    diagnostics = []
    messages = [{'role': 'system', 'content': 'Follow the user instruction exactly.'},
                {'role': 'user', 'content': 'Reply with only the word BANANA.'}]
    wrapped = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    banana = {'wrapped_prompt': wrapped, 'prompt_token_ids': tokenizer.encode(wrapped, add_special_tokens=False),
              'prompt_sha256': digest(wrapped)}
    diagnostics.append({**banana, 'request_id': 'diagnostic/ordinary_banana', 'output_ids': [], 'guided': False,
                        'diagnostic_group': 'ordinary_banana'})
    for i, ids in enumerate(([1], [1, 2, 3, 4], [2, 3, 4, 1], [9, 10, 11], [11])):
        diagnostics.append({**banana, 'request_id': f'diagnostic/guided_banana/{i}', 'output_ids': ids,
                            'grammar': grammar(ids), 'grammar_sha256': digest(grammar(ids)), 'guided': True,
                            'diagnostic_group': 'guided_banana'})
    for task in TASKS:
        for scope in (*SCOPES, 'singleton'):
            for context in (CONTEXTS if scope != 'singleton' else ('none',)):
                records = synthetic_records(task, 'near' if context == 'none' else context)
                if scope == 'singleton':
                    records = records[:1]
                p = make_prompt(tokenizer, task, records, scope)
                diagnostics.append({**p, 'request_id': f'diagnostic/smoke/{task}/{scope}/{context}',
                    'task': task, 'scope': scope, 'context': context, 'guided': True,
                    'diagnostic_group': 'synthetic_format_smoke'})
    assert len(diagnostics) == 20
    write_lines(ROOT/'prepared'/f'{model}_diagnostic_requests.jsonl', diagnostics)
    parser_rows = parser_checks()
    write_json(ROOT/'prepared/PARSER_CHECKS.json', {'status': 'PASS', 'checks': parser_rows})
    output_counts = []
    for ids in ([1], [1, 2, 3, 4], [2, 3, 4, 1], [9, 10, 11], [11]):
        for labels in itertools.product('YN', repeat=len(ids)):
            raw = compact({str(i): value for i, value in zip(ids, labels)})
            assert parse(raw, ids) is not None
            output_counts.append(len(tokenizer.encode(raw, add_special_tokens=False)))
    report = {'created_unix': time.time(), 'record_type': f'Current CPU-only {model} preparation; no model inference.',
        'model_snapshot_provenance': 'See separate current model verification; local tokenizer preparation does not certify official provenance.',
        'tokenizer_class': type(tokenizer).__name__, 'tokenizer_is_fast': tokenizer.is_fast,
        'tokenizer_source': model_path,'model':model,
        'local_tokenizer_only_trust_remote_code': model=='glm',
        'main_calls': len(jobs), 'serial_calls': len(serial), 'diagnostic_calls': len(diagnostics),
        'main_item_labels': sum(len(j['output_ids']) for j in jobs),
        'serial_item_labels': sum(len(j['output_ids']) for j in serial),
        'all_scientific_item_labels': sum(len(j['output_ids']) for j in jobs+serial),
        'selected_targets': dict(collections.Counter(r['task'] for r in targets)),
        'serial_targets': dict(collections.Counter(r['task'] for r in targets if r['selected_ordinal'] < 32)),
        'all_original_targets_and_near_contexts_retained': True, 'targets_resampled_or_replaced': 0,
        'redrawn_random_targets':sum(r.get('glm_primary_random_context_redrawn',False) for r in targets),
        'unchanged_random_targets':sum(not r.get('glm_primary_random_context_redrawn',False) for r in targets),
        'original_random_contexts_preserved_in_secondary':True,
        'unchanged_contexts_have_original_systems_unwrapped_prompts_requested_ids_output_ids_grammars_id_maps': 'PASS',
        'redrawn_random_contexts_keep_original_target_system_output_ids_grammar_and_complete_bank_features': 'PASS',
        'full_first_last_prompt_and_token_identity': 'PASS', 'fixed_target_byte_positions_and_token_prefixes': 'PASS',
        'serial_exact_main_prompt_grammar_token_identity': 'PASS',
        'max_main_prompt_tokens': max(j['prompt_tokens_preflight'] for j in jobs),
        'max_serial_prompt_tokens': max(j['prompt_tokens_preflight'] for j in serial),
        'max_main_prompt_plus_output_cap': max(j['prompt_tokens_preflight'] + 128 for j in jobs),
        'all_prompt_plus_128_within_4096': all(j['prompt_tokens_preflight']+128 <= 4096 for j in jobs),
        'main_input_tokens': sum(len(j['prompt_token_ids']) for j in jobs),
        'serial_input_tokens': sum(len(j['prompt_token_ids']) for j in serial),
        'diagnostic_input_tokens': sum(len(j['prompt_token_ids']) for j in diagnostics),
        'original_24_token_length_match_in_this_tokenizer': all(r['absolute_difference'] <= 24 for r in length_rows),
        'length_match_max_absolute_difference': max(r['absolute_difference'] for r in length_rows),
        'length_match_violations': [r for r in length_rows if r['absolute_difference'] > 24],
        'per_task_length_match': {task: {'comparisons': sum(r['task']==task for r in length_rows),
            'max_abs_difference': max(r['absolute_difference'] for r in length_rows if r['task']==task),
            'violations': sum(r['task']==task and r['absolute_difference']>24 for r in length_rows)} for task in TASKS},
        'max_selected_complete_feature_tokens_model': max(r['feature_tokens_model'] for r in feature_lengths),
        'feature_over_512_occurrences': sum(r['feature_tokens_model'] > 512 for r in feature_lengths),
        'max_valid_json_output_tokens': max(output_counts), 'output_cap': 128, 'max_model_len': 4096,
        'request_hashes': {name: sha(ROOT/'prepared'/name) for name in
            (f'{model}_requests.jsonl', f'{model}_serial_requests.jsonl', f'{model}_diagnostic_requests.jsonl', 'schedule.jsonl', 'serial_schedule.jsonl')}}
    write_json(ROOT/'prepared'/f'{model.upper()}_CONTEXT_PREFLIGHT.json', report)
    print(compact({k: v for k, v in report.items() if k != 'length_match_violations'}), flush=True)


if __name__ == '__main__':
    main()
