#!/usr/bin/env python3
"""Rebuild the paper's original DEV dependency, not the old TEST study.

Reconstruct 36 locked policies and all observed calibration token ledgers;
rebuild complete source relations, partitions, gold, DEV schedules and native
tokenizer prompts from redistributed sources and tokenizer assets. No weights.
"""
import argparse
import importlib
import json
import os
import sys
import time
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    bundle, out = a.bundle.resolve(), a.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    sys.dont_write_bytecode = True
    os.environ.update(CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1', USE_TORCH='0',
                      HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
    sys.path.insert(0, str(bundle))
    import common
    from lock_policies import derive_lock
    from check_cpu import run_checks
    payload = derive_lock(bundle)
    archived = json.loads((bundle / 'POLICY_LOCK.json').read_text())
    if payload != archived['payload']:
        raise ValueError('Complete 36-policy DEV payload differs from the immutable lock')
    common.write_json(out / 'RECOMPUTED_POLICY_PAYLOAD.json', payload)
    checks = run_checks()
    if checks.get('status') != 'PASS':
        raise ValueError('Decision CPU contract check failed')
    common.write_json(out / 'CPU_CONTRACT_CHECKS.json', checks)
    # Frozen prepare.py initializes local cache directories at import. Redirect
    # that initialization explicitly to this new output tree, then bind its
    # pure constructors to the relocated read-only input bundle.
    original_root = common.ROOT
    common.ROOT = out / 'prepare_import_workspace'
    try:
        prep = importlib.import_module('prepare')
    finally:
        common.ROOT = original_root
    prep.ROOT = bundle
    rows = prep.relations_from_reserves()
    partitions, index_audits, random_bases = prep.partitions(rows)
    gold = prep.get_gold(rows)
    schedules = prep.schedules(partitions)
    signatures = common.digest(common.compact({'relations': rows, 'partitions': partitions, 'schedules': schedules}))
    inputs, prepared = out / 'inputs', out / 'prepared'
    inputs.mkdir()
    prepared.mkdir()
    for name, obj, lines in [
        ('relations.jsonl', rows, True), ('partitions.jsonl', partitions, True),
        ('gold.jsonl', gold, True), ('LEXICAL_INDEX_AUDIT.json', index_audits, False),
        ('RANDOM_BASE_PERMUTATIONS.json', random_bases, False),
        ('FEATURE_ONLY_PLAN_HASH.json', {'sha256': signatures, 'gold_used_for_plan_construction': False}, False)
    ]:
        (common.write_lines if lines else common.write_json)(inputs / name, obj)
    common.write_lines(prepared / 'dev_schedule.jsonl', schedules['dev'])
    common.write_json(prepared / 'dev_block_membership.json',
                      [[j['request_id'] for j in schedules['dev'][s:s + 64]] for s in range(0, len(schedules['dev']), 64)])
    from transformers import AutoTokenizer
    lookup = {(r['task'], r['phase'], r['occurrence_id']): r for r in rows}
    for model in ('qwen', 'smol'):
        location = bundle / 'recovery_decision_notes/offline_tokenizers' / model
        tokenizer = AutoTokenizer.from_pretrained(location, local_files_only=True, trust_remote_code=False)
        requests = []
        for job in schedules['dev']:
            records = [lookup[job['task'], 'dev', i] for i in job['occurrence_ids']]
            prompt = common.prepare_prompt(tokenizer, job['task'], records, job['execution_type'])
            longest = common.compact({str(i): 'Y' for i in prompt['output_ids']})
            lengths = [len(tokenizer.encode(common.compact({str(i): v for i in prompt['output_ids']}), add_special_tokens=False)) for v in ('Y', 'N')]
            if prompt['prompt_tokens_preflight'] + 128 > 4096:
                raise ValueError('Prepared prompt exceeds the frozen context envelope')
            requests.append({**job, **prompt, 'canonical_output_token_lengths_allY_allN': lengths,
                             'max_ascii_output_characters': len(longest)})
        common.write_lines(prepared / f'{model}_dev_requests.jsonl', requests)
    compared = {}
    for directory in (inputs, prepared):
        for file in sorted(directory.iterdir()):
            rel = file.relative_to(out)
            if common.sha(file) != common.sha(bundle / rel):
                raise ValueError('Regenerated decision input differs: ' + str(rel))
            compared[str(rel)] = common.sha(file)
    dev_costs = {}
    for model in ('qwen', 'smol'):
        costs = payload['development_research_costs'][model]
        scientific = common.read_lines(bundle / f'run_dev_{model}/responses.jsonl')
        controls = common.read_lines(bundle / f'run_dev_{model}/diagnostics.jsonl')
        literal = {'scientific_calls': len(scientific), 'diagnostic_calls': len(controls)}
        for kind in ('input', 'output'):
            scientific_tokens = sum(r[f'actual_{kind}_tokens'] for r in scientific)
            control_tokens = sum(r[f'actual_{kind}_tokens'] for r in controls)
            if scientific_tokens != costs['scientific_tokens'][f'actual_{kind}_tokens'] or control_tokens != costs['diagnostic_tokens'][kind]:
                raise ValueError('Original calibration actual token accounting mismatch')
            literal[f'scientific_actual_{kind}_tokens'] = scientific_tokens
            literal[f'diagnostic_actual_{kind}_tokens'] = control_tokens
            literal[f'all_actual_{kind}_tokens'] = scientific_tokens + control_tokens
        dev_costs[model] = literal
    common.write_json(out / 'DEV_ACTUAL_TOKENS.json', dev_costs)
    common.write_json(out / 'DECISION_DEPENDENCY_REPLAY.json', {
        'status': 'PASS', 'replayed_unix': time.time(), 'gpu_used': False, 'new_inference': False,
        'policy_count': 36, 'complete_policy_payload_equal': True,
        'policy_payload_sha256': common.digest(common.compact(payload)),
        'CPU_contract_checks': checks['checks'], 'input_file_hashes_equal': compared,
        'DEV_actual_tokens': dev_costs,
        'coverage': 'Original DEV input closure, all 36 policy payloads, selector ledgers and actual calibration cost. Full source/partition/gold inputs are reconstructed, but no old TEST inference or TEST-result claim is replayed.',
        'historical_boundary': 'Preserved historical 73/242-file inventories remain incomplete. Missing old certificates are not recreated; the old freeze is not represented as fully recovered.'})
    print('PASS: original DEV, 36 locks, all selector ledgers, actual tokens and deterministic inputs match.')


if __name__ == '__main__':
    main()
