#!/usr/bin/env python3
"""Forward, finite, label-free three-tokenizer rematching of 34 random contexts."""
import copy
import importlib.util
import random
import time
import numpy as np
from common import *
from glm_tokenizer import load_tokenizer


def lengths(task, target, background, tokenizers):
    result = {}
    for model, tokenizer in tokenizers.items():
        result[model] = {}
        for scope in SCOPES:
            system, user, _ = prompt(task, [target]+background, scope)
            wrapped = tokenizer.apply_chat_template([{'role': 'system', 'content': system},
                {'role': 'user', 'content': user}], tokenize=False, add_generation_prompt=True)
            result[model][scope] = len(tokenizer.encode(wrapped, add_special_tokens=False, truncation=False))
    return result


def rematch(feature_targets, banks, tokenizers, original_failures, proposal_path):
    rng = random.Random(2026110603)
    changes = []
    chosen = {}
    with proposal_path.open('x', encoding='utf8') as sink:
        for target in feature_targets:
            key = target['task'], target['target_id']
            if key not in original_failures:
                chosen[key] = target['contexts']['random']
                continue
            bank = banks[target['task']]
            near = target['contexts']['near']
            near_identities = set().union(*(identities(r) for r in near))
            bank_ids = [identities(r) for r in bank]
            near_lengths = lengths(target['task'], target['target'], near, tokenizers)
            bank_lengths = {m: np.array([r['token_lengths'][m] for r in bank]) for m in tokenizers}
            near_feature_lengths = {m: [len(tok.encode(feature_string(r), add_special_tokens=False, truncation=False))
                                       for r in near] for m, tok in tokenizers.items()}
            attempts = {}
            accepted = None
            accepted_lengths = None
            accepted_width = None
            for width in (16, 32, 64):
                attempts[str(width)] = 0
                pools = []
                for position in range(3):
                    mask = np.ones(len(bank), dtype=bool)
                    for model in tokenizers:
                        mask &= np.abs(bank_lengths[model]-near_feature_lengths[model][position]) <= width
                        mask &= bank_lengths[model] <= 512
                    pools.append([i for i in np.flatnonzero(mask).tolist() if not bank_ids[i] & near_identities])
                for attempt in range(1, 2001):
                    attempts[str(width)] += 1
                    selected, used = [], set()
                    for pool in pools:
                        remaining = [i for i in pool if not bank_ids[i] & used]
                        if not remaining:
                            break
                        i = rng.choice(remaining)
                        selected.append(i)
                        used.update(bank_ids[i])
                    row = {'task': target['task'], 'target_id': target['target_id'],
                           'selected_ordinal': target['selected_ordinal'], 'width': width,
                           'attempt_at_width': attempt, 'pool_sizes': list(map(len, pools)),
                           'proposal_ids': [bank[i]['source_id'] for i in selected]}
                    if len(selected) != 3:
                        row['status'] = 'empty_uniform_draw_pool'
                        sink.write(compact(row)+'\n')
                        break
                    records = [bank[i] for i in selected]
                    random_lengths = lengths(target['task'], target['target'], records, tokenizers)
                    matched = all(abs(near_lengths[m][s]-random_lengths[m][s]) <= 24 for m in tokenizers for s in SCOPES)
                    capped = all(random_lengths[m][s]+128 <= 4096 and near_lengths[m][s]+128 <= 4096
                                 for m in tokenizers for s in SCOPES)
                    row.update(near_prompt_lengths=near_lengths, random_prompt_lengths=random_lengths,
                               near_minus_random={m: {s: near_lengths[m][s]-random_lengths[m][s] for s in SCOPES} for m in tokenizers},
                               status='accepted' if matched and capped else 'length_mismatch' if not matched else 'context_overflow')
                    sink.write(compact(row)+'\n')
                    if matched and capped:
                        accepted, accepted_lengths, accepted_width = records, random_lengths, width
                        break
                if accepted is not None:
                    break
            sink.flush()
            result = {'task': target['task'], 'target_id': target['target_id'], 'selected_ordinal': target['selected_ordinal'],
                      'original_near_ids': [r['source_id'] for r in near],
                      'original_random_ids': [r['source_id'] for r in target['contexts']['random']],
                      'accepted': accepted is not None, 'attempts_by_width': attempts, 'accepted_width': accepted_width,
                      'new_random_ids': [r['source_id'] for r in accepted] if accepted is not None else None,
                      'near_prompt_lengths': near_lengths, 'new_random_prompt_lengths': accepted_lengths}
            changes.append(result)
            chosen[key] = accepted
            print(target['task'], target['selected_ordinal'], 'accepted', accepted is not None, 'attempts', sum(attempts.values()), flush=True)
    return chosen, changes


def main():
    from transformers import AutoTokenizer
    paths = {'qwen': Path('/root/models/Qwen/Qwen3-4B-Instruct-2507'),
             'smol': Path('/home/gpu/intent2s_models/SmolLM2-1.7B-Instruct')}
    tokenizers = {m: AutoTokenizer.from_pretrained(str(p), trust_remote_code=False, local_files_only=True) for m, p in paths.items()}
    tokenizers['glm'] = load_tokenizer()
    historical = read_lines(ROOT/'historical_context_arm/selection/targets.jsonl')
    preflight = json.loads((ROOT/'historical_context_arm/prepared/GLM_CONTEXT_PREFLIGHT.json').read_text())
    failures = {(r['task'], r['target_id']) for r in preflight['length_match_violations']}
    assert len(failures) == 34
    feature_targets = [{k: copy.deepcopy(r[k]) for k in ('task', 'target_id', 'selected_ordinal', 'target', 'contexts')} for r in historical]
    # No gold fields are passed into the selection function or its proposal pools.
    assert all(not {'gold', 'background_gold'} & set(r) and
               not {'gold', 'background_gold'} & set(record)
               for r in feature_targets
               for record in [r['target'], *r['contexts']['near'], *r['contexts']['random']])
    banks = {task: read_lines(ROOT/'selection'/f'{task}_background_bank.jsonl') for task in TASKS}
    bank_length_audit = []
    for task, bank in banks.items():
        for start in range(0, len(bank), 128):
            subset = bank[start:start+128]
            enc = tokenizers['glm']([feature_string(r) for r in subset], add_special_tokens=False, truncation=False)['input_ids']
            for r, ids in zip(subset, enc):
                r['token_lengths'] = {**r['token_lengths'], 'glm': len(ids)}
                bank_length_audit.append({'task': task, 'source_id': r['source_id'], 'feature_sha256': r['feature_sha256'],
                                         'token_lengths': r['token_lengths'], 'glm_over_512': len(ids)>512})
    write_lines(ROOT/'prepared/THREE_TOKENIZER_BANK_LENGTHS.jsonl', bank_length_audit)
    input_signature = digest(compact({'targets': feature_targets, 'banks': banks}))
    chosen, changes = rematch(feature_targets, banks, tokenizers, failures, ROOT/'prepared/REMATCH_PROPOSALS.jsonl')
    signature = {f'{task}/{target}': None if records is None else [r['source_id'] for r in records]
                 for (task, target), records in chosen.items()}
    write_json(ROOT/'prepared/FEATURE_ONLY_REMATCH_SIGNATURE.json', {'input_sha256': input_signature,
        'chosen_random_ids': signature, 'seed': 2026110603, 'selection_function_receives_gold_fields': False})
    # Actual selection replay with all source labels inverted. Gold never enters
    # selection inputs: the independently re-derived feature arguments are equal.
    inverted = copy.deepcopy(historical)
    for r in inverted:
        r['gold'] = 1-r['gold']
        r['background_gold'] = {c: [1-v for v in values] for c, values in r['background_gold'].items()}
    changed_features = [{k: copy.deepcopy(r[k]) for k in ('task', 'target_id', 'selected_ordinal', 'target', 'contexts')} for r in inverted]
    assert changed_features == feature_targets
    replay, replay_changes = rematch(changed_features, banks, tokenizers, failures, ROOT/'verification/REMATCH_GOLD_INVERSION_REPLAY_PROPOSALS.jsonl')
    assert {key: None if rows is None else [r['source_id'] for r in rows] for key, rows in replay.items()} == {
        key: None if rows is None else [r['source_id'] for r in rows] for key, rows in chosen.items()}
    assert replay_changes == changes
    write_json(ROOT/'verification/REMATCH_GOLD_INVERSION_CHECK.json', {'status': 'PASS',
        'actual_selection_function_reexecuted': True, 'all_selected_target_and_background_gold_inverted': True,
        'feature_selection_inputs_identical': True, 'chosen_ids_and_full_attempt_logs_identical': True,
        'proposal_sha256': sha(ROOT/'prepared/REMATCH_PROPOSALS.jsonl'),
        'replay_proposal_sha256': sha(ROOT/'verification/REMATCH_GOLD_INVERSION_REPLAY_PROPOSALS.jsonl')})
    report = {'created_unix': time.time(), 'seed': 2026110603, 'affected_targets': len(failures),
              'accepted_targets': sum(r['accepted'] for r in changes), 'failed_targets': [r for r in changes if not r['accepted']],
              'unchanged_random_targets': len(historical)-len(failures), 'all_target_and_near_features_unchanged': True,
              'proposal_records': sum(1 for _ in (ROOT/'prepared/REMATCH_PROPOSALS.jsonl').open()),
              'bank_glm_over_512_excluded_from_new_draws': sum(r['glm_over_512'] for r in bank_length_audit),
              'changes': changes}
    write_json(ROOT/'prepared/REMATCH_REPORT.json', report)
    if report['failed_targets']:
        raise SystemExit('Finite proposal rule did not produce all contexts. Retain failure report; do not submit GPU inference.')
    # Attach evaluation labels only after the feature-only chosen-ID signature
    # and its inversion replay have been saved and compared.
    spec = importlib.util.spec_from_file_location('source_prepare_contracts', ROOT/'source_reference/prepare.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    gold = module.load_gold(ROOT/'sources')
    primary = copy.deepcopy(historical)
    for r in primary:
        key = r['task'], r['target_id']
        if key in failures:
            r['contexts']['random'] = chosen[key]
            r['background_gold']['random'] = [gold[r['task'], b['source_id']] for b in chosen[key]]
            r['length_audit_historical'] = r.pop('length_audit')
            change = next(c for c in changes if (c['task'], c['target_id']) == key)
            r['length_audit'] = {'new_three_tokenizer_matching': change, 'seed': 2026110603}
        r['glm_primary_random_context_redrawn'] = key in failures
    path = ROOT/'selection/targets.jsonl'
    out = ROOT/'selection/primary_targets.new.jsonl'
    write_lines(out, primary)
    out.replace(path)
    # Preserve all initial compiled files; the following native preparation
    # writes a fresh primary arm rather than replacing this CPU observation.
    previous = ROOT/'initial_cpu_preflight'
    previous.mkdir()
    for p in list((ROOT/'prepared').glob('glm_*.jsonl')) + list((ROOT/'prepared').glob('GLM_*.json*')):
        p.rename(previous/p.name)
    print(compact({k: v for k, v in report.items() if k != 'changes'}), flush=True)


if __name__ == '__main__':
    main()
