#!/usr/bin/env python3
"""Create explicitly synthetic CPU transport fixtures; never model evidence."""
import argparse
import csv
import json
from pathlib import Path
import shutil
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE/'public'))
from portable_common import compact, digest, import_file, load, sha, write_json, write_lines


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    out = a.out.resolve()
    if not out.is_relative_to(HERE):
        raise ValueError('Fixture writes must stay inside portable_artifact')
    out.mkdir(parents=True, exist_ok=False)
    for rel in ('experiment_common.py', 'PROTOCOL.txt', 'analyze_finite_audits.py',
                'novelty_theory/finite_audit.py', 'real_engine_analysis/analyze_study.py'):
        q = out/rel;q.parent.mkdir(parents=True, exist_ok=True);shutil.copyfile(ROOT/rel, q)
    for name in ('analyze_budget_followup.py','analysis_common.py'):
        rel='real_engine_analysis/budget_followup/'+name
        q=out/rel;q.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/rel,q)
    common = import_file('fixture_constructor', out/'experiment_common.py')
    common.ROOT = out
    from scipy.stats import beta
    m = import_file('fixture_parser', out/'real_engine_analysis/analyze_study.py')
    q3_rel = next(k for k in load(ROOT/'PREINFERENCE_FREEZE.json')['files'] if k.endswith('/lotus_runner.py'))
    (out/q3_rel).parent.mkdir(parents=True);shutil.copyfile(ROOT/q3_rel, out/q3_rel)
    q3_csv = out/'fixture_original_q3.csv'
    with q3_csv.open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=('id', 'reviewId', 'scoreSentiment'));writer.writeheader()
        for i in range(120):writer.writerow({'id': 'taken_3', 'reviewId': str(i), 'scoreSentiment': 'POSITIVE' if i%2 else 'NEGATIVE'})
    records = [{'record_id': f'CPU_FIXTURE_{i}', 'fields': {'reviewText': f'Synthetic fixture record {i}; not a real review.'}} for i in range(16)]
    requests = [{'record_id': r['record_id'], 'messages': [{'role': 'system', 'content': 'CPU fixture'},
                 {'role': 'user', 'content': r['fields']['reviewText']}]} for r in records]
    for request in requests:
        request['messages_sha256'] = digest(json.dumps(request['messages'], ensure_ascii=False, sort_keys=True, separators=(',', ':')))
    for name, values in [('records', records), ('gold', [{'record_id': r['record_id'], 'gold': bool(i%2)} for i, r in enumerate(records)]), ('native_requests', requests)]:
        write_lines(out/'prepared'/('fixture.'+name+'.jsonl'), values)
    workload = {'workload_id': 'CPU_FIXTURE', 'domain': 'movie', 'corpus_id': 'SYNTHETIC_NOT_MODEL_EVIDENCE',
                'predicate': 'Synthetic fixture {reviewText}', 'gold_field': 'gold'}
    for field in ('records', 'gold', 'native_requests'):
        path = out/'prepared'/('fixture.'+field+'.jsonl');workload[field] = str(path);workload[field+'_sha256'] = sha(path)
    cases = [{'case_id': f'CPU_FIXTURE__n16__cache{int(cache)}', 'partition_key': 'CPU_FIXTURE__n16',
              'workload_id': 'CPU_FIXTURE', 'n': 16, 'prefix_cache': cache, 'primary_matrix': bool(cache)} for cache in (True, False)]
    study = {'config': common.CONFIG, 'models': {'qwen': 'SYNTHETIC_NOT_A_MODEL', 'glm': 'SYNTHETIC_NOT_A_MODEL'}, 'workloads': [workload], 'cases': cases}
    write_json(out/'prepared/STUDY.json', study)
    frozen = {str(p.relative_to(out)): sha(p) for p in out.rglob('*') if p.is_file()}
    write_json(out/'PREINFERENCE_FREEZE.json', {'scope': 'Synthetic CPU fixture only', 'files': frozen})
    write_json(out/'SYNTHETIC_CPU_FIXTURE.json', {'fixture': True, 'scientific_model_evidence': False, 'new_GPU_or_model_execution': False})
    for model in study['models']:
        for case in cases:
            cache = case['prefix_cache'];run = out/f'run_{model}_cache{int(cache)}';run.mkdir()
            response, effective, submissions, raw = [], [], [], []

            def generate(messages, answers, stage, grammar=None):
                jobs, returned, result = [], [], []
                for message, answer in zip(messages, answers):
                    logical = f'{model}:{int(cache)}:{len(response)}';engine = 'CPU_FIXTURE_ENGINE_'+logical
                    pids = [1, 2, int(digest(compact(message))[:4], 16)];oids = [5, 6]
                    message_hash, prompt_hash = digest(compact(message)), digest(compact(pids))
                    job = {'logical_id': logical, 'messages_sha256': message_hash, 'prompt_token_ids_sha256': prompt_hash,
                           'prompt_token_ids': pids, 'grammar': grammar, 'max_output': 128 if grammar else 32}
                    jobs.append(job)
                    cached = None if len(response)%13 == 0 else 1 if cache else 0
                    value = {'logical_id': logical, 'stage': stage, 'engine_request_id': engine, 'raw': answer,
                        'messages_sha256': message_hash, 'prompt_token_ids_sha256': prompt_hash,
                        'actual_input_tokens': len(pids), 'actual_output_tokens': len(oids),
                        'cached_input_tokens': cached, 'observed_uncached_input_tokens': None if cached is None else len(pids)-cached,
                        'finish_reason': 'stop', 'stop_reason': None}
                    controls = {'n': 1, 'temperature': 0, 'top_k': 0, 'top_p': 1, 'min_p': 0, 'max_tokens': job['max_output'],
                        'ignore_eos': False, 'repetition_penalty': 1, 'presence_penalty': 0, 'frequency_penalty': 0,
                        'guided': grammar is not None, 'guided_backend': 'xgrammar' if grammar else None,
                        'guided_disable_fallback': True if grammar else None, 'grammar_sha256': digest(grammar) if grammar else None,
                        'all_stop_token_ids': [2], 'sampling_type': 'GREEDY'}
                    effective.append({'logical_id': logical, 'engine_request_id': engine, 'stage': stage,
                                      'prompt_token_ids_sha256': prompt_hash, 'controls': controls, 'observed_unix': 1})
                    returned.append({'position': len(returned), 'engine_request_id': engine, 'prompt_token_ids': pids,
                                     'num_cached_tokens': cached, 'outputs': [{'raw': answer, 'output_token_ids': oids, 'finish_reason': 'stop', 'stop_reason': None}]})
                    response.append(value);result.append(value)
                submissions.append({'stage': stage, 'jobs': jobs, 'time_unix': 1})
                raw.append({'stage': stage, 'expected_ids': [j['logical_id'] for j in jobs], 'returns': returned,
                            'seconds': .01, 'time_unix': 1, 'raw_saved_before_validation': True})
                return result

            def native(indices, directory, stage, *, q3=False, noncontract_first=False):
                directory.mkdir(parents=True)
                ids = [f'sembench-demo-q3:row-{i}:review-{i}' for i in indices] if q3 else [records[i]['record_id'] for i in indices]
                messages = [[{'role': 'system', 'content': 'CPU fixture'}, {'role': 'user', 'content': 'Q3 fixture '+str(i)}] for i in indices] if q3 else [requests[i]['messages'] for i in indices]
                answers = [('True because of fixture explanation' if j == 3 and stage.endswith('/S2') else 'Answer: True' if i%2 else 'Answer: False') for j, i in enumerate(indices)]
                if noncontract_first:
                    answers[0] = ('True' if indices[0]%2 else 'False')+' with a synthetic noncontract explanation'
                generated = generate(messages, answers, stage)
                rows = [{'record_id': rid, 'raw_output': answer, 'native_label': m.native_label(answer)[0],
                         'strict_label': m.strict_label(answer), 'strict_valid': m.strict_label(answer) is not None} for rid, answer in zip(ids, answers)]
                strict = [r['strict_label'] for r in rows]
                obj = {'rows': rows, 'native_count': sum(r['native_label'] for r in rows),
                       'strict_count': sum(strict) if all(type(x) is bool for x in strict) else None,
                       'strict_valid_rows': sum(x is not None for x in strict),
                       'transport_invocations': [{'messages': messages}]}
                if q3:obj.update(data=str(q3_csv), method_source=str(out/q3_rel), data_sha256=sha(q3_csv), method_sha256='b62ce8e4ab8393f1a33be9f3f52708e69d3387a372c6efbe8e091fd96b0858f7')
                write_json(directory/'native_result.json', obj)
                return obj

            def batch(groups, directory, stage, *, probe_slots=None, fail_first=False):
                directory.mkdir(parents=True)
                messages = [common.batch_messages([requests[i]['messages'] for i in group]) for group in groups]
                labels = [[bool(i%2) for i in group] for group in groups]
                if fail_first:
                    labels[0][probe_slots[0]] = not labels[0][probe_slots[0]]
                answers = [compact({str(i+1): 'True' if v else 'False' for i, v in enumerate(row)}) for row in labels]
                generated = generate(messages, answers, stage, common.batch_grammar(len(groups[0])))
                calls = [{**value, 'group_ordinal': j, 'indices': group, 'strict_valid': True, 'labels': label}
                         for j, (value, group, label) in enumerate(zip(generated, groups, labels))]
                write_lines(directory/'batch_calls.jsonl', calls)
                if probe_slots is not None:
                    return [{'probe_index': group[pos], 'probe_slot': pos, 'label': label[pos], 'strict_valid': True, 'logical_id': call['logical_id']}
                            for group, pos, label, call in zip(groups, probe_slots, labels, calls)]
                vector = [None]*16
                for group, values in zip(groups, labels):
                    for i, value in zip(group, values):vector[i] = value
                return {'labels': vector, 'strict_complete': True, 'strict_count': sum(vector), 'groups': groups}

            def usage(start):
                rows = response[start:];v = m.usage(rows)
                return {k: v[k] for k in ('requests', 'input_tokens', 'output_tokens', 'cached_tokens_unknown_requests', 'cached_tokens_observed_lower_bound', 'cached_tokens', 'uncached_tokens')}

            write_json(run/'START.json', {'model': model, 'prefix_cache': cache, 'prelabel_freeze_sha256': sha(out/'PREINFERENCE_FREEZE.json'), 'started_unix': 0})
            q3_start = len(response);native(list(range(120)), run/'official_sembench_q3', 'OFFICIAL_SEMBENCH_Q3', q3=True)
            write_json(run/'OFFICIAL_Q3_RECEIPT.json', {'usage': usage(q3_start), 'result_file': 'official_sembench_q3/native_result.json'})
            case_root = run/case['case_id'];case_root.mkdir();results = {}
            for policy in m.POLICIES:
                stage = case['case_id']+'/'+policy;start = len(response);directory = case_root/policy
                if policy in ('S1', 'S2'):
                    value = native(list(range(16)), directory, stage);result = {'labels': [r['native_label'] for r in value['rows']], 'strict_labels': [r['strict_label'] for r in value['rows']], 'strict_count': value['strict_count'], 'native_count': value['native_count'], 'strict_complete': value['strict_count'] is not None}
                elif policy in ('RANDOM8', 'SIMILAR8'):
                    groups = common.random_partition(16, 8, case['partition_key']) if policy == 'RANDOM8' else [list(range(8)), list(range(8, 16))]
                    result = batch(groups, directory, stage)
                else:
                    directory.mkdir();indices = common.probes(16, case['partition_key'], 'calibration_probes', 80)
                    reference = native(indices, directory/'calibration_singletons', stage)['rows'];decisions = []
                    selected = 4
                    for k in (8, 4):
                        groups, slots = common.probe_batches(16, k, case['partition_key'], 'calibration_batches', indices)
                        take = 16 if k == 8 else 80;checked = []
                        for offset in range(0, take, 16):
                            rows = batch(groups[offset:offset+16], directory/f'candidate_{k}_chunk_{offset}', stage,
                                         probe_slots=slots[offset:offset+16], fail_first=k==8 and offset==0)
                            for j, row in enumerate(rows, offset):
                                row.update(reference_strict_label=reference[j]['strict_label'], agreement=row['label']==reference[j]['strict_label'])
                            checked.extend(rows)
                        x = sum(r['agreement'] for r in checked)
                        lower = float(beta.ppf(.05/3, x, 80-x+1)) if x else 0.0
                        decisions.append({'k': k, 'probe_budget': 80, 'actually_checked': take, 'agreements': x,
                                          'lower': lower, 'passed': k==4, 'probes': checked})
                    cal_usage = usage(start);execution_start = len(response)
                    result = batch(common.random_partition(16, selected, case['partition_key']), directory/'execution_partition', stage+'/execution')
                    result.update(selected_k=selected, calibration=decisions, calibration_usage=cal_usage,
                                  calibration_seconds=.5, execution_usage=usage(execution_start))
                result.update(policy=policy, usage=usage(start), elapsed_seconds=1.0)
                results[policy] = result;write_json(case_root/(policy+'_RESULT.json'), result)
            stage = case['case_id']+'/INDEPENDENT_VALIDATION';start = len(response)
            indices = common.probes(16, case['partition_key'], 'independent_validation_probes', 80)
            groups, slots = common.probe_batches(16, 4, case['partition_key'], 'independent_validation_batches', indices)
            reference = native(indices, case_root/'validation_singletons', stage)
            batched = batch(groups, case_root/'validation_batches', stage, probe_slots=slots)
            write_json(case_root/'INDEPENDENT_VALIDATION.json', {'status': 'complete', 'selected_k': 4, 'indices': indices,
                'fresh_singletons': reference['rows'], 'batched_probes': batched, 'usage': usage(start), 'not_used_for_selection': True})
            write_json(case_root/'CASE_COMPLETE.json', {'case': case, 'policies': list(m.POLICIES),
                'results_sha256': {p: sha(case_root/(p+'_RESULT.json')) for p in m.POLICIES},
                'validation_sha256': sha(case_root/'INDEPENDENT_VALIDATION.json')})
            for name, values in [('responses', response), ('effective_requests', effective), ('submissions', submissions), ('raw_returns', raw)]:write_lines(run/(name+'.jsonl'), values)
            write_json(run/'ALL_WORK_COMPLETE.json', {'status': 'complete', 'model': model, 'prefix_cache': cache,
                'cases_complete': 1, 'all_usage': usage(0), 'all_returned_requests': len(response),
                'responses_sha256': sha(run/'responses.jsonl'), 'raw_returns_sha256': sha(run/'raw_returns.jsonl'),
                'effective_requests_sha256': sha(run/'effective_requests.jsonl')})
    jobs = [{'job_id': f'CPU_FIXTURE__budget{b}__rep{r}', 'workload_id': 'CPU_FIXTURE', 'n': 16,
             'budget': b, 'repeat': r, 'sampling_key': f'CPU_FIXTURE_BUDGET_REP{r}'} for b in (80, 320) for r in (1,2)]
    budget_config = {'models': list(study['models']), 'selected_workloads': [workload],
                     'jobs_per_model': jobs, 'model_policy_executions': len(jobs)*len(study['models']),
                     'candidates': [8,4,2], 'eta': .05, 'delta': .05,
                     'n':16,'budgets':[80,320],'repeats':[1,2],'probe_chunk':16,
                     'status': 'SYNTHETIC_CPU_FIXTURE_NOT_MODEL_EVIDENCE'}
    write_json(out/'BUDGET_FOLLOWUP_STUDY.json', budget_config)
    write_json(out/'BUDGET_FOLLOWUP_FREEZE.json', {'scope': 'synthetic followup fixture only',
               'files': {'BUDGET_FOLLOWUP_STUDY.json': sha(out/'BUDGET_FOLLOWUP_STUDY.json')}})
    for model in study['models']:
        cache = True;response, effective, submissions, raw = [], [], [], []
        run = out/f'budget_run_{model}';run.mkdir()
        write_json(run/'START.json', {'model': model, 'freeze_sha256': sha(out/'BUDGET_FOLLOWUP_FREEZE.json'), 'fixture': True})
        generate([[{'role':'user','content':'CPU fixture control'}]], ['BANANA'], 'controls')
        generate([[{'role':'user','content':'CPU fixture control'}]], ['{"1":"True"}'], 'controls', common.batch_grammar(1))
        resets=[]
        for job in jobs:
            budget, key = job['budget'], job['sampling_key'];directory = run/job['job_id'];directory.mkdir()
            start = len(response);stage = job['job_id']+'/calibration'
            resets.append({'case_id':job['job_id'],'policy':'NATIVE_CP','prefix_cache':True,
                           'native_reset_observations':[{'native_result':True,'cached_hash_count_after':0,
                             'num_gpu_blocks':100,'num_free_blocks_after':99,'pid':999,
                             'native_class':'vllm.v1.core.block_pool.BlockPool'}]})
            indices = common.probes(16, key, 'native_budget_probes', budget)
            reference = native(indices, directory/'calibration_singletons', stage, noncontract_first=True)
            groups, slots = common.probe_batches(16, 8, key, 'native_budget_batches', indices)
            checked = []
            for offset in range(0, budget, 16):
                values = batch(groups[offset:offset+16], directory/f'candidate_8_chunk_{offset}', stage, probe_slots=slots[offset:offset+16])
                for j, value in enumerate(values, offset):
                    ref = reference['rows'][j]
                    value.update(reference_native_label=ref['native_label'], reference_strict_label=ref['strict_label'],
                                 reference_strict_valid=ref['strict_valid'], agreement=value['label']==ref['native_label'])
                checked.extend(values)
            lower = float(beta.ppf(.05/3, budget, 1))
            decision = {'k':8,'budget':budget,'actually_checked':budget,'agreements':budget,
                        'lower':lower,'passed':True,'probes':checked}
            calibration_usage = usage(start);execution_start = len(response)
            result = batch(common.random_partition(16, 8, key), directory/'execution_partition', job['job_id']+'/execution')
            result.update(job=job, native_complete=True, native_count=result['strict_count'], selected_k=8,
                          calibration=[decision], calibration_usage=calibration_usage,
                          execution_usage=usage(execution_start), usage=usage(start), elapsed_seconds=1.0,
                          calibration_seconds=.5, native_ref_format_valid_rows=reference['strict_valid_rows'], native_ref_rows=budget)
            write_json(directory/'POLICY_RESULT.json', result)
            write_json(directory/'POLICY_COMPLETE.json', {'job':job,'result_sha256':sha(directory/'POLICY_RESULT.json')})
        for name, values in [('responses',response),('effective_requests',effective),('submissions',submissions),('raw_returns',raw)]:
            write_lines(run/(name+'.jsonl'),values)
        write_lines(run/'cache_resets.jsonl',resets)
        write_json(run/'BUDGET_ALL_WORK_COMPLETE.json', {'status':'complete','model':model,'jobs_complete':len(jobs),
                   'all_usage':usage(0),'responses_sha256':sha(run/'responses.jsonl'),'raw_returns_sha256':sha(run/'raw_returns.jsonl')})
    failures = out/'failed_attempts';failures.mkdir()
    for i in range(3):
        attempt = failures/f'CPU_FIXTURE_STARTUP_{i+1}';attempt.mkdir()
        write_json(attempt/'START.json', {'started_unix': 0, 'fixture_not_model_evidence': True})
        write_json(attempt/'FAILURE.json', {'error': 'CPU-only startup fixture; not an actual engine failure', 'time_unix': 1})
        if i == 0:
            base = out/'run_qwen_cache1'
            rows = [json.loads(line) for line in (base/'responses.jsonl').read_text().splitlines()][:10]
            observed = [json.loads(line) for line in (base/'effective_requests.jsonl').read_text().splitlines()][:10]
            submitted = json.loads((base/'submissions.jsonl').read_text().splitlines()[0])
            returns = json.loads((base/'raw_returns.jsonl').read_text().splitlines()[0])
            for row in rows+observed:row['stage'] = 'controls'
            submitted['stage'] = returns['stage'] = 'controls'
            submitted['jobs'] = submitted['jobs'][:10]
            returns['returns'] = returns['returns'][:10];returns['expected_ids'] = returns['expected_ids'][:10]
            for name, values in [('responses', rows), ('effective_requests', observed), ('submissions', [submitted]), ('raw_returns', [returns])]:
                write_lines(attempt/(name+'.jsonl'), values)
    print('Synthetic CPU-only fixture tree created; never actual model evidence:', out)


if __name__ == '__main__':
    main()
