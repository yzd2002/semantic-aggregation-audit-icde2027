#!/usr/bin/env python3
"""Build a public compact evidence tree after real completion. Never publish.

All writes are constrained to this portable_artifact directory. Original
inputs, attempts, runtime and full private native returns remain unchanged.
"""
import argparse
import ast
import csv
from collections import defaultdict
import json
import os
from pathlib import Path
import re
import shutil
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE/'public'))
from portable_common import compact, digest, import_file, lines, load, sha, write_json, write_lines

FORBIDDEN = {'.venv', 'venv', '__pycache__', '.git', 'cache', 'runtime_cache', 'credentials', 'manuscript', 'submission'}


def native_source_templates(root, workloads, text_field):
    from fetch_agnews import PLACEHOLDER
    templates, expected = [], {}
    for workload in workloads:
        records = list(lines(workload['records']))
        path = Path(workload['native_requests'])
        requests = list(lines(path))
        if len(records) != len(requests):
            raise ValueError('Native request/full source occurrence count differs')
        template = None
        serialization = None
        with path.open() as text_stream:
            for record, row, literal in zip(records, requests, text_stream):
                text = record['fields'][text_field]
                item = json.loads(json.dumps(row, ensure_ascii=False))
                if not any(text in message['content'] for message in item['messages']):
                    raise ValueError('Native source formatter does not preserve exact full record text')
                for message in item['messages']:
                    message['content'] = message['content'].replace(text, PLACEHOLDER)
                item['record_id'], item['messages_sha256'] = '__RECORD_ID__', '__CANONICAL_MESSAGE_SHA256__'
                mode = 'compact' if literal == compact(row)+'\n' else 'spaced' if literal == json.dumps(row, ensure_ascii=False)+'\n' else None
                if mode is None:
                    raise ValueError('Unrecognized original request JSONL serialization')
                if template is None:
                    template, serialization = item, mode
                if template != item or serialization != mode:
                    raise ValueError('No exact constant source-text-free request template exists')
        templates.append({'workload_id': workload['workload_id'], 'row_template': template,
                          'relative_output': str(path.relative_to(root)), 'serialization': serialization,
                          'relative_records': str(Path(workload['records']).relative_to(root)),
                          'text_field': text_field})
        for field in ('records', 'gold', 'native_requests'):
            p = Path(workload[field]);expected[str(p.relative_to(root))] = sha(p)
        if 'metadata' in workload:
            p = Path(workload['metadata']);expected[str(p.relative_to(root))] = sha(p)
    return templates, expected


def agnews_fetch_plan(root, study):
    templates, expected = native_source_templates(root,
        [w for w in study['workloads'] if w.get('domain') == 'agnews'], 'text')
    return {'source_repository': 'https://huggingface.co/datasets/fancyzhx/ag_news',
        'revision': 'eb185aade064a813bc0b7f42de02595523103ca4',
        'publisher_url': 'https://huggingface.co/datasets/fancyzhx/ag_news/resolve/eb185aade064a813bc0b7f42de02595523103ca4/data/test-00000-of-00001.parquet',
        'recorded_mirror_url': 'https://hf-mirror.com/datasets/fancyzhx/ag_news/resolve/eb185aade064a813bc0b7f42de02595523103ca4/data/test-00000-of-00001.parquet',
        'source_sha256': '71de87ec66bc5737752a2502204dfa6d7fe9856ade3ea444dc6317789a4f13fb',
        'source_bytes': 1234829, 'feature_seed': '2026110617',
        'rights': 'Publisher card license unknown. Anonymous acquisition does not assign a new data/token license.',
        'native_request_templates': templates, 'expected_reconstruction_sha256': expected}


def movie_fetch_plan(root, study):
    templates, expected = native_source_templates(root,
        [w for w in study['workloads'] if w.get('domain') == 'movie'], 'reviewText')
    return {'source_repository': 'https://www.kaggle.com/datasets/andrezaza/clapper-massive-rotten-tomatoes-movies-and-reviews/versions/4',
        'source_zip_sha256': '92d95672cd39260fca12e6e64d814f7d324ad0600823e0d9207ed1fb52ead707',
        'rights': 'Kaggle version4 host declares CC0:Public Domain; no additional rights warranty.',
        'native_request_templates': templates, 'expected_reconstruction_sha256': expected}


def allowed_write(path):
    path = Path(path).resolve()
    if not path.is_relative_to(HERE):
        raise ValueError('Writes must stay inside portable_artifact')
    return path


def check_complete(root, study, *, fixture=False):
    missing = []
    if not fixture and len(study['models'])*len(study['cases']) != 36:
        missing.append('Expected all 36 prespecified model/configuration cells')
    freeze = sha(root/'PREINFERENCE_FREEZE.json')
    for model in study['models']:
        for cache in sorted({c['prefix_cache'] for c in study['cases']}):
            run = root/f'run_{model}_cache{int(cache)}'
            marker = run/'ALL_WORK_COMPLETE.json'
            if not marker.exists():
                missing.append(str(marker.relative_to(root)))
                continue
            complete = load(marker)
            if complete.get('status') != 'complete' or (run/'FAILURE.json').exists():
                missing.append(str(run.relative_to(root)) + ': unresolved failure/incomplete marker')
            if load(run/'START.json')['prelabel_freeze_sha256'] != freeze:
                missing.append(str(run.relative_to(root)) + ': actual start binds another freeze')
            cases = [c for c in study['cases'] if c['prefix_cache'] == cache]
            if complete['cases_complete'] != len(cases):
                missing.append(str(run.relative_to(root)) + ': missing configurations')
            for case in cases:
                case_root = run/case['case_id']
                if not (case_root/'CASE_COMPLETE.json').exists():
                    missing.append(str(case_root.relative_to(root)))
                else:
                    done = load(case_root/'CASE_COMPLETE.json')
                    for policy, expected in done['results_sha256'].items():
                        if sha(case_root/(policy+'_RESULT.json')) != expected:
                            missing.append(str(case_root.relative_to(root)) + ': policy result drift')
                    if sha(case_root/'INDEPENDENT_VALIDATION.json') != done['validation_sha256']:
                        missing.append(str(case_root.relative_to(root)) + ': validation drift')
    if missing:
        raise ValueError('Actual scientific completion is required; no imputation:\n' + '\n'.join(missing))


def source_files(root):
    frozen = load(root/'PREINFERENCE_FREEZE.json')['files']
    selected = {'PREINFERENCE_FREEZE.json', 'PROTOCOL.txt', 'real_engine_analysis/analyze_study.py'}
    for rel in frozen:
        p = Path(rel)
        if FORBIDDEN.intersection(p.parts):
            continue
        if p.suffix == '.py' or rel in ('prepared/STUDY.json', 'prepared/MODEL_PROVENANCE.json',
                'prepared/HOST_VERSIONS.json', 'prepared/TOKENIZER_FEASIBILITY.json', 'RUNTIME_TECHNICAL_AMENDMENT_V2.json'):
            selected.add(rel)
    for rel in ('PUBLIC_PROTOCOL_FREEZE_VERIFIED.json', 'PUBLIC_RUNTIME_AMENDMENT_V2_VERIFIED.json',
                'run_study_glm_v3.py', 'native_runtime_glm_v3.py', 'launch_glm_v3.py',
                'GLM_NATIVE_TOKENIZER_GUARD_AMENDMENT_V3.json', 'PUBLIC_GLM_TOKENIZER_V3_VERIFIED.json',
                'novelty_theory/finite_analysis_review/GLM_V3_REVIEW.json',
                'novelty_theory/finite_analysis_review/RUNTIME_V2_REVIEW.json',
                'prepare_budget_followup.py', 'BUDGET_FOLLOWUP_STUDY.json', 'BUDGET_FOLLOWUP_FREEZE.json',
                'run_budget_followup.py', 'launch_budget_followup.py', 'PUBLIC_BUDGET_FOLLOWUP_VERIFIED.json',
                'novelty_theory/followup_design/FINAL_REVIEW_RECEIPT.json',
                'real_engine_analysis/budget_followup/analyze_budget_followup.py',
                'real_engine_analysis/budget_followup/analysis_common.py',
                'real_engine_analysis/budget_followup/README.md',
                'real_engine_analysis/budget_followup/BUDGET_ANALYZER_READY.json',
                'real_engine_analysis/budget_followup/SELF_TEST_BUDGET_RECEIPT.json',
                'real_engine_analysis/budget_followup/COMMON_HELPER_PROVENANCE.json',
                'real_engine/REAL_ENGINE_PROTOCOL.md', 'real_engine/sources/movie_v4_download_receipt.json',
                'data_agnews/DOWNLOAD_RECEIPTS.json', 'prepared/agnews/manifest.json'):
        if (root/rel).is_file():
            selected.add(rel)
    repo = root/'real_engine/repos'
    for base in repo.glob('*'):
        if base.is_dir():
            for p in base.glob('*LICENSE*'):
                if p.is_file():
                    selected.add(str(p.relative_to(root)))
    return {rel: root/rel for rel in selected}


def copy_model_license_notices(root, payload):
    """Copy only the pinned license bytes, never model assets or login files."""
    provenance = load(root/'prepared/MODEL_PROVENANCE.json')
    notices = {}
    official = {'qwen': 'https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507',
                'glm': 'https://huggingface.co/THUDM/glm-4-9b-chat'}
    for model, record in provenance.items():
        if model not in official:
            raise ValueError('Model notice needs an explicit official source: '+model)
        source = Path(record['path'])/'LICENSE'
        expected = record['files']['LICENSE']
        if sha(source) != expected:
            raise ValueError('Pinned model license changed: '+model)
        target = payload/'model_license_notices'/model/'LICENSE'
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        notices[model] = {'official_model_page': official[model],
            'revision': record.get('revision'), 'license_sha256': expected,
            'source_binding': 'Matches prepared/MODEL_PROVENANCE.json exactly; no model weights copied',
            'license_file': str(target.relative_to(payload))}
    write_json(payload/'model_license_notices/SOURCE_NOTICE_PROVENANCE.json',
        {'models': notices, 'attribution': 'Built with glm-4.',
         'scope': 'Unmodified pinned model license notices only; no new rights grant for data or outputs.'})


def check_budget_complete(root, fixture=False):
    study = load(root/'BUDGET_FOLLOWUP_STUDY.json')
    if not fixture and study['model_policy_executions'] != 16:
        raise ValueError('Expected the separately frozen 16-policy bounded budget family')
    freeze = sha(root/'BUDGET_FOLLOWUP_FREEZE.json')
    for model in study['models']:
        run = root/f'budget_run_{model}'
        complete = load(run/'BUDGET_ALL_WORK_COMPLETE.json')
        if complete['status'] != 'complete' or complete['jobs_complete'] != len(study['jobs_per_model']) or (run/'FAILURE.json').exists():
            raise ValueError('Budget follow-up has incomplete/failed jobs: '+model)
        if load(run/'START.json')['freeze_sha256'] != freeze:
            raise ValueError('Budget follow-up does not bind its final prospective freeze')
        for job in study['jobs_per_model']:
            folder = run/job['job_id'];marker = load(folder/'POLICY_COMPLETE.json')
            if sha(folder/'POLICY_RESULT.json') != marker['result_sha256']:
                raise ValueError('Budget policy completion binding differs')
    for rel, expected in load(root/'BUDGET_FOLLOWUP_FREEZE.json')['files'].items():
        if sha(root/rel) != expected:
            raise ValueError('Budget prospective input/source file changed: '+rel)


def compact_native(source, target):
    data = load(source)
    hashes = [digest(compact(messages)) for invocation in data.get('transport_invocations', [])
              for messages in invocation.get('messages', [])]
    result = {k: v for k, v in data.items() if k != 'transport_invocations'}
    result['compact_transport_messages_sha256'] = hashes
    result['compact_projection'] = {'original_file_sha256': sha(source),
        'omitted': 'full transport input messages; original private file retained unchanged'}
    write_json(target, result)


def compact_run(source, target, bindings, files, root):
    target.mkdir(parents=True)
    for name in ('responses.jsonl', 'effective_requests.jsonl', 'ALL_WORK_COMPLETE.json', 'START.json',
                 'FAILURE.json', 'EFFECTIVE_NATIVE_RUNTIME.json', 'cache_resets.jsonl',
                 'OFFICIAL_Q3_RECEIPT.json', 'CONTROLS.json', 'BUDGET_ALL_WORK_COMPLETE.json'):
        p = source/name
        if p.is_file():
            shutil.copyfile(p, target/name)
            files[str(p.relative_to(root))] = sha(p)
    submissions = source/'submissions.jsonl'
    jobs = {}
    def submitted_rows():
        if not submissions.exists():
            return
        for batch in lines(submissions):
            copy = {k: v for k, v in batch.items() if k != 'jobs'}
            copy['jobs'] = []
            for job in batch['jobs']:
                check_hash = digest(compact(job['prompt_token_ids']))
                if check_hash != job['prompt_token_ids_sha256']:
                    raise ValueError('Private submitted prompt array hash mismatch')
                item = {k: v for k, v in job.items() if k != 'prompt_token_ids'}
                item['prompt_token_count'] = len(job['prompt_token_ids'])
                copy['jobs'].append(item)
                jobs[item['logical_id']] = item
            yield copy
    write_lines(target/'submitted_calls.jsonl', submitted_rows())
    raw = source/'raw_returns.jsonl'
    def snapshots():
        if not raw.exists():
            return
        for batch in lines(raw):
            copy = {k: v for k, v in batch.items() if k != 'returns'}
            copy['returns'] = []
            for record in batch['returns']:
                pids = record['prompt_token_ids']
                if pids is None:
                    raise ValueError('Unknown actual prompt length cannot be compacted as zero')
                item = {k: v for k, v in record.items() if k not in ('prompt_token_ids', 'outputs')}
                item.update(prompt_token_count=len(pids), prompt_token_ids_sha256=digest(compact(pids)))
                item['outputs'] = []
                for output in record['outputs']:
                    answer = {k: v for k, v in output.items() if k != 'output_token_ids'}
                    answer.update(output_token_count=len(output['output_token_ids']),
                                  output_token_ids_sha256=digest(compact(output['output_token_ids'])))
                    item['outputs'].append(answer)
                copy['returns'].append(item)
            yield copy
    write_lines(target/'raw_return_snapshots.jsonl', snapshots())
    for p in sorted(source.rglob('*')):
        if not p.is_file() or p.is_symlink() or FORBIDDEN.intersection(p.relative_to(source).parts):
            continue
        rel = p.relative_to(source)
        if len(rel.parts) == 1:
            continue
        q = target/rel
        if p.name == 'records.jsonl' or p.name.endswith('stderr.log'):
            continue
        if p.name == 'native_result.json':
            compact_native(p, q)
        elif p.name == 'batch_calls.jsonl':
            q.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(p, q)
            for call in lines(p):
                if call['logical_id'] not in jobs:
                    raise ValueError('Physical batch has no actual submitted request')
                bindings[source.name+'|'+call['logical_id']] = {'indices': call['indices'],
                    'expected_messages_sha256': call['messages_sha256'],
                    'submitted_prompt_token_sha256': jobs[call['logical_id']]['prompt_token_ids_sha256']}
        elif p.name.endswith('_RESULT.json') or p.name in ('CASE_COMPLETE.json', 'POLICY_COMPLETE.json', 'START.json', 'INDEPENDENT_VALIDATION.json', 'transport_receipt.json'):
            q.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(p, q)
        else:
            continue
        files[str(p.relative_to(root))] = sha(p)
    if (source/'OFFICIAL_Q3_RECEIPT.json').exists():
        receipt = load(source/'OFFICIAL_Q3_RECEIPT.json')
        original = load(source/receipt.get('result_file', 'official_sembench_q3/native_result.json'))
        with Path(original['data']).open(newline='') as stream:
            rows = [(i, row) for i, row in enumerate(csv.DictReader(stream)) if row['id'] == 'taken_3']
        ids = [f"sembench-demo-q3:row-{i}:review-{row['reviewId']}" for i, row in rows]
        gold = [{'POSITIVE': True, 'NEGATIVE': False}.get(row['scoreSentiment']) for _, row in rows]
        write_json(target/'OFFICIAL_Q3_COMPACT_SOURCE.json', {'record_ids': ids, 'gold': gold,
            'source_csv_sha256': sha(Path(original['data'])), 'method_sha256': original['method_sha256'],
            'source_text_omitted': True})
    for p in (submissions, raw):
        if p.exists():
            files[str(p.relative_to(root))] = sha(p)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--fixture', action='store_true')
    p.add_argument('--check-only', action='store_true')
    p.add_argument('--with-budget', action='store_true', help='Also require and include the separately frozen 16-policy native-budget family')
    a = p.parse_args()
    root, out = a.root.resolve(), allowed_write(a.out)
    study = load(root/'prepared/STUDY.json')
    fixture_present = (root/'SYNTHETIC_CPU_FIXTURE.json').is_file()
    if fixture_present != a.fixture:
        raise ValueError('Fixture/actual selection must be explicit and consistent')
    try:
        check_complete(root, study, fixture=a.fixture)
        if a.with_budget:
            check_budget_complete(root, fixture=a.fixture)
    except ValueError as error:
        write_json(HERE/'ACTUAL_COMPLETION_STATUS.json', {'status': 'REFUSED_INCOMPLETE' if not a.fixture else 'FIXTURE_INCOMPLETE', 'details': str(error), 'real_archive_hashes': None})
        raise
    if a.check_only:
        print('All actual complete markers and result bindings are present.')
        return
    out.mkdir(parents=True, exist_ok=False)
    payload = out/'bundle'
    payload.mkdir()
    files = source_files(root)
    for rel, source in files.items():
        destination = payload/rel
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    if not a.fixture:
        copy_model_license_notices(root, payload)
    tools = payload/'_portable_tools';tools.mkdir()
    for tool in (HERE/'public').glob('*.py'):
        shutil.copyfile(tool, tools/tool.name)
    for pattern in ('*.md', '*.json', 'requirements-cpu.txt'):
        for file in (HERE/'public').glob(pattern):
            shutil.copyfile(file, tools/file.name)
    original_hashes = {rel: sha(source) for rel, source in files.items()}
    for workload in study['workloads']:
        record_file, requests_file, gold_file = [Path(workload[k]) for k in ('records', 'native_requests', 'gold')]
        for field, source in [('records', record_file), ('native_requests', requests_file)]:
            target = payload/source.relative_to(root)
            if target.exists():
                continue  # Four AG predicates share one occurrence vector.
            if field == 'records':
                values = ({'record_id': r['record_id']} for r in lines(source))
            else:
                values = ({'record_id': r['record_id'], 'messages_sha256': digest(compact(r['messages'])),
                           'canonical_messages_sha256': r.get('messages_sha256'),
                           'full_messages_omitted': True} for r in lines(source))
            write_lines(target, values)
            original_hashes[str(source.relative_to(root))] = sha(source)
        target = payload/gold_file.relative_to(root)
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(gold_file, target)
        original_hashes[str(gold_file.relative_to(root))] = sha(gold_file)
    if not a.fixture:
        write_json(payload/'AGNEWS_FETCH_PLAN.json', agnews_fetch_plan(root, study))
        write_json(payload/'MOVIE_FETCH_PLAN.json', movie_fetch_plan(root, study))
    bindings = {}
    for model in study['models']:
        for cache in sorted({c['prefix_cache'] for c in study['cases']}):
            rel = f'run_{model}_cache{int(cache)}'
            compact_run(root/rel, payload/rel, bindings, original_hashes, root)
    for attempt in sorted((root/'failed_attempts').glob('*')):
        if attempt.is_dir() and (attempt/'FAILURE.json').exists():
            compact_run(attempt, payload/'failed_attempts'/attempt.name, bindings, original_hashes, root)
    if a.with_budget:
        for model in load(root/'BUDGET_FOLLOWUP_STUDY.json')['models']:
            rel = 'budget_run_'+model
            compact_run(root/rel, payload/rel, bindings, original_hashes, root)
    write_json(payload/'COMPACT_BINDINGS.json', {'batch_message_bindings': bindings,
        'binding_key_format': 'run_directory|logical_id',
        'private_full_input_verification_required_for_source_text': True,
        'original_file_sha256': original_hashes})
    # Establish expected mathematics by the original independent analyzer on
    # actual complete private inputs. Writes remain inside this build directory.
    sys.path.insert(0, str(HERE/'public'))
    import replay_compact
    independent = import_file('private_unchanged_analyzer', root/'real_engine_analysis/analyze_study.py')
    expected, integrity = independent.analyze(root/'prepared/STUDY.json', root, out/'private_full_input_analysis', deep=True)
    if integrity['error_count'] or expected['analysis_status'] != 'complete':
        raise ValueError('Original full-input independent analysis did not complete without errors')
    write_json(payload/'EXPECTED_MATHEMATICS.json', replay_compact.normalise_tables(expected))
    # Run original finite mathematics against private actual vectors and labels.
    common = import_file('experiment_common', root/'experiment_common.py')
    sys.modules['experiment_common'] = common
    finite = import_file('private_unchanged_finite', root/'analyze_finite_audits.py')
    finite.main(out/'private_finite_analysis')
    shutil.copyfile(out/'private_finite_analysis/FIXED_BUDGET_AUDITS.json', payload/'EXPECTED_FINITE_AUDITS.json')
    if a.with_budget:
        from replay_budget import analyze_budget
        own_budget = analyze_budget(root, out/'private_budget_analysis', compact_mode=False)
        shutil.copyfile(out/'private_budget_analysis/BUDGET_RECOMPUTATION.json', payload/'EXPECTED_BUDGET_AUDITS.json')
        from replay_official_budget import run_official, compare_views
        if (root/'real_engine_analysis/budget_followup/analyze_budget_followup.py').exists():
            official_budget, official_tables = run_official(root, out/'private_separate_author_budget_analysis')
            compare_views(own_budget, official_budget, out/'PRIVATE_BUDGET_ANALYZER_AGREEMENT.json')
            write_json(payload/'EXPECTED_OFFICIAL_BUDGET_TABLES.json', official_tables)
            shutil.copyfile(out/'PRIVATE_BUDGET_ANALYZER_AGREEMENT.json', payload/'PRIVATE_BUDGET_ANALYZER_AGREEMENT.json')
    kind = 'SYNTHETIC_CPU_FIXTURE_NOT_MODEL_EVIDENCE' if a.fixture else 'ACTUAL_COMPACT_EVIDENCE'
    write_json(payload/'COMPACT_SCOPE.json', {'payload_kind': kind, 'original_root': str(root),
        'actual_configurations': len(study['models'])*len(study['cases']),
        'with_budget_followup': a.with_budget,
        'budget_followup_policy_jobs': len(load(root/'BUDGET_FOLLOWUP_STUDY.json')['models'])*len(load(root/'BUDGET_FOLLOWUP_STUDY.json')['jobs_per_model']) if a.with_budget else 0,
        'original_full_private_data_modified': False,
        'omitted': ['raw source news/review input text', 'full native transport messages', 'input/output token ID arrays', 'weights', 'venv', 'caches', 'credentials', 'user submission instructions'],
        'retained': ['actual model raw output strings', 'original canonical response bytes', 'raw-return input/output lengths and SHA256', 'native effective controls', 'cache/reset receipts', 'all physical groups', 'CP probes', 'label vectors', 'all failed attempts', 'original analysis/math source'],
        'source_rights': 'AG News card says unknown; pinned anonymous acquisition/reconstruction is separate, with no new license grant. Movie Kaggle version4 declares CC0 as hosted; no broader rights warranty.',
        'no_cleanroom_GPU_reproduction_claim': True})
    write_json(payload/'COMPACT_SELECTED_FILES.json', {'scope': 'Post-run compact-public file integrity, not the original full-input preinference freeze',
        'files': {str(f.relative_to(payload)): sha(f) for f in sorted(payload.rglob('*')) if f.is_file()}})
    write_json(payload/'FILE_SHA256.json', {'files': {str(f.relative_to(payload)): sha(f) for f in sorted(payload.rglob('*')) if f.is_file()}})
    for rel, expected_hash in original_hashes.items():
        if sha(root/rel) != expected_hash:
            raise ValueError('Private original changed during compact build: ' + rel)
    write_json(out/'BUILD_STATUS.json', {'status': 'BUILT_AWAITING_ACTUAL_RELOCATION_REPLAY', 'payload_kind': kind, 'originals_unchanged': True, 'public_release_performed': False})
    print('Compact tree built; actual relocation replay is still required:', payload)


if __name__ == '__main__':
    os.environ.update(CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    main()
