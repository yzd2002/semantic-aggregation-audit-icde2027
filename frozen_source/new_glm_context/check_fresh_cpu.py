#!/usr/bin/env python3
"""Synthetic full-chain checks; generated labels are never model observations."""
import ast
import copy
import time
from common import *
from analyze_fresh_replication import analyze_all,load_unique,RAW_DIRECTORIES


def main():
    out=ROOT/'cpu_checks/fresh_full_chain';out.mkdir()
    for file in ROOT.glob('*.py'):ast.parse(file.read_text())
    for file in ('analyze.py','secondary.py'):
        assert sha(ROOT/file)==sha(ROOT/'source_reference'/file)
    original=(ROOT/'source_reference/common.py').read_text()
    assert (ROOT/'common.py').read_text()==original.replace("MODELS=('qwen','smol')","MODELS=('qwen','smol','glm')")
    # Every model uses exactly the same logical IDs/backgrounds/scopes/schedule.
    keyfields=('request_id','task','target_id','selected_ordinal','scope','context','repeat','ordinal','block',
               'system','unwrapped_user','requested_ids','output_ids','grammar','id_map')
    reference=None
    for model in MODELS:
        p=json.loads((ROOT/'prepared'/f'{model.upper()}_CONTEXT_PREFLIGHT.json').read_text())
        assert p['original_24_token_length_match_in_this_tokenizer'] and p['all_prompt_plus_128_within_4096']
        assert p['length_match_max_absolute_difference']<=24 and p['feature_over_512_occurrences']==0
        jobs=read_lines(ROOT/'prepared'/f'{model}_requests.jsonl')
        fields=[{k:r[k] for k in keyfields} for r in jobs]
        if reference is None:reference=fields
        else:assert fields==reference
        serial=read_lines(ROOT/'prepared'/f'{model}_serial_requests.jsonl');byid={r['request_id']:r for r in jobs}
        for row in serial:
            source=byid[row['source_main_request_id']]
            for key in ('wrapped_prompt','prompt_token_ids','grammar','output_ids','requested_ids','id_map'):
                assert row[key]==source[key]
        for mode in ('mixed','serial'):
            mapping=[r for r in read_lines(ROOT/'prepared'/f'HISTORICAL_SHARED_CALL_MAP_{model}.jsonl') if r['historical_mode']==mode]
            assert sum(not r['uses_shared_primary_observation'] for r in mapping)==(204 if mode=='mixed' else 66)
    # Build an explicitly synthetic complete GLM ledger to exercise all four
    # raw modes, dependent historical lookup views and original statistics.
    model='glm';fixture=out/'SYNTHETIC_NOT_MODEL_RUN';fixture.mkdir()
    (fixture/'prepared').symlink_to(ROOT/'prepared',target_is_directory=True)
    (fixture/'selection').symlink_to(ROOT/'selection',target_is_directory=True)
    (fixture/'historical_context_arm').symlink_to(ROOT/'historical_context_arm',target_is_directory=True)
    run=fixture/f'run_{model}';run.mkdir()
    envelope=json.loads((ROOT/'prepared'/f'REQUEST_ENVELOPE_{model}.json').read_text())
    cfg=json.loads((ROOT/'prepared'/f'runtime_configuration_{model}.json').read_text())
    effective=[];engine_counter=0
    def synthetic_row(job,diagnostic=False):
        nonlocal engine_counter
        engine_id='SYNTHETIC/'+str(engine_counter);engine_counter+=1
        values={str(i):('Y' if (job.get('selected_ordinal',0)+i+job.get('repeat',1))%3 else 'N') for i in job['output_ids']}
        raw=compact(values) if job['output_ids'] else 'BANANA'
        parsed=parse(raw,job['output_ids']) if job['output_ids'] else None
        row={**{k:v for k,v in job.items() if k not in ('prompt_token_ids','wrapped_prompt','target_input_prefix')},
             'engine_request_id':engine_id,'raw':raw,'strict_valid':parsed is not None,
             'target_label':int(parsed[1]=='Y') if parsed is not None and 1 in parsed else None,
             'actual_input_tokens':len(job['prompt_token_ids']),'actual_output_tokens':1,'output_token_ids':[0],
             'input_token_ids_sha256':digest(compact(job['prompt_token_ids'])),'diagnostic':diagnostic,
             'synthetic_cpu_fixture_not_a_model_observation':True}
        effective.append({'engine_request_id':engine_id,'prompt_token_ids_sha256':row['input_token_ids_sha256'],
            'actual_processed_controls':{'n':1,'temperature':0,'top_p':1,'top_k':0,'max_tokens':128,
            'all_stop_token_ids':cfg['stop_token_ids'],'primary_eos_token_id':cfg['primary_eos_token_id'],
            'guided_backend':'xgrammar' if job.get('grammar') else None,
            'guided_disable_fallback':True if job.get('grammar') else None,'grammar_sha256':job.get('grammar_sha256')}})
        return row
    for mode,entry in envelope['modes'].items():
        destination=run/RAW_DIRECTORIES[mode];destination.mkdir(parents=True,exist_ok=True)
        jobs=read_lines(ROOT/entry['file'])
        write_lines(destination/'responses.jsonl',[synthetic_row(job) for job in jobs])
        write_lines(destination/'block_submissions.jsonl',[{'block':block,'request_ids':[j['request_id'] for j in jobs if j['block']==block]}
            for block in sorted({j['block'] for j in jobs})])
    diagnostics=read_lines(ROOT/'prepared/glm_diagnostic_requests.jsonl')
    write_lines(run/'diagnostics.jsonl',[synthetic_row(job,True) for job in diagnostics])
    write_lines(run/'diagnostic_submissions.jsonl',[{'request_ids':[r['request_id'] for r in diagnostics]}])
    write_lines(run/'effective_requests.jsonl',effective)
    result=analyze_all(fixture,model,out/'complete_results')
    assert result['costs']['unique_scientific_returned']==4750 and result['costs']['unique_all_returned']==4770
    for task in TASKS:
        assert result['primary']['tasks'][task]['common12_n']==128
        assert result['historical']['tasks'][task]['common12_n']==128
        assert result['paired']['tasks'][task]['scopes']['full-first']['excess']['mean']==0
    # Keep a whole scientific block unreturned; all submitted attempts remain.
    responses=run/'historical_extra_serial/responses.jsonl'
    saved=responses.read_bytes();rows=read_lines(responses);missing=rows[-1]['request_id']
    responses.write_text(''.join(compact(r)+'\n' for r in rows[:-1]))
    interrupted=analyze_all(fixture,model,out/'interrupted_results')
    assert interrupted['costs']['unique_scientific_submitted']==4750
    assert interrupted['costs']['unique_scientific_returned']==4749
    assert missing in interrupted['costs']['modes']['historical-extra-serial']['submitted_unreturned']
    responses.write_bytes(saved)
    # Detect a wrong raw prompt hash rather than silently accepting it.
    corrupt=copy.deepcopy(rows);corrupt[0]['prompt_sha256']='0'*64
    responses.write_text(''.join(compact(r)+'\n' for r in corrupt))
    rejected=False
    try:load_unique(fixture,model)
    except ValueError as error:
        rejected='identity mismatch' in str(error)
    assert rejected
    responses.write_bytes(saved)
    write_json(out/'CHECKS.json',{'status':'PASS','created_unix':time.time(),'model_inference':False,
        'all_three_model_logical_inputs_identical':True,'all_three_tokenizer_length_bounds':True,
        'original_statistical_files_and_functions_unchanged':True,'full_4750_plus_20_synthetic_chain':True,
        'shared_historical_calls_add_no_unique_cost':True,'interrupted_extra_serial_retained_and_counted':True,
        'corrupt_raw_prompt_hash_rejected':True,'native_gpu_runtime_still_unverified':True})
    print('CPU-only shared-call, full-chain, completion and identity checks PASS',flush=True)


if __name__=='__main__':main()
