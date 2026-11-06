#!/usr/bin/env python3
"""Fresh raw-ledger analysis and explicit dependent historical/subset views.

The original statistical functions are imported without modification. Derived
view ledgers are logical observation lookups, never new inference submissions.
"""
import argparse
import collections
import shutil
from common import *
from analyze import analyze, maybe_lines
from secondary import main as secondary_analysis, load_trace, paired_modes
from generation_receipts import return_accounting,compact as receipt_compact,digest as receipt_digest
from global_primary.primary_family import primary_interval as suite_primary_interval

RAW_DIRECTORIES={'mixed':'','historical-extra-mixed':'historical_extra','serial':'serial','historical-extra-serial':'historical_extra_serial'}


def load_unique(bundle,model):
    bundle=Path(bundle)
    envelope=json.loads((bundle/'prepared'/f'REQUEST_ENVELOPE_{model}.json').read_text())
    run=bundle/f'run_{model}'
    observations={};submissions={};source_paths={};all_rows=[];all_submitted=[]
    detail={}
    for mode,entry in envelope['modes'].items():
        jobs={r['request_id']:r for r in read_lines(bundle/entry['file'])}
        destination=run/RAW_DIRECTORIES[mode]
        blocks=maybe_lines(destination/'block_submissions.jsonl')
        submitted={}
        for block in blocks:
            expected=[r['request_id'] for r in jobs.values() if r['block']==block['block']]
            if block['request_ids']!=expected:raise ValueError('Actual submission differs from frozen block in '+mode)
            for rid in block['request_ids']:
                if rid in submitted:raise ValueError('Duplicate actual submission '+rid)
                submitted[rid]=block
        rows=maybe_lines(destination/'responses.jsonl');lookup={}
        for row in rows:
            rid=row['request_id']
            if rid not in jobs or rid not in submitted or rid in lookup:raise ValueError('Unknown, repeated or unsubmitted raw call '+rid)
            job=jobs[rid]
            for key in ('request_id','task','target_id','selected_ordinal','scope','context','repeat','ordinal','block',
                        'requested_ids','output_ids','prompt_sha256','grammar_sha256','id_map_sha256'):
                if row[key]!=job[key]:raise ValueError('Raw/frozen request identity mismatch '+rid+'/'+key)
            if row['actual_input_tokens']!=len(job['prompt_token_ids']) or row['input_token_ids_sha256']!=digest(compact(job['prompt_token_ids'])):
                raise ValueError('Actual input tokens differ from frozen native token IDs '+rid)
            values=parse(row['raw'],job['output_ids'])
            if row['strict_valid']!=(values is not None) or row['target_label']!=(int(values[1]=='Y') if values is not None else None):
                raise ValueError('Strict raw parse mismatch '+rid)
            if row['actual_output_tokens']!=len(row['output_token_ids']) or row['actual_output_tokens']>128:
                raise ValueError('Output token/cap mismatch '+rid)
            lookup[rid]=row
        observations[mode]=lookup;submissions[mode]=submitted
        source_paths[mode]=destination/'responses.jsonl'
        all_rows.extend(rows);all_submitted.extend(submitted)
        detail[mode]={'scheduled':len(jobs),'submitted_to_generate':len(submitted),'returned':len(rows),
            'strict_valid':sum(r['strict_valid'] for r in rows),
            'input_tokens':sum(r['actual_input_tokens'] for r in rows),'output_tokens':sum(r['actual_output_tokens'] for r in rows),
            'submitted_unreturned':sorted(set(submitted)-set(lookup)),'unsubmitted':sorted(set(jobs)-set(submitted)),
            'raw_response_sha256':sha(destination/'responses.jsonl') if (destination/'responses.jsonl').exists() else None}
    if len(all_submitted)!=len(set(all_submitted)):raise ValueError('Actual call IDs are not globally unique within this model')
    diagnostics=maybe_lines(run/'diagnostics.jsonl')
    diagnostic_submissions=maybe_lines(run/'diagnostic_submissions.jsonl')
    diag_ids=[rid for block in diagnostic_submissions for rid in block['request_ids']]
    if len(diag_ids)!=len(set(diag_ids)):raise ValueError('Duplicate diagnostic attempt')
    if len({r['request_id'] for r in diagnostics})!=len(diagnostics):raise ValueError('Duplicate diagnostic response')
    if any(r['request_id'] not in diag_ids for r in diagnostics):raise ValueError('Unsubmitted diagnostic response')
    effective=maybe_lines(run/'effective_requests.jsonl')
    effective_by={r['engine_request_id']:r for r in effective}
    if len(effective_by)!=len(effective):raise ValueError('Repeated engine-processed request identity')
    cfg=json.loads((bundle/'prepared'/f'runtime_configuration_{model}.json').read_text())
    canonical_rows=all_rows+diagnostics
    engine_ids=[r['engine_request_id'] for r in canonical_rows]
    if len(set(engine_ids))!=len(engine_ids):raise ValueError('Canonical rows reuse an engine request ID')
    receipt_rows=maybe_lines(run/'raw_return_receipts.jsonl')
    validation_rows=maybe_lines(run/'generation_validation.jsonl')
    receipt_by={r['generation_call_id']:r for r in receipt_rows}
    validation_by={r['generation_call_id']:r for r in validation_rows}
    if len(receipt_by)!=len(receipt_rows) or len(validation_by)!=len(validation_rows):
        raise ValueError('Repeated generation call ID in raw receipt or validation evidence')
    synthetic_only=canonical_rows and all(r.get('synthetic_cpu_fixture_not_a_model_observation') is True for r in canonical_rows)
    if canonical_rows and not receipt_rows and not synthetic_only:
        raise ValueError('Canonical model rows lack the authoritative raw-return receipts')
    for row in all_rows+diagnostics:
        controls=effective_by.get(row['engine_request_id'])
        if controls is None:raise ValueError('Returned raw call lacks actual effective-parameter evidence')
        p=controls['actual_processed_controls']
        if (p['temperature'],p['top_p'],p['top_k'],p['max_tokens'],p['n'])!=(0,1,0,128,1):raise ValueError('Nonfrozen greedy controls')
        if p['all_stop_token_ids']!=cfg['stop_token_ids'] or p['primary_eos_token_id']!=cfg['primary_eos_token_id']:
            raise ValueError('Model-specific native stop/EOS mismatch')
        if controls['prompt_token_ids_sha256']!=row['input_token_ids_sha256']:raise ValueError('Engine-processed input mismatch')
        if row.get('grammar_sha256'):
            if p['guided_backend']!='xgrammar' or not p['guided_disable_fallback'] or p['grammar_sha256']!=row['grammar_sha256']:
                raise ValueError('Actual grammar/fallback differs')
        if receipt_rows:
            call_id=row.get('generation_call_id');receipt=receipt_by.get(call_id);validation=validation_by.get(call_id)
            if receipt is None or validation is None or not validation['batch_valid']:
                raise ValueError('Canonical row lacks successful generation-call receipt validation')
            if controls.get('generation_call_id')!=call_id:
                raise ValueError('Canonical and engine-processed generation call IDs differ')
            if row.get('raw_return_receipt_sha256')!=receipt_digest(receipt_compact(receipt)) or row['raw_return_receipt_sha256']!=validation['raw_return_receipt_sha256']:
                raise ValueError('Canonical raw-return receipt hash differs')
            position=row.get('raw_return_position')
            if type(position) is not int or not 0<=position<len(receipt['actual_returns']):
                raise ValueError('Canonical raw-return position is invalid')
            observed=receipt['actual_returns'][position];binding=validation['returned_bindings'][position]
            if binding['identity_validation_status']!='verified' or binding['verified_expected_request_id']!=row['request_id']:
                raise ValueError('Canonical request is not the verified raw-return binding')
            if observed['engine_request_id']!=row['engine_request_id'] or binding['engine_request_id']!=row['engine_request_id']:
                raise ValueError('Canonical engine request differs from raw-return evidence')
            if observed['actual_prompt_token_ids_sha256']!=row['input_token_ids_sha256']:
                raise ValueError('Canonical prompt hash differs from raw-return evidence')
            if len(observed['outputs'])!=1:
                raise ValueError('Canonical row points to an invalid completion count')
            completion=observed['outputs'][0]
            if completion['raw']!=row['raw'] or completion['raw_sha256']!=row.get('raw_sha256') or completion['output_token_ids']!=row['output_token_ids']:
                raise ValueError('Canonical raw/text/token payload differs from durable raw-return evidence')
    costs={'model':model,'budget_scientific_calls':4750,'budget_diagnostic_calls':20,'modes':detail,
        'unique_scientific_submitted':len(all_submitted),'unique_scientific_returned':len(all_rows),
        'unique_diagnostic_submitted':len(diag_ids),'unique_diagnostic_returned':len(diagnostics),
        'unique_all_submitted':len(all_submitted)+len(diag_ids),'unique_all_returned':len(all_rows)+len(diagnostics),
        'engine_processed_requests':len(effective),
        'all_observed_tokens':{k:sum(r['actual_'+k] for r in all_rows+diagnostics) for k in ('input_tokens','output_tokens')},
        'derived_view_rows_add_no_calls':True,'no_old_run_outputs_used':True}
    if (run/'generation_invocations.jsonl').exists():
        actual_returns=return_accounting(run)
        costs['canonical_mapped_observed_tokens']=costs['all_observed_tokens']
        costs['all_observed_tokens']={'input_tokens':actual_returns['complete_total_input_tokens'],
            'output_tokens':actual_returns['complete_total_output_tokens']}
        costs['actual_generation_return_accounting']=actual_returns
        costs['canonical_rows_are_not_all_engine_returns_on_integrity_failure']=True
        costs['costs_are_unknown_rather_than_zero_when_any_invocation_is_unresolved']=bool(actual_returns['unresolved_invocations'])
    return observations,submissions,source_paths,costs


def project_view(bundle,model,destination,observations,submissions,source_paths,length_pass_only=False):
    bundle=Path(bundle);destination=Path(destination)
    for directory in ('prepared','selection',f'run_{model}',f'run_{model}/serial'):
        (destination/directory).mkdir(parents=True,exist_ok=True)
    primary=read_lines(bundle/'selection/targets.jsonl')
    eligible={(r['task'],r['target_id']) for r in primary if not length_pass_only or not r['glm_primary_random_context_redrawn']}
    targets=[r for r in read_lines(bundle/'historical_context_arm/selection/targets.jsonl') if (r['task'],r['target_id']) in eligible]
    write_lines(destination/'selection/targets.jsonl',targets)
    maps=read_lines(bundle/'prepared'/f'HISTORICAL_SHARED_CALL_MAP_{model}.jsonl')
    mapping={(r['historical_mode'],r['historical_request_id']):r for r in maps}
    provenance=[]
    for mode,schedule_name,subdir in [('mixed','schedule.jsonl',''),('serial','serial_schedule.jsonl','serial')]:
        schedule=[r for r in read_lines(bundle/'prepared'/schedule_name) if (r['task'],r['target_id']) in eligible]
        # A canonical view has one reference lookup per logical block. Original
        # physical mixed/serial block shapes remain in unique raw ledgers.
        schedule=[{**r,'block':r['ordinal']} for r in schedule]
        write_lines(destination/'prepared'/schedule_name,schedule)
        projected=[];lookup_submissions=[]
        for job in schedule:
            link=mapping[mode,job['request_id']]
            source_mode,source_id=link['raw_mode'],link['raw_request_id']
            actual_submission=submissions[source_mode].get(source_id)
            actual=observations[source_mode].get(source_id)
            origin={'raw_mode':source_mode,'raw_request_id':source_id,
                'raw_response_file':str(source_paths[source_mode].relative_to(bundle)),
                'uses_shared_primary_observation':link['uses_shared_primary_observation']}
            provenance.append({'view_request_id':job['request_id'],'view_mode':mode,**origin,
                'actual_submitted':actual_submission is not None,'actual_returned':actual is not None})
            if actual_submission is not None:
                lookup_submissions.append({'block':job['block'],'request_ids':[job['request_id']],
                    'ledger_type':'Derived observation lookup, not an inference submission',
                    'source_submission_block':actual_submission['block'],**origin})
            if actual is not None:
                projected.append({**actual,**job,'projection_origin':origin,
                    'projection_is_not_new_model_call':True})
        run=destination/f'run_{model}'/subdir
        write_lines(run/'responses.jsonl',projected)
        write_lines(run/'block_submissions.jsonl',lookup_submissions)
    write_lines(destination/'OBSERVATION_LOOKUP_PROVENANCE.jsonl',provenance)
    write_json(destination/'DERIVED_VIEW.json',{'kind':'Dependent observation lookup view; no inference is performed here',
        'length_pass_only':length_pass_only,'targets_by_task':dict(collections.Counter(r['task'] for r in targets)),
        'derived_blocks_are_single_reference_lookups':True,'physical_submission_shapes_are_in_the_unique_raw_ledgers':True,
        'view_call_costs_are_not_additional_inference_costs':True})


def analyze_all(bundle,model,out):
    bundle=Path(bundle);out=Path(out);out.mkdir(parents=True,exist_ok=False)
    observations,submissions,source_paths,costs=load_unique(bundle,model)
    write_json(out/'UNIQUE_RAW_COSTS_AND_COMPLETION.json',costs)
    primary=analyze(bundle,model,out/'primary_main')
    point_rows=read_lines(out/'primary_main/per_target.jsonl')
    primary_components=[]
    target_rows=read_lines(bundle/'selection/targets.jsonl')
    for task in TASKS:
        values=sorted([r for r in point_rows if r['task']==task],key=lambda r:r['selected_ordinal'])
        assert [r['target_id'] for r in values]==primary['tasks'][task]['common12_target_ids']
        excess=[r['scopes']['full-first']['excess'] for r in values]
        component=suite_primary_interval(excess,task=task,model=model)
        assert component['mean_excess']==primary['tasks'][task]['scopes']['full-first']['excess']['mean']
        component['common12_target_ids']=[r['target_id'] for r in values]
        component['common12_selected_ordinals']=[r['selected_ordinal'] for r in values]
        complete=set(component['common12_target_ids'])
        component['excluded_target_ids']=[r['target_id'] for r in target_rows if r['task']==task and r['target_id'] not in complete]
        component['source_per_target_sha256']=sha(out/'primary_main/per_target.jsonl')
        component['source_primary_analysis_sha256']=sha(out/'primary_main/analysis.json')
        primary_components.append(component)
    write_json(out/'global_primary_component.json',{'scope':'Two DBLP/BoolQ cases for this model; component of the prospective nine-case suite',
        'model':model,'cases':primary_components,'complete_nine_case_suite':False,
        'SNLI_cases_are_not_fabricated_here':True,'legacy_ci98_75_flags_are_not_authoritative':True,
        'amendment':'global_primary/GLOBAL_PRIMARY_AMENDMENT.md'})
    secondary=secondary_analysis(bundle,model,out/'primary_secondary')
    historical=out/'derived_historical_view'
    project_view(bundle,model,historical,observations,submissions,source_paths)
    historical_main=analyze(historical,model,out/'historical_main')
    historical_secondary=secondary_analysis(historical,model,out/'historical_secondary')
    lengthpass=out/'derived_original_length_pass_view'
    project_view(bundle,model,lengthpass,observations,submissions,source_paths,length_pass_only=True)
    analyze(lengthpass,model,out/'original_length_pass_main')
    secondary_analysis(lengthpass,model,out/'original_length_pass_secondary')
    fresh_labels,_=load_trace(bundle,model,'mixed')
    historical_labels,_=load_trace(historical,model,'mixed')
    targets=read_lines(bundle/'selection/targets.jsonl')
    paired={'direction':'primary matched-context arm minus exact-historical-context arm',
        'status':'secondary dependent comparison, 95% intervals; shared identical observations are not independent', 'tasks':{}}
    for task in TASKS:
        rows=sorted([r for r in targets if r['task']==task],key=lambda r:r['selected_ordinal'])
        result=paired_modes(rows,historical_labels,fresh_labels)
        result['direction']=paired['direction']
        paired['tasks'][task]=result
    write_json(out/'PAIRED_PRIMARY_MINUS_HISTORICAL.json',paired)
    write_json(out/'REPORT_INDEX.json',{'model':model,'cost_authority':'UNIQUE_RAW_COSTS_AND_COMPLETION.json',
        'authoritative_suite_primary_component':'global_primary_component.json',
        'legacy_point_estimates_and_intervals':'primary_main/analysis.json','primary_serial_and_transport':'primary_secondary/secondary.json',
        'secondary_historical':'historical_main/analysis.json','secondary_historical_serial':'historical_secondary/secondary.json',
        'secondary_length_pass':'original_length_pass_main/analysis.json','paired_arms':'PAIRED_PRIMARY_MINUS_HISTORICAL.json',
        'interpretation':'New forward replication after old results were known. The current nine-case suite uses the root prospective 179/180 intervals; original 98.75% flags are legacy only. Selected-target finite-bootstrap intervals are approximate and do not establish exact population FWER.'})
    return {'costs':costs,'primary':primary,'historical':historical_main,'paired':paired}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--bundle',type=Path,default=ROOT)
    parser.add_argument('--model',choices=MODELS,required=True);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();analyze_all(args.bundle,args.model,args.out)
