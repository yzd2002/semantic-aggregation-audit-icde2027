#!/usr/bin/env python3
"""Validate durable native receipts, then call unchanged SNLI statistics.

Only the one observed SNLI case for this model is emitted. The original legacy
98.75% fields remain available verbatim and do not decide the nine-case family.
"""
import argparse
import collections
import traceback
from common import *
from analyze import analyze,maybe_lines
from secondary import main as secondary_analysis,load_trace,label,CONDITIONS
from generation_receipts import GenerationReceiptLedger,return_accounting,compact as receipt_compact,digest as receipt_digest
from runtime_contract import load_diagnostic_jobs,atomic_json
from global_primary.primary_family import primary_interval as suite_primary_interval,excess_vector as suite_excess_vector

RAW_DIRECTORIES={'mixed':'','serial':'serial'}
IDENTITY_KEYS=('request_id','task','target_id','selected_ordinal','scope','context','repeat','ordinal','block',
    'requested_ids','output_ids','prompt_sha256','grammar_sha256','id_map_sha256','execution_mode','source_main_request_id')


def unique_index(rows,key,description):
    result={row[key]:row for row in rows}
    if len(result)!=len(rows):raise ValueError('Duplicate '+description)
    return result


def check_snapshot(snapshot,position):
    if snapshot['observed_return_position']!=position:raise ValueError('Raw snapshot position mismatch')
    tokens=snapshot['actual_prompt_token_ids']
    if tokens is not None:
        if snapshot['actual_input_tokens']!=len(tokens) or snapshot['actual_prompt_token_ids_sha256']!=digest(compact(tokens)):
            raise ValueError('Raw snapshot prompt token/count/hash mismatch')
    elif snapshot['actual_input_tokens'] is not None or snapshot['actual_prompt_token_ids_sha256'] is not None:
        raise ValueError('Unknown raw prompt costs must remain null')
    outputs=snapshot['outputs']
    if outputs is None:
        if snapshot['observed_output_count'] is not None:raise ValueError('Unknown completion count must remain null')
    else:
        if snapshot['observed_output_count']!=len(outputs):raise ValueError('Raw completion count mismatch')
        for index,output in enumerate(outputs):
            if output['observed_completion_position']!=index:raise ValueError('Raw completion position mismatch')
            output_tokens=output['output_token_ids'];text=output['raw']
            if output['raw_sha256']!=(digest(text) if isinstance(text,str) else None):raise ValueError('Raw completion hash mismatch')
            if output['actual_output_tokens']!=(len(output_tokens) if output_tokens is not None else None):
                raise ValueError('Raw completion token count mismatch')
    complete=not snapshot['capture_errors'] and outputs is not None and all(not c['capture_errors'] for c in outputs)
    if snapshot['capture_complete']!=complete:raise ValueError('Raw snapshot capture flag mismatch')
    if 'UNVERIFIED' not in snapshot['identity_validation_status']:
        raise ValueError('Raw positional candidates must remain explicitly unverified')


def audit_generation_chain(run,expected_jobs,submission_batches):
    invocations=maybe_lines(run/'generation_invocations.jsonl')
    invocation_by=unique_index(invocations,'generation_call_id','generation invocation identity')
    receipts=maybe_lines(run/'raw_return_receipts.jsonl')
    receipt_by=unique_index(receipts,'generation_call_id','raw receipt identity')
    validations=maybe_lines(run/'generation_validation.jsonl')
    validation_by=unique_index(validations,'generation_call_id','generation validation identity')
    processed=maybe_lines(run/'generation_processed_bindings.jsonl')
    failures=maybe_lines(run/'generation_failures.jsonl')
    for event in receipts+validations+processed+failures:
        if event['generation_call_id'] not in invocation_by:raise ValueError('Generation evidence lacks an invocation')
    if len(invocations)>len(submission_batches):raise ValueError('Generation invocation lacks a frozen submission batch')
    used=set();checker=GenerationReceiptLedger(run,'analysis_only_no_writes');verified={}
    for index,invocation in enumerate(invocations):
        call_id=invocation['generation_call_id'];batch=submission_batches[index]
        if invocation['mode']!=batch['mode'] or invocation['diagnostic']!=batch['diagnostic'] or invocation['automatic_retry'] is not False:
            raise ValueError('Invocation mode, diagnostic status or retry policy differs')
        ids=batch['request_ids']
        if invocation['expected_batch_request_ids']!=ids:raise ValueError('Invocation/request-submission sequence differs')
        if any(rid in used for rid in ids):raise ValueError('A request was attempted more than once')
        used.update(ids);jobs=[expected_jobs[rid] for rid in ids]
        if invocation['expected_input_token_ids_sha256']!=[digest(compact(j['prompt_token_ids'])) for j in jobs]:
            raise ValueError('Invocation frozen input-token hashes differ')
        if invocation['expected_grammar_sha256']!=[j.get('grammar_sha256') for j in jobs]:
            raise ValueError('Invocation frozen grammar hashes differ')
        submitted=[r for r in processed if r['generation_call_id']==call_id]
        for position,binding in enumerate(submitted):
            if binding['processed_position']!=position:raise ValueError('Processed-binding order differs')
            if binding['actual_prompt_token_ids_sha256']!=digest(compact(binding['actual_prompt_token_ids'])):
                raise ValueError('Processed-binding prompt hash differs')
        event=receipt_by.get(call_id);validation=validation_by.get(call_id)
        if event is None:
            if validation is not None:raise ValueError('Validation has no durable raw-return receipt')
            continue
        if event['expected_batch_request_ids']!=ids or event['mode']!=batch['mode'] or event['diagnostic']!=batch['diagnostic']:
            raise ValueError('Raw-return receipt request/mode differs')
        snapshots=event['actual_returns'];capture_errors=event['return_sequence_capture_errors']
        for position,snapshot in enumerate(snapshots):check_snapshot(snapshot,position)
        if event['observed_return_count']!=(None if capture_errors else len(snapshots)):
            raise ValueError('Raw-return sequence count differs')
        if event['all_actual_returns_captured']!=(not capture_errors and all(r['capture_complete'] for r in snapshots)):
            raise ValueError('Raw-return sequence completeness flag differs')
        checker.active={'processed':submitted,'batch':jobs}
        recomputed=checker._validate(jobs,snapshots,capture_errors)
        if validation is not None:
            for key,value in recomputed.items():
                if validation[key]!=value:raise ValueError('Saved return validation differs from durable submission/raw evidence: '+key)
            if validation['raw_return_receipt_sha256']!=receipt_digest(receipt_compact(event)):
                raise ValueError('Saved raw-return receipt hash differs')
            if validation['batch_valid']:verified[call_id]=(event,validation)
    return invocations,receipt_by,verified,submission_batches[len(invocations):]


def load_unique(bundle,model):
    bundle=Path(bundle);run=bundle/f'run_{model}'
    envelope=json.loads((bundle/'prepared'/f'REQUEST_ENVELOPE_{model}.json').read_text())
    cfg=json.loads((bundle/'prepared'/f'runtime_configuration_{model}.json').read_text())
    expected_jobs={};observations={};submissions={};source_paths={};all_rows=[];all_submitted=[];detail={}
    submission_batches=[]
    diagnostics=load_diagnostic_jobs(bundle,model)
    diagnostic_jobs=unique_index(diagnostics,'request_id','frozen diagnostic request')
    expected_jobs.update(diagnostic_jobs)
    diag_blocks=maybe_lines(run/'diagnostic_submissions.jsonl')
    if len(diag_blocks)>len(envelope['diagnostic_execution_order']):raise ValueError('Extra diagnostic submission group')
    diag_ids=[]
    for index,block in enumerate(diag_blocks):
        group=envelope['diagnostic_execution_order'][index]
        if block['name']!=group or block['request_ids']!=envelope['diagnostic_groups'][group]:
            raise ValueError('Diagnostic submission differs from frozen order/group')
        diag_ids.extend(block['request_ids'])
        submission_batches.append({'mode':'diagnostic/'+group,'diagnostic':True,'request_ids':block['request_ids']})
    if len(set(diag_ids))!=len(diag_ids):raise ValueError('Duplicate diagnostic attempt')
    for mode in envelope['execution_order']:
        entry=envelope['modes'][mode];jobs_list=read_lines(bundle/entry['file'])
        jobs=unique_index(jobs_list,'request_id','frozen scientific request');expected_jobs.update(jobs)
        destination=run/RAW_DIRECTORIES[mode]
        blocks=maybe_lines(destination/'block_submissions.jsonl');submitted={}
        width=entry['requests_per_submission']
        if len(blocks)>(len(jobs_list)+width-1)//width:raise ValueError('Extra scientific submission batch')
        for index,block in enumerate(blocks):
            planned=jobs_list[index*width:(index+1)*width]
            expected=[j['request_id'] for j in planned]
            if block['request_ids']!=expected or any(j['block']!=block['block'] for j in planned):
                raise ValueError('Actual submission differs from frozen block/order in '+mode)
            for rid in block['request_ids']:
                if rid in submitted:raise ValueError('Duplicate scientific submission '+rid)
                submitted[rid]=block
            submission_batches.append({'mode':mode,'diagnostic':False,'request_ids':block['request_ids']})
        rows=maybe_lines(destination/'responses.jsonl');lookup=unique_index(rows,'request_id','canonical scientific response')
        if any(rid not in jobs or rid not in submitted for rid in lookup):raise ValueError('Unknown/unsubmitted canonical scientific response')
        observations[mode]=lookup;submissions[mode]=submitted;source_paths[mode]=destination/'responses.jsonl'
        all_rows.extend(rows);all_submitted.extend(submitted)
        detail[mode]={'scheduled':len(jobs),'submitted_to_generate':len(submitted),'returned':len(rows),
            'strict_valid':sum(r['strict_valid'] for r in rows),'input_tokens':sum(r['actual_input_tokens'] for r in rows),
            'output_tokens':sum(r['actual_output_tokens'] for r in rows),
            'submitted_unreturned':sorted(set(submitted)-set(lookup)),'unsubmitted':[j['request_id'] for j in jobs_list if j['request_id'] not in submitted],
            'raw_response_sha256':sha(destination/'responses.jsonl') if (destination/'responses.jsonl').exists() else None}
    if len(all_submitted+diag_ids)!=len(set(all_submitted+diag_ids)):raise ValueError('Request attempts are not globally unique')
    diag_rows=maybe_lines(run/'diagnostics.jsonl');diag_lookup=unique_index(diag_rows,'request_id','canonical diagnostic response')
    if any(rid not in diag_ids for rid in diag_lookup):raise ValueError('Unsubmitted diagnostic response')
    canonical_rows=all_rows+diag_rows
    invocations,receipts,verified,not_invoked=audit_generation_chain(run,expected_jobs,submission_batches)
    effective=maybe_lines(run/'effective_requests.jsonl')
    effective_by=unique_index(effective,'engine_request_id','engine-processed request identity')
    engine_ids=[r['engine_request_id'] for r in canonical_rows]
    if len(set(engine_ids))!=len(engine_ids):raise ValueError('Canonical rows reuse an engine request ID')
    for row in canonical_rows:
        job=expected_jobs[row['request_id']]
        for key in IDENTITY_KEYS:
            if key in job and row.get(key)!=job[key]:raise ValueError('Canonical/frozen identity differs: '+row['request_id']+'/'+key)
        is_diagnostic=row['request_id'] in diagnostic_jobs
        if row.get('diagnostic') is not is_diagnostic:raise ValueError('Canonical diagnostic/scientific flag differs')
        call_id=row.get('generation_call_id')
        if call_id not in verified:raise ValueError('Canonical row lacks successful durable generation validation')
        receipt,validation=verified[call_id]
        if row.get('raw_return_receipt_sha256')!=receipt_digest(receipt_compact(receipt)):
            raise ValueError('Canonical raw-return receipt hash differs')
        position=row.get('raw_return_position')
        if type(position) is not int or not 0<=position<len(receipt['actual_returns']):raise ValueError('Canonical raw-return position is invalid')
        observed=receipt['actual_returns'][position];binding=validation['returned_bindings'][position]
        if binding['identity_validation_status']!='verified' or binding['verified_expected_request_id']!=row['request_id']:
            raise ValueError('Canonical request does not match the verified raw-return binding')
        if observed['engine_request_id']!=row['engine_request_id'] or binding['engine_request_id']!=row['engine_request_id']:
            raise ValueError('Canonical engine request differs from durable raw-return evidence')
        if observed['actual_prompt_token_ids']!=job['prompt_token_ids'] or observed['actual_prompt_token_ids_sha256']!=row['input_token_ids_sha256']:
            raise ValueError('Canonical/frozen/durable prompt tokens differ')
        if len(observed['outputs'])!=1:raise ValueError('Canonical row references an invalid completion count')
        completion=observed['outputs'][0]
        for raw_key,canonical_key in [('raw','raw'),('raw_sha256','raw_sha256'),('output_token_ids','output_token_ids'),
                ('actual_output_tokens','actual_output_tokens'),('finish_reason','finish_reason'),('stop_reason','stop_reason')]:
            if completion[raw_key]!=row[canonical_key]:raise ValueError('Canonical completion differs from durable raw payload: '+canonical_key)
        if row['actual_input_tokens']!=len(job['prompt_token_ids']) or row['actual_input_tokens']!=observed['actual_input_tokens']:
            raise ValueError('Canonical actual input count differs')
        if row['actual_output_tokens']!=len(row['output_token_ids']) or not 0<=row['actual_output_tokens']<=128:
            raise ValueError('Canonical output token count/cap differs')
        controls=effective_by.get(row['engine_request_id'])
        if controls is None or controls.get('generation_call_id')!=call_id:raise ValueError('Canonical row lacks matching engine-processed controls')
        if controls['prompt_token_ids_sha256']!=row['input_token_ids_sha256']:raise ValueError('Processed input hash differs')
        p=controls['actual_processed_controls']
        for key,value in cfg['cpu_effective_sampling'].items():
            if key not in ('guided_backend','guided_disable_fallback') and p.get(key)!=value:
                raise ValueError('Actual native sampling differs: '+key)
        if p['primary_eos_token_id']!=cfg['primary_eos_token_id']:raise ValueError('Actual primary EOS differs')
        guided=job.get('grammar') is not None
        if p['grammar_sha256']!=job.get('grammar_sha256') or p['guided_backend']!=('xgrammar' if guided else None) or p['guided_disable_fallback']!=(True if guided else None):
            raise ValueError('Actual native grammar or fallback differs')
        # Parsing occurs only after the canonical row is connected to its
        # complete, durable return object and actual engine controls.
        values=parse(row['raw'],job['output_ids']) if job['output_ids'] else None
        target=int(values[1]=='Y') if values is not None and 1 in values else None
        if row['strict_valid']!=(values is not None) or row['target_label']!=target:
            raise ValueError('Canonical strict parsing differs')
        expected_parsed={str(k):v for k,v in values.items()} if values is not None else None
        if row.get('parsed')!=expected_parsed:raise ValueError('Canonical parsed object differs')
    actual=return_accounting(run)
    unresolved=bool(actual['unresolved_invocations'] or not_invoked)
    costs={'model':model,'budget_scientific_calls':envelope['unique_scientific_calls'],
        'budget_diagnostic_calls':envelope['unique_diagnostic_calls'],'modes':detail,
        'unique_scientific_submitted':len(all_submitted),'unique_scientific_returned':len(all_rows),
        'unique_diagnostic_submitted':len(diag_ids),'unique_diagnostic_returned':len(diag_rows),
        'unique_all_submitted':len(all_submitted)+len(diag_ids),'unique_all_returned':len(canonical_rows),
        'engine_processed_requests':len(effective),'canonical_mapped_observed_tokens':{
            k:sum(r['actual_'+k] for r in canonical_rows) for k in ('input_tokens','output_tokens')},
        'all_observed_tokens':{'input_tokens':None if unresolved else actual['complete_total_input_tokens'],
            'output_tokens':None if unresolved else actual['complete_total_output_tokens']},
        'actual_generation_return_accounting':actual,'submitted_batches_without_invocation':not_invoked,
        'costs_unknown_rather_than_zero_when_any_attempt_is_unresolved':unresolved,
        'canonical_rows_are_not_all_engine_returns_on_integrity_failure':True,
        'no_automatic_retry':True,'all_canonical_rows_verified_against_durable_receipts':True}
    return observations,submissions,source_paths,costs


def analyze_all(bundle,model,out):
    bundle=Path(bundle);out=Path(out);out.mkdir(parents=True,exist_ok=False)
    run=bundle/f'run_{model}'
    if not (run/'START.json').exists():raise ValueError('No real run-start record; cannot emit a prospective primary case')
    synthetic=(bundle/'SYNTHETIC_CPU_FIXTURE.json').exists()
    atomic_json(out/'RAW_RETURN_COSTS_BEFORE_IDENTITY_ANALYSIS.json',{
        'synthetic_cpu_fixture_not_model_evidence':synthetic,'accounting':return_accounting(run)})
    try:
        observations,submissions,source_paths,costs=load_unique(bundle,model)
        write_json(out/'UNIQUE_RAW_COSTS_AND_COMPLETION.json',costs)
        primary=analyze(bundle,model,out/'legacy_main')
        secondary=secondary_analysis(bundle,model,out/'legacy_secondary')
        point_rows=read_lines(out/'legacy_main/per_target.jsonl')
        targets=read_lines(bundle/'selection/targets.jsonl')
        mixed,_=load_trace(bundle,model,'mixed')
        assert TASKS==('snli',)
        values=sorted(point_rows,key=lambda r:r['selected_ordinal'])
        common_ids=primary['tasks']['snli']['common12_target_ids']
        assert [r['target_id'] for r in values]==common_ids
        selected=sorted(targets,key=lambda r:r['selected_ordinal'])
        common=[r for r in selected if r['target_id'] in set(common_ids)]
        arrays=[[label(mixed,r,'full-first',c,j) for r in common] for c,j in [('near',1),('near',2),('random',1),('random',2)]]
        excess=suite_excess_vector(*arrays).tolist()
        assert excess==[r['scopes']['full-first']['excess'] for r in values]
        component=suite_primary_interval(excess,task='snli',model=model)
        assert component['mean_excess']==primary['tasks']['snli']['scopes']['full-first']['excess']['mean']
        component.update(common12_target_ids=common_ids,common12_selected_ordinals=[r['selected_ordinal'] for r in values],
            excluded_target_ids=[r['target_id'] for r in selected if r['target_id'] not in set(common_ids)],
            source_per_target_sha256=sha(out/'legacy_main/per_target.jsonl'),source_legacy_analysis_sha256=sha(out/'legacy_main/analysis.json'),
            synthetic_cpu_fixture_not_model_evidence=synthetic,
            observed_scientific_returns=costs['unique_scientific_returned'])
        write_json(out/'global_primary_component.json',{
            'scope':'One observed SNLI/model case; component of the prospective nine-case suite',
            'model':model,'cases':[component],'complete_nine_case_suite':False,'other_six_context_cases_are_not_fabricated':True,
            'synthetic_cpu_fixture_not_model_evidence':synthetic,'legacy_ci98_75_flags_are_not_authoritative':True,
            'run_start_sha256':sha(run/'START.json'),'amendment':'global_primary/GLOBAL_PRIMARY_AMENDMENT.md',
            'amendment_sha256':sha(bundle/'global_primary/GLOBAL_PRIMARY_AMENDMENT.md'),
            'targets_sha256':sha(bundle/'selection/targets.jsonl')})
        write_json(out/'REPORT_INDEX.json',{'model':model,'synthetic_cpu_fixture_not_model_evidence':synthetic,
            'cost_authority':'UNIQUE_RAW_COSTS_AND_COMPLETION.json','authoritative_suite_primary_component':'global_primary_component.json',
            'legacy_point_estimates_and_intervals':'legacy_main/analysis.json','secondary_serial_transport_COUNT':'legacy_secondary/secondary.json',
            'original_statistics_byte_identical':True,'new_primary_nominal_confidence':179/180,'new_primary_quantiles':[1/360,359/360],
            'interpretation':'Selected-target finite-bootstrap intervals are approximate; original 98.75% flags are legacy only; the global nine-case criterion is authoritative.'})
        return {'costs':costs,'primary':primary,'secondary':secondary,'component':component}
    except BaseException as error:
        atomic_json(out/'ANALYSIS_FAILURE.json',{'type':type(error).__name__,'message':str(error),'traceback':traceback.format_exc(),
            'raw_costs_retained_before_identity_checks':True,'authoritative_primary_component_not_available':True})
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--bundle',type=Path,default=ROOT)
    parser.add_argument('--model',choices=MODELS,required=True);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();analyze_all(args.bundle,args.model,args.out)
