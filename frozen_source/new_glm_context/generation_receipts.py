"""CPU-only, reusable write-before-validation receipts for blocking generation.

All returned requests and every completion are archived before count, order,
identity or parse checks. A positional expected ID is only an unverified
candidate until a separate validation event binds it to an engine request ID.
The module never loads a model, changes sampling, parses labels or retries.
"""
from pathlib import Path
import hashlib
import json
import os
import time


def compact(value):
    return json.dumps(value,ensure_ascii=False,separators=(',',':'),default=lambda x:{'unserializable_type':type(x).__name__,'repr':repr(x)})


def digest(value):
    return hashlib.sha256(value.encode('utf8')).hexdigest()


def append_durable(path,value):
    encoded=compact(value)+'\n'
    with Path(path).open('a',encoding='utf8') as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    return digest(encoded[:-1])


def _attribute(obj,name,errors,required=True):
    try:
        return getattr(obj,name)
    except Exception as error:
        if required:errors.append({'field':name,'error_type':type(error).__name__,'error':str(error)})
        return None


def _sequence(value,errors,name):
    if value is None:
        errors.append({'field':name,'error':'None; token/output count is unknown, not zero'})
        return None
    try:
        return list(value)
    except Exception as error:
        errors.append({'field':name,'error_type':type(error).__name__,'error':str(error)})
        return None


def snapshot_response(response,position,expected_ids):
    errors=[]
    prompt_ids=_sequence(_attribute(response,'prompt_token_ids',errors),errors,'prompt_token_ids')
    outputs=_sequence(_attribute(response,'outputs',errors),errors,'outputs')
    completions=[] if outputs is not None else None
    if outputs is not None:
        for output_position,output in enumerate(outputs):
            output_errors=[]
            token_ids=_sequence(_attribute(output,'token_ids',output_errors),output_errors,'token_ids')
            text=_attribute(output,'text',output_errors)
            completions.append({'observed_completion_position':output_position,
                'engine_completion_index':_attribute(output,'index',output_errors,False),
                'raw':text,'raw_sha256':digest(text) if isinstance(text,str) else None,
                'output_token_ids':token_ids,'actual_output_tokens':len(token_ids) if token_ids is not None else None,
                'finish_reason':_attribute(output,'finish_reason',output_errors,False),
                'stop_reason':_attribute(output,'stop_reason',output_errors,False),
                'capture_errors':output_errors})
    engine_id=_attribute(response,'request_id',errors)
    complete=not errors and completions is not None and all(not c['capture_errors'] for c in completions)
    return {'observed_return_position':position,'engine_request_id':engine_id,
        'actual_prompt_token_ids':prompt_ids,'actual_input_tokens':len(prompt_ids) if prompt_ids is not None else None,
        'actual_prompt_token_ids_sha256':digest(compact(prompt_ids)) if prompt_ids is not None else None,
        'outputs':completions,'observed_output_count':len(completions) if completions is not None else None,
        'candidate_expected_position':position if position<len(expected_ids) else None,
        'candidate_expected_request_id':expected_ids[position] if position<len(expected_ids) else None,
        'identity_validation_status':'UNVERIFIED; positional candidate is not a trusted scientific binding',
        'capture_complete':complete,'capture_errors':errors}


def return_accounting(directory):
    """Observed payload totals, including unmapped/invalid/duplicate returns.

    Missing, failed or malformed returns retain unknown totals as None rather
    than masquerading as zero cost. Canonical parsed response rows are not the
    authority for these counts.
    """
    directory=Path(directory)
    def rows(name):
        path=directory/name
        return [json.loads(line) for line in path.read_text().splitlines() if line] if path.exists() else []
    invocations=rows('generation_invocations.jsonl')
    receipts=rows('raw_return_receipts.jsonl')
    validations=rows('generation_validation.jsonl')
    failures=rows('generation_failures.jsonl')
    returned={r['generation_call_id']:r for r in receipts}
    validation_by={r['generation_call_id']:r for r in validations}
    actual_count=sum(len(r['actual_returns']) for r in receipts)
    input_payload=sum(r['actual_input_tokens'] for event in receipts for r in event['actual_returns'] if r['actual_input_tokens'] is not None)
    output_payload=sum(c['actual_output_tokens'] for event in receipts for r in event['actual_returns']
                     for c in (r['outputs'] or []) if c['actual_output_tokens'] is not None)
    unresolved=[]
    by_mode={}
    for invocation in invocations:
        call_id=invocation['generation_call_id'];event=returned.get(call_id);validation=validation_by.get(call_id)
        reason=[]
        if event is None:reason.append('generate did not provide an archived complete return sequence; completion compute cost unknown')
        else:
            if not event['all_actual_returns_captured']:reason.append('some returned fields or the return sequence could not be fully captured')
            if len(event['actual_returns'])<len(invocation['expected_batch_request_ids']):reason.append('submitted requests have no returned observation; their token cost is unknown')
        if validation is None or not validation['batch_valid']:reason.append('count/order/identity validation absent or failed; costs/labels cannot be completely attributed')
        if reason:unresolved.append({'generation_call_id':call_id,'reasons':reason})
        mode=invocation['mode'];entry=by_mode.setdefault(mode,{'generate_invocations':0,'expected_request_attempts':0,
            'observed_returned_requests':0,'observed_input_token_payload_total':0,'observed_output_token_payload_total':0,
            'unresolved_invocations':0})
        entry['generate_invocations']+=1;entry['expected_request_attempts']+=len(invocation['expected_batch_request_ids'])
        if event:
            entry['observed_returned_requests']+=len(event['actual_returns'])
            entry['observed_input_token_payload_total']+=sum(r['actual_input_tokens'] for r in event['actual_returns'] if r['actual_input_tokens'] is not None)
            entry['observed_output_token_payload_total']+=sum(c['actual_output_tokens'] for r in event['actual_returns'] for c in (r['outputs'] or []) if c['actual_output_tokens'] is not None)
        entry['unresolved_invocations']+=bool(reason)
    return {'accounting_authority':'append-only generation invocation and raw-return receipts; includes unmapped, extra and invalid returned requests',
        'generate_invocations':len(invocations),'expected_request_attempts':sum(len(r['expected_batch_request_ids']) for r in invocations),
        'observed_returned_requests':actual_count,'observed_completion_outputs':sum(len(r['outputs'] or []) for event in receipts for r in event['actual_returns']),
        'observed_input_token_payload_total':input_payload,'observed_output_token_payload_total':output_payload,
        'payload_totals_are_not_attributed_compute_totals_when_identity_or_completeness_fails':True,
        'complete_total_input_tokens':None if unresolved else input_payload,
        'complete_total_output_tokens':None if unresolved else output_payload,
        'unresolved_invocations':unresolved,'failed_generate_events':len(failures),'by_mode':by_mode,
        'unknown_cost_is_not_zero':True,'unverified_returns_are_never_silently_assigned_to_scientific_conditions':True}


class GenerationReceiptLedger:
    def __init__(self,directory,model):
        self.directory=Path(directory);self.model=model;self.counter=0;self.active=None
        self.previous_engine_ids=set()

    def note_processed(self,request,controls):
        """Record actual engine IDs at ordered input submission, before returns."""
        if self.active is None:raise RuntimeError('Engine processed a request outside the recorded blocking generate call')
        position=len(self.active['processed'])
        candidate=self.active['batch'][position] if position<len(self.active['batch']) else None
        record={'generation_call_id':self.active['generation_call_id'],'processed_position':position,
            'engine_request_id':request.request_id,'actual_prompt_token_ids':list(request.prompt_token_ids),
            'actual_prompt_token_ids_sha256':digest(compact(list(request.prompt_token_ids))),
            'actual_grammar_sha256':controls.get('grammar_sha256'),
            'candidate_expected_request_id':candidate['request_id'] if candidate else None,
            'binding_status':'Provisional ordered submission binding; returned identity not yet checked'}
        self.active['processed'].append(record)
        append_durable(self.directory/'generation_processed_bindings.jsonl',record)

    def generate(self,llm,batch,parameters,mode,diagnostic=False):
        if self.active is not None:raise RuntimeError('Concurrent public generate calls are forbidden by this ledger')
        self.counter+=1
        call_id=f'{self.model}/{os.getpid()}/{self.counter:06d}'
        invocation={'schema_version':1,'generation_call_id':call_id,'model':self.model,'mode':mode,
            'diagnostic':diagnostic,'started_unix':time.time(),'expected_batch_request_ids':[j['request_id'] for j in batch],
            'expected_input_token_ids_sha256':[digest(compact(j['prompt_token_ids'])) for j in batch],
            'expected_grammar_sha256':[j.get('grammar_sha256') for j in batch],
            'position_bindings_are_initially_unverified':True,'automatic_retry':False}
        append_durable(self.directory/'generation_invocations.jsonl',invocation)
        self.active={'generation_call_id':call_id,'batch':batch,'processed':[]}
        try:
            try:
                values=llm.generate([{'prompt_token_ids':j['prompt_token_ids']} for j in batch],parameters,use_tqdm=False)
            except BaseException as error:
                append_durable(self.directory/'generation_failures.jsonl',{'generation_call_id':call_id,'mode':mode,
                    'diagnostic':diagnostic,'time_unix':time.time(),'expected_batch_request_ids':invocation['expected_batch_request_ids'],
                    'processed_engine_request_ids':[r['engine_request_id'] for r in self.active['processed']],
                    'exception_type':type(error).__name__,'message':str(error),'actual_return_count':None,
                    'reason':'generate raised without returning a sequence; no zero-cost assumption or retry',
                    'completion_compute_cost_unknown':True})
                raise
            capture_errors=[];materialized=[]
            try:
                for value in values:materialized.append(value)
            except Exception as error:
                capture_errors.append({'stage':'materialize_return_sequence','error_type':type(error).__name__,'message':str(error)})
            snapshots=[]
            for position,response in enumerate(materialized):
                # Capture every later return even if one object's fields are
                # malformed. Neither identity nor JSON-label parsing runs here.
                snapshots.append(snapshot_response(response,position,invocation['expected_batch_request_ids']))
            receipt={'schema_version':1,'generation_call_id':call_id,'model':self.model,'mode':mode,'diagnostic':diagnostic,
                'returned_unix':time.time(),'expected_batch_request_ids':invocation['expected_batch_request_ids'],
                'observed_return_count':len(materialized) if not capture_errors else None,
                'actual_returns':snapshots,'return_sequence_capture_errors':capture_errors,
                'all_actual_returns_captured':not capture_errors and all(r['capture_complete'] for r in snapshots),
                'validation_status':'All returns persisted before count/order/identity/parse validation',
                'durability':'append, flush, fsync before validation'}
            receipt_sha=append_durable(self.directory/'raw_return_receipts.jsonl',receipt)
            validation=self._validate(batch,snapshots,capture_errors)
            validation.update(generation_call_id=call_id,mode=mode,diagnostic=diagnostic,
                raw_return_receipt_sha256=receipt_sha,time_unix=time.time())
            append_durable(self.directory/'generation_validation.jsonl',validation)
            if not validation['batch_valid']:
                raise RuntimeError('Returned generate batch failed validation after complete raw receipt persistence: '+compact(validation['issues']))
            return materialized,{'generation_call_id':call_id,'raw_return_receipt_sha256':receipt_sha,
                'mapping_validation_status':'verified by engine-ID submission binding, order, complete prompt tokens and grammar'}
        finally:
            self.active=None

    def _validate(self,batch,snapshots,capture_errors):
        issues=[];processed=self.active['processed'];bindings=[]
        if capture_errors:issues.append('Return sequence capture failed; actual returned count/cost remains unknown')
        if len(snapshots)!=len(batch):issues.append(f'Returned count {len(snapshots)} differs from expected {len(batch)}')
        if len(processed)!=len(batch):issues.append(f'Engine-processed count {len(processed)} differs from expected {len(batch)}')
        by_id={};bad_processed=set()
        for position,row in enumerate(processed):
            engine_id=row['engine_request_id']
            if engine_id in by_id or engine_id in self.previous_engine_ids:
                issues.append('Repeated engine request ID in processed binding: '+str(engine_id));bad_processed.add(engine_id)
            by_id[engine_id]=(position,row)
            if position>=len(batch):issues.append('Unexpected additional engine-processed request');continue
            job=batch[position]
            if row['actual_prompt_token_ids']!=job['prompt_token_ids'] or row['actual_grammar_sha256']!=job.get('grammar_sha256'):
                issues.append('Engine-processed input/grammar differs from expected request '+job['request_id']);bad_processed.add(engine_id)
        seen=set()
        for position,snapshot in enumerate(snapshots):
            row_issues=[];engine_id=snapshot['engine_request_id'];bound=by_id.get(engine_id)
            if engine_id in seen:row_issues.append('duplicate returned engine request ID')
            seen.add(engine_id)
            if bound is None:row_issues.append('unknown returned engine request ID')
            if engine_id in bad_processed:row_issues.append('engine submission identity was not validated')
            expected_id=None
            if bound is not None:
                source_position,_=bound
                if source_position<len(batch):
                    job=batch[source_position];expected_id=job['request_id']
                    if source_position!=position:row_issues.append('returned order differs from ordered engine submission binding')
                    if snapshot['actual_prompt_token_ids']!=job['prompt_token_ids']:row_issues.append('returned prompt token IDs differ from bound expected request')
            if not snapshot['capture_complete']:row_issues.append('required returned fields were not completely captured')
            if snapshot['observed_output_count']!=1:row_issues.append('returned completion count differs from frozen n=1')
            for completion in snapshot['outputs'] or []:
                if not isinstance(completion['raw'],str):row_issues.append('returned raw completion is not a string')
            bindings.append({'observed_return_position':position,'engine_request_id':engine_id,
                'candidate_expected_request_id':snapshot['candidate_expected_request_id'],
                'verified_expected_request_id':expected_id if not row_issues else None,
                'identity_validation_status':'verified' if not row_issues else 'FAILED; no trusted scientific assignment',
                'issues':row_issues})
            issues.extend(f'return[{position}]: '+reason for reason in row_issues)
        self.previous_engine_ids.update(by_id)
        verified={r['verified_expected_request_id'] for r in bindings if r['verified_expected_request_id'] is not None}
        return {'batch_valid':not issues,'expected_count':len(batch),'observed_return_count':len(snapshots),
            'processed_count':len(processed),'issues':issues,'returned_bindings':bindings,
            'expected_ids_without_verified_return':[r['request_id'] for r in batch if r['request_id'] not in verified],
            'raw_receipt_remains_authoritative_even_when_validation_fails':True}
