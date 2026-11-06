#!/usr/bin/env python3
"""Separate SNLI fresh-result extension; original six-case evidence is immutable.
Only our hash-frozen independent mathematical helpers are imported. No producer
module, tokenizer/model implementation or GPU library is imported/executed.
"""
import collections
import hashlib
import json
import math
import sys
import time
from pathlib import Path
sys.dont_write_bytecode=True
import numpy as np
import pyarrow.parquet as pq
from independent_audit import (compact,digest,file_hash,require,indexed,ordered_ids,regexp,parse_label,
    vector_for,estimate,primary,quality,count_condition,count_pair,get,completed,scope_vectors,
    transport,compare,lookup_from_jobs,SCOPES,CONDS,EXPECTED,MODELS,COMPARISONS)
BASE=Path('/home/gpu/IntentSat2S_revision_20261106/new_snli_context')
PARENT=Path(__file__).resolve().parent
DEST=PARENT/'snli_extension'
DIRS={'mixed':'','serial':'serial'}
HASHES={}
SNLI_SYSTEM="Decide whether each premise entails its paired hypothesis. Use only that record's own complete premise and hypothesis. Other records are not additional evidence for it. Output Y if the hypothesis must be true given the premise, and N if it is neutral or contradicts the premise. Judge only the requested record IDs. Return an exact JSON object with quoted record IDs and quoted Y or N values, with no whitespace or explanation."


def remember(path):
    path=Path(path)
    require(path.is_relative_to(BASE),f'Read outside authorized bundle: {path}')
    HASHES[str(path.relative_to(BASE))]=file_hash(path)

def jread(path):
    path=Path(path);remember(path)
    return json.loads(path.read_text())

def rows(path, optional=False):
    path=Path(path)
    if optional and not path.exists():
        return []
    remember(path)
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]

def write_json(name,value):
    (DEST/name).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def write_rows(name,values):
    with (DEST/name).open('w') as stream:
        for value in values:
            stream.write(compact(value)+'\n')


def source_check(targets):
    pinned={'test':('4696deda851c4d2385f26b58f2e13f9ed9f08ea7b42a3f4c2b97a9d08448878c',411531,10000),
            'train':('ef9a7b25d97390a62aeda7abe26aec8640600f50b818eaeb9107097d60ac6620',19614612,550152)}
    tables={}
    for split,(sha,size,n) in pinned.items():
        p=BASE/f'sources/{split}.parquet';remember(p)
        require(file_hash(p)==sha and p.stat().st_size==size,'Pinned SNLI source payload')
        tables[split]=pq.read_table(p,columns=['premise','hypothesis','label'])
        require(tables[split].num_rows==n,'Pinned SNLI source row count')
    checked={}
    def record(r):
        split=r['source_split'];i=r['source_index'];table=tables[split]
        feature={key:table[key][i].as_py() for key in ('premise','hypothesis')}
        original=table['label'][i].as_py()
        require(original in (0,1,2),'Selected SNLI original label valid')
        require(r['task']=='snli' and r['source_id']==f'{split}:{i:06d}','SNLI row source identity')
        require(r['features']==feature and r['feature_sha256']==digest(compact(feature)),'SNLI complete source features')
        checked[split,i]=original
        return original,int(original==0)
    require(len(targets)==128 and [r['selected_ordinal'] for r in targets]==list(range(128)),'SNLI fixed target order')
    for r in targets:
        require(r['task']=='snli' and r['target_id']==r['target']['source_id'],'SNLI target identity')
        original,binary=record(r['target'])
        require(r['gold']==binary and r['original_gold_class']==original,'SNLI source binary gold mapping')
        for context in ('near','random'):
            require(len(r['contexts'][context])==3,'SNLI background triplets')
            labels=[record(x) for x in r['contexts'][context]]
            require(r['background_original_gold_class'][context]==[x[0] for x in labels] and r['background_gold'][context]==[x[1] for x in labels],'SNLI background gold')
    counts=collections.Counter(str(r['original_gold_class']) for r in targets)
    plan=jread(BASE/'prepared/GOLD_MAPPING_AND_QUALITY_PLAN.json')
    require(dict(counts)==plan['selected_original_class_counts'] and sum(r['gold'] for r in targets)==plan['selected_binary_reference_count'],'SNLI gold quality plan')
    return {'fixed_targets':128,'distinct_source_records_checked':len(checked),'full_complete_fields_and_original_gold_verified':True,
            'binary_mapping':{'0':1,'1':0,'2':0},'original_class_counts':dict(counts),
            'reference_positive_count':sum(r['gold'] for r in targets),'serial_reference_positive_count':sum(r['gold'] for r in targets[:32])}

def prepared_check(model,targets):
    envelope=jread(BASE/f'prepared/REQUEST_ENVELOPE_{model}.json');selection={r['target_id']:r for r in targets}
    jobs={};by_mode={};keyed={};prefixes={}
    for mode in envelope['execution_order']:
        entry=envelope['modes'][mode];path=BASE/entry['file'];require(file_hash(path)==entry['sha256'],'SNLI envelope request SHA')
        requests=rows(path);require(len(requests)==entry['planned_calls'],'SNLI planned request count');by_mode[mode]=requests
        require(sum(len(j['prompt_token_ids']) for j in requests)==entry['planned_input_tokens'],'SNLI planned input-token budget')
        for job in requests:
            rid=job['request_id'];require(rid not in jobs,'Duplicate SNLI request ID');jobs[rid]=job
            r=selection[job['target_id']];require(job['selected_ordinal']==r['selected_ordinal'] and job['task']=='snli','SNLI frozen target/ordinal')
            require(job['execution_mode']==mode and job['max_tokens']==128,'SNLI request mode/cap')
            if mode=='serial':require(r['selected_ordinal']<32,'SNLI serial fixed ordinals')
            records=[r['target']] if job['scope']=='singleton' else [r['target']]+r['contexts'][job['context']]
            lines=[compact({'id':str(i+1),'features':v['features']}) for i,v in enumerate(records)]
            requested=[1] if job['scope'] in ('singleton','target-only') else [1,2,3,4]
            ids=ordered_ids(job['scope']);grammar=regexp(ids)
            user='Records:\n'+'\n'.join(lines)+'\nRequested IDs: '+compact(requested)+'\nReturn one Y or N value for each requested ID.'
            require(job['system']==SNLI_SYSTEM and job['unwrapped_user']==user,'SNLI complete task prompt')
            require(job['requested_ids']==requested and job['output_ids']==ids and job['grammar']==grammar and job['grammar_sha256']==digest(grammar),'SNLI exact output contract')
            require(job['prompt_sha256']==digest(job['wrapped_prompt']) and job['wrapped_prompt'].count(user)==1,'SNLI wrapped prompt identity')
            actual_map={str(i+1):{k:v[k] for k in ('source_id','source_split','feature_sha256')} for i,v in enumerate(records)}
            require(job['id_map']==actual_map and job['id_map_sha256']==digest(compact(actual_map)),'SNLI local source ID mapping')
            tokens=job['prompt_token_ids'];require(all(type(v) is int for v in tokens) and len(tokens)==job['prompt_tokens_preflight'] and len(tokens)+128<=4096,'SNLI exact native prompt token limits')
            start=job['wrapped_prompt'].index(user)+len('Records:\n');end=start+len(lines[0])
            require(job['target_input_line']==lines[0] and job['target_input_span_chars']==[start,end] and job['target_input_span_bytes']==[len(job['wrapped_prompt'][:start].encode()),len(job['wrapped_prompt'][:end].encode())],'SNLI complete first target spans')
            prefix=job['wrapped_prompt'][:end];prefix_n=job['target_prefix_token_count']
            require(job['target_input_prefix']==prefix and job['target_input_prefix_sha256']==digest(prefix) and job['target_prefix_token_ids_sha256']==digest(compact(tokens[:prefix_n])),'SNLI target prefix')
            value=(prefix,tuple(tokens[:prefix_n]),start,end)
            require(prefixes.setdefault(job['target_id'],value)==value,'SNLI fixed target prefixes across modes/contexts/scopes')
            require(job['fixed_output_prefix_before_target']==(None if job['scope']=='full-last' else '{"1":"'),'SNLI target output prefix')
            keyed[mode,job['target_id'],job['scope'],job['context'],job['repeat']]=job
    for (mode,target,scope,context,repeat),job in keyed.items():
        if repeat==2:
            other=keyed[mode,target,scope,context,1]
            require(all(job[k]==other[k] for k in ('wrapped_prompt','prompt_token_ids','grammar','id_map')),'SNLI repeat request identity')
        if scope=='full-last':
            other=keyed[mode,target,'full-first',context,repeat]
            require(job['wrapped_prompt']==other['wrapped_prompt'] and job['prompt_token_ids']==other['prompt_token_ids'],'SNLI first/last same inputs')
        if mode=='serial':
            other=keyed['mixed',target,scope,context,repeat]
            require(job['source_main_request_id']==other['request_id'] and all(job[k]==other[k] for k in ('wrapped_prompt','prompt_token_ids','grammar','id_map','output_ids')),'SNLI serial exact main copy')
        if context=='near':
            other=keyed[mode,target,scope,'random',repeat]
            require(abs(len(job['prompt_token_ids'])-len(other['prompt_token_ids']))<=24,'SNLI native length matching')
    path=BASE/envelope['diagnostic_file'];require(file_hash(path)==envelope['diagnostic_sha256'],'SNLI diagnostic request SHA')
    diagnostics=rows(path);require(len(diagnostics)==20,'SNLI twenty diagnostic calls')
    for job in diagnostics:
        require(job['request_id'] not in jobs,'SNLI diagnostic/science ID disjointness')
        groups=[group for group,ids in envelope['diagnostic_groups'].items() if job['request_id'] in ids]
        require(len(groups)==1,'SNLI diagnostic group identity');job['diagnostic_group']=groups[0]
        job['grammar_sha256']=digest(job['grammar']) if job['grammar'] is not None else None
        require(job['prompt_sha256']==digest(job['wrapped_prompt']) and job['max_tokens']==128,'SNLI diagnostic prompt/cap')
        if job['guided']:require(job['grammar']==regexp(job['output_ids']),'SNLI diagnostic grammar')
        jobs[job['request_id']]=job
    require(len(jobs)==2260,'SNLI unique frozen requests')
    return jobs,by_mode,diagnostics


def raw_chain(model,jobs,by_mode,diagnostic_jobs):
    run=BASE/f'run_{model}';started=jread(run/'START.json')
    require(started['frozen_manifest_sha256']==file_hash(BASE/'PREINFERENCE_FROZEN_SHA256.json'),'Actual freeze binding')
    require(started['request_envelope_sha256']==file_hash(BASE/f'prepared/REQUEST_ENVELOPE_{model}.json'),'Actual envelope binding')
    canonical={};submitted_by_mode={};canonical_modes={}
    for mode,modejobs in by_mode.items():
        directory=run/DIRS[mode]
        blocks=rows(directory/'block_submissions.jsonl');returned=rows(directory/'responses.jsonl')
        seen=[]
        for block in blocks:
            expected=[j['request_id'] for j in modejobs if j['block']==block['block']]
            require(block['request_ids']==expected,'Submission vs frozen block')
            require(len(expected)==1 if 'serial' in mode else len(expected)<=64,'Submission shape')
            seen.extend(expected)
        require(len(set(seen))==len(seen) and set(seen)=={j['request_id'] for j in modejobs},'Science submitted completeness')
        require(len(returned)==len(seen),'Science returned completeness')
        canonical_modes[mode]=returned;submitted_by_mode[mode]=seen
        for r in returned:
            rid=r['request_id'];require(rid not in canonical and rid in seen,'Canonical unique submitted ID');canonical[rid]=r
            j=jobs[rid]
            if 'source_main_request_id' in j:require(r.get('source_main_request_id')==j['source_main_request_id'],'SNLI canonical main-copy identity')
            for key in ('request_id','task','target_id','selected_ordinal','scope','context','repeat','ordinal','block','requested_ids','output_ids','prompt_sha256','grammar_sha256','id_map_sha256','execution_mode'):
                require(r[key]==j[key],f'Canonical frozen identity {model}/{rid}/{key}')
    diagnostic_rows=rows(run/'diagnostics.jsonl');diagblocks=rows(run/'diagnostic_submissions.jsonl')
    diagids=[r for b in diagblocks for r in b['request_ids']]
    require(len(diagids)==len(set(diagids))==20 and set(diagids)=={j['request_id'] for j in diagnostic_jobs},'Diagnostic submissions')
    for r in diagnostic_rows:
        require(r['request_id'] not in canonical,'Repeated diagnostic canonical');canonical[r['request_id']]=r
    require(len(canonical)==len(jobs)==2260,'Canonical completion count')
    inv=rows(run/'generation_invocations.jsonl');ret=rows(run/'raw_return_receipts.jsonl')
    proc=rows(run/'generation_processed_bindings.jsonl');valid=rows(run/'generation_validation.jsonl')
    effective=rows(run/'effective_requests.jsonl');failed=rows(run/'generation_failures.jsonl',True)
    require(not failed,'A generate failure requires explicit unresolved accounting')
    inv_by=indexed(inv,'generation_call_id','invocations');ret_by=indexed(ret,'generation_call_id','receipts')
    valid_by=indexed(valid,'generation_call_id','validations');eff_by=indexed(effective,'engine_request_id','processed controls')
    require(set(inv_by)==set(ret_by)==set(valid_by),'Unresolved receipt/validation invocation')
    proc_by=collections.defaultdict(list)
    for r in proc:proc_by[r['generation_call_id']].append(r)
    require(set(proc_by)==set(inv_by),'Unresolved processed invocation')
    require(len(effective)==len(proc)==2260 and len(inv)==479,'Engine and generate count')
    expected_batches=[('diagnostic/'+b['name'],True,b['request_ids']) for b in diagblocks]
    for mode,modejobs in by_mode.items():
        for block,group in __import__('itertools').groupby(modejobs,key=lambda j:j['block']):
            expected_batches.append((mode,False,[j['request_id'] for j in group]))
    require(len(expected_batches)==479,'SNLI frozen generate batches')
    derived={};by_cost=collections.defaultdict(lambda:collections.Counter());bindings=[];all_engine=set();seen_expected=[];ordinary_control=None
    eos=EXPECTED[model][2];stops=EXPECTED[model][3]
    for invocation_index,call in enumerate(inv):
        require((call['mode'],call['diagnostic'],call['expected_batch_request_ids'])==expected_batches[invocation_index],'SNLI exact invocation/submission order')
        cid=call['generation_call_id'];mode=call['mode'];expected=call['expected_batch_request_ids'];receipt=ret_by[cid];validation=valid_by[cid]
        require(cid.startswith(f'{model}/{started["pid"]}/'),'Fresh invocation process identity')
        require(call['started_unix']>=started['started_unix'] and receipt['returned_unix']>=call['started_unix'],'Fresh invocation/receipt time order')
        require(call['automatic_retry'] is False,'Automatic retry prohibited')
        require(call['expected_input_token_ids_sha256']==[digest(compact(jobs[rid]['prompt_token_ids'])) for rid in expected],'Invocation complete input SHA')
        require(call['expected_grammar_sha256']==[jobs[rid].get('grammar_sha256') for rid in expected],'Invocation grammar SHA')
        require(receipt['expected_batch_request_ids']==expected and receipt['mode']==mode and receipt['model']==model,'Receipt invocation identity')
        require(receipt['diagnostic']==call['diagnostic'] and receipt['all_actual_returns_captured'] and not receipt['return_sequence_capture_errors'],'Incomplete return capture')
        require(receipt['observed_return_count']==len(receipt['actual_returns'])==len(expected),'Return cardinality')
        pp=proc_by[cid];require(len(pp)==len(expected),'Processed cardinality')
        require([p['processed_position'] for p in pp]==list(range(len(pp))),'Processed order')
        p_by=indexed(pp,'engine_request_id','engine bindings')
        require(validation['batch_valid'] and not validation['issues'] and not validation['expected_ids_without_verified_return'],'Producer rejected invocation')
        receipt_hash=digest(compact(receipt))
        require(validation['raw_return_receipt_sha256']==receipt_hash,'Validation durable receipt SHA')
        require(validation['expected_count']==validation['observed_return_count']==validation['processed_count']==len(expected),'Validation count')
        for i,p in enumerate(pp):
            rid=expected[i];job=jobs[rid];engine=p['engine_request_id'];control=eff_by[engine]
            require(engine not in all_engine,'Engine identity reused');all_engine.add(engine)
            require(p['candidate_expected_request_id']==rid and p['actual_prompt_token_ids']==job['prompt_token_ids'],'Processed engine-ID frozen input binding')
            ihash=digest(compact(job['prompt_token_ids']))
            require(p['actual_prompt_token_ids_sha256']==ihash and p['actual_grammar_sha256']==job.get('grammar_sha256'),'Processed prompt/grammar hash')
            require(control['generation_call_id']==cid and control['prompt_token_ids_sha256']==ihash,'Processed parameter binding')
            cp=control['actual_processed_controls']
            for k,v in {'n':1,'temperature':0,'top_k':0,'top_p':1.0,'min_p':0.0,'max_tokens':128,'ignore_eos':False,'repetition_penalty':1.0,'presence_penalty':0.0,'frequency_penalty':0.0,'sampling_type':'GREEDY','primary_eos_token_id':eos,'all_stop_token_ids':stops,'stop_token_ids':stops}.items():
                require(cp[k]==v,f'Actual processed control {model}/{rid}/{k}')
            if job.get('grammar_sha256'):
                require((cp['guided_backend'],cp['guided_disable_fallback'],cp['grammar_sha256'])==('xgrammar',True,job['grammar_sha256']),'Actual guidance/fallback')
            else:require(cp['guided_backend'] is None and cp['grammar_sha256'] is None,'Ordinary control unexpectedly guided')
        for position,actual in enumerate(receipt['actual_returns']):
            engine=actual['engine_request_id'];require(engine in p_by,'Unmapped returned engine ID')
            p=p_by[engine];rid=p['candidate_expected_request_id'];job=jobs[rid];row=canonical[rid]
            require(actual['observed_return_position']==position==p['processed_position'],'Return order matches processed identity')
            require(actual['capture_complete'] and not actual['capture_errors'] and 'UNVERIFIED' in actual['identity_validation_status'],'Incomplete/unmarked raw request payload')
            require(actual['actual_prompt_token_ids']==p['actual_prompt_token_ids']==job['prompt_token_ids'],'Returned complete input tokens')
            ihash=digest(compact(actual['actual_prompt_token_ids']))
            require(actual['actual_prompt_token_ids_sha256']==ihash==row['input_token_ids_sha256'],'Returned input hash')
            require(actual['actual_input_tokens']==len(actual['actual_prompt_token_ids'])==row['actual_input_tokens'],'Actual input token count')
            require(len(actual['outputs'])==actual['observed_output_count']==1,'Returned completion count')
            output=actual['outputs'][0]
            require(output['observed_completion_position']==0 and output['engine_completion_index']==0 and not output['capture_errors'],'Completion identity/capture')
            require(output['raw_sha256']==digest(output['raw'])==row['raw_sha256'],'Raw text SHA')
            require(output['raw']==row['raw'] and output['output_token_ids']==row['output_token_ids'],'Canonical exact durable payload')
            require(output['actual_output_tokens']==len(output['output_token_ids'])==row['actual_output_tokens']<=128,'Actual output token cap/count')
            require(output['finish_reason']==row['finish_reason'] and output['stop_reason']==row['stop_reason'],'Finish metadata binding')
            require(row['engine_request_id']==engine and row['generation_call_id']==cid and row['raw_return_position']==position,'Canonical returned identity')
            require(row['raw_return_receipt_sha256']==receipt_hash,'Canonical durable receipt SHA')
            vb=validation['returned_bindings'][position]
            require(vb['engine_request_id']==engine and vb['verified_expected_request_id']==rid and vb['identity_validation_status']=='verified' and not vb['issues'],'Validation identity corroboration')
            labels=parse_label(output['raw'],job['output_ids'])
            target_label=labels.get(1) if labels else None
            require(row['strict_valid']==(labels is not None) and row['target_label']==target_label,'Independent strict parser vs canonical')
            if 'task' in job:
                require(labels is not None,f'Invalid scientific raw remains retained: {model}/{rid}')
                derived[rid]=target_label
            elif job.get('guided'):
                require(labels is not None,'Guided diagnostic invalid')
            else:
                # The frozen gate is BANANA containment and incompatibility with
                # the five guided JSON contracts. Preserve leading whitespace.
                require('BANANA' in output['raw'] and all(parse_label(output['raw'],ids) is None for ids in ([1],[1,2,3,4],[2,3,4,1],[9,10,11],[11])),'Ordinary/guided control incompatibility')
                ordinary_control={'raw':output['raw'],'literal_exact_BANANA':output['raw']=='BANANA',
                                  'contains_BANANA':True,'input_tokens':actual['actual_input_tokens'],
                                  'output_tokens':output['actual_output_tokens'],'whitespace_preserved':True}
            b=by_cost[mode];b['requests']+=1;b['input_tokens']+=len(actual['actual_prompt_token_ids']);b['output_tokens']+=len(output['output_token_ids']);b['outputs']+=1
            if labels is not None:
                b['strict_valid']+=1
                b['item_labels']+=len(labels)
            bindings.append({'model':model,'request_id':rid,'generation_call_id':cid,'engine_request_id':engine,
                             'raw_return_position':position,'raw_return_receipt_sha256':receipt_hash,
                             'input_token_ids_sha256':ihash,'grammar_sha256':job.get('grammar_sha256'),
                             'raw_sha256':output['raw_sha256'],'target_label':target_label,'diagnostic':call['diagnostic']})
        seen_expected.extend(expected);by_cost[mode]['generate_invocations']+=1
    require(len(seen_expected)==len(set(seen_expected))==2260 and set(seen_expected)==set(jobs),'Lost/extra/repeated attempts')
    native=jread(run/'EFFECTIVE_NATIVE_RUNTIME.json')
    require(native['model_impl']=='vllm' and native['dtype']=='torch.bfloat16','Native BF16 engine')
    for k,v in {'max_model_len':4096,'max_num_seqs':16,'tensor_parallel_size':1,'enforce_eager':True,'prefix_caching':False,'gpu_memory_utilization':.7,'primary_eos_token_id':eos,'guided_decoding_backend':'xgrammar','guided_decoding_disable_fallback':True}.items():
        require(native[k]==v,f'Native engine {k}')
    require(not native['remote_model_modules_loaded'],'Remote modeling module in parent')
    for worker in native['worker_identity']:
        require((worker['model_class'],worker['model_module'])==EXPECTED[model][:2] and worker['parameter_dtypes']==['torch.bfloat16'] and not worker['remote_model_modules_loaded'],'Native actual worker identity')
    saved=jread(run/'RAW_RETURN_ACCOUNTING.json')
    totals={k:sum(v[k] for v in by_cost.values()) for k in ('requests','input_tokens','output_tokens','outputs','generate_invocations')}
    require(saved['generate_invocations']==totals['generate_invocations'] and saved['expected_request_attempts']==saved['observed_returned_requests']==totals['requests'],'Saved raw accounting counts')
    require(saved['complete_total_input_tokens']==totals['input_tokens'] and saved['complete_total_output_tokens']==totals['output_tokens'] and not saved['unresolved_invocations'],'Saved raw accounting tokens')
    scientific={k:sum(v[k] for mode,v in by_cost.items() if not mode.startswith('diagnostic/')) for k in ('requests','input_tokens','output_tokens')}
    controls={k:totals[k]-scientific[k] for k in ('requests','input_tokens','output_tokens')}
    require(scientific['requests']==2240 and controls['requests']==20,'Science/control unique counts')
    scientific['item_labels']=sum(v['item_labels'] for mode,v in by_cost.items() if not mode.startswith('diagnostic/'))
    require(scientific['item_labels']==6080,'Scientific returned item-label budget')
    cfg=jread(BASE/f'prepared/runtime_configuration_{model}.json')
    return derived,bindings,{'by_mode':dict(by_cost),'totals':totals,'scientific':scientific,'controls':controls,
         'engine_processed_requests':len(effective),'unresolved_invocations':[],'actual_native_runtime':native,
         'recorded_package_versions':started['runtime_versions'],'frozen_sampling_normalization':cfg['cpu_effective_sampling'],
         'all_science_calls_strict_valid':True,'guided_diagnostic_calls_strict_valid':19,'ordinary_BANANA_is_not_a_science_parse':True,
         'ordinary_control':ordinary_control,
         'fresh_process_pid':started['pid'],'fresh_start_unix':started['started_unix']}

def statistics(selection,lookup,main=False):
    n=len(selection);common12=completed(selection,lookup,CONDS[:12]);common14=completed(selection,lookup,CONDS)
    gold=[r['gold'] for r in selection];cg=[r['gold'] for r in common12]
    choices=np.random.default_rng(2026110611).integers(0,len(common12),size=(4000,len(common12))) if common12 else None
    constant={'all_N':quality([0]*n,gold),'all_Y':quality([1]*n,gold)}
    result={'common12_n':len(common12),'common14_n':len(common14),'common12_target_ids':[r['target_id'] for r in common12],
            'common14_target_ids':[r['target_id'] for r in common14],'condition_quality':[],'scopes':{},'scope_contrasts':{},
            'fixed_relation_conditions':[],'fixed_relation_comparisons':[]}
    result['constants_fixed_targets' if main else 'constants']=constant
    if main:result.update(selected_targets=n,reference_positive_count=sum(gold))
    else:result.update(selected_subset_n=n,selected_subset_target_ids=[r['target_id'] for r in selection])
    vectors={};preds={};per_target=[]
    for s,c,j in CONDS:
        valid=[r for r in selection if get(lookup,r,s,c,j) is not None]
        p=[get(lookup,r,s,c,j) for r in valid];preds[s,c,j]=p
        common=common14 if s=='singleton' else common12
        q={'scope':s,'context':c,'repeat':j,'own_valid':quality(p,[r['gold'] for r in valid]),
           'common_set':'common14' if s=='singleton' else 'common12',
           'common_quality':quality([get(lookup,r,s,c,j) for r in common],[r['gold'] for r in common])}
        if main:q['selected_denominator']=n
        result['condition_quality'].append(q)
        cnt={'scope':s,'context':c,'repeat':j,**count_condition(p,gold,n)}
        if main:cnt['missing_target_ids']=[r['target_id'] for r in selection if get(lookup,r,s,c,j) is None]
        result['fixed_relation_conditions'].append(cnt)
    for scope in SCOPES:
        v=scope_vectors(common12,lookup,scope);vectors[scope]=v
        result['scopes'][scope]={k:estimate(v[k],choices,main and scope=='full-first' and k=='excess') for k in ('replacement','replay','excess','near_minus_random_error')}
        result['scopes'][scope].update(mean_wrong_to_right_count_over_repeats=sum(v['wrong_to_right']),mean_right_to_wrong_count_over_repeats=sum(v['right_to_wrong']))
        if main:
            result['scopes'][scope]['per_repeat_correctness_changes']=[{'repeat':j,**count_pair([get(lookup,r,scope,'random',j) for r in common12],[get(lookup,r,scope,'near',j) for r in common12],cg,len(common12))} for j in (1,2)]
        else:
            def stable(rows_):
                valid=completed(rows_,lookup,[(scope,c,j) for c in ('near','random') for j in (1,2)])
                ids=[r['target_id'] for r in valid if get(lookup,r,scope,'near',1)==get(lookup,r,scope,'near',2) and get(lookup,r,scope,'random',1)==get(lookup,r,scope,'random',2) and get(lookup,r,scope,'near',1)!=get(lookup,r,scope,'random',1)]
                return {'four_call_valid_n':len(valid),'stable_change_count':len(ids),'stable_change_fraction':len(ids)/len(valid) if valid else None,'stable_change_target_ids':ids}
            result['scopes'][scope]['stable_changes_common12']=stable(common12)
            result.setdefault('stable_changes_own_four_call_valid',{})[scope]=stable(selection)
        entries=[]
        for name,c0,j0,c1,j1 in [('replacement_repeat1','random',1,'near',1),('replacement_repeat2','random',2,'near',2),('replay_near','near',1,'near',2),('replay_random','random',1,'random',2)]:
            pair={'scope':scope,'comparison':name,**count_pair(preds[scope,c0,j0],preds[scope,c1,j1],gold,n)}
            if main:pair.update(from_context=c0,from_repeat=j0,to_context=c1,to_repeat=j1)
            result['fixed_relation_comparisons'].append(pair);entries.append(pair)
        if main:
            if all(x['status']=='complete' for x in entries):
                replacement=(entries[0]['absolute_ratio_change']+entries[1]['absolute_ratio_change'])/2
                replay=(entries[2]['absolute_ratio_change']+entries[3]['absolute_ratio_change'])/2
                summary={'status':'complete','fixed_denominator':n,'mean_absolute_replacement_ratio_change':replacement,'mean_absolute_replay_ratio_change':replay,'excess_absolute_ratio_change_descriptive':replacement-replay}
            else:summary={'status':'abstain','fixed_denominator':n}
            result.setdefault('fixed_relation_ratio_summaries',{})[scope]=summary
    for scope in SCOPES[1:]:
        result['scope_contrasts'][scope+' minus full-first']=estimate([a-b for a,b in zip(vectors[scope]['excess'],vectors['full-first']['excess'])],choices)
    for i,r in enumerate(common12):
        per_target.append({'task':r['task'],'target_id':r['target_id'],'selected_ordinal':r['selected_ordinal'],'gold':r['gold'],
                           'scopes':{s:{k:v[i] for k,v in values.items()} for s,values in vectors.items()},
                           'raw_derived_labels':{s:{c:[get(lookup,r,s,c,j) for j in (1,2)] for c in ('near','random')} for s in SCOPES}})
    if main:
        result['singleton_comparisons']=[]
        for scope in SCOPES:
            for c in ('near','random'):
                for j in (1,2):
                    result['singleton_comparisons'].append({'scope':scope,'context':c,'repeat':j,'set':'common14','n':len(common14),**count_pair([get(lookup,r,'singleton','none',j) for r in common14],[get(lookup,r,scope,c,j) for r in common14],[r['gold'] for r in common14],len(common14))})
        result['singleton_replay_common14']=count_pair([get(lookup,r,'singleton','none',1) for r in common14],[get(lookup,r,'singleton','none',2) for r in common14],[r['gold'] for r in common14],len(common14))
        result['singleton_replay_fixed_relation']=count_pair(preds['singleton','none',1],preds['singleton','none',2],gold,n)
    return result,per_target

def paired_modes(selection,a,b):
    common=completed(completed(selection,a,CONDS[:12]),b,CONDS[:12])
    choices=np.random.default_rng(2026110611).integers(0,len(common),size=(4000,len(common))) if common else None
    result={'common12_both_modes_n':len(common),'common12_both_modes_target_ids':[r['target_id'] for r in common],
            'scopes':{},'paired_scope_contrast_differences':{},'paired_condition_error_differences':[]}
    differences={}
    for s in SCOPES:
        va=scope_vectors(common,a,s);vb=scope_vectors(common,b,s)
        differences[s]=[x-y for x,y in zip(vb['excess'],va['excess'])]
        result['scopes'][s]={k:estimate([x-y for x,y in zip(vb[k],va[k])],choices) for k in ('replacement','replay','excess','near_minus_random_error')}
    for s in SCOPES[1:]:
        result['paired_scope_contrast_differences'][s+' minus full-first']=estimate([x-y for x,y in zip(differences[s],differences['full-first'])],choices)
    for s,c,j in CONDS[:12]:
        delta=[int(get(b,r,s,c,j)!=r['gold'])-int(get(a,r,s,c,j)!=r['gold']) for r in common]
        result['paired_condition_error_differences'].append({'scope':s,'context':c,'repeat':j,**estimate(delta,choices)})
    return result


def main():
    DEST.mkdir(parents=True,exist_ok=True)
    # Do not open live scientific ledgers until all three completed runs and
    # their producer analysis exist. Unavailable SNLI is never converted to zero.
    missing=[]
    for m in MODELS:
        for rel in (f'run_{m}/ALL_WORK_COMPLETE.json',f'analysis_{m}/legacy_main/analysis.json',f'analysis_{m}/legacy_secondary/secondary.json',f'analysis_{m}/global_primary_component.json'):
            if not (BASE/rel).exists():missing.append(rel)
    if missing:
        write_json('PENDING_REAL_COMPLETION.json',{'status':'PENDING; no SNLI result calculated or imputed','missing':missing,'primary_family_size':9})
        print(compact({'status':'PENDING','missing':missing}),flush=True)
        return
    baseline=json.loads((PARENT/'SIX_CASE_BASELINE_FREEZE_SHA256.json').read_text())
    for name,expected in baseline['files'].items():require(file_hash(PARENT/name)==expected,'Six-case baseline changed '+name)
    cert6=json.loads((PARENT/'RESULTS_CERTIFICATE.json').read_text())
    source6=json.loads((PARENT/'SOURCE_RAW_ANALYSIS_SHA256.json').read_text())
    six_source=Path(cert6['read_only_source_bundle'])
    for rel,expected in source6.items():require(file_hash(six_source/rel)==expected,'Six-case scientific source changed '+rel)
    frozen=jread(BASE/'PREINFERENCE_FROZEN_SHA256.json')
    require(len(frozen['files'])==163,'Expected reviewed SNLI freeze163')
    for rel,expected in frozen['files'].items():
        require('..' not in Path(rel).parts and not Path(rel).is_absolute(),'SNLI frozen path escapes')
        p=BASE/rel;remember(p);require(file_hash(p)==expected,'SNLI frozen file drift '+rel)
    targets=sorted(rows(BASE/'selection/targets.jsonl'),key=lambda r:r['selected_ordinal'])
    gold=source_check(targets);write_json('SNLI_SOURCE_AND_FREEZE.json',{'freeze_sha256':file_hash(BASE/'PREINFERENCE_FROZEN_SHA256.json'),'frozen_files':163,'all_frozen_hashes_match':True,'source':gold})
    results={};costs={};bindings=[];all_points=[];cases=[]
    for model in MODELS:
        print('Independent SNLI CPU audit: '+model,flush=True)
        completed_record=jread(BASE/f'run_{model}/ALL_WORK_COMPLETE.json');require(completed_record['status']=='complete','SNLI completion is not complete')
        require(not (BASE/f'run_{model}/FAILURE.json').exists(),'SNLI technical failure requires separately reviewed continuation')
        jobs,mode_jobs,diag=prepared_check(model,targets)
        labels,binding,cost=raw_chain(model,jobs,mode_jobs,diag);costs[model]=cost;bindings.extend(binding)
        mixed=lookup_from_jobs(mode_jobs,labels,'mixed');serial=lookup_from_jobs(mode_jobs,labels,'serial')
        own_main,points=statistics(targets,mixed,True)
        subset=targets[:32];mixed_subset,_=statistics(subset,mixed);serial_subset,serial_points=statistics(subset,serial)
        secondary={'reference_transport':transport(targets,mixed),'mixed_subset':mixed_subset,'serial_subset':serial_subset,'paired_serial_minus_mixed':paired_modes(subset,mixed,serial)}
        producer_main=jread(BASE/f'analysis_{model}/legacy_main/analysis.json');producer_secondary=jread(BASE/f'analysis_{model}/legacy_secondary/secondary.json')
        compare(own_main,producer_main['tasks']['snli'],model+'/snli/main');compare(secondary,producer_secondary['tasks']['snli'],model+'/snli/secondary')
        producer_points=rows(BASE/f'analysis_{model}/legacy_main/per_target.jsonl');compare(points,producer_points,model+'/snli/per_target')
        case=primary([p['scopes']['full-first']['excess'] for p in points],'snli',model,[p['target_id'] for p in points],[p['selected_ordinal'] for p in points])
        producer_case=jread(BASE/f'analysis_{model}/global_primary_component.json')
        require(producer_case['synthetic_cpu_fixture_not_model_evidence'] is False,'SNLI producer is synthetic fixture')
        compare(case,producer_case['cases'][0],model+'/snli/authoritative_primary');cases.append(case)
        results[model]={'main':own_main,'secondary':secondary}
        all_points.extend({'model':model,'mode':'primary_mixed',**p} for p in points)
        all_points.extend({'model':model,'mode':'primary_serial',**p} for p in serial_points)
        write_json(f'SNLI_RECOMPUTED_{model}.json',results[model]);write_json(f'SNLI_RAW_CHAIN_AND_COSTS_{model}.json',cost)
    write_rows('SNLI_INDEPENDENT_RAW_BINDINGS.jsonl',bindings);write_rows('SNLI_INDEPENDENT_PER_TARGET.jsonl',all_points)
    six=json.loads((PARENT/'NINE_FAMILY_PRIMARY_SIX_OBSERVED.json').read_text())
    final_cases=[c for c in six['cases'] if c['task']!='snli']+cases
    final_cases.sort(key=lambda c:(('dblp_acm','boolq','snli').index(c['task']),MODELS.index(c['model'])))
    require(len(final_cases)==9 and all(c['available'] for c in final_cases),'Nine real cases required')
    write_json('NINE_CASE_PRIMARY_INDEPENDENT.json',{'status':'Nine real fresh primary cases independently verified','cases':final_cases,
        'primary_family_size':9,'confidence':179/180,'quantiles':[1/360,359/360],'bootstrap_draws':4000,
        'positive_evidence_count':sum(c['positive_primary_evidence'] for c in final_cases),'all_zero_negative_unfavorable_results_retained':True,
        'finite_selected_target_bootstrap':True,'exact_population_coverage_claimed':False,'exact_FWER_claimed':False,'legacy_98_75_is_not_authoritative':True})
    write_json('SNLI_SOURCE_RAW_ANALYSIS_SHA256.json',dict(sorted(HASHES.items())))
    for rel,expected in HASHES.items():require(file_hash(BASE/rel)==expected,'SNLI hash drift during audit '+rel)
    for name,expected in baseline['files'].items():require(file_hash(PARENT/name)==expected,'Six-case baseline changed during extension '+name)
    artifacts={p.name:file_hash(p) for p in DEST.iterdir() if p.is_file() and p.name not in ('SNLI_AND_NINE_CASES_CERTIFICATE.json','RUN.log','PENDING_REAL_COMPLETION.json')}
    certificate={'status':'PASS: all nine real fresh primary cases independently verified','no_producer_modules_imported':True,
       'no_GPU_or_new_inference':True,'six_case_baseline_unchanged':True,'six_case_original_certificate_sha256':file_hash(PARENT/'RESULTS_CERTIFICATE.json'),
       'six_case_baseline_freeze_sha256':file_hash(PARENT/'SIX_CASE_BASELINE_FREEZE_SHA256.json'),
       'independent_math_helper_sha256':file_hash(PARENT/'independent_audit.py'),'extension_script_sha256':file_hash(Path(__file__)),
       'SNLI_source_manifest_sha256':file_hash(DEST/'SNLI_SOURCE_RAW_ANALYSIS_SHA256.json'),'SNLI_artifact_sha256':artifacts,
       'SNLI_frozen_files_checked':163,'SNLI_unique_scientific_requests':sum(c['scientific']['requests'] for c in costs.values()),
       'SNLI_scientific_item_labels':sum(c['scientific']['item_labels'] for c in costs.values()),'SNLI_control_requests':sum(c['controls']['requests'] for c in costs.values()),
       'SNLI_independent_raw_bindings':len(bindings),'SNLI_comparison_fields':dict(COMPARISONS),'SNLI_comparison_mismatches':0,
       'all_task_model_main_common12':{f"{c['task']}/{c['model']}":c['common12_complete_n'] for c in final_cases},
       'primary_positive_cases':[f"{c['task']}/{c['model']}" for c in final_cases if c['positive_primary_evidence']],
       'SNLI_source_gold':gold,'primary_family_size':9,'legacy_98_75_used_as_primary':False,
       'limits':['Finite 4000-draw selected-target bootstrap is not a population-coverage or exact-FWER certificate.',
                 'SNLI Y=entailment; N combines neutral and contradiction. Binary quality does not measure three-class SNLI accuracy.',
                 'No independent model execution or token decoding. Exact durable input/output arrays, raw strings, grammar, processor controls and native worker identity were verified.',
                 'Serial follows mixed and is secondary. Native family comparisons are not isolated model/size causal effects.'],
       'finished_unix':time.time()}
    write_json('SNLI_AND_NINE_CASES_CERTIFICATE.json',certificate)
    print(compact({'certificate':str(DEST/'SNLI_AND_NINE_CASES_CERTIFICATE.json'),'SNLI_comparison_fields':dict(COMPARISONS),'SNLI_cases':cases}),flush=True)

if __name__=='__main__':main()
