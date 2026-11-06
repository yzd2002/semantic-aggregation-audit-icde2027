#!/usr/bin/env python3
"""Read-only, CPU-only independent audit. No producer modules are imported.

Scientific inputs are the durable return payloads after an independently checked
engine-ID/complete-input binding. Producer analysis is read only for comparison.
All writes stay beside this script, or an explicitly supplied output directory.
"""
import argparse
import collections
import csv
import hashlib
import itertools
import json
import math
import re
import sys
import time
from pathlib import Path

import numpy as np

BASE = Path('/home/gpu/IntentSat2S_revision_20261106/new_glm_context')
DEST = Path(__file__).resolve().parent
MODELS = ('qwen', 'smol', 'glm')
TASKS = ('dblp_acm', 'boolq')
SCOPES = ('full-first', 'target-only', 'full-last')
CONDS = [(s,c,j) for s in SCOPES for c in ('near','random') for j in (1,2)] + [('singleton','none',j) for j in (1,2)]
DIRS = {'mixed':'', 'serial':'serial', 'historical-extra-mixed':'historical_extra', 'historical-extra-serial':'historical_extra_serial'}
EXPECTED = {'qwen':('Qwen3ForCausalLM','vllm.model_executor.models.qwen3',151645,[151643,151645]),
            'smol':('LlamaForCausalLM','vllm.model_executor.models.llama',2,[2]),
            'glm':('ChatGLMForCausalLM','vllm.model_executor.models.chatglm',151329,[151329,151336,151338])}
SYSTEMS = {
 'dblp_acm': "Decide whether each pair of bibliographic citations refers to the same publication. Use only that pair's own title, authors, venue, and year fields. Other records are not additional evidence for it. Output Y for the same publication and N for different publications. Judge only the requested record IDs. Return an exact JSON object with quoted record IDs and quoted Y or N values, with no whitespace or explanation.",
 'boolq': 'Answer each yes/no question using only its own passage. Other records are not additional evidence for it. Output Y if the answer is yes and N if the answer is no. Judge only the requested record IDs. Return an exact JSON object with quoted record IDs and quoted Y or N values, with no whitespace or explanation.'}
HASHES = {}
COMPARISONS = collections.Counter()

def require(condition, message):
    if not condition:
        raise AssertionError(message)

def compact(obj):
    return json.dumps(obj,ensure_ascii=False,separators=(',',':'))

def digest(value):
    return hashlib.sha256(value if isinstance(value,bytes) else value.encode('utf-8')).hexdigest()

def file_hash(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4<<20),b''):
            h.update(block)
    return h.hexdigest()

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

def indexed(values,key,where):
    result={r[key]:r for r in values}
    require(len(result)==len(values),f'Duplicate {key}: {where}')
    return result

def ordered_ids(scope):
    return [2,3,4,1] if scope=='full-last' else ([1] if scope in ('target-only','singleton') else [1,2,3,4])

def regexp(ids):
    return r'\{'+','.join('"'+str(i)+'":"[YN]"' for i in ids)+r'\}'

def parse_label(raw, ids):
    # Independently demand the exact byte-level object shape before JSON decode.
    if not isinstance(raw,str) or re.fullmatch(regexp(ids),raw) is None:
        return None
    pairs=json.loads(raw,object_pairs_hook=list)
    if [k for k,_ in pairs]!=[str(i) for i in ids]:
        return None
    return {int(k):int(v=='Y') for k,v in pairs}

def vector_for(a,b,c,d,gold):
    values={k:[] for k in ('replacement','replay','excess','near_minus_random_error','wrong_to_right','right_to_wrong')}
    for n1,n2,r1,r2,g in zip(a,b,c,d,gold):
        replacement=(int(n1!=r1)+int(n2!=r2))/2
        replay=(int(n1!=n2)+int(r1!=r2))/2
        wr=(int(r1!=g and n1==g)+int(r2!=g and n2==g))/2
        rw=(int(r1==g and n1!=g)+int(r2==g and n2!=g))/2
        error=(int(n1!=g)+int(n2!=g)-int(r1!=g)-int(r2!=g))/2
        require(error==rw-wr,'Paired error identity')
        for k,v in zip(values,(replacement,replay,replacement-replay,error,wr,rw)):
            values[k].append(v)
    return values

def estimate(values, draws, legacy=False):
    if not values:
        return {'n':0,'mean':None,'ci95':None,**({'ci98_75':None,'positive_primary_evidence':False} if legacy else {})}
    x=np.asarray(values,dtype=float)
    simulations=x[draws].mean(axis=1)
    answer={'n':len(values),'mean':float(x.mean()),'ci95':np.quantile(simulations,[.025,.975],method='linear').tolist()}
    if legacy:
        answer['ci98_75']=np.quantile(simulations,[.00625,.99375],method='linear').tolist()
        answer['positive_primary_evidence']=bool(answer['ci98_75'][0]>0)
    return answer

def primary(values,task,model,ids,ordinals):
    seed=int.from_bytes(hashlib.sha256(f'2026110613|{task}|{model}'.encode()).digest()[:8],'big')
    x=np.asarray(values,dtype=float)
    require(all(v in (-1,0,1) for v in values),'Nonternary binary four-call excess')
    if len(x):
        choices=np.random.default_rng(seed).integers(0,len(x),size=(4000,len(x)))
        sims=x[choices].mean(axis=1)
        interval=np.quantile(sims,[1/360,359/360],method='linear').tolist()
    else:
        interval=None
    return {'task':task,'model':model,'mode':'primary_mixed','scope':'full-first','planned_target_n':128,
            'common12_complete_n':len(x),'excluded_n':128-len(x),'bootstrap_draws':4000,'bootstrap_seed':seed,
            'confidence':179/180,'quantiles':[1/360,359/360],'nominal_family_alpha':.05,'primary_family_size':9,
            'available':bool(len(x)),'mean_excess':float(x.mean()) if len(x) else None,'interval':interval,
            'positive_primary_evidence':bool(interval and interval[0]>0),
            'exact_population_coverage_claimed':False,'exact_FWER_claimed':False,
            'common12_target_ids':ids,'common12_selected_ordinals':ordinals,
            'excess_value_counts':{str(v):values.count(v) for v in (-1,0,1)}}

def quality(pred,gold):
    require(len(pred)==len(gold),'Pred/gold length mismatch')
    tp=tn=fp=fn=0
    for p,g in zip(pred,gold):
        require(p in (0,1) and g in (0,1),'Invalid prediction or gold')
        if p==1 and g==1:tp+=1
        elif p==0 and g==0:tn+=1
        elif p==1:fp+=1
        else:fn+=1
    n=len(gold);pr=tp/(tp+fn) if tp+fn else None;nr=tn/(tn+fp) if tn+fp else None
    return {'n':n,'tp':tp,'fp':fp,'tn':tn,'fn':fn,'errors':fp+fn,'error_rate':(fp+fn)/n if n else None,
            'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else (0.0 if n else None),
            'positive_recall':pr,'negative_recall':nr,'balanced_accuracy':(pr+nr)/2 if pr is not None and nr is not None else None,
            'predicted_positive_count':tp+fp,'reference_positive_count':tp+fn,
            'predicted_positive_fraction':(tp+fp)/n if n else None,
            'reference_error_cancellation_count':fp+fn-abs(fp-fn)}

def count_condition(pred,gold,n):
    if n==0 or len(pred)!=n:
        return {'status':'abstain','fixed_denominator':n,'valid_targets':len(pred),
                'reason':'no_selected_targets' if n==0 else 'not_all_selected_targets_valid'}
    q=quality(pred,gold);delta=sum(pred)-sum(gold)
    return {'status':'complete','fixed_denominator':n,'valid_targets':n,'predicted_positive_count':sum(pred),
            'reference_positive_count':sum(gold),'ratio':sum(pred)/n,'signed_count_error':delta,'absolute_count_error':abs(delta),
            'absolute_ratio_error':abs(delta)/n,'false_positive_count':q['fp'],'false_negative_count':q['fn'],
            'candidate_reference_errors':q['errors'],'reference_error_cancellation_count':q['reference_error_cancellation_count']}

def count_pair(before,after,gold,n):
    if not n or len(before)!=n or len(after)!=n:
        return {'status':'abstain','fixed_denominator':n,'reason':'at_least_one_fixed_relation_condition_incomplete'}
    changes=[(b,a,g) for b,a,g in zip(before,after,gold) if b!=a]
    ny=sum(b==0 and a==1 for b,a,g in changes);yn=len(changes)-ny
    wr=sum(b!=g and a==g for b,a,g in changes);rw=len(changes)-wr
    delta=sum(after)-sum(before)
    require(delta==ny-yn and rw-wr==sum(a!=g for a,g in zip(after,gold))-sum(b!=g for b,g in zip(before,gold)),'COUNT pair invariants')
    return {'status':'complete','fixed_denominator':n,'before_positive_count':sum(before),'after_positive_count':sum(after),
            'changed_target_count':len(changes),'n_to_y':ny,'y_to_n':yn,'signed_count_change':delta,'absolute_count_change':abs(delta),
            'signed_ratio_change':delta/n,'absolute_ratio_change':abs(delta)/n,
            'reference_free_cancelled_flip_count':len(changes)-abs(delta),
            'reference_free_cancelled_flip_fraction':(len(changes)-abs(delta))/len(changes) if changes else None,
            'wrong_to_right':wr,'right_to_wrong':rw,'signed_candidate_reference_error_change':rw-wr,
            'correctness_change_cancellation_count':len(changes)-abs(rw-wr),
            'before_absolute_ratio_error':abs(sum(before)-sum(gold))/n,'after_absolute_ratio_error':abs(sum(after)-sum(gold))/n}

def source_selection_check(targets,historical):
    def csv_rows(path):
        remember(path)
        with path.open() as f:return list(csv.DictReader(f))
    A={r['id']:r for r in csv_rows(BASE/'sources/dblp_acm/tableA.csv')}
    B={r['id']:r for r in csv_rows(BASE/'sources/dblp_acm/tableB.csv')}
    pairs={s:csv_rows(BASE/f'sources/dblp_acm/{s}.csv') for s in ('train','valid','test')}
    boolq={s:rows(BASE/f'sources/boolq/{s}.jsonl') for s in ('train','validation')}
    verified={}
    def record(r):
        identity=(r['task'],r['source_id']);split=r['source_split'];i=r['source_index']
        if r['task']=='dblp_acm':
            p=pairs[split][i];left=p['ltable_id'];right=p['rtable_id']
            require((left,right)==(r['left_id'],r['right_id']),f'Source join IDs {identity}')
            features={'left':{k:A[left][k] for k in ('title','authors','venue','year')},'right':{k:B[right][k] for k in ('title','authors','venue','year')}}
            require(r['source_id']==f'{split}:{i:05d}:L{left}:R{right}',f'Pair source ID {identity}')
            gold=int(p['label'])
        else:
            p=boolq[split][i];features={k:p[k] for k in ('question','passage')};gold=int(p['answer'])
            require(p['row_idx']==i and r['source_id']==f'{split}:{i:05d}',f'BoolQ source index {identity}')
        require(r['features']==features,f'Complete source feature mismatch {identity}')
        require(r['feature_sha256']==digest(compact(features)),f'Feature SHA mismatch {identity}')
        verified[identity]=gold
        return gold
    for selection in (targets,historical):
        require(len(selection)==256,'Fixed target count')
        for task in TASKS:
            taskrows=sorted([r for r in selection if r['task']==task],key=lambda r:r['selected_ordinal'])
            require([r['selected_ordinal'] for r in taskrows]==list(range(128)),f'Fixed ordinals {task}')
        for r in selection:
            require(r['gold']==record(r['target']),f'Target gold {r["target_id"]}')
            require(r['target_id']==r['target']['source_id'],'Target identity')
            for c in ('near','random'):
                require(len(r['contexts'][c])==3,'Triplet size')
                require(r['background_gold'][c]==[record(x) for x in r['contexts'][c]],'Background gold')
    historical_by={(r['task'],r['target_id']):r for r in historical}
    changes=collections.Counter()
    for r in targets:
        old=historical_by[r['task'],r['target_id']]
        require((r['target'],r['gold'],r['contexts']['near'])==(old['target'],old['gold'],old['contexts']['near']),'Historical target/near changed')
        changed=r['contexts']['random']!=old['contexts']['random']
        require(changed==r['glm_primary_random_context_redrawn'],'Changed-random marker mismatch')
        if changed:changes[r['task']]+=1
    require(dict(changes)=={'dblp_acm':4,'boolq':30},'Historical random-change counts')
    return {'verified_distinct_source_records':len(verified),'fixed_targets_per_task':128,'source_gold_verified':True,
            'all_complete_fields_verified':True,'historical_random_changed_targets':dict(changes),'target_and_near_unchanged':True}

def prepared_check(model,targets,historical):
    envelope=jread(BASE/f'prepared/REQUEST_ENVELOPE_{model}.json')
    sels={False:{(r['task'],r['target_id']):r for r in targets},True:{(r['task'],r['target_id']):r for r in historical}}
    jobs={};by_mode={};target_prefix={};keyed={}
    for mode,entry in envelope['modes'].items():
        path=BASE/entry['file'];require(file_hash(path)==entry['sha256'],f'Envelope SHA {model}/{mode}')
        requests=rows(path);require(len(requests)==entry['calls'],f'Envelope calls {model}/{mode}')
        by_mode[mode]=requests
        for job in requests:
            rid=job['request_id'];require(rid not in jobs,'Repeated frozen request ID')
            jobs[rid]=job
            r=sels[mode.startswith('historical-extra')][job['task'],job['target_id']]
            require(job['selected_ordinal']==r['selected_ordinal'],'Prepared target ordinal')
            if 'serial' in mode:require(r['selected_ordinal']<32,'Serial selected subset')
            records=[r['target']] if job['scope']=='singleton' else [r['target']]+r['contexts'][job['context']]
            lines=[compact({'id':str(i+1),'features':v['features']}) for i,v in enumerate(records)]
            requested=[1] if job['scope'] in ('singleton','target-only') else [1,2,3,4]
            ids=ordered_ids(job['scope']);grammar=regexp(ids)
            user='Records:\n'+'\n'.join(lines)+'\nRequested IDs: '+compact(requested)+'\nReturn one Y or N value for each requested ID.'
            require(job['system']==SYSTEMS[job['task']] and job['unwrapped_user']==user,f'Complete prompt {model}/{rid}')
            require(job['requested_ids']==requested and job['output_ids']==ids and job['grammar']==grammar,'Requested grammar/order')
            require(job['grammar_sha256']==digest(grammar) and job['prompt_sha256']==digest(job['wrapped_prompt']),'Prompt/grammar hashes')
            require(job['wrapped_prompt'].count(user)==1,'Unwrapped prompt location')
            actual_map={str(i+1):{k:v[k] for k in ('source_id','source_split','feature_sha256')} for i,v in enumerate(records)}
            require(job['id_map']==actual_map and job['id_map_sha256']==digest(compact(actual_map)),'Local/source ID mapping')
            ids_in=job['prompt_token_ids'];require(all(type(v) is int for v in ids_in),'Noninteger frozen tokens')
            require(job['prompt_tokens_preflight']==len(ids_in) and len(ids_in)+128<=4096,'Native input/context limit')
            start=job['wrapped_prompt'].index(user)+len('Records:\n');end=start+len(lines[0])
            require(job['target_input_line']==lines[0] and job['target_input_span_chars']==[start,end],'Target full line/span')
            require(job['target_input_span_bytes']==[len(job['wrapped_prompt'][:start].encode()),len(job['wrapped_prompt'][:end].encode())],'Target byte span')
            prefix=job['wrapped_prompt'][:end];prefix_n=job['target_prefix_token_count']
            require(job['target_input_prefix']==prefix and job['target_input_prefix_sha256']==digest(prefix),'Target byte prefix hash')
            require(job['target_prefix_token_ids_sha256']==digest(compact(ids_in[:prefix_n])),'Target token prefix hash')
            pkey=(job['task'],job['target_id']);pvalue=(prefix,tuple(ids_in[:prefix_n]),start,end)
            require(target_prefix.setdefault(pkey,pvalue)==pvalue,'Fixed target prefix across all contexts/scopes/modes')
            require(job['fixed_output_prefix_before_target']==(None if job['scope']=='full-last' else '{"1":"'),'Target output prefix')
            keyed[mode,job['task'],job['target_id'],job['scope'],job['context'],job['repeat']]=job
    for key,job in keyed.items():
        mode,task,target,scope,c,j=key
        if scope=='full-last':
            first=keyed[mode,task,target,'full-first',c,j]
            require(job['wrapped_prompt']==first['wrapped_prompt'] and job['prompt_token_ids']==first['prompt_token_ids'],'First/last input identity')
        if j==2:
            first=keyed[mode,task,target,scope,c,1]
            require((job['wrapped_prompt'],job['prompt_token_ids'],job['grammar'])==(first['wrapped_prompt'],first['prompt_token_ids'],first['grammar']),'Repeat input identity')
        if mode=='serial':
            mixed=keyed['mixed',task,target,scope,c,j]
            require(all(job[k]==mixed[k] for k in ('wrapped_prompt','prompt_token_ids','grammar','id_map','output_ids')),'Serial/mixed request identity')
        if mode=='mixed' and c=='near':
            other=keyed[mode,task,target,scope,'random',j]
            require(abs(len(job['prompt_token_ids'])-len(other['prompt_token_ids']))<=24,'Three-tokenizer primary length matching')
    diag_entry=envelope['diagnostics'];path=BASE/diag_entry['file'];require(file_hash(path)==diag_entry['sha256'],'Diagnostic envelope SHA')
    diagnostics=rows(path)
    require(len(diagnostics)==20,'Diagnostic fixed count')
    for job in diagnostics:
        require(job['request_id'] not in jobs,'Diagnostic/science duplicate ID');jobs[job['request_id']]=job
        require(job['prompt_sha256']==digest(job['wrapped_prompt']),'Diagnostic prompt SHA')
        if job.get('guided'):require(job['grammar']==regexp(job['output_ids']) and job['grammar_sha256']==digest(job['grammar']),'Diagnostic grammar')
    require(len(jobs)==4770,'All unique prepared calls')
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
            for key in ('request_id','task','target_id','selected_ordinal','scope','context','repeat','ordinal','block','requested_ids','output_ids','prompt_sha256','grammar_sha256','id_map_sha256'):
                require(r[key]==j[key],f'Canonical frozen identity {model}/{rid}/{key}')
    diagnostic_rows=rows(run/'diagnostics.jsonl');diagblocks=rows(run/'diagnostic_submissions.jsonl')
    diagids=[r for b in diagblocks for r in b['request_ids']]
    require(len(diagids)==len(set(diagids))==20 and set(diagids)=={j['request_id'] for j in diagnostic_jobs},'Diagnostic submissions')
    for r in diagnostic_rows:
        require(r['request_id'] not in canonical,'Repeated diagnostic canonical');canonical[r['request_id']]=r
    require(len(canonical)==len(jobs)==4770,'Canonical completion count')
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
    require(len(effective)==len(proc)==4770 and len(inv)==1025,'Engine and generate count')
    derived={};by_cost=collections.defaultdict(lambda:collections.Counter());bindings=[];all_engine=set();seen_expected=[];ordinary_control=None
    eos=EXPECTED[model][2];stops=EXPECTED[model][3]
    for call in inv:
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
            require(actual['capture_complete'] and not actual['capture_errors'],'Incomplete returned request payload')
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
    require(len(seen_expected)==len(set(seen_expected))==4770 and set(seen_expected)==set(jobs),'Lost/extra/repeated attempts')
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
    require(scientific['requests']==4750 and controls['requests']==20,'Science/control unique counts')
    scientific['item_labels']=sum(v['item_labels'] for mode,v in by_cost.items() if not mode.startswith('diagnostic/'))
    require(scientific['item_labels']==12970,'Scientific returned item-label budget')
    cfg=jread(BASE/f'prepared/runtime_configuration_{model}.json')
    return derived,bindings,{'by_mode':dict(by_cost),'totals':totals,'scientific':scientific,'controls':controls,
         'engine_processed_requests':len(effective),'unresolved_invocations':[],'actual_native_runtime':native,
         'recorded_package_versions':started['runtime_versions'],'frozen_sampling_normalization':cfg['cpu_effective_sampling'],
         'all_science_calls_strict_valid':True,'guided_diagnostic_calls_strict_valid':19,'ordinary_BANANA_is_not_a_science_parse':True,
         'ordinary_control':ordinary_control,
         'fresh_process_pid':started['pid'],'fresh_start_unix':started['started_unix']}

def lookup_from_jobs(jobs,labels,mode):
    result={}
    for j in jobs[mode]:
        key=(j['task'],j['target_id'],j['scope'],j['context'],j['repeat'])
        require(key not in result,'Duplicate logical condition')
        result[key]=labels.get(j['request_id'])
    return result

def get(lookup,row,scope,context,repeat):
    return lookup.get((row['task'],row['target_id'],scope,context,repeat))

def completed(selection,lookup,conditions):
    return [r for r in selection if all(get(lookup,r,*c) is not None for c in conditions)]

def scope_vectors(selection,lookup,scope):
    arrays=[[get(lookup,r,scope,c,j) for r in selection] for c,j in [('near',1),('near',2),('random',1),('random',2)]]
    return vector_for(*arrays,[r['gold'] for r in selection])

def statistics(selection,lookup,main=False):
    n=len(selection);common12=completed(selection,lookup,CONDS[:12]);common14=completed(selection,lookup,CONDS)
    gold=[r['gold'] for r in selection];cg=[r['gold'] for r in common12]
    choices=np.random.default_rng(2026110204).integers(0,len(common12),size=(4000,len(common12))) if common12 else None
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
    choices=np.random.default_rng(2026110204).integers(0,len(common),size=(4000,len(common))) if common else None
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

def transport(selection,lookup):
    details=[]
    for r in selection:
        refs=[get(lookup,r,s,c,j) for s,c,j in [('singleton','none',1),('singleton','none',2),('full-first','random',1),('full-first','random',2)]]
        near=[get(lookup,r,'full-first','near',j) for j in (1,2)]
        valid=all(v is not None for v in refs);passes=valid and len(set(refs))==1
        d={'target_id':r['target_id'],'selected_ordinal':r['selected_ordinal'],'reference_calls_valid':valid,'passes':passes,
           'reference_labels':refs,'near_labels':near,'valid_near_calls':sum(x is not None for x in near),'gold':r['gold']}
        if passes:
            ref=refs[0];d.update(reference_label=ref,reference_is_gold_error=ref!=r['gold'])
            if all(x is not None for x in near):
                d.update(any_flip=any(x!=ref for x in near),stable_flip=all(x!=ref for x in near),
                         wrong_to_right_count_over_two_near_calls=sum(ref!=r['gold'] and x==r['gold'] for x in near),
                         right_to_wrong_count_over_two_near_calls=sum(ref==r['gold'] and x!=r['gold'] for x in near))
        details.append(d)
    passing=[d for d in details if d['passes']];ev=[d for d in passing if d['valid_near_calls']==2]
    return {'selected_n':len(selection),'four_reference_calls_valid_n':sum(d['reference_calls_valid'] for d in details),
            'four_call_pass_n':len(passing),'passing_both_near_calls_valid_n':len(ev),
            'passing_invalid_or_unreturned_near_call_count':sum(2-d['valid_near_calls'] for d in passing),
            'passing_near_validity_counts':{str(k):sum(d['valid_near_calls']==k for d in passing) for k in (0,1,2)},
            'passing_not_evaluable_n':len(passing)-len(ev),'any_flip_count':sum(d['any_flip'] for d in ev),'stable_flip_count':sum(d['stable_flip'] for d in ev),
            'any_flip_fraction':sum(d['any_flip'] for d in ev)/len(ev) if ev else None,'stable_flip_fraction':sum(d['stable_flip'] for d in ev)/len(ev) if ev else None,
            'reference_gold_errors_on_all_passing':sum(d['reference_is_gold_error'] for d in passing),
            'reference_gold_error_fraction_on_all_passing':sum(d['reference_is_gold_error'] for d in passing)/len(passing) if passing else None,
            'reference_gold_errors_on_near_evaluable':sum(d['reference_is_gold_error'] for d in ev),
            'wrong_to_right_count_over_two_near_calls':sum(d['wrong_to_right_count_over_two_near_calls'] for d in ev),
            'right_to_wrong_count_over_two_near_calls':sum(d['right_to_wrong_count_over_two_near_calls'] for d in ev),
            'new_calls_added':0,'reference_calls_already_in_main':4*len(selection),'near_diagnostic_calls_already_in_main':2*len(selection),'details':details}

def compare(ours,theirs,path):
    # Compare our independently produced fields, allowing irrelevant metadata.
    if isinstance(ours,dict):
        require(isinstance(theirs,dict),f'Comparison object {path}')
        for key,value in ours.items():
            if key in ('raw_derived_labels','excess_value_counts'):continue
            require(key in theirs,f'Missing producer field {path}/{key}')
            compare(value,theirs[key],f'{path}/{key}')
    elif isinstance(ours,list):
        require(isinstance(theirs,list) and len(ours)==len(theirs),f'Comparison list {path}')
        for i,(a,b) in enumerate(zip(ours,theirs)):compare(a,b,f'{path}/{i}')
    elif isinstance(ours,float):
        require(isinstance(theirs,(int,float)) and math.isclose(ours,theirs,rel_tol=1e-12,abs_tol=1e-12),f'Numeric mismatch {path}: {ours!r} vs {theirs!r}')
        COMPARISONS['numerical_fields']+=1
    else:
        require(ours==theirs,f'Value mismatch {path}: {ours!r} vs {theirs!r}')
        COMPARISONS['other_fields']+=1

def failure_check():
    attempts=[]
    for d in sorted((BASE/'failed_attempts').iterdir()):
        model=d.name.split('_')[0];run=d/f'run_{model}';failure=jread(run/'FAILURE.json');start=jread(run/'START.json')
        counts={name:len(rows(run/f'{name}.jsonl',True)) for name in ('generation_invocations','raw_return_receipts','generation_processed_bindings','generation_validation','effective_requests','diagnostics','responses')}
        require(not any(counts.values()),f'Failed startup generated outputs {d.name}')
        for sub in ('serial','historical_extra','historical_extra_serial'):
            require(not rows(run/sub/'responses.jsonl',True),'Failed startup has nested outputs')
        account=jread(run/'RAW_RETURN_ACCOUNTING.json')
        require(account['generate_invocations']==account['expected_request_attempts']==account['observed_returned_requests']==0 and not account['unresolved_invocations'],'Failed startup accounting')
        if 'before_rpc' in d.name:
            category='Unix-domain IPC path exceeds 107 characters';require(failure['type']=='ZMQError' and '107 characters' in failure['message'] and "llm=LLM(" in failure['traceback'],'IPC failure location')
        else:
            category='Worker-identity callable rejected by default serializer';require(failure['type']=='TypeError' and 'not serializable' in failure['message'] and 'collective_rpc(worker_identity)' in failure['traceback'],'Serialization failure location')
        attempts.append({'attempt':d.name,'model':model,'category':category,'failure_type':failure['type'],'started_unix':start['started_unix'],
                         'generate_invocations':0,'request_attempts':0,'returned_requests':0,'saved_artifacts_preserved':True,
                         'zero_generation_is_not_zero_startup_GPU_compute':True,'ledger_row_counts':counts})
        for p in d.rglob('*'):
            if p.is_file() and p.suffix in ('.json','.jsonl','.log'):remember(p)
    require(len(attempts)==6 and all(sum(a['model']==m for a in attempts)==2 for m in MODELS),'Six retained startup attempts')
    return attempts

def self_checks():
    truth=[]
    for values in itertools.product((0,1),repeat=4):
        v=vector_for(*[[x] for x in values],[0]);excess=v['excess'][0]
        require(excess in (-1,0,1),'Four-call truth table')
        truth.append({'near1':values[0],'near2':values[1],'random1':values[2],'random2':values[3],**{k:x[0] for k,x in v.items()}})
    require(any(r['excess']==-1 for r in truth),'Negative results retained')
    require(count_condition([0]*127,[0]*128,128)['status']=='abstain','Missing fixed COUNT must abstain')
    require(count_pair([0]*128,[0]*127,[0]*128,128)['status']=='abstain','Missing COUNT comparison must abstain')
    require(quality([1]*4,[0,1,0,1])['balanced_accuracy']==.5,'Constant baseline quality')
    for raw in (' {"1":"Y"}','{"1":"Y"}\n','{"1": "Y"}','{"1":"Y","1":"N"}','{"1":"yes"}','{"1":"Y","2":"N"}'):
        require(parse_label(raw,[1]) is None,'Strict parser rejection')
    require(parse_label('{"2":"N","3":"Y","4":"N","1":"Y"}',[2,3,4,1])=={2:0,3:1,4:0,1:1},'Full-last strict order')
    return {'all_16_binary_four_call_truth_table':truth,'missing_1_of_128_COUNT_abstains':True,'missing_pair_COUNT_abstains':True,
            'constant_baseline_and_strict_parser_checks':True,'negative_excess_preserved':True}

def main():
    started=time.time();write_json('SELF_CHECKS.json',self_checks())
    frozen=jread(BASE/'PREINFERENCE_FROZEN_SHA256.json');freeze_checks=[]
    for rel,expected in frozen['files'].items():
        p=BASE/rel;require(p.is_relative_to(BASE) and '..' not in Path(rel).parts,'Freeze path escapes source')
        actual=file_hash(p);HASHES[rel]=actual
        require(actual==expected,f'Frozen source mutated: {rel}')
        freeze_checks.append(rel)
    targets=rows(BASE/'selection/targets.jsonl');historical=rows(BASE/'historical_context_arm/selection/targets.jsonl')
    source_check=source_selection_check(targets,historical)
    write_json('SOURCE_AND_FREEZE_CHECK.json',{'source_selection':source_check,'frozen_file_count':len(freeze_checks),
                                            'all_frozen_file_hashes_match':True,'freeze_sha256':file_hash(BASE/'PREINFERENCE_FROZEN_SHA256.json')})
    results={};cases=[];all_points=[];all_bindings=[];costs={}
    for model in MODELS:
        print(f'Independent CPU audit: {model}',flush=True)
        jobs,mode_jobs,diagnostic_jobs=prepared_check(model,targets,historical)
        labels,bindings,cost=raw_chain(model,jobs,mode_jobs,diagnostic_jobs)
        all_bindings.extend(bindings);costs[model]=cost
        mixed=lookup_from_jobs(mode_jobs,labels,'mixed');serial=lookup_from_jobs(mode_jobs,labels,'serial')
        maps=rows(BASE/f'prepared/HISTORICAL_SHARED_CALL_MAP_{model}.jsonl');historical_lookups={'mixed':{},'serial':{}}
        for link in maps:
            j=jobs[link['raw_request_id']];require(link['prompt_sha256']==j['prompt_sha256'] and link['grammar_sha256']==j['grammar_sha256'],'Historical shared call map prompt/grammar')
            key=(j['task'],j['target_id'],j['scope'],j['context'],j['repeat'])
            destination=historical_lookups[link['historical_mode']]
            require(key not in destination,'Historical duplicate condition');destination[key]=labels[link['raw_request_id']]
            require(link['uses_shared_primary_observation']==(link['raw_mode'] in ('mixed','serial')),'Historical shared observation marker')
        require(len(maps)==4480 and len(historical_lookups['mixed'])==3584 and len(historical_lookups['serial'])==896,'Historical lookup cardinality')
        model_results={}
        for view,selection,lookup,serial_lookup,prefix in [('primary',targets,mixed,serial,'primary'),('historical',historical,historical_lookups['mixed'],historical_lookups['serial'],'historical'),
              ('original_length_pass',[r for r in historical if not next(t for t in targets if t['task']==r['task'] and t['target_id']==r['target_id'])['glm_primary_random_context_redrawn']],historical_lookups['mixed'],historical_lookups['serial'],'original_length_pass')]:
            own_main={};own_secondary={};points=[]
            for task in TASKS:
                selected=sorted([r for r in selection if r['task']==task],key=lambda r:r['selected_ordinal'])
                subset=[r for r in selected if r['selected_ordinal']<32]
                summary,per_target=statistics(selected,lookup,True);own_main[task]=summary;points.extend(per_target)
                mix_sub,_=statistics(subset,lookup);ser_sub,_=statistics(subset,serial_lookup)
                own_secondary[task]={'reference_transport':transport(selected,lookup),'mixed_subset':mix_sub,'serial_subset':ser_sub,'paired_serial_minus_mixed':paired_modes(subset,lookup,serial_lookup)}
                if view=='primary':
                    values=[r['scopes']['full-first']['excess'] for r in per_target]
                    case=primary(values,task,model,[r['target_id'] for r in per_target],[r['selected_ordinal'] for r in per_target]);cases.append(case)
                    all_points.extend({'model':model,'mode':'primary_mixed',**r} for r in per_target)
                if view=='primary':
                    _,serial_points=statistics(subset,serial_lookup)
                    all_points.extend({'model':model,'mode':'primary_serial',**r} for r in serial_points)
            producer_main=jread(BASE/f'analysis_{model}/{prefix}_main/analysis.json')
            producer_secondary=jread(BASE/f'analysis_{model}/{prefix}_secondary/secondary.json')
            compare(own_main,producer_main['tasks'],f'{model}/{view}/main')
            compare(own_secondary,producer_secondary['tasks'],f'{model}/{view}/secondary')
            producer_points=rows(BASE/f'analysis_{model}/{prefix}_main/per_target.jsonl')
            compare(points,producer_points,f'{model}/{view}/per_target')
            model_results[view]={'main':own_main,'secondary':own_secondary}
        producer_case=jread(BASE/f'analysis_{model}/global_primary_component.json')
        for our_case,their_case in zip([c for c in cases if c['model']==model],producer_case['cases']):
            compare(our_case,their_case,f'{model}/authoritative_primary')
        paired=jread(BASE/f'analysis_{model}/PAIRED_PRIMARY_MINUS_HISTORICAL.json')
        own_paired={task:paired_modes(sorted([r for r in targets if r['task']==task],key=lambda r:r['selected_ordinal']),historical_lookups['mixed'],mixed) for task in TASKS}
        compare(own_paired,paired['tasks'],f'{model}/paired_primary_minus_historical')
        model_results['paired_primary_minus_historical']=own_paired
        results[model]=model_results
        write_json(f'RECOMPUTED_{model}.json',model_results)
        write_json(f'RAW_CHAIN_AND_COSTS_{model}.json',cost)
    attempts=failure_check();write_json('SIX_STARTUP_FAILURES.json',attempts)
    write_rows('INDEPENDENT_RAW_BINDINGS.jsonl',all_bindings);write_rows('INDEPENDENT_PER_TARGET.jsonl',all_points)
    for model in MODELS:
        cases.append({'task':'snli','model':model,'available':False,'status':'Not audited here; awaiting authorized fresh SNLI durable raw',
                      'primary_family_size':9,'positive_primary_evidence':False,'mean_excess':None,'interval':None})
    cases.sort(key=lambda c:(('dblp_acm','boolq','snli').index(c['task']),MODELS.index(c['model'])))
    write_json('NINE_FAMILY_PRIMARY_SIX_OBSERVED.json',{'cases':cases,'observed_and_independently_bound_cases':6,'family_size_remains_nine':True,
                    'confidence':179/180,'quantiles':[1/360,359/360],'draws':4000,'seed_rule':'uint64 big endian first8 SHA256("2026110613|task|model")',
                    'selected_target_finite_bootstrap':True,'exact_population_coverage_claimed':False,'exact_FWER_claimed':False,
                    'legacy_98_75_is_not_new_primary':True,'null_negative_constant_and_unavailable_cases_retained':True})
    write_json('SOURCE_RAW_ANALYSIS_SHA256.json',dict(sorted(HASHES.items())))
    artifacts={p.name:file_hash(p) for p in DEST.iterdir() if p.is_file() and p.name not in ('RESULTS_CERTIFICATE.json','RUN.log')}
    certificate={'status':'PASS for six DBLP/BoolQ fresh cases; SNLI remains unaudited/unavailable in this certificate',
          'read_only_source_bundle':str(BASE),'own_output_directory':str(DEST),'independent_math':True,
          'producer_modules_imported':False,'no_GPU_or_model_inference':True,'all_frozen_source_files_match':True,'frozen_files':len(freeze_checks),
          'source_gold_and_full_fields_verified':source_check,'science_requests':sum(c['scientific']['requests'] for c in costs.values()),
          'control_requests':sum(c['controls']['requests'] for c in costs.values()),'all_request_bindings_verified':len(all_bindings),
          'all_engine_processed_controls_verified':True,'six_failed_startups_preserved_and_zero_generate':True,
          'comparison_counts':dict(COMPARISONS),'comparison_mismatches':0,'null_negative_and_zero_results_retained':True,
          'scientific_input_authority':'Complete durable raw_return_receipts, independently bound by engine ID and exact prepared input tokens/grammar, then canonical payload corroboration',
          'fresh_vs_old':'Only current new_glm_context/run_* durable return payloads are scientific outcomes. Old known results inform the documented design timing; old model outputs were never imported.',
          'primary_rule':'fixed nine-case suite; 4000 draws, case-SHA seeds, common12 selected ordinal order, 179/180 two-sided intervals at 1/360 and 359/360; positive only if lower > 0',
          'inference_limits':['Finite bootstrap of selected fixed targets; no population or exact FWER certificate.',
                              'No independent reexecution of model inference or tokenizer decoding. The audit checks preserved native input arrays, all returned output arrays/text, actual processor controls, and native-worker evidence.',
                              'Zero generate in failed startups excludes generation; it does not establish zero initialization GPU time or energy.',
                              'Historical/subset projections share raw calls and add no inference cost. Serial is secondary and follows mixed.',
                              'No SNLI scientific result is inferred or fabricated. Family size stays nine.'],
          'source_manifest_sha256':file_hash(DEST/'SOURCE_RAW_ANALYSIS_SHA256.json'),'artifact_sha256':artifacts,
          'finished_unix':time.time(),'elapsed_CPU_audit_seconds':time.time()-started}
    write_json('RESULTS_CERTIFICATE.json',certificate)
    print(compact({'certificate':str(DEST/'RESULTS_CERTIFICATE.json'),'comparisons':dict(COMPARISONS),
                   'primary':[{k:c.get(k) for k in ('task','model','common12_complete_n','mean_excess','interval','positive_primary_evidence')} for c in cases if c['available']]}),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source',type=Path,default=BASE);parser.add_argument('--out',type=Path,default=DEST)
    args=parser.parse_args();BASE=args.source.resolve();DEST=args.out.resolve();DEST.mkdir(parents=True,exist_ok=True)
    main()
