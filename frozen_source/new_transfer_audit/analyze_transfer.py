#!/usr/bin/env python3
"""Frozen 36-policy transfer analysis. Missing/invalid outcomes never shrink denominators."""
import argparse, collections
from fractions import Fraction
from common import *
from decision_metrics import complete,disagreement,gold_metrics,evaluation,empirical_indicator,arithmetic_break_even
def policy_id(p):return '/'.join(p[k] for k in ('model','task','tolerance','selector'))

def optional(path):return read_lines(path) if Path(path).exists() else []

def sum_known(values):
    return None if any(v is None for v in values) else sum(values)

def cost(jobs,returned,submitted=None):
    submitted=set(returned) if submitted is None else set(submitted)
    present=[returned[j['request_id']] for j in jobs if j['request_id'] in returned]
    result={'planned_calls':len(jobs),'submitted_calls':sum(j['request_id'] in submitted for j in jobs),
        'returned_calls':len(present),'planned_requested_items':sum(len(j.get('occurrence_ids',[])) for j in jobs)}
    for kind in ('input','output'):
        key='actual_'+kind+'_tokens';vs=[r.get(key) for r in present]
        known=[v for v in vs if type(v) is int and v>=0]
        result[key]=sum(known) if len(known)==len(jobs) else None
        result['observed_'+kind+'_token_lower_bound']=sum(known)
        result['unknown_'+kind+'_token_call_count']=len(jobs)-len(known)
    result['actual_total_tokens']=sum_known([result['actual_input_tokens'],result['actual_output_tokens']])
    result['observed_total_token_lower_bound']=result['observed_input_token_lower_bound']+result['observed_output_token_lower_bound']
    result['token_cost_complete']=result['actual_total_tokens'] is not None
    return result

def plus_calibration(c,cal):
    return {'planned_calls':cal['planned_calls']+c['planned_calls'],
        'returned_calls':cal['returned_calls']+c['returned_calls'],
        'planned_requested_items':cal['planned_requested_items']+c['planned_requested_items'],
        **{k:sum_known([cal[k],c[k]]) for k in ('actual_input_tokens','actual_output_tokens','actual_total_tokens')},
        'unknown_execution_input_token_call_count':c['unknown_input_token_call_count'],
        'unknown_execution_output_token_call_count':c['unknown_output_token_call_count'],
        'calibration_counted_once':True}

def journal_cost(entries):
    returns=[r for e in entries for r in e['actual_returns']]
    out={'returned_engine_records':len(returns),'submitted_batch_expected_records':sum(len(e['expected_batch_request_ids']) for e in entries),
        'batches_with_return_count_mismatch':sum(len(e['expected_batch_request_ids'])!=e['actual_return_count'] for e in entries)}
    out['expected_records_without_observed_return']=max(0,out['submitted_batch_expected_records']-len(returns))
    out['extra_return_records_beyond_expected_count']=max(0,len(returns)-out['submitted_batch_expected_records'])
    for kind in ('input','output'):
        values=[r.get('actual_'+kind+'_tokens') for r in returns];known=[v for v in values if type(v) is int and v>=0]
        out['observed_'+kind+'_token_sum_from_actual_returns']=sum(known)
        out['unknown_'+kind+'_token_return_count']=len(values)-len(known)
    out['interpretation']='Literal observed tokens from every actual returned engine record, including unmatched or invalid records. Unreturned submitted requests retain unknown additional usage; these sums are not a fabricated complete cost.'
    return out

def load_trace(model):
    run=ROOT/f'run_{model}';schedule=read_lines(ROOT/'prepared'/f'{model}_schedule.jsonl')
    jobs={j['request_id']:j for j in schedule};requests={}
    # Keep prompt token provenance without retaining every large full prompt in memory.
    with (ROOT/'prepared'/f'{model}_requests.jsonl').open() as f:
        for line in f:
            j=json.loads(line);rid=j['request_id'];assert all(j[k]==v for k,v in jobs[rid].items())
            requests[rid]={k:j[k] for k in ('requested_ids','output_ids','prompt_sha256','grammar_sha256','id_map_sha256')}
            requests[rid].update(input_token_count=len(j['prompt_token_ids']),input_token_ids_sha256=digest(compact(j['prompt_token_ids'])))
    submissions=optional(run/'block_submissions.jsonl');submitted_list=[]
    for ix,b in enumerate(submissions):
        expected=[j['request_id'] for j in schedule[ix*64:(ix+1)*64]]
        assert b['block']==ix and b['request_ids']==expected
        assert b['first_ordinal']==ix*64 and b['last_ordinal']==ix*64+len(expected)-1
        submitted_list.extend(b['request_ids'])
    assert len(set(submitted_list))==len(submitted_list)
    submitted_set=set(submitted_list)
    returned={};vectors={};invalid=[];engine_ids=set()
    for j in schedule:
        vectors.setdefault(j['relation_id'],{}).setdefault(j['configuration_id'],{}).setdefault(j['repeat'],[None]*128)
    for r in optional(run/'responses.jsonl'):
        rid=r['request_id'];assert rid in jobs and rid in submitted_set and rid not in returned
        j=jobs[rid];p=requests[rid]
        for k,v in j.items():assert r.get(k)==v,(rid,k)
        for k in ('requested_ids','output_ids','prompt_sha256','grammar_sha256','id_map_sha256'):assert r.get(k)==p[k]
        assert r['diagnostic'] is False
        n=r.get('actual_input_tokens');assert n is None or n==p['input_token_count']
        if n is not None:assert r['input_token_ids_sha256']==p['input_token_ids_sha256']
        n=r.get('actual_output_tokens');assert n is None or n==len(r['output_token_ids'])
        parsed=parse(r['raw'],p['output_ids']);saved={int(k):v for k,v in r['labels'].items()} if r.get('labels') is not None else None
        assert parsed==saved and r['strict_valid']==(parsed is not None)
        assert r['engine_request_id'] not in engine_ids;engine_ids.add(r['engine_request_id'])
        returned[rid]=r
        if parsed is None:invalid.append(rid)
        for i in j['occurrence_ids']:vectors[j['relation_id']][j['configuration_id']][j['repeat']][i-1]=parsed[i] if parsed is not None else None
    diag=optional(run/'diagnostics.jsonl');diag_jobs=read_lines(ROOT/'prepared'/f'{model}_controls.jsonl')
    diag_sub=optional(run/'diagnostic_submissions.jsonl');control_submitted=[rid for b in diag_sub for rid in b['request_ids']]
    assert len(control_submitted)==len(set(control_submitted))
    assert control_submitted==[j['request_id'] for j in diag_jobs][:len(control_submitted)]
    diag_byid={r['request_id']:r for r in diag};assert len(diag_byid)==len(diag);invalid_controls=[]
    for j in diag_jobs:
        if j['request_id'] not in diag_byid:continue
        r=diag_byid[j['request_id']];assert r['diagnostic'] is True and r['wrapped_prompt']==j['wrapped_prompt']
        assert r['prompt_sha256']==j['prompt_sha256'] and r['output_ids']==j['output_ids'] and r['grammar']==j.get('grammar')
        if r.get('actual_input_tokens') is not None:assert r['actual_input_tokens']==len(j['prompt_token_ids'])
        if r.get('actual_output_tokens') is not None:assert r['actual_output_tokens']==len(r['output_token_ids'])
        assert r['input_token_ids_sha256']==digest(compact(j['prompt_token_ids']))
        parsed=parse(r['raw'],j['output_ids']) if j['output_ids'] else None
        saved={int(k):v for k,v in r['labels'].items()} if r.get('labels') is not None else None
        assert parsed==saved and r['strict_valid']==(parsed is not None)
        assert r['engine_request_id'] not in engine_ids;engine_ids.add(r['engine_request_id'])
        if (j['guided'] and parsed is None) or (not j['guided'] and 'BANANA' not in r['raw']):invalid_controls.append(j['request_id'])
    journals=optional(run/'engine_returns.jsonl');journal_by_engine={};science_journals=[];control_journals=[]
    for entry in journals:
        assert entry['journal_written_before_count_identity_or_parser_validation'] is True
        assert entry['actual_return_count']==len(entry['actual_returns'])
        if entry['stage']=='science':
            ix=entry['block'];assert entry['expected_batch_request_ids']==[j['request_id'] for j in schedule[ix*64:(ix+1)*64]]
            assert all(rid in submitted_set for rid in entry['expected_batch_request_ids'])
            science_journals.append(entry)
        else:
            name=entry['stage'].removeprefix('diagnostic/')
            assert entry['expected_batch_request_ids']==[j['request_id'] for j in diag_jobs if j['name']==name]
            assert all(rid in control_submitted for rid in entry['expected_batch_request_ids'])
            control_journals.append(entry)
        for actual in entry['actual_returns']:
            eid=actual['engine_request_id']
            # Ambiguous engine IDs are retained as failure evidence, never assigned to targets.
            journal_by_engine.setdefault(eid,[]).append(actual)
            ids=actual['actual_prompt_token_ids']
            assert actual['actual_input_tokens']==(len(ids) if ids is not None else None)
            outs=actual['outputs']
            computed=sum(len(o['output_token_ids']) for o in outs) if outs and all(o['output_token_ids'] is not None for o in outs) else None
            assert computed==actual['actual_output_tokens']
    # Validated target rows must each match their already persisted native return.
    for r in list(returned.values())+diag:
        candidates=journal_by_engine.get(r['engine_request_id'],[]);assert len(candidates)==1
        actual=candidates[0];assert len(actual['outputs'])==1
        answer=actual['outputs'][0]
        assert r['raw']==answer['raw'] and r['output_token_ids']==answer['output_token_ids']
        assert r['input_token_ids_sha256']==digest(compact(actual['actual_prompt_token_ids']))
    summary=json.loads((run/'summary.json').read_text()) if (run/'summary.json').exists() else None
    complete_run=len(returned)==len(jobs) and len(submitted_list)==len(jobs)
    if summary:
        assert summary['status']=='complete' and summary['model']==model and complete_run and len(diag)==20
        assert not invalid_controls
        assert summary['response_sha256']==sha(run/'responses.jsonl')
        assert summary['engine_return_journal_sha256']==sha(run/'engine_returns.jsonl')
        assert summary['scientific_calls']==len(returned) and summary['invalid_scientific_calls']==len(invalid)
        assert summary['valid_scientific_calls']==len(returned)-len(invalid) and summary['diagnostic_calls']==20 and summary['all_calls']==len(returned)+20
        blocks=optional(run/'blocks.jsonl')
        assert [(b['block'],b['requests']) for b in blocks]==[(b['block'],len(b['request_ids'])) for b in submissions]
        assert len(science_journals)==len(submissions) and len(control_journals)==3
        assert summary['blocks_sha256']==sha(run/'blocks.jsonl')
        for kind in ('input','output'):
            science_tokens=sum_known([r.get('actual_'+kind+'_tokens') for r in returned.values()])
            control_tokens=sum_known([r.get('actual_'+kind+'_tokens') for r in diag])
            assert summary['scientific_observed_tokens'][kind+'_tokens']==science_tokens
            assert summary['diagnostic_observed_tokens'][kind+'_tokens']==control_tokens
            assert summary['all_observed_tokens'][kind+'_tokens']==sum_known([science_tokens,control_tokens])
        start=json.loads((run/'START.json').read_text());data=json.loads((run/'DATA_START.json').read_text())
        assert start['model']==data['model']==model and start['planned_scientific_calls']==len(jobs)
        assert start['freeze_sha256']==sha(ROOT/'PREINFERENCE_FROZEN_SHA256.json')
        assert start['request_sha256']==sha(ROOT/'prepared'/f'{model}_requests.jsonl')
        assert data['control_sha256']==sha(run/'CONTROL_RESULT.json')
        control_result=json.loads((run/'CONTROL_RESULT.json').read_text())
        assert control_result['status']=='PASS' and control_result['diagnostic_sha256']==sha(run/'diagnostics.jsonl') and control_result['diagnostic_calls']==20
        assert [rid for b in diag_sub for rid in b['request_ids']]==[j['request_id'] for j in diag_jobs]
        assert start['started_unix']<=max(b['submitted_unix'] for b in diag_sub)<=data['time_unix']
        assert all(data['time_unix']<=b['submitted_unix']<=summary['completed_unix'] for b in submissions)
    return schedule,returned,set(submitted_list),vectors,{'phase_complete':complete_run,'invalid_request_ids':invalid,
        'missing_request_ids':[rid for rid in jobs if rid not in returned],
        'science_union':cost(schedule,returned,submitted_list),'controls':cost(diag_jobs,diag_byid,control_submitted),
        'invalid_control_request_ids':invalid_controls,
        'control_gate_status':json.loads((run/'CONTROL_RESULT.json').read_text())['status'] if (run/'CONTROL_RESULT.json').exists() else 'not_reached_or_incomplete',
        'all_native_returns_science_journal':journal_cost(science_journals),
        'all_native_returns_control_journal':journal_cost(control_journals),
        'unmapped_scientific_native_returns':sum(e['actual_return_count'] for e in science_journals)-len(returned),
        'unmapped_control_native_returns':sum(e['actual_return_count'] for e in control_journals)-len(diag),
        'science_submitted_requests_without_return_journal':len(submitted_list)-sum(len(e['expected_batch_request_ids']) for e in science_journals),
        'control_submitted_requests_without_return_journal':len(control_submitted)-sum(len(e['expected_batch_request_ids']) for e in control_journals),
        'engine_return_journal_sha256':sha(run/'engine_returns.jsonl') if (run/'engine_returns.jsonl').exists() else None,
        'raw_sha256':sha(run/'responses.jsonl') if (run/'responses.jsonl').exists() else None,
        'failure':json.loads((run/'FAILURE.json').read_text()) if (run/'FAILURE.json').exists() else None}

def mean_all(values):return sum(values)/len(values) if values and all(v is not None for v in values) else None

def relation_summary(replays,tau=None):
    losses=[replays[str(r)]['singleton_fidelity'] for r in (1,2)]
    mean=mean_all([v['value'] if v else None for v in losses])
    both=None if any(v is None or not v['complete'] for v in losses) else all(empirical_indicator(v,tau) for v in losses) if tau is not None else None
    # MEAN uses exact integer loss over 2*128; BOTH requires both 128-item replays.
    within_mean=None if tau is None or mean is None else tau.denominator*sum(v['numerator'] for v in losses)<=tau.numerator*256
    metrics=('error','accuracy','balanced_accuracy','f1','absolute_ratio_error','predicted_positive_fraction','signed_count_bias','canceled_gold_error_mass')
    gold={key:mean_all([replays[str(r)]['gold'][key] for r in (1,2)]) for key in metrics}
    gold['mean_confusion_counts']={key:mean_all([replays[str(r)]['gold']['confusion_counts_on_valid_outcomes'][key]
        if replays[str(r)]['gold']['complete'] else None for r in (1,2)]) for key in ('tp','tn','fp','fn')}
    return {'mean_S_disagreement':mean,'within_tolerance_MEAN':within_mean,'within_tolerance_BOTH':both,
        'public_gold_mean_over_two_replays':gold,'replays_are_execution_diagnostics_not_independent_calibrations':True}

def aggregate_relations(rels):
    rows=list(rels.values());mean_loss=mean_all([x['summary']['mean_S_disagreement'] for x in rows])
    metrics=('error','accuracy','balanced_accuracy','f1','absolute_ratio_error','predicted_positive_fraction','signed_count_bias','canceled_gold_error_mass')
    result={'relation_count_fixed_denominator':len(rows),'mean_S_disagreement_over_relations':mean_loss,
        'public_gold_relation_macro_mean':{key:mean_all([x['summary']['public_gold_mean_over_two_replays'][key] for x in rows]) for key in metrics},
        'public_gold_relation_macro_confusion_counts':{key:mean_all([x['summary']['public_gold_mean_over_two_replays']['mean_confusion_counts'][key] for x in rows]) for key in ('tp','tn','fp','fn')}}
    for name in ('MEAN','BOTH'):
        vals=[x['summary']['within_tolerance_'+name] for x in rows]
        result['tolerance_'+name]={'pass':sum(v is True for v in vals),'fail':sum(v is False for v in vals),'unknown':sum(v is None for v in vals),
            'fixed_relation_denominator':len(rows),'pass_fraction':sum(v is True for v in vals)/len(rows) if rows and all(v is not None for v in vals) else None}
    return result

def prefix_costs(policy,relation_ids,executed,schedule,returned,submitted):
    cal=policy['observed_standalone_calibration_cost'];prefixes=[]
    for count in range(1,len(relation_ids)+1):
        selected=set(relation_ids[:count]);row={'prefix_relations':count,'relation_ids':relation_ids[:count],'replays':{}}
        for repeat in (1,2):
            subset=[j for j in schedule if j['relation_id'] in selected and j['repeat']==repeat]
            exe=cost([j for j in subset if j['configuration_id']==executed],returned,submitted)
            baseline=cost([j for j in subset if j['execution_type']=='P'],returned,submitted)
            total=plus_calibration(exe,cal)
            row['replays'][str(repeat)]={'execution':exe,'old_calibration_once_plus_execution':total,'independent_P_baseline':baseline,
                'planned_calls_difference_vs_P':total['planned_calls']-baseline['planned_calls'],
                'calls_amortized':total['planned_calls']<=baseline['planned_calls'],
                'observed_tokens_difference_vs_P':None if total['actual_total_tokens'] is None or baseline['actual_total_tokens'] is None else total['actual_total_tokens']-baseline['actual_total_tokens'],
                'tokens_amortized':None if total['actual_total_tokens'] is None or baseline['actual_total_tokens'] is None else total['actual_total_tokens']<=baseline['actual_total_tokens']}
        prefixes.append(row)
    return prefixes

def fixed_prefix_costs(relation_ids,config,schedule,returned,submitted):
    prefixes=[]
    for count in range(1,len(relation_ids)+1):
        selected=set(relation_ids[:count]);row={'prefix_relations':count,'calibration_calls':0,'replays':{}}
        for repeat in (1,2):
            current=[j for j in schedule if j['relation_id'] in selected and j['repeat']==repeat]
            exe=cost([j for j in current if j['configuration_id']==config],returned,submitted)
            baseline=cost([j for j in current if j['execution_type']=='P'],returned,submitted)
            row['replays'][str(repeat)]={'execution_cost':exe,'independent_P_baseline':baseline,
                'planned_calls_difference_vs_P':exe['planned_calls']-baseline['planned_calls'],
                'observed_tokens_difference_vs_P':None if exe['actual_total_tokens'] is None or baseline['actual_total_tokens'] is None else exe['actual_total_tokens']-baseline['actual_total_tokens']}
        prefixes.append(row)
    return prefixes

def analyze(model,out):
    schedule,returned,submitted,vectors,ledger=load_trace(model)
    all_policies=json.loads((ROOT/'POLICY_LOCK.json').read_text())['payload']['policies']
    policies=[p for p in all_policies if p['model']==model];assert len(policies)==18
    rawgold=read_lines(ROOT/'inputs/gold.jsonl');source=read_lines(ROOT/'inputs/relations.jsonl')
    gold={};source_lookup={(r['relation_id'],r['occurrence_id']):r for r in source}
    for relation in sorted(vectors):
        gs=sorted([r for r in rawgold if r['relation_id']==relation],key=lambda r:r['occurrence_id'])
        assert [r['occurrence_id'] for r in gs]==list(range(1,129))
        assert all(r['source_id']==source_lookup[relation,r['occurrence_id']]['source_id'] for r in gs)
        gold[relation]=[r['gold'] for r in gs]
    evaluated={};per_target=[]
    for relation,configs in vectors.items():
        evaluated[relation]={}
        for config,reps in configs.items():
            replays={}
            for repeat in (1,2):
                pred=reps[repeat];ref=configs['singleton'][repeat]
                replays[str(repeat)]={**evaluation(pred,ref,gold[relation],reference_only=config=='singleton'),
                    'execution_cost':cost([j for j in schedule if (j['relation_id'],j['configuration_id'],j['repeat'])==(relation,config,repeat)],returned,submitted)}
                for i,(p,s,g) in enumerate(zip(pred,ref,gold[relation]),1):
                    per_target.append({'model':model,'relation_id':relation,'configuration_id':config,'repeat':repeat,'occurrence_id':i,
                        'source_id':source_lookup[relation,i]['source_id'],'prediction':p,'common_S_reference':s,'public_gold':g})
            evaluated[relation][config]={'replays':replays,'summary':relation_summary(replays)}
    results={}
    for p in policies:
        pid=policy_id(p);task=p['task'];relation_ids=sorted(k for k in vectors if k.startswith(task+'/'))
        logical=p['chosen']['configuration_id'];exe='singleton_execution' if logical=='singleton' else logical
        tau=Fraction(p['tolerance_numerator'],p['tolerance_denominator']);rels={}
        for relation in relation_ids:
            item=evaluated[relation][exe]
            policy_replays={repeat:{**entry,'empirical_tolerance_satisfied':empirical_indicator(entry['singleton_fidelity'],tau)} for repeat,entry in item['replays'].items()}
            rels[relation]={'replays':policy_replays,'summary':relation_summary(policy_replays,tau)}
        results[pid]={'policy':p,'executed_configuration_id':exe,'relations':rels,'aggregate':aggregate_relations(rels),
            'all_prefix_costs':prefix_costs(p,relation_ids,exe,schedule,returned,submitted),
            'call_break_even_arithmetic':arithmetic_break_even(p['chosen']['k']),
            'calibration_counted_once_per_prefix':True,'new_calibration_calls':0}
    fixed={}
    for task in TASKS:
        fixed[task]={}
        for label,config in [('P_baseline','singleton_execution'),('random_k8_C','k8/random/C'),('diversity_k8_C','k8/diversity/C'),('S_reference','singleton')]:
            ids=sorted(k for k in vectors if k.startswith(task+'/'));rels={r:evaluated[r][config] for r in ids}
            fixed[task][label]={'calibration_calls':0,'configuration_id':config,'relations':rels,'aggregate':aggregate_relations(rels),
                'all_prefix_execution_costs':fixed_prefix_costs(ids,config,schedule,returned,submitted)}
        fixed[task]['zero_call_gold_constants']={r:{'Y':gold_metrics([1]*128,gold[r]),'N':gold_metrics([0]*128,gold[r])} for r in ids}
    contrasts={}
    for task in TASKS:
        for tol in ('0.05','0.10','0.20'):
            a=results[f'{model}/{task}/{tol}/Input'];b=results[f'{model}/{task}/{tol}/Output'];rows={}
            for relation in a['relations']:
                rows[relation]={}
                for repeat in ('1','2'):
                    av=a['relations'][relation]['replays'][repeat];bv=b['relations'][relation]['replays'][repeat]
                    af=av['singleton_fidelity'];bf=bv['singleton_fidelity']
                    rows[relation][repeat]={'output_minus_input_S_disagreement':None if not af['complete'] or not bf['complete'] else bf['value']-af['value'],
                        'output_minus_input_execution_calls':bv['execution_cost']['planned_calls']-av['execution_cost']['planned_calls']}
            contrasts[task+'/'+tol]={'same_locked_plan':a['executed_configuration_id']==b['executed_configuration_id'],'per_relation_replay':rows}
    lock_payload=json.loads((ROOT/'POLICY_LOCK.json').read_text())['payload']
    result={'schema_version':1,'new_real_replication':True,'old_transfer_raw_lost':True,'prior_23_relation_aggregate_results_known_before_this_run':True,
        'model':model,'policy_count':18,'policy_lock_sha256':sha(ROOT/'POLICY_LOCK.json'),
        'all_policies':results,'fixed_zero_calibration_references':fixed,'all_executed_configurations':evaluated,
        'output_minus_input':contrasts,'physical_execution_union_cost':ledger,
        'original_development_union_cost_from_immutable_lock':lock_payload['development_research_costs'][model],
        'new_calibration_calls':0,
        'scope':'Fixed finite panel and execution replays. Not unseen-outcome first research, population inference, IID studies, or a SOTA claim.',
        'design_boundary':'.05/.10 policies cannot amortize their original 1152 calibration calls within this panel: DBLP has only 3 relations and most locked choices are singleton; BoolQ .05/.10 are singleton. Outcome cannot change this call arithmetic.',
        'accounting':'Each prefix counts original observed calibration once plus one execution replay. Shared roles reuse physical calls; tolerances and identical plans never imply independent calibration studies. In the original locks 8/12 Input-vs-Output pairs choose the same plan. P/S/replay physical calls remain distinct. Invalid and missing data retain fixed denominators; unknown token usage is null, not zero.',
        'relations_sha256':sha(ROOT/'inputs/relations.jsonl'),'gold_sha256':sha(ROOT/'inputs/gold.jsonl'),
        'freeze_sha256':sha(ROOT/'PREINFERENCE_FROZEN_SHA256.json') if (ROOT/'PREINFERENCE_FROZEN_SHA256.json').exists() else None}
    out=Path(out);out.mkdir(exist_ok=False);write_json(out/'analysis.json',result);write_lines(out/'per_target.jsonl',per_target)
    write_json(out/'ANALYSIS_SHA256.json',{name:sha(out/name) for name in ('analysis.json','per_target.jsonl')})
    print(compact({'model':model,'policy_count':18,'phase_complete':ledger['phase_complete'],'analysis':str(out/'analysis.json')}))
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',choices=MODELS,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();analyze(a.model,a.out)
