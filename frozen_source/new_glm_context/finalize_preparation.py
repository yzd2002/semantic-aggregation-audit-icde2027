#!/usr/bin/env python3
"""Compile the finite extra schedules, shared-call map and exact request budget."""
import random
import time
import argparse
from common import *


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--model',choices=MODELS,default='glm');model=parser.parse_args().model
    changes=json.loads((ROOT/'prepared/REMATCH_REPORT.json').read_text())['changes']
    changed={(r['task'],r['target_id']) for r in changes}
    primary=read_lines(ROOT/'prepared'/f'{model}_requests.jsonl')
    serial=read_lines(ROOT/'prepared'/f'{model}_serial_requests.jsonl')
    historical=read_lines(ROOT/'historical_context_arm/prepared'/f'{model}_requests.jsonl')
    historical_serial=read_lines(ROOT/'historical_context_arm/prepared'/f'{model}_serial_requests.jsonl')
    primary_by={r['request_id']:r for r in primary}
    serial_by={r['request_id']:r for r in serial}
    scope_order={scope:i for i,scope in enumerate((*SCOPES,'singleton'))}
    def extra(rows,seed,mode,is_serial):
        selected=[r for r in rows if (r['task'],r['target_id']) in changed and r['context']=='random']
        selected.sort(key=lambda r:(TASKS.index(r['task']),r['selected_ordinal'],scope_order[r['scope']],r['repeat']))
        random.Random(seed).shuffle(selected)
        result=[]
        for ordinal,r in enumerate(selected):
            result.append({**r,'request_id':'historical-extra/'+r['request_id'],
                'source_historical_request_id':r['request_id'],'execution_mode':mode,
                'ordinal':ordinal,'block':ordinal if is_serial else ordinal//64})
        return result
    extras=extra(historical,2026110604,'historical-extra-mixed',False)
    extras_serial=extra(historical_serial,2026110605,'historical-extra-serial',True)
    assert len(extras)==204 and len(extras_serial)==66
    modes={'mixed':primary,'historical-extra-mixed':extras,'serial':serial,'historical-extra-serial':extras_serial}
    filenames={'mixed':f'{model}_requests.jsonl','historical-extra-mixed':f'{model}_historical_extra_requests.jsonl',
        'serial':f'{model}_serial_requests.jsonl','historical-extra-serial':f'{model}_historical_extra_serial_requests.jsonl'}
    meta_keys=('request_id','task','target_id','selected_ordinal','scope','context','repeat','ordinal','block',
        'source_main_request_id','source_historical_request_id','execution_mode')
    for mode,rows in [('historical-extra-mixed',extras),('historical-extra-serial',extras_serial)]:
        requests_path=ROOT/'prepared'/filenames[mode]
        if requests_path.exists():assert read_lines(requests_path)==rows
        else:write_lines(requests_path,rows)
        schedule_path=ROOT/'prepared'/(mode+'_schedule.jsonl')
        schedule=[{k:r[k] for k in meta_keys if k in r} for r in rows]
        if schedule_path.exists():assert read_lines(schedule_path)==schedule
        else:write_lines(schedule_path,schedule)
        membership=[[r['request_id'] for r in rows if r['block']==b] for b in sorted({r['block'] for r in rows})]
        write_json(ROOT/'prepared'/(mode+'_block_membership.json'),membership)
    extra_by={r['source_historical_request_id']:r for r in extras}
    extra_serial_by={r['source_historical_request_id']:r for r in extras_serial}
    shared=[]
    for mode,rows,ordinary,extra_lookup in [('mixed',historical,primary_by,extra_by),
                                         ('serial',historical_serial,serial_by,extra_serial_by)]:
        for row in rows:
            is_extra=row['request_id'] in extra_lookup
            source=extra_lookup[row['request_id']] if is_extra else ordinary[row['request_id']]
            for key in ('wrapped_prompt','prompt_token_ids','requested_ids','output_ids','grammar','id_map'):
                assert row[key]==source[key],(row['request_id'],key)
            shared.append({'historical_mode':mode,'historical_request_id':row['request_id'],
                'raw_mode':('historical-extra-'+mode) if is_extra else mode,
                'raw_request_id':source['request_id'],'uses_shared_primary_observation':not is_extra,
                'prompt_sha256':row['prompt_sha256'],'grammar_sha256':row['grammar_sha256']})
    write_lines(ROOT/'prepared'/f'HISTORICAL_SHARED_CALL_MAP_{model}.jsonl',shared)
    diagnostics=read_lines(ROOT/'prepared'/f'{model}_diagnostic_requests.jsonl')
    envelope={'created_unix':time.time(),'status':'CPU-prepared; root review and explicit GPU GO still required',
        'model':model,'design_timing':'New three-family follow-up designed with earlier Qwen/Smol context and Qwen/Smol/GLM decision results known; no new context judgments used.',
        'execution_order':list(modes),'primary_target_counts':{task:128 for task in TASKS},
        'serial_target_counts':{task:32 for task in TASKS},'historical_changed_random_targets':34,
        'diagnostics':{'calls':len(diagnostics),'input_tokens':sum(len(r['prompt_token_ids']) for r in diagnostics),
            'file':f'prepared/{model}_diagnostic_requests.jsonl','sha256':sha(ROOT/'prepared'/f'{model}_diagnostic_requests.jsonl')},
        'modes':{},'sampling':json.loads((ROOT/'prepared'/f'runtime_configuration_{model}.json').read_text())['requested_sampling'],
        'literal_complete_features_no_truncation':True,'unique_scientific_calls':sum(map(len,modes.values())),
        'unique_all_calls':sum(map(len,modes.values()))+len(diagnostics),
        'unique_scientific_item_labels':sum(len(r['output_ids']) for rows in modes.values() for r in rows),
        'all_input_tokens':sum(len(r['prompt_token_ids']) for rows in modes.values() for r in rows)+sum(len(r['prompt_token_ids']) for r in diagnostics),
        'max_all_output_tokens_hard_cap':128*(sum(map(len,modes.values()))+len(diagnostics)),
        'shared_observation_note':'Historical analysis views reuse unique primary raw calls according to HISTORICAL_SHARED_CALL_MAP; copied/projected rows are not new calls or independent samples.',
        'main_primary_criterion':'All six fresh full-first contrasts retain 98.75% lower endpoint > 0 as a conservative inherited rule; no six-comparison FWER claim.',
        'paired_arm_and_serial_and_length_pass_results':'secondary descriptive, unchanged functions, 95% intervals'}
    for mode,rows in modes.items():
        envelope['modes'][mode]={'calls':len(rows),'item_labels':sum(len(r['output_ids']) for r in rows),
            'input_tokens':sum(len(r['prompt_token_ids']) for r in rows),
            'max_prompt_tokens':max(len(r['prompt_token_ids']) for r in rows),
            'output_token_cap_per_request':128,'requests_per_submission':1 if 'serial' in mode else 64,
            'file':'prepared/'+filenames[mode],'sha256':sha(ROOT/'prepared'/filenames[mode])}
    assert envelope['unique_scientific_calls']==4750 and envelope['unique_all_calls']==4770
    write_json(ROOT/'prepared'/f'REQUEST_ENVELOPE_{model}.json',envelope)
    print(compact(envelope),flush=True)


if __name__=='__main__':main()
