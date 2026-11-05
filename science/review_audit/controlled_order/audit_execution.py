#!/usr/bin/env python3
"""Post-run independent completeness and retained-provenance audit."""
import argparse,json
from common import ROOT,read,sha,write_json

def audit(model):
    folder=ROOT/f'run_{model}';load=lambda name:json.loads((folder/name).read_text())
    summary=load('summary.json');start=load('START.json');data=load('DATA_START.json')
    records=read(folder/'responses.jsonl');blocks=read(folder/'blocks.jsonl');submitted=read(folder/'block_submissions.jsonl')
    planned=read(ROOT/'selection/schedule.jsonl')
    assert len(records)==len(planned)==summary['jobs']==summary['valid_jobs']==11894
    assert summary['invalid_jobs']==0 and summary['status']=='complete'
    assert len(blocks)==len(submitted)==186
    assert len({r['engine_request_id'] for r in records})==len(records)
    assert all(r['ordinal']==i and r['request_id']==planned[i]['request_id'] for i,r in enumerate(records))
    for i,(block,submit) in enumerate(zip(blocks,submitted)):
        assert block['block']==submit['block']==i
        selected=records[64*i:64*(i+1)]
        assert block['requests']==len(selected)
        assert block['first_ordinal']==submit['first_ordinal']==64*i
        assert block['last_ordinal']==submit['last_ordinal']==64*i+len(selected)-1
        assert all(r['block']==i for r in selected)
        assert data['time_unix']<=submit['submitted_unix']<=block['completed_unix']
    controls=load('negative_controls.json');smoke=load('synthetic_smoke.json')
    assert controls['valid'] and all(x['valid'] for x in controls['checks'])
    assert smoke['valid'] and all(x['valid'] for x in smoke['records'])
    assert sha(folder/'negative_controls.json')==data['negative_control_sha256']
    assert sha(folder/'synthetic_smoke.json')==data['smoke_sha256']
    assert sha(folder/'responses.jsonl')==summary['response_sha256']
    assert sha(folder/'blocks.jsonl')==summary['block_sha256']
    assert sha(ROOT/'prepared'/f'{model}_requests.jsonl')==start['request_sha256']
    assert sha(ROOT/'RUNNER_V4_FROZEN_SHA256.json')==start['runtime_amendment_sha256']
    for kind in ('input','output'):
        assert summary['actual_tokens'][kind+'_tokens']==sum(r['actual_'+kind+'_tokens'] for r in records)
    result={'model':model,'status':'PASS','requests':len(records),'submitted_blocks':len(submitted),
        'completed_blocks':len(blocks),'unique_engine_request_ids':len(records),
        'unreturned_submitted_requests':0,'invalid_requests':0,
        'controls_and_smoke_passed':True,'actual_tokens':summary['actual_tokens'],
        'mixed_workload_seconds':summary['mixed_workload_seconds'],
        'response_sha256':summary['response_sha256'],
        'scope':'Independent post-run completeness, submission, control, token and retained-file checks; raw parsing/statistics are checked separately.'}
    write_json(folder/'EXECUTION_AUDIT.json',result)
    print('PASS complete submitted/returned execution, controls, retained provenance and actual tokens:',model)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',choices=['qwen','smol'],required=True);audit(p.parse_args().model)
