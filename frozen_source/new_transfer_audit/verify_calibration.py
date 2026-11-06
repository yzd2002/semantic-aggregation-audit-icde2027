#!/usr/bin/env python3
"""Recheck the old DEV raw and all 36 fixed policies, without fitting anything new."""
import importlib.util
from common import *

def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod

def verify():
    original=ROOT.parent/'reconstruction/decision_audit'
    trace=module('restored_decision_trace',ROOT/'source_reference/decision_trace_records.py')
    selector=module('restored_decision_selector',ROOT/'source_reference/decision_selector.py')
    lock=json.loads((ROOT/'POLICY_LOCK.json').read_text());policies=lock['payload']['policies'];checks=[];manifests={}
    assert sha(ROOT/'POLICY_LOCK.json')==sha(original/'POLICY_LOCK.json')
    for model in MODELS:
        dev=trace.load_trace(original,'dev',model,require_complete=True)
        manifests[model]={'raw_sha256':dev['raw_sha256'],'submissions_sha256':dev['submissions_sha256'],
            'summary_sha256':dev['summary_sha256'],'control_evidence':dev['control_evidence']}
        for task in TASKS:
            regenerated=selector.select_all_tolerances(dev['vectors'][task])
            for tolerance,_ in selector.TOLERANCES:
                for which in ('Input','Output','Repeat'):
                    p=next(p for p in policies if (p['model'],p['task'],p['tolerance'],p['selector'])==(model,task,tolerance,which))
                    expected=regenerated[tolerance][which]
                    assert all(p[k]==v for k,v in expected.items())
                    cost=dev['ledger']['by_task'][task]['selector_calibration'][which]
                    assert p['observed_standalone_calibration_cost']==cost
                    assert cost['planned_calls']==cost['returned_calls']==1152
                    assert all(type(cost[k]) is int and cost[k]>0 for k in ('actual_input_tokens','actual_output_tokens','actual_total_tokens'))
                    checks.append({'model':model,'task':task,'tolerance':tolerance,'selector':which,
                        'chosen':p['chosen'],'verified_calibration_calls':1152,'observed_calibration_cost':cost})
    assert len(checks)==36
    return {'status':'PASS','policy_count':36,'original_dev_raw_reverified':True,
        'policy_lock_sha256':sha(ROOT/'POLICY_LOCK.json'),'development_evidence':manifests,'checks':checks,
        'new_calibration_calls':0,'shared_trace_warning':'36 policies reuse overlapping DEV traces and three tolerances. Identical plans are not independent calibration studies.'}

if __name__=='__main__':
    result=verify();write_json(ROOT/'verification/CALIBRATION_RECHECK.json',result);print(compact({k:result[k] for k in ('status','policy_count','original_dev_raw_reverified')}))
