#!/usr/bin/env python3
"""Development-only decision lock. Does not open gold or evaluation response files."""
import argparse,time
from common import *
from trace_records import load_trace
from selector import select_all_tolerances,TOLERANCES

def derive_lock(bundle):
 bundle=Path(bundle);records=[];costs={};evidence={}
 for model in MODELS:
  dev=load_trace(bundle,'dev',model,require_complete=True)
  evidence[model]={k:dev[k] for k in ('raw_sha256','submissions_sha256','summary_sha256','control_evidence')}
  costs[model]=dev['ledger']
  for task in TASKS:
   policies=select_all_tolerances(dev['vectors'][task])
   for tolerance,_ in TOLERANCES:
    for selector in ('Input','Output','Repeat'):
     records.append({'model':model,'task':task,'tolerance':tolerance,'primary_operating_point':tolerance=='0.05',
      'selector':selector,**policies[tolerance][selector],
      'observed_standalone_calibration_cost':dev['ledger']['by_task'][task]['selector_calibration'][selector]})
 assert len(records)==36 and len({(r['model'],r['task'],r['tolerance'],r['selector']) for r in records})==36
 return {'schema_version':1,'policy_count':36,'policies':records,'development_evidence':evidence,'development_research_costs':costs,
  'scientific_freeze_sha256':sha(bundle/'PREINFERENCE_FROZEN_SHA256.json'),
  'relation_file_sha256':sha(bundle/'inputs/relations.jsonl'),'partition_file_sha256':sha(bundle/'inputs/partitions.jsonl'),
  'selector_source_sha256':sha(bundle/'selector.py'),'heldout_outputs_read':False,'gold_read':False}

def main(bundle):
 bundle=Path(bundle)
 if any((bundle/f'run_test_{model}').exists() for model in MODELS):raise RuntimeError('Refusing to create a decision lock after an evaluation run directory exists')
 payload=derive_lock(bundle)
 path=bundle/'POLICY_LOCK.json';write_json(path,{'locked_unix':time.time(),'status':'all36 mechanically locked; evaluation still awaits independent/root release','payload':payload})
 write_json(bundle/'POLICY_LOCK_SHA256.json',{'POLICY_LOCK.json':sha(path),'policy_count':36,'payload_sha256':digest(compact(payload))})
 print(compact({'status':'locked','policies':36,'lock_sha256':sha(path),'evaluation_authorized':False}),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,default=ROOT);a=p.parse_args();main(a.bundle)
