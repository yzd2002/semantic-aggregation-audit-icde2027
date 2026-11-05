#!/usr/bin/env python3
"""Append-only pre-inference supplement: key-order effects and failure ledger."""
import argparse,json
import numpy as np
from common import *
from analyze_controlled import quality,distances,interval
from verify_controlled import raw_parse

def main(model):
    inputs={d:read(ROOT/'selection'/f'{d}_inputs.jsonl') for d in ('esci','wands')};schedule_jobs=read(ROOT/'selection/schedule.jsonl')
    path=ROOT/f'run_{model}/responses.jsonl';records=read(path) if path.exists() else []
    expected={j['request_id']:j for j in schedule_jobs};seen=set();attempted=set();invalid=[]
    vectors={(d,r['query_id'],c,k):[None]*len(r['items']) for d,rows in inputs.items() for r in rows for c in (*CONDITIONS,'S') for k in (0,1)}
    for r in records:
        assert r['request_id'] in expected and r['request_id'] not in seen;seen.add(r['request_id']);j=expected[r['request_id']]
        assert all(r[k]==j[k] for k in ('ordinal','condition','repeat','dataset','query_id','ids'));attempted.add((r['dataset'],r['query_id']))
        p=raw_parse(r['raw'],r['ids'],r['condition'].endswith('D'))
        if p is None:invalid.append(r['request_id']);continue
        vec=vectors[r['dataset'],r['query_id'],r['condition'],r['repeat']]
        for i,value in p.items():assert vec[int(i)-1] is None;vec[int(i)-1]=value=='Y'
    complete=lambda key:None not in vectors[key]
    result={'model':model,'exposure':'Supplement frozen after selected-label composition, before new GPU judgments.',
        'execution_ledger':{'planned_jobs':len(schedule_jobs),'returned_jobs':len(records),'valid_returned_jobs':len(records)-len(invalid),
            'invalid_returned_jobs':len(invalid),'invalid_request_ids':invalid,'unreturned_jobs':len(schedule_jobs)-len(records),
            'unreturned_request_ids':[j['request_id'] for j in schedule_jobs if j['request_id'] not in seen]},'datasets':{}}
    for di,(dataset,rows) in enumerate(inputs.items()):
        seed=2026110220+2*(model=='smol')+di
        batch=[r for r in rows if all(complete((dataset,r['query_id'],c,k)) for c in CONDITIONS for k in (0,1))]
        single=[r for r in rows if all(complete((dataset,r['query_id'],'S',k)) for k in (0,1))]
        entry={'selected_pools':len(rows),'attempted_pools':sum((dataset,r['query_id']) in attempted for r in rows),
            'wholly_unattempted_query_ids':[r['query_id'] for r in rows if (dataset,r['query_id']) not in attempted],
            'completion':{c+str(k):sum(complete((dataset,r['query_id'],c,k)) for r in rows) for c in (*CONDITIONS,'S') for k in (0,1)},
            'all8_complete_query_ids':[r['query_id'] for r in batch],'singleton_own_complete_query_ids':[r['query_id'] for r in single],
            'singleton_quality_own':{str(k):quality(single,[np.asarray(vectors[dataset,r['query_id'],'S',k],dtype=bool) for r in single]) for k in (0,1)},
            'actual_tokens':{c+str(k):{kind:sum(r['actual_'+kind+'_tokens'] for r in records if r['dataset']==dataset and r['condition']==c and r['repeat']==k) for kind in ('input','output')} for c in (*CONDITIONS,'S') for k in (0,1)},
            'secondary_key_order':{},'per_pool':[],'bootstrap_seed':seed}
        if batch:
            draw=np.random.default_rng(seed).integers(0,len(batch),size=(4000,len(batch)))
            for row in batch:
                q=row['query_id'];g=np.array([i['gold']=='Y' for i in row['items']]);item={'query_id':q,'input_orders':{}}
                for inp,ca,cd in [('A','AA','AD'),('R','RA','RD')]:
                    v={(c,k):np.asarray(vectors[dataset,q,c,k],dtype=bool) for c in (ca,cd) for k in (0,1)}
                    rowchange=[distances(v[ca,k],v[cd,k]) for k in (0,1)];rep=[distances(v[c,0],v[c,1]) for c in (ca,cd)];stat={}
                    for key in ('candidate_disagreement','absolute_ratio_change'):
                        change=float(np.mean([x[key] for x in rowchange]));replay=float(np.mean([x[key] for x in rep]));stat[key]={'key_change':change,'mean_replay':replay,'excess':change-replay}
                    mae=lambda p:abs(int(p.sum())-int(g.sum()))/len(g);cer=lambda p:float(np.mean(p!=g))
                    stat['reference_ratio_mae_D_minus_A']=float(np.mean([mae(v[cd,k])-mae(v[ca,k]) for k in (0,1)]))
                    stat['candidate_error_D_minus_A']=float(np.mean([cer(v[cd,k])-cer(v[ca,k]) for k in (0,1)]));item['input_orders'][inp]=stat
                entry['per_pool'].append(item)
            for inp in ('A','R'):
                entry['secondary_key_order'][inp]={key:{which:interval([r['input_orders'][inp][key][which] for r in entry['per_pool']],draw) for which in ('key_change','mean_replay','excess')} for key in ('candidate_disagreement','absolute_ratio_change')}
                for key in ('reference_ratio_mae_D_minus_A','candidate_error_D_minus_A'):entry['secondary_key_order'][inp][key]=interval([r['input_orders'][inp][key] for r in entry['per_pool']],draw)
        result['datasets'][dataset]=entry
    write_json(ROOT/f'run_{model}/supplementary_analysis.json',result)
    print('Supplement written',model,'returned/invalid/unreturned',len(records),len(invalid),len(schedule_jobs)-len(records),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',choices=['qwen','smol'],required=True);main(p.parse_args().model)
