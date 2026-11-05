#!/usr/bin/env python3
"""Frozen all-eight-complete descriptive analysis; never clip excess below zero."""
import argparse,json
import numpy as np
from common import *

def assemble(model):
    inputs={d:read(ROOT/'selection'/f'{d}_inputs.jsonl') for d in ('esci','wands')}
    plans={(d,r['query_id'],c,k):[None]*len(r['items']) for d,rows in inputs.items() for r in rows for c in (*CONDITIONS,'S') for k in (0,1)}
    records=read(ROOT/f'run_{model}/responses.jsonl');jobs=read(ROOT/'selection/schedule.jsonl')
    if len(records)!=len(jobs):raise ValueError('Incomplete execution; retain failure evidence separately instead of silently dropping jobs')
    for r,j in zip(records,jobs):
        if any(r[k]!=j[k] for k in ('ordinal','request_id','block','dataset','query_id','condition','repeat','ids')):raise ValueError('Response schedule differs')
        parsed=parse(r['raw'],r['ids'],r['condition'].endswith('D'))
        if r['parsed']!=({str(k):v for k,v in parsed.items()} if parsed else None):raise ValueError('Saved parse differs')
        if parsed:
            vec=plans[r['dataset'],r['query_id'],r['condition'],r['repeat']]
            for i,label in parsed.items():
                if vec[i-1] is not None:raise ValueError('Duplicate occurrence decision')
                vec[i-1]=label=='Y'
    return inputs,{k:(np.asarray(v,dtype=bool) if None not in v else None) for k,v in plans.items()},records

def quality(rows,vecs):
    tp=fp=fn=tn=0;mae=[];cer=[]
    for row,p in zip(rows,vecs):
        g=np.asarray([i['gold']=='Y' for i in row['items']]);tp+=int(np.count_nonzero(g&p));fp+=int(np.count_nonzero(~g&p));fn+=int(np.count_nonzero(g&~p));tn+=int(np.count_nonzero(~g&~p))
        mae.append(abs(int(p.sum())-int(g.sum()))/len(g));cer.append(float(np.mean(p!=g)))
    return {'pools':len(rows),'candidates':sum(len(r['items']) for r in rows),'tp':tp,'fp':fp,'fn':fn,'tn':tn,
        'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.0,'balanced_accuracy':(tp/(tp+fn)+tn/(tn+fp))/2 if tp+fn and tn+fp else None,
        'ratio_mae':float(np.mean(mae)) if mae else None,'candidate_error_macro':float(np.mean(cer)) if cer else None}
def interval(values,draw):
    a=np.asarray(values,dtype=float)
    return {'mean':float(a.mean()),'unadjusted_query_interval_95':np.quantile(a[draw].mean(axis=1),[.025,.975]).tolist()}
def distances(a,b):return {'candidate_disagreement':float(np.mean(a!=b)),'absolute_ratio_change':abs(int(a.sum())-int(b.sum()))/len(a)}

def main(model):
    inputs,plans,records=assemble(model);result={'model':model,'scope':'Post-hoc project design fixed before new sample/inference; WANDS exposed; inherited same-machine environment.',
        'responses_sha256':sha(ROOT/f'run_{model}/responses.jsonl'),'analysis_code_sha256':sha(Path(__file__)),'datasets':{}}
    for di,(d,rows) in enumerate(inputs.items()):
        seed=2026110220+2*(model=='smol')+di
        complete={f'{c}{k}':[r['query_id'] for r in rows if plans[d,r['query_id'],c,k] is not None] for c in (*CONDITIONS,'S') for k in (0,1)}
        primary=[r for r in rows if all(plans[d,r['query_id'],c,k] is not None for c in CONDITIONS for k in (0,1))]
        singleton=[r for r in rows if all(plans[d,r['query_id'],'S',k] is not None for k in (0,1))]
        shared=[r for r in primary if all(plans[d,r['query_id'],'S',k] is not None for k in (0,1))]
        entry={'attempted_pools':len(rows),'completion':{k:len(v) for k,v in complete.items()},'primary_common_pools':len(primary),
            'primary_common_query_ids':[r['query_id'] for r in primary],'primary_excluded_query_ids':[r['query_id'] for r in rows if r not in primary],
            'singleton_both_complete_pools':len(singleton),'batch_singleton_common_pools':len(shared),'batch_singleton_common_query_ids':[r['query_id'] for r in shared],
            'bootstrap_draws':4000,'bootstrap_seed':int(seed),'quality_all8':{},'per_pool':[]}
        if not primary:result['datasets'][d]=entry;continue
        draw=np.random.default_rng(seed).integers(0,len(primary),size=(4000,len(primary)))
        for c in CONDITIONS:
            for k in (0,1):entry['quality_all8'][f'{c}{k}']=quality(primary,[plans[d,r['query_id'],c,k] for r in primary])
        entry['constants_all8']={v:quality(primary,[np.full(len(r['items']),v=='Y',dtype=bool) for r in primary]) for v in ('Y','N')}
        for row in primary:
            q=row['query_id'];g=np.asarray([i['gold']=='Y' for i in row['items']]);v={(c,k):plans[d,q,c,k] for c in CONDITIONS for k in (0,1)}
            stat={'query_id':q,'n':len(g),'gold_positive':int(g.sum()),'orders':{},'replay':{}}
            for out,ca,cr in [('A','AA','RA'),('D','AD','RD')]:
                rowchange=[distances(v[ca,k],v[cr,k]) for k in (0,1)];replay=[distances(v[c,0],v[c,1]) for c in (ca,cr)]
                val={}
                for key in rowchange[0]:
                    dr=float(np.mean([x[key] for x in rowchange]));dp=float(np.mean([x[key] for x in replay]))
                    val[key]={'row_change':dr,'mean_replay':dp,'excess':dr-dp}
                mae=lambda p:abs(int(p.sum())-int(g.sum()))/len(g)
                cer=lambda p:float(np.mean(p!=g))
                val['reference_ratio_mae_reverse_minus_canonical']=float(np.mean([mae(v[cr,k])-mae(v[ca,k]) for k in (0,1)]))
                val['candidate_error_reverse_minus_canonical']=float(np.mean([cer(v[cr,k])-cer(v[ca,k]) for k in (0,1)]))
                stat['orders'][out]=val
            stat['replay']={c:distances(v[c,0],v[c,1]) for c in CONDITIONS};entry['per_pool'].append(stat)
        entry['co_primary']={out:{key:{kind:interval([r['orders'][out][key][kind] for r in entry['per_pool']],draw)
            for kind in ('row_change','mean_replay','excess')} for key in ('candidate_disagreement','absolute_ratio_change')} for out in ('A','D')}
        entry['effect_modification_D_minus_A']={key:interval([r['orders']['D'][key]['excess']-r['orders']['A'][key]['excess'] for r in entry['per_pool']],draw)
            for key in ('candidate_disagreement','absolute_ratio_change')}
        entry['reference_quality_row_order']={out:{key:interval([r['orders'][out][key] for r in entry['per_pool']],draw)
            for key in ('reference_ratio_mae_reverse_minus_canonical','candidate_error_reverse_minus_canonical')} for out in ('A','D')}
        entry['replay']={c:{key:interval([r['replay'][c][key] for r in entry['per_pool']],draw) for key in ('candidate_disagreement','absolute_ratio_change')} for c in CONDITIONS}
        entry['singleton_quality_own_common']={str(k):quality(singleton,[plans[d,r['query_id'],'S',k] for r in singleton]) for k in (0,1)}
        entry['singleton_quality_all10']={str(k):quality(shared,[plans[d,r['query_id'],'S',k] for r in shared]) for k in (0,1)}
        entry['batch_singleton']={}
        if shared:
            draw10=np.random.default_rng(seed+100).integers(0,len(shared),size=(4000,len(shared)))
            for c in CONDITIONS:
                vals=[{key:float(np.mean([distances(plans[d,r['query_id'],c,k],plans[d,r['query_id'],'S',k])[key] for k in (0,1)]))
                    for key in ('candidate_disagreement','absolute_ratio_change')} for r in shared]
                entry['batch_singleton'][c]={key:interval([r[key] for r in vals],draw10) for key in ('candidate_disagreement','absolute_ratio_change')}
            entry['singleton_replay']={key:interval([distances(plans[d,r['query_id'],'S',0],plans[d,r['query_id'],'S',1])[key] for r in shared],draw10)
                for key in ('candidate_disagreement','absolute_ratio_change')}
        entry['actual_tokens']={f'{c}{k}':{'input':sum(r['actual_input_tokens'] for r in records if r['dataset']==d and r['condition']==c and r['repeat']==k),
            'output':sum(r['actual_output_tokens'] for r in records if r['dataset']==d and r['condition']==c and r['repeat']==k)} for c in (*CONDITIONS,'S') for k in (0,1)}
        result['datasets'][d]=entry
        print(model,d,'complete',len(primary),'/',len(rows),'excess',json.dumps({o:{k:v['excess'] for k,v in vals.items()} for o,vals in entry['co_primary'].items()}),flush=True)
    write_json(ROOT/f'run_{model}/analysis.json',result)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',choices=list(MODELS),required=True);main(p.parse_args().model)
