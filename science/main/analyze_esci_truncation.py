#!/usr/bin/env python3
"""Analyze paired short/long/replay ESCI judgments on jointly complete pools."""
import argparse
import collections
import hashlib
import json
from pathlib import Path

import numpy as np


def read(path):return [json.loads(x) for x in path.read_text(encoding='utf8').splitlines()]


def measure(rows,predictions):
    tp=tn=fp=fn=0;errors=[]
    for row,pred in zip(rows,predictions):
        gold=[x['gold'] for x in row['items']]
        if not gold or len(pred)!=len(gold) or set(gold+pred)-{'Y','N'}:raise ValueError('Invalid binary vector')
        for g,p in zip(gold,pred):
            tp+=g=='Y' and p=='Y';tn+=g=='N' and p=='N';fp+=g=='N' and p=='Y';fn+=g=='Y' and p=='N'
        errors.append(abs(pred.count('Y')/len(gold)-gold.count('Y')/len(gold)))
    return {'candidates':tp+tn+fp+fn,'tp':tp,'tn':tn,'fp':fp,'fn':fn,
            'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,
            'balanced_accuracy':(tp/(tp+fn)+tn/(tn+fp))/2 if tp+fn and tn+fp else None,
            'ratio_mae':float(np.mean(errors))},np.asarray(errors,dtype=float)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--selection',type=Path,required=True)
    ap.add_argument('--run',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args()
    if a.out.exists():raise FileExistsError(a.out)
    rows=read(a.selection/'inputs.jsonl');selection=json.loads((a.selection/'selection.json').read_text())
    run=json.loads((a.run/'summary.json').read_text());modes=('short','long','replay')
    if run['selection_inputs_sha256']!=selection['inputs_sha256']:raise ValueError('Selection mismatch')
    records={m:read(a.run/(m+'_responses.jsonl')) for m in modes}
    flat=[(row,it) for row in rows for it in row['items']]
    for mode in modes:
        if len(records[mode])!=len(flat):raise ValueError('Unexpected response count')
        for (row,it),rec in zip(flat,records[mode]):
            if (rec['query_id'],rec['item_id'],rec['product_id'],rec['text_changed']) != (
                row['query_id'],it['id'],it['product_id'],it['text_changed']):raise ValueError('Response alignment mismatch')
    grouped={m:collections.defaultdict(list) for m in modes}
    for m in modes:
        for rec in records[m]:grouped[m][rec['query_id']].append(rec['final'] if rec['final'] in ('Y','N') else None)
    complete={m:sum(len(grouped[m][r['query_id']])==len(r['items']) and None not in grouped[m][r['query_id']]
                    for r in rows) for m in modes}
    if any(complete[m]!=run['modes'][m]['complete_pools'] for m in modes):
        raise ValueError('Run completion count does not match responses')
    common=[r for r in rows if all(len(grouped[m][r['query_id']])==len(r['items'])
                                   and None not in grouped[m][r['query_id']] for m in modes)]
    if not common:raise RuntimeError('No common-complete pools; quality comparison undefined')
    preds={m:[grouped[m][r['query_id']] for r in common] for m in modes}
    metrics={};errors={}
    for m in modes:metrics[m],errors[m]=measure(common,preds[m])
    delta=errors['long']-errors['short'];rng=np.random.default_rng(20271101)
    draw=rng.integers(0,len(common),size=(4000,len(common)));samples=delta[draw].mean(axis=1)
    changes={}
    for mode in ('long','replay'):
        subsets={'all':[0,0],'text_changed':[0,0],'text_unchanged':[0,0]}
        for row,short,other in zip(common,preds['short'],preds[mode]):
            for item,x,y in zip(row['items'],short,other):
                key='text_changed' if item['text_changed'] else 'text_unchanged'
                for sub in ('all',key):
                    subsets[sub][0]+=x!=y;subsets[sub][1]+=1
        changes[mode+'_vs_short']={k:{'changed':v[0],'total':v[1],
            'fraction':v[0]/v[1] if v[1] else None} for k,v in subsets.items()}
    docids=sorted({it['product_id'] for r in common for it in r['items']})
    index={x:i for i,x in enumerate(docids)}
    offsets=[];doc_index=[];gold=[];short=[];long=[];pos=0
    for row,sp,lp in zip(common,preds['short'],preds['long']):
        offsets.append(pos);pos+=len(row['items'])
        for it,x,y in zip(row['items'],sp,lp):
            doc_index.append(index[it['product_id']]);gold.append(it['gold']=='Y');short.append(x=='Y');long.append(y=='Y')
    doc_index=np.asarray(doc_index);offsets=np.asarray(offsets)
    gold=np.asarray(gold,dtype=float);short=np.asarray(short,dtype=float);long=np.asarray(long,dtype=float)
    rng=np.random.default_rng(20271102);sensitivity=[]
    for _ in range(4000):
        qdraw=rng.integers(0,len(common),len(common));pweights=rng.exponential(1.,len(docids));w=pweights[doc_index]
        denom=np.add.reduceat(w,offsets)
        if np.any(denom<=0):raise RuntimeError('A selected query has zero product weight')
        gr=np.add.reduceat(w*gold,offsets)/denom;sr=np.add.reduceat(w*short,offsets)/denom
        lr=np.add.reduceat(w*long,offsets)/denom
        sensitivity.append(float((np.abs(lr-gr)-np.abs(sr-gr))[qdraw].mean()))
    result={'protocol':'PROTOCOL_esci_truncation_sensitivity.md','selection_inputs_sha256':selection['inputs_sha256'],
        'attempted_pools':len(rows),'attempted_candidates':len(flat),'completion':complete,
        'common_complete_pools':len(common),'common_query_ids':[r['query_id'] for r in common],
        'changed_text_candidates_all':selection['changed_candidates'],
        'changed_text_candidates_common':sum(it['text_changed'] for r in common for it in r['items']),
        'metrics':metrics,'changes':changes,
        'primary_long_minus_short':{'ratio_mae_difference':float(delta.mean()),
            'query_bootstrap_95':[float(x) for x in np.percentile(samples,[2.5,97.5])],
            'draws':4000,'seed':20271101},
        'product_weight_sensitivity':{'distinct_products':len(docids),'draws':4000,'seed':20271102,
            'reweighted_range_95':[float(x) for x in np.percentile(sensitivity,[2.5,97.5])]},
        'limitations':'Post-hoc same public labeled sample; product-weight sensitivity changes the estimand, not a fixed-pool confidence interval.'}
    a.out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print('Common-complete pools:',len(common),'of',len(rows))
    print('Ratio MAE short, long, replay:',[metrics[m]['ratio_mae'] for m in modes])
    print('Longer minus shorter:',result['primary_long_minus_short'])
    print('Product-weight sensitivity:',result['product_weight_sensitivity'])
    print('Candidate changes:',changes)


if __name__=='__main__':main()
