#!/usr/bin/env python3
"""Paired ratio, candidate and resource statistics from saved recovery records."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from batch_prompts import parse

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def read(p):return [json.loads(s) for s in p.read_text(encoding='utf8').splitlines() if s.strip()]

def keyed(rows,records,size):
    expected=[(r['pool_id'],start) for r in rows for start in range(0,32,size)]
    if [(r['pool_id'],r['start']) for r in records]!=expected:raise ValueError('Unexpected record order')
    n=32//size;out=[]
    for q in range(len(rows)):
        chunks=[parse(x['final'],size) for x in records[q*n:(q+1)*n]]
        out.append(sum(chunks,[]) if all(x is not None for x in chunks) else None)
    return out

def point(rows,records):
    if [(r['pool_id'],r['start']) for r in records]!=[(q['pool_id'],i) for q in rows for i in range(32)]:raise ValueError('Unexpected pointwise order')
    out=[]
    for q in range(len(rows)):
        labels=[x['final'] for x in records[q*32:(q+1)*32]]
        out.append(labels if all(x in ('Y','N') for x in labels) else None)
    return out

def metrics(gold,pred):
    tp=fp=fn=tn=0;err=[]
    for g,p in zip(gold,pred):
        err.append(abs(g.count('Y')-p.count('Y'))/32)
        for a,b in zip(g,p):tp+=a==b=='Y';tn+=a==b=='N';fp+=a=='N' and b=='Y';fn+=a=='Y' and b=='N'
    return {'tp':tp,'fp':fp,'fn':fn,'tn':tn,'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,
            'balanced_accuracy':(tp/(tp+fn)+tn/(tn+fp))/2 if tp+fn and tn+fp else None,'ratio_mae':float(np.mean(err))}

def main(selection,recovery,prior,out):
    if out.exists():raise FileExistsError('Use a fresh analysis output')
    data=selection/'inputs.jsonl';rows=read(data);meta=json.loads((selection/'selection.json').read_text())
    new=json.loads((recovery/'summary.json').read_text());old=json.loads((prior/'summary.json').read_text())
    if not(new.get('completed') and old.get('completed')) or sha(data)!=meta['inputs_sha256'] or sha(data)!=new['selection_sha256'] or sha(data)!=old['selection_inputs_sha256']:raise ValueError('Incomplete or mismatched input')
    sizes={'eight_original':8,'four_original':4,'four_stable_reverse':4,'four_replay':4,'eight_replay':8}
    vectors={n:keyed(rows,read(recovery/(n+'_responses.jsonl')),size) for n,size in sizes.items()}
    previous=keyed(rows,read(prior/'original_responses.jsonl'),8);pointwise=point(rows,read(prior/'pointwise_responses.jsonl'))
    result={'scope':'Post-hoc exposed pools, same unsupported recovery environment for both keyed plans; prior pointwise came from earlier software.',
            'selection_sha256':sha(data),'recovery_summary_sha256':sha(recovery/'summary.json'),'prior_summary_sha256':sha(prior/'summary.json'),
            'primary':'case-control new four-minus-new eight query-macro ratio MAE','resamples':4000,'scenarios':{},
            'all_pool_original_costs':{'four':new['modes']['four_original'],'eight':new['modes']['eight_original']}}
    for scenario,seed in (('case_control',20261113),('uniform',20261114)):
        attempted=[i for i,r in enumerate(rows) if r['scenario']==scenario]
        common=[i for i in attempted if vectors['four_original'][i] is not None and vectors['eight_original'][i] is not None]
        info={'attempted_pools':len(attempted),'common_complete_pools':len(common),'query_bootstrap_seed':seed,
              'completion':{n:sum(v[i] is not None for i in attempted) for n,v in vectors.items()}}
        for name,ref in (('four_stable_reverse','four_original'),('four_replay','four_original'),('eight_replay','eight_original')):
            both=[i for i in attempted if vectors[ref][i] is not None and vectors[name][i] is not None]
            info[name+'_changed_labels']=sum(sum(a!=b for a,b in zip(vectors[ref][i],vectors[name][i])) for i in both)
            info[name+'_compared_labels']=32*len(both)
        both=[i for i in attempted if vectors['eight_original'][i] is not None and previous[i] is not None]
        info['prior_eight_vs_recovery_changed_labels']=sum(sum(a!=b for a,b in zip(vectors['eight_original'][i],previous[i])) for i in both)
        info['prior_eight_vs_recovery_compared_labels']=32*len(both)
        if not common:result['scenarios'][scenario]=info;continue
        gold=[[x['gold'] for x in rows[i]['items']] for i in common];f=[vectors['four_original'][i] for i in common];e=[vectors['eight_original'][i] for i in common]
        info['metrics']={'four_original':metrics(gold,f),'eight_original':metrics(gold,e),'always_N':metrics(gold,[['N']*32 for _ in common])}
        if all(pointwise[i] is not None for i in common):info['metrics']['prior_full_pointwise']=metrics(gold,[pointwise[i] for i in common])
        delta=np.asarray([(abs(g.count('Y')-a.count('Y'))-abs(g.count('Y')-b.count('Y')))/32 for g,a,b in zip(gold,f,e)])
        draws=np.random.default_rng(seed).integers(0,len(common),size=(4000,len(common)));sample=delta[draws].mean(axis=1)
        info['four_minus_eight_ratio_mae']=float(delta.mean());info['query_bootstrap_95']=[float(y) for y in np.quantile(sample,[.025,.975])]
        info['common_pool_ids']=[rows[i]['pool_id'] for i in common];result['scenarios'][scenario]=info
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print('Analyzed all recovery modes and earlier references without inference.')

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('selection','recovery','prior','out'):p.add_argument('--'+n,type=Path,required=True)
    x=p.parse_args();main(x.selection,x.recovery,x.prior,x.out)
