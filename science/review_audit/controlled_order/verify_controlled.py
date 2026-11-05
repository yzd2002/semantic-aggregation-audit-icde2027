#!/usr/bin/env python3
"""Independent source selection, schedule, raw parser and integer-count analysis audit."""
import argparse,hashlib,json,math,random
from fractions import Fraction
from pathlib import Path
import numpy as np
from common import ROOT,SOURCE,FIELDS,CONDITIONS,sha,read

def close(a,b):
    if a is None or b is None:assert a is None and b is None,(a,b)
    else:assert np.allclose(a,b,rtol=0,atol=1e-12),(a,b)
def selection():
    import pandas as pd
    import pyarrow.dataset as ds
    out=ROOT/'selection';meta=json.loads((out/'exclusion_manifest.json').read_text());q=set();pid=set();text=set();train=set()
    for item in meta['sources']:
        p=Path(item['path']);assert sha(p)==item['sha256'];rows=read(p);q.update(str(r['query']).strip().casefold() for r in rows)
        if item['kind']=='training_manifest':
            ids={str(r['product_id']) for r in rows};pid|=ids;train|=ids
        elif item['kind']=='esci':
            for row in rows:
                for x in row['items']:
                    if 'product_id' in x:pid.add(str(x['product_id']))
                    values=[x[k] for k in ('text','short','long') if isinstance(x.get(k),str)];assert values
                    text.update(x[:600] for x in values)
    ep=SOURCE/'shopping_queries_dataset_examples.parquet';pp=SOURCE/'shopping_queries_dataset_products.parquet'
    assert sha(ep)==meta['source_hashes']['examples'] and sha(pp)==meta['source_hashes']['products']
    ex=pd.read_parquet(ep,columns=['example_id','query','query_id','product_id','product_locale','esci_label','small_version','split'],filters=[('product_locale','=','us'),('small_version','=',1),('split','=','test')])
    ids=train|set(ex.product_id.astype(str));tab=ds.dataset(pp,format='parquet').to_table(columns=['product_id',*FIELDS],filter=(ds.field('product_locale')=='us')&ds.field('product_id').isin(sorted(ids))).to_pylist()
    products={}
    for row in tab:
        key=str(row['product_id']);assert key not in products
        products[key]=' '.join(row[k] for k in FIELDS if isinstance(row[k],str) and row[k])[:600]
    assert train<=products.keys();text.update(products[i] for i in train)
    eligible=[]
    for qid,group in ex.groupby('query_id'):
        names={str(s).strip().casefold() for s in group['query']};pids=[str(i) for i in group.product_id]
        if len(names)==1 and '' not in names and not names&q and not set(pids)&pid and not any(products.get(i) in text for i in pids):eligible.append(int(qid))
    eligible.sort();assert eligible==meta['eligible_query_ids'];selected=random.Random(2026110201).sample(eligible,128);assert selected==meta['selected_query_ids']
    inputs={d:read(out/f'{d}_inputs.jsonl') for d in ('esci','wands')};assert [r['query_id'] for r in inputs['esci']]==list(map(str,selected))
    for row in inputs['esci']:
        source=ex[ex.query_id==int(row['query_id'])].sort_values('example_id').to_dict('records')
        sd=int.from_bytes(hashlib.sha256(('2026110201\0'+row['query_id']).encode()).digest()[:8],'big');random.Random(sd).shuffle(source)
        assert len(source)==len(row['items']) and len(source)>0
        assert row['query']==source[0]['query'] and row['query'].strip().casefold() not in q
        for index,(x,y) in enumerate(zip(row['items'],source),1):
            assert x['id']==index and x['example_id']==str(y['example_id']) and x['product_id']==str(y['product_id'])
            assert x['text']==products[x['product_id']] and x['text'] not in text and x['product_id'] not in pid
            assert x['grade']==y['esci_label'] and x['gold']==('Y' if y['esci_label'] in ('E','S') else 'N')
    old=Path('/home/gpu/IntentSat2S_ICDE2027_package/02_reproducibility/wands_selective_execution_selection/inputs.jsonl')
    assert inputs['wands']==read(old) and sha(old)==meta['source_hashes']['wands_input']
    units=[]
    for d,rows in inputs.items():
        for row in rows:
            assert [x['id'] for x in row['items']]==list(range(1,len(row['items'])+1))
            units.extend((d,row['query_id'],start,list(range(start+1,min(start+9,len(row['items'])+1)))) for start in range(0,len(row['items']),8))
    rng=random.Random(2026110217);rng.shuffle(units);expected=[]
    for ci,(d,qid,start,ids) in enumerate(units):
        jobs=[(c,k,start,ids) for c in CONDITIONS for k in (0,1)]+[('S',k,i-1,[i]) for i in ids for k in (0,1)];rng.shuffle(jobs)
        for c,k,s,ix in jobs:
            ordinal=len(expected);expected.append({'dataset':d,'query_id':qid,'start':s,'size':len(ix),'condition':c,'repeat':k,'ids':ix,
                'chunk_ordinal':ci,'ordinal':ordinal,'block':ordinal//64,'request_id':f'r{ordinal:06d}'})
    assert read(out/'schedule.jsonl')==expected
    frozen=json.loads((out/'PREINFERENCE_INPUT_CODE_SHA256.json').read_text())
    for rel,h in frozen['files'].items():assert sha(ROOT/rel)==h
    print('PASS selection: 128 entire new ESCI pools; all available prior IDs and visible texts/training prefixes excluded; WANDS52 unchanged; complete numeric-ID schedule and block membership independently reproduced.')

def raw_parse(raw,ids,desc):
    try:pairs=json.loads(raw,object_pairs_hook=lambda x:x)
    except (ValueError,TypeError):return None
    if not isinstance(pairs,list) or any(not isinstance(x,tuple) or len(x)!=2 for x in pairs):return None
    if [k for k,v in pairs]!=list(map(str,sorted(ids,reverse=desc))) or any(v not in ('Y','N') for k,v in pairs):return None
    canonical='{'+','.join(json.dumps(k)+':'+json.dumps(v) for k,v in pairs)+'}'
    return dict(pairs) if raw==canonical else None

def run_verify(model):
    location=ROOT/f'run_{model}';analysis=json.loads((location/'analysis.json').read_text());rs=read(location/'responses.jsonl');jobs=read(ROOT/'prepared'/f'{model}_requests.jsonl')
    assert len(rs)==len(jobs) and len({r['engine_request_id'] for r in rs})==len(rs)
    inputs={d:read(ROOT/'selection'/f'{d}_inputs.jsonl') for d in ('esci','wands')};vectors={(d,r['query_id'],c,k):[None]*len(r['items']) for d,rows in inputs.items() for r in rows for c in (*CONDITIONS,'S') for k in (0,1)}
    for r,j in zip(rs,jobs):
        for k in ('request_id','ordinal','block','dataset','query_id','condition','repeat','ids','prompt_sha256','grammar_sha256','id_map_sha256'):assert r[k]==j[k]
        p=raw_parse(r['raw'],j['ids'],j['condition'].endswith('D'));assert p==r['parsed']
        assert r['actual_input_tokens']==len(j['prompt_token_ids']) and r['actual_output_tokens']==len(r['output_token_ids'])
        assert r['input_token_ids_sha256']==hashlib.sha256(json.dumps(j['prompt_token_ids'],separators=(',',':')).encode()).hexdigest()
        if p:
            v=vectors[r['dataset'],r['query_id'],r['condition'],r['repeat']]
            for i,value in p.items():assert v[int(i)-1] is None;v[int(i)-1]=int(value=='Y')
    def done(key):return None not in vectors[key]
    def dist(x,y):return {'candidate_disagreement':Fraction(sum(a!=b for a,b in zip(x,y)),len(x)),
        'absolute_ratio_change':Fraction(abs(sum(x)-sum(y)),len(x))}
    def qm(rows,preds):
        tp=fp=fn=tn=0;ratio=[];error=[]
        for row,p in zip(rows,preds):
            g=[int(i['gold']=='Y') for i in row['items']]
            tp+=sum(x==y==1 for x,y in zip(g,p));fp+=sum(x==0 and y==1 for x,y in zip(g,p));fn+=sum(x==1 and y==0 for x,y in zip(g,p));tn+=sum(x==y==0 for x,y in zip(g,p))
            ratio.append(Fraction(abs(sum(p)-sum(g)),len(g)));error.append(Fraction(sum(x!=y for x,y in zip(g,p)),len(g)))
        return {'pools':len(rows),'candidates':sum(len(r['items']) for r in rows),'tp':tp,'fp':fp,'fn':fn,'tn':tn,'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0,
            'balanced_accuracy':(tp/(tp+fn)+tn/(tn+fp))/2 if tp+fn and tn+fp else None,'ratio_mae':float(sum(ratio)/len(rows)) if rows else None,'candidate_error_macro':float(sum(error)/len(rows)) if rows else None}
    def qcheck(rows,preds,old):
        for k,v in qm(rows,preds).items():close(v,old[k])
    def icheck(vals,weights,old):
        a=np.asarray(vals,dtype=float);close(float(a.mean()),old['mean']);close(np.quantile(weights@a/len(a),[.025,.975]),old['unadjusted_query_interval_95'])
    for di,(d,rows) in enumerate(inputs.items()):
        old=analysis['datasets'][d];seed=2026110220+2*(model=='smol')+di;assert old['bootstrap_seed']==seed
        for c in (*CONDITIONS,'S'):
            for k in (0,1):assert old['completion'][c+str(k)]==sum(done((d,r['query_id'],c,k)) for r in rows)
        primary=[r for r in rows if all(done((d,r['query_id'],c,k)) for c in CONDITIONS for k in (0,1))];n=len(primary)
        assert old['primary_common_query_ids']==[r['query_id'] for r in primary] and n==old['primary_common_pools']
        if not n:continue
        ix=np.random.default_rng(seed).integers(0,n,size=(4000,n));weights=np.stack([np.bincount(x,minlength=n) for x in ix]);computed=[]
        for r,stored in zip(primary,old['per_pool']):
            q=r['query_id'];g=[int(i['gold']=='Y') for i in r['items']];item={}
            for o,ca,cr in [('A','AA','RA'),('D','AD','RD')]:
                v={(c,k):vectors[d,q,c,k] for c in (ca,cr) for k in (0,1)};item[o]={}
                for key in ('candidate_disagreement','absolute_ratio_change'):
                    row=(dist(v[ca,0],v[cr,0])[key]+dist(v[ca,1],v[cr,1])[key])/2
                    replay=(dist(v[ca,0],v[ca,1])[key]+dist(v[cr,0],v[cr,1])[key])/2
                    item[o][key]={'row_change':float(row),'mean_replay':float(replay),'excess':float(row-replay)}
                    for which,value in item[o][key].items():close(value,stored['orders'][o][key][which])
                mae=lambda p:Fraction(abs(sum(p)-sum(g)),len(g));cer=lambda p:Fraction(sum(x!=y for x,y in zip(p,g)),len(g))
                for key,metric in [('reference_ratio_mae_reverse_minus_canonical',mae),('candidate_error_reverse_minus_canonical',cer)]:
                    item[o][key]=float(sum(metric(v[cr,k])-metric(v[ca,k]) for k in (0,1))/2);close(item[o][key],stored['orders'][o][key])
            computed.append(item)
        for o in ('A','D'):
            for key in ('candidate_disagreement','absolute_ratio_change'):
                for which in ('row_change','mean_replay','excess'):icheck([x[o][key][which] for x in computed],weights,old['co_primary'][o][key][which])
            for key in ('reference_ratio_mae_reverse_minus_canonical','candidate_error_reverse_minus_canonical'):icheck([x[o][key] for x in computed],weights,old['reference_quality_row_order'][o][key])
        for key in ('candidate_disagreement','absolute_ratio_change'):icheck([x['D'][key]['excess']-x['A'][key]['excess'] for x in computed],weights,old['effect_modification_D_minus_A'][key])
        for c in CONDITIONS:
            for k in (0,1):qcheck(primary,[vectors[d,r['query_id'],c,k] for r in primary],old['quality_all8'][c+str(k)])
            for key in ('candidate_disagreement','absolute_ratio_change'):icheck([float(dist(vectors[d,r['query_id'],c,0],vectors[d,r['query_id'],c,1])[key]) for r in primary],weights,old['replay'][c][key])
        for label in ('Y','N'):qcheck(primary,[[int(label=='Y')]*len(r['items']) for r in primary],old['constants_all8'][label])
        single=[r for r in rows if all(done((d,r['query_id'],'S',k)) for k in (0,1))];shared=[r for r in primary if r in single]
        assert old['batch_singleton_common_query_ids']==[r['query_id'] for r in shared]
        for k in (0,1):
            qcheck(single,[vectors[d,r['query_id'],'S',k] for r in single],old['singleton_quality_own_common'][str(k)])
            qcheck(shared,[vectors[d,r['query_id'],'S',k] for r in shared],old['singleton_quality_all10'][str(k)])
        if shared:
            ns=len(shared);draw=np.random.default_rng(seed+100).integers(0,ns,size=(4000,ns));w=np.stack([np.bincount(x,minlength=ns) for x in draw])
            for key in ('candidate_disagreement','absolute_ratio_change'):
                for c in CONDITIONS:icheck([float(sum(dist(vectors[d,r['query_id'],c,k],vectors[d,r['query_id'],'S',k])[key] for k in (0,1))/2) for r in shared],w,old['batch_singleton'][c][key])
                icheck([float(dist(vectors[d,r['query_id'],'S',0],vectors[d,r['query_id'],'S',1])[key]) for r in shared],w,old['singleton_replay'][key])
        for c in (*CONDITIONS,'S'):
            for k in (0,1):
                for label in ('input','output'):assert old['actual_tokens'][c+str(k)][label]==sum(r['actual_'+label+'_tokens'] for r in rs if r['dataset']==d and r['condition']==c and r['repeat']==k)
        print('PASS',model,d,': independent raw parser, all-8/all-10 sets, every quality, co-primary excess including negatives, effect modification, replay, singleton and bootstrap interval verified.')
    freeze=json.loads((ROOT/'prepared/EXECUTION_FROZEN_SHA256.json').read_text())
    for rel,h in freeze['files'].items():assert sha(ROOT/rel)==h
    print('PASS all frozen code/input/model-metadata/request hashes unchanged.')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--selection',action='store_true');p.add_argument('--model',choices=['qwen','smol']);a=p.parse_args()
    if a.selection:selection()
    if a.model:run_verify(a.model)
