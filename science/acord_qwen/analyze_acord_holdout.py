#!/usr/bin/env python3
"""Analyze a completed ACORD run; no model inference or data mutation."""
import argparse,hashlib,json,re
from pathlib import Path
import numpy as np
from acord_prompts import IDS,parse_keyed,parse_point
from order_fallback import combine

def lines(path):
    with path.open(encoding='utf8') as f:return [json.loads(x) for x in f if x.strip()]

def vector(rows,recs):
    expected=[(r['pool_id'],start) for r in rows for start in (0,8,16,24)]
    if [(x['pool_id'],x['start']) for x in recs]!=expected:raise ValueError('unexpected keyed request order')
    out=[]
    for i,r in enumerate(rows):
        v=[]
        for rec in recs[4*i:4*i+4]:
            obj=parse_keyed(rec['final'])
            if obj is None:break
            v.extend(obj[j] for j in IDS)
        out.append(v if len(v)==32 else None)
    return out

def f1(tp,fp,fn):
    den=2*tp+fp+fn
    return np.divide(2*tp,den,out=np.zeros_like(den,dtype=float),where=den!=0)

def metrics(gold,pred):
    a=np.asarray([v=='Y' for row in gold for v in row]);b=np.asarray([v=='Y' for row in pred for v in row]);tp=int((a&b).sum());fp=int((~a&b).sum());fn=int((a&~b).sum());tn=int((~a&~b).sum())
    return {'tp':tp,'fp':fp,'fn':fn,'tn':tn,'f1':float(f1(np.array(tp),np.array(fp),np.array(fn))),
            'balanced_accuracy':float((tp/(tp+fn)+tn/(tn+fp))/2) if tp+fn and tn+fp else None,
            'ratio_mae':float(np.mean([abs(g.count('Y')-p.count('Y'))/len(g) for g,p in zip(gold,pred)]))}

def main(selection,run,out):
    if out.exists():raise FileExistsError('Choose a fresh output path')
    rows=lines(selection/'inputs.jsonl');meta=json.loads((selection/'selection.json').read_text());summary=json.loads((run/'summary.json').read_text())
    sha=hashlib.sha256((selection/'inputs.jsonl').read_bytes()).hexdigest()
    if sha!=meta['inputs_sha256'] or sha!=summary['selection_inputs_sha256'] or not summary.get('completed'):
        raise ValueError('Input mismatch or incomplete run')
    originals=vector(rows,lines(run/'original_responses.jsonl'));reverse=vector(rows,lines(run/'stable_reverse_responses.jsonl'))
    replays=vector(rows,lines(run/'replay_responses.jsonl'))
    selected=lines(run/'selective_pointwise_responses.jsonl');lookup={(x['pool_id'],x['start']):parse_point(x['final']) for x in selected}
    if len(lookup)!=len(selected):raise ValueError('Duplicate selective pointwise request')
    expected={(r['pool_id'],j) for r,a,b in zip(rows,originals,reverse) if a is not None and b is not None for j,(x,y) in enumerate(zip(a,b)) if x!=y}
    if set(lookup)!=expected:raise ValueError('Selective routing differs from keyed disagreements')
    sel=[combine(a,b,[lookup.get((r['pool_id'],j)) for j in range(32)]) for r,a,b in zip(rows,originals,reverse)]
    saved=lines(run/'selective_outputs.jsonl')
    if [(x['pool_id'],x['pred']) for x in saved]!=[(r['pool_id'],v) for r,v in zip(rows,sel)]:raise ValueError('Saved selective vectors mismatch')
    points=lines(run/'pointwise_responses.jsonl');want=[(r['pool_id'],j) for r in rows for j in range(32)]
    if [(x['pool_id'],x['start']) for x in points]!=want:raise ValueError('Full pointwise request order mismatch')
    full=[]
    for i in range(len(rows)):
        v=[parse_point(x['final']) for x in points[32*i:32*i+32]];full.append(v if None not in v else None)
    result={'scope':'fixed pools; not population prevalence or legal advice','protocol':meta['protocol'],
            'selection_sha256':sha,'run_sha256':hashlib.sha256((run/'summary.json').read_bytes()).hexdigest(),
            'primary':'case_control selective-minus-original query-macro MAE','scenarios':{},
            'model_calls':summary['selective_pipeline'],'resamples':4000}
    for scenario,seed in (('case_control',20261101),('uniform',20261102)):
        attempted=[i for i,r in enumerate(rows) if r['scenario']==scenario]
        common=[i for i in attempted if originals[i] is not None and reverse[i] is not None and sel[i] is not None]
        info={'attempted_pools':len(attempted),'common_complete_pools':len(common),'query_bootstrap_seed':seed,
              'completion':{k:sum(v[i] is not None for i in attempted) for k,v in [('original',originals),('reverse',reverse),('selective',sel),('replay',replays),('pointwise',full)]}}
        if not common:result['scenarios'][scenario]=info;continue
        gold=[[x['gold'] for x in rows[i]['items']] for i in common]
        pred={'original':[originals[i] for i in common],'reverse':[reverse[i] for i in common],
              'selective':[sel[i] for i in common]}
        pred['lexical']=[];pred['lexical_fallback']=[];pred['always_Y']=[];pred['always_N']=[]
        for i in common:
            row=rows[i];terms=set(re.findall(r'\w+',row['query'].casefold()))
            lex=['Y' if terms and terms.issubset(set(re.findall(r'\w+',x['text'].casefold()))) else 'N' for x in row['items']]
            pred['lexical'].append(lex);pred['lexical_fallback'].append(combine(originals[i],reverse[i],lex))
            pred['always_Y'].append(['Y']*32);pred['always_N'].append(['N']*32)
        info['metrics']={name:metrics(gold,vals) for name,vals in pred.items()}
        n=len(common);g=np.asarray([[z=='Y' for z in v] for v in gold],float);vec={k:np.asarray([[z=='Y' for z in row] for row in values],float) for k,values in pred.items()}
        vec['midpoint']=(vec['original']+vec['reverse'])/2
        err={k:abs(v.mean(axis=1)-g.mean(axis=1)) for k,v in vec.items()}
        info['midpoint_ratio_mae']=float(err['midpoint'].mean())
        rng=np.random.default_rng(seed);draw=rng.integers(0,n,size=(4000,n));info['selective_minus']={}
        for reference in ('original','midpoint','lexical_fallback'):
            d=err['selective']-err[reference];info['selective_minus'][reference]={'ratio_mae':float(d.mean()),
                'query_bootstrap_95':[float(x) for x in np.quantile(d[draw].mean(axis=1),[.025,.975])]}
        ids=sorted({item['doc_id'] for i in common for item in rows[i]['items']});lookup={d:j for j,d in enumerate(ids)}
        mapping=np.asarray([[lookup[x['doc_id']] for x in rows[i]['items']] for i in common]);rng=np.random.default_rng(seed+1000)
        ranges={k:[] for k in ('original','midpoint','lexical_fallback')}
        for _ in range(4000):
            qdraw=rng.integers(0,n,size=n);w=rng.exponential(size=len(ids))[mapping];den=w.sum(axis=1);target=(w*g).sum(axis=1)/den
            wr={k:abs((w*v).sum(axis=1)/den-target) for k,v in vec.items()}
            for k in ranges:ranges[k].append(float(np.mean((wr['selective']-wr[k])[qdraw])))
        for k,values in ranges.items():info['selective_minus'][k]['clause_reweighted_range_95']=[float(x) for x in np.quantile(values,[.025,.975])]
        info['reweighting_warning']='Changing clause weights changes each fixed pool and its estimand; not a fixed-pool confidence interval.'
        info['common_pool_ids']=[rows[i]['pool_id'] for i in common];result['scenarios'][scenario]=info
    out.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
    print('Analyzed completed run; primary common pools:',result['scenarios']['case_control']['common_complete_pools'])

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--selection',type=Path,required=True);p.add_argument('--run',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();main(a.selection,a.run,a.out)
