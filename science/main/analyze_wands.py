#!/usr/bin/env python3
"""Cluster-bootstrap WANDS quality on the four-plan common-completion subset."""
import argparse,json,re
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
def lines(path):
    with path.open(encoding='utf8') as f:return [json.loads(s) for s in f]
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,default=HERE/'run_wands_160');a=ap.parse_args()
    rows=lines(a.run/'inputs.jsonl');outs=lines(a.run/'outputs.jsonl');methods=['pointwise','batch_json','two_stage_json','two_stage_compact_fallback']
    assert len(rows)==len(outs)==160 and all(r['query_id']==o['query_id'] for r,o in zip(rows,outs))
    shared=[i for i,o in enumerate(outs) if all(o['plans'][m]['pred'] is not None for m in methods)]
    result={'description':'Query-cluster bootstrap on pools completed by all plans. This is a sampled judged workload, not a user-satisfaction interval.',
            'common_complete_pools':len(shared),'attempted_pools':len(rows),'resamples':4000,'seed':20271111,'methods':{},'differences':{}}
    if not shared:
        (a.run/'paired_analysis.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2));return
    records={};constants=['always_yes','always_no','lexical_all_terms']
    for method in methods+constants:
        values=[]
        for i in shared:
            row=rows[i];gold=[x['gold'] for x in row['items']]
            if method in methods:pred=outs[i]['plans'][method]['pred']
            elif method=='lexical_all_terms':
                query=set(re.findall(r'\w+',row['query'].casefold()))
                pred=['Y' if query and query.issubset(set(re.findall(r'\w+',it['text'].casefold()))) else 'N' for it in row['items']]
            else:pred=[('Y' if method=='always_yes' else 'N')]*len(gold)
            assert len(gold)==len(pred)
            tp=sum(g=='Y' and y=='Y' for g,y in zip(gold,pred));fp=sum(g=='N' and y=='Y' for g,y in zip(gold,pred))
            tn=sum(g=='N' and y=='N' for g,y in zip(gold,pred));fn=sum(g=='Y' and y=='N' for g,y in zip(gold,pred))
            mae=abs(sum(y=='Y' for y in pred)/len(pred)-sum(g=='Y' for g in gold)/len(gold))
            values.append((tp,fp,tn,fn,mae))
        records[method]=np.array(values,dtype=float)
    rng=np.random.default_rng(20271111);draw=rng.integers(0,len(shared),size=(4000,len(shared)));boot={}
    for name,v in records.items():
        tp,fp,tn,fn=v[:,:4].sum(axis=0);den=2*tp+fp+fn
        measured={'tp':int(tp),'fp':int(fp),'tn':int(tn),'fn':int(fn),
                  'f1':float(2*tp/den) if den else 0.0,
                  'balanced_accuracy':float((tp/(tp+fn)+tn/(tn+fp))/2) if (tp+fn) and (tn+fp) else None,
                  'mae':float(v[:,4].mean())}
        b=v[draw,:4].sum(axis=1);tp,fp,tn,fn=b.T;den=2*tp+fp+fn
        f1=np.divide(2*tp,den,out=np.zeros_like(tp),where=den>0);mae=v[draw,4].mean(axis=1)
        boot[name]={'f1':f1,'mae':mae}
        for key,x in boot[name].items():measured[key+'_ci']=[float(y) for y in np.quantile(x,[.025,.975])]
        result['methods'][name]=measured
    for name in methods[1:]+constants:
        result['differences'][name]={k+'_ci':[float(y) for y in np.quantile(boot[name][k]-boot['pointwise'][k],[.025,.975])]
                                     for k in ('f1','mae')}
    (a.run/'paired_analysis.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
