#!/usr/bin/env python3
"""Post-hoc fixed-pool confusion/ratio decomposition; no model execution."""
import argparse,hashlib,json,re
from fractions import Fraction
from pathlib import Path

SETTINGS=(('wands52','wands_selective_execution_selection','run_wands_selective_execution_qwen_v1','run_wands_selective_execution_qwen_v1/paired_analysis.json',52),
          ('esci250','esci_selective_holdout_selection','run_esci_selective_holdout_qwen','analysis_esci_selective_holdout_qwen/summary.json',250))
FILES=('original_responses.jsonl','stable_reverse_responses.jsonl','pointwise_responses.jsonl','selective_pointwise_responses.jsonl','selective_outputs.jsonl')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return [json.loads(s) for s in p.read_text(encoding='utf8').split('\n') if s.strip()]
def keyed(text,n):
    match=re.fullmatch(r'\{'+','.join('"'+str(i)+'":"([YN])"' for i in range(1,n+1))+r'\}',text)
    return list(match.groups()) if match else None
def keyed_pools(rows,records):
    want=[(r['query_id'],s,min(8,len(r['items'])-s)) for r in rows for s in range(0,len(r['items']),8)]
    if [(r['query_id'],r['start'],r['size']) for r in records]!=want:raise ValueError('Keyed order or size changed')
    outputs=[];offset=0
    for row in rows:
        labels=[]
        for start in range(0,len(row['items']),8):
            n=min(8,len(row['items'])-start);v=keyed(records[offset]['final'],n);offset+=1
            if v is None:raise ValueError('This diagnostic requires complete keyed pools')
            labels+=v
        outputs.append(labels)
    return outputs
def vectors(base,selection,run):
    rows=read(base/selection/'inputs.jsonl');root=base/run
    a=keyed_pools(rows,read(root/FILES[0]));b=keyed_pools(rows,read(root/FILES[1]));raw=read(root/FILES[2])
    want=[(r['query_id'],i) for r in rows for i in range(len(r['items']))]
    if [(r['query_id'],r['start']) for r in raw]!=want:raise ValueError('Pointwise order changed')
    if any(r['final'] not in ('Y','N') for r in raw):raise ValueError('Incomplete pointwise pool')
    full=[];offset=0
    for row in rows:
        n=len(row['items']);full.append([x['final'] for x in raw[offset:offset+n]]);offset+=n
    selected=read(root/FILES[3]);needed=[(r['query_id'],i) for r,x,y in zip(rows,a,b) for i,(u,v) in enumerate(zip(x,y)) if u!=v]
    if [(r['query_id'],r['start']) for r in selected]!=needed:raise ValueError('Selective routes changed')
    if any(r['final'] not in ('Y','N') for r in selected):raise ValueError('Incomplete selective response')
    h={(r['query_id'],r['start']):r['final'] for r in selected}
    selective=[[u if u==v else h[row['query_id'],i] for i,(u,v) in enumerate(zip(x,y))] for row,x,y in zip(rows,a,b)]
    saved=read(root/FILES[4]);assert [(r['query_id'],r['pred']) for r in saved]==[(r['query_id'],v) for r,v in zip(rows,selective)]
    return rows,{'keyed':a,'selective':selective,'pointwise':full,
                 'always_Y':[['Y']*len(r['items']) for r in rows],'always_N':[['N']*len(r['items']) for r in rows]}
def summarize(rows,preds):
    confusion=dict(tp=0,fp=0,fn=0,tn=0);candidate=[];ratio=[];cancel=[];bias=[];exact=hidden=opposite=0
    for row,pred in zip(rows,preds):
        gold=[x['gold'] for x in row['items']];n=len(gold);assert len(pred)==n
        fp=sum(g=='N' and p=='Y' for g,p in zip(gold,pred));fn=sum(g=='Y' and p=='N' for g,p in zip(gold,pred))
        tp=sum(g==p=='Y' for g,p in zip(gold,pred));tn=n-tp-fp-fn
        for k,v in zip(('tp','fp','fn','tn'),(tp,fp,fn,tn)):confusion[k]+=v
        e=Fraction(fp+fn,n);r=Fraction(abs(fp-fn),n);c=Fraction(2*min(fp,fn),n)
        assert e-r==c and abs(Fraction(pred.count('Y')-gold.count('Y'),n))==r
        candidate.append(e);ratio.append(r);cancel.append(c);bias.append(Fraction(fp-fn,n))
        exact+=fp==fn;hidden+=fp==fn and fp+fn>0;opposite+=min(fp,fn)>0
    count=len(rows);mean=lambda x:float(sum(x,Fraction(0))/count)
    tp,fp,fn=confusion['tp'],confusion['fp'],confusion['fn']
    return {**confusion,'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,
            'candidate_error_macro':mean(candidate),'ratio_mae':mean(ratio),'cancelled_error_macro':mean(cancel),
            'signed_ratio_bias':mean(bias),'exact_ratio_pools':exact,'exact_ratio_with_candidate_errors':hidden,'pools_with_opposing_errors':opposite}
def main(base,out):
    if out.exists():raise FileExistsError('Use a fresh result file')
    result={'scope':'Post-hoc re-analysis of already observed outputs; no inference, no causal or population claim.',
            'identity':'candidate_error - absolute_ratio_error = 2*min(FP,FN)/n, pool by pool',
            'weighting':'Query-macro means normalize each fixed pool by its own size; F1 uses pooled confusion counts.',
            'source_sha256':{},'settings':{}}
    for name,sel,run,prior,n in SETTINGS:
        for rel in [sel+'/inputs.jsonl',prior]+[run+'/'+f for f in FILES]:result['source_sha256'][rel]=sha(base/rel)
        rows,pred=vectors(base,sel,run);assert len(rows)==n
        old=json.loads((base/prior).read_text());assert old['common_complete_pools']==n
        stats={k:summarize(rows,v) for k,v in pred.items()}
        for current,original in (('keyed','original'),('selective','actual_fallback'),('pointwise','pointwise')):
            before=old['quality_common'][original]
            for field in ('tp','fp','fn','tn','f1','ratio_mae'):assert abs(stats[current][field]-before[field])<1e-12
        result['settings'][name]={'pools':n,'candidates':sum(len(r['items']) for r in rows),'plans':stats}
    out.write_text(json.dumps(result,indent=2)+'\n');print('Recomputed confusion/ratio decomposition for both fixed samples; prior metrics match.')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--artifact',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();main(a.artifact,a.out)
