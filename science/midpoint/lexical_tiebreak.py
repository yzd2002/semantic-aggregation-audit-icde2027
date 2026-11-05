#!/usr/bin/env python3
"""Post-hoc lexical tie-breaker on already saved two-order model traces."""
import argparse,json,re,tarfile
from pathlib import Path
import numpy as np
import midpoint_reference as prior

def f1(tp,fp,fn):
    den=2*tp+fp+fn
    return np.divide(2*tp,den,out=np.zeros_like(den,dtype=float),where=den!=0)

def baseline(row,trec):
    if trec:
        terms=set(re.findall(r'[a-z0-9]+',row['query'].lower()))
        if not terms:raise ValueError('empty TREC query')
        return ['Y' if terms.issubset(set(re.findall(r'[a-z0-9]+',(x['title']+' '+x['abstract']).lower()))) else 'N' for x in row['items']]
    terms=set(re.findall(r'\w+',row['query'].casefold()))
    return ['Y' if terms and terms.issubset(set(re.findall(r'\w+',x['text'].casefold()))) else 'N' for x in row['items']]

def compute(root,previous,out):
    baseline_results=json.loads(previous.read_text())['settings']
    result={'scope':'post-hoc virtual lexical tie-breaker; existing outputs and labels exposed; no new inference',
            'protocol':'LEXICAL_PROTOCOL.md','settings':{}}
    with tarfile.open(root/'posthoc_7b/posthoc_7b_artifacts.tar.gz','r:gz') as archive:
        for idx,(name,selection,run,member) in enumerate(prior.CONFIG):
            rows=prior.lines((root/selection/'inputs.jsonl').read_bytes());key='topic_id' if name.startswith('trec') else 'query_id'
            def get(file):return (root/run/file).read_bytes() if run else archive.extractfile(member+'/'+file).read()
            orig=prior.keyed(rows,get('original_responses.jsonl'),key,'original');rev=prior.keyed(rows,get('stable_reverse_responses.jsonl'),key,'reverse')
            if name=='wands96_qwen':sel,_,_=prior.virtual_outputs(rows,orig,rev,get('pointwise_responses.jsonl'),key)
            else:sel=prior.selection_outputs(rows,get('selective_outputs.jsonl'),key)
            common=[i for i in range(len(rows)) if all(v[i] is not None for v in (orig,rev,sel))]
            if [rows[i][key] for i in common]!=baseline_results[name]['query_ids']:raise ValueError('prior common sample changed: '+name)
            counts=[];errors=[];weight_data=[];ids=[]
            for i in common:
                row=rows[i];a=orig[i];b=rev[i];s=sel[i];x=baseline(row,key=='topic_id');g=[z['gold'] for z in row['items']]
                lex=[u if u==v else c for u,v,c in zip(a,b,x)]
                if not(len(a)==len(b)==len(s)==len(x)==len(g)) or any(z not in ('Y','N') for z in g):raise ValueError('nonbinary or mismatched vectors')
                vectors=[s,lex,x];row_counts=[];row_errors=[]
                for pred in vectors:
                    tp=sum(p=='Y' and t=='Y' for p,t in zip(pred,g));fp=sum(p=='Y' and t=='N' for p,t in zip(pred,g));fn=sum(p=='N' and t=='Y' for p,t in zip(pred,g));tn=sum(p=='N' and t=='N' for p,t in zip(pred,g))
                    row_counts.append((tp,fp,fn,tn));row_errors.append(abs(pred.count('Y')-g.count('Y'))/len(g))
                counts.append(row_counts);errors.append(row_errors)
                weight_data.append((np.asarray([t=='Y' for t in g],float),np.asarray([p=='Y' for p in s],float),np.asarray([p=='Y' for p in lex],float)))
                ids.append([z['doc_id'] if key=='topic_id' else z['product_id'] for z in row['items']])
            ct=np.asarray(counts,dtype=int);er=np.asarray(errors,float);n=len(common);metrics={}
            for k,label in enumerate(('selective','lexical_fallback','lexical_only')):
                tp,fp,fn,tn=ct[:,k,:].sum(axis=0);value=float(f1(np.array(tp),np.array(fp),np.array(fn)))
                if (tp+fn)==0 or (tn+fp)==0:bal=None
                else:bal=float((tp/(tp+fn)+tn/(tn+fp))/2)
                metrics[label]={'f1':value,'balanced_accuracy':bal,'ratio_mae':float(er[:,k].mean()),'tp':int(tp),'fp':int(fp),'fn':int(fn),'tn':int(tn)}
            if abs(metrics['selective']['ratio_mae']-baseline_results[name]['mean_absolute_ratio_error']['selective'])>1e-12:raise ValueError('prior selective metric mismatch '+name)
            if name=='wands96_qwen':
                old=json.loads((root/run/'paired_analysis.json').read_text())['quality_common']
                for key2,label in [('lexical_fallback','lexical_fallback'),('lexical','lexical_only')]:
                    for m in ('f1','balanced_accuracy','ratio_mae'):
                        if abs(old[key2][m]-metrics[label][m])>1e-12:raise ValueError('prior WANDS96 baseline mismatch '+m)
            rng=np.random.default_rng(20261121+idx);draw=rng.integers(0,n,size=(4000,n));qcounts=ct[draw].sum(axis=1);sample_f1=f1(qcounts[:,0,0],qcounts[:,0,1],qcounts[:,0,2])-f1(qcounts[:,1,0],qcounts[:,1,1],qcounts[:,1,2]);sample_mae=(er[draw,0]-er[draw,1]).mean(axis=1)
            rng=np.random.default_rng(20262121+idx);unique=sorted({z for row in ids for z in row});lookup={z:i for i,z in enumerate(unique)};ix=np.array([lookup[z] for row in ids for z in row]);lengths=np.array([len(z) for z in ids]);offset=np.r_[0,np.cumsum(lengths)[:-1]];gold=np.concatenate([t[0] for t in weight_data]);sp=np.concatenate([t[1] for t in weight_data]);lp=np.concatenate([t[2] for t in weight_data]);cross=[]
            for _ in range(4000):
                qdraw=rng.integers(0,n,size=n);w=rng.exponential(size=len(unique))[ix];den=np.add.reduceat(w,offset);g=np.add.reduceat(w*gold,offset)/den;s=np.add.reduceat(w*sp,offset)/den;l=np.add.reduceat(w*lp,offset)/den
                cross.append(float(np.mean((abs(s-g)-abs(l-g))[qdraw])))
            result['settings'][name]={'attempted':len(rows),'common':n,'metrics':metrics,'selective_minus_lexical_fallback':{'ratio_mae':float(er[:,0].mean()-er[:,1].mean()),
                'ratio_mae_query_bootstrap_95':[float(x) for x in np.quantile(sample_mae,[.025,.975])],
                'ratio_mae_product_weighted_range_95':[float(x) for x in np.quantile(cross,[.025,.975])],
                'pooled_f1':metrics['selective']['f1']-metrics['lexical_fallback']['f1'],
                'pooled_f1_query_bootstrap_95':[float(x) for x in np.quantile(sample_f1,[.025,.975])]},
                'lexical_fallback_cost':baseline_results[name]['midpoint_cost'],
                'selective_cost':baseline_results[name]['selective_cost'],
                'cost_note':'logical model calls/input tokens only; lexical CPU time not measured',
                'query_seed':20261121+idx,'product_weight_seed':20262121+idx,'query_ids':[rows[i][key] for i in common]}
            print(name,'MAE lexical',round(metrics['lexical_fallback']['ratio_mae'],4),'selective',round(metrics['selective']['ratio_mae'],4),
                  'F1 lexical',round(metrics['lexical_fallback']['f1'],4),'selective',round(metrics['selective']['f1'],4),flush=True)
    out.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf8')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--artifact',required=True,type=Path);p.add_argument('--prior-result',required=True,type=Path);p.add_argument('--out',required=True,type=Path);a=p.parse_args()
    if a.out.exists():raise FileExistsError('Use a fresh output file')
    compute(a.artifact,a.prior_result,a.out)
