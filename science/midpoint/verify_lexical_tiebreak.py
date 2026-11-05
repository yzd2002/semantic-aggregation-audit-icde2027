#!/usr/bin/env python3
"""Independent strict-parser, routing, and query-bootstrap verification."""
import argparse,json,math,re,sys,tarfile
from pathlib import Path
import numpy as np
import midpoint_reference as prior

def read(raw):return [json.loads(x) for x in raw.decode('utf8').split('\n') if x.strip()]
def close(a,b):return math.isclose(a,b,rel_tol=0,abs_tol=1e-12)
def f1(tp,fp,fn):return 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.0

def lexical(row,trec):
    # Separate tokenization implementation and data structures.
    if trec:
        words=re.findall('[a-z0-9]+',row['query'].lower());texts=[i['title']+' '+i['abstract'] for i in row['items']]
        return [('Y' if all(w in set(re.findall('[a-z0-9]+',x.lower())) for w in words) else 'N') for x in texts]
    words=list(set(re.findall(r'\w+',row['query'].casefold())))
    return [('Y' if words and all(w in set(re.findall(r'\w+',it['text'].casefold())) for w in words) else 'N') for it in row['items']]

def main(root,result):
    sys.path.insert(0,str(root))
    from run_keyed_order_audit import parse_keyed
    actual=json.loads(result.read_text())['settings'];assert list(actual)==[x[0] for x in prior.CONFIG]
    arc=root/'posthoc_7b/posthoc_7b_artifacts.tar.gz'
    with tarfile.open(arc,'r:gz') as archive:
        for idx,(name,selection,run,member) in enumerate(prior.CONFIG):
            rows=read((root/selection/'inputs.jsonl').read_bytes());key='topic_id' if name.startswith('trec') else 'query_id'
            def get(file):return (root/run/file).read_bytes() if run else archive.extractfile(member+'/'+file).read()
            orig=read(get('original_responses.jsonl'));rev=read(get('stable_reverse_responses.jsonl'))
            om={(x[key],x['start']):x for x in orig};rm={(x[key],x['start']):x for x in rev};assert len(om)==len(orig) and len(rm)==len(rev)
            points=read(get('pointwise_responses.jsonl' if name=='wands96_qwen' else 'selective_pointwise_responses.jsonl'))
            pm={(x[key],x['start']):x['final'] for x in points};assert len(pm)==len(points)
            saved={} if name=='wands96_qwen' else {x[key]:x['pred'] for x in read(get('selective_outputs.jsonl'))}
            labels=[];deltas=[];data=[];items=[];wanted=set()
            for row in rows:
                a=[];b=[]
                for start in range(0,len(row['items']),8):
                    size=min(8,len(row['items'])-start);ids=list(range(1,size+1));q=row[key]
                    x=parse_keyed(om[(q,start)]['final'],ids);y=parse_keyed(rm[(q,start)]['final'],ids)
                    if x is None or y is None:break
                    a += [x[i] for i in ids];b += [y[i] for i in ids]
                if len(a)!=len(row['items']) or len(b)!=len(row['items']):continue
                lex=lexical(row,key=='topic_id');model=[];tie=[]
                for i,(x,y,l) in enumerate(zip(a,b,lex)):
                    if x==y:model.append(x);tie.append(x)
                    else:
                        wanted.add((row[key],i));model.append(pm[(row[key],i)]);tie.append(l)
                if any(x not in ('Y','N') for x in model):model=None
                if name!='wands96_qwen':assert model==saved[row[key]],name
                if model is None:continue
                gold=[i['gold'] for i in row['items']];n=len(gold);group=[];errs=[]
                for pred in (model,tie,lex):
                    tp=sum(g=='Y' and p=='Y' for g,p in zip(gold,pred));fp=sum(g=='N' and p=='Y' for g,p in zip(gold,pred));fn=sum(g=='Y' and p=='N' for g,p in zip(gold,pred));tn=sum(g=='N' and p=='N' for g,p in zip(gold,pred));group.append((tp,fp,fn,tn));errs.append(abs(pred.count('Y')-gold.count('Y'))/n)
                labels.append((row[key],group,errs));deltas.append(errs[0]-errs[1]);data.append(([x=='Y' for x in gold],[x=='Y' for x in model],[x=='Y' for x in tie]));items.append([x['doc_id'] if key=='topic_id' else x['product_id'] for x in row['items']])
            if name!='wands96_qwen':assert set(pm)==wanted,name
            x=actual[name];assert [q[0] for q in labels]==x['query_ids'] and len(labels)==x['common']
            for method,mi in [('selective',0),('lexical_fallback',1),('lexical_only',2)]:
                summed=[sum(v[1][mi][j] for v in labels) for j in range(4)];tp,fp,fn,tn=summed
                m=x['metrics'][method];assert (tp,fp,fn,tn)==(m['tp'],m['fp'],m['fn'],m['tn'])
                assert close(f1(tp,fp,fn),m['f1']) and close(sum(v[2][mi] for v in labels)/len(labels),m['ratio_mae'])
            rng=np.random.default_rng(20261121+idx);n=len(labels);draws=rng.integers(0,n,size=(4000,n));f1_delta=[];mae_delta=[]
            for draw in draws:
                one=[sum(labels[i][1][0][j] for i in draw) for j in range(3)];two=[sum(labels[i][1][1][j] for i in draw) for j in range(3)]
                f1_delta.append(f1(*one)-f1(*two));mae_delta.append(sum(deltas[i] for i in draw)/n)
            details=x['selective_minus_lexical_fallback'];assert close(sum(deltas)/n,details['ratio_mae'])
            for v1,v2 in zip(np.quantile(mae_delta,[.025,.975]),details['ratio_mae_query_bootstrap_95']):assert close(v1,v2),name
            for v1,v2 in zip(np.quantile(f1_delta,[.025,.975]),details['pooled_f1_query_bootstrap_95']):assert close(v1,v2),name
            if name in {'wands52_qwen','esci250_qwen','trec50_qwen'}:
                unique=sorted({item for group in items for item in group});lookup={v:i for i,v in enumerate(unique)};rng=np.random.default_rng(20262121+idx);samples=[]
                for _ in range(4000):
                    qdraw=rng.integers(0,n,size=n);w=rng.exponential(size=len(unique));per=[]
                    for group,(gold,model,tie) in zip(items,data):
                        den=gg=ss=ll=0.0
                        for item,g,s,l in zip(group,gold,model,tie):
                            wt=float(w[lookup[item]]);den+=wt;gg+=wt*g;ss+=wt*s;ll+=wt*l
                        per.append(abs(ss/den-gg/den)-abs(ll/den-gg/den))
                    samples.append(sum(per[i] for i in qdraw)/n)
                for v1,v2 in zip(np.quantile(samples,[.025,.975]),details['ratio_mae_product_weighted_range_95']):assert close(v1,v2),name
            print(name,'strict parsing, routing, metrics and paired intervals verified',flush=True)
    print('All 12 settings verified; product-reweighted sensitivity recomputed for three settings.')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--artifact',required=True,type=Path);p.add_argument('--result',required=True,type=Path);a=p.parse_args();main(a.artifact,a.result)
