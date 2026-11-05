#!/usr/bin/env python3
"""Independently rebuild all candidate vectors using the original strict parser."""
import argparse, json, math, sys, tarfile
from pathlib import Path
import numpy as np

CONFIG = [
 ('wands96_qwen','wands_fallback_validation_selection','run_wands_fallback_validation_qwen_v1',None),
 ('wands52_qwen','wands_selective_execution_selection','run_wands_selective_execution_qwen_v1',None),
 ('wands52_smollm','wands_selective_execution_selection','run_wands_selective_execution_smollm_v1',None),
 ('esci250_qwen','esci_selective_holdout_selection','run_esci_selective_holdout_qwen',None),
 ('esci250_smollm','esci_selective_holdout_selection','run_esci_selective_holdout_smollm',None),
 ('trec50_qwen','trec_covid_transfer_selection','run_trec_covid_transfer',None),
 ('trec50_smollm','trec_covid_transfer_selection','run_trec_covid_smollm',None),
 ('esci250_7b','esci_selective_holdout_selection',None,'posthoc/run_esci_7b'),
 ('esci250_7b_schema','esci_selective_holdout_selection',None,'posthoc/run_esci_7b_specguided'),
 ('wands52_7b','wands_selective_execution_selection',None,'posthoc/run_wands_7b'),
 ('wands52_7b_schema','wands_selective_execution_selection',None,'posthoc/run_wands_7b_specguided'),
 ('trec50_7b','trec_covid_transfer_selection',None,'posthoc/run_trec_7b'),
]

def records(raw):return [json.loads(x) for x in raw.decode('utf8').split('\n') if x.strip()]
def close(a,b):return math.isclose(a,b,rel_tol=0,abs_tol=1e-12)

def main(root,result):
    sys.path.insert(0,str(root))
    from run_keyed_order_audit import parse_keyed
    from order_fallback import combine
    saved=json.loads(result.read_text())['settings']
    assert list(saved)==[x[0] for x in CONFIG]
    checks=0
    for n in range(1,9):
        for a in range(n+1):
            for b in range(n+1):
                for g in range(n+1):
                    assert abs(a+b-2*g)<=abs(a-g)+abs(b-g)
                    checks+=1
    assert checks==2024
    assert abs((1+0)/(2*8)-1/8)>0 and (1+0)%(2)!=0
    print('Midpoint inequality checked for',checks,'integer-count cases; midpoint need not be an aligned label ratio.',flush=True)
    archive_path=root/'posthoc_7b/posthoc_7b_artifacts.tar.gz'
    with tarfile.open(archive_path,'r:gz') as archive:
        for index,(name,selection,run,member) in enumerate(CONFIG):
            rows=records((root/selection/'inputs.jsonl').read_bytes());key='topic_id' if name.startswith('trec') else 'query_id'
            def get(file):return (root/run/file).read_bytes() if run else archive.extractfile(member+'/'+file).read()
            orig=records(get('original_responses.jsonl'));rev=records(get('stable_reverse_responses.jsonl'))
            o={(r[key],r['start']):r for r in orig};v={(r[key],r['start']):r for r in rev}
            assert len(o)==len(orig) and len(v)==len(rev)
            point=records(get('pointwise_responses.jsonl' if name=='wands96_qwen' else 'selective_pointwise_responses.jsonl'))
            points={(x[key],x['start']):x['final'] for x in point};assert len(points)==len(point)
            expected_disagreements=set();allvectors=[];deltas=[];ratio_values=[];item_ids=[]
            actual={} if name=='wands96_qwen' else {x[key]:x['pred'] for x in records(get('selective_outputs.jsonl'))}
            for row in rows:
                a=[];b=[]
                for start in range(0,len(row['items']),8):
                    size=min(8,len(row['items'])-start);ids=list(range(1,size+1));q=row[key]
                    pa=parse_keyed(o[(q,start)]['final'],ids);pb=parse_keyed(v[(q,start)]['final'],ids)
                    if pa is None or pb is None:break
                    a.extend(pa[x] for x in ids);b.extend(pb[x] for x in ids)
                if len(a)!=len(row['items']) or len(b)!=len(row['items']):continue
                h=[None]*len(a)
                for i,(x,y) in enumerate(zip(a,b)):
                    if x!=y:
                        expected_disagreements.add((row[key],i))
                        h[i]=points[(row[key],i)] if points[(row[key],i)] in ('Y','N') else None
                pred=combine(a,b,h)
                if name!='wands96_qwen':assert pred==actual[row[key]],name
                if pred is None:continue
                n=len(a);g=sum(x['gold']=='Y' for x in row['items']);ca=a.count('Y');cb=b.count('Y');cs=pred.count('Y')
                em=abs(ca+cb-2*g)/(2*n);es=abs(cs-g)/n;deltas.append(es-em)
                ratio_values.append((abs(ca-g)/n,abs(cb-g)/n,em,es))
                allvectors.append(([x['gold']=='Y' for x in row['items']],[x=='Y' for x in a],[x=='Y' for x in b],[x=='Y' for x in pred]))
                item_ids.append([x['doc_id'] if key=='topic_id' else x['product_id'] for x in row['items']])
            if name!='wands96_qwen':assert set(points)==expected_disagreements,name
            found=saved[name];assert len(deltas)==found['common'];names=('original','reverse','midpoint','selective')
            for i,k in enumerate(names):assert close(sum(x[i] for x in ratio_values)/len(deltas),found['mean_absolute_ratio_error'][k]),(name,k)
            assert close(sum(deltas)/len(deltas),found['selective_minus_midpoint'])
            rng=np.random.default_rng(20261101+index)
            draws=rng.integers(0,len(deltas),size=(4000,len(deltas)))
            sample=[sum(deltas[i] for i in draw)/len(deltas) for draw in draws]
            ci=np.quantile(sample,[.025,.975]);assert all(close(x,y) for x,y in zip(ci,found['query_bootstrap_95'])),name
            if name in {'wands52_qwen','esci250_qwen','trec50_7b'}:
                unique=sorted({q for group in item_ids for q in group});ids={v:i for i,v in enumerate(unique)}
                rng=np.random.default_rng(20262101+index);samples=[];n=len(deltas)
                for _ in range(4000):
                    qdraw=rng.integers(0,n,size=n);w=rng.exponential(size=len(unique));local=[]
                    for group,(g,a,b,s) in zip(item_ids,allvectors):
                        denominator=gold=mid=sel=0.0
                        for t,gv,av,bv,sv in zip(group,g,a,b,s):
                            weight=float(w[ids[t]]);denominator+=weight;gold+=weight*gv;mid+=weight*(av+bv)/2;sel+=weight*sv
                        local.append(abs(sel/denominator-gold/denominator)-abs(mid/denominator-gold/denominator))
                    samples.append(sum(local[i] for i in qdraw)/n)
                ci=np.quantile(samples,[.025,.975]);assert all(close(x,y) for x,y in zip(ci,found['product_reweighted_range_95'])),name
            print(name,'strict parser, fallback reconstruction and bootstrap verified',flush=True)
    print('All 12 settings verified; product reweighting independently recomputed for three settings.')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--artifact',required=True,type=Path);p.add_argument('--result',required=True,type=Path);a=p.parse_args();main(a.artifact,a.result)
