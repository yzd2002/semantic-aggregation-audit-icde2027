#!/usr/bin/env python3
"""Independently reconstruct ACORD sample positions and source associations."""
import argparse,collections,csv,hashlib,io,json,random
from pathlib import Path
from zipfile import ZipFile

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main(archive,selection,protocol):
    meta=json.loads((selection/'selection.json').read_text());inputs=selection/'inputs.jsonl'
    if sha(inputs)!=meta['inputs_sha256'] or sha(archive)!=meta['source_sha256'] or sha(protocol)!=meta['protocol_sha256']:
        raise ValueError('Recorded source, selection, or protocol hash mismatch')
    with inputs.open(encoding='utf8') as f:rows=[json.loads(line) for line in f if line.strip()]
    with ZipFile(archive) as z:
        def path(s):
            x=[n for n in z.namelist() if n.endswith('/'+s)]
            if len(x)!=1:raise ValueError('Unexpected archive path')
            return x[0]
        with z.open(path('queries.jsonl')) as f:queries={x['_id']:x for x in map(json.loads,f)}
        with z.open(path('corpus.jsonl')) as f:docs={x['_id']:x['text'] for x in map(json.loads,f)}
        scores=collections.defaultdict(lambda:collections.defaultdict(list))
        with z.open(path('qrels/test.tsv')) as f:
            for x in csv.DictReader(io.TextIOWrapper(f,encoding='utf-8-sig',newline=''),delimiter='\t'):
                scores[x['query-id']][x['corpus-id']].append(int(x['score']))
    eligible=[];usable={}
    for q,edges in scores.items():
        if q not in queries or queries[q]['metadata'].get('split')!='test' or not queries[q]['text'].strip():continue
        usable[q]={d:s[0] for d,s in edges.items() if len(set(s))==1 and d in docs and docs[d].strip()}
        if len(usable[q])>=32:eligible.append(q)
    eligible.sort();assert eligible==meta['eligible_query_ids']
    assert [(r['scenario'],r['query_id']) for r in rows]==[(scenario,q) for scenario in ('uniform','case_control') for q in eligible]
    actual_summaries={}
    for scenario in ('uniform','case_control'):
        grouped=[r for r in rows if r['scenario']==scenario];hist=collections.Counter();truncated=0;pos=0;negpools=0
        for r in grouped:
            q=r['query_id'];source=queries[q];assert r['pool_id']==scenario+':'+q and r['query']==source['text']
            assert r['query_category']==source['metadata'].get('category') and r['query_type']==source['metadata'].get('type')
            cand=usable[q];keys=sorted(cand);seed=int.from_bytes(hashlib.sha256(('ACORD-20261101\0'+scenario+'\0'+q).encode()).digest(),'big');rng=random.Random(seed)
            if scenario=='uniform':want=rng.sample(keys,32)
            else:
                groups=([d for d in keys if cand[d] in (3,4)],[d for d in keys if cand[d] in (1,2)],[d for d in keys if cand[d]==0])
                want=[]
                for group,quota in zip(groups,(8,8,16)):want += rng.sample(group,min(quota,len(group)))
                remaining=sorted(set(keys).difference(want));want += rng.sample(remaining,32-len(want))
            rng.shuffle(want)
            assert [i['doc_id'] for i in r['items']]==want and len(set(want))==32
            for it in r['items']:
                text=docs[it['doc_id']];grade=cand[it['doc_id']]
                assert it['score']==grade and it['gold']==('Y' if grade>=3 else 'N') and it['text']==text[:600] and it['source_chars']==len(text)
                hist[grade]+=1;pos+=grade>=3;truncated+=len(text)>600
            negpools+=not any(it['gold']=='Y' for it in r['items'])
        saved=meta['scenarios'][scenario]
        assert saved['pools']==len(grouped) and saved['candidate_occurrences']==32*len(grouped)
        assert saved['grade_counts']=={str(k):v for k,v in sorted(hist.items())} and saved['positive_occurrences']==pos
        assert saved['all_negative_pools']==negpools and saved['truncated_occurrences']==truncated
        actual_summaries[scenario]={'pools':len(grouped),'positive':pos,'truncated':truncated,'all_negative_pools':negpools}
    by={(r['scenario'],r['query_id']):{i['doc_id'] for i in r['items']} for r in rows}
    overlap=sum(len(by['uniform',q]&by['case_control',q]) for q in eligible)
    assert overlap==meta['candidate_overlap_within_query_across_scenarios']
    print('Independently reconstructed',len(rows),'pools from',len(eligible),'test queries;',actual_summaries,'cross-scenario overlaps',overlap)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--archive',type=Path,required=True);p.add_argument('--selection',type=Path,required=True);p.add_argument('--protocol',type=Path,required=True);a=p.parse_args();main(a.archive,a.selection,a.protocol)
