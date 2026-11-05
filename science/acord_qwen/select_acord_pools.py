#!/usr/bin/env python3
"""Select explicitly scored ACORD test pools under the frozen design."""
import argparse,collections,csv,hashlib,io,json,random
from pathlib import Path
from zipfile import ZipFile

PROTOCOL='PROTOCOL_acord_holdout.md'
SOURCE_SHA='c9e851456a14aed161033b856e9f40f3cc9f46cf7282c2007b143420e93a65c3'

def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def find(z,suffix):
    paths=[x.filename for x in z.infolist() if x.filename.endswith('/'+suffix)]
    if len(paths)!=1:raise ValueError('Unexpected archive contents: '+suffix)
    return paths[0]

def read_jsonl(z,suffix):
    with z.open(find(z,suffix)) as f:
        for line in f:
            if line.strip():yield json.loads(line)

def sample(qid,scenario,pairs):
    seed=int.from_bytes(hashlib.sha256(f'ACORD-20261101\0{scenario}\0{qid}'.encode('utf8')).digest(),'big')
    rng=random.Random(seed)
    ids=sorted(pairs)
    if scenario=='uniform':selected=rng.sample(ids,32)
    elif scenario=='case_control':
        strata=[([d for d in ids if pairs[d]>=3],8),([d for d in ids if pairs[d] in (1,2)],8),([d for d in ids if pairs[d]==0],16)]
        selected=[]
        for group,quota in strata:selected.extend(rng.sample(group,min(len(group),quota)))
        have=set(selected);remainder=[d for d in ids if d not in have]
        selected.extend(rng.sample(remainder,32-len(selected)))
    else:raise ValueError('unknown scenario')
    rng.shuffle(selected)
    if len(selected)!=32 or len(set(selected))!=32:raise ValueError('invalid sampled pool')
    return selected

def main(archive,protocol,out):
    if out.exists():raise FileExistsError('Choose a new output directory')
    if digest(archive)!=SOURCE_SHA:raise ValueError('Pinned archive SHA mismatch')
    if protocol.name!=PROTOCOL:raise ValueError('Wrong protocol')
    with ZipFile(archive) as z:
        if z.testzip() is not None:raise ValueError('Source archive CRC failed')
        queries={};docs={}
        for x in read_jsonl(z,'queries.jsonl'):
            q=x.get('_id');t=x.get('text');meta=x.get('metadata',{})
            if not isinstance(q,str) or not isinstance(t,str) or not isinstance(meta,dict) or q in queries:raise ValueError('invalid or duplicate query')
            queries[q]={'text':t,'metadata':meta}
        for x in read_jsonl(z,'corpus.jsonl'):
            d=x.get('_id');t=x.get('text')
            if not isinstance(d,str) or not isinstance(t,str) or d in docs:raise ValueError('invalid or duplicate corpus item')
            docs[d]=t
        qrels=collections.defaultdict(lambda:collections.defaultdict(set));raw=0
        with z.open(find(z,'qrels/test.tsv')) as f:
            reader=csv.DictReader(io.TextIOWrapper(f,encoding='utf-8-sig',newline=''),delimiter='\t')
            if reader.fieldnames!=['query-id','corpus-id','score']:raise ValueError('Unexpected qrel header')
            for x in reader:
                q,d,raw_score=x['query-id'],x['corpus-id'],x['score'];score=int(raw_score)
                if score not in range(5):raise ValueError('Unexpected score: '+raw_score)
                qrels[q][d].add(score);raw+=1
    valid={};missing_queries=missing_docs=empty_docs=empty_queries=0;conflicts=duplicates=0
    for q,pairs in qrels.items():
        if q not in queries:missing_queries+=len(pairs);continue
        if queries[q]['metadata'].get('split')!='test':raise ValueError('qrel points outside official test split')
        if not queries[q]['text'].strip():empty_queries+=1;continue
        outpairs={}
        for d,scores in pairs.items():
            if len(scores)>1:conflicts+=1;continue
            if d not in docs:missing_docs+=1;continue
            if not docs[d].strip():empty_docs+=1;continue
            outpairs[d]=next(iter(scores))
        valid[q]=outpairs
    distinct=sum(len(v) for v in qrels.values());duplicates=raw-distinct
    eligible=[q for q in sorted(valid) if len(valid[q])>=32]
    if not eligible:raise ValueError('No eligible test queries')
    rows=[]
    for scenario in ('uniform','case_control'):
        for q in eligible:
            pairs=valid[q];ids=sample(q,scenario,pairs);meta=queries[q]['metadata'];items=[]
            for d in ids:
                text=docs[d];score=pairs[d]
                items.append({'doc_id':d,'text':text[:600],'source_chars':len(text),'score':score,'gold':'Y' if score>=3 else 'N'})
            rows.append({'pool_id':scenario+':'+q,'query_id':q,'scenario':scenario,'query':queries[q]['text'],
                         'query_category':meta.get('category'),'query_type':meta.get('type'),'items':items})
    if len({r['pool_id'] for r in rows})!=len(rows):raise ValueError('duplicate pool id')
    out.mkdir(parents=True,exist_ok=False)
    data=out/'inputs.jsonl'
    with data.open('w',encoding='utf8',newline='\n') as f:
        for row in rows:f.write(json.dumps(row,ensure_ascii=False,separators=(',',':'))+'\n')
    reports={}
    for name in ('uniform','case_control'):
        sub=[r for r in rows if r['scenario']==name];grades=collections.Counter(i['score'] for r in sub for i in r['items'])
        lengths=[i['source_chars'] for r in sub for i in r['items']]
        reports[name]={'pools':len(sub),'candidate_occurrences':len(lengths),'grade_counts':dict(sorted(grades.items())),
                       'positive_occurrences':sum(i['gold']=='Y' for r in sub for i in r['items']),
                       'all_negative_pools':sum(all(i['gold']=='N' for i in r['items']) for r in sub),
                       'truncated_occurrences':sum(x>600 for x in lengths),
                       'distinct_clauses':len({i['doc_id'] for r in sub for i in r['items']})}
    lookup={(r['scenario'],r['query_id']):{i['doc_id'] for i in r['items']} for r in rows}
    overlap=sum(len(lookup[('uniform',q)]&lookup[('case_control',q)]) for q in eligible)
    meta={'protocol':PROTOCOL,'protocol_sha256':digest(protocol),'source_sha256':digest(archive),
          'source_commit':'942b33a7c6fed7ead362c7c296401086f8bc0321','source_git_blob':'6de8c6c01a5221fe7669bc3117cfc02e5bc02fbc',
          'selection_code_sha256':digest(Path(__file__)),'inputs_sha256':digest(data),'query_records':len(queries),'clause_records':len(docs),
          'raw_test_qrels':raw,'duplicate_qrel_rows':duplicates,'conflicting_query_clause_pairs':conflicts,
          'missing_query_pairs':missing_queries,'missing_clause_pairs':missing_docs,'empty_query_count':empty_queries,'empty_clause_pairs':empty_docs,
          'eligible_queries':len(eligible),'eligible_query_ids':eligible,'candidate_overlap_within_query_across_scenarios':overlap,
          'parent_query_id_empty_for_test_queries':sum(not queries[q]['metadata'].get('parent_query_id') for q in eligible),
          'scenarios':reports,'limitations':['case-control sampling uses reference scores','zero scores include source-expanded negatives',
                  'not a live retrieval distribution','clause text limited to 600 Unicode code points','pretraining exposure unknown']}
    (out/'selection.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print('Selected',len(eligible),'test queries:',{k:{'pools':v['pools'],'positives':v['positive_occurrences'],'truncated':v['truncated_occurrences']} for k,v in reports.items()})
    print('Frozen inputs SHA-256:',meta['inputs_sha256'])

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--archive',type=Path,required=True);p.add_argument('--protocol',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();main(a.archive,a.protocol,a.out)
