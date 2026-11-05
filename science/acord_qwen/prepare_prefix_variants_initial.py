#!/usr/bin/env python3
"""Extend fixed ACORD candidate prefixes without changing any selected labels."""
import argparse,copy,hashlib,json
from pathlib import Path
from zipfile import ZipFile

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main(archive,base,amendment,out,length):
    if length not in (1800,2400):raise ValueError('Expected a fixed cap of 1800 or 2400')
    if out.exists():raise FileExistsError('Choose a new output directory')
    meta=json.loads((base/'selection.json').read_text());source=base/'inputs.jsonl'
    if sha(archive)!=meta['source_sha256'] or sha(source)!=meta['inputs_sha256']:
        raise ValueError('Original fixed source or selection has changed')
    with source.open(encoding='utf8') as f:rows=[json.loads(x) for x in f if x.strip()]
    with ZipFile(archive) as z:
        paths=[p for p in z.namelist() if p.endswith('/corpus.jsonl')]
        if len(paths)!=1:raise ValueError('Unexpected source archive')
        with z.open(paths[0]) as f:
            docs={}
            for line in f:
                x=json.loads(line);key=x['_id']
                if key in docs:raise ValueError('Duplicate source clause ID')
                docs[key]=x['text']
    changed=0;truncated={'uniform':0,'case_control':0}
    for row in rows:
        for item in row['items']:
            text=docs[item['doc_id']]
            if len(text)!=item['source_chars'] or item['text']!=text[:600]:
                raise ValueError('Source text does not reconstruct original selection')
            new=text[:length];changed+=new!=item['text'];item['text']=new
            truncated[row['scenario']]+=len(text)>length
    out.mkdir(parents=True,exist_ok=False);data=out/'inputs.jsonl'
    with data.open('w',encoding='utf8',newline='\n') as f:
        for row in rows:f.write(json.dumps(row,ensure_ascii=False,separators=(',',':'))+'\n')
    result=copy.deepcopy(meta)
    result['base_inputs_sha256']=meta['inputs_sha256'];result['inputs_sha256']=sha(data)
    result['prefix_length']=length;result['prefix_amendment_sha256']=sha(amendment)
    result['long_prefix_code_sha256']=sha(Path(__file__))
    result['changed_candidate_occurrences']=changed
    for scenario in ('uniform','case_control'):
        result['scenarios'][scenario]['truncated_at_600_occurrences']=meta['scenarios'][scenario]['truncated_occurrences']
        result['scenarios'][scenario]['truncated_occurrences']=truncated[scenario]
    result['limitations']=[x.replace('600 Unicode code points',f'{length} Unicode code points') for x in meta['limitations']]
    (out/'selection.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print('Extended',changed,'candidate occurrences; still truncated at 1200:',truncated,'inputs SHA-256',result['inputs_sha256'])

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--archive',type=Path,required=True);p.add_argument('--base',type=Path,required=True);p.add_argument('--amendment',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--length',type=int,required=True);a=p.parse_args();main(a.archive,a.base,a.amendment,a.out,a.length)
