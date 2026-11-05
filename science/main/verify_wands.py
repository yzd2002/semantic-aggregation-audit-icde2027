#!/usr/bin/env python3
"""Reconstruct WANDS selection and reparse all saved operator responses."""
import argparse,collections,json
from pathlib import Path
from run_wands import sample
from run_experiment import parse_batch
from run_validation import parse_compact
HERE=Path(__file__).resolve().parent
def lines(path):
    with path.open(encoding='utf8') as f:return [json.loads(s) for s in f]
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,default=HERE/'run_wands_160')
    ap.add_argument('--wands-dir',type=Path,default=Path('/home/gpu/intent2s_wands_official_2026'));a=ap.parse_args()
    rows=lines(a.run/'inputs.jsonl');out=lines(a.run/'outputs.jsonl');summary=json.loads((a.run/'summary.json').read_text())
    rebuilt,meta=sample(a.wands_dir);assert rows==rebuilt and summary['selection']==meta
    assert len(rows)==len(out)==160 and summary['protocol']=='PROTOCOL_wands.md'
    assert dict(collections.Counter(it['grade'] for row in rows for it in row['items']))==summary['grade_counts']
    for r,o in zip(rows,out):
        assert r['query']==o['query'] and r['query_id']==o['query_id'] and len(r['items'])==32
        assert [it['id'] for it in r['items']]==list(range(1,33))
        assert all(it['gold']==('Y' if it['grade']=='Exact' else 'N') for it in r['items'])
    for method,ms in summary['plans'].items():
        complete=tp=fp=tn=fn=0
        for r,o in zip(rows,out):
            entry=o['plans'][method];pred=entry['pred'];raw=entry['raw']
            if method=='pointwise':recovered=[z['final'] if z['final'] in ('Y','N') else None for z in raw]
            else:
                recovered=[None]*len(r['items']);parser=parse_compact if 'compact' in method else parse_batch
                for chunk in raw:
                    val=parser(chunk['final'],chunk['size'])
                    if val is not None:recovered[chunk['start']:chunk['start']+chunk['size']]=val
            assert pred==(recovered if None not in recovered else None),(method,r['query_id'])
            if pred is None:continue
            complete+=1
            for it,y in zip(r['items'],pred):
                g=it['gold'];tp+=g=='Y' and y=='Y';fp+=g=='N' and y=='Y';tn+=g=='N' and y=='N';fn+=g=='Y' and y=='N'
        assert (complete,tp,fp,tn,fn)==tuple(ms[k] for k in ['complete_pools','tp','fp','tn','fn'])
        print('WANDS',method,'complete',complete,'confusion',tp,fp,tn,fn,'OK')
    print('WANDS source hashes, selection, mappings, parsers and summaries verified.')
if __name__=='__main__':main()
