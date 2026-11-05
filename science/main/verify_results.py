#!/usr/bin/env python3
"""Independently audit saved predictions, output parsers, counts and overlap."""
import collections,json
from pathlib import Path
from run_experiment import parse_batch
from run_validation import parse_compact
root=Path(__file__).resolve().parent
folders=['run_test_160','run_validation_500','run_full_pools_250']
if (root/'run_full_pools_smollm_250'/'summary.json').exists():folders.append('run_full_pools_smollm_250')
seen=set()
for name in folders:
 p=root/name; rows=[json.loads(x) for x in (p/'inputs.jsonl').open()];out=[json.loads(x) for x in (p/'outputs.jsonl').open()]
 summary=json.loads((p/'summary.json').read_text());assert len(rows)==len(out)==summary['selection']['sample_size']
 if name=='run_full_pools_smollm_250':
  reference=[json.loads(x) for x in (root/'run_full_pools_250'/'inputs.jsonl').open()]
  assert rows==reference,'Sensitivity-model candidate inputs differ'
 else:
  overlap=seen & {x['query'] for x in rows};assert not overlap,(name,'query overlap',len(overlap));seen.update(x['query'] for x in rows)
 grades=collections.Counter(it['grade'] for r in rows for it in r['items'])
 assert dict(grades)==summary['grade_counts']
 for r,o in zip(rows,out):
  assert r['query']==o['query'];assert [it['id'] for it in r['items']]==list(range(1,len(r['items'])+1))
  assert all(it['gold']==('Y' if it['grade'] in ('E','S') else 'N') for it in r['items'])
 for method,ms in summary['plans'].items():
  complete=tp=fp=tn=fn=0
  for r,o in zip(rows,out):
   entry=o['plans'][method];pred=entry['pred'];raw=entry['raw']
   if method=='pointwise':
    recovered=[z['final'] if z['final'] in ('Y','N') else None for z in raw]
   elif name.startswith('run_full_pools_'):
    recovered=[None]*len(r['items'])
    for chunk in raw:
     parser=parse_compact if 'compact' in method else parse_batch
     x=parser(chunk['final'],chunk['size'])
     if x is not None:recovered[chunk['start']:chunk['start']+chunk['size']]=x
   elif raw is None:recovered=[None]*len(r['items'])
   else:
    parser=parse_compact if 'compact' in method else parse_batch
    recovered=parser(raw['final'],len(r['items']))
    if recovered is None:recovered=[None]*len(r['items'])
   assert pred==(recovered if None not in recovered else None),(name,method,r['query'])
   if pred is None:continue
   complete+=1
   for it,y in zip(r['items'],pred):
    g=it['gold'];tp+=g=='Y' and y=='Y';fp+=g=='N' and y=='Y';tn+=g=='N' and y=='N';fn+=g=='Y' and y=='N'
  assert (complete,tp,fp,tn,fn)==tuple(ms[k] for k in ['complete_pools','tp','fp','tn','fn']),(name,method)
  print(name,method,'complete',complete,'confusion',tp,fp,tn,fn,'OK')
print('Data, parser, summary and disjoint query-string checks passed.')
