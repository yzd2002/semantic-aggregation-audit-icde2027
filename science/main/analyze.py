#!/usr/bin/env python3
"""Recompute paired, query-clustered uncertainty from the recorded test outputs."""
import argparse,json,random,statistics
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,default=Path(__file__).resolve().parent/'run_test_160');p=ap.parse_args().run
rows=[json.loads(x) for x in (p/'inputs.jsonl').open()]
outputs=[json.loads(x) for x in (p/'outputs.jsonl').open()]
assert len(rows)==len(outputs) and all(r['source_line']==o['source_line'] for r,o in zip(rows,outputs))
methods=['pointwise','batch','two_stage']
shared=[i for i,o in enumerate(outputs) if all(o['plans'][m]['pred'] is not None for m in methods)]

def metrics(ids,method):
 tp=fp=fn=tn=0;mae=[]
 for i in ids:
  row=rows[i];pred=outputs[i]['plans'][method]['pred'];gold=[x['gold'] for x in row['items']]
  assert len(pred)==len(gold)
  mae.append(abs(sum(x=='Y' for x in gold)/len(gold)-sum(x=='Y' for x in pred)/len(pred)))
  for g,y in zip(gold,pred):
   tp+=g=='Y' and y=='Y';fp+=g=='N' and y=='Y';fn+=g=='Y' and y=='N';tn+=g=='N' and y=='N'
 return {'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0,'mae':statistics.mean(mae),'accuracy':(tp+tn)/(tp+fp+fn+tn)}

def ci(vals):
 vals.sort();return [vals[int(0.025*(len(vals)-1))],vals[int(0.975*(len(vals)-1))]]
rng=random.Random(20271031);B=4000
traces={m:{'f1':[],'mae':[]} for m in methods};differences={m:{'f1':[],'mae':[]} for m in methods[1:]}
for _ in range(B):
 ix=[rng.choice(shared) for _ in shared];v={m:metrics(ix,m) for m in methods}
 for m in methods:
  for k in ('f1','mae'): traces[m][k].append(v[m][k])
 for m in methods[1:]:
  for k in ('f1','mae'):differences[m][k].append(v[m][k]-v['pointwise'][k])
result={'description':'Descriptive 95% query-cluster bootstrap on the common complete subset; not population prevalence. Differences are method minus pointwise.',
        'common_complete_pools':len(shared),'resamples':B,'seed':20271031,'methods':{},'differences':{}}
for m in methods:
 result['methods'][m]={**metrics(shared,m),'f1_ci':ci(traces[m]['f1']),'mae_ci':ci(traces[m]['mae'])}
for m in methods[1:]:
 result['differences'][m]={k+'_ci':ci(differences[m][k]) for k in ('f1','mae')}
(p/'paired_analysis.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
