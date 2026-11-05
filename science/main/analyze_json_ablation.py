#!/usr/bin/env python3
"""Paired JSON plan contrast on pools complete under both JSON plans."""
import argparse,json
from pathlib import Path
import numpy as np
ap=argparse.ArgumentParser();ap.add_argument('--run',default='run_full_pools_250');args=ap.parse_args()
p=Path(__file__).resolve().parent/args.run
r=[json.loads(x) for x in (p/'inputs.jsonl').open()];o=[json.loads(x) for x in (p/'outputs.jsonl').open()]
keys=['batch_json','two_stage_json'];ix=[i for i,x in enumerate(o) if all(x['plans'][k]['pred'] is not None for k in keys)]
result={'common_pools':len(ix),'description':'Post-hoc paired comparison of indexed JSON plans, conditioned on both completing. Two-stage adds specification text and calls.'}
if ix:
 rng=np.random.default_rng(20271106);draw=rng.integers(0,len(ix),size=(4000,len(ix)));metrics={};boots={}
 for k in keys:
  vals=[]
  for i in ix:
   gold=[x['gold'] for x in r[i]['items']];pred=o[i]['plans'][k]['pred']
   tp=sum(g=='Y' and y=='Y' for g,y in zip(gold,pred));fp=sum(g=='N' and y=='Y' for g,y in zip(gold,pred));fn=sum(g=='Y' and y=='N' for g,y in zip(gold,pred));tn=sum(g=='N' and y=='N' for g,y in zip(gold,pred))
   mae=abs(sum(x=='Y' for x in gold)/len(gold)-sum(x=='Y' for x in pred)/len(pred));vals.append((tp,fp,fn,tn,mae))
  v=np.array(vals,dtype=float);tp,fp,fn,tn=v[:,:4].sum(axis=0);s=v[draw,:4].sum(axis=1)
  boots[k]={'f1':2*s[:,0]/(2*s[:,0]+s[:,1]+s[:,2]),'mae':v[draw,4].mean(axis=1)}
  metrics[k]={'f1':float(2*tp/(2*tp+fp+fn)),'mae':float(v[:,4].mean())}
 result['metrics']=metrics
 result['two_stage_minus_batch']={key+'_ci':[float(v) for v in np.quantile(boots[keys[1]][key]-boots[keys[0]][key],[.025,.975])] for key in ['f1','mae']}
(p/'json_ablation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
