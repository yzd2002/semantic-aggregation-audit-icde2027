#!/usr/bin/env python3
"""Compare model/plan metrics on pools completed by all plans and models."""
import argparse,json
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parent
ap=argparse.ArgumentParser();ap.add_argument('--qwen-run',type=Path,default=root/'run_full_pools_250');ap.add_argument('--smollm-run',type=Path,default=root/'run_full_pools_smollm_250');args=ap.parse_args()
names={'qwen':args.qwen_run,'smollm':args.smollm_run}
methods=['pointwise','batch_json','two_stage_json','two_stage_compact_fallback']
rows={k:[json.loads(s) for s in (root/p/'inputs.jsonl').open()] for k,p in names.items()}
outs={k:[json.loads(s) for s in (root/p/'outputs.jsonl').open()] for k,p in names.items()}
assert rows['qwen']==rows['smollm'],'Different candidate inputs across models'
shared=[i for i in range(len(rows['qwen'])) if all(outs[k][i]['plans'][m]['pred'] is not None for k in names for m in methods)]
report={'description':'Post-result metrics on the pre-specified intersection of pools completed by all four plans for both models; conditional on completion and not a causal model comparison.',
        'attempted_pools':len(rows['qwen']),'common_pools':len(shared),'models':{}}
if shared:
 rng=np.random.default_rng(20271105);draw=rng.integers(0,len(shared),size=(4000,len(shared)))
 for model in names:
  report['models'][model]={}
  for m in methods:
   a=[]
   for i in shared:
    r=rows[model][i];gold=[x['gold'] for x in r['items']];pred=outs[model][i]['plans'][m]['pred']
    assert len(gold)==len(pred)
    tp=sum(g=='Y' and y=='Y' for g,y in zip(gold,pred));fp=sum(g=='N' and y=='Y' for g,y in zip(gold,pred))
    fn=sum(g=='Y' and y=='N' for g,y in zip(gold,pred));tn=sum(g=='N' and y=='N' for g,y in zip(gold,pred))
    mae=abs(sum(y=='Y' for y in pred)/len(pred)-sum(g=='Y' for g in gold)/len(gold))
    a.append((tp,fp,fn,tn,mae))
   a=np.array(a,dtype=float);tp,fp,fn,tn=a[:,:4].sum(axis=0)
   v=a[draw,:4].sum(axis=1);den=2*v[:,0]+v[:,1]+v[:,2];f1=np.divide(2*v[:,0],den,out=np.zeros_like(den),where=den>0);mae=a[draw,4].mean(axis=1)
   report['models'][model][m]={'f1':float(2*tp/(2*tp+fp+fn)) if 2*tp+fp+fn else 0.0,'balanced_accuracy':float((tp/(tp+fn)+tn/(tn+fp))/2) if (tp+fn) and (tn+fp) else None,
      'ratio_mae':float(a[:,4].mean()),'f1_ci':[float(x) for x in np.quantile(f1,[.025,.975])],
      'mae_ci':[float(x) for x in np.quantile(mae,[.025,.975])]}
(names['smollm']/'crossmodel_analysis.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
