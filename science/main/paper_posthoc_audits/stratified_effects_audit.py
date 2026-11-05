#!/usr/bin/env python3
"""Post-hoc paired query-macro MAE effects by reference-label composition."""
import argparse,collections,hashlib,json,re
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--artifact',type=Path,default=Path(__file__).resolve().parents[1])
p.add_argument('--out',type=Path,help='new JSON output path; no overwrite')
a=p.parse_args();root=a.artifact;used=set()
def lines(name):
 used.add(name)
 with (root/name).open(encoding='utf8') as f:return [json.loads(x) for x in f if x.strip()]
def obj(name):used.add(name);return json.loads((root/name).read_text(encoding='utf8'))
def digest(name):
 h=hashlib.sha256()
 with (root/name).open('rb') as f:
  for part in iter(lambda:f.read(1<<20),b''):h.update(part)
 return h.hexdigest()
def original(rows,path):
 events=lines(path);i=0;vectors=[]
 for row in rows:
  pred=[]
  for start in range(0,len(row['items']),8):
   n=min(8,len(row['items'])-start);event=events[i];i+=1
   if (event['query_id'],event['start'],event['size'])!=(row['query_id'],start,n):raise ValueError('Keyed chunk mismatch')
   pat=r'\{'+','.join(re.escape(json.dumps(str(k)))+r':"[YN]"' for k in range(1,n+1))+r'\}'
   if re.fullmatch(pat,event['final']) is None:raise ValueError('Invalid keyed response')
   decoded=json.loads(event['final']);pred.extend(decoded[str(k)] for k in range(1,n+1))
  vectors.append(pred)
 if i!=len(events):raise ValueError('Unexpected keyed events')
 return vectors
specs=[('esci_new','esci_selective_holdout_selection/inputs.jsonl','run_esci_selective_holdout_{model}','analysis_esci_selective_holdout_{model}/summary.json'),
       ('wands_followup','wands_selective_execution_selection/inputs.jsonl','run_wands_selective_execution_{model}_v1','run_wands_selective_execution_{model}_v1/paired_analysis.json')]
results={};seed=20261031;replicates=4000
for data_idx,(dataset,selection,runfmt,summaryfmt) in enumerate(specs):
 rows=lines(selection);results[dataset]={}
 for model_idx,model in enumerate(('qwen','smollm')):
  run=runfmt.format(model=model);baseline=original(rows,run+'/original_responses.jsonl');selected=lines(run+'/selective_outputs.jsonl');prior=obj(summaryfmt.format(model=model))
  if len(rows)!=len(selected) or prior['common_complete_pools']!=len(rows):raise ValueError('Expected common-complete sample')
  delta=[];groups=collections.defaultdict(list)
  for row,x,y in zip(rows,baseline,selected):
   gold=[v['gold'] for v in row['items']];n=len(gold)
   for item in row['items']:
    grade=item['grade']
    allowed={'E','S','C','I'} if dataset=='esci_new' else {'Exact','Partial','Irrelevant'}
    expected=('Y' if grade in ('E','S') else 'N') if dataset=='esci_new' else ('Y' if grade=='Exact' else 'N')
    if grade not in allowed or item['gold']!=expected:raise ValueError('Reference mapping mismatch')
   if not n or y['query_id']!=row['query_id'] or y['pred'] is None or len(y['pred'])!=n or not set(gold)<=set('YN') or not set(y['pred'])<=set('YN'):raise ValueError('Unexpected vectors')
   r=gold.count('Y')/n;old=abs(x.count('Y')/n-r);new=abs(y['pred'].count('Y')/n-r);d=new-old;delta.append(d)
   group='all_positive' if gold.count('Y')==n else 'all_negative' if gold.count('Y')==0 else 'mixed';groups[group].append((old,new,d))
  if abs(sum(delta)/len(delta)-prior['comparisons']['actual_fallback_minus_original']['ratio_mae_difference'])>1e-12:raise ValueError('Stored overall effect mismatch')
  group_summary={}
  for j,(name,records) in enumerate(sorted(groups.items())):
   x=np.asarray(records,dtype=np.float64);n=len(x);rng=np.random.default_rng(seed+100*data_idx+10*model_idx+j)
   draw=rng.integers(0,n,size=(replicates,n));boot=x[draw,2].mean(axis=1);lo,hi=np.quantile(boot,[.025,.975])
   group_summary[name]={'pools':n,'keyed_ratio_mae':float(x[:,0].mean()),'selective_ratio_mae':float(x[:,1].mean()),
     'difference_selective_minus_keyed':float(x[:,2].mean()),'unadjusted_posthoc_query_bootstrap_95':[float(lo),float(hi)],
     'weighted_contribution_to_overall_difference':float(x[:,2].sum()/len(rows)),'seed':seed+100*data_idx+10*model_idx+j}
  if abs(sum(g['weighted_contribution_to_overall_difference'] for g in group_summary.values())-sum(delta)/len(delta))>1e-12:raise ValueError('Group effects do not sum to overall')
  results[dataset][model]={'pools':len(rows),'overall_difference':sum(delta)/len(delta),'groups':group_summary}
report={'description':'Post-hoc strata determined by observed reference labels; not a pre-registered or causal analysis.',
        'bootstrap':'4,000 within-stratum query resamples, unadjusted, conditional on the selected samples; ignores product reuse across queries.',
        'results':results,'input_sha256':{n:digest(n) for n in sorted(used)}}
text=json.dumps(report,indent=2)+'\n'
if a.out:
 with a.out.open('x',encoding='utf8') as f:f.write(text)
else:print(text,end='')
