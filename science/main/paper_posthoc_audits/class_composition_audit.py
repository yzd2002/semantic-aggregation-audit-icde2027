#!/usr/bin/env python3
"""Describe reference-label composition and constant-plan behavior in fixed samples."""
import argparse,collections,hashlib,json
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--artifact',type=Path,default=Path(__file__).resolve().parents[1])
p.add_argument('--out',type=Path,help='new JSON output; never overwrites an existing file')
a=p.parse_args();root=a.artifact;files=set()
def readlines(n):
 files.add(n)
 with (root/n).open(encoding='utf8') as f:return [json.loads(x) for x in f if x.strip()]
def readjson(n):files.add(n);return json.loads((root/n).read_text(encoding='utf8'))
def sha(n):
 h=hashlib.sha256()
 with (root/n).open('rb') as f:
  for v in iter(lambda:f.read(1<<20),b''):h.update(v)
 return h.hexdigest()
paths={'esci_original':'run_full_pools_250/inputs.jsonl','esci_new':'esci_selective_holdout_selection/inputs.jsonl',
       'wands_holdout':'wands_holdout_selection/inputs.jsonl','wands_followup':'wands_selective_execution_selection/inputs.jsonl',
       'trec':'trec_covid_transfer_selection/inputs.jsonl'}
rows={k:readlines(v) for k,v in paths.items()};composition={}
for name,pools in rows.items():
 counts=collections.Counter();grades=collections.Counter();candidates=0
 for row in pools:
  items=row['items']
  if not items:raise ValueError('Empty candidate pool')
  for item in items:
   g,label=item['grade'],item['gold'];grades[str(g)]+=1
   expected=('Y' if g in ('E','S') else 'N') if name.startswith('esci') else ('Y' if g=='Exact' else 'N') if name.startswith('wands') else ('Y' if g==2 else 'N')
   allowed={'E','S','C','I'} if name.startswith('esci') else {'Exact','Partial','Irrelevant'} if name.startswith('wands') else {0,1,2}
   if g not in allowed or label!=expected:raise ValueError('Unexpected grade or binary mapping')
  n=len(items);y=sum(x['gold']=='Y' for x in items);candidates+=n
  counts['all_positive' if y==n else 'all_negative' if y==0 else 'mixed']+=1
 composition[name]={'pools':len(pools),'candidates':candidates,'all_positive':counts['all_positive'],'all_negative':counts['all_negative'],'mixed':counts['mixed'],'grades':dict(sorted(grades.items()))}
expected={'esci_original':(250,73,0),'esci_new':(250,96,0),'wands_holdout':(160,1,60),'wands_followup':(52,0,20),'trec':(50,0,1)}
for name,(n,positive,negative) in expected.items():
 q=composition[name]
 if (q['pools'],q['all_positive'],q['all_negative'])!=(n,positive,negative):raise ValueError('Unexpected composition')
stratified={}
for dataset,source,runfmt,summaryfmt,constant in [
 ('esci_new','esci_new','run_esci_selective_holdout_{model}/selective_outputs.jsonl','analysis_esci_selective_holdout_{model}/summary.json','always_Y'),
 ('wands_followup','wands_followup','run_wands_selective_execution_{model}_v1/selective_outputs.jsonl','run_wands_selective_execution_{model}_v1/paired_analysis.json','always_N')]:
 pools=rows[source];data={}
 for model in ('qwen','smollm'):
  predictions=readlines(runfmt.format(model=model));summary=readjson(summaryfmt.format(model=model))
  if len(predictions)!=len(pools) or summary['common_complete_pools']!=len(pools):raise ValueError('Expected complete aligned pools')
  errors=[]
  for row,pred in zip(pools,predictions):
   truth=[x['gold'] for x in row['items']];v=pred['pred']
   if row['query_id']!=pred['query_id'] or v is None or len(v)!=len(truth) or not set(v)<=set('YN'):raise ValueError('Unexpected prediction')
   errors.append(abs(v.count('Y')/len(v)-truth.count('Y')/len(truth)))
  if abs(sum(errors)/len(errors)-summary['quality_common']['actual_fallback']['ratio_mae'])>1e-12:raise ValueError('Stored selective MAE differs')
  reference_ratios=[sum(x['gold']=='Y' for x in row['items'])/len(row['items']) for row in pools]
  const=sum(1-v if constant=='always_Y' else v for v in reference_ratios)/len(reference_ratios)
  if abs(const-summary['quality_common'][constant]['ratio_mae'])>1e-12:raise ValueError('Stored constant baseline differs')
  data[model]=errors
 groups={}
 for typ in ('all_positive','all_negative','mixed'):
  indices=[]
  for i,row in enumerate(pools):
   n=len(row['items']);y=sum(x['gold']=='Y' for x in row['items']);kind='all_positive' if y==n else 'all_negative' if y==0 else 'mixed'
   if kind==typ:indices.append(i)
  if not indices:continue
  def mean(x):return sum(x)/len(x)
  groups[typ]={'pools':len(indices),'always_Y_ratio_mae':mean([1-sum(x['gold']=='Y' for x in pools[i]['items'])/len(pools[i]['items']) for i in indices]),
   'always_N_ratio_mae':mean([sum(x['gold']=='Y' for x in pools[i]['items'])/len(pools[i]['items']) for i in indices]),
   'selective_ratio_mae':{model:mean([vals[i] for i in indices]) for model,vals in data.items()}}
 stratified[dataset]=groups
result={'interpretation':'post-hoc descriptive grouping of observed labeled pools, not a controlled intervention or a new sample',
        'composition':composition,'stratified':stratified,'input_sha256':{name:sha(name) for name in sorted(files)}}
text=json.dumps(result,indent=2,ensure_ascii=False)+'\n'
if a.out:
 with a.out.open('x',encoding='utf8') as f:f.write(text)
else:print(text,end='')
