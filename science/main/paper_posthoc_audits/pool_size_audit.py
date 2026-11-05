#!/usr/bin/env python3
"""Post-hoc pool-size concentration and descriptive errors for recorded ESCI pools."""
import argparse,collections,hashlib,json,re
from pathlib import Path
parser=arg.ArgumentParser(description=__doc__) if False else argparse.ArgumentParser(description=__doc__)
parser.add_argument('--artifact',type=Path,default=Path(__file__).resolve().parents[1])
parser.add_argument('--out',type=Path,help='Optional new JSON output path; never overwrite an existing file.')
args=parser.parse_args();root=args.artifact
files=set()
def read_jsonl(name):
 p=root/name;files.add(name)
 with p.open(encoding='utf8') as f:return [json.loads(x) for x in f if x.strip()]
def read_json(name):
 files.add(name);return json.loads((root/name).read_text(encoding='utf8'))
def digest(name):
 h=hashlib.sha256()
 with (root/name).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def observations(rows,items):
 if len(rows)!=len(items):raise ValueError('Prediction count differs from pool count')
 result=[]
 for row,pred in zip(rows,items):
  if row['query_id']!=pred['query_id'] or len(row['items'])!=len(pred['pred']):raise ValueError('Query or vector mismatch')
  truth=[item['gold'] for item in row['items']]
  if not set(truth)<=set('YN') or not set(pred['pred'])<=set('YN'):raise ValueError('Unexpected label')
  n=len(truth);result.append({'n':n,'reference_ratio':truth.count('Y')/n,
   'absolute_ratio_error':abs(pred['pred'].count('Y')/n-truth.count('Y')/n)})
 return result
def original_vectors(rows,model):
 events=read_jsonl(f'run_esci_selective_holdout_{model}/original_responses.jsonl');index=0;pred=[]
 for row in rows:
  labels=[]
  for start in range(0,len(row['items']),8):
   size=min(8,len(row['items'])-start);event=events[index];index+=1
   if (event['query_id'],event['start'],event['size'])!=(row['query_id'],start,size):raise ValueError('Chunk mismatch')
   expected=r'\{'+','.join(re.escape(json.dumps(str(i)))+r':"[YN]"' for i in range(1,size+1))+r'\}'
   if re.fullmatch(expected,event['final']) is None:raise ValueError('Invalid keyed response')
   obj=json.loads(event['final']);labels.extend(obj[str(i)] for i in range(1,size+1))
  pred.append({'query_id':row['query_id'],'pred':labels})
 if index!=len(events):raise ValueError('Unmatched keyed records')
 return pred
def pointwise_vectors(rows,model):
 events=read_jsonl(f'run_esci_selective_holdout_{model}/pointwise_responses.jsonl');index=0;pred=[]
 for row in rows:
  labels=[]
  for i in range(len(row['items'])):
   event=events[index];index+=1
   if (event['query_id'],event['start'])!=(row['query_id'],i) or event['final'] not in ('Y','N'):raise ValueError('Pointwise response mismatch')
   labels.append(event['final'])
  pred.append({'query_id':row['query_id'],'pred':labels})
 if index!=len(events):raise ValueError('Unmatched pointwise records')
 return pred
def describe(rows):
 if len(rows)!=250 or len({r['query_id'] for r in rows})!=250:raise ValueError('Unexpected sample size or duplicate query')
 hist=collections.Counter(len(r['items']) for r in rows)
 return {'pools':len(rows),'candidates':sum(n*k for n,k in hist.items()),'minimum':min(hist),'maximum':max(hist),'n_16':hist[16],
         'size_histogram':{str(n):hist[n] for n in sorted(hist)}}
original=read_jsonl('run_full_pools_250/inputs.jsonl')
rows=read_jsonl('esci_selective_holdout_selection/inputs.jsonl')
full,holdout=describe(original),describe(rows)
if (full['candidates'],full['n_16'],holdout['candidates'],holdout['n_16'])!=(5160,172,4428,197):raise ValueError('Sample summary mismatch')
preds={'all_Y':[{'query_id':r['query_id'],'pred':['Y']*len(r['items'])} for r in rows],
       'supervised':read_jsonl('run_esci_selective_supervised_reference/predictions.jsonl')}
for model in ('qwen','smollm'):
 preds[model+'_original']=original_vectors(rows,model)
 preds[model+'_selective']=read_jsonl(f'run_esci_selective_holdout_{model}/selective_outputs.jsonl')
 preds[model+'_pointwise']=pointwise_vectors(rows,model)
# Cross-check the keyed positive counts against a separately reconstructed earlier audit.
floor=read_json('posthoc_variable_pool_floor/summary.json')['models']
for model in ('qwen','smollm'):
 recorded=floor[model]['pool_counts']; vectors=preds[model+'_original']
 if len(recorded)!=len(rows):raise ValueError('Prior variable-pool audit length mismatch')
 for row,pred,prior in zip(rows,vectors,recorded):
  if (row['query_id'],len(row['items']),pred['pred'].count('Y')) != (prior['query_id'],prior['n'],prior['original_positive']):
   raise ValueError('Independent positive-count audit mismatch')
obs={name:observations(rows,value) for name,value in preds.items()}
for model in ('qwen','smollm'):
 prior=read_json(f'analysis_esci_selective_holdout_{model}/summary.json')['quality_common']
 for method,prior_key in [('original','original'),('selective','actual_fallback'),('pointwise','pointwise')]:
  actual=sum(x['absolute_ratio_error'] for x in obs[f'{model}_{method}'])/len(rows)
  if abs(actual-prior[prior_key]['ratio_mae'])>1e-12:raise ValueError('Reported aggregate mismatch')
sup=read_json('analysis_esci_selective_supervised_reference/summary.json')['all_250_supervised']['ratio_mae']
if abs(sum(x['absolute_ratio_error'] for x in obs['supervised'])/len(rows)-sup)>1e-12:raise ValueError('Supervised aggregate mismatch')
all_y=read_json('analysis_esci_selective_holdout_qwen/summary.json')['quality_common']['always_Y']['ratio_mae']
if abs(sum(x['absolute_ratio_error'] for x in obs['all_Y'])/len(rows)-all_y)>1e-12:raise ValueError('All-Y aggregate mismatch')
strata={}
for label,predicate in [('exactly_16',lambda n:n==16),('other_sizes',lambda n:n!=16)]:
 indexes=[i for i,r in enumerate(rows) if predicate(len(r['items']))]
 strata[label]={'pools':len(indexes),'candidates':sum(len(rows[i]['items']) for i in indexes),
 'mean_reference_ratio':sum(obs['all_Y'][i]['reference_ratio'] for i in indexes)/len(indexes),
 'ratio_mae':{name:sum(values[i]['absolute_ratio_error'] for i in indexes)/len(indexes) for name,values in obs.items()}}
result={'analysis':'post-hoc descriptive stratification after viewing the dataset and outputs; no new inference, causal interpretation or uncertainty claim',
        'original_sample':full,'new_sample':holdout,'strata_new_sample':strata,
        'input_sha256':{name:digest(name) for name in sorted(files)}}
text=json.dumps(result,indent=2,ensure_ascii=False)+'\n'
if args.out is not None:
 with args.out.open('x',encoding='utf8') as f:f.write(text)
else:print(text,end='')
