#!/usr/bin/env python3
"""Post-hoc 96-query composition audit reconstructed from recorded responses."""
import argparse,collections,hashlib,json,re
from pathlib import Path
import numpy as np

def read(path):
 with path.open(encoding='utf8') as f:return [json.loads(x) for x in f if x.strip()]
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for x in iter(lambda:f.read(1024*1024),b''):h.update(x)
 return h.hexdigest()
def close(x,y,label):
 if abs(float(x)-float(y))>1e-12:raise ValueError(label+' mismatch')
def main():
 p=argparse.ArgumentParser();p.add_argument('--artifact',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();r=a.artifact
 if a.out.exists():raise FileExistsError(a.out)
 src={'selection':'wands_fallback_validation_selection/inputs.jsonl','original':'run_wands_fallback_validation_qwen_v1/original_responses.jsonl',
 'reverse':'run_wands_fallback_validation_qwen_v1/stable_reverse_responses.jsonl','pointwise':'run_wands_fallback_validation_qwen_v1/pointwise_responses.jsonl',
 'paired':'run_wands_fallback_validation_qwen_v1/paired_analysis.json','saved_reference':'run_wands_supervised_reference/predictions.jsonl'}
 rows=read(r/src['selection']);old=read(r/src['original']);rev=read(r/src['reverse']);point=read(r/src['pointwise']);prior=json.loads((r/src['paired']).read_text());saved=[x for x in read(r/src['saved_reference']) if x['sample']=='fallback96']
 if len(rows)!=96 or len(old)!=384 or len(rev)!=384 or len(point)!=3072 or len(saved)!=96:raise ValueError('Unexpected source lengths')
 if prior['common_complete_pools']!=96 or prior['common_query_ids']!=[x['query_id'] for x in rows] or any(x['query_id']!=row['query_id'] for row,x in zip(rows,saved)):raise ValueError('Query alignment')
 pattern=r'\{'+','.join(re.escape(json.dumps(str(i)))+r':"[YN]"' for i in range(1,9))+r'\}'
 groups=collections.defaultdict(list);allres=[];routed_total=0
 for q,row in enumerate(rows):
  if len(row['items'])!=32:raise ValueError('Unexpected pool size')
  gold=[]
  for it in row['items']:
   if it['grade'] not in ('Exact','Partial','Irrelevant') or it['gold']!=('Y' if it['grade']=='Exact' else 'N'):raise ValueError('Gold mapping')
   gold.append(it['gold'])
  original=[];reverse=[]
  for c in range(4):
   for events,order,dst in ((old,list(range(1,9)),original),(rev,list(range(8,0,-1)),reverse)):
    event=events[q*4+c]
    if (event['query_id'],event['start'],event['size'],event['presented_ids'])!=(row['query_id'],8*c,8,order) or not re.fullmatch(pattern,event['final']):raise ValueError('Keyed response mismatch')
    val=json.loads(event['final']);dst.extend(val[str(k)] for k in range(1,9))
  block=point[32*q:32*q+32];helper=[]
  for i,event in enumerate(block):
   if (event['query_id'],event['start'],event['size'],event['presented_ids'])!=(row['query_id'],i,1,None) or event['final'] not in ('Y','N'):raise ValueError('Pointwise response mismatch')
   helper.append(event['final'])
  words=set(re.findall(r'\w+',row['query'].casefold()));lex=['Y' if words and words.issubset(set(re.findall(r'\w+',it['text'].casefold()))) else 'N' for it in row['items']]
  select=[v if v==w else h for v,w,h in zip(original,reverse,helper)];lexselect=[v if v==w else h for v,w,h in zip(original,reverse,lex)]
  if select!=saved[q]['qwen'] or lex!=saved[q]['lexical']:raise ValueError('Saved reference predictions mismatch')
  nroute=sum(v!=w for v,w in zip(original,reverse));routed_total+=nroute
  count=gold.count('Y');group='all_negative' if count==0 else 'all_positive' if count==32 else 'mixed'
  def err(pred):return abs(pred.count('Y')-count)/32
  o=err(original);f=err(select);l=err(lexselect);h=err(helper);allres.append((o,f,l,h,nroute));groups[group].append((o,f,l,h,nroute))
 if routed_total!=prior['disagreement']['candidates']:raise ValueError('Disagreement count mismatch')
 overall=np.asarray(allres,dtype=float)
 for i,n in enumerate(('original','pointwise_fallback','lexical_fallback','pointwise')):close(overall[:,i].mean(),prior['quality_common'][n]['ratio_mae'],n+' MAE')
 for i,n in enumerate(('pointwise_fallback','lexical_fallback','pointwise'),1):close((overall[:,i]-overall[:,0]).mean(),prior['comparisons'][n+'_minus_original']['ratio_mae_difference'],n+' delta')
 results={};group_order=('all_negative','mixed','all_positive')
 for j,name in enumerate(group_order):
  if name not in groups:continue
  x=np.asarray(groups[name],dtype=float);n=len(x);delta=x[:,1]-x[:,0];seed=20271224+j
  draw=np.random.default_rng(seed).integers(0,n,size=(4000,n));ci=np.quantile(delta[draw].mean(axis=1),[.025,.975])
  results[name]={'pools':n,'keyed_mae':float(x[:,0].mean()),'pointwise_fallback_mae':float(x[:,1].mean()),'lexical_fallback_mae':float(x[:,2].mean()),'all_pointwise_mae':float(x[:,3].mean()),'fallback_minus_keyed':float(delta.mean()),'weighted_contribution_to_overall':float(delta.sum()/len(rows)),'routed_candidates':int(x[:,4].sum()),'routed_fraction':float(x[:,4].sum()/(n*32)),'bootstrap_seed':seed,'bootstrap_draws':4000,'unadjusted_posthoc_query_bootstrap_95':[float(z) for z in ci]}
 if sum(g['pools'] for g in results.values())!=96 or sum(g['routed_candidates'] for g in results.values())!=routed_total:raise ValueError('Group totals mismatch')
 close(sum(g['weighted_contribution_to_overall'] for g in results.values()),(overall[:,1]-overall[:,0]).mean(),'weighted contribution')
 out={'scope':'post-hoc analysis of exposed labeled pools and saved traces, not an independent or causal study',
      'contrast':'trace-derived pointwise fallback minus keyed query-macro ratio MAE',
      'overall':{'pools':96,'routed_candidates':routed_total,'keyed_mae':float(overall[:,0].mean()),'pointwise_fallback_mae':float(overall[:,1].mean()),'difference':float((overall[:,1]-overall[:,0]).mean())},
      'groups':results,'input_sha256':{name:sha(r/path) for name,path in src.items()}}
 with a.out.open('x',encoding='utf8') as f:json.dump(out,f,indent=2);f.write('\n')
 print(json.dumps({'overall':out['overall'],'groups':results},indent=2))
if __name__=='__main__':main()
