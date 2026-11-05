#!/usr/bin/env python3
"""Post-hoc crossed query/product weighting for saved WANDS ratio references."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
SETS=[('holdout160','wands_holdout_selection','run_keyed_order_qwen_v1',20261031,160),
      ('fallback96','wands_fallback_validation_selection','run_wands_fallback_validation_qwen_v1',20271128,96),
      ('selective52','wands_selective_execution_selection','run_wands_selective_execution_qwen_v1',20271201,52)]
def lines(p):
 with p.open(encoding='utf8') as f:return [json.loads(x) for x in f if x.strip()]
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for x in iter(lambda:f.read(1<<20),b''):h.update(x)
 return h.hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--artifact',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();root=a.artifact.resolve();sys.path.insert(0,str(root))
 from analyze_order_fallback_validation import keyed_vectors
 prior=root/'run_wands_ratio_baseline/summary.json';predpath=root/'run_wands_supervised_reference/predictions.jsonl';pred=lines(predpath);before=json.loads(prior.read_text());threshold=before['threshold_selection']['threshold']
 if threshold!=0.5837454647017262:raise ValueError('Unexpected frozen threshold')
 result={'scope':'post-hoc query–product reweighting; not a fixed-pool confidence interval','samples':{},'input_sha256':{}}
 source=[prior,predpath,root/'analyze_order_fallback_validation.py',root/'run_keyed_order_audit.py']
 for name,selection,run,seed,n in SETS:
  inp=root/selection/'inputs.jsonl';source.append(inp);rows=lines(inp);stored=[x for x in pred if x['sample']==name]
  if len(rows)!=n or len(stored)!=n or any(r['query_id']!=s['query_id'] or len(r['items'])!=32 for r,s in zip(rows,stored)):raise ValueError(name+' alignment')
  if any(len(set(int(x['product_id']) for x in r['items']))!=32 for r in rows):raise ValueError(name+' duplicate product within query')
  product_ids=sorted({int(x['product_id']) for r in rows for x in r['items']});index={v:i for i,v in enumerate(product_ids)}
  qidx=np.repeat(np.arange(n),32);pidx=np.array([index[int(x['product_id'])] for r in rows for x in r['items']]);truth=np.array([x['gold']=='Y' for r in rows for x in r['items']],float)
  for r in rows:
   if any(x['gold'] not in ('Y','N') or (x['gold']=='Y')!=(x['grade']=='Exact') for x in r['items']):raise ValueError('Gold mismatch')
  qwen=np.array([y=='Y' for s in stored for y in s['qwen']],float);lex=np.array([y=='Y' for s in stored for y in s['lexical']],float)
  score=np.array([y for s in stored for y in s['positive_probability']],float)
  if any(len(s['qwen'])!=32 or len(s['lexical'])!=32 or len(s['positive_probability'])!=32 for s in stored) or not np.isfinite(score).all() or (score<0).any() or (score>1).any():raise ValueError(name+' bad outputs')
  hard=(score>=threshold).astype(float);methods={'qwen':qwen,'lexical':lex,'hard':hard,'score':score}
  fixed_ref=truth.reshape(n,32).mean(axis=1)
  for k,record in [('qwen',before['samples'][name]['binary_metrics']['qwen']),('hard',before['samples'][name]['ratio_results']['ratio_threshold']),('score',before['samples'][name]['ratio_results']['score_mean']),('lexical',before['samples'][name]['binary_metrics']['lexical'])]:
   observed=float(np.abs(methods[k].reshape(n,32).mean(axis=1)-fixed_ref).mean())
   if abs(observed-record['ratio_mae'])>1e-12:raise ValueError(name+' fixed-pool mismatch '+k)
  if name!='holdout160':
   originalfile=root/run/'original_responses.jsonl';source.append(originalfile);original=keyed_vectors(rows,lines(originalfile))
   if any(x is None or len(x)!=32 for x in original):raise ValueError(name+' invalid keyed output')
   methods['original']=np.array([y=='Y' for row in original for y in row],float)
   previous=root/run/'two_way_analysis.json';source.append(previous);priorrange=json.loads(previous.read_text())
  rng=np.random.default_rng(seed);contrasts={k:[] for k in ['hard_minus_qwen','hard_minus_lexical','lexical_minus_qwen','score_minus_qwen']}
  if name!='holdout160':contrasts['qwen_minus_keyed']=[]
  for _ in range(4000):
   qw=np.bincount(rng.integers(0,n,n),minlength=n);pw=np.bincount(rng.integers(0,len(product_ids),len(product_ids)),minlength=len(product_ids));weight=pw[pidx]
   den=np.bincount(qidx,weights=weight,minlength=n)
   if np.any((qw>0)&(den==0)):raise RuntimeError(name+' selected query without product weight')
   safe=np.where(den==0,1,den);ref=np.bincount(qidx,weights=weight*truth,minlength=n)/safe
   err={k:np.abs(np.bincount(qidx,weights=weight*v,minlength=n)/safe-ref) for k,v in methods.items()}
   for label,left,right in [('hard_minus_qwen','hard','qwen'),('hard_minus_lexical','hard','lexical'),('lexical_minus_qwen','lexical','qwen'),('score_minus_qwen','score','qwen')]:
    contrasts[label].append(float(np.dot(qw,err[left]-err[right])/qw.sum()))
   if name!='holdout160':contrasts['qwen_minus_keyed'].append(float(np.dot(qw,err['qwen']-err['original'])/qw.sum()))
  output={k:[float(x) for x in np.quantile(v,[.025,.975])] for k,v in contrasts.items()}
  if name!='holdout160':
   key='pointwise_fallback_minus_original_ratio_mae_query_product_ci' if name=='fallback96' else 'actual_fallback_minus_original_ratio_mae_query_product_ci'
   if not np.allclose(output['qwen_minus_keyed'],priorrange[key],rtol=0,atol=1e-12):raise ValueError(name+' prior crossed analysis mismatch')
  result['samples'][name]={'queries':n,'distinct_products':len(product_ids),'repeated_product_ids':int(sum(x>1 for x in np.bincount(pidx))),
   'seed':seed,'draws':4000,'ranges':output}
 for p in source:result['input_sha256'][str(p.relative_to(root))]=sha(p)
 if a.out.exists() or a.out.is_symlink():raise FileExistsError(a.out)
 a.out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
 for name,*_ in SETS:
  x=result['samples'][name];print(name,'repeated products',x['repeated_product_ids'],'threshold − Qwen:',x['ranges']['hard_minus_qwen'],'lexical − Qwen:',x['ranges']['lexical_minus_qwen'])
if __name__=='__main__':main()
