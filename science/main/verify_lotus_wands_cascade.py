#!/usr/bin/env python3
"""Independently reparse recorded LOTUS WANDS cascade, samples and metrics."""
import argparse,hashlib,json,math,re
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
SYSTEM=('The user will provide a claim and some relevant context.\n'
        '    Your job is to determine whether the claim is true for the given context.\n'
        '     Use the following format to provide your answer:\n'
        '            Answer: <Your answer here. The answer should be either True or False>\n'
        '            ')
CLAIM=('The Product fully matches the shopper query Query, including the requested product type '
       'and every stated essential modifier; use only supplied information and do not assume unknown properties.')
def lines(p):
 with p.open(encoding='utf8') as f:return [json.loads(x) for x in f]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def message_hash(m):return hashlib.sha256(json.dumps(m,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def parsed(text):
 a=text.split('Answer:',1)[1].strip() if 'Answer:' in text else text.strip();a=a.lower()
 return True if 'true' in a else False if 'false' in a else True
def audit(texts):
 counts={'neither':0,'both':0,'empty':0}
 for text in texts:
  a=text.split('Answer:',1)[1].strip() if 'Answer:' in text else text.strip();lo=a.lower();t='true' in lo;f='false' in lo
  counts['neither']+=not t and not f;counts['both']+=t and f;counts['empty']+=not bool(a)
 return counts
def expected_message(row,cols):
 text=''.join(f"[{c.capitalize()}]: «{row[c]}»\n" for c in cols)
 return [{'role':'system','content':SYSTEM},{'role':'user','content':f'Context:\n{text}\n\nClaim: {CLAIM}'}]
def independent_threshold(scores,answers,factors,recall_target,precision_target,failure_probability):
 pairs=sorted(zip(scores,answers,factors),key=lambda x:x[0],reverse=True);n=len(pairs);assert n
 def rec(pos,neg):
  accepted=[x for x in pairs if x[0]>=pos or x[0]<=neg];sent=[x for x in pairs if x[0]<pos and x[0]>neg]
  total=sum(x[1]*x[2] for x in pairs)
  return (sum(1 for x in accepted if x[0]>=pos and x[1])+sum(x[1]*x[2] for x in sent))/total if total>0 else 0.0
 def tneg(pos,target):return max((x[0] for x in pairs[::-1] if rec(pos,x[0])>=target),default=0)
 neg=tneg(1.0,recall_target)
 z1=[int(x[1])*x[2] for x in pairs if x[0]>=neg];z2=[int(x[1])*x[2] for x in pairs if x[0]<neg]
 m1=float(np.mean(z1)) if z1 else 0.;s1=float(np.std(z1)) if z1 else 0.
 m2=float(np.mean(z2)) if z2 else 0.;s2=float(np.std(z2)) if z2 else 0.
 upper=m1+s1/math.sqrt(n)*math.sqrt(2*np.log(1/(failure_probability/2)))
 lower=m2-s2/math.sqrt(n)*math.sqrt(2*np.log(1/(failure_probability/2)))
 target=1. if upper+lower==0 else min(1.,upper/(upper+lower))
 neg=tneg(1.0,target);possible=[1.0]
 for x in pairs:
  vals=[int(y[1]) for y in pairs if y[0]>=x[0]]
  mean=float(np.mean(vals)) if vals else 0.;std=float(np.std(vals)) if vals else 0.
  bound=mean-std/math.sqrt(len(vals))*math.sqrt(2*np.log(1/(failure_probability/n)))
  if bound>precision_target:possible.append(x[0])
 return max(neg,min(possible)),neg
def query_counts(rows,pred):
 out=[]
 for row,v in zip(rows,pred):
  y=np.asarray([x['gold']=='Y' for x in row['items']]);p=np.asarray([x=='Y' for x in v]);assert len(y)==len(p)==32
  out.append([int((y&p).sum()),int((~y&p).sum()),int((~y&~p).sum()),int((y&~p).sum()),abs(int(p.sum())-int(y.sum()))/32])
 return np.asarray(out,dtype=float)
def quality(z):
 tp,fp,tn,fn=z[:,:4].sum(axis=0);den=2*tp+fp+fn
 return {'tp':int(tp),'fp':int(fp),'tn':int(tn),'fn':int(fn),'f1':float(2*tp/den) if den else 0.,
  'balanced_accuracy':float((tp/(tp+fn)+tn/(tn+fp))/2),'ratio_mae':float(z[:,4].mean())}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);ap.add_argument('--main-model');ap.add_argument('--helper-model');a=ap.parse_args()
 if bool(a.main_model)!=bool(a.helper_model):raise ValueError('Provide both model tokenizers or neither')
 pre=json.loads((a.run/'preflight.json').read_text());summary=json.loads((a.run/'summary.json').read_text());sampling=json.loads((a.run/'sampling_trace.json').read_text())
 rows=lines(HERE/'wands_selective_execution_selection/inputs.jsonl');prior=lines(HERE/'run_wands_selective_execution_qwen_v1/selective_outputs.jsonl');shuffled=lines(a.run/'shuffled_rows.jsonl');decisions=lines(a.run/'decisions.jsonl');events=lines(a.run/'model_requests.jsonl')
 assert pre['input_sha256']==sha(HERE/'wands_selective_execution_selection/inputs.jsonl') and pre['prior_selective_sha256']==sha(HERE/'run_wands_selective_execution_qwen_v1/selective_outputs.jsonl')
 assert len(rows)==len(prior)==52 and pre['rows']==len(shuffled)==len(decisions)==1664 and pre['pools']==52
 assert pre['lotus_version']=='1.2.4' and pre['wheel_sha256']=='4803bcba7a30d50095b0c6fc5036942ebd1cfceb42a3f907454892c11171fa89' and pre['installed_lotus_python_files_verified']==69
 flat=[{'query_id':r['query_id'],'candidate_id':it['id'],'pool_index':i,'candidate_index':j,'query':r['query'],'product':it['text']} for i,r in enumerate(rows) for j,it in enumerate(r['items'])]
 perm=np.random.default_rng(20271305).permutation(len(flat));assert shuffled==[flat[int(i)] for i in perm]
 cols=pre['columns'];assert set(cols)=={'query','product'} and len(cols)==2 and pre['formatted_instruction']==CLAIM
 groups={}
 for e in events:groups.setdefault((e['phase'],e['role'],e['batch']),[]).append(e)
 def group(phase,role,batch,indices):
  v=groups.pop((phase,role,batch));assert len(v)==len(indices)
  for i,(e,rowidx) in enumerate(zip(v,indices)):
   assert e['ordinal']==i and e['messages']==expected_message(shuffled[rowidx],cols) and e['prompt_sha256']==message_hash(e['messages'])
   assert isinstance(e['output'],str) and e['usage'] is not None and all(isinstance(e['usage'][k],int) and e['usage'][k]>=0 for k in ('prompt_tokens','completion_tokens','total_tokens'))
  return v
 n=len(flat);allidx=list(range(n));base=group('base','main',0,allidx);helper=group('cascade','helper',0,allidx)
 assert all(e['logprobs'] is None for e in base) and all(isinstance(e['logprobs'],list) for e in helper)
 raw_probs=[];first_positions={};fallback=0
 for e in helper:
  probs=None;tokens=[x['token'] for x in e['logprobs']]
  for pos,item in enumerate(e['logprobs']):
   top={v['token']:np.exp(v['logprob']) for v in item['top_logprobs']}
   if 'True' in top and 'False' in top:
    probs=float(top['True']/(top['True']+top['False']));first_positions[str(pos)]=first_positions.get(str(pos),0)+1;break
  if probs is None:probs=1. if 'True' in tokens else 0.;fallback+=1
  raw_probs.append(probs)
 q=np.percentile(raw_probs,np.linspace(0,100,51));scores=np.clip((np.digitize(raw_probs,q)-1)/50,0,1)
 assert np.allclose(scores,sampling['proxy_scores'],rtol=0,atol=1e-12)
 diag=summary['proxy_diagnostics'];assert diag['missing_both_label_logprobs']==fallback and diag['first_both_label_position_counts']==first_positions
 assert diag['helper_raw']==audit([e['output'] for e in helper])
 weights=np.sqrt(scores);weights=.9*weights/weights.sum()+.1*np.ones(n)/n
 prob=weights/weights.sum();rng=np.random.RandomState(20271231);indices=rng.choice(np.arange(n),int(.1*n),p=prob)
 factors=(1/n)/weights
 assert list(indices)==sampling['sample_indices'] and np.allclose(factors,sampling['correction_factors'],rtol=0,atol=1e-12)
 sample=group('cascade','main',0,list(indices));assert all(e['logprobs'] is None for e in sample)
 t=summary['routing']['thresholds'];assert t['recall_target']==.8 and t['precision_target']==.8 and t['failure_probability']==.2 and t['sampling_percentage']==.1
 assert t['cascade_IS_random_seed']==20271231 and t['cascade_IS_max_sample_range']==n and t['cascade_IS_weight']==.9 and t['cascade_num_calibration_quantiles']==50
 pos,neg=independent_threshold(scores[indices],[parsed(e['output']) for e in sample],factors[indices],.8,.8,.2)
 assert abs(pos-t['filter_pos_cascade_threshold'])<1e-12 and abs(neg-t['filter_neg_cascade_threshold'])<1e-12
 helperidx=[i for i,s in enumerate(scores) if s>=pos or s<=neg];mainidx=[i for i,s in enumerate(scores) if s<pos and s>neg]
 assert len(helperidx)==summary['routing']['helper'] and len(mainidx)==summary['routing']['main'] and len(helperidx)+len(mainidx)==n
 routed=group('cascade','main',1,mainidx) if mainidx else []
 assert all(e['logprobs'] is None for e in routed) and not groups
 main_outputs={i:e['output'] for i,e in zip(mainidx,routed)}
 vectors={'lotus_base':[[None]*32 for _ in rows],'lotus_cascade':[[None]*32 for _ in rows],'selective':[r['pred'] for r in prior]}
 terms=lambda x:set(re.findall(r'\w+',x.casefold()))
 vectors['lexical']=[['Y' if terms(r['query']) and terms(r['query'])<=terms(it['text']) else 'N' for it in r['items']] for r in rows]
 for i,(row,d) in enumerate(zip(shuffled,decisions)):
  expected= True if scores[i]>=pos else False if scores[i]<=neg else parsed(main_outputs[i])
  raw=helper[i]['output'] if i in set(helperidx) else main_outputs[i]
  assert d=={'shuffled_index':i,'query_id':row['query_id'],'candidate_id':row['candidate_id'],'pool_index':row['pool_index'],'candidate_index':row['candidate_index'],
             'base':parsed(base[i]['output']),'cascade':expected,'base_raw':base[i]['output'],'cascade_raw':raw}
  vectors['lotus_base'][row['pool_index']][row['candidate_index']]='Y' if d['base'] else 'N'
  vectors['lotus_cascade'][row['pool_index']][row['candidate_index']]='Y' if d['cascade'] else 'N'
 assert summary['raw_diagnostics']['base']==audit([e['output'] for e in base]) and summary['raw_diagnostics']['cascade']==audit([d['cascade_raw'] for d in decisions])
 assert summary['logical_requests']=={'base_main':n,'cascade_helper':n,'cascade_calibration_main':len(indices),'cascade_routed_main':len(mainidx)}
 for key,ev in [('base_main',base),('cascade_main',sample+routed),('cascade_helper',helper)]:
  sums={k:sum(e['usage'][k] for e in ev) for k in ('prompt_tokens','completion_tokens','total_tokens')};stored=summary['usage'][key]
  assert stored['physical']==stored['virtual']==sums and stored['cache_hits']==stored['operator_cache_hits']==0
 mats={k:query_counts(rows,v) for k,v in vectors.items()}
 for k,m in mats.items():
  for f,v in quality(m).items():assert abs(v-summary['quality'][k][f])<1e-12,(k,f)
 draw=np.random.default_rng(20271306).integers(0,len(rows),size=(4000,len(rows)))
 def bootf1(m):
  b=m[draw,:4].sum(axis=1);den=2*b[:,0]+b[:,1]+b[:,3]
  return np.divide(2*b[:,0],den,out=np.zeros(len(draw)),where=den!=0)
 diff=bootf1(mats['lotus_cascade'])-bootf1(mats['lotus_base']);dm=mats['lotus_cascade'][draw,4].mean(axis=1)-mats['lotus_base'][draw,4].mean(axis=1);c=summary['cascade_minus_lotus_base']
 assert np.allclose(np.quantile(diff,[.025,.975]),c['f1_difference_ci'],rtol=0,atol=1e-12)
 assert np.allclose(np.quantile(dm,[.025,.975]),c['ratio_mae_difference_ci'],rtol=0,atol=1e-12)
 assert abs(c['f1_difference']-(quality(mats['lotus_cascade'])['f1']-quality(mats['lotus_base'])['f1']))<1e-12
 assert abs(c['ratio_mae_difference']-(quality(mats['lotus_cascade'])['ratio_mae']-quality(mats['lotus_base'])['ratio_mae']))<1e-12
 x=[v for row in vectors['lotus_base'] for v in row];y=[v for row in vectors['lotus_cascade'] for v in row]
 tp=sum(a==b=='Y' for a,b in zip(x,y));fp=sum(a=='N' and b=='Y' for a,b in zip(x,y));fn=sum(a=='Y' and b=='N' for a,b in zip(x,y));compare=summary['cascade_vs_separately_run_base']
 assert compare['tp']==tp and compare['fp']==fp and compare['fn']==fn
 assert abs(compare['candidate_agreement']-sum(a==b for a,b in zip(x,y))/n)<1e-12
 assert compare['precision']==(tp/(tp+fp) if tp+fp else None) and compare['recall']==(tp/(tp+fn) if tp+fn else None)
 if a.main_model:
  from transformers import AutoTokenizer
  for key,path in [('main',a.main_model),('helper',a.helper_model)]:
   tok=AutoTokenizer.from_pretrained(path,local_files_only=True)
   assert all(len(tok.encode(x,add_special_tokens=False))==1 for x in ('True','False'))
   longest=max(len(tok.apply_chat_template(e['messages'],tokenize=True,add_generation_prompt=True))+16 for e in base)
   assert longest==pre['max_prompt_plus_output_tokens'][key] and longest<=4096
 print('All row/prompt associations, raw parser outcomes, proxy probabilities, sampling, thresholds, routing, per-request usage, metrics and bootstrap verified.')
 print('Helper exact-label fallback count:',fallback,'first positions:',first_positions,'tokenizer context checked:',bool(a.main_model))
if __name__=='__main__':main()
