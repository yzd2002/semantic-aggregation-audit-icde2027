#!/usr/bin/env python3
"""Post-hoc pool-error and ratio-residual calibration on saved WANDS traces."""
import argparse,hashlib,json,re
from pathlib import Path

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for z in iter(lambda:f.read(1024*1024),b''):h.update(z)
 return h.hexdigest()
def lines(p):
 with p.open(encoding='utf8') as f:return [json.loads(x) for x in f if x.strip()]
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--artifact',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();r=a.artifact
 if a.out.exists():raise FileExistsError(a.out)
 specs=[('cal160','wands_holdout_selection/inputs.jsonl','run_keyed_order_qwen_v1/original_outputs.jsonl','run_keyed_order_qwen_v1/paired_analysis.json','keyed_original',160),('test96','wands_fallback_validation_selection/inputs.jsonl','run_wands_fallback_validation_qwen_v1/original_responses.jsonl','run_wands_fallback_validation_qwen_v1/paired_analysis.json','original',96),('test52','wands_selective_execution_selection/inputs.jsonl','run_wands_selective_execution_qwen_v1/original_responses.jsonl','run_wands_selective_execution_qwen_v1/paired_analysis.json','original',52)]
 pat=r'\{'+','.join(re.escape(json.dumps(str(i)))+r':"[YN]"' for i in range(1,9))+r'\}'
 allsets={};provenance={}
 for name,source,events,summary,qual,n in specs:
  rows=lines(r/source);raw=lines(r/events);prior=json.loads((r/summary).read_text())
  if len(rows)!=n or len(raw)!=(n if name=='cal160' else n*4) or prior['common_complete_pools']!=n or prior['common_query_ids']!=[x['query_id'] for x in rows]:raise ValueError('Unexpected source or completion: '+name)
  scores=[]
  for i,row in enumerate(rows):
   if len(row['items'])!=32:raise ValueError('Unexpected pool size')
   if any(v['grade'] not in ('Exact','Partial','Irrelevant') or v['gold']!=('Y' if v['grade']=='Exact' else 'N') for v in row['items']):raise ValueError('Gold mapping')
   chunks=raw[i]['raw'] if name=='cal160' else raw[4*i:4*i+4]
   if len(chunks)!=4 or (name=='cal160' and raw[i]['query_id']!=row['query_id']):raise ValueError('Chunk count or calibration query')
   pred=[]
   for j,c in enumerate(chunks):
    if (c['start'],c['size'],c['presented_ids'])!=(8*j,8,list(range(1,9))) or (name!='cal160' and c['query_id']!=row['query_id']) or not re.fullmatch(pat,c['final']):raise ValueError('Keyed response mismatch')
    val=json.loads(c['final']);pred.extend(val[str(k)] for k in range(1,9))
   if name=='cal160' and raw[i]['pred']!=pred:raise ValueError('Parsed vector mismatch')
   gold=[v['gold'] for v in row['items']];p=pred.count('Y');g=gold.count('Y');e=sum(x!=y for x,y in zip(pred,gold));res=abs(p-g)
   if res>e:raise ValueError('Ratio residual exceeds candidate mistakes')
   scores.append({'p':p,'g':g,'error_count':e,'ratio_residual_count':res})
  mae=sum(x['ratio_residual_count'] for x in scores)/(32*n)
  if abs(mae-prior['quality_common'][qual]['ratio_mae'])>1e-12:raise ValueError('Saved keyed MAE mismatch: '+name)
  allsets[name]={'rows':rows,'scores':scores,'mae':mae}
  for path in (source,events,summary):provenance[path]=sha(r/path)
 for x,y in [('cal160','test96'),('cal160','test52'),('test96','test52')]:
  ax=allsets[x]['rows'];by=allsets[y]['rows'];aq={z['query'].strip().casefold() for z in ax};bq={z['query'].strip().casefold() for z in by};ai={str(it['product_id']) for z in ax for it in z['items']};bi={str(it['product_id']) for z in by for it in z['items']}
  if aq&bq or ai&bi:raise ValueError('Query/product overlap: '+x+' '+y)
 cal=allsets['cal160']['scores'];rank=(9*(len(cal)+1)+9)//10
 if rank>len(cal):ke=kr=32
 else:ke=sorted(x['error_count'] for x in cal)[rank-1];kr=sorted(x['ratio_residual_count'] for x in cal)[rank-1]
 if kr>ke:raise ValueError('Quantile ordering violated')
 output={}
 for name in ('test96','test52'):
  s=allsets[name]['scores'];n=len(s);record={'pools':n,'keyed_ratio_mae':allsets[name]['mae']}
  for label,t in [('candidate_error',ke),('ratio_residual',kr)]:
   widths=[];covered=0;score_covered=0
   for x in s:
    lo=max(0,x['p']-t);hi=min(32,x['p']+t);widths.append((hi-lo)/32);covered+=lo<=x['g']<=hi
    score_covered+=x['error_count']<=t if label=='candidate_error' else x['ratio_residual_count']<=t
   record[label]={'threshold':t,'interval_covered':covered,'score_at_most_threshold':score_covered,'mean_width':sum(widths)/n,'full_width_pools':sum(w==1 for w in widths)}
  for x in s:
   if not (max(0,x['p']-ke)<=max(0,x['p']-kr)<=min(32,x['p']+kr)<=min(32,x['p']+ke)):raise ValueError('Interval containment failed')
  output[name]=record
 result={'scope':'post-hoc diagnostic on previously exposed WANDS pools; no new model inference; exchangeability and future coverage not established',
         'nominal_miscoverage':0.1,'calibration':{'pools':len(cal),'order_statistic_rank':rank,'candidate_error_threshold':ke,'ratio_residual_threshold':kr,'keyed_ratio_mae':allsets['cal160']['mae']},'test':output,'input_sha256':provenance}
 with a.out.open('x',encoding='utf8') as f:json.dump(result,f,indent=2);f.write('\n')
 print(json.dumps({'calibration':result['calibration'],'test':output},indent=2))
if __name__=='__main__':main()
