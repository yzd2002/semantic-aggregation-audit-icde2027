#!/usr/bin/env python3
"""Independently recheck threshold selection and fixed-output WANDS ratios."""
import argparse,bisect,json,math
from pathlib import Path

def lines(p):
 with p.open(encoding='utf8') as f:return [json.loads(x) for x in f if x.strip()]
def compare(x,y,label):
 if not math.isclose(x,y,rel_tol=0,abs_tol=2e-12):raise ValueError(label+': '+str(x)+' != '+str(y))
def main():
 p=argparse.ArgumentParser();p.add_argument('--artifact',type=Path,required=True);p.add_argument('--result',type=Path,required=True);a=p.parse_args();r=a.artifact
 result=json.loads(a.result.read_text());train=lines(r/'run_wands_160/inputs.jsonl');oof=lines(r/'run_wands_calibrated_lexical/out_of_fold.jsonl')
 if len(train)!=160 or len(oof)!=160:raise ValueError('Training count')
 sorted_scores=[];truth=[];values=set()
 for row,pred in zip(train,oof):
  if row['query_id']!=pred['query_id'] or len(row['items'])!=32 or len(pred['positive_probability'])!=32:raise ValueError('Training order')
  s=sorted(pred['positive_probability']);sorted_scores.append(s);values.update(s)
  truth.append(sum(it['gold']=='Y' for it in row['items']))
 candidates=[math.nextafter(max(values),math.inf)]+sorted(values,reverse=True)
 losses=[]
 for t in candidates:
  loss=sum(abs((32-bisect.bisect_left(scores,t))-g) for scores,g in zip(sorted_scores,truth))
  losses.append(loss)
 index=min(range(len(candidates)),key=lambda i:losses[i]);choice=candidates[index]
 compare(choice,result['threshold_selection']['threshold'],'threshold')
 compare(losses[index]/(160*32),result['threshold_selection']['oof_selected_ratio_mae'],'training loss')
 median=sorted(truth)[79]/32;compare(median,result['threshold_selection']['constant_lower_median_train_ratio'],'median')
 pred=lines(r/'run_wands_supervised_reference/predictions.jsonl')
 names=[('holdout160','wands_holdout_selection'),('fallback96','wands_fallback_validation_selection'),('selective52','wands_selective_execution_selection')]
 for name,selection in names:
  rows=lines(r/selection/'inputs.jsonl');saved=[x for x in pred if x['sample']==name]
  if len(rows)!=len(saved):raise ValueError(name+' count')
  errors={'qwen':[],'fixed_0_5':[],'ratio_threshold':[],'lexical':[],'score_mean':[],'training_median':[]}
  for row,p in zip(rows,saved):
   if row['query_id']!=p['query_id']:raise ValueError('Alignment')
   g=sum(it['gold']=='Y' for it in row['items'])/32
   scores=p['positive_probability']
   estimates={'qwen':sum(x=='Y' for x in p['qwen'])/32,'fixed_0_5':sum(x>=.5 for x in scores)/32,
              'ratio_threshold':sum(x>=choice for x in scores)/32,'lexical':sum(x=='Y' for x in p['lexical'])/32,
              'score_mean':math.fsum(scores)/32,'training_median':median}
   for key,v in estimates.items():errors[key].append(abs(v-g))
  for key,e in errors.items():
   expected=(result['samples'][name]['binary_metrics'][key]['ratio_mae'] if key in ['qwen','fixed_0_5','lexical']
             else result['samples'][name]['ratio_results'][key]['ratio_mae'])
   compare(math.fsum(e)/len(e),expected,name+' '+key)
 print('Independent sorted-score sweep and all six ratio MAEs verified for 160, 96 and 52 pools.')
if __name__=='__main__':main()
