#!/usr/bin/env python3
"""Post-hoc audit of 7B specification formatting and observed fallback."""
import argparse,collections,json,re
from pathlib import Path

def no_duplicates(pairs):
 d={}
 for k,v in pairs:
  if k in d:raise ValueError('duplicate key')
  d[k]=v
 return d

def valid(text):
 try:
  x=json.loads(text,object_pairs_hook=no_duplicates)
  return (isinstance(x,dict) and set(x)=={'product_type','required','preferred'}
          and isinstance(x['product_type'],str) and bool(x['product_type'].strip())
          and all(isinstance(x[k],list) and all(isinstance(y,str) for y in x[k]) for k in ['required','preferred']))
 except (TypeError,ValueError):return False

def diagnose(records):
 result={}
 for key in ('first','final'):
  count=collections.Counter()
  for row in records:
   text=row[key].strip()
   if valid(text):count['strict_valid']+=1;continue
   match=re.fullmatch(r'```(?:json)?\s*\n(.*?)\n```',text,flags=re.I|re.S)
   if match:
    count['fenced']+=1
    if valid(match.group(1)):count['valid_after_fence_removal']+=1
   else:count['other_invalid']+=1
  result[key]=dict(count)
 return result

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--esci-run',type=Path,required=True);ap.add_argument('--wands-run',type=Path,required=True);ap.add_argument('--esci-analysis',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
 if a.out.exists():raise FileExistsError(a.out)
 rows={}
 for label,p in [('esci',a.esci_run/'specifications_responses.jsonl'),('wands',a.wands_run/'specifications.jsonl')]:
  with p.open(encoding='utf8') as f:rows[label]=[json.loads(x) for x in f]
 report={k:{'pools':len(r),'formats':diagnose(r)} for k,r in rows.items()}
 esci=json.loads((a.esci_analysis/'summary.json').read_text());wands=json.loads((a.wands_run/'summary.json').read_text())
 fallback={'esci':esci['specification_fallback_pools'],'wands':wands['specifications']['fallback_pools']}
 for domain in rows:
  invalid=report[domain]['pools']-report[domain]['formats']['final'].get('strict_valid',0)
  if invalid!=fallback[domain]:raise RuntimeError('Fallback count differs from strict invalid specifications')
  report[domain]['observed_fallback_pools']=fallback[domain]
 report['limitations']='Post-hoc format diagnosis only; stripping fences was not run as a new plan and does not validate semantic specifications.'
 a.out.write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
 print(json.dumps(report,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
