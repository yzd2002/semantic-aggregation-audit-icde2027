#!/usr/bin/env python3
"""Descriptive, post-hoc completion breakdown by pool size."""
import argparse,json
from pathlib import Path
from run_experiment import parse_batch,parse_spec
from run_validation import parse_compact
ap=argparse.ArgumentParser();ap.add_argument('--run',default='run_full_pools_250');a=ap.parse_args()
p=Path(__file__).resolve().parent/a.run
rows=[json.loads(s) for s in (p/'inputs.jsonl').open()];outs=[json.loads(s) for s in (p/'outputs.jsonl').open()]
methods=['pointwise','batch_json','two_stage_json','two_stage_compact_fallback']
result={'description':'Post-hoc descriptive breakdown by pool size, not a causal effect.','bins':{},'failures':{}}
for label,fn in [('8-16',lambda n:n<=16),('17-24',lambda n:17<=n<=24),('25+',lambda n:n>=25)]:
 ix=[i for i,r in enumerate(rows) if fn(len(r['items']))]
 result['bins'][label]={'pools':len(ix),'complete':{m:sum(outs[i]['plans'][m]['pred'] is not None for i in ix) for m in methods}}
for m in methods:
 spec_fail=label_fail=0
 for r,o in zip(rows,outs):
  z=o['plans'][m]
  if z['pred'] is not None:continue
  if m!='pointwise' and z.get('spec') is not None and parse_spec(z['spec']['final']) is None and not z.get('fallback',False):spec_fail+=1
  else:label_fail+=1
 result['failures'][m]={'specification_failure':spec_fail,'label_or_chunk_failure':label_fail}
(p/'failure_analysis.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
