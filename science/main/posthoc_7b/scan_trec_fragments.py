#!/usr/bin/env python3
"""Count exact 60-character article fragments in a directory of UTF-8 traces."""
import argparse,json,re
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--selection',type=Path,required=True,help='reconstructed TREC inputs.jsonl')
p.add_argument('--run',type=Path,required=True,help='directory of saved TREC trace files')
p.add_argument('--out',type=Path,required=True,help='new JSON output')
a=p.parse_args()
if a.out.exists():raise FileExistsError(a.out)
norm=lambda s:re.sub(r'\s+',' ',s).strip().casefold()
fragments=set()
with a.selection.open(encoding='utf8') as f:
 for line in f:
  for item in json.loads(line)['items']:
   for field in ('title','abstract'):
    t=norm(item[field]);offset=max(0,len(t)//2-30)
    if len(t)>=60:fragments.update((t[:60],t[offset:offset+60]))
results={}
for path in sorted(a.run.iterdir()):
 if path.is_file():
  content=norm(path.read_text(encoding='utf8'))
  results[path.name]=sum(1 for fragment in fragments if fragment in content)
result={'fragment_length_characters':60,'distinct_fragments':len(fragments),'matches_per_trace_file':results,
        'interpretation':'Exact-fragment diagnostic only; not a copyright or licensing determination. Other third-party content remains.'}
a.out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
print('Tested fragments:',len(fragments),'trace files:',len(results),'total exact matches:',sum(results.values()))
