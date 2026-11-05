#!/usr/bin/env python3
"""Check that the Figure 3 renderer rejects missing or inconsistent data."""
import argparse,copy,json,os,subprocess,sys,tempfile
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);a=p.parse_args();data=json.loads(a.data.read_text())
plot=Path(__file__).with_name('plot_readable_model_sensitivity.py')
variants=[]
x=copy.deepcopy(data);del x['values']['ESCI']['Qwen2.5-7B'];variants.append(('missing',x))
x=copy.deepcopy(data);x['values']['WANDS']['Qwen3-4B']['pools']+=1;variants.append(('count',x))
x=copy.deepcopy(data);x['values']['TREC-COVID']['Qwen2.5-7B']['difference']+=.1;variants.append(('contrast',x))
with tempfile.TemporaryDirectory(prefix='figure3-check-') as tmp:
 root=Path(tmp)
 for name,variant in variants:
  source=root/(name+'.json');source.write_text(json.dumps(variant));out=root/(name+'-output')
  r=subprocess.run([sys.executable,str(plot),'--data',str(source),'--out',str(out)],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
  if r.returncode==0 or out.exists():raise AssertionError('Renderer accepted invalid '+name+' input')
print('Missing model, differing pool counts and inconsistent contrast all rejected.')
