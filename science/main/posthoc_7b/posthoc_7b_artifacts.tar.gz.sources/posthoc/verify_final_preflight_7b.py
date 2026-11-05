#!/usr/bin/env python3
"""Verify pre-inference orchestration, tokenizer checks and figure plan."""
import argparse,hashlib,json
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--artifact',type=Path,default=Path(__file__).resolve().parent)
p.add_argument('--posthoc',type=Path,default=Path(__file__).resolve().parent)
p.add_argument('--model',type=Path,help='optional external directory for the downloaded 7B model')
a=p.parse_args();b=a.artifact.resolve();w=a.posthoc.resolve()
for manifest in ['PRE_7B_DRIVER_SHA256.json','PRE_7B_CONTEXT_SHA256.json','PRE_7B_FIGURE_SHA256.json','PRE_7B_FIGURE_TEST_AMENDMENT_SHA256.json']:
 entries=json.loads((w/manifest).read_text(encoding='utf8'))
 for key,want in entries.items():
  model_prefix='posthoc/model/Qwen2.5-7B-Instruct/'
  if a.model is not None and key.startswith(model_prefix):path=a.model.resolve()/key.removeprefix(model_prefix)
  else:path=(b/key.removeprefix('artifact/')) if key.startswith('artifact/') else (w/key.removeprefix('posthoc/')) if key.startswith('posthoc/') else w/key
  h=hashlib.sha256()
  with path.open('rb') as f:
   for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
  if h.hexdigest()!=want:raise RuntimeError(f'{manifest}: {key} changed')
 print(manifest,'verified',len(entries),'entries')
print('Final pre-inference driver, context and figure checks passed.')
