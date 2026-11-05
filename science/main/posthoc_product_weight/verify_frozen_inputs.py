#!/usr/bin/env python3
"""Verify the inputs and scripts frozen before the product-weight sensitivity audit."""
import argparse,hashlib,json
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--artifact',type=Path,required=True);p.add_argument('--work',type=Path,required=True);p.add_argument('--audit',type=Path,required=True);a=p.parse_args()
manifest=json.loads((a.audit/'FROZEN_INPUTS_SHA256.json').read_text());roots={'artifact':a.artifact,'posthoc':a.work,'audit':a.audit}
for key,want in manifest.items():
 prefix,relative=key.split('/',1);path=roots[prefix]/relative;h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 if h.hexdigest()!=want:raise ValueError('Frozen input mismatch: '+key)
print('Verified',len(manifest),'frozen pre-analysis inputs and scripts.')
