#!/usr/bin/env python3
"""Verify the complete post-hoc spec-guidance frozen input and code manifest."""
import argparse,hashlib,json
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--artifact',type=Path,default=Path(__file__).resolve().parent);p.add_argument('--work',type=Path,default=Path(__file__).resolve().parent);a=p.parse_args();b=a.artifact.resolve();w=a.work.resolve()
obj=json.loads((w/'SPEC_GUIDED_FROZEN_SHA256.json').read_text())
for key,want in obj.items():
 path=(b/key.removeprefix('artifact/')) if key.startswith('artifact/') else w/key.removeprefix('posthoc/')
 h=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 if h.hexdigest()!=want:raise RuntimeError('Frozen spec-guided input changed: '+key)
print('Frozen spec-guided inputs verified:',len(obj))
