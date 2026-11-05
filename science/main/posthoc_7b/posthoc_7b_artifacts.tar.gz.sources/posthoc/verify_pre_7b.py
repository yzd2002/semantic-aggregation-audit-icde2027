#!/usr/bin/env python3
"""Verify pre-output sources and the explicitly superseded cross-model amendment."""
import argparse,hashlib,json
from pathlib import Path
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--artifact',type=Path,default=Path(__file__).resolve().parent)
parser.add_argument('--posthoc',type=Path,default=Path(__file__).resolve().parent)
args=parser.parse_args()
b=args.artifact.resolve()
w=args.posthoc.resolve()
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for x in iter(lambda:f.read(1024*1024),b''):h.update(x)
 return h.hexdigest()
def verify_manifest(name,artifact_only=False):
 obj=json.loads((w/name).read_text(encoding='utf8'));checked=0
 for key,want in obj.items():
  if artifact_only and key.startswith('posthoc/'):continue
  if key.startswith('artifact/'):p=b/key.removeprefix('artifact/')
  elif key.startswith('posthoc/'):p=w/key.removeprefix('posthoc/')
  else:p=w/key
  if not p.is_file() or sha(p)!=want:raise RuntimeError(f'Frozen file mismatch in {name}: {key}')
  checked+=1
 print(name,'verified',checked,'entries')
for name in ('PRE_7B_SOURCE_SHA256.json','PRE_7B_MULTIDOMAIN_SHA256.json','PRE_7B_CROSSED_SHA256.json',
             'PRE_7B_GUARDED_BATCH_SHA256.json','PRE_7B_CROSSMODEL_AMENDMENT_SHA256.json'):
 verify_manifest(name)
# The initial cross-model plan's three posthoc files were amended before any
# new model outputs; the amendment manifest preserves and verifies that record.
verify_manifest('PRE_7B_CROSSMODEL_SHA256.json',artifact_only=True)
print('All pre-output source, input, plan and supersession checks passed.')
