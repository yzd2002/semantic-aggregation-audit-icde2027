#!/usr/bin/env python3
"""Freeze all scientific inputs/code/history; omit runtime caches and model weights."""
import argparse,time
from common import *

def verify(bundle):
 bundle=Path(bundle);manifest=json.loads((bundle/'PREINFERENCE_FROZEN_SHA256.json').read_text())
 for name,want in manifest['files'].items():
  if sha(bundle/name)!=want:raise ValueError('Frozen file mismatch: '+name)
 return {'status':'PASS','files':len(manifest['files']),'freeze_sha256':sha(bundle/'PREINFERENCE_FROZEN_SHA256.json')}

def freeze(bundle):
 bundle=Path(bundle)
 if any(p.is_dir() for p in bundle.glob('run_*')):raise ValueError('No preinference freeze after a run directory exists')
 paths=[]
 for p in bundle.rglob('*'):
  if not p.is_file():continue
  rel=p.relative_to(bundle)
  if rel.parts[0] in ('cache','tmp','__pycache__') or '__pycache__' in rel.parts:continue
  if p.name=='PREINFERENCE_FROZEN_SHA256.json':continue
  paths.append(p)
 files={p.relative_to(bundle).as_posix():sha(p) for p in sorted(paths)}
 write_json(bundle/'PREINFERENCE_FROZEN_SHA256.json',{'schema_version':1,'created_unix':time.time(),'files':files,
  'file_count':len(files),'bytes':sum(p.stat().st_size for p in paths),'relative_root':'decision_audit',
  'gpu_authorization':'none; development requires root GO; heldout evaluation also requires reviewed 36-policy lock and TEST_RELEASE',
  'planned_scientific_calls':15616,'planned_requested_items':46080,'planned_diagnostic_calls':80})
 return verify(bundle)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,default=ROOT);p.add_argument('--check',action='store_true');a=p.parse_args()
 print(compact(verify(a.bundle) if a.check else freeze(a.bundle)))
