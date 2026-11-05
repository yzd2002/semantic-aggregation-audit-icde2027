#!/usr/bin/env python3
"""Check frozen source hashes for the post-hoc midpoint analysis."""
import argparse, hashlib, json
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('--artifact',required=True,type=Path);p.add_argument('--supplement',required=True,type=Path)
a=p.parse_args();counts=0
pre=json.loads((a.supplement/'PRECOMPUTATION_INPUT_SHA256.json').read_text())
for name,expected in pre.items():
    # The preserved initial script was renamed after its first run failed.
    path=(a.supplement/'midpoint_reference_initial.py' if name=='midpoint_reference.py' else
          a.supplement/name if name=='PROTOCOL.md' else a.artifact/name)
    found=hashlib.sha256(path.read_bytes()).hexdigest()
    if found!=expected:raise ValueError('Frozen input mismatch: '+name)
    counts+=1
for name,expected in json.loads((a.supplement/'POSTCOMPUTATION_DEPENDENCY_SHA256.json').read_text()).items():
    found=hashlib.sha256((a.artifact/name).read_bytes()).hexdigest()
    if found!=expected:raise ValueError('Additional dependency mismatch: '+name)
    counts+=1
print('Verified',counts,'pre- and post-computation source hashes; the latter were not frozen before results.')
