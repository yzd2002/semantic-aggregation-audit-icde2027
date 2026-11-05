#!/usr/bin/env python3
"""Verify the saved lexical-tiebreaker protocol and pre-calculation code hashes."""
import argparse,hashlib,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--supplement',required=True,type=Path);a=p.parse_args()
manifest=json.loads((a.supplement/'LEXICAL_PRECALC_SHA256.json').read_text())
for name,expected in manifest.items():
    found=hashlib.sha256((a.supplement/name).read_bytes()).hexdigest()
    if found!=expected:raise ValueError('Lexical analysis source mismatch: '+name)
print('Verified',len(manifest),'lexical protocol, code and prior-artifact hashes; outcomes and inputs were already exposed.')
