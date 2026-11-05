#!/usr/bin/env python3
"""Check that the manuscript summary exactly matches the recorded weighted draws."""
import argparse,copy,json
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--result',type=Path,required=True);p.add_argument('--summary',type=Path,required=True);a=p.parse_args()
r=json.loads(a.result.read_text());s=json.loads(a.summary.read_text());expected=copy.deepcopy(r)
for d in expected['datasets'].values():d.pop('reweighted_differences')
if expected!=s:raise ValueError('Summary differs from raw result')
print('The compact manuscript summary matches both complete weighted analyses.')
