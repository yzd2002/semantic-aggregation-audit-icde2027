#!/usr/bin/env python3
"""Independently check the incompatible-instruction specification control."""
import argparse,json
from pathlib import Path

def nodup(pairs):
 d={}
 for k,v in pairs:
  if k in d:raise ValueError('duplicate key')
  d[k]=v
 return d

def valid(text):
 try:
  x=json.loads(text,object_pairs_hook=nodup)
  return (isinstance(x,dict) and set(x)=={'product_type','required','preferred'} and
          isinstance(x['product_type'],str) and bool(x['product_type'].strip()) and
          all(isinstance(x[k],list) and all(isinstance(y,str) for y in x[k]) for k in ('required','preferred')))
 except (ValueError,TypeError):return False

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);a=p.parse_args()
 x=json.loads((a.run/'guidance_sanity.json').read_text(encoding='utf8'))
 if valid(x['unconstrained']) or not valid(x['guided_spec']) or x['unconstrained_valid_spec'] or not x['guided_spec_valid']:
  raise ValueError('Specification negative control failed')
 print('Independent specification negative control passed; semantic validity is not established.')
if __name__=='__main__':main()
