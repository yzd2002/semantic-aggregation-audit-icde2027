#!/usr/bin/env python3
"""Rebuild all selected inputs from the verified source files."""
import argparse,json
from pathlib import Path
from run_experiment import read_rows
from run_validation import sample_rows
from run_full_pools import sample_official
ap=argparse.ArgumentParser()
ap.add_argument('--derived',default='/data/esci/test.jsonl.zst')
ap.add_argument('--examples',default='/home/gpu/intent2s_esci_official_2026/shopping_queries_dataset_examples.parquet')
ap.add_argument('--products',default='/home/gpu/intent2s_esci_official_2026/shopping_queries_dataset_products.parquet')
a=ap.parse_args();root=Path(__file__).resolve().parent
sample=lambda p:[json.loads(x) for x in (root/p/'inputs.jsonl').open()]
pilot=sample('run_test_160');check,_=read_rows(a.derived)
assert check==pilot,'Pilot selection mismatch';print('Pilot selection reproduced.',flush=True)
validation=sample('run_validation_500');check,_=sample_rows(a.derived,{x['query'] for x in pilot})
assert check==validation,'Derived validation selection mismatch';print('Derived validation selection reproduced.',flush=True)
full=sample('run_full_pools_250');check,_=sample_official(a.examples,a.products)
assert check==full,'Original full-pool selection mismatch';print('Original full-pool selection reproduced.',flush=True)
if (root/'run_full_pools_smollm_250'/'inputs.jsonl').exists():
 assert sample('run_full_pools_smollm_250')==full,'Sensitivity-model input mismatch'
 print('Sensitivity-model inputs match.',flush=True)
