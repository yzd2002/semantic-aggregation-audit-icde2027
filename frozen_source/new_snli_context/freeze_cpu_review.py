#!/usr/bin/env python3
"""Freeze a review candidate; explicitly not GPU authorization or a final runtime freeze."""
from common import *
import time

def main():
 assert json.loads((ROOT/'prepared/AUTHOR_CPU_VERIFICATION.json').read_text())['status']=='PASS'
 assert json.loads((ROOT/'prepared/PREINFERENCE_STATUS.json').read_text())['status']=='NO_GPU_GO'
 paths=[]
 for name in ('sources','selection','prepared','tokenizers','gold_perturbation_sources','cpu_checks'):
  paths.extend(p for p in (ROOT/name).rglob('*') if p.is_file() and '__pycache__' not in p.parts)
 paths.extend(p for p in ROOT.iterdir() if p.is_file() and p.name!='CPU_REVIEW_CANDIDATE_SHA256.json' and not p.name.endswith('.sha256'))
 files={str(p.relative_to(ROOT)):sha(p) for p in sorted(paths)}
 result={'created_unix':time.time(),'freeze_status':'CPU_REVIEW_CANDIDATE_NO_GPU_GO','new_model_judgments':0,
  'files':files,'remaining_before_inference':['independent review','GPU ownership/runtime compatibility review',
   'final inference runner and any adapter freeze','final pre-inference manifest including runner and review evidence','explicit root GPU GO'],
  'mutable_runtime_cache_excluded':'cpu_cache (tokenizer module cache); source tokenizer assets are frozen separately'}
 write_json(ROOT/'CPU_REVIEW_CANDIDATE_SHA256.json',result)
 (ROOT/'CPU_REVIEW_CANDIDATE_SHA256.json.sha256').write_text(sha(ROOT/'CPU_REVIEW_CANDIDATE_SHA256.json')+'  CPU_REVIEW_CANDIDATE_SHA256.json\n')
 print(compact({'status':result['freeze_status'],'files':len(files),'manifest_sha256':sha(ROOT/'CPU_REVIEW_CANDIDATE_SHA256.json')}))

if __name__=='__main__':main()
