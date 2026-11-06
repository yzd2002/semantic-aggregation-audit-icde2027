#!/usr/bin/env python3
"""Explicit root-GO-gated launch; no model imports until authorization is checked."""
import hashlib
import json
import os
from pathlib import Path
import sys
import argparse

root=Path(__file__).resolve().parent
parser=argparse.ArgumentParser();parser.add_argument('--model',choices=('qwen','smol','glm'),required=True)
parser.add_argument('--device',type=int,required=True);args=parser.parse_args()
approval=root/f'GPU_GO_{args.model}.json'
freeze=root/'PREINFERENCE_FROZEN_SHA256.json'
if not approval.exists() or not freeze.exists():
    raise SystemExit('CPU preparation only. Explicit root GPU_GO.json tied to the final freeze is required.')
authorization=json.loads(approval.read_text())
if authorization.get('authorized_by')!='root' or authorization.get('device')!=args.device or authorization.get('model')!=args.model or authorization.get('freeze_sha256')!=hashlib.sha256(freeze.read_bytes()).hexdigest():
    raise SystemExit('GPU approval does not match the reviewed frozen preparation.')
env=os.environ.copy()
cache=root/'runtime_cache'/args.model
paths={'TMPDIR':'tmp','XDG_CACHE_HOME':'xdg','HF_HOME':'hf','HF_MODULES_CACHE':'hf_modules','TORCH_HOME':'torch',
       'VLLM_CACHE_ROOT':'vllm','VLLM_CONFIG_ROOT':'vllm_config','NUMBA_CACHE_DIR':'numba',
       'TORCHINDUCTOR_CACHE_DIR':'inductor','TRITON_CACHE_DIR':'triton','CUDA_CACHE_PATH':'cuda',
       'TORCH_EXTENSIONS_DIR':'extensions','FLASHINFER_WORKSPACE_BASE':'flashinfer-base'}
for key,relative in paths.items():
    path=cache/relative;path.mkdir(parents=True,exist_ok=True);env[key]=str(path)
env.update(CUDA_VISIBLE_DEVICES=str(args.device),PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
    TOKENIZERS_PARALLELISM='false',PYTHONPATH=str(root/'runtime_override'),INTENTSAT_SINGLE_RANK_NCCL_COMPAT='1',
    VLLM_USE_V1='1',VLLM_NO_USAGE_STATS='1',DO_NOT_TRACK='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
os.execve(sys.executable,[sys.executable,'-B',str(root/'run_gpu.py'),'--model',args.model,'--device',str(args.device)],env)
