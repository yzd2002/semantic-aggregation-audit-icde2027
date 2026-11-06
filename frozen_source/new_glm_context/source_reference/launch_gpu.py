#!/usr/bin/env python3
"""Launch the frozen one-model runner with every writable cache under this bundle."""
import os,sys
from pathlib import Path
root=Path(__file__).resolve().parent
if len(sys.argv)!=2 or sys.argv[1] not in ('qwen','smol'):raise SystemExit('Usage: launch_gpu.py qwen|smol')
env=os.environ.copy()
paths={'TMPDIR':'tmp','XDG_CACHE_HOME':'cache','HF_HOME':'cache/hf','TORCH_HOME':'cache/torch',
 'VLLM_CACHE_ROOT':'cache/vllm','VLLM_CONFIG_ROOT':'cache/vllm_config','NUMBA_CACHE_DIR':'cache/numba',
 'TORCHINDUCTOR_CACHE_DIR':'cache/inductor','TRITON_CACHE_DIR':'cache/triton','CUDA_CACHE_PATH':'cache/cuda',
 'TORCH_EXTENSIONS_DIR':'cache/extensions'}
for key,rel in paths.items():
 path=root/rel;path.mkdir(parents=True,exist_ok=True);env[key]=str(path)
env.update(PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',
 PYTHONPATH=str(root/'runtime_override'),INTENTSAT_SINGLE_RANK_NCCL_COMPAT='1',
 VLLM_USE_V1='1',VLLM_NO_USAGE_STATS='1',DO_NOT_TRACK='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
os.execve(sys.executable,[sys.executable,'-B',str(root/'run_gpu.py'),'--model',sys.argv[1]],env)
