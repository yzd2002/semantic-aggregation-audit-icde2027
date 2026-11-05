#!/usr/bin/env python3
"""Record only non-secret launch settings from the already-running local process."""
import argparse,json,time
from pathlib import Path
from common import ROOT,write_json

NAMES=('PYTHONDONTWRITEBYTECODE','OPENBLAS_NUM_THREADS','INTENTSAT_SINGLE_RANK_NCCL_COMPAT',
       'PYTHONPATH','VLLM_USE_V1','HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','TMPDIR','XDG_CACHE_HOME',
       'VLLM_CACHE_ROOT','NUMBA_CACHE_DIR','TORCHINDUCTOR_CACHE_DIR','TRITON_CACHE_DIR','CUDA_CACHE_PATH',
       'HF_HOME','TORCH_EXTENSIONS_DIR','VLLM_NO_USAGE_STATS','VLLM_CONFIG_ROOT','CUDA_VISIBLE_DEVICES')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--model',choices=['qwen','smol'],required=True)
    name=parser.parse_args().model;folder=ROOT/f'run_{name}';start=json.loads((folder/'START.json').read_text())
    entries=Path(f"/proc/{start['pid']}/environ").read_bytes().split(b'\0')
    selected={}
    for entry in entries:
        key,separator,value=entry.partition(b'=')
        if separator and key.decode(errors='replace') in NAMES:
            selected[key.decode()]=value.decode(errors='replace')
    write_json(folder/'OBSERVED_LAUNCH_ENVIRONMENT.json',{'captured_unix':time.time(),'pid':start['pid'],
        'scope':'Post-start observation of an explicit non-secret whitelist; not an additional pre-inference freeze.',
        'settings':selected,'unset':[name for name in NAMES if name not in selected]})
    print('Recorded whitelisted launch settings:',name)
