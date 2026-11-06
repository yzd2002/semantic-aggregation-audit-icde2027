#!/usr/bin/env python3
"""Never launches without independent review and a fresh root-authored GPU release."""
import argparse,os,sys
from pathlib import Path
from common import ROOT,MODELS
from gpu_gate import verify
from runtime_observer import launch_environment

def main():
    p=argparse.ArgumentParser();p.add_argument('--model',choices=MODELS,required=True);p.add_argument('--gpu',required=True);a=p.parse_args()
    verify(a.model,a.gpu)
    env=os.environ.copy()
    launch_env=launch_environment(ROOT,a.model)
    for key,value in launch_env.items():
        if key not in ('VLLM_ALLOW_INSECURE_SERIALIZATION',):Path(value).mkdir(parents=True,exist_ok=True)
        env[key]=value
    env.update(CUDA_VISIBLE_DEVICES=a.gpu,PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',
        PIP_DISABLE_PIP_VERSION_CHECK='1',PYTHONPATH=str(ROOT/'runtime_override'),INTENTSAT_SINGLE_RANK_NCCL_COMPAT='1',
        VLLM_USE_V1='1',VLLM_NO_USAGE_STATS='1',DO_NOT_TRACK='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    os.execve(sys.executable,[sys.executable,'-B',str(ROOT/'run_gpu.py'),'--model',a.model,'--gpu',a.gpu],env)

if __name__=='__main__':main()
