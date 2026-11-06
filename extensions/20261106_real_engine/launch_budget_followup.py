"""Single selected idle GPU, separate caches, fixed final follow-up only."""
from experiment_common import *
import argparse
import os
import sys


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=MODELS, required=True)
    p.add_argument("--gpu", required=True)
    a = p.parse_args()
    tag = "budget_"+a.model
    base = ROOT / "runtime_cache" / tag
    env = os.environ.copy()
    for key, sub in {
        "TMPDIR": "tmp", "XDG_CACHE_HOME": "xdg", "XDG_CONFIG_HOME": "config",
        "HF_HOME": "hf", "HF_MODULES_CACHE": "hf_modules", "TORCH_HOME": "torch",
        "VLLM_CACHE_ROOT": "vllm", "VLLM_CONFIG_ROOT": "vllm_config", "NUMBA_CACHE_DIR": "numba",
        "TORCHINDUCTOR_CACHE_DIR": "inductor", "TRITON_CACHE_DIR": "triton", "CUDA_CACHE_PATH": "cuda",
        "TORCH_EXTENSIONS_DIR": "extensions", "FLASHINFER_WORKSPACE_BASE": "flashinfer_base",
    }.items():
        path = base / sub
        path.mkdir(parents=True, exist_ok=True)
        env[key] = str(path)
    rpc = Path("/home/gpu/i2u1106") / ("b"+a.model)
    rpc.mkdir(parents=True, exist_ok=True)
    env.update(CUDA_VISIBLE_DEVICES=a.gpu, PYTHONDONTWRITEBYTECODE="1", HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false",
        PYTHONPATH=str(ROOT / "runtime_override")+":"+str(ROOT), INTENTSAT_SINGLE_RANK_NCCL_COMPAT="1",
        INTENTSAT_CACHE_RESET_OBSERVER="1", INTENTSAT_CACHE_RESET_LOG=str(base / "actual_cache_resets.jsonl"),
        VLLM_USE_V1="1", VLLM_ALLOW_INSECURE_SERIALIZATION="1", VLLM_RPC_BASE_PATH=str(rpc),
        VLLM_NO_USAGE_STATS="1", DO_NOT_TRACK="1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    os.execve(sys.executable, [sys.executable, "-B", str(ROOT / "run_budget_followup.py"),
              "--model", a.model, "--gpu", a.gpu], env)


if __name__ == "__main__":
    main()
