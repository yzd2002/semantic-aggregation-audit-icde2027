#!/usr/bin/env bash
set -euo pipefail
work=/dev/shm/intentsat-7b-followup-20261031
artifact=/home/gpu/IntentSat2S_ICDE2027_package/02_reproducibility
python=/dev/shm/intentsat-offline-gpu-D9xHRe/venv/bin/python
model="$work/model/Qwen2.5-7B-Instruct"
mkdir -p "$work/runtime/tmp" "$work/runtime/cache/hf" "$work/runtime/cache/xdg" "$work/runtime/cache/triton" "$work/runtime/cache/torchinductor" "$work/runtime/cache/cuda"
export TMPDIR="$work/runtime/tmp" XDG_CACHE_HOME="$work/runtime/cache/xdg" HF_HOME="$work/runtime/cache/hf"
export TRITON_CACHE_DIR="$work/runtime/cache/triton" TORCHINDUCTOR_CACHE_DIR="$work/runtime/cache/torchinductor" CUDA_CACHE_PATH="$work/runtime/cache/cuda"
export PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$work:$artifact" VLLM_USE_V1=1 VLLM_NO_USAGE_STATS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
ensure_gpu_idle() {
  if [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; then
    printf 'GPU is in use; leaving other work untouched.\n' >&2
    return 1
  fi
}
printf 'Checking frozen inputs, synthetic tests and model hashes.\n'
"$python" "$work/verify_pre_7b.py" --artifact "$artifact" --posthoc "$work" > "$work/preflight.log"
"$python" "$work/test_crossmodel_7b.py" >> "$work/preflight.log"
"$python" "$work/test_guarded_batches.py" >> "$work/preflight.log"
(cd "$model" && sha256sum -c "$work/MODEL_LOCAL_SHA256.txt") >> "$work/preflight.log"
ensure_gpu_idle
printf 'Running 7B on the exposed ESCI pools.\n'
"$python" "$work/run_esci_7b_posthoc.py" --model "$model" --selection "$artifact/esci_selective_holdout_selection" --out "$work/run_esci_7b" > "$work/esci_run.log" 2>&1
"$python" "$artifact/analyze_esci_selective_holdout.py" --selection "$artifact/esci_selective_holdout_selection" --run "$work/run_esci_7b" --out "$work/analysis_esci_7b" > "$work/esci_analysis.log" 2>&1
"$python" "$artifact/verify_esci_selective_holdout.py" --selection "$artifact/esci_selective_holdout_selection" --run "$work/run_esci_7b" --analysis "$work/analysis_esci_7b" --model "$model" > "$work/esci_verify.log" 2>&1
"$python" "$work/crossmodel_7b.py" --selection "$artifact/esci_selective_holdout_selection" --older-run "$artifact/run_esci_selective_holdout_qwen" --newer-run "$work/run_esci_7b" --out "$work/crossmodel_7b_results" > "$work/esci_crossmodel.log" 2>&1
ensure_gpu_idle
printf 'Running 7B on the exposed WANDS pools.\n'
"$python" "$work/run_wands_7b_posthoc.py" --model "$model" --selection "$artifact/wands_selective_execution_selection" --out "$work/run_wands_7b" > "$work/wands_run.log" 2>&1
"$python" "$work/analyze_wands_7b_posthoc.py" --selection "$artifact/wands_selective_execution_selection" --run "$work/run_wands_7b" > "$work/wands_analysis.log" 2>&1
"$python" "$work/verify_wands_7b_posthoc.py" --selection "$artifact/wands_selective_execution_selection" --run "$work/run_wands_7b" --model "$model" > "$work/wands_verify.log" 2>&1
"$python" "$work/analyze_wands_7b_two_way.py" --selection "$artifact/wands_selective_execution_selection" --run "$work/run_wands_7b" > "$work/wands_crossed.log" 2>&1
"$python" "$work/verify_wands_7b_two_way.py" --selection "$artifact/wands_selective_execution_selection" --run "$work/run_wands_7b" > "$work/wands_crossed_verify.log" 2>&1
ensure_gpu_idle
printf 'Running 7B on the exposed TREC-COVID topics.\n'
"$python" "$work/run_trec_7b_posthoc.py" --model "$model" --selection "$artifact/trec_covid_transfer_selection" --out "$work/run_trec_7b" > "$work/trec_run.log" 2>&1
"$python" "$artifact/analyze_trec_covid_transfer.py" --selection "$artifact/trec_covid_transfer_selection" --run "$work/run_trec_7b" --out "$work/trec_7b_analysis.json" > "$work/trec_analysis.log" 2>&1
"$python" "$artifact/verify_trec_covid_transfer.py" --selection "$artifact/trec_covid_transfer_selection" --run "$work/run_trec_7b" --analysis "$work/trec_7b_analysis.json" --model "$model" > "$work/trec_verify.log" 2>&1
(cd "$model" && sha256sum -c "$work/MODEL_LOCAL_SHA256.txt") > "$work/post_model_hash.log"
"$python" "$work/verify_pre_7b.py" --artifact "$artifact" --posthoc "$work" > "$work/post_source_hash.log"
printf 'All three post-hoc 7B runs and their independent verifications completed.\n'
