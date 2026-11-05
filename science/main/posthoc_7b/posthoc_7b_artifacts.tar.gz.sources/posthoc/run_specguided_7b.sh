#!/usr/bin/env bash
set -euo pipefail
work=/dev/shm/intentsat-7b-followup-20261031
artifact=/home/gpu/IntentSat2S_ICDE2027_package/02_reproducibility
python=/dev/shm/intentsat-offline-gpu-D9xHRe/venv/bin/python
model="$work/model/Qwen2.5-7B-Instruct"
export TMPDIR="$work/runtime/tmp" XDG_CACHE_HOME="$work/runtime/cache/xdg" HF_HOME="$work/runtime/cache/hf"
export TRITON_CACHE_DIR="$work/runtime/cache/triton" TORCHINDUCTOR_CACHE_DIR="$work/runtime/cache/torchinductor" CUDA_CACHE_PATH="$work/runtime/cache/cuda"
export PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$work:$artifact" VLLM_USE_V1=1 VLLM_NO_USAGE_STATS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
ensure_gpu_idle() {
  if [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; then
    printf 'GPU is in use; leaving other work untouched.\n' >&2; return 1
  fi
}
printf 'Checking schema-guided frozen inputs, model and synthetic controls.\n'
"$python" "$work/verify_pre_7b.py" --artifact "$artifact" --posthoc "$work" > "$work/specguided_preflight.log"
"$python" "$work/verify_specguided_frozen.py" --artifact "$artifact" --work "$work" >> "$work/specguided_preflight.log"
"$python" "$work/test_specguided_analysis.py" >> "$work/specguided_preflight.log"
"$python" "$work/test_spec_schema.py" >> "$work/specguided_preflight.log"
(cd "$model" && sha256sum -c "$work/MODEL_LOCAL_SHA256.txt") >> "$work/specguided_preflight.log"
ensure_gpu_idle
printf 'Running schema-guided 7B specifications on exposed ESCI pools.\n'
"$python" "$work/run_esci_7b_specguided.py" --model "$model" --selection "$artifact/esci_selective_holdout_selection" --out "$work/run_esci_7b_specguided" > "$work/esci_specguided_run.log" 2>&1
"$python" "$work/verify_specguided_control.py" --run "$work/run_esci_7b_specguided" > "$work/esci_specguided_control.log" 2>&1
"$python" "$artifact/analyze_esci_selective_holdout.py" --selection "$artifact/esci_selective_holdout_selection" --run "$work/run_esci_7b_specguided" --out "$work/analysis_esci_7b_specguided" > "$work/esci_specguided_analysis.log" 2>&1
"$python" "$artifact/verify_esci_selective_holdout.py" --selection "$artifact/esci_selective_holdout_selection" --run "$work/run_esci_7b_specguided" --analysis "$work/analysis_esci_7b_specguided" --model "$model" > "$work/esci_specguided_verify.log" 2>&1
"$python" "$work/compare_spec_guided.py" --selection "$artifact/esci_selective_holdout_selection" --old-run "$work/run_esci_7b" --old-analysis "$work/analysis_esci_7b/summary.json" --new-run "$work/run_esci_7b_specguided" --new-analysis "$work/analysis_esci_7b_specguided/summary.json" --out "$work/esci_specguided_comparison.json" > "$work/esci_specguided_comparison.log" 2>&1
"$python" "$work/verify_specguided_comparison.py" --selection "$artifact/esci_selective_holdout_selection" --old-run "$work/run_esci_7b" --old-analysis "$work/analysis_esci_7b/summary.json" --new-run "$work/run_esci_7b_specguided" --new-analysis "$work/analysis_esci_7b_specguided/summary.json" --result "$work/esci_specguided_comparison.json" > "$work/esci_specguided_comparison_verify.log" 2>&1
ensure_gpu_idle
printf 'Running schema-guided 7B specifications on exposed WANDS pools.\n'
"$python" "$work/run_wands_7b_specguided.py" --model "$model" --selection "$artifact/wands_selective_execution_selection" --out "$work/run_wands_7b_specguided" > "$work/wands_specguided_run.log" 2>&1
"$python" "$work/verify_specguided_control.py" --run "$work/run_wands_7b_specguided" > "$work/wands_specguided_control.log" 2>&1
"$python" "$work/analyze_wands_7b_specguided.py" --selection "$artifact/wands_selective_execution_selection" --run "$work/run_wands_7b_specguided" > "$work/wands_specguided_analysis.log" 2>&1
"$python" "$work/verify_wands_7b_specguided.py" --selection "$artifact/wands_selective_execution_selection" --run "$work/run_wands_7b_specguided" --model "$model" > "$work/wands_specguided_verify.log" 2>&1
"$python" "$work/analyze_wands_7b_specguided_two_way.py" --selection "$artifact/wands_selective_execution_selection" --run "$work/run_wands_7b_specguided" > "$work/wands_specguided_crossed.log" 2>&1
"$python" "$work/verify_wands_7b_specguided_two_way.py" --selection "$artifact/wands_selective_execution_selection" --run "$work/run_wands_7b_specguided" > "$work/wands_specguided_crossed_verify.log" 2>&1
"$python" "$work/compare_spec_guided.py" --selection "$artifact/wands_selective_execution_selection" --old-run "$work/run_wands_7b" --old-analysis "$work/run_wands_7b/paired_analysis.json" --new-run "$work/run_wands_7b_specguided" --new-analysis "$work/run_wands_7b_specguided/paired_analysis.json" --out "$work/wands_specguided_comparison.json" > "$work/wands_specguided_comparison.log" 2>&1
"$python" "$work/verify_specguided_comparison.py" --selection "$artifact/wands_selective_execution_selection" --old-run "$work/run_wands_7b" --old-analysis "$work/run_wands_7b/paired_analysis.json" --new-run "$work/run_wands_7b_specguided" --new-analysis "$work/run_wands_7b_specguided/paired_analysis.json" --result "$work/wands_specguided_comparison.json" > "$work/wands_specguided_comparison_verify.log" 2>&1
(cd "$model" && sha256sum -c "$work/MODEL_LOCAL_SHA256.txt") > "$work/specguided_post_model_hash.log"
"$python" "$work/verify_specguided_frozen.py" --artifact "$artifact" --work "$work" > "$work/specguided_post_source_hash.log"
printf 'Both schema-guided post-hoc runs and independent verifications completed.\n'
