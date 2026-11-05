#!/usr/bin/env bash
# Recheck selected archived results with CPU only. No model inference is run.
set -eu
cd "$(dirname "$0")"
TASK_PYTHON=${INTENTSAT_AUDIT_PYTHON:-python3}
TASK_WANDS=''
if [ "$#" -gt 0 ]; then
  if [ "$#" -ne 2 ] || [ "$1" != '--wands-dir' ]; then
    printf 'Usage: %s [--wands-dir /path/to/WANDS/dataset]\n' "$0" >&2
    exit 2
  fi
  TASK_WANDS=$2
fi
unset PYTHONPATH
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
run() { printf '\nChecking %s\n' "$1"; "$TASK_PYTHON" "$@"; }
for TASK_MANIFEST in LOCAL_SERVICE_BENCHMARK_FROZEN_SHA256.txt LOCAL_SERVICE_CONTINUATION_FROZEN_SHA256.txt WANDS_EMBEDDING_REFERENCE_FROZEN_SHA256.txt WANDS_EMBEDDING_REPLAY_FROZEN_SHA256.txt WANDS_CALIBRATED_LEXICAL_FROZEN_SHA256.txt; do
  sha256sum -c "$TASK_MANIFEST" > /dev/null
  printf 'Verified %s\n' "$TASK_MANIFEST"
done
run test_ratio_bounds.py
run test_stability_bounds.py
run test_local_service_benchmark.py
run test_prepare_submission_authors.py
run test_wands_calibrated_threshold.py
if [ -n "$TASK_WANDS" ]; then
  run verify_wands_holdout_selection.py --wands-dir "$TASK_WANDS" --selection wands_holdout_selection
  run verify_order_fallback_selection.py --wands-dir "$TASK_WANDS" --selection wands_fallback_validation_selection
  run verify_selective_execution_selection.py --wands-dir "$TASK_WANDS" --selection wands_selective_execution_selection
else
  printf '\nNo WANDS source directory supplied: source-table selection checks skipped.\n'
fi
for TASK_RUN in run_wands_holdout_qwen_v1 run_wands_holdout_smollm_v1; do
  run verify_wands_holdout.py --run "$TASK_RUN" --selection wands_holdout_selection
  run verify_wands_holdout_two_way.py --run "$TASK_RUN" --selection wands_holdout_selection
done
run verify_order_fallback_validation.py --run run_wands_fallback_validation_qwen_v1 --selection wands_fallback_validation_selection
run verify_order_fallback_two_way.py --run run_wands_fallback_validation_qwen_v1 --selection wands_fallback_validation_selection
run verify_selective_execution.py --run run_wands_selective_execution_qwen_v1 --selection wands_selective_execution_selection
run verify_selective_two_way.py --run run_wands_selective_execution_qwen_v1 --selection wands_selective_execution_selection
run verify_selective_model_sensitivity.py --run run_wands_selective_execution_smollm_v1 --selection wands_selective_execution_selection
run verify_selective_model_two_way.py --run run_wands_selective_execution_smollm_v1 --selection wands_selective_execution_selection
run verify_local_service_benchmark.py --run run_local_service_benchmark_resumed --selection wands_selective_execution_selection
run verify_wands_supervised_reference.py --run run_wands_supervised_reference
run verify_wands_calibrated_lexical.py --run run_wands_calibrated_lexical
run verify_wands_embedding_reference.py --run run_wands_embedding_reference
run verify_wands_embedding_replay.py --prior run_wands_embedding_reference --replay run_wands_embedding_replay
printf '\nSelected CPU verification suite passed. This does not rerun GPU inference.\n'
