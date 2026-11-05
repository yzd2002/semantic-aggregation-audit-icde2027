#!/usr/bin/env bash
set -euo pipefail
if [ "$#" -ne 4 ]; then
  printf '%s\n' 'Usage: rebuild_smol_followup.sh QWEN_REBUILT_WORK SMOL_TOKENIZER_DIR ORIGINAL_REPRO_DIR NEW_WORK_DIR' >&2
  exit 2
fi
SMOL_QWEN_WORK=$1
SMOL_MODEL=$2
SMOL_REPRO=$3
SMOL_WORK=$4
SMOL_HERE=$(cd "$(dirname "$0")" && pwd)
SMOL_PYTHON=${INTENTSAT_PYTHON:-python3}
test -f "$SMOL_QWEN_WORK/selection_2400_corrected/inputs.jsonl"
test -f "$SMOL_QWEN_WORK/restored_results/summary.json"
test -d "$SMOL_MODEL"
test "$(basename "$SMOL_MODEL")" = SmolLM2-1.7B-Instruct
test -f "$SMOL_REPRO/order_fallback.py"
test -f "$SMOL_REPRO/run_keyed_order_audit.py"
test ! -e "$SMOL_WORK"
test ! -L "$SMOL_WORK"
(cd "$SMOL_HERE" && sha256sum -c SHA256SUMS.txt > /dev/null)
mkdir "$SMOL_WORK"
export PYTHONDONTWRITEBYTECODE=1 USE_TORCH=0 USE_TF=0 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$SMOL_REPRO:$SMOL_HERE${PYTHONPATH:+:$PYTHONPATH}"
"$SMOL_PYTHON" "$SMOL_HERE/preflight_acord.py" --selection "$SMOL_QWEN_WORK/selection_2400_corrected" --model "$SMOL_MODEL" --out "$SMOL_WORK/preflight"
cmp "$SMOL_HERE/preflight_summary.json" "$SMOL_WORK/preflight/summary.json"
"$SMOL_PYTHON" "$SMOL_HERE/restore_pseudonymous_results.py" --selection "$SMOL_QWEN_WORK/selection_2400_corrected" --redacted "$SMOL_HERE/results_pseudonymous" --manifest "$SMOL_HERE/RESTORE_SHA256.json" --out "$SMOL_WORK/restored_results"
"$SMOL_PYTHON" "$SMOL_HERE/analyze_acord_holdout.py" --selection "$SMOL_QWEN_WORK/selection_2400_corrected" --run "$SMOL_WORK/restored_results" --out "$SMOL_WORK/analysis.json"
"$SMOL_PYTHON" "$SMOL_HERE/verify_acord_run.py" --selection "$SMOL_QWEN_WORK/selection_2400_corrected" --preflight "$SMOL_WORK/preflight" --run "$SMOL_WORK/restored_results" --analysis "$SMOL_WORK/analysis.json"
"$SMOL_PYTHON" "$SMOL_HERE/analyze_acord_pointwise_supplement.py" --selection "$SMOL_QWEN_WORK/selection_2400_corrected" --run "$SMOL_WORK/restored_results" --out "$SMOL_WORK/pointwise.json"
cmp "$SMOL_HERE/pointwise.json" "$SMOL_WORK/pointwise.json"
"$SMOL_PYTHON" "$SMOL_HERE/verify_pointwise_supplement.py" --selection "$SMOL_QWEN_WORK/selection_2400_corrected" --run "$SMOL_WORK/restored_results" --primary "$SMOL_WORK/analysis.json" --supplement "$SMOL_WORK/pointwise.json"
"$SMOL_PYTHON" "$SMOL_HERE/compare_models.py" --selection "$SMOL_QWEN_WORK/selection_2400_corrected" --qwen "$SMOL_QWEN_WORK/restored_results" --smol "$SMOL_WORK/restored_results" --out "$SMOL_WORK/model_comparison.json"
cmp "$SMOL_HERE/model_comparison.json" "$SMOL_WORK/model_comparison.json"
"$SMOL_PYTHON" "$SMOL_HERE/verify_model_comparison.py" --selection "$SMOL_QWEN_WORK/selection_2400_corrected" --qwen "$SMOL_QWEN_WORK/restored_results" --smol "$SMOL_WORK/restored_results" --comparison "$SMOL_WORK/model_comparison.json"
"$SMOL_PYTHON" - "$SMOL_HERE/RESTORE_SHA256.json" "$SMOL_WORK/analysis.json" <<'PYCODE'
import hashlib,json,sys
with open(sys.argv[1],encoding='utf8') as f: m=json.load(f)
with open(sys.argv[2],'rb') as f: actual=hashlib.sha256(f.read()).hexdigest()
assert actual==m['primary_analysis_sha256'], 'Smol primary analysis hash mismatch'
print('Smol tokenizer, saved responses and analyses verified; no inference rerun.')
PYCODE
