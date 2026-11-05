#!/usr/bin/env bash
set -euo pipefail
if [ "$#" -ne 3 ]; then
  printf '%s\n' 'Usage: rebuild_batch_supplement.sh QWEN_REBUILT_WORK TOKENIZER_MODEL_DIR NEW_WORK_DIR' >&2
  exit 2
fi
ACORD_PRIOR_WORK=$1
ACORD_MODEL=$2
ACORD_NEW_WORK=$3
ACORD_HERE=$(cd "$(dirname "$0")" && pwd)
ACORD_PYTHON=${INTENTSAT_PYTHON:-python3}
test -f "$ACORD_PRIOR_WORK/selection_2400_corrected/inputs.jsonl"
test -f "$ACORD_PRIOR_WORK/restored_results/summary.json"
test -d "$ACORD_MODEL"
test ! -e "$ACORD_NEW_WORK"
test ! -L "$ACORD_NEW_WORK"
(cd "$ACORD_HERE" && sha256sum -c SHA256SUMS.txt > /dev/null)
mkdir "$ACORD_NEW_WORK"
export PYTHONDONTWRITEBYTECODE=1 USE_TORCH=0 USE_TF=0 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$ACORD_HERE${PYTHONPATH:+:$PYTHONPATH}"
"$ACORD_PYTHON" "$ACORD_HERE/preflight_batch.py" --selection "$ACORD_PRIOR_WORK/selection_2400_corrected" --model "$ACORD_MODEL" --protocol "$ACORD_HERE/PROTOCOL_recovery_batch.md" --out "$ACORD_NEW_WORK/preflight"
cmp "$ACORD_HERE/preflight_summary.json" "$ACORD_NEW_WORK/preflight/summary.json"
"$ACORD_PYTHON" "$ACORD_HERE/restore_pseudonymous_results.py" --selection "$ACORD_PRIOR_WORK/selection_2400_corrected" --redacted "$ACORD_HERE/results_pseudonymous" --manifest "$ACORD_HERE/RESTORE_SHA256.json" --out "$ACORD_NEW_WORK/restored_results"
"$ACORD_PYTHON" "$ACORD_HERE/analyze_batch.py" --selection "$ACORD_PRIOR_WORK/selection_2400_corrected" --recovery "$ACORD_NEW_WORK/restored_results" --prior "$ACORD_PRIOR_WORK/restored_results" --out "$ACORD_NEW_WORK/analysis.json"
"$ACORD_PYTHON" "$ACORD_HERE/verify_batch.py" --selection "$ACORD_PRIOR_WORK/selection_2400_corrected" --preflight "$ACORD_NEW_WORK/preflight" --recovery "$ACORD_NEW_WORK/restored_results" --prior "$ACORD_PRIOR_WORK/restored_results" --analysis "$ACORD_NEW_WORK/analysis.json"
"$ACORD_PYTHON" - "$ACORD_HERE/RESTORE_SHA256.json" "$ACORD_NEW_WORK/analysis.json" <<'PYCODE'
import hashlib,json,sys
with open(sys.argv[1],encoding='utf8') as f: manifest=json.load(f)
with open(sys.argv[2],'rb') as f: actual=hashlib.sha256(f.read()).hexdigest()
assert actual==manifest['analysis_sha256'], 'Paired analysis hash mismatch'
print('Prompt preflight, saved outputs and paired statistics reproduced; no model inference was rerun.')
PYCODE
