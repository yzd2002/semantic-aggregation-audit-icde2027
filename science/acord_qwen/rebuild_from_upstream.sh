#!/usr/bin/env bash
set -euo pipefail
if [ "$#" -ne 4 ]; then
  printf '%s\n' 'Usage: rebuild_from_upstream.sh UPSTREAM_ZIP TOKENIZER_MODEL_DIR ORIGINAL_REPRO_DIR NEW_WORK_DIR' >&2
  exit 2
fi
ACORD_ARCHIVE=$1
ACORD_MODEL=$2
ACORD_REPRO=$3
ACORD_WORK=$4
ACORD_HERE=$(cd "$(dirname "$0")" && pwd)
ACORD_PYTHON=${INTENTSAT_PYTHON:-python3}
test -f "$ACORD_ARCHIVE"
test -d "$ACORD_MODEL"
test -f "$ACORD_REPRO/order_fallback.py"
test -f "$ACORD_REPRO/run_keyed_order_audit.py"
test ! -e "$ACORD_WORK"
test ! -L "$ACORD_WORK"
(cd "$ACORD_HERE" && sha256sum -c SHA256SUMS.txt > /dev/null)
mkdir "$ACORD_WORK"
cp -R "$ACORD_HERE/cap1200" "$ACORD_HERE/cap1800" "$ACORD_HERE/cap2400" "$ACORD_WORK/"
cp "$ACORD_HERE/AMENDMENT_prefix_search.md" "$ACORD_WORK/"
export PYTHONDONTWRITEBYTECODE=1 USE_TORCH=0 USE_TF=0 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$ACORD_REPRO:$ACORD_HERE${PYTHONPATH:+:$PYTHONPATH}"
"$ACORD_PYTHON" "$ACORD_HERE/select_acord_pools.py" --archive "$ACORD_ARCHIVE" --protocol "$ACORD_HERE/PROTOCOL_acord_holdout.md" --out "$ACORD_WORK/selection_600"
"$ACORD_PYTHON" "$ACORD_HERE/verify_acord_selection.py" --archive "$ACORD_ARCHIVE" --selection "$ACORD_WORK/selection_600" --protocol "$ACORD_HERE/PROTOCOL_acord_holdout.md"
"$ACORD_PYTHON" "$ACORD_HERE/prepare_long_prefix.py" --archive "$ACORD_ARCHIVE" --base "$ACORD_WORK/selection_600" --amendment "$ACORD_HERE/AMENDMENT_1200_preflight.md" --out "$ACORD_WORK/selection_1200"
for ACORD_N in 1800 2400; do
  "$ACORD_PYTHON" "$ACORD_HERE/prepare_prefix_variants.py" --archive "$ACORD_ARCHIVE" --base "$ACORD_WORK/selection_600" --amendment "$ACORD_HERE/AMENDMENT_prefix_search.md" --out "$ACORD_WORK/selection_${ACORD_N}_corrected" --length "$ACORD_N"
done
for ACORD_N in 1200 1800 2400; do
  ACORD_SUFFIX=''
  if [ "$ACORD_N" != 1200 ]; then ACORD_SUFFIX='_corrected'; fi
  "$ACORD_PYTHON" "$ACORD_WORK/cap${ACORD_N}/preflight_acord.py" --selection "$ACORD_WORK/selection_${ACORD_N}${ACORD_SUFFIX}" --model "$ACORD_MODEL" --out "$ACORD_WORK/preflight_${ACORD_N}"
  cmp "$ACORD_HERE/preflight_${ACORD_N}_summary.json" "$ACORD_WORK/preflight_${ACORD_N}/summary.json"
done
"$ACORD_PYTHON" "$ACORD_HERE/verify_prefix_variants.py" --source "$ACORD_ARCHIVE" --base "$ACORD_WORK/selection_600" --root "$ACORD_WORK" --out "$ACORD_WORK/decision.json"
cmp "$ACORD_HERE/PREFIX_DECISION.json" "$ACORD_WORK/decision.json"
"$ACORD_PYTHON" "$ACORD_HERE/restore_pseudonymous_results.py" --selection "$ACORD_WORK/selection_2400_corrected" --redacted "$ACORD_HERE/results_pseudonymous" --manifest "$ACORD_HERE/RESTORE_SHA256.json" --out "$ACORD_WORK/restored_results"
"$ACORD_PYTHON" "$ACORD_HERE/analyze_acord_holdout.py" --selection "$ACORD_WORK/selection_2400_corrected" --run "$ACORD_WORK/restored_results" --out "$ACORD_WORK/analysis.json"
"$ACORD_PYTHON" "$ACORD_HERE/analyze_acord_pointwise_supplement.py" --selection "$ACORD_WORK/selection_2400_corrected" --run "$ACORD_WORK/restored_results" --out "$ACORD_WORK/pointwise.json"
cmp "$ACORD_HERE/pointwise_supplement.json" "$ACORD_WORK/pointwise.json"
"$ACORD_PYTHON" "$ACORD_HERE/verify_acord_run.py" --selection "$ACORD_WORK/selection_2400_corrected" --preflight "$ACORD_WORK/preflight_2400" --run "$ACORD_WORK/restored_results" --analysis "$ACORD_WORK/analysis.json"
"$ACORD_PYTHON" "$ACORD_HERE/verify_pointwise_supplement.py" --selection "$ACORD_WORK/selection_2400_corrected" --run "$ACORD_WORK/restored_results" --primary "$ACORD_WORK/analysis.json" --supplement "$ACORD_WORK/pointwise.json"
"$ACORD_PYTHON" - "$ACORD_HERE/RESTORE_SHA256.json" "$ACORD_WORK/analysis.json" <<'PYCODE'
import hashlib,json,sys
with open(sys.argv[1],encoding='utf8') as f: manifest=json.load(f)
with open(sys.argv[2],'rb') as f: actual=hashlib.sha256(f.read()).hexdigest()
assert actual==manifest['primary_analysis_sha256'], 'Primary analysis hash mismatch'
print('Reconstructed selections, all tokenizer checks, saved responses and both analyses verified; no model inference was rerun.')
PYCODE
