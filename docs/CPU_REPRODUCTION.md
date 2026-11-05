# CPU reproduction from retained traces

First verify and unpack all selected assets with the two repository tools.
Set `INTENTSAT_WORK` to that newly extracted workspace and `INTENTSAT_NEW` to
a new result directory whose parent already exists. Use unoptimized Python:
the independent historical checks use assertions. No GPU inference is involved.

The workspace preparation tool also copies the unchanged delivered
`midpoint_ratio_supplement/routing_budget_result.json` to the frozen consensus
analyzer's expected `reproducibility/posthoc_routing_budget/` location. Its
SHA-256 must match the original consensus dependency. The mapping is recorded
in `EXTRACTION_RECEIPT.json`; no archive or frozen implementation is edited.

```sh
export PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1
INTENTSAT_WORK=/path/to/extracted-workspace
INTENTSAT_NEW=/path/to/new-results
mkdir "$INTENTSAT_NEW"
```

## 1. Standard-library cancellation diagnostic

This covers all 52 WANDS and 250 ESCI pools in the saved Qwen comparison.
The analyzer and independent verifier are separate retained implementations.

```sh
python3 -B "$INTENTSAT_WORK/midpoint_ratio_supplement/error_cancellation.py" \
  --artifact "$INTENTSAT_WORK/reproducibility" \
  --out "$INTENTSAT_NEW/error_cancellation.json"
python3 -B "$INTENTSAT_WORK/midpoint_ratio_supplement/verify_error_cancellation.py" \
  --artifact "$INTENTSAT_WORK/reproducibility" \
  --result "$INTENTSAT_NEW/error_cancellation.json"
```

## 2. Standard-library count-envelope diagnostic

The launcher verifies the original inputs and all 604 pool–model rows, with
302 distinct query pools. The delivered consensus JSON is hash-identified;
its reference count makes this a diagnostic rather than an online rule.

```sh
python3 -B "$INTENTSAT_WORK/review_audit/envelope/portable_envelope.py" \
  --artifact "$INTENTSAT_WORK/reproducibility" \
  --consensus-audit "$INTENTSAT_WORK/review_audit/consensus/consensus_audit.json" \
  --output "$INTENTSAT_NEW/envelope"
```

## 3. Controlled-order replay (NumPy 2.2.6)

Use Python 3.12 with `numpy==2.2.6`. The replay reads the full retained
requests, responses and schedule and recomputes primary/supplementary results
for both models. It compares with saved values at tolerance `1e-12` and invokes
independent raw parsers and frequency-weight bootstrap checks. It needs neither
Torch nor vLLM.

```sh
python3 -B "$INTENTSAT_WORK/review_audit/controlled_order/replay_cpu.py" \
  --bundle "$INTENTSAT_WORK/review_audit/controlled_order" \
  --out "$INTENTSAT_NEW/controlled_order"
```

## 4. Consensus and routing (NumPy 2.2.6)

```sh
python3 -B "$INTENTSAT_WORK/review_audit/consensus/audit_consensus_baselines.py" \
  --artifact "$INTENTSAT_WORK/reproducibility" --out "$INTENTSAT_NEW/consensus.json"
python3 -B "$INTENTSAT_WORK/review_audit/consensus/verify_consensus_audit.py" \
  --artifact "$INTENTSAT_WORK/reproducibility" --result "$INTENTSAT_NEW/consensus.json"
python3 -B "$INTENTSAT_WORK/midpoint_ratio_supplement/routing_budget.py" \
  --artifact "$INTENTSAT_WORK/reproducibility" --out "$INTENTSAT_NEW/routing.json"
python3 -B "$INTENTSAT_WORK/midpoint_ratio_supplement/verify_routing_budget.py" \
  --artifact "$INTENTSAT_WORK/reproducibility" --result "$INTENTSAT_NEW/routing.json"
```

## Additional historical checks

The historical main `requirements-verify.txt` and `run_cpu_verification_suite.sh`
record the wider NumPy 1.26.4 environment. Use a separate environment for that
suite and the midpoint/lexical calculations; the supplied historical logs are
not a claim that this new staging session re-executed every command.

TREC-specific checks require reconstruction of the two omitted input files
first. ACORD reconstruction needs the exact upstream ZIP plus tokenizer-only
assets and its recorded NumPy/Transformers/Tokenizers versions. Read
[DATA.md](DATA.md) and the preserved historical guide for the precise commands.
The figure supplement records Matplotlib/font requirements. The manuscript can
compile from rendered source and bundled figures without those model/data steps.

Fresh model generation is a separate, resource-intensive task. Historical
runtime conflicts, grammar controls, local communication adapter and failed
attempts remain disclosed; none of these CPU commands establishes a complete
clean-machine GPU reproduction or reconstructs unavailable later GLM volumes.
