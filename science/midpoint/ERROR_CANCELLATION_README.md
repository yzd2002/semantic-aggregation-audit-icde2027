# Post-hoc candidate-error and ratio-error decomposition

This is a descriptive re-analysis of previously observed Qwen outputs on the
52-query WANDS actual-selective study and the 250-query new ESCI study. It is
not a new model run, held-out test, preregistered analysis, or confidence bound.
Both fixed samples are used in full; no pools are excluded.

For a complete pool of size n, let FP/FN count disagreements with its fixed
binary reference. The absolute ratio error is |FP-FN|/n, candidate error is
(FP+FN)/n, and their difference is 2 min(FP,FN)/n. These elementary identities
explain how opposing candidate errors can cancel in the aggregate. Reported
means normalize each pool first, then weight queries equally; the F1 check
instead pools confusion counts, as in the original study.

Plans: canonical original keyed output, actual selective fallback, full
pointwise output, all-Y and all-N. The analysis verifies every required keyed
and pointwise output, exact routing of selective requests, saved selective
vectors, and agreement with earlier confusion counts, F1 and MAE. Output JSON
contains SHA-256 values of all 14 source files. JSONL is split on ASCII LF,
preserving Unicode line-separator characters inside valid JSON strings.

Use Python 3.12 (standard library only), the main artifact's `reproducibility/`
directory, and a new output path. No model, GPU or additional data is needed:

```sh
python3 test_error_cancellation.py
python3 error_cancellation.py --artifact /path/to/reproducibility --out /path/to/new_error_cancellation.json
python3 verify_error_cancellation.py --artifact /path/to/reproducibility --result /path/to/new_error_cancellation.json
```

`verify_error_cancellation.py` independently parses keyed pairs, reconstructs
vectors using query/position maps, and computes cancellation from the integer
number of candidate mistakes minus the absolute positive-count difference.
The test exhausts 5,460 binary vector pairs of lengths 1–6. The saved result
is `error_cancellation_summary.json`; the local verification log records the
checks, not a fresh-machine inference replication. Reference-label ambiguity,
sampling bias, truncation and prior sample exposure remain unchanged.
