# Readability evidence from the unchanged frozen study

These are explanatory traces and exact cost decompositions from existing outputs, not new trials or a protocol amendment. No source news/review text or token arrays are included. The six original workloads still cover two text domains.

After the core artifact has been safely expanded, recompute with Python 3 standard library only:

```bash
python3 -B recompute_frozen_evidence.py --root /path/to/core/expanded --out /path/to/fresh_readability_evidence
```

Both `candidate_stop_trace.csv` and `phase_exact_cost.csv` must be byte-identical to the supplied CSVs. The generated JSON records the chosen local input root and source fingerprints, so those provenance fields differ between the original private tree and compact expansion. `EVIDENCE_RECOMPUTE_RECEIPT.json` preserves the actual prior original-versus-compact CSV check; the final public commit receives its own anonymous-download CPU replay receipt.

The trace scores and bills all requests in each returned 16-request chunk. Its first-decisive position is reconstructed in frozen trial order, not an online stopping point or an estimate of mismatch probability. Calibration references and candidate probes are derived accounting categories within the recorded calibration stage. The displayed reply examples are actual model completions and contain no source-news text. No extra model call, GPU job or random trial is performed. Upstream data/model rights remain as documented in the original scientific module.
