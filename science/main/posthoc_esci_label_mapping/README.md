# Post-hoc ESCI label-mapping sensitivity

The same 250 new-query pools and already saved outputs are scored under two
different reference mappings: E/S positive (the original task) and E-only
positive. The analysis and fixed bootstrap seed are described in
`PROTOCOL_label_mapping_sensitivity.md`. `PRE_ANALYSIS_SHA256.json` records
the data, protocol and analysis code before this post-hoc calculation.
`verify_label_mapping.py` was written afterward as a separate verification and
reconstructs selective decisions from the raw response events using the
previously saved execution analysis.

The E-only calculation is **not a new E-only inference or training run**: the
original prompts explicitly include functional substitutes, and the supervised
reference was trained with E/S-positive labels. Both references are proxies,
not independently adjudicated purchase satisfaction. Intervals describe the
observed sampled pools and are not multiplicity-adjusted or causal guarantees.

From the parent reproducibility directory, with NumPy installed and an unused
output path:

```bash
python3 posthoc_esci_label_mapping/test_label_mapping.py
python3 posthoc_esci_label_mapping/analyze_label_mapping.py \
  --root . --out /path/to/new_mapping_result.json
python3 posthoc_esci_label_mapping/verify_label_mapping.py \
  --root . --analysis-dir posthoc_esci_label_mapping
```

The verifier checks the included `result.json`; independently compare a newly
generated result with it. It imports the existing selective-run reconstruction
code, whose dependencies are included elsewhere in the reproducibility bundle.
