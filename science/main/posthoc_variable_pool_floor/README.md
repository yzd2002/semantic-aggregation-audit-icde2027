# Variable-pool deterministic ratio floor

For Q aligned pairs of complete binary output vectors with pool sizes n_q and
positive counts a_q, b_q, the larger query-macro absolute ratio error against a
common fixed binary reference is at least

`sum_q(abs(a_q-b_q)/n_q) / (2*Q)`.

Unlike the integer-corrected equal-size floor in the manuscript, this universal
unequal-size bound need not be attainable. A constructive synthetic reference
count vector gives an upper bound on the sharp minimum; the code proves the
construction is no more than `1/(2*Q*min(n_q))` above the lower bound. The
construction assumes unrestricted occurrence labels. It does not establish a
bound's attainability if duplicate products or other references are coupled.

The included post-hoc ESCI result uses 250 previously observed variable-size
pools and two models. The floor uses output counts alone; stored E/S labels are
used only to check that the observed MAEs obey it. A fixed shared reference and
correct candidate alignment are required. This is not a confidence interval,
an independent sample, a causal order effect or evidence of user satisfaction.

From the parent reproducibility directory, using Python 3.11 or later:

```bash
python3 posthoc_variable_pool_floor/test_variable_pool_floor.py
python3 posthoc_variable_pool_floor/analyze_variable_pool_floor.py \
  --root . --out /path/to/new_variable_floor_summary.json
python3 posthoc_variable_pool_floor/verify_variable_pool_floor.py \
  --root . --analysis-dir posthoc_variable_pool_floor
```

`PRE_ANALYSIS_SHA256.json` records the protocol, code and inputs before this
post-hoc calculation. The verifier was written afterward and reconstructs the
two keyed outputs independently from the raw events.
