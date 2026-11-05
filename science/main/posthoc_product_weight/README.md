# Post-hoc shared-product weighting sensitivity

This is an after-the-fact sensitivity audit of the 7B schema-guided versus unguided **selective** pipelines, using already observed ESCI and WANDS queries. It resamples queries and uses positive exponential weights shared by product ID. Reweighting products inside pools changes the target: the percentile ranges are not confidence intervals for the original fixed-pool MAE. The source protocol, both datasets, seeds and checks were frozen before any values from this sensitivity calculation were computed. This does not make the underlying model experiment prospective.

First extract and verify the existing `posthoc_7b/posthoc_7b_artifacts.tar.gz` as documented in `posthoc_7b/README.md`, resulting in `AUDIT=/path/to/new_7b_audit/posthoc`. In the extracted `reproducibility` directory, using a Python environment with NumPy from `requirements-verify.txt`:

```bash
python3 posthoc_product_weight/verify_frozen_inputs.py --artifact . --work "$AUDIT" --audit posthoc_product_weight
PYTHONPATH=posthoc_product_weight python3 posthoc_product_weight/test_product_weight_sensitivity.py
python3 posthoc_product_weight/verify_product_weight_sensitivity.py --artifact . --work "$AUDIT" --result posthoc_product_weight/product_weight_sensitivity.json
```

The raw 4,000 weighted draws per dataset are retained for independent checks. To recompute a fresh result, use `analyze_product_weight_sensitivity.py --artifact . --work "$AUDIT" --out /path/to/new_result.json`. It refuses an existing output. Neither product reweighting nor parsing establishes semantic validity, generalization or purchase satisfaction.

Check the compact summary injected into the manuscript against the full saved draws with:

```bash
python3 posthoc_product_weight/verify_summary.py --result posthoc_product_weight/product_weight_sensitivity.json --summary posthoc_product_weight/product_weight_summary.json
```
