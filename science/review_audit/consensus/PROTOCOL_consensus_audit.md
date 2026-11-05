# Post-hoc audit of consensus-preserving baselines

This protocol was written during independent manuscript review on 2026-11-02, **after** the manuscript, existing traces and preliminary AND/OR results were inspected. It is an exploratory, conditional re-analysis, not preregistration, a new blind test, or evidence of causal or population effects. Existing authoritative files are read only. No model inference is performed.

## Question and fixed policies

The selective rule retains agreement between canonical and stable-ID-reversed keyed outputs and uses a pointwise answer on disagreement. Two elementary comparators preserve exactly the same agreement: AND returns N on every disagreement; OR returns Y on every disagreement. Both use the existing two keyed passes and specification, but require no pointwise calls. Both comparators are reported in every setting; selecting the best policy using test labels is not proposed as an online algorithm.

Analyze all four existing model-by-sample settings: Qwen and SmolLM2 on the same 52 WANDS pools and the same 250 new ESCI pools. There are 604 pool-model instances, **302 distinct pools**, and four distinct within-setting comparisons. Do not combine their bootstrap samples as independent observations. Preserve the authoritative fixed pool denominator, source label mapping and exact candidate order. Candidate outputs are reconstructed from raw JSON records and verified against saved selective outputs. Actual selective responses and later full-pointwise responses are kept distinct.

## Metrics and uncertainty

For canonical, reversed, full pointwise, actual selective, cached selective, AND and OR, report pooled F1, pooled balanced accuracy, query-macro candidate error and query-macro ratio MAE. For selective-minus-AND and selective-minus-OR compute paired query-bootstrap percentile intervals using 4,000 draws. Use seeds 2026110210, 2026110211, 2026110212, 2026110213 in the setting order above. All intervals are descriptive and unadjusted for multiple comparisons. They condition on complete saved traces and the chosen pools, ignore finite-frame corrections, and do not guarantee coverage or independent replication across models.

Recompute the exact uniform-k-subset reference with the same **later full-pointwise** labels as cached selective. Match k separately in each pool to the observed keyed disagreement count. In the primary implementation use multivariate hypergeometric counts; verify every pool separately with a position-by-position integer dynamic program. These are matched **pointwise-label** budgets, not equal total resources: the second keyed pass is part of routing cost. Report the existing trace-derived total calls/tokens of the two-keyed baseline by subtracting actual selective pointwise requests from the recorded selective pipeline. Do not invent latency for virtual AND/OR.

## Label-composition attribution

Report all-negative, mixed and all-positive pools separately, using gold labels only for descriptive attribution. Give subgroup counts, within-group mean cached-minus-expected-uniform MAE, unadjusted paired intervals, and each group's contribution `(group count / total count) * group contrast` to the original total contrast. These groups are unavailable without reference labels and cannot be used to route new queries. Do not claim that a nonsignificant subgroup effect equals zero, or that subgroup contrasts establish interaction significance. Do not filter the headline sample or change the primary fixed-pool estimand.

## Verification and reporting

Save source SHA-256 hashes, script, JSON results and an independent verification log under the reviewer directory. Independently reconstruct label vectors from raw responses, verify saved selective vectors, all exact uniform expectations, all point estimates and paired intervals. Explain which metrics trade off: lower ratio MAE alone does not prove dominance in F1/balanced accuracy. Report unfavorable results for both fixed policies and both models.

Run with Python 3 and NumPy only, `PYTHONDONTWRITEBYTECODE=1`, `OPENBLAS_NUM_THREADS=1`; put outputs in `/dev/shm` because the root disk is full. No GPU, network, deletions or changes to the authoritative package.
