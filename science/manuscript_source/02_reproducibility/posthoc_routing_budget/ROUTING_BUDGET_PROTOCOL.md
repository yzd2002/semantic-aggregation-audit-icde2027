# Post-hoc matched-label-budget routing ablation

This protocol is written before computing this particular ablation. The samples,
model outputs, and earlier aggregate results were already exposed. It is not
preregistration, new GPU inference, or an independent held-out experiment.

Use all 52 WANDS and 250 new ESCI pools for each of Qwen3-4B and SmolLM2-1.7B.
Require every pool's original keyed, stable-ID reversed, full pointwise, and
actual selective records to be complete; fail instead of silently dropping pools.
Verify the prior original/full-pointwise/full-trace-fallback MAEs and confusion
counts, the exact selective routes, and the saved actual selective vectors.

For a fixed pool, let a and b be the two keyed vectors, h the saved full-pointwise
vector, n its size and k the number of a/b disagreements. The trace-routed plan
uses h on those k positions and a elsewhere. Compare it with a uniform subset
of exactly k positions, using the same h and a. Both calculations hold all
responses fixed; neither regenerates judgments under a different schedule.
The random reference is conditional on the observed, per-pool k; it is not an
executable scheme that knows future budgets without the keyed passes.

Compute the random reference's expectation exactly over all k-subsets, without
Monte Carlo. For ratio MAE, h-a has plus/zero/minus counts; use the corresponding
multivariate hypergeometric counts and sum the absolute positive-count error.
For candidate error, use linear inclusion probability k/n. Average each metric
equally over queries. Report routed minus expected-random MAE and candidate
error, regardless of direction. Use 4,000 paired query-bootstrap draws with
NumPy default_rng seeds 20261103, 20261104, 20261105, 20261106 for the four
settings in the script's fixed order. Percentile intervals are descriptive,
unadjusted across comparisons, and condition on saved traces and sample design.
They do not predict the outcome of one newly executed random policy.

Also report route counts and base keyed errors on agreeing/disagreeing candidates
to explain the signal's scope. Check all numerical results independently using
a position-by-position integer dynamic program, separate parsing/reconstruction,
and exhaustive small-pool enumeration. Source SHA-256 values and code hashes
will accompany results. No post-result model or pool selection is allowed.

This matches the number of inspected pointwise labels per query. It does not
match prompt-token totals or establish end-to-end cost/latency: random positions
can have different text lengths, and actual selective pointwise outputs can
differ from the later full-pointwise trace. Existing measured pipelines remain
the source of actual cost claims. Public-data, proxy-label, truncation, overlap,
pretraining and post-hoc limitations remain unchanged.
