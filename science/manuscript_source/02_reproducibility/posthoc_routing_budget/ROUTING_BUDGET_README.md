# Equal-label-budget routing ablation

This post-hoc analysis compares cached disagreement routing with the exact
expected error of a uniform subset of the same size on each fixed pool. It
does not execute a random policy or establish a latency advantage.

Let a be the canonical keyed vector, h the saved full-pointwise vector, g the
reference positive count and k the observed keyed disagreement count. Among
the n values h-a, let u/v/w count +1/-1/0. The expected absolute ratio error
of uniform checking is

    sum_{i,j} C(u,i) C(v,j) C(w,k-i-j) |sum(a)-g+i-j|
    -------------------------------------------------
                     n C(n,k)

over feasible i,j. The candidate-error expectation is
`(1-k/n)*e(a,gold) + (k/n)*e(h,gold)`. These are elementary finite-population
expectations, not a new routing algorithm. The independent verifier uses a
position-wise integer dynamic program instead of the combinatorial formula.

Use the main artifact's extracted `reproducibility/` directory. Python 3.12
and NumPy 2.2.6 match the recorded routing analysis; the older midpoint/lexical
analyses retain their separately recorded NumPy 1.26.4 environment. This module
also uses the included `error_cancellation.py` parser/reconstruction helpers.
Use a new output path:

```sh
python3 test_routing_budget.py
python3 routing_budget.py --artifact /path/to/reproducibility --out /path/to/new_routing_result.json
python3 verify_routing_budget.py --artifact /path/to/reproducibility --result /path/to/new_routing_result.json
```

`ROUTING_BUDGET_PROTOCOL.md` fixes all four settings before this calculation;
the data, responses and previous outcomes were already exposed. The
precomputation snapshot identifies the protocol, implementation, test, helper
and 26 source files. The independent verifier was added after computation and
is covered by the supplement's final SHA manifest, not that earlier snapshot.

`routing_budget_result.json` contains exact rational expectations per pool,
query-macro means, 4,000-draw unadjusted query-bootstrap intervals, base errors
on agreement/disagreement regions, counts and source hashes. These intervals
condition on the frozen traces and describe sampled-query variation. They are
not confidence bounds for a newly generated random policy or a population
deployment. The protocol records the four fixed seeds and limitations.

Qwen's routed/expected-uniform MAEs are 0.118389/0.130066 on WANDS and
0.372986/0.384355 on ESCI. The paired intervals are below zero in these samples.
SmolLM2's WANDS difference spans zero; on ESCI the routed MAE 0.812698 exceeds
0.803540 expected uniform. All 604 pools are retained. Actual selective outputs
can differ from the later full-pointwise trace; the paper reports their measured
pipeline results separately. Equal label budgets do not equal prompt tokens,
runtime, or future inference behavior. No new model inference was performed.
