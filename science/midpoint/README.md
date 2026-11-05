# Post-hoc ratio, tie-breaker, error-decomposition and routing analyses

This small supplement belongs beside the full `reproducibility_artifact.zip`; it does not contain the original model responses, full datasets or weights. Extract that artifact and this supplement into separate folders. Point `--artifact` to the extracted `reproducibility/` directory and `--supplement` to this directory. The reference is fractional and ratio-only; it does not provide candidate labels or F1. Inputs and results were already exposed.

The following commands use Python and NumPy 1.26.4. Use a fresh output path outside the artifact.

```bash
python3 verify_source_hashes.py --artifact /path/to/reproducibility --supplement .
python3 midpoint_reference.py --artifact /path/to/reproducibility --out /path/to/new_midpoint_result.json
python3 verify_midpoint_reference.py --artifact /path/to/reproducibility --result /path/to/new_midpoint_result.json
```

The source hash JSON marked PRECOMPUTATION records the original inputs and the initial implementation before this specific computation. The initial script was renamed to `midpoint_reference_initial.py` after printing three WANDS outcomes and then failing on an ESCI Unicode separator. The amendment and failure log document the correction; do not claim that the corrected implementation was frozen before outcomes. `POSTCOMPUTATION_DEPENDENCY_SHA256.json` records extra verifier and prior-summary inputs added after computation. `result.json` is the saved amended run, and `verification.log` records the independent strict-parser reconstruction and selected sensitivity checks. `SHA256SUMS.txt` records the exact supplement contents.

The reduced artifact that omits sampled TREC article text requires those two selection files to be reconstructed from fixed upstream sources before running the full analysis; follow its accompanying instructions. Third-party content and self-authored code remain subject to the authors' publication-rights review. Local rerunning is not a fresh-machine GPU inference or a new experiment.


## Post-hoc lexical disagreement tie-breaker

The same supplement also includes a virtual tie-breaker that keeps the common label from two keyed outputs and applies the existing all-query-words rule on disagreement. It returns candidate labels, uses no additional language-model calls, and was evaluated after the samples and earlier outcomes were exposed. It is not the fractional midpoint. Its fixed-before-this-computation protocol and source hashes are recorded; they do not make the analysis prospective. CPU time and online latency were not measured.

```bash
python3 verify_lexical_sources.py --supplement .
python3 lexical_tiebreak.py --artifact /path/to/reproducibility --prior-result result.json --out /path/to/new_lexical_result.json
python3 verify_lexical_tiebreak.py --artifact /path/to/reproducibility --result /path/to/new_lexical_result.json
```

`lexical_result.json` reports all 12 predetermined settings and unadjusted query-bootstrap intervals; product-weighted ranges change the evaluated pools. `lexical_verification.log` records independent strict-parser, routing and statistical checks. Run the original `verify_source_hashes.py` as well to verify the underlying 55 source dependencies.


## Candidate-error cancellation diagnostic

The supplement now also includes a post-hoc fixed-pool decomposition of candidate error and absolute ratio error for Qwen on the 52 WANDS and 250 new ESCI pools. See `ERROR_CANCELLATION_README.md` for the elementary identity, query weighting, source hashes, reconstruction commands and independent check. All pools in both samples are retained; no new model inference was performed.


## Matched-label-budget routing ablation

`ROUTING_BUDGET_README.md` describes the new post-hoc ablation on all 604 product pools across two models. It uses exact expectations of uniform subsets, fixed per-pool disagreement budgets and identical saved full-pointwise traces on both plans. `ROUTING_BUDGET_PROTOCOL.md` and `ROUTING_PRECOMPUTATION_SHA256.json` preserve the design before this calculation, with previously exposed samples. Python 3.12/NumPy 2.2.6 match this analysis; older midpoint/lexical environments remain as documented above. See `ROUTING_VERIFICATION.log` for independent combinatorial/position-DP and archive-source checks. This matches label counts, not runtime or total resource use.
