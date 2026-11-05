# Frozen follow-up: actual selective fallback on new ESCI query pools

This follow-up is fixed after the earlier ESCI, WANDS and biomedical results and
after a counts-only feasibility check, but **before selecting the new query IDs,
examining their labels, or running either model**. It evaluates the previously
specified selective-fallback mechanism on new query pools from the *same public
ESCI source*. It is not a new source, a user-satisfaction study, or proof that
either model has not seen the public data during pretraining.

## Selection

Use the previously pinned Amazon example and product Parquet hashes. From the
US-English, small-version, test split, exclude the normalized (trimmed,
case-folded) query strings in `run_test_160`, `run_validation_500`,
`run_full_pools_250`, and the saved 50,000-example supervised training
manifest. Exclude a whole query if **any** of its labeled example product IDs
occurs in the earlier 250 full pools or that training manifest. This does not
prove product-ID isolation from the two derived samples, whose IDs were not
retained. Require nonempty normalized queries, a unique normalized query per
source query ID and a nonmissing product ID for every example. Reject ambiguous
source keys, unsupported grades, failed product joins or empty product text;
do not silently replace selected queries. Prior to this protocol, a counts-only
inspection found 3,963 distinct eligible query IDs/normalized strings under the
query and product-ID exclusions, without printing grade counts or selecting
queries. Fail if the eligible count differs.

Sort eligible numeric query IDs and sample 250 without replacement using
`random.Random(20271201)`. Retain **all** their judged examples in this split,
not a fixed 32-item sample and not a live result page. Sort by source example
ID, then shuffle each query using `random.Random` seeded by the first eight
bytes (big endian) of SHA-256 of the UTF-8 string `20271201`, a NUL, and its
decimal query ID. Assign one-based pool positions after shuffling. Preserve
source query, example and product IDs and grade, and concatenate the same five
product fields in the original order, truncated to 600 characters. E and S are
positive; C and I negative, as in the prior ESCI study. Products may repeat
within or across the new pools. Source annotation and label validity do not
establish actual shopper satisfaction, and the 600-character text may omit
information available to annotators.

## Execution

Before either model is run, freeze the selection, code, this protocol, pinned
model/tokenizer hashes and run configuration. Run Qwen3-4B-Instruct-2507 and
SmolLM2-1.7B-Instruct separately on the exact same fixed pools and system prompt
from the original ESCI study. Their sizes, families, tokenizers and templates
differ; the same unretuned prompts are an implementation comparison, not a
controlled model-family experiment. The Qwen contrast is primary and SmolLM2 is
a secondary pre-specified sensitivity. Use the documented local-copy isolated
environment, vLLM 0.10.2 V1, bfloat16, one RTX 5090, temperature zero, 4,096
token context, eager mode, prefix caching and 16 sequences. Check that no other
GPU process is using the device, and recheck ownership between batches; do not
stop other processes. Run batches of at most 64 prompts and record all outcomes.

First verify incompatible-instruction negative controls for the one-label and
keyed-output grammars. Fail closed if they do not work. For each query, generate
the prior query specification with one retry; direct fallback is allowed when
the specification remains invalid. Split candidates into consecutive chunks of
at most eight with stable chunk-local IDs. Generate original-order and reversed
order keyed JSON for each chunk, preserving ID association, with the same
`keyed_prompt`, fixed-key regex, cap 128 and at most one retry. The final chunk
can be smaller than eight; its regex contains exactly its IDs. If either keyed
response for a chunk remains invalid, its pool is non-estimable for selective
execution. For each disagreement in valid paired chunks, actually issue a
pointwise `[YN]`-constrained request (cap four, at most one retry) with the
same query, specification/fallback and item. Keep agreements and replace
disagreements with these actual pointwise outputs; invalid required outputs
cause pool abstention. Then run original-order keyed replay and the full
pointwise comparator on every candidate. The later full comparator is a
diagnostic and cannot supply the actual selective output.

Save per-request prompt hashes, local IDs and order, initial/final outputs,
retry and finish reasons, token counts and context checks. Record actual calls,
logical tokens, completions and offline batch wall time, with model loading,
negative controls, replay and full-pointwise calls excluded from the selective
plan. One fixed-order offline run with prefix caching is not a production
latency measurement or a universal speedup claim. Original and reverse runs
are not randomized in time and change prompt context as well as position.

## Analysis

Before observing either model's new outcomes, the primary contrast is Qwen's
query-macro ratio MAE of actual selective fallback **minus** canonical keyed
original on queries for which original, reverse, actual selective, replay and
full-pointwise all complete. Use 4,000 paired query bootstrap resamples with
NumPy `default_rng(20271202)` and percentile endpoints; report the observed
effect and full interval, whether better, worse, null or incomplete. This
criterion is exploratory across the broader project because it follows earlier
results, and failure-conditioned selection and repeated products limit the
bootstrap interpretation. Do not treat the interval as guaranteed population
coverage. Use the same fixed analysis for SmolLM2 as a secondary sensitivity.

For each model also report completions over all 250 attempted pools; candidate
F1, balanced accuracy and ratio MAE on the common set; all-Y, all-N and
all-query-words baselines; pointwise quality; keyed label changes after reversal
and repeated identical prompts; disagreement and actual selective-vs-later-full
pointwise changes; calls, retry counts and logical input/output tokens. Clearly
distinguish candidate-weighted F1 from query-macro MAE and predictions from
reference labels. Report all model outcomes and failures even if a method is
worse. No other primary comparisons or multiple-testing correction are implied.
