# Controlled input-row / output-key order audit

This follow-up follows all earlier results and reviewer-inspired redesign. It is
an exploratory project-level study; the design and analysis below are fixed
before the new ESCI selection or either model's new judgments. WANDS is an
explicitly exposed-sample mechanistic replication. Do not replace old results.

## Target and intervention

The target remains the reference-positive proportion of each complete fixed
candidate pool. Candidate error and disagreement are separate diagnostics;
an exact ratio is a correct answer to the ratio-only task even if errors cancel.
This is not a new aggregation or selective-voting algorithm.

Hold the candidate list, text prefixes, binary relevance rule, chunk membership,
occurrence ID, system prompt, user-template wording, output alphabet and token
cap fixed. In each chunk of at most eight, cross input row order (canonical vs
reverse) with output-key serialization (ascending vs descending numeric ID).
The prompt explicitly permits either key order; the decoding grammar specifies
the order. Thus the output-order intervention changes the grammar, not the user
instruction. IDs are original one-based pool occurrence IDs, never reassigned
by input order or singleton execution. No generated query specification is used.
The four conditions are AA, RA, AD, RD (input first, output second).

Make two identical-prompt/grammar repetitions of every condition. Interleave all
eight condition–repeat requests in a deterministic random order within each
chunk, and randomize chunk order using seed 2026110217. Serve sequential blocks
of at most 64 requests, with at most 16 active sequences, prefix cache disabled,
temperature 0, max_tokens=128, max_model_len=4096, bfloat16 and eager vLLM.
The fixed order is shared between models. Resource timing is diagnostic for this
mixed workload only, not a per-plan latency or speedup experiment.

Also issue two repeats of the same keyed-JSON template with a single candidate
and that candidate's unchanged occurrence ID. These are interleaved with chunk
jobs in deterministic order, not treated as human truth. Check singleton F1,
balanced accuracy, ratio MAE and candidate error as well as agreement to batches.
No retries are permitted in this new experiment: any raw output that fails the
exact order-specific grammar is recorded as invalid and causes pool abstention.
This isolates the guided-output path and differs from older one-retry plans.

## Samples

ESCI: select 128 US-English small-version test query IDs, retaining all judged
candidates. Use the previously pinned Amazon tables and field serialization,
with 600-character prefixes. Exclude normalized query strings, product IDs and
exact serialized candidate texts used in all previous ESCI samples or supervised
training. Include prior WANDS query strings in the query exclusion as well.
Training product text is reconstructed by the same five-field serialization.
Eligibility is determined without grade-stratification. Sample uniformly from
sorted eligible IDs with seed 2026110201; order candidates with a per-query SHA-256
seed derived from that seed and query ID. A missing join, ambiguous ID, empty
text or unsupported selected grade is an error, not grounds to replace a query.
First freeze the selected inputs and code hashes; summarize labels only later.

WANDS: reuse the existing 52 selective-execution pools, same 32 candidates each,
600-character texts, and Exact-only positive rule. This is not a new holdout.
Both models use identical pool inputs and schedule. Qwen3-4B-Instruct-2507 is the
primary execution; SmolLM2-1.7B-Instruct is a planned secondary sensitivity.
Source pretraining exposure and reference-label noise remain unknown.

## Primary and secondary descriptive comparisons

For each output order, compute per-pool candidate disagreement between input
orders within repeat 0 and repeat 1, averaged over repeats. Subtract the mean
same-condition replay disagreement for those two input orders. This excess may
be negative; never clip it. Query-macro average it and compute 4,000 paired
query-bootstrap percentile draws with fixed seed 2026110220 for Qwen/ESCI,
2026110221 for Qwen/WANDS, 2026110222 for Smol/ESCI, 2026110223 for Smol/WANDS.
Both output-order contrasts are co-primary descriptive contrasts, not a single
winner-selected test. Report their difference (effect modification by key order).

Report analogous absolute ratio changes, all four condition/repeat qualities,
aligned label changes, completion across every attempted pool, per-condition
replay, batch–singleton disagreement, and constants. Quality contrasts use the
common-complete set across the compared conditions; list any exclusions.
Report input-row order contrasts for reference ratio MAE and candidate error.
All intervals are unadjusted and sample-descriptive; these few repetitions do
not imply deterministic execution, population coverage, or an isolated model
cognitive mechanism. Correlated repeated products can limit query bootstrap.

## Implementation, provenance, and failure rules

Freeze code, this protocol, sample selection and model-file SHA-256 before new
GPU judgments. Record exact wrapped prompts, prompt/grammar/ID-map hashes, raw
outputs without trimming, finish reasons, tokens and execution order. Verify
output guidance against incompatible BANANA instructions before any data run.
Reject token overflow before execution. Check GPU is idle initially and only
owned processes use it during batches. Do not stop other tasks. Install any
dependency overrides in a process-local temporary directory; preserve install
failures and disclose inherited dependency conflicts if they remain. Never
describe an inherited environment as a fresh-machine replication.

If prerequisites fail, retain failure evidence and do not invent results. Any
amendment after selection/inference starts must be separately timestamped and
its exposure status stated. All successful and negative results are retained.
