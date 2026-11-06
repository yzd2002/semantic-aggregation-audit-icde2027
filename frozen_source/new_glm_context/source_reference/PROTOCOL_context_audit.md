# Fixed-target context audit — pre-selection protocol

Status: design fixed before target sampling, model judgments, or inspection of any new model outcome. The earlier manuscript and experiments are exposed. This is a new follow-up, not a retrospective preregistration of the earlier study.

## Question and scope

Does changing other input records change a target's judgment when the target text, identifier, input position and requested-label prefix are fixed? Does this sensitivity occur when no other record's label precedes the target output? Evaluate the correctness and aggregate consequences against the original public reference labels, without assuming every change is harmful.

The fixed requested-label prefix applies to full-first and target-only; full-last deliberately allows preceding generated background labels as a diagnostic arm. This is a controlled execution-contract audit of two semantic predicates, not a new matching/QA algorithm or a claim about a model's internal reasoning. Nearest contexts are selected by lexical TF–IDF similarity, not by semantic or gold labels. The intervention is a matched-length context replacement. A difference between target-only and full output is a whole output-scope contrast, not a pure mediation estimate.

## Models and unchanged runtime commitments

Use the already available Qwen3-4B-Instruct-2507 and SmolLM2-1.7B-Instruct, both in bfloat16, on the same RTX 5090. Do not replace a model or alter its prompt/cap after seeing task outputs. A contemplated SmolLM3 download was unavailable through the local network; no new-model judgments occurred. The two chosen families already appeared in the earlier study; this follow-up extends tasks and interventions, not the number of model families.

Use vLLM 0.10.2 / PyTorch 2.8.0 / Transformers 4.57.1, eager execution, tensor parallel size one, context 4096, temperature 0, output cap 128, maximum 16 sequences, no prefix caching, no retries, and explicit xgrammar guidance with fallback disabled. Preserve the inherited dependency conflicts and the previously tested process-local singleton NCCL adapter as disclosed runtime conditions; do not change installed packages or permissions. Record package/model/source hashes and grammar controls before data. All caches and outputs go to the new shared-memory work directory.

## Sources and eligibility

1. Bibliographic entity matching: DBLP–ACM official DeepMatcher files, with the independently matching Git mirror pinned to 84478b2082eb60834e37852637f7c403b34e857d. Use the public test pair split for targets and train pairs for backgrounds. Join tableA/tableB by the provided IDs. Preserve complete title/authors/venue/year fields, including empty authors. Y means the two citations identify the same publication. The official split is pair-disjoint, not entity-disjoint.
2. BoolQ passage-conditioned yes/no QA: google/boolq at repository revision 35b264d03638db9f4ce671b711558bf7ff0f80d5, official maintained Hub mirror. Use its labeled validation split (3270 records), not the hidden-label test split, for targets; use train (9427) for backgrounds. Preserve complete questions and passages. Y means the reference answer is yes. The original GCS project currently reports a closed billing account; do not claim byte comparison to GCS.

Read only feature strings/IDs during sampling and context selection. Gold labels are attached for evaluation after the selected inputs have been determined. Drop malformed/nonjoinable records, empty titles/questions/passages, and feature serializations exceeding 512 tokens with either model tokenizer. Do not truncate. Record all exclusion counts. Model pretraining exposure is unknown.

## Target reserve and background bank

Use deterministic seeds recorded in code: target seed 2026110201, context seed 2026110202, schedule seed 2026110203, bootstrap seed 2026110204.

For DBLP–ACM, randomly permute eligible left IDs, then independently permute each left ID's eligible test pairs. Select the first pair whose right ID and normalized complete citation fingerprints have not appeared among reserved targets. Reserve up to 256 targets with no reused left ID, right ID, or normalized full citation on either side. For BoolQ, seed-shuffle eligible validation rows and reserve up to 256 with distinct normalized questions and passages. Normalization is Unicode casefold plus whitespace collapse; it does not add inferred entity links. This is seeded selection with exclusions, not uniform sampling over all possible matchings.

Exclude from the training background bank every record sharing any reserved target's known entity ID or complete citation fingerprint (DBLP–ACM), or normalized question or passage (BoolQ). Feature-deduplicate the remaining bank without reference to labels. These exclusions concern observed identifiers/text, not all possible latent entity/topic overlap. Keep the source train/validation/test provenance distinct.

## Neighbor selection and length matching

Fit a word unigram/bigram TF–IDF model on the eligible background bank only: Unicode word tokens, lowercase, L2 normalization, no label features or supervised fitting. Use each target's complete serialized features as its search query. Rank by cosine similarity with a deterministic source-ID tie break. Select the top three background records greedily subject to distinct IDs/text within the triplet; for bibliographic pairs, also avoid repeating a left/right entity or full citation within the triplet.

For each near-neighbor position, a random-match proposal draws uniformly from remaining background records whose unwrapped feature token lengths differ from that position by at most w tokens in BOTH model tokenizers. Exclude the three near records and their observed entity/text identities, as well as duplicates within the random triplet. Try w=16, then 32, then 64, with at most 2000 seeded proposals per width. Accept the first triplet whose FULL prepared prompt length differs between near and random contexts by at most 24 tokens in each tokenizer, for every output scope. Do not select proposals based on lexical similarity, labels, or model predictions. Record the width and actual length/similarity differences.

Process reserved targets in their fixed order and select the first 128 that admit both contexts and pass all-model prompt-plus-cap preflight. If fewer than 128 are eligible, use all eligible targets and report the shortfall; never relax criteria after judgments. The background bank excludes all reserved targets and therefore stays fixed during this feasibility filter. Preserve unused reserve IDs, rejection reasons and chosen neighbor IDs.

## Fourteen requests per target and model

Every input starts with the target record at local ID 1, followed by three background records at IDs 2,3,4 in a fixed per-context order. Target features, local ID and input prefix are identical between the near and random contexts. Each task uses one fixed predicate instruction and explicitly says to judge each record only from its own fields; other records are not additional evidence for it.

For each near/random context run three output scopes, each twice:

- full-first: request IDs [1,2,3,4], grammar order 1,2,3,4;
- full-last: exactly the same prompt and requested-ID list, grammar order 2,3,4,1;
- target-only: same four input records, request ID [1], grammar order 1.

Also run two singleton references: only the target input, request ID [1], same task instruction and output syntax. Every response is an exact JSON object with quoted IDs and quoted Y/N values, no explanations, whitespace, duplicate/extra/missing keys or partial salvage. Full-first and target-only share the fixed output prefix before the target label. The output-scope instruction changes for target-only; this is disclosed, not treated as a perfectly isolated internal computation intervention.

Freeze one complete seeded randomized schedule of all requests, identical for both models. Submit blocks of at most 64 through a maximum-16-sequence engine. Preserve submitted blocks, returned raw strings, completion/caps, actual prompt/output tokens, timing, errors and all controls. No answer-based reordering, retries, stopping or choice of displayed conditions. At 128 targets per task this is 3584 requests per model, 7168 total, excluding synthetic controls.

Before inference independently verify target identity and token-prefix positions, requested IDs, full-first/full-last prompt byte equality, all maximum lengths, strict parser rejection examples and incompatible-instruction guidance controls. A technical failure may be repaired only in a new version with an append-only amendment; retain attempts and do not silently replace judged records.

## Estimands fixed before outcomes

For a given task/model/scope let a_t^(j) and b_t^(j) be the target binary decisions in near and random contexts, repeat j=1,2. Define each target's replacement excess:

e_t = (1[a_t^1 != b_t^1] + 1[a_t^2 != b_t^2])/2 - (1[a_t^1 != a_t^2] + 1[b_t^1 != b_t^2])/2.

The primary statistic is mean_t(e_t) for full-first, for each of the four task/model combinations. Report negative values. Report a 95% target-bootstrap interval and a predeclared 98.75% percentile interval for each of the four primary contrasts (Bonferroni family allocation); intervals remain approximate descriptions of these selected target sets, not population or causal guarantees. Use 4000 shared-index bootstrap draws. The predeclared positive-primary-evidence criterion is a 98.75% interval lower endpoint greater than zero; the 95% intervals are descriptive only. Failure to meet this criterion establishes neither no effect nor equivalence. Target-only and full-last excesses, the two paired scope contrasts E_target-only minus E_full-first and E_full-last minus E_full-first, and signed near-minus-random reference-error differences are secondary, with unadjusted 95% intervals, all reported.

Also report both directions of changed gold correctness from random to near context (wrong-to-right is improvement, right-to-wrong is worsening), candidate error/F1/balanced accuracy, constants all-Y/all-N, and the same-template singleton. Do not conflate low variation with high accuracy or treat every context-induced change as an error increase.

For COUNT/ratio, each task's selected targets form ONE fixed logical relation. The denominator is the number of selected targets; background records NEVER enter that relation. First sum the aligned target predictions over this whole relation, then compute predicted positive count, signed change, absolute change, and absolute error against its reference count, separately for every scope/context/repeat and both singletons. Do not average single-target absolute changes and call that an aggregate result: it would equal the candidate-flip metric. Report reference-free cancellation (flips vs net count change) and correctness-dependent cancellation separately. No arbitrary post-hoc groups are primary.

Completion is reported on all scheduled attempts. The primary paired candidate analyses use the common set complete across the 12 contextual requests; singleton comparisons additionally require both singleton responses. State attempted/common counts. A complete fixed-relation aggregate is emitted only if every selected target validates in that condition; otherwise report abstention, without silently changing the denominator. Keep all condition-specific results, including failures.

Sum actual calls and observed tokens, including every diagnostic request; no online speedup is claimed from interleaved timings. Do not claim a deployable optimized policy from these audits.

## Independent verification and interpretation

An independent implementation must reconstruct labels directly from untrimmed saved responses, re-evaluate all counts, flip directions and global COUNTs, independently compute bootstrap summaries, and check sampling/length constraints. Test that changing only gold labels does not change selected target/background IDs or prompts. Re-analysis must run from the delivered archive without original absolute directories or model inference.

If full-first replacement excess survives replay comparison, the effect does not require previously generated background labels. This does not identify attention, reasoning, training contamination or a token-level internal mechanism. If it does not survive, retain the null/adverse finding. A different target-only result concerns the requested output scope as a whole. Nearest and random contexts may differ beyond their matched lengths; make no unconditional semantic-similarity causal claim. External evidence covers these complete-field tasks, selected targets, two previously studied model families and one runtime, not all semantic operators or production workloads.
