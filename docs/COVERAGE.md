# Historical manuscript–artifact coverage

This map concerns the saved 2026-11-02 review manuscript, identified by exact
PDF/LaTeX hashes in the asset index. Its source has five manuscript figures and
seven tables. A later manuscript must be audited against its own final claims.
No asset completeness claim for later missing volumes is made here.

| Manuscript component | Required saved evidence / entry point |
| --- | --- |
| Target, contract, bounds (Section III) | Main: contract/parsers, occurrence IDs, `ratio_bounds.py`, `coupled_ratio_bounds.py`, `ratio_disagreement_bounds.py`, `stability_bounds.py`, exhaustive checks and conditional calibration protocols. |
| Execution plans / original Qwen and SmolLM2 studies (Sections IV–VI; Tables I–III; quality figure) | Main: fixed selections, all attempted responses, retries/failures, pointwise/direct/specification plans, original/full-pool analyses, LOTUS and supervised comparators. Completion and cost use attempted pools; quality uses specified common-complete pools. |
| WANDS independent, holdout and fallback studies; Table VI | Main: WANDS160, holdout, fallback-validation and selective-execution selections/runs; supervised, lexical and embedding references; binary Exact/Partial mapping sensitivity. |
| Row/key order, replay, random permutations | Main: original controls, stable-ID keyed/reversal runs, replay and random-order records; review-audit supplement: new 128 ESCI / exposed 52 WANDS controlled samples, full schedule, both model requests/responses, runtime amendments and independent CPU replay. Controlled-order figure and Table IV use the latter. |
| Ratio midpoint, lexical disagreement rule, cancellation, equal-label-budget routing figure | Midpoint supplement plus main: all settings, source hashes, exposed-sample amendments, exact uniform-subset expectations and independent verifier. 302 distinct pools are reused for two models, producing 604 pool–model cases. |
| Consensus AND/OR references; Table V; reference-count envelope | Review-audit `consensus/`, `envelope/`: frozen saved-trace computation, independent position/DP check, portable launcher. The envelope uses a known reference count and is not a deployable oracle-free rule. |
| Local HTTP service; Table VII | Main: original/resumed local service traces, workload, server settings, repeated completion/latencies and failed-attempt history. Does not establish hardware-independent deployment latency. |
| Historical third-model sensitivity / model-sensitivity figure | Main nested `posthoc_7b/` archive with Qwen2.5-7B tokenizer/metadata, prompt hashes, recorded responses, analyses, guided-specification amendments and integrity checks. Its unguided product specifications all used direct fallback. This material survives; later GLM material is a different missing study. |
| Historical cross-domain TREC-COVID | Main saved traces/source hashes plus fixed BEIR reconstruction of the omitted two text files. Exact selection-source narrative amendments are preserved. |
| Historical cross-domain ACORD / chunk-size sensitivity | All three ACORD reconstruction supplements plus main, independently downloaded fixed upstream ZIP and tokenizers. 57 query IDs are shared across the two scenarios. Pseudonyms are linkable; original contracts/queries are not supplied in these supplements. Replacement four/eight-candidate runs and dependency conflicts are disclosed. |
| Output recording / provenance | Response-provenance supplement: early product outputs were trimmed before saving. Whitespace lost at recording cannot be restored. |
| All displayed manuscript figures and rendered source | Figure supplement plus midpoint routing JSON and current controlled analyses; standalone LaTeX source includes rendered `main.tex`, class, figure PDFs and newer renderer. Historical figure alternatives remain for audit. |

## Later revision status

Context/decision recovery is a separate in-progress effort and is not silently
merged with this historical evidence. Complete later transfer/GLM volumes were
lost and are unavailable here. A plan, schema, checksum stub, or empty directory
does not reproduce an experiment. Publication of the old collection can establish
availability of that collection; it cannot establish complete EAB coverage for
a later manuscript that relies on the missing work.

## What verification establishes

Archive hashes, per-member inspection and CPU recomputation establish the
identity and selected arithmetic of saved materials. They do not establish a
complete clean installation elsewhere, new GPU generation, semantic validity,
population coverage, legal utility, novelty, or acceptance. Failures, adverse
results, post-hoc decisions, repeated queries and runtime modifications remain
part of the scientific record.
