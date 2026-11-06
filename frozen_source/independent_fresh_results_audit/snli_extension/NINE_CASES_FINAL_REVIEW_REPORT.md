# Independent final nine-case fresh-results review

PASS for evidence integrity and numerical recomputation. This does not recommend a publication decision. Five of nine fixed primary cases meet the prospective strictly-positive-lower-bound criterion; the other four are retained. The six-case baseline script, source data and original certificate remain unchanged.

All nine cases have common12=128 and common14=128; all serial relations have 32 targets. Every raw scientific return is strict-valid. The complete new suite has 20,970 scientific requests, 120 controls, 57,150 scientific item labels and 21,090 total returned requests, from 4,512 blocking generate invocations. The six retained prior startup failures had zero generation; their initialization compute is not described as zero.

| Task | Model | Replacement pp | Replay pp | Full-first excess pp | 179/180 interval pp | Positive criterion |
|---|---|---:|---:|---:|---|---|
| dblp_acm | qwen | 1.953125 | 0.390625 | 1.5625 | [0, 5.46875] | No |
| dblp_acm | smol | 0.390625 | 0.390625 | 0 | [0, 0] | No |
| dblp_acm | glm | 10.15625 | 0 | 10.15625 | [3.90625, 17.96875] | Yes |
| boolq | qwen | 8.984375 | 0.390625 | 8.59375 | [3.125, 16.40625] | Yes |
| boolq | smol | 26.5625 | 3.90625 | 22.65625 | [12.58463542, 33.59375] | Yes |
| boolq | glm | 9.375 | 0 | 9.375 | [3.125, 17.1875] | Yes |
| snli | qwen | 2.34375 | 0.78125 | 1.5625 | [0, 5.46875] | No |
| snli | smol | 18.359375 | 3.515625 | 14.84375 | [7.03125, 24.21875] | Yes |
| snli | glm | 3.125 | 0 | 3.125 | [0, 7.8125] | No |

The primary family is fixed at nine, with confidence 179/180, 4,000 draws, linear quantiles 1/360 and 359/360, and per-case SHA256 seed from `2026110613|task|model`. These are finite-Monte-Carlo intervals for the selected target sets, not exact population-coverage or exact-FWER certificates. Original 98.75% fields remain legacy only. Failed primary criteria show neither equivalence nor absence of a population effect.

## SNLI results and adverse evidence

SNLI original labels independently rejoin the pinned Parquet rows: 47 entailment, 39 neutral and 42 contradiction among 128 targets. Y maps entailment to 1; N combines the other two classes. The serial32 subset has 11 positive reference labels. This is binary entailment quality, not three-class SNLI accuracy.

- Qwen–SNLI: full-first excess 1.5625 pp, interval [0,5.46875] pp, not positive. Near has 12/12 errors and random 13/11; mean gold-error change is zero. The four-reference qualification has 120 passing targets, 7 reference errors and zero near any/stable flips (0/120). This null transport result is retained.

- Smol–SNLI: full-first excess 14.84375 pp, interval [7.03125,24.21875] pp, positive. Near has 49/48 errors and random 48/50; all four exceed the all-N baseline of 47 errors, although balanced accuracies are about .68–.69 versus .5 for constants. Repeat 2 changes 22 labels (11 each way) with COUNT 91 to 91; 12 wrong-to-right and 10 right-to-wrong changes give 50 to 48 gold errors. Reference-free cancellation is 22, correctness-change cancellation 20. Its reference qualification passes 90 targets, of which 20 references are wrong; near shows 10 any flips and 7 stable flips, with 2 wrong-to-right versus 15 right-to-wrong events across two near calls.

- GLM–SNLI: full-first excess 3.125 pp, interval [0,7.8125] pp, not positive. Both repeats change the same four labels; all four move from correct to wrong (0 wrong-to-right, 4 right-to-wrong). Errors increase 18 to 22; COUNT increases 63 to 65 even though four rows change, with two reference-free cancelled flips. The four-reference qualification passes 123 targets, includes 16 reference errors and yields 2 stable/any near flips, all four near events worsening gold correctness.

## SNLI secondary scopes and serial32

| Model | Full-first excess pp | Target-only excess pp | Full-last excess pp | Serial32 full-first pp | Secondary 95% interval pp |
|---|---:|---:|---:|---:|---|
| qwen | 1.5625 | 4.6875 | 3.125 | 3.125 | [0, 9.375] |
| smol | 14.84375 | 9.375 | 34.375 | 28.125 | [12.5, 43.75] |
| glm | 3.125 | 2.34375 | 3.90625 | 3.125 | [0, 9.375] |

Serial is secondary, uses the same fixed 32 targets per task, and follows mixed on the loaded model. Persistence does not isolate a scheduler or an internal reasoning/attention mechanism. Target-only/full-last do not replace failed primary contrasts. The six earlier fresh cases retain every null, all-Y-near-constant behavior, baseline failure, label-vs-COUNT cancellation and GLM–DBLP correctness worsening in the separately frozen `AUDIT_REPORT.md`.

## SNLI complete durable-return token bill

| Model | Science requests | Science input | Science output | Controls | Control input | Control output | Total input | Total output |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| qwen | 2240 | 558262 | 30425 | 20 | 3221 | 272 | 561483 | 30697 |
| smol | 2240 | 597052 | 34840 | 20 | 3459 | 315 | 600511 | 35155 |
| glm | 2240 | 544816 | 28800 | 20 | 3101 | 264 | 547917 | 29064 |

SNLI total: 1,709,911 input and 94,916 output tokens. Grand suite: 10,138,202 input and 295,392 output tokens, including all controls and historical extras; dependent derived views add no calls.

Every SNLI request was checked against actual native BF16 worker class, actual engine-processed temperature=0/top-p=1/top-k=0/n=1/cap128, exact xgrammar with disabled fallback where guided, native EOS/stop sets, complete frozen input token arrays and output tokens/text. Qwen/Smol ordinary controls emit literal BANANA; GLM emits untrimmed `\nBANANA` (5 output tokens). All 19 guided controls per model are strict-valid. All three producer REPORT_INDEX files and logs indicate complete real model evidence.

## Independent evidence

No producer statistical or return-validation module was imported. The six-case numerical engine is hash-frozen and reused as our own independent helper; the separate SNLI extension uses its frozen legacy secondary seed 2026110611 and the same authoritative SHA-derived primary seeds. Native durable snapshots are independently bound by processed engine identity, exact complete input arrays and grammar before parsing; canonical outputs and producer statistics are comparators.

The new extension matches 10,343 numerical and 18,706 other producer fields (29,049 total), with zero mismatches. Together with the unchanged 168,429-field six-case audit, this verifies 197,478 comparison fields. All 163 SNLI frozen files, selected-source fields/gold, raw snapshots, processed evidence and analysis hashes are bound into the separate final certificate.

- `SNLI_AND_NINE_CASES_CERTIFICATE.json` links the immutable original six-case certificate, own independent helper, extension code, SNLI source/raw/analysis manifest and new artifacts.
- `NINE_CASE_PRIMARY_INDEPENDENT.json` reports all nine real primary cases, including zeros and failed criteria.
- `NINE_CASE_MANUSCRIPT_NUMBERS.json` provides the independently derived numbers, every condition’s confusion/quality and fixed COUNT, all serial results and the complete cost bill.
- `SNLI_INDEPENDENT_RAW_BINDINGS.jsonl` and `SNLI_INDEPENDENT_PER_TARGET.jsonl` trace every new request and per-target four-call vector.

The audit checks the preserved evidence; it does not independently rerun models or decode token arrays. The design follows earlier known outcomes, so it is not a retrospective preregistration. Public sources, selected relations, unknown pretraining overlap and the recorded same-machine runtime bound interpretation.
