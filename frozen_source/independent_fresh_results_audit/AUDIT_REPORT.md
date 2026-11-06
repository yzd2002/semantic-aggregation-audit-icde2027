# Independent fresh-results audit: DBLP–ACM and BoolQ

PASS for the six observed fresh cases. This is a data-integrity and numerical-recomputation certificate, not an acceptance recommendation. SNLI remains unavailable in this certificate; the planned primary family remains nine.

No producer module was imported. Scientific labels were recovered from complete durable raw-return payloads only after independent engine-ID, processed-input-token, grammar, frozen-request, source-feature, and source-gold checks. Canonical rows and producer analyses were then used as corroboration and numerical comparators.

All 269 frozen files match. All 14,310 unique requests (14,250 science and 60 controls) have complete independently checked return bindings and actual processor evidence. There are 3,075 blocking generate invocations. Each model contributes 4,750 scientific requests and 12,970 item labels. Every primary common12 and common14 contains all 128 selected targets per task; every serial subset contains all 32. Missing-one-of-128 COUNT abstention and negative-excess handling were independently checked.

Independent calculations match 60,586 numerical and 107,843 other producer fields (168,429 fields total; zero mismatches), covering all primary/historical/original-length-pass condition confusions, quality, COUNT, cancellation, scope contrasts, per-target vectors, serial32, reference transport, and dependent paired comparisons.

## Authoritative new primary results

Percentages below are percentage points of target-label disagreement. Intervals use 4,000 selected-target bootstrap draws, the first eight SHA256 bytes of `2026110613|task|model` as an unsigned big-endian seed, common12 selected-ordinal order, and linear quantiles 1/360 and 359/360 (confidence 179/180). A positive result requires the lower endpoint to be strictly greater than zero. Legacy 98.75% fields do not decide these cases.

| Task | Model | Replacement pp | Replay pp | Excess pp | 179/180 interval pp | Positive criterion |
|---|---|---:|---:|---:|---|---|
| dblp_acm | qwen | 1.953125 | 0.390625 | 1.5625 | [0, 5.46875] | No |
| dblp_acm | smol | 0.390625 | 0.390625 | 0 | [0, 0] | No |
| dblp_acm | glm | 10.15625 | 0 | 10.15625 | [3.90625, 17.96875] | Yes |
| boolq | qwen | 8.984375 | 0.390625 | 8.59375 | [3.125, 16.40625] | Yes |
| boolq | smol | 26.5625 | 3.90625 | 22.65625 | [12.58463542, 33.59375] | Yes |
| boolq | glm | 9.375 | 0 | 9.375 | [3.125, 17.1875] | Yes |

Four of the six observed cases satisfy the new criterion. Qwen–DBLP does not because its lower bound is zero. Smol–DBLP has zero excess and interval [0,0]. A failed criterion establishes neither absence of an effect nor equivalence.

## Results that must not be omitted

- Smol–DBLP full-first is not a globally constant-model case. Random repeats and near repeat 1 produce all Y (95 errors/128; balanced accuracy .5); near repeat 2 produces 127 Y and 1 N (96 errors/128; balanced accuracy 0.48484848484848486). Replacement and replay both equal 0.390625 pp, so excess is zero. Other output scopes differ and are retained.
- GLM–DBLP full-first has positive excess but worse candidate accuracy: random has 5 errors/128 and near has 14 errors/128 in each repeat. There are 2 wrong-to-right and 11 right-to-wrong changes per repeat, an error increase of 9/128 = 7.03125 pp. Context sensitivity does not imply a benefit.
- Smol–BoolQ full-first has 60/62 near errors and 68/70 random errors; all exceed the 46-error all-Y baseline. Balanced accuracy is recorded separately because the selected relation is imbalanced.
- Qwen–BoolQ full-first repeat 1 has 12 changed labels (6 N-to-Y, 6 Y-to-N), COUNT 89 to 89, 7 wrong-to-right and 5 right-to-wrong, and 33 to 31 candidate errors. Reference-free cancelled flips = 12; correctness-change cancellation = 10. These are different quantities.
- GLM–BoolQ full-first repeat 1 likewise has 12 changed labels (6+6), COUNT 101 to 101, 7 wrong-to-right and 5 right-to-wrong. Zero aggregate COUNT change does not mean unchanged row decisions or zero reference error.

## All contextual scopes and serial sensitivity

| Model | Task | Full-first excess pp | Target-only excess pp | Full-last excess pp | Serial32 full-first excess pp | Serial32 secondary 95% interval pp |
|---|---|---:|---:|---:|---:|---|
| qwen | dblp_acm | 1.5625 | 6.25 | 3.90625 | 0 | [0, 0] |
| qwen | boolq | 8.59375 | 7.8125 | 14.84375 | 3.125 | [0, 9.375] |
| smol | dblp_acm | 0 | 0.78125 | 13.28125 | 0 | [0, 0] |
| smol | boolq | 22.65625 | 23.4375 | 42.1875 | 28.125 | [12.5, 43.75] |
| glm | dblp_acm | 10.15625 | 5.46875 | 4.6875 | 12.5 | [3.125, 25] |
| glm | boolq | 9.375 | 4.6875 | 16.40625 | 3.125 | [0, 9.375] |

Serial is a secondary sensitivity check on fixed ordinals 0–31, after mixed execution and intervening historical mixed work. These intervals are not additional primary-family claims. Full-first does not require preceding generated background labels, but this does not identify an internal attention/reasoning mechanism or a sole scheduler cause.

## Actual durable-return token accounting

| Model | Science requests | Science input tokens | Science output tokens | Controls | Control input tokens | Control output tokens | Total input tokens | Total output tokens |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| qwen | 4750 | 2789708 | 63900 | 20 | 3830 | 267 | 2793538 | 64167 |
| smol | 4750 | 2942538 | 74350 | 20 | 4135 | 315 | 2946673 | 74665 |
| glm | 4750 | 2684470 | 61380 | 20 | 3610 | 264 | 2688080 | 61644 |

Total: 8,428,291 input tokens and 200,476 output tokens, including all controls and historical extras. Projected historical/subset views add zero new requests.

Each returned request has recorded actual greedy processing (temperature 0, top-p 1, top-k 0, n=1, cap 128), xgrammar with fallback disabled where guided, and the native EOS/stop sets. Native BF16 worker classes are Qwen3ForCausalLM, LlamaForCausalLM, and ChatGLMForCausalLM, with no recorded remote modeling modules. GLM’s ordinary unguided control is the untrimmed string `\nBANANA` and uses 5 output tokens; Qwen and Smol emit literal `BANANA` using 4. All 19 guided controls per model strictly satisfy their respective frozen grammar. No scientific format invalids were found.

All six failed startups are retained: three Unix-domain IPC path-length errors and three worker-identity callable serialization errors. Each has zero generate invocations, zero submitted generation requests and zero generation returns. Some startup work loaded/initialized the model; zero generate must not be described as zero startup GPU compute, time or energy.

## Evidence and limits

- `RESULTS_CERTIFICATE.json` binds this audit’s artifact hashes to `SOURCE_RAW_ANALYSIS_SHA256.json`, which includes source exports, frozen requests/code/protocol, complete durable raw receipts, processed bindings, canonical outputs, failures and producer analyses.
- `INDEPENDENT_RAW_BINDINGS.jsonl` provides the unique request/engine/generate-call/receipt/input/grammar/output mapping. `INDEPENDENT_PER_TARGET.jsonl` preserves every main/serial label tuple and independently recomputed replacement, replay, excess, signed error and wrong/right vector.
- `RECOMPUTED_<model>.json` retains all conditions, constant baselines, gold confusions, F1, balanced accuracy, fixed-denominator COUNT, both cancellation definitions and secondary views. `RAW_CHAIN_AND_COSTS_<model>.json` records full token bills and actual runtime evidence.
- Only new current-run durable outputs supplied scientific labels. The design was made after old results were known; this is a fresh forward replication, not a preregistration of past research.
- These finite-Monte-Carlo percentile intervals describe selected fixed targets. Nominal Bonferroni allocation is not exact population coverage or an exact FWER certificate. No SNLI outcome is invented. No new inference or independent tokenizer decoding was performed.
