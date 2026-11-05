# IntentSat-2S reproducibility artifact

See `ARTIFACT_CONTENTS_AND_RIGHTS.md` for included materials, upstream source notices, and the limits of the recorded verification.

The artifact contains the prompts, selection logic, strict parsers, model outputs and exact computed summaries for a 160-query exploratory pilot, a disjoint 500-query derived-format validation, a further disjoint 250-query study of the original labeled candidate pools, a follow-up second-model run on the same 250 pools, and post-hoc LOTUS and supervised lexical baselines. The `PROTOCOL*.md` files describe the fixed designs and their limitations. `run_test_160/`, `run_validation_500/`, `run_full_pools_250/`, `run_full_pools_smollm_250/`, and `run_lotus_baseline/` contain the sampled inputs or references to them, raw model responses, selection logs, summaries, and paired analyses. `run_supervised_baseline/` and `run_supervised_baseline_1000/` preserve the initial nonconverged fit and the follow-up fit, respectively, including training-selection manifests and per-candidate predictions. They are derived from the dataset below. No model weights or full dataset files are bundled.

## Data provenance

Data: https://huggingface.co/datasets/nixiesearch/amazon-esci at commit `c092c898ccc036577f25a998fd429b84facfbd86`; dataset card declares Apache-2.0. Test file at that commit:

https://huggingface.co/datasets/nixiesearch/amazon-esci/resolve/c092c898ccc036577f25a998fd429b84facfbd86/data/test/test.jsonl.zst

Expected SHA-256: `ede7f2ed1e45a1563bdea8d8e96184c4dd5eb585af0856b2268bc82e37124087`.

The exact conversion script is at https://huggingface.co/datasets/nixiesearch/amazon-esci/raw/c092c898ccc036577f25a998fd429b84facfbd86/convert.py . It identifies the source as the US small version of the Shopping Queries dataset and encodes E/S/C/I as 1/0.1/0.01/0. It emits one record per E product with all non-E products. It drops IDs and the other E products in the same query. We sample query **strings**, not recoverable query IDs; see the protocols before interpreting the scores. Original dataset: https://github.com/amazon-science/esci-data and https://arxiv.org/abs/2206.06588 . Check underlying content licensing before redistributing a public artifact.

Model: https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507 ; local safetensors SHA-256 values are in `MODEL_SHA256.txt`. `MODEL_QWEN_ASSETS_SHA256.txt` is the unchanged original frozen record of tokenizer and configuration hashes and the observed official revision. The four small Qwen configuration files were matched byte-for-byte and are preserved in `qwen_metadata/`. A later official-metadata comparison also matched the local merges and vocabulary files at that revision; see `qwen3_official_asset_recheck/` for the captured metadata, verification code and limits. The follow-up compared recorded weight hashes with official LFS metadata but did not re-hash the local weight bytes. Verify hashes and use the preserved configuration when reproducing. The primary GPU environment recorded Python 3.12.11, PyTorch 2.8.0+cu128, transformers 4.57.1, vLLM 0.10.2, zstandard 0.25.0, pandas 2.3.3, numpy 2.2.6, pyarrow 22.0.0; one NVIDIA RTX 5090 32 GB, bfloat16, 4096-token maximum context, eager mode, offline batched generation, zero temperature. The LOTUS client used NumPy 1.26.4 separately. A later metadata audit found dependency conflicts in the GPU and LOTUS client environments; see `GPU_ENVIRONMENT_POSTRUN_SNAPSHOT.json` and the two `GPU_ENVIRONMENT_PIP_CHECK_*.txt` files. The audit is not a complete pre-run lock and does not by itself identify failed calls. Exact versions, scheduler, GPU architecture and model files can change results and timings.

### Original full candidate pools

The original Amazon Science tables at revision `7916cdf6ab75a462e77f20ab40428a10923998d5` are available at:

- https://github.com/amazon-science/esci-data/raw/7916cdf6ab75a462e77f20ab40428a10923998d5/shopping_queries_dataset/shopping_queries_dataset_examples.parquet
- https://github.com/amazon-science/esci-data/raw/7916cdf6ab75a462e77f20ab40428a10923998d5/shopping_queries_dataset/shopping_queries_dataset_products.parquet

The expected SHA-256 values are listed in `OFFICIAL_DATA_SHA256.txt`; they match the Git LFS pointers. The full tables are not included in this artifact. The sampled 250 original query-ID candidate lists are preserved in `run_full_pools_250/inputs.jsonl` with the original example IDs, product IDs and labels. The input joins and pool construction are implemented in `run_full_pools.py`. See `ESCI_LICENSE.txt`, `ESCI_NOTICE.txt`, `NOTICE`, and the publisher's license terms before redistribution. These labeled lists are not necessarily exhaustive live-search results.

### Second-model sensitivity

The follow-up model is https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct at revision `31b70e2e869a7173562077fd711b654946d38674`; the expected weight hash is in `MODEL_SENSITIVITY_SHA256.txt`. This model is a different family and size, not a controlled architecture-only comparison. The follow-up is conducted on the same 250 original pools after observing the first model's results; it is not another dataset. `analyze_crossmodel.py` checks that the saved inputs match and computes metrics on the intersection of all completed plans when possible.

### Independent WANDS source and sampled pools

Wayfair WANDS is a separate human-annotated product-search dataset: https://github.com/wayfair/WANDS/tree/3b74dcf4ba29ab8ff3e6a50b5b09fc627cb882b5 . The frozen source files and SHA-256 values are in `WANDS_DATA_SHA256.txt`; the publisher's license and label definitions are recorded in `WANDS_LICENSE.txt` and `WANDS_SOURCE_NOTES.md`. Download the three TSV files `query.csv`, `product.csv`, and `label.csv` from its `dataset/` directory into a directory of your choice. The complete source data are not bundled. `PROTOCOL_wands.md` specifies deduplication, the exact-only relevance proxy, the 160-query sample and the different text fields available to the model versus annotators. This source is independent of ESCI, but the decision to add it followed the earlier results, and it is not evidence of real user purchases.

The following commands run the same selected WANDS pools on both local models, then reconstruct the source selection, reparse saved predictions, check actual prompt lengths, and analyze the common completed pools. Use new output directories and run the GPU jobs sequentially:

```bash
python3 preflight_wands.py --wands-dir /path/to/wands --model /path/to/Qwen3-4B-Instruct-2507
VLLM_USE_V1=0 python3 run_wands.py --wands-dir /path/to/wands --model /path/to/Qwen3-4B-Instruct-2507 --out wands_qwen_new
python3 verify_wands.py --wands-dir /path/to/wands --run wands_qwen_new
python3 analyze_wands.py --run wands_qwen_new
python3 audit_wands_context.py --run wands_qwen_new --model /path/to/Qwen3-4B-Instruct-2507
python3 analyze_partial_bounds.py --run wands_qwen_new
python3 preflight_wands.py --wands-dir /path/to/wands --model /path/to/SmolLM2-1.7B-Instruct
VLLM_USE_V1=0 python3 run_wands.py --wands-dir /path/to/wands --model /path/to/SmolLM2-1.7B-Instruct --out wands_smollm_new
python3 verify_wands.py --wands-dir /path/to/wands --run wands_smollm_new
python3 analyze_wands.py --run wands_smollm_new
python3 audit_wands_context.py --run wands_smollm_new --model /path/to/SmolLM2-1.7B-Instruct
python3 analyze_partial_bounds.py --run wands_smollm_new
python3 analyze_crossmodel.py --qwen-run wands_qwen_new --smollm-run wands_smollm_new
```

`analyze_partial_bounds.py --run run_full_pools_250` also reconstructs completion intervals for failed ESCI pools; those intervals constrain completions of the accepted **model labels**, not the true relevance ratio unless accepted labels are correct. `ratio_bounds.py` also computes sharp bounds under an externally supplied accepted-label error budget. The code does not estimate that budget; `test_ratio_bounds.py` exhaustively checks the endpoints for all binary pools of size at most six.

### Post-hoc compact-prompt and parser checks

`PROTOCOL_compact_prompt_ablation.md` removes the original four-item example, whose trailing period conflicts with the strict format, and reuses the saved SmolLM2 specifications. `PROTOCOL_compact_parse_audit.md` then reanalyzes the original and example-removed raw outputs with a restricted delimiter-tolerant parser; it never adds a missing label. Both are post-hoc diagnostics and do not replace the original strict-plan results.

```bash
VLLM_USE_V1=0 python3 run_compact_prompt_ablation.py --model /path/to/SmolLM2-1.7B-Instruct --esci-run /path/to/smollm_esci_run --wands-run /path/to/smollm_wands_run --out compact_ablation_new
python3 verify_compact_prompt_ablation.py --run compact_ablation_new --esci-run /path/to/smollm_esci_run --wands-run /path/to/smollm_wands_run
python3 test_tolerant_compact.py
python3 analyze_compact_parse_audit.py --ablation compact_ablation_new --esci-run /path/to/smollm_esci_run --wands-run /path/to/smollm_wands_run --out compact_parse_new
python3 verify_compact_parse_audit.py --run compact_parse_new --ablation compact_ablation_new --esci-run /path/to/smollm_esci_run --wands-run /path/to/smollm_wands_run
```

### Post-hoc guided decoding

`PROTOCOL_guided_compact.md` specifies the example-removed compact prompts and per-chunk exact-length Y/N regex used for a post-hoc comparison. A first attempt to use vLLM 0.10.2's V0 engine produced a byte-for-byte repeat of all Qwen unconstrained outputs, including invalid chunks, despite the requested regex. Its outputs, logs, original code snapshot and protocol amendment are preserved; they are not treated as active constrained decoding. The corrected matched comparison uses the V1 engine in both conditions and first runs an incompatible-answer negative control to verify that the format constraint is actually active. The constraint enforces an output format, not correctness of the decisions.

The commands below use the recorded input/specification runs and explicitly regenerate the matched V1 baselines for both models. Run GPU jobs sequentially and use fresh output directories.

```bash
python3 verify_v0_guidance_attempts.py
VLLM_USE_V1=1 python3 run_compact_prompt_ablation.py --model /path/to/Qwen3-4B-Instruct-2507 --esci-run run_full_pools_250 --wands-run run_wands_160 --out compact_qwen_v1_new
python3 verify_compact_prompt_ablation.py --run compact_qwen_v1_new --esci-run run_full_pools_250 --wands-run run_wands_160
VLLM_USE_V1=1 python3 run_guided_compact.py --model /path/to/Qwen3-4B-Instruct-2507 --esci-run run_full_pools_250 --wands-run run_wands_160 --out guided_qwen_new
python3 analyze_guided_compact.py --guided-run guided_qwen_new --unconstrained-run compact_qwen_v1_new --esci-run run_full_pools_250 --wands-run run_wands_160
python3 verify_guided_compact.py --guided-run guided_qwen_new --unconstrained-run compact_qwen_v1_new --esci-run run_full_pools_250 --wands-run run_wands_160
VLLM_USE_V1=1 python3 run_compact_prompt_ablation.py --model /path/to/SmolLM2-1.7B-Instruct --esci-run run_full_pools_smollm_250 --wands-run run_wands_smollm_160 --out compact_smollm_v1_new
VLLM_USE_V1=1 python3 run_guided_compact.py --model /path/to/SmolLM2-1.7B-Instruct --esci-run run_full_pools_smollm_250 --wands-run run_wands_smollm_160 --out guided_smollm_new
python3 analyze_guided_compact.py --guided-run guided_smollm_new --unconstrained-run compact_smollm_v1_new --esci-run run_full_pools_smollm_250 --wands-run run_wands_smollm_160
python3 verify_guided_compact.py --guided-run guided_smollm_new --unconstrained-run compact_smollm_v1_new --esci-run run_full_pools_smollm_250 --wands-run run_wands_smollm_160
```

### Post-hoc within-chunk order audit

`PROTOCOL_order_sensitivity.md` fixes one within-chunk reversal after the earlier results were observed. Each chunk contains the same products, with local identifiers reassigned to the reversed positions; generated labels are mapped back to original candidate indices. The saved `run_reverse_order_qwen_v1/` and `run_reverse_order_smollm_v1/` include the negative controls, prompt hashes, exact position mappings, raw responses, paired analysis and query-cluster bootstrap intervals. The agreement-only diagnostic abstains on differing decisions; its `k=0` intervals condition on accepted labels being correct and are not calibrated coverage guarantees. The comparison also changes the product-to-local-ID association and prompt context, so it cannot isolate a pure position effect or characterize every permutation.

With the recorded baseline and input directories, reproduce each model on the GPU sequentially:

```bash
VLLM_USE_V1=1 python3 run_reverse_order.py --model /path/to/Qwen3-4B-Instruct-2507 --esci-run run_full_pools_250 --wands-run run_wands_160 --baseline run_guided_compact_qwen_v1 --out reverse_qwen_new
python3 analyze_reverse_order.py --run reverse_qwen_new --baseline run_guided_compact_qwen_v1 --esci-run run_full_pools_250 --wands-run run_wands_160
python3 verify_reverse_order.py --run reverse_qwen_new --baseline run_guided_compact_qwen_v1 --esci-run run_full_pools_250 --wands-run run_wands_160
VLLM_USE_V1=1 python3 run_reverse_order.py --model /path/to/SmolLM2-1.7B-Instruct --esci-run run_full_pools_smollm_250 --wands-run run_wands_smollm_160 --baseline run_guided_compact_smollm_v1 --out reverse_smollm_new
python3 analyze_reverse_order.py --run reverse_smollm_new --baseline run_guided_compact_smollm_v1 --esci-run run_full_pools_smollm_250 --wands-run run_wands_smollm_160
python3 verify_reverse_order.py --run reverse_smollm_new --baseline run_guided_compact_smollm_v1 --esci-run run_full_pools_smollm_250 --wands-run run_wands_smollm_160
```

### Stable identifiers and identical-prompt replay

`PROTOCOL_order_controls.md` was fixed after the original reversal results. It adds an identical-prompt replay and a reversal that keeps each product's local ID. The saved outputs in `run_order_controls_qwen_v1/` and `run_order_controls_smollm_v1/` support four common-pool contrasts: identical replay, reversal with stable IDs, ID reassignment at the same reversed order, and the original reversal with reassigned IDs. Output labels are mapped back to the same candidates. `verify_order_controls.py` reconstructs all three user-prompt hashes, checks identifier preservation, candidate alignment, parsing, quality, completeness, and paired bootstrap intervals. These are post-hoc input-level controls in fixed run order; a single replay cannot characterize all inference variability. For the paper's prior-reversal table column, the renderer reuses the earlier figure's bootstrap interval when the common query IDs match, avoiding differences caused only by the bootstrap seed.

```bash
VLLM_USE_V1=1 python3 run_order_controls.py --model /path/to/Qwen3-4B-Instruct-2507 --esci-run run_full_pools_250 --wands-run run_wands_160 --baseline run_guided_compact_qwen_v1 --renumbered run_reverse_order_qwen_v1 --out order_controls_qwen_new
python3 analyze_order_controls.py --run order_controls_qwen_new --baseline run_guided_compact_qwen_v1 --renumbered run_reverse_order_qwen_v1 --esci-run run_full_pools_250 --wands-run run_wands_160
python3 verify_order_controls.py --run order_controls_qwen_new --baseline run_guided_compact_qwen_v1 --renumbered run_reverse_order_qwen_v1 --esci-run run_full_pools_250 --wands-run run_wands_160
VLLM_USE_V1=1 python3 run_order_controls.py --model /path/to/SmolLM2-1.7B-Instruct --esci-run run_full_pools_smollm_250 --wands-run run_wands_smollm_160 --baseline run_guided_compact_smollm_v1 --renumbered run_reverse_order_smollm_v1 --out order_controls_smollm_new
python3 analyze_order_controls.py --run order_controls_smollm_new --baseline run_guided_compact_smollm_v1 --renumbered run_reverse_order_smollm_v1 --esci-run run_full_pools_smollm_250 --wands-run run_wands_smollm_160
python3 verify_order_controls.py --run order_controls_smollm_new --baseline run_guided_compact_smollm_v1 --renumbered run_reverse_order_smollm_v1 --esci-run run_full_pools_smollm_250 --wands-run run_wands_smollm_160
python3 test_stability_bounds.py
```

The elementary disagreement bound implemented by `stability_bounds.py` assumes one fixed binary reference for both outputs. For n candidates and D disagreements, the sum of their candidate-error rates is at least D/n, and the larger rate is at least ceil(D/2)/n. Both bounds are attainable. Earlier frozen analysis files retain the weaker D/(2n) maximum-error floor; it remains valid but does not exploit integer counts. Neither floor is a confidence interval or a lower bound on aggregate-ratio error.

### Disjoint WANDS holdout

After the earlier order results, `PROTOCOL_wands_holdout.md` fixed the selection and primary test before sampling and inference. `wands_holdout_selection/` contains 160 new queries and 5,120 judged products; query strings and product IDs sampled in the earlier WANDS study were excluded. This is a separate sample from the same public source, not a separate dataset or proof of pretraining exclusion. The selection uses 308 eligible query IDs after exclusions, and product reuse within the holdout remains possible. `WANDS_HOLDOUT_FROZEN_SHA256.txt` records the frozen protocol, selection and principal code hashes. `WANDS_HOLDOUT_ENGINE_NOTE.md` records that holdout specifications use V1, whereas earlier WANDS specifications used V0; within the holdout, all three order conditions reuse the same fresh specification or fallback.

The primary Qwen check compares identical-prompt replay with stable-ID reversal and asks whether the lower endpoint of its 95% query-cluster interval exceeds the pre-specified 10% practical threshold. SmolLM2 is secondary. Before inspecting holdout outputs, `PROTOCOL_wands_holdout_two_way.md` added a secondary query–product resampling sensitivity because sampled products can recur across queries. Neither bootstrap interval is an exact population guarantee. The two pre-outcome manifests, raw outputs, context audits, selection verifier and independent analysis verifiers are retained.

```bash
python3 select_wands_holdout.py --wands-dir /path/to/wands --out wands_holdout_selection_new
python3 verify_wands_holdout_selection.py --wands-dir /path/to/wands --selection wands_holdout_selection_new
VLLM_USE_V1=1 python3 run_wands_holdout.py --model /path/to/Qwen3-4B-Instruct-2507 --selection wands_holdout_selection_new --out wands_holdout_qwen_new
python3 analyze_wands_holdout.py --run wands_holdout_qwen_new --selection wands_holdout_selection_new
python3 verify_wands_holdout.py --run wands_holdout_qwen_new --selection wands_holdout_selection_new
python3 analyze_wands_holdout_two_way.py --run wands_holdout_qwen_new --selection wands_holdout_selection_new
python3 verify_wands_holdout_two_way.py --run wands_holdout_qwen_new --selection wands_holdout_selection_new
python3 audit_wands_holdout_context.py --run wands_holdout_qwen_new --selection wands_holdout_selection_new --model /path/to/Qwen3-4B-Instruct-2507
VLLM_USE_V1=1 python3 run_wands_holdout.py --model /path/to/SmolLM2-1.7B-Instruct --selection wands_holdout_selection_new --out wands_holdout_smollm_new
python3 analyze_wands_holdout.py --run wands_holdout_smollm_new --selection wands_holdout_selection_new
python3 verify_wands_holdout.py --run wands_holdout_smollm_new --selection wands_holdout_selection_new
python3 analyze_wands_holdout_two_way.py --run wands_holdout_smollm_new --selection wands_holdout_selection_new
python3 verify_wands_holdout_two_way.py --run wands_holdout_smollm_new --selection wands_holdout_selection_new
python3 audit_wands_holdout_context.py --run wands_holdout_smollm_new --selection wands_holdout_selection_new --model /path/to/SmolLM2-1.7B-Instruct
```

### Post-hoc supervised baseline

The trained lexical comparator uses 50,000 labeled examples from the original train split and excludes the case-folded, trimmed evaluation query strings. Train and test are not product-disjoint: 357 distinct product IDs occur in both the sampled training set and the 250-pool test set. It uses supervised labels, unlike the evaluated zero-shot LLM plans. Training was CPU-only with scikit-learn 1.7.2, SciPy 1.16.2, joblib 1.5.2 and threadpoolctl 3.6.0. The first fit hit the 200-iteration limit; the additional, documented run used an upper limit of 1,000 and converged after 275. No test-set tuning of features or threshold was performed, and both outputs are preserved. `verify_supervised_baseline.py` reconstructs the training sample and checks query isolation, predictions and reported metrics.

### Post-hoc LOTUS semantic filter

`run_lotus_baseline/` contains a direct run of the released LOTUS `sem_filter` operator over the same 250 original pools using the local Qwen model. It is one filter configuration, not a benchmark of LOTUS cascades or its optimizer. LOTUS 1.2.4, LiteLLM 1.106.6, NumPy 1.26.4 and pandas 2.3.3 were used in a separate client environment; the vLLM 0.10.2 model server ran in the earlier GPU environment. The LOTUS 1.2.4 PyPI wheel `lotus_ai-1.2.4-py3-none-any.whl` has SHA-256 `4803bcba7a30d50095b0c6fc5036942ebd1cfceb42a3f907454892c11171fa89` and is distributed under Apache-2.0. See https://pypi.org/project/lotus-ai/1.2.4/ and https://lotus-ai.readthedocs.io/en/latest/sem_filter.html . `verify_lotus_baseline.py` reproduces its parser's postprocessing, counts defaults and ambiguous responses, and checks the saved metrics. The online API time is not comparable with the paper's offline inference time.

To repeat it, install LOTUS in an isolated Python environment (its NumPy constraint differs from the GPU environment), then launch the original Qwen model in a separate vLLM environment on loopback. An example server command is:

```bash
VLLM_USE_V1=0 python3 -m vllm.entrypoints.openai.api_server --model /path/to/Qwen3-4B-Instruct-2507 --served-model-name qwen-intentsat --dtype bfloat16 --max-model-len 4096 --gpu-memory-utilization 0.70 --max-num-seqs 16 --enforce-eager --host 127.0.0.1 --port 18483
```

After the local server is ready, from this directory in the LOTUS client environment run:

```bash
LITELLM_LOCAL_MODEL_COST_MAP=True LITELLM_TELEMETRY=False NO_PROXY='*' no_proxy='*' python3 run_lotus_baseline.py --out lotus_new
python3 verify_lotus_baseline.py --run lotus_new
```

The client rejects non-loopback endpoints and uses an inert local authentication placeholder. Stop only the model-server process launched for this test. The saved server log includes two initial synthetic smoke-test calls in addition to the 5,160 evaluation calls. Its final `KeyboardInterrupt` and process-group cleanup warning were emitted when the test's own server was deliberately stopped after the completed evaluation; the saved client output and independently verified results precede that shutdown.

## Reproduce

Run from this directory after installing compatible dependencies and obtaining the verified data and model files. Choose **new output directory names**: the programs refuse to overwrite prior runs.

```bash
python3 test_contract.py
python3 test_ratio_bounds.py
VLLM_USE_V1=0 python3 run_experiment.py --data /path/to/test.jsonl.zst --model /path/to/Qwen3-4B-Instruct-2507 --out pilot_new
VLLM_USE_V1=0 python3 run_validation.py --data /path/to/test.jsonl.zst --model /path/to/Qwen3-4B-Instruct-2507 --out validation_new
python3 analyze.py --run pilot_new
python3 analyze_validation.py --run validation_new
VLLM_USE_V1=0 python3 run_full_pools.py --examples /path/to/shopping_queries_dataset_examples.parquet --products /path/to/shopping_queries_dataset_products.parquet --model /path/to/Qwen3-4B-Instruct-2507 --out full_pools_new
python3 analyze_full_pools.py --run full_pools_new
VLLM_USE_V1=0 python3 run_full_pools.py --examples /path/to/shopping_queries_dataset_examples.parquet --products /path/to/shopping_queries_dataset_products.parquet --model /path/to/SmolLM2-1.7B-Instruct --out full_pools_smollm_new
python3 analyze_smollm.py --run full_pools_smollm_new
python3 analyze_crossmodel.py --qwen-run full_pools_new --smollm-run full_pools_smollm_new
python3 analyze_json_ablation.py --run full_pools_new
python3 analyze_failures.py --run full_pools_new
python3 analyze_json_ablation.py --run full_pools_smollm_new
python3 analyze_failures.py --run full_pools_smollm_new
python3 run_supervised_baseline.py --examples /path/to/shopping_queries_dataset_examples.parquet --products /path/to/shopping_queries_dataset_products.parquet --out supervised_200_new --max-iter 200
python3 run_supervised_baseline.py --examples /path/to/shopping_queries_dataset_examples.parquet --products /path/to/shopping_queries_dataset_products.parquet --out supervised_1000_new --max-iter 1000
python3 verify_supervised_baseline.py --run supervised_1000_new --examples /path/to/shopping_queries_dataset_examples.parquet
python3 verify_results.py
python3 verify_selections.py --derived /path/to/test.jsonl.zst --examples /path/to/shopping_queries_dataset_examples.parquet --products /path/to/shopping_queries_dataset_products.parquet
```

The validation program uses the already recorded `run_test_160/inputs.jsonl` to exclude its exact query strings; the original-pool sampler likewise excludes the recorded pilot and validation inputs. Those recorded samples must be retained when reproducing the same selection. The analysis scripts default to the recorded directories if `--run` is omitted; the commands above analyze the new directories. `verify_results.py` checks the recorded runs rather than the new directories. Captured output tokens, wall times and execution order are in each `summary.json`; output completion errors and retries are in `outputs.jsonl`. Latency is offline aggregate chat-template rendering plus model-generation time, **not** interactive or end-to-end service latency; model loading and most other Python-side assembly and parsing are excluded. The old manuscript's 0.3997 figure is not produced by these programs and must not be presented as a result of the new validated operator.

The optional manuscript-rendering, submission-field and author-bundle scripts require the surrounding complete preparation package, including its `01_manuscript/` and `03_submission/` directories. They are not required to rerun the model evaluations in this standalone artifact. The author-bundle script checks ORCID format and checksum, not ownership, and creates a new directory without overwriting the review manuscript; see `03_submission/03_作者信息最后填写.md` in the complete package.

## Exploratory keyed-output order audit

`PROTOCOL_keyed_order_audit.md` records the motivation, pre-inference design and the limitations of a post-hoc audit on the already exposed WANDS holdout. `KEYED_ORDER_AUDIT_FROZEN_SHA256.txt` covers the fixed protocol, runner, analysis, selected inputs and reused prior specifications/outputs. The compact protocol's original listed-order parsing and its primary result remain unchanged. The keyed conditions reuse each model's saved specifications and issue three guided JSON runs with identical local product IDs, in original order, identical replay, and reversed presentation order. Keyed JSON fixes the parser's association; it cannot prove the model used the right semantics. The compact and keyed conditions change the prompt, grammar and token cap, and were run in different inference instances.

The recorded runs can be checked without model inference. The optional `--model` argument loads a local tokenizer and independently checks prompt length against the context cap:

```bash
sha256sum -c KEYED_ORDER_AUDIT_FROZEN_SHA256.txt
python3 test_keyed_order_audit.py
python3 verify_keyed_order_audit.py --run run_keyed_order_qwen_v1 --prior run_wands_holdout_qwen_v1 --selection wands_holdout_selection --model /path/to/Qwen3-4B-Instruct-2507
python3 verify_keyed_order_audit.py --run run_keyed_order_smollm_v1 --prior run_wands_holdout_smollm_v1 --selection wands_holdout_selection --model /path/to/SmolLM2-1.7B-Instruct
```

To repeat a run, use a **new**, nonexistent output directory; model weights and the GPU environment are required:

```bash
VLLM_USE_V1=1 python3 run_keyed_order_audit.py --model /path/to/Qwen3-4B-Instruct-2507 --selection wands_holdout_selection --prior run_wands_holdout_qwen_v1 --out keyed_qwen_new
python3 analyze_keyed_order_audit.py --run keyed_qwen_new --prior run_wands_holdout_qwen_v1 --selection wands_holdout_selection
python3 verify_keyed_order_audit.py --run keyed_qwen_new --prior run_wands_holdout_qwen_v1 --selection wands_holdout_selection --model /path/to/Qwen3-4B-Instruct-2507
VLLM_USE_V1=1 python3 run_keyed_order_audit.py --model /path/to/SmolLM2-1.7B-Instruct --selection wands_holdout_selection --prior run_wands_holdout_smollm_v1 --out keyed_smollm_new
python3 analyze_keyed_order_audit.py --run keyed_smollm_new --prior run_wands_holdout_smollm_v1 --selection wands_holdout_selection
python3 verify_keyed_order_audit.py --run keyed_smollm_new --prior run_wands_holdout_smollm_v1 --selection wands_holdout_selection --model /path/to/SmolLM2-1.7B-Instruct
```

## Disagreement-fallback validation on a further WANDS sample

`PROTOCOL_order_fallback_validation.md` was fixed after the previous experiments but before selection and inference on this sample. The sampled queries and products are disjoint from the two earlier WANDS samples, though all are from the same public source. The primary outcome is the pre-specified pointwise-fallback minus canonical-keyed query-macro ratio MAE. The actual pointwise baseline ran on every candidate; the selective fallback's call and token totals are computed from recorded traces, not measured online latency. `PROTOCOL_order_fallback_two_way.md` fixes a separate crossed query–product sensitivity after selection and inference began but before inspecting model outputs.

To verify the recorded data and results without rerunning inference:

```bash
sha256sum -c ORDER_FALLBACK_PRESELECTION_SHA256.txt
sha256sum -c ORDER_FALLBACK_FROZEN_SHA256.txt
sha256sum -c ORDER_FALLBACK_TWO_WAY_FROZEN_SHA256.txt
python3 test_order_fallback.py
python3 test_order_fallback_analysis.py
python3 verify_order_fallback_selection.py --wands-dir /path/to/wands --selection wands_fallback_validation_selection
python3 verify_order_fallback_validation.py --run run_wands_fallback_validation_qwen_v1 --selection wands_fallback_validation_selection --model /path/to/Qwen3-4B-Instruct-2507
python3 verify_order_fallback_two_way.py --run run_wands_fallback_validation_qwen_v1 --selection wands_fallback_validation_selection
```

The optional `--model` argument loads the local tokenizer to check exact input-token counts and prompt lengths; omit it to verify saved outputs, associations, quality and intervals without the tokenizer. To reproduce model execution use a new, nonexistent output directory:

```bash
VLLM_USE_V1=1 python3 run_order_fallback_validation.py --model /path/to/Qwen3-4B-Instruct-2507 --selection wands_fallback_validation_selection --out fallback_qwen_new
python3 analyze_order_fallback_validation.py --run fallback_qwen_new --selection wands_fallback_validation_selection
python3 analyze_order_fallback_two_way.py --run fallback_qwen_new --selection wands_fallback_validation_selection
python3 verify_order_fallback_validation.py --run fallback_qwen_new --selection wands_fallback_validation_selection --model /path/to/Qwen3-4B-Instruct-2507
python3 verify_order_fallback_two_way.py --run fallback_qwen_new --selection wands_fallback_validation_selection
```

Rerunning model calls can change outputs despite temperature zero. The symmetry guarantee concerns the same cached, ID-aligned pair and fallback outputs; it does not certify semantic correctness or stability when calls, chunks or identifiers change.

## Actual selective execution on further disjoint WANDS queries

`PROTOCOL_selective_execution.md` was fixed after observing the prior 96-query result, before selecting products or running the new model evaluation. It includes all 52 remaining eligible WANDS queries after excluding earlier query strings and 12,646 previously sampled WANDS product IDs. The program really sends pointwise requests only for valid keyed-order disagreements, then separately runs a keyed replay and an all-candidate pointwise diagnostic. The single recorded offline-batch duration excludes model loading and the later diagnostic calls; it does not establish online latency or general speedup. `PROTOCOL_selective_two_way.md` records the secondary crossed sensitivity fixed after sample selection and inference began but before inspecting outputs.

To verify the recorded sample, actual request selection, outputs, metrics and resampling without running inference:

```bash
sha256sum -c SELECTIVE_EXECUTION_PRESELECTION_SHA256.txt
sha256sum -c SELECTIVE_EXECUTION_FROZEN_SHA256.txt
sha256sum -c SELECTIVE_EXECUTION_TWO_WAY_FROZEN_SHA256.txt
python3 verify_selective_execution_selection.py --wands-dir /path/to/wands --selection wands_selective_execution_selection
python3 verify_selective_execution.py --run run_wands_selective_execution_qwen_v1 --selection wands_selective_execution_selection --model /path/to/Qwen3-4B-Instruct-2507
python3 verify_selective_two_way.py --run run_wands_selective_execution_qwen_v1 --selection wands_selective_execution_selection
```

The optional `--model` argument independently checks input tokens and context lengths with a local tokenizer. To rerun the actual scheduler, use a new output directory and the required GPU environment:

```bash
VLLM_USE_V1=1 python3 run_selective_execution.py --model /path/to/Qwen3-4B-Instruct-2507 --selection wands_selective_execution_selection --out selective_qwen_new
python3 analyze_selective_execution.py --run selective_qwen_new --selection wands_selective_execution_selection
python3 analyze_selective_two_way.py --run selective_qwen_new --selection wands_selective_execution_selection
python3 verify_selective_execution.py --run selective_qwen_new --selection wands_selective_execution_selection --model /path/to/Qwen3-4B-Instruct-2507
python3 verify_selective_two_way.py --run selective_qwen_new --selection wands_selective_execution_selection
```

The two Qwen samples use the same public WANDS source, relevance proxies, one model and one GPU. The additional crossed resampling changes within-query product weights and is not exact coverage for the fixed-pool estimand.

## Secondary SmolLM2 selective-execution sensitivity

`PROTOCOL_selective_model_sensitivity.md` was fixed after observing the Qwen result and before the SmolLM2 inference. It uses the exact same 52 queries and products, generates model-specific specifications, and actually sends pointwise requests only for disagreements. It is a secondary within-sample model sensitivity, not a new independent dataset or a causal model-family comparison. Pointwise SmolLM2 predicted N for all 1,664 candidates and selective fallback made no true-positive prediction; the failure is retained in the paper and saved outputs. Model families, sizes, tokenizers, chat templates and specifications differ together.

Verify the recorded result without running inference (the local tokenizer is optional):

```bash
sha256sum -c SELECTIVE_MODEL_SENSITIVITY_FROZEN_SHA256.txt
python3 verify_selective_model_sensitivity.py --run run_wands_selective_execution_smollm_v1 --selection wands_selective_execution_selection --model /path/to/SmolLM2-1.7B-Instruct
python3 verify_selective_model_two_way.py --run run_wands_selective_execution_smollm_v1 --selection wands_selective_execution_selection
```

To repeat, use a new output directory and the required GPU environment; reruns may differ even at temperature zero:

```bash
VLLM_USE_V1=1 python3 run_selective_model_sensitivity.py --model /path/to/SmolLM2-1.7B-Instruct --selection wands_selective_execution_selection --out selective_smollm_new
python3 analyze_selective_model_sensitivity.py --run selective_smollm_new --selection wands_selective_execution_selection
python3 analyze_selective_model_two_way.py --run selective_smollm_new --selection wands_selective_execution_selection
python3 verify_selective_model_sensitivity.py --run selective_smollm_new --selection wands_selective_execution_selection --model /path/to/SmolLM2-1.7B-Instruct
python3 verify_selective_model_two_way.py --run selective_smollm_new --selection wands_selective_execution_selection
```

## Post-hoc supervised WANDS reference

`PROTOCOL_wands_supervised_reference.md` fixes a feature-limited logistic baseline after earlier WANDS outcomes were observed. It trains on 5,120 already-labeled candidates from 160 earlier WANDS queries, then evaluates the existing 160-, 96- and 52-query samples. All three evaluation samples are query-string and sampled-product disjoint from the training sample. The classifier uses supervised labels, unlike the zero-shot LLM plans; the samples and outputs were already exposed, and this is not a new independent dataset or a strong tuned ranker. The saved predictions and all three results, including poor supervised performance, are retained.

The recorded run uses scikit-learn 1.7.2, SciPy 1.16.2 and threadpoolctl 3.6.0. With those dependencies installed, verify the frozen inputs and independently refit and analyze the saved results:

```bash
sha256sum -c WANDS_SUPERVISED_REFERENCE_FROZEN_SHA256.txt
PYTHONHASHSEED=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 verify_wands_supervised_reference.py --run run_wands_supervised_reference
```

To repeat the fit, use a new, nonexistent output directory:

```bash
PYTHONHASHSEED=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 run_wands_supervised_reference.py --out wands_supervised_new
PYTHONHASHSEED=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 verify_wands_supervised_reference.py --run wands_supervised_new
```

The original Qwen conditions, model and label comparisons, and query-bootstrap intervals are post-hoc and unadjusted for multiple comparisons. Repeated products within the test pools and the query/product sampling design limit interpretation.

## Calibrated Qwen3 embedding reference

`PROTOCOL_wands_embedding_reference.md` fixes a further post-hoc reference after prior outcomes were observed. Qwen3-Embedding-0.6B is used as a bi-encoder; one cosine threshold is fitted using labels from the first WANDS sample, then applied unchanged to the three existing test samples. Training and testing have no overlapping normalized query strings, sampled product IDs or exact serialized product text. This supervised calibration is not equivalent to a zero-shot method, a new blind test or a state-of-the-art reranker. The source model revision, local hashes and which assets match the official revision are in `MODEL_EMBEDDING_SHA256.txt`; weights are not included. Saved float32 vectors, scores, threshold and candidate mappings are in `run_wands_embedding_reference/`.

Verify the frozen experiment and independently reconstruct stored scores, threshold, decisions, metrics and bootstrap intervals:

```bash
sha256sum -c WANDS_EMBEDDING_REFERENCE_FROZEN_SHA256.txt
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 verify_wands_embedding_reference.py --run run_wands_embedding_reference
sha256sum -c WANDS_EMBEDDING_REPLAY_FROZEN_SHA256.txt
python3 verify_wands_embedding_replay.py --prior run_wands_embedding_reference --replay run_wands_embedding_replay
```

`run_embedding_synthetic_smoke.py` and `run_wands_embedding_reference/synthetic_smoke.json` preserve the unrelated example used to confirm that the model loaded and ranked its relevant passage first. The replay repeats 32 query vectors and five fixed batches of 32 document vectors on the same hardware; it does not prove numerical determinism across hardware or all inputs. To rerun embedding inference, use a fresh output directory, the recorded GPU environment and the locally downloaded, hash-matched model:

```bash
PYTHONHASHSEED=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 run_wands_embedding_reference.py --model /path/to/Qwen3-Embedding-0.6B --out embedding_new
PYTHONHASHSEED=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 run_wands_embedding_replay.py --model /path/to/Qwen3-Embedding-0.6B --prior embedding_new --out embedding_replay_new
```

Install scikit-learn 1.7.2, SciPy 1.16.2 and threadpoolctl 3.6.0 before running these scripts, along with the listed PyTorch and Transformers dependencies. Timing is one offline batch, not an online speed measurement.

### Post-hoc local HTTP service workload

`PROTOCOL_local_service_benchmark.md` describes the 52 previously exposed WANDS pools, six plan/concurrency conditions and two repetitions. `LOCAL_SERVICE_SERVER_CONFIG.md` records the local server; `LOCAL_SERVICE_BENCHMARK_FROZEN_SHA256.txt` freezes the original design, prompts and client before measurement. The first condition completed 49 of 52 pools and recorded three loopback `ReadError` requests; the original client then stopped. `run_local_service_initial_aborted/` preserves that condition and its traceback. The after-observation continuation, new client connection pool and unchanged fixed remaining schedule are explicitly documented in `LOCAL_SERVICE_BENCHMARK_AMENDMENT.md`; `LOCAL_SERVICE_CONTINUATION_FROZEN_SHA256.txt` freezes the amendment, continuation code and original measurements before continuation. The resumed results include the original first-condition records verbatim; there is no replacement, retry or imputation for HTTP failures. The record of nonsecret server-startup excerpts is in `run_local_service_benchmark_resumed/server_startup_evidence.txt`.

To audit the retained results without a GPU or server (Python with NumPy and httpx 0.28.1):

```bash
sha256sum -c LOCAL_SERVICE_BENCHMARK_FROZEN_SHA256.txt
sha256sum -c LOCAL_SERVICE_CONTINUATION_FROZEN_SHA256.txt
python3 test_local_service_benchmark.py
python3 verify_local_service_benchmark.py --run run_local_service_benchmark_resumed --selection wands_selective_execution_selection
```

A fresh execution needs the recorded Qwen model, vLLM 0.10.2 and an available compatible GPU. Start the following server in a separate terminal, adjusting the local model path and choosing an unused local port; launch no concurrent GPU workload. The same server must stay running throughout the client run. This launch is local-only and does not require a credential.

```bash
VLLM_USE_V1=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python3 -m vllm.entrypoints.openai.api_server \
  --model /path/to/Qwen3-4B-Instruct-2507 --served-model-name intentsat-local \
  --dtype bfloat16 --max-model-len 4096 --gpu-memory-utilization 0.70 \
  --max-num-seqs 16 --enforce-eager --no-enable-prefix-caching --disable-log-stats \
  --host 127.0.0.1 --port 35069
```

Then run the unamended client using a new output path. It checks incompatible-output controls, warms up on an unrelated synthetic query and records each condition; it stops after a condition with an HTTP failure. Do not overwrite the provided results. `resume_local_service_benchmark.py` applies specifically to the retained first-condition observation described in the amendment and is not a general crash recovery facility.

```bash
PYTHONHASHSEED=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 run_local_service_benchmark.py \
  --endpoint http://127.0.0.1:35069 --model intentsat-local \
  --selection wands_selective_execution_selection --out run_local_service_new
python3 verify_local_service_benchmark.py --run run_local_service_new --selection wands_selective_execution_selection
```

Each percentile is over complete query pipelines and excludes the client admission queue, model loading, startup and warmup. Server batching, fan-out, loopback transport, generation and parsing remain included. Calls that fail before usage is returned have unobserved token usage; recorded token totals should not be read as total server consumption. The same 52 pools are repeated, specifications are generated afresh, and timing depends on hardware, load and software. These measurements do not establish public Internet latency, production reliability, an externally valid speedup, or cost.

## Isolated CPU verification environment

`requirements-verify.txt` records the exact Python packages resolved in a virtual environment on the same Linux x86-64 host, using Python 3.12.11. `run_cpu_verification_suite.sh` runs selected saved-output checks with one CPU thread. Binary wheels were installed from PyPI without system site-packages and `pip check` passed in this CPU environment. Earlier isolated audits passed 21, 23 and then 25 invocations; the current suite passes 26 invocations across 24 scripts after adding synthetic checks for the model-asset verifier. The current environment uses NumPy 1.26.4, matching the LOTUS cascade's original run; the earlier 23-check environment used 2.2.6. Logs and snapshots are identified in `CPU_VERIFICATION_HISTORY.md`. WANDS source tables were the already downloaded, hash-verified local copy, not a new independent download or a different physical machine.

For a fresh CPU-only verification in a compatible Python 3.12 Linux environment, extract this artifact, enter its `reproducibility` directory, then run:

```bash
python3 -m venv /tmp/intentsat-verify-env
/tmp/intentsat-verify-env/bin/python -m pip install -r requirements-verify.txt
/tmp/intentsat-verify-env/bin/python -m pip check
sha256sum -c CPU_VERIFICATION_SHA256.txt
INTENTSAT_AUDIT_PYTHON=/tmp/intentsat-verify-env/bin/python bash run_cpu_verification_suite.sh --wands-dir /path/to/WANDS/dataset
```

The optional WANDS path must contain the pinned `query.csv`, `product.csv`, and `label.csv`; omitting `--wands-dir` skips only the three checks that reconstruct selections from the original source tables. The suite checks recorded outputs, model-label associations, bootstrap calculations, selective-request records, local-service timings, supervised refitting, saved embedding calculations and elementary bounds. The requirements snapshot is not a hashed wheel lock and is not a complete GPU environment. These steps do not rerun vLLM inference, fetch model weights, reproduce GPU timing on another machine, validate rights in catalog content, or publish the artifact.

## Cross-validated decision threshold for the WANDS supervised reference

`PROTOCOL_wands_calibrated_lexical.md` specifies a further post-hoc reference after all earlier WANDS results were observed. It addresses the original classifier's fixed 0.5 decision threshold: five query-level folds use training labels, excluding the validation fold's product IDs and exact product texts from fold training. An F1-maximizing threshold is selected on the combined out-of-fold scores, then applied unchanged to the already saved probabilities from the classifier fitted on all 160 training queries. This chooses a decision threshold rather than calibrating probabilities; the selected out-of-fold F1 uses the same scores and labels used for threshold selection, and the three test samples were already exposed. F1 and ratio MAE can move in opposite directions. The procedure, fixed inputs, pre-fit setup note, fold scores, predictions, adverse outcomes, and independent refit are retained.

With the CPU verification dependencies installed, audit the frozen setup and result:

```bash
sha256sum -c WANDS_CALIBRATED_LEXICAL_FROZEN_SHA256.txt
PYTHONHASHSEED=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 test_wands_calibrated_threshold.py
PYTHONHASHSEED=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 verify_wands_calibrated_lexical.py --run run_wands_calibrated_lexical
```

To independently rerun the five folds, use a new output directory and compare only after recording any failures:

```bash
PYTHONHASHSEED=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 run_wands_calibrated_lexical.py --out wands_calibrated_lexical_new
PYTHONHASHSEED=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 verify_wands_calibrated_lexical.py --run wands_calibrated_lexical_new
```

`WANDS_CALIBRATED_LEXICAL_PREFIT_NOTE.md` records the initial missing-pyarrow unit-test failure before any fitting and the added query-uniqueness assertions before the final freeze. The bootstrap intervals are descriptive and do not correct for exposed test samples, multiple comparisons, or query/product dependencies.

## LOTUS helper cascade on the exposed WANDS sample

`PROTOCOL_lotus_wands_cascade.md` records one post-hoc configuration of the actual LOTUS 1.2.4 `sem_filter`: Qwen-only versus a SmolLM2 helper plus Qwen cascade on the already exposed 52 pools. The fixed source, setup, model hashes, unrelated synthetic tests and pre-measurement checks are in `LOTUS_WANDS_CASCADE_FROZEN_SHA256.txt` and `LOTUS_CASCADE_PACKAGE_EVIDENCE.md`. The `run_lotus_wands_cascade/` directory retains the exact row order, prompts, raw outputs, helper token log probabilities, calibration samples, routing, usage and results. `server_startup_evidence.json` is a post-run extraction of selected original server startup fields, not a signed attestation; some runner preflight fields are hard-coded. The comparison is not a benchmark of the complete LOTUS optimizer. Nominal cascade targets concern the reference-model procedure, not human-label accuracy, and the separate full-filter run does not establish a guarantee.

LOTUS 1.2.4 finds both exact label tokens in the top-logprob list at the first generated token of all 1,664 helper responses, which was `Answer`; this score is not necessarily the label probability at the subsequent answer position. A post-hoc position audit is saved as `proxy_position_audit.json`. The exact NumPy version matters: under 2.2.6, tiny floating differences cross quantile boundaries for 155 proxy scores, and the strict trace verification fails. The first failure is retained in `CPU_LOTUS_NUMPY2_INITIAL_FAILURE.log`. Use the pinned 1.26.4 snapshot for this saved-output verification:

```bash
sha256sum -c LOTUS_WANDS_CASCADE_FROZEN_SHA256.txt
PYTHONHASHSEED=0 python3 verify_lotus_wands_cascade.py --run run_lotus_wands_cascade
python3 analyze_lotus_proxy_position.py --verify-only
```

`analyze_lotus_numpy_versions.py` recomputes raw and quantized proxy scores from the same saved helper log probabilities in the active NumPy environment; archived outputs for both versions are in `run_lotus_wands_cascade/numpy_versions/`. To repeat, run the script separately in each environment with `--run run_lotus_wands_cascade --out /a/new/output.json`; it does not change the recorded cascade or rerun either model.

The CPU suite does not require LOTUS or model weights. The additional `test_lotus_wands_cascade.py` cross-checks the independent threshold/parser/prompt implementation against the exact official LOTUS 1.2.4 wheel and requires that separate full LOTUS environment. `LOTUS_CASCADE_RERUN.md` gives the two local server commands, the complete client command and model-hash verification, together with the server-identity and environment limitations. Re-running model inference requires the two original model weights and local vLLM servers; GPU timing and cross-version numerical identity are not guaranteed.

## Synthetic check with vLLM's declared dependency versions

After the original GPU environment's dependency conflicts were found, `PROTOCOL_vllm_declared_deps_smoke.md` fixed a small unrelated check using a process-local overlay containing the xgrammar, outlines-core and setuptools versions declared by vLLM 0.10.2. The three wheel digests match official PyPI metadata; 389 installed code files match their wheels. The original environment and prior results were not changed. In this inherited overlay, `pip check` no longer reports a vLLM mismatch, but it still reports conflicts involving sglang, outlines and pygobject; it is **not** a clean or fully consistent environment. Wheel metadata, setup logs and the remaining conflicts are in `vllm_declared_deps_setup/`.

The Qwen/V1 run's unconstrained `BANANA` negative control failed both strict output formats as expected; compact and keyed guided outputs passed their parsers, as did two manually authored eight-product prompts. The process exited successfully but logged nanobind cleanup warnings. These are synthetic structural checks, not human-label accuracy, benchmark or fresh-machine replication. Raw prompts and outputs are in `run_vllm_declared_deps_smoke/`; do not substitute them for the paper's experimental results.

```bash
sha256sum -c VLLM_DECLARED_DEPS_SMOKE_FROZEN_SHA256.txt
python3 verify_vllm_declared_deps_smoke.py --run run_vllm_declared_deps_smoke
```

The verifier independently reconstructs the prompts and parses the recorded outputs. An optional `--model /path/to/Qwen3-4B-Instruct-2507` also checks prompt token lengths using the local tokenizer, without a GPU. The separately downloaded package wheels and model weights are not bundled; official wheel names and hashes are in `vllm_declared_deps_setup/wheel_evidence.json`.

To try the same **limited synthetic check** on a comparable Linux x86-64 host with the recorded base packages and a free GPU, create a new overlay and output directory. The hash check below intentionally fails if a different platform wheel is downloaded. Inspect `pip check` separately: inherited package conflicts are expected and must not be represented as resolved.

```bash
TASK_REPRO_DIR="$PWD"
TASK_OVERLAY_DIR=/path/to/new-overlay
TASK_MODEL_DIR=/path/to/Qwen3-4B-Instruct-2507
mkdir "$TASK_OVERLAY_DIR"
mkdir "$TASK_OVERLAY_DIR/wheels" "$TASK_OVERLAY_DIR/site" "$TASK_OVERLAY_DIR/cache"
python3 -m pip download --no-cache-dir --no-deps --only-binary=:all: \
  --dest "$TASK_OVERLAY_DIR/wheels" xgrammar==0.1.23 outlines-core==0.2.11 setuptools==79.0.1
(cd "$TASK_OVERLAY_DIR/wheels" && sha256sum -c "$TASK_REPRO_DIR/vllm_declared_deps_setup/wheel_sha256.txt")
python3 -m pip install --no-index --no-deps --no-cache-dir \
  --target "$TASK_OVERLAY_DIR/site" "$TASK_OVERLAY_DIR"/wheels/*.whl
PYTHONPATH="$TASK_OVERLAY_DIR/site:$TASK_REPRO_DIR" VLLM_USE_V1=1 VLLM_NO_USAGE_STATS=1 \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  VLLM_CACHE_ROOT="$TASK_OVERLAY_DIR/cache" TRITON_CACHE_DIR="$TASK_OVERLAY_DIR/cache/triton" \
  python3 run_vllm_declared_deps_smoke.py --model "$TASK_MODEL_DIR" --out vllm_deps_smoke_new
python3 verify_vllm_declared_deps_smoke.py --run vllm_deps_smoke_new --model "$TASK_MODEL_DIR"
```

The overlay is an experiment on the inherited environment, not an independently resolved installation. Its runner requires the recorded base versions and refuses to start if another GPU process is detected. Use a verified model directory, keep failure logs and do not overwrite the archived output.

## TREC-COVID cross-domain transfer check

`PROTOCOL_trec_covid_transfer.md` fixes the 50-topic, judged-document sampling and comparison before model inference. The original selector failed before creating a sample because two qrel rows had an unrecognized `-1` grade; `TREC_COVID_SELECTION_AMENDMENT.md` and the original failure and frozen source are retained. The corrected selector excludes those rows without treating them as negatives. `TREC_COVID_CROSSED_SENSITIVITY_AMENDMENT.md` was fixed after sampling but before inference. `PROTOCOL_trec_covid_smollm.md` adds a second-model sensitivity check on the same exposed sample after the Qwen result. This is an exploratory biomedical relevance adaptation; it is not medical advice, a TREC leaderboard result, or evidence of shopper satisfaction. Sampled article titles and abstracts may have different underlying rights: see `trec_covid_sources/SOURCES.md` before public release.

The NIST topics and qrels are included. The full BEIR corpus archive is not included; download the exact archive and check its published MD5 and recorded SHA-256:

```bash
curl -fL https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/trec-covid.zip -o /path/to/trec-covid.zip
md5sum /path/to/trec-covid.zip
sha256sum /path/to/trec-covid.zip
sha256sum -c TREC_COVID_SELECTION_FROZEN_SHA256.txt
sha256sum -c TREC_COVID_SELECTION_AMENDMENT_FROZEN_SHA256.txt
sha256sum -c TREC_COVID_INFERENCE_FROZEN_SHA256.txt
python3 verify_trec_covid_selection.py \
  --topics trec_covid_sources/topics-rnd5.xml \
  --qrels trec_covid_sources/qrels-covid_d5_j0.5-5.txt \
  --archive /path/to/trec-covid.zip --selection trec_covid_transfer_selection
python3 verify_trec_covid_transfer.py --selection trec_covid_transfer_selection \
  --run run_trec_covid_transfer --analysis TREC_COVID_TRANSFER_ANALYSIS.json
```

The expected archive MD5 is `ce62140cb23feb9becf6270d0d1fe6d1` and SHA-256 is `120f42a7864d2214234537733c0d2c6684e42fdfafff2c5eacf98afca6656aa0`. Pass `--model /path/to/Qwen3-4B-Instruct-2507` to the last verifier to independently check every wrapped prompt hash and input token count with the local tokenizer. `test_trec_covid_transfer.py` checks prompt/label isolation, parsing, reversal and fallback; `test_trec_covid_pipeline.py` exercises analysis and the independent verifier using fabricated traces. The saved run used the process-local dependency overlay described above, not a fresh environment; it exited successfully but recorded nanobind cleanup warnings. Never replace saved outputs when repeating inference:

```bash
python3 select_trec_covid_transfer_v2.py --topics trec_covid_sources/topics-rnd5.xml \
  --qrels trec_covid_sources/qrels-covid_d5_j0.5-5.txt \
  --archive /path/to/trec-covid.zip --out trec_selection_new
python3 preflight_trec_covid_context.py --model /path/to/Qwen3-4B-Instruct-2507 \
  --selection trec_selection_new --out trec_context_new.json
PYTHONPATH="$TASK_OVERLAY_DIR/site:$TASK_REPRO_DIR" VLLM_USE_V1=1 VLLM_NO_USAGE_STATS=1 \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python3 run_trec_covid_transfer.py --model /path/to/Qwen3-4B-Instruct-2507 \
  --selection trec_selection_new --out trec_run_new
python3 analyze_trec_covid_transfer.py --selection trec_selection_new --run trec_run_new --out trec_analysis_new.json
```

All five conditions completed 50/50 pools in the saved run, but the query-bootstrap and query–document sensitivity intervals for selective-minus-keyed ratio MAE both span zero. The latter reweights observed pools and is not an exact fixed-pool confidence interval; many abstracts are truncated, repeated documents occur across topics, and model pretraining exposure is unknown.

For the second model, the frozen follow-up and independent checks are:

```bash
sha256sum -c TREC_COVID_SMOL_FROZEN_SHA256.txt
sha256sum -c TREC_COVID_SMOL_RESULTS_SHA256.txt
python3 verify_trec_covid_transfer.py --selection trec_covid_transfer_selection \
  --run run_trec_covid_smollm --analysis TREC_COVID_SMOL_ANALYSIS.json
```

Optionally add `--model /path/to/SmolLM2-1.7B-Instruct` to verify its tokenizer. To rerun, replace the model and runner in the earlier command with `SmolLM2-1.7B-Instruct` and `run_trec_covid_smollm.py`, and choose a new output directory; use the same dependency overlay and an idle GPU. This follow-up is post-hoc on the same sample, not a new dataset. Its original and selective MAE are 0.2131 and 0.2288; both the query and crossed sensitivity intervals for the difference are positive on this sample. Model family, size, tokenizer and template all differ, and prompts were not separately tuned.

If sampled article text cannot be redistributed, `make_artifact_without_sampled_trec_text.py` can create a **separate** ZIP that omits the two TREC sample files and includes instructions to reconstruct them from the pinned official sources. It refuses to overwrite an existing output and byte-checks every retained file. This does not remove product text or other third-party content and is not a blanket license to publish. From this directory, after building the full reproducibility archive:

```bash
python3 make_artifact_without_sampled_trec_text.py \
  --source ../03_submission/reproducibility_artifact.zip \
  --out /path/to/new_artifact_without_sampled_trec_text.zip
```

## Post-hoc partial-aggregate diagnostic

`PROTOCOL_partial_aggregate_bias.md` fixes a descriptive analysis after the original outcomes were already observed. On Qwen's failed compact-fallback pools in the 250-query ESCI and 160-query WANDS runs, it reconstructs only strictly valid final chunks and compares two hypothetical ratios: accepted positives divided by accepted labels, and accepted positives divided by the full pool size with unresolved labels treated as negative. These are not executed service plans. The saved labels also allow an exact arithmetic split of available-only signed error into observed-subset composition and accepted-label discrepancy terms; this is not a causal effect of changing a model. All reported pools have at least one accepted label, but the code and tests explicitly retain the undefined case when none is accepted.

```bash
sha256sum -c PARTIAL_AGGREGATE_BIAS_FROZEN_SHA256.txt
sha256sum -c PARTIAL_AGGREGATE_BIAS_RESULTS_SHA256.txt
python3 test_partial_aggregate_bias.py
python3 verify_partial_aggregate_bias.py --run run_partial_aggregate_bias
```

`run_partial_aggregate_bias/details.jsonl` retains per-query counts; `composition_secondary.json` was computed after inspecting the first diagnostic to expose cancellation in signed averages. The separately recorded spread is descriptive, not a pre-specified test or population estimate. Neither labels nor failures establish actual customer satisfaction.

## Post-hoc failure-aware lexical fallback on all attempted pools

`PROTOCOL_failure_lexical_fallback.md` is fixed after all original outcomes and labels were observed. Without new inference, it compares whole-pool lexical fallback with retention of valid saved chunks followed by lexical completion. It includes all 250 ESCI and 160 WANDS attempted pools for each model; the two models reuse those same queries. These are trace-derived alternative policies, not measured online executions. The deterministic lexical rule uses only query and candidate text. Query bootstrap intervals are retrospective and unadjusted, and the product-relevance references do not measure real shopper satisfaction.

```bash
sha256sum -c FAILURE_LEXICAL_FALLBACK_FROZEN_SHA256.txt
sha256sum -c FAILURE_LEXICAL_FALLBACK_RESULTS_SHA256.txt
python3 test_failure_lexical_fallback.py
python3 verify_failure_lexical_fallback.py --run run_failure_lexical_fallback
python3 analyze_failure_lexical_fallback.py --out failure_lexical_fallback_new
```

The independent verifier reconstructs strict chunk validity, all fallback decisions, completed-pool consistency, source counts, metrics and paired intervals, and compares available prior failure-bound traces. The analysis command requires a new directory and never overwrites the archived result.

## ESCI product-text truncation sensitivity

`PROTOCOL_esci_truncation_sensitivity.md` fixes a follow-up after earlier results were observed but before reconstructing longer source texts or running these comparisons. It reuses the existing 250-query ESCI sample and compares 600- and 1,200-character prefixes of the same official source fields, plus an identical 600-character replay. The complete original product Parquet and model weights are not bundled. The extended sampled product text in `esci_truncation_selection/inputs.jsonl` remains in **both** reproducibility archive variants, and its redistribution rights must be reviewed alongside other product content.

```bash
sha256sum -c ESCI_TRUNCATION_SELECTION_FROZEN_SHA256.txt
sha256sum -c ESCI_TRUNCATION_INFERENCE_FROZEN_SHA256.txt
sha256sum -c ESCI_TRUNCATION_RESULTS_SHA256.txt
python3 verify_esci_truncation_selection.py \
  --products /path/to/shopping_queries_dataset_products.parquet \
  --prior run_full_pools_250 --selection esci_truncation_selection
python3 verify_esci_truncation.py --selection esci_truncation_selection \
  --run run_esci_truncation --analysis ESCI_TRUNCATION_ANALYSIS.json
```

The independent verifier optionally accepts `--model /path/to/Qwen3-4B-Instruct-2507` to check wrapped prompts and input-token counts. `test_esci_truncation_pipeline.py` exercises fabricated paired traces and checks that prompt tampering is detected. The longer prefix can still be incomplete and is not automatically more accurate. Query-bootstrap results are descriptive on a public judged sample; the secondary query–product exponential-weight range changes the candidate-pool estimand and is not a frequentist confidence interval.

To reproduce inference, generate a new selection and context preflight, then use the recorded dependency overlay, an idle GPU and a new output directory; this does not overwrite any saved result:

```bash
python3 select_esci_truncation.py --products /path/to/shopping_queries_dataset_products.parquet \
  --prior run_full_pools_250 --out esci_truncation_selection_new
python3 preflight_esci_truncation.py --model /path/to/Qwen3-4B-Instruct-2507 \
  --selection esci_truncation_selection_new --out esci_truncation_context_new.json
PYTHONPATH="$TASK_OVERLAY_DIR/site:$TASK_REPRO_DIR" VLLM_USE_V1=1 VLLM_NO_USAGE_STATS=1 \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python3 run_esci_truncation.py --model /path/to/Qwen3-4B-Instruct-2507 \
  --selection esci_truncation_selection_new --preflight esci_truncation_context_new.json --out esci_truncation_run_new
python3 analyze_esci_truncation.py --selection esci_truncation_selection_new \
  --run esci_truncation_run_new --out esci_truncation_analysis_new.json
```

## Same-machine offline isolated-environment diagnostic

The separate public-index installation attempt in `PROTOCOL_clean_gpu_environment.md` could not connect to PyPI or the PyTorch CUDA index; both failed resolution logs are preserved. `PROTOCOL_offline_isolated_gpu_diagnostic.md` then describes a **local-copy**, isolated Python environment assembled from the existing host's frozen dependency closure and the three-wheel overlay. Its 147 distributions passed an independent metadata dependency check, and both source and destination hashes for 41,092 installed-file records matched. The 47 omitted entrypoints or manuals are listed in `CLEAN_GPU_OFFLINE_COPY/copy_summary.json`. The host Python interpreter, GPU driver, model files, and operating system remain dependencies. Copy hashes do not certify the public provenance of the host's packages; this is not a clean online install or a fresh-machine reproduction.

The five synthetic format outputs and token counts match the earlier overlay check. After that check, `PROTOCOL_offline_wands_replay.md` fixed a replay of the first 16 already observed WANDS holdout queries, with their saved specifications and candidate IDs. All 16 pools and 64 chunks per condition completed without retries. Original-order labels differ on 24/512 candidates from the earlier run; the new run's own identical-prompt replay differs on 13/512. The batch size and runtime changed together, and these first 16 pools are not an independent random sample; neither result identifies a cause or guarantees reproducibility.

From this directory, the saved evidence can be audited without GPU inference:

```bash
sha256sum -c CLEAN_GPU_PRESETUP_FROZEN_SHA256.txt
sha256sum -c CLEAN_GPU_OFFLINE_FROZEN_SHA256.txt
sha256sum -c OFFLINE_WANDS_REPLAY_FROZEN_SHA256.txt
sha256sum -c OFFLINE_WANDS_REPLAY_RESULTS_SHA256.txt
python3 test_offline_isolated_gpu.py
python3 test_offline_wands_replay.py
python3 verify_vllm_declared_deps_smoke.py --run run_clean_gpu_offline_smoke
python3 verify_offline_wands_replay.py --selection wands_holdout_selection \
  --prior run_wands_holdout_qwen_v1 --run run_offline_wands_replay
```

The last two verifiers optionally accept `--model /path/to/Qwen3-4B-Instruct-2507` to check tokenizer-dependent prompt lengths and hashes. The copied-file manifest contains absolute source paths from the original host, so its `verify_offline_isolated_gpu.py --site ... --copy CLEAN_GPU_OFFLINE_COPY --graph CLEAN_GPU_OFFLINE_DEPENDENCY_GRAPH.json` check needs those original source files plus the corresponding copied environment; it cannot verify package provenance on a different machine. To repeat the local-copy diagnostic, create a **new** venv and use `prepare_offline_isolated_gpu.py` with the documented overlay, frozen graph and that venv's site-packages; it refuses version or dependency changes and existing output paths. Check the new environment with an independent dependency checker, then run `run_vllm_declared_deps_smoke.py` and `run_offline_wands_replay.py` only on an idle GPU with V1 and new output directories. Preserve failures and never overwrite saved outputs.

## Deterministic ratio-error floors

`PROTOCOL_ratio_instability_bounds.md` specifies a post-hoc, label-free analysis of the previously observed WANDS holdout replay and stable-ID reversal. The single-pool floor is `ceil(abs(p_a-p_b)/2)/n`. For `Q` equally sized pools it additionally gives a sharp lower bound on the **larger of the two query-macro MAEs**, `ceil(sum_q(abs(p_aq-p_bq))/2)/(Q*n)`, over unrestricted reference counts. The mean of the per-pool worst-error floors is a different quantity and must not be reported as a floor on the worse macro MAE. No probabilistic coverage or outcome attribution follows.

```bash
sha256sum -c RATIO_INSTABILITY_FROZEN_SHA256.txt
sha256sum -c RATIO_INSTABILITY_RESULTS_SHA256.txt
python3 test_ratio_disagreement_bounds.py
python3 verify_ratio_instability.py --selection wands_holdout_selection \
  --prior run_wands_holdout_qwen_v1 --run run_ratio_instability
```

The saved result has 160 common-complete pools of 32 candidates, a total positive-count gap of 247, and a sharp worse macro-MAE floor of 124/5,120. The independent verifier constructs a synthetic attainable reference-count witness; the actual labels are not needed to compute the floor. The result reuses the public, previously observed sample and cannot be generalized to unobserved search pools.

## Post-hoc multi-permutation sensitivity

`PROTOCOL_random_order_sensitivity.md` fixes three seeded within-chunk orders per query for the previously observed 160-pool WANDS holdout, plus three interleaved original-order baselines. Both Qwen and SmolLM2 use the same saved mappings and their respective earlier specifications; all six conditions completed all 160 pools for each model. Across the three paired random-order comparisons, Qwen changed 34.6--35.1% and SmolLM2 8.2--8.7% of parsed labels; both identical-prompt replay comparisons changed none in this run. These are post-hoc descriptive results on exposed data. Fixed execution order, prompt context, output association and numerical behavior prevent attributing changes to a single cause. Prior 16-query identical-prompt checks did show changes, so the zero replays here do not prove deterministic inference.

```bash
sha256sum -c RANDOM_ORDER_FROZEN_SHA256.txt
sha256sum -c RANDOM_ORDER_RESULTS_SHA256.txt
python3 verify_random_order_selection.py --selection wands_holdout_selection --orders random_order_selection
python3 test_random_order_sensitivity.py
python3 verify_random_order_sensitivity.py --selection wands_holdout_selection --orders random_order_selection \
  --prior run_wands_holdout_qwen_v1 --run run_random_order_qwen --analysis analysis_random_order_qwen
python3 verify_random_order_sensitivity.py --selection wands_holdout_selection --orders random_order_selection \
  --prior run_wands_holdout_smollm_v1 --run run_random_order_smollm --analysis analysis_random_order_smollm
```

Each verifier optionally takes `--model /path/to/the/corresponding/model` to check wrapped prompt hashes and token counts. To repeat inference, use the documented local-copy isolated environment, V1, an idle GPU and new output paths, replacing the model and prior directory for the second model. The runner checks GPU ownership between batches and does not overwrite saved results. The local package copy is not a clean installation from public indexes, and no new machine, production traffic or causal identification is claimed.

## New ESCI selective-fallback query pools

After earlier results, `PROTOCOL_esci_selective_holdout.md` fixed the sample and primary comparison before selection and inference. From 3,963 eligible US-English test queries, 250 query IDs were sampled; all 4,428 labeled examples in their source pools were retained. Normalized queries exclude previous ESCI and supervised-training queries, and product IDs exclude earlier original pools and the 50,000-example training sample. Derived pilot/validation product IDs were not retained. A post-hoc audit found 20 exact serialized-text matches in 18 queries despite these ID exclusions; none of those queries were replaced. A further post-hoc sensitivity drops those 18 queries. Exact text differences do not establish semantic or pretraining isolation.

Both models complete all 250 pools. Qwen selective versus keyed ratio MAE is 0.373 versus 0.407 (difference -0.0342; query-bootstrap 95% interval [-0.0451, -0.0235]); all-Y is 0.185. SmolLM2 selective versus keyed is 0.813 versus 0.802 (difference +0.0108; interval [0.0075, 0.0144]). Qwen selective makes 2,133 calls versus 4,690 including specification for full pointwise, but uses 1.52M versus 1.05M logical input tokens. The supervised reference uses 50,000 additional training labels. These results are conditional on selected public judged pools, one fixed run per model and a relevance proxy, not a population guarantee, production latency or evidence of user purchases.

```bash
sha256sum -c ESCI_SELECTIVE_PRESELECTION_SHA256.txt
sha256sum -c ESCI_SELECTIVE_INFERENCE_FROZEN_SHA256.txt
sha256sum -c ESCI_SELECTIVE_SUPERVISED_FROZEN_SHA256.txt
sha256sum -c ESCI_SELECTIVE_RESULTS_SHA256.txt
python3 test_esci_selective_holdout.py
python3 test_esci_selective_pipeline.py
python3 verify_esci_selective_selection.py --examples /path/to/shopping_queries_dataset_examples.parquet \
  --products /path/to/shopping_queries_dataset_products.parquet --selection esci_selective_holdout_selection
python3 verify_esci_selective_holdout.py --selection esci_selective_holdout_selection \
  --run run_esci_selective_holdout_qwen --analysis analysis_esci_selective_holdout_qwen
python3 verify_esci_selective_holdout.py --selection esci_selective_holdout_selection \
  --run run_esci_selective_holdout_smollm --analysis analysis_esci_selective_holdout_smollm
python3 verify_esci_selective_secondary.py --selection esci_selective_holdout_selection \
  --reference run_esci_selective_supervised_reference --reference-analysis analysis_esci_selective_supervised_reference \
  --qwen-run run_esci_selective_holdout_qwen --qwen-analysis analysis_esci_selective_holdout_qwen \
  --smollm-run run_esci_selective_holdout_smollm --smollm-analysis analysis_esci_selective_holdout_smollm
```

The two synthetic tests import the runner and require `psutil==7.0.0` as well as the documented NumPy and CPU dependencies; they do not start GPU inference. Two indirect analysis imports were omitted from the new pre-inference manifest. Afterward, they were byte-compared with the earlier complete archive and its earlier frozen manifest; `ESCI_SELECTIVE_DEPENDENCY_PROVENANCE.json` records this narrower evidence rather than claiming they were in the new manifest. Each model verifier optionally accepts `--model /path/to/the/corresponding/model` to check the wrapped prompt hashes and tokenizer-dependent input lengths. The supervised reference can be checked with `verify_esci_selective_supervised_reference.py --examples ... --products ... --selection esci_selective_holdout_selection --reference run_esci_selective_supervised_reference --refit`; the recorded independent refit used NumPy 2.2.6, pandas 2.3.3, SciPy 1.16.2 and scikit-learn 1.7.2. The separate clean CPU verification environment uses NumPy 1.26.4 and checks output reconstruction but is not the exact original refit environment. The first GPU-isolated test could not import scikit-learn; this is recorded, and the CPU reference was fit in the documented original training overlay. GPU inference was not rerun on another machine. The full results, model-weight hashes and rights caveats remain applicable.

## Chunk-local identities and duplicate coupling

`test_occurrence_contract.py` checks how local IDs restart in each chunk and map to unique pool positions. It also gives small exhaustive counterexamples showing that the stated conditional bounds remain valid, but their endpoints or floors may not be attainable, if duplicate occurrences must share a reference label. The paper's sharpness statements are over unrestricted occurrence labels. `audit_occurrence_ids.py` reports within-pool repeated source IDs across seven saved datasets; the derived ESCI samples have no source IDs, and distinct IDs can still have the same content. No sample is modified.

```bash
python3 test_occurrence_contract.py
sha256sum -c OCCURRENCE_AUDIT_SHA256.txt
python3 audit_occurrence_ids.py --out /path/to/new_occurrence_audit.json
```

The reduced archive omits the TREC sample input, so reconstruct that file using its included notice before running the complete hash check or the seven-dataset ID audit.

## Exact bounds with shared duplicate references

`PROTOCOL_coupled_ratio_bounds.md` and `coupled_ratio_bounds.py` give exact conditional ratio endpoints for a supplied partition of duplicate occurrences that must share a reference, or report an infeasible error budget. Some intermediate ratios may be unattainable. The algorithm does not identify duplicate products or justify the error budget; it is tested on synthetic pools and does not change the existing experiment results.

```bash
python3 test_coupled_ratio_bounds.py
sha256sum -c COUPLED_RATIO_BOUNDS_SHA256.txt
```


## Post-hoc third-model sensitivity

The Qwen2.5-7B product and TREC sensitivity checks, subsequent schema-guided specification comparison, frozen protocols, traces, model hashes, source snapshots and independent checks are described in `posthoc_7b/README.md`. Model weights and complete upstream datasets are not included. The original 7B product runs all fell back to direct judgments after strict specification parsing failed. New schema-guided runs reuse the same exposed queries and are descriptive whole-pipeline comparisons. The third model changes family, tokenizer, template and runtime settings together; no causal size comparison is implied.


## Post-hoc shared-product weighting sensitivity

The third-model schema-guided versus unguided selective pipelines have an exploratory query/product-weight sensitivity on both previously exposed ESCI and WANDS pools. Frozen inputs, 8,000 saved draws, scripts, independent verification and exact interpretation are under `posthoc_product_weight/`. Product reweighting changes the estimand and is not a fixed-pool confidence interval.

### Post-hoc ESCI label-mapping sensitivity

`posthoc_esci_label_mapping/` rescores the same 250 saved new-query pools under the original E/S-positive and alternate E-only reference mappings. The original prompts and supervised training still treat substitutes as positive. The alternate score is target sensitivity, not a new E-only task run. Protocol, input hashes, output, synthetic checks and independent raw-event verification are included in that directory.

### Variable-pool ratio-instability floor

`posthoc_variable_pool_floor/` gives a deterministic macro-ratio error floor for unequal pool sizes, an unrestricted reference-count witness, exhaustive synthetic tests and an independent raw-response reconstruction on the previously observed 250 new ESCI pools. The floor does not identify which plan is worse or estimate population coverage.

### Public-index installation retry (incomplete)

`PUBLIC_INDEX_RETRY_OUTCOME.md` preserves a later post-hoc public-index resolution, the 163 pinned versions, the installation command and log, and the space-guard termination. Resolution succeeded but installation, dependency checks and GPU inference did not complete. This is not a fresh-machine reproduction.

### Figure 2 and Figure 4 readability

`figure_readability/` preserves the previous and current vector/PNG figures and hashes of the source summaries. `make_figures.py` uses larger labels and 300 dpi PNG output for these two figures without changing the data. Set `INTENTSAT_FIGURE_OUTPUT_DIR` to an empty directory to render separately instead of overwriting manuscript figures.


### Figure 1: execution and selective plans

`make_figures.py` draws the batch output contract and the selective disagreement plan as a vector PDF and 300 dpi preview. The diagram distinguishes the plan-dependent treatment of an invalid specification from unresolved candidate judgments. It is an illustration of the recorded control flow, not new experimental data. The original pre-third-model figure code remains in `pre_7b_snapshot/` for the frozen archive checks.


### WANDS ratio-oriented lexical references

`PROTOCOL_wands_ratio_baseline.md` describes the post-hoc comparison using existing training-fold and test predictions. The method selects a ratio-MAE threshold using only the 160 training queries and also reports ratio-only mean-score and training-median references. Test samples and previous outcomes were already exposed; these are descriptive comparisons, not new model runs. A separate pre-analysis manifest records the protocol, initial analysis code and available inputs before computing these contrasts; the post-analysis manifest records additional dependencies. Run from this artifact directory:

```bash
sha256sum -c WANDS_RATIO_BASELINE_PRE_ANALYSIS_SHA256.txt
sha256sum -c WANDS_RATIO_BASELINE_POST_ANALYSIS_SHA256.txt
python3 run_wands_ratio_baseline.py --artifact . --out /path/to/new_ratio_baseline.json
python3 verify_wands_ratio_baseline.py --artifact . --result /path/to/new_ratio_baseline.json
```

`run_wands_ratio_baseline/summary.json` preserves the recorded values. The selected training-fold objective is not an independent validation score; score means do not provide candidate labels. The reference uses labeled training data. Query-bootstrap intervals are unadjusted and do not account for cross-query product reuse.


### Query–product sensitivity for WANDS ratio references

`PROTOCOL_wands_ratio_product_sensitivity.md` specifies an exploratory, post-hoc product- and query-reweighting check of the saved ratio-threshold, mean-score and lexical references. Product reweighting changes the within-query pools; its percentile ranges are not confidence intervals for the original fixed-pool MAE. The 96- and 52-query reruns reproduce the earlier selective-versus-keyed crossed ranges; an independent loop implementation checks the 52-query threshold and lexical contrasts. From this artifact directory:

```bash
sha256sum -c WANDS_RATIO_PRODUCT_PRE_ANALYSIS_SHA256.txt
python3 run_wands_ratio_product_sensitivity.py --artifact . --out /path/to/new_product_sensitivity.json
python3 verify_wands_ratio_product_sensitivity.py --artifact . --result /path/to/new_product_sensitivity.json
```

The recorded results are in `run_wands_ratio_product_sensitivity/summary.json`. Comparisons are post hoc, use previously exposed labeled pools, and are not adjusted for multiplicity.


### Ratio-oriented threshold for the WANDS embedding reference

`PROTOCOL_wands_embedding_ratio_reference.md` specifies a post-hoc analysis of saved embeddings. A scalar threshold minimizes query-macro ratio MAE on 160 labeled training queries only; test results were already exposed, the model was not fine-tuned, and no inference was repeated. The selected training score is not independent validation. From this artifact directory:

```bash
sha256sum -c WANDS_EMBEDDING_RATIO_PRE_ANALYSIS_SHA256.txt
python3 run_wands_embedding_ratio_reference.py --artifact . --out /path/to/new_embedding_ratio.json
python3 verify_wands_embedding_ratio_reference.py --artifact . --result /path/to/new_embedding_ratio.json
```

`run_wands_embedding_ratio_reference/summary.json` records all comparisons. Query-bootstrap intervals and product-reweighted sensitivity ranges are unadjusted; changing product weights changes the pools and estimand, so those ranges are not fixed-pool confidence intervals. The verifier was written after the results were observed.


### Post-hoc accounting of WANDS embedding inputs

The encoder's recorded GPU run saved only its maximum input length, not total token counts. `PROTOCOL_wands_embedding_token_accounting.md` reconstructs the original joint encoding inputs after the fact, with the exact pinned tokenizer, original batch order, vector-index mappings and saved candidate text. The recorded joint input comprises training and all exposed test samples. The reconstructed per-sample isolated counts are not separate measured runs, and token counts are not comparable latency, energy, FLOPs or dollars across encoder and generator models. No model weights or GPU are used in this accounting. The local model directory must contain the pinned tokenizer and config files matching the hashes checked by both scripts; its weights are not required. From this artifact directory:

```bash
sha256sum -c WANDS_EMBEDDING_TOKEN_PRE_ANALYSIS_SHA256.txt
TOKENIZERS_PARALLELISM=false python3 run_wands_embedding_token_accounting.py --artifact . --model /path/to/Qwen3-Embedding-0.6B --out /path/to/new_embedding_tokens.json
USE_TORCH=0 TOKENIZERS_PARALLELISM=false python3 verify_wands_embedding_token_accounting.py --artifact . --model /path/to/Qwen3-Embedding-0.6B --result /path/to/new_embedding_tokens.json
```

`run_wands_embedding_token_accounting/summary.json` records the counts. The verifier uses the same underlying pinned tokenizer library through the original Transformers interface; it is not a different tokenizer implementation or an independent hardware run.


### Reference-composition sensitivity for the 96-query WANDS validation

`PROTOCOL_wands96_composition.md` records a post-hoc decomposition of the already exposed 96-query trace-derived fallback contrast. The analysis reconstructs the original and reversed keyed outputs, pointwise and lexical predictions from saved traces, checks the existing full-sample results, and groups pools by their reference-label composition. These reference-defined groups cannot be used to route unlabeled queries. Within-group intervals are unadjusted and do not account for selection bias or product reuse. No new model inference or selective-scheduler timing was performed. From this artifact directory:

```bash
sha256sum -c WANDS96_COMPOSITION_PRE_ANALYSIS_SHA256.txt
python3 run_wands96_composition.py --artifact . --out /path/to/new_wands96_composition.json
python3 verify_wands96_composition.py --artifact . --result /path/to/new_wands96_composition.json
```

The recorded values are in `run_wands96_composition/summary.json`. The independent integer-count verifier was written after the results were observed.


### Post-hoc pool error-budget calibration

`PROTOCOL_pool_error_calibration.md` describes an exploratory reconstruction using 160 previously exposed complete WANDS pools for calibration and already exposed 96/52 pools for description. For a fixed prediction rule and exchangeable query-level scores, standard split-conformal rank calibration gives marginal coverage; the recorded selection and reuse do not establish those assumptions. The diagnostic compares calibrating candidate error counts with directly calibrating aggregate count residuals and records empirical interval coverage and width, including adverse outcomes. It runs on saved data and responses without model inference. From this artifact directory:

```bash
sha256sum -c POOL_ERROR_CALIBRATION_PRE_ANALYSIS_SHA256.txt
python3 run_pool_error_calibration.py --artifact . --out /path/to/new_pool_error_calibration.json
python3 verify_pool_error_calibration.py --artifact . --result /path/to/new_pool_error_calibration.json
```

`pool_error_calibration/summary.json` records the descriptive results. The independent verifier was written after the results were observed and exhaustively checks small rank/tie and binary-vector cases; neither the verifier nor these data establish future or per-query coverage.


### Manuscript post-hoc composition and size audits

`paper_posthoc_audits/` contains portable copies of the manuscript's prior pool-size, reference-label composition and stratified-effect audits, their unchanged recorded summaries, and an independent exhaustive integer check of the variable-pool ratio floor. The programs use saved results and perform no model inference; the observational groups and unadjusted resampling intervals do not establish causality or population coverage. Their default artifact root is the directory above `paper_posthoc_audits/`, so these commands also work when this reproducibility ZIP is unpacked on its own:

```bash
sha256sum -c PAPER_POSTHOC_AUDITS_SHA256.txt
python3 paper_posthoc_audits/pool_size_audit.py --out /path/to/new_pool_size.json
python3 paper_posthoc_audits/class_composition_audit.py --out /path/to/new_class_composition.json
python3 paper_posthoc_audits/stratified_effects_audit.py --out /path/to/new_stratified_effects.json
python3 paper_posthoc_audits/check_three_pool_ratio.py
```

Compare the first three new files to `pool_size_summary.json`, `class_composition_summary.json`, and `stratified_effects_summary.json` in that directory. The class-composition audit includes TREC-COVID and, in the TREC-text-omitting archive, runs only after the two omitted sample files have been reconstructed and hash-verified as described in `NOTICE_RECONSTRUCT_TREC_SAMPLE.md`. The results retain the original input hashes and post-hoc scope. The portable scripts differ from the earlier review-note scripts only in their default artifact-root path.
