# Auditing LLM-Based Semantic Aggregation: reproduction guide

**Release status:** This is a draft accompanying locally prepared EAB candidate artifacts. EAB requires all necessary artifacts to be publicly accessible at submission; no missing-artifact exemption is claimed. The authors must complete the rights and privacy review, choose the artifact variant, and publish the verified files before supplying a public URL. This document does not itself make the artifacts publicly accessible. The full local submission package also contains private audit material and is not itself the proposed public release.

## Files and scope

Place this guide, `ARTIFACT_ARCHIVES_SHA256.txt`, and the following files together in the reviewed public release. Publish **one** main artifact variant and all applicable supplements:

1. `reproducibility_artifact.zip` is the full local verification artifact and includes sampled TREC-COVID titles and abstracts. Alternatively, `reproducibility_without_trec_article_text.zip` omits two sampled TREC input files and supplies reconstruction instructions in `reproducibility/NOTICE_RECONSTRUCT_TREC_SAMPLE.md`. The reduced variant still contains other third-party material. Reconstruct the missing TREC inputs before running analyses that need them. Neither variant includes full upstream datasets or model weights.
2. `midpoint_ratio_supplement.zip` contains the post-hoc two-order midpoint, lexical tie-breaker, and candidate/ratio error-decomposition and matched-label-budget routing analyses; all use saved outputs from the main artifact.
3. `response_provenance_supplement.zip` contains the response-recording clarification and its audit.
4. `figure_style_supplement.zip` contains the original figure generator, data and figure checksums, PDF/SVG/PNG figures and instructions. It reads statistical input files from the main artifact.
5. `ACORD_无原文可重建复现候选.zip` reconstructs the Qwen ACORD selection and saved responses from the fixed upstream archive; `ACORD_SmolLM2_无原文复现候选.zip` and `ACORD_批次敏感性_无原文复现候选.zip` depend on that reconstructed Qwen work directory.

These candidate supplements omit ACORD contract and query text, but contain unsalted, linkable pool pseudonyms, model outputs, prompt hashes and aggregates. Pseudonymization is **not** anonymization or a redistribution license. The authors must review all included content and upstream terms; the full internal `05_进行中的独立验证/` directory is not part of this proposed release.

The additional required `review_audit_supplement.zip` contains the new controlled experiment, consensus-reference and count-envelope analyses, with its own reproduction guide. It does not replace any preceding archive.

With all nine archive variants present locally, run `sha256sum -c ARTIFACT_ARCHIVES_SHA256.txt` from this directory. If only one main variant is released, verify the lines for the files actually supplied; the other main-variant line will necessarily report a missing file.

## Response-recording correction

Some historical files and the main artifact README use fields or descriptions named “raw.” In early product-study runners, the strings were saved **after leading and trailing whitespace was stripped**, before subsequent parsing. The discarded whitespace cannot be recovered. Later TREC-COVID and ACORD keyed and pointwise outputs were recorded and validated without that trimming; LOTUS output uses its separate interface. See `response_provenance/NOTICE.md`. Analyses use the parser and saved strings from each actual run.

## Reconstruct and verify without new model inference

Extract the selected main archive to obtain its `reproducibility/` directory, and extract each supplement into a separate location. In the commands below, replace every path with a real path; `NEW_*` output paths must not already exist. The shell variables are examples, not preconfigured paths.

```sh
INTENT2S_BASE=/path/to/extracted/reproducibility
INTENT2S_QWEN_SUPP=/path/to/extracted/acord_reconstructable
INTENT2S_SMOL_SUPP=/path/to/extracted/acord_smol_reconstructable
INTENT2S_BATCH_SUPP=/path/to/extracted/acord_batch_recovery
INTENT2S_QWEN_TOKENIZER=/path/to/Qwen3-4B-Instruct-2507
INTENT2S_SMOL_TOKENIZER=/path/to/SmolLM2-1.7B-Instruct
INTENT2S_ACORD_ARCHIVE=/path/to/ACORD_upstream.zip
INTENT2S_QWEN_WORK=/path/to/new_qwen_work
INTENT2S_SMOL_WORK=/path/to/new_smol_work
INTENT2S_BATCH_WORK=/path/to/new_batch_work
export INTENTSAT_PYTHON=/path/to/compatible/python
```

Obtain the fixed ACORD source archive from [the recorded upstream revision](https://raw.githubusercontent.com/The-Atticus-Project/acord/942b33a7c6fed7ead362c7c296401086f8bc0321/ACORD%20Dataset%20%26%20ReadMe.zip). Its expected SHA-256 is `c9e851456a14aed161033b856e9f40f3cc9f46cf7282c2007b143420e93a65c3`. Qwen and Smol tokenizer/configuration files must match the hashes in their respective preflight summaries; the directory basenames must be as shown above. These CPU checks use the recorded NumPy 2.2.6, transformers 4.57.1 and tokenizers 0.22.0; the complete environment requirements and source notices are in the individual artifacts.

```sh
(cd "$INTENT2S_QWEN_SUPP" && sha256sum -c SHA256SUMS.txt)
bash "$INTENT2S_QWEN_SUPP/rebuild_from_upstream.sh" \
  "$INTENT2S_ACORD_ARCHIVE" "$INTENT2S_QWEN_TOKENIZER" \
  "$INTENT2S_BASE" "$INTENT2S_QWEN_WORK"

bash "$INTENT2S_SMOL_SUPP/rebuild_smol_followup.sh" \
  "$INTENT2S_QWEN_WORK" "$INTENT2S_SMOL_TOKENIZER" \
  "$INTENT2S_BASE" "$INTENT2S_SMOL_WORK"

bash "$INTENT2S_BATCH_SUPP/rebuild_batch_supplement.sh" \
  "$INTENT2S_QWEN_WORK" "$INTENT2S_QWEN_TOKENIZER" "$INTENT2S_BATCH_WORK"
```

The scripts check hashes, regenerate prompt-tokenization records, restore saved outputs and independently verify the analyses. They do **not** rerun the GPU model. The first four-item run lost its per-response traces after a restart; the reported replacement comparison reran both chunk sizes in the same process after earlier outcomes were known. The replacement runtime had conflicting vLLM dependency requirements. Its guided-output controls and saved results were checked, but it is not a clean independent replication. See `PROTOCOL_recovery_batch.md` and `RECOVERY_HISTORY.md` in the batch supplement.

For midpoint, lexical and error-decomposition analyses, follow `midpoint_ratio_supplement/README.md`; the midpoint/lexical CPU environment uses NumPy 1.26.4, while the error-decomposition diagnostic uses the Python standard library only. The routing ablation uses Python 3.12/NumPy 2.2.6 and exact uniform-subset expectations. It covers 302 distinct pools (52 WANDS and 250 ESCI) reused for two models, hence 604 pool–model cases, not 604 independent pools. Those results were independently reconstructed from both main artifact variants. The error-decomposition check covers the same 302 Qwen pools. The figure supplement also reads `routing_budget_result.json` from the updated two-order supplement. The corrected midpoint implementation followed an initial failed run after some results were observed. Figure regeneration instructions and source hashes are in `intent_sat_figure_supplement/README.md` and `figures/FIGURE_MANIFEST.json`; matching rendering libraries and fonts are needed for byte-identical files.

## Interpretation and verification limits

ACORD uses the same 57 queries in two scenarios, one constructed with reference labels; source-expanded negatives and truncated clauses remain, and pretraining exposure is unknown. The models, prompts and runs are not a controlled model-family or batch-size causal study. Reconstructed saved outputs support local re-analysis, not future-population coverage, deployment latency, legal utility, or a fresh-machine GPU reproduction. The main artifact records post-run dependency conflicts and a partial isolated same-machine smoke check; a clean installation and complete rerun on another machine have not been established. Consult the protocols, source notices and per-study logs before extending the claims.


## New review-audit supplement

Extract `review_audit_supplement.zip` separately and follow `review_audit/README.md`. The consensus and count-envelope analyses reuse the 302 distinct WANDS/ESCI pools for two models (604 pool–model cases); they are post-hoc trace analyses. The controlled experiment selects 128 additional ESCI pools and reuses 52 exposed WANDS pools. Its input/code/schedule freezes, pre-inference amendments, attempted-run records, both models' raw responses and independently verified statistics are retained. The experiment records a single-rank process-local NCCL compatibility adapter after standard Gloo initialization was denied access to PCI enumeration; both models use that same disclosed runtime. No original failure is silently replaced or described as a successful clean installation.

The latest standalone LaTeX package compiles the current manuscript without rerunning models. Rerendering statistics requires the main artifact plus the newer analysis directories and renderer from this supplement/source package. Historical scripts and results in the main archive remain unchanged; the current figure and audit supplements supersede their presentation. The supplied publication file hashes identify the intended revision.
