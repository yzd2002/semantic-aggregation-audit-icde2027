# Third-model post-hoc sensitivity and specification audit

The compressed trace archive contains all new 7B prompts' hashes, raw model responses, route and retry records, analyses, controls, frozen protocols and code, verification logs, model metadata and tokenizer files. The four model weight shards and full upstream datasets are **not included**. The saved official model revision and weight hashes are in the archive; download them separately only if rerunning inference. The old unguided product runs all used direct fallback because their fenced specification responses failed the strict parser. The later schema-guided experiment and all three datasets use already exposed samples; no causal size or semantic correctness claim follows.

From the extracted `reproducibility` directory use a new path:

```bash
python3 posthoc_7b/extract_and_verify.py --artifact . --out /path/to/new_7b_audit
```

The command checks the compressed archive, safely extracts it, checks every archived file, and verifies the original frozen source manifests. Three later-edited publication scripts (manuscript rendering, editable Word conversion, and figure generation) are checked against their exact pre-experiment copies in `pre_7b_snapshot/`; all other source references use the live artifact. The superseded pre-output cross-model and figure-test manifest entries are checked under their recorded amendment, without claiming the amended files match the superseded bytes. When using the reduced TREC package, first reconstruct its two omitted TREC sample files as directed by `NOTICE_RECONSTRUCT_TREC_SAMPLE.md`.

For a positive and negative integrity regression on a POSIX system, use a new output directory with approximately 46 MB available. It retains the extracted positive and deliberately altered negative-control data for inspection, and does not run model inference:

```bash
python3 posthoc_7b/test_snapshot_integrity.py --artifact . --out /path/to/new_snapshot_test
```

For independent detailed checks, use a Python environment with the dependencies in `requirements-verify.txt` and `transformers` for token checking; set `AUDIT=/path/to/new_7b_audit/posthoc` and run from this directory:

```bash
python3 verify_esci_selective_holdout.py --selection esci_selective_holdout_selection --run "$AUDIT/run_esci_7b" --analysis "$AUDIT/analysis_esci_7b" --model "$AUDIT/model/Qwen2.5-7B-Instruct"
python3 verify_esci_selective_holdout.py --selection esci_selective_holdout_selection --run "$AUDIT/run_esci_7b_specguided" --analysis "$AUDIT/analysis_esci_7b_specguided" --model "$AUDIT/model/Qwen2.5-7B-Instruct"
PYTHONPATH="$PWD:$AUDIT" python3 "$AUDIT/verify_wands_7b_posthoc.py" --selection wands_selective_execution_selection --run "$AUDIT/run_wands_7b" --model "$AUDIT/model/Qwen2.5-7B-Instruct"
PYTHONPATH="$PWD:$AUDIT" python3 "$AUDIT/verify_wands_7b_specguided.py" --selection wands_selective_execution_selection --run "$AUDIT/run_wands_7b_specguided" --model "$AUDIT/model/Qwen2.5-7B-Instruct"
python3 verify_trec_covid_transfer.py --selection trec_covid_transfer_selection --run "$AUDIT/run_trec_7b" --analysis "$AUDIT/trec_7b_analysis.json" --model "$AUDIT/model/Qwen2.5-7B-Instruct"
```

`verify_specguided_comparison.py` independently recomputes paired differences from raw traces; its exact invocations are in `run_specguided_7b.sh`. Query-cluster intervals are unadjusted and descriptive. Rights and access to any material before public distribution remain to be checked by the authors.

## Re-running generation

`run_all_7b.sh` and `run_specguided_7b.sh` preserve this machine's original paths for audit and are not portable runbooks. Download the official fixed revision in `MODEL_OFFICIAL_EXPECTED.json` to a separate folder named `Qwen2.5-7B-Instruct`, verify every file with its recorded hash, and use a compatible isolated vLLM environment. The frozen individual runners accept explicit paths and refuse to start if the GPU is in use. For example, after defining a new output path and setting the expected dependencies and `AUDIT`:

```bash
VLLM_USE_V1=1 PYTHONPATH="$PWD:$AUDIT" python3 "$AUDIT/run_esci_7b_specguided.py" \
  --model /path/to/Qwen2.5-7B-Instruct \
  --selection esci_selective_holdout_selection --out /path/to/new_esci_run
```

The WANDS and biomedical runners similarly accept `--model`, `--selection` and a non-existing `--out`. See the frozen protocols for the exact sample, settings and order. Model outputs and timing need not match across numerical environments; a successful parser check does not validate semantics.


## Readability-only revision of Figure 3

After the results were observed, the figure was restyled with larger labels and the common-complete pool count in each panel title. All nine stored contrasts, confidence intervals and absolute MAEs are unchanged. The revised plotting source checks the shared pool counts, data arithmetic, header overlap and bar/label overlap. The original figure and plotting script remain in the archived experiment for audit; the separate revised script reads the same `POSTHOC_7B_FIGURE_DATA.json`. Matplotlib 3.11.2 was used (it is an additional plotting dependency), and the PNG for the editable Word copy is rendered at 300 dpi.

```bash
python3 posthoc_7b/plot_readable_model_sensitivity.py --data POSTHOC_7B_FIGURE_DATA.json --out /path/to/new_figure_directory
python3 posthoc_7b/test_readable_model_sensitivity.py --data POSTHOC_7B_FIGURE_DATA.json
sha256sum -c posthoc_7b/FIGURE3_READABILITY_SHA256.txt
```

Figure data are fixed; exact PDF/PNG byte identity after regeneration is not guaranteed across software versions or metadata timestamps. The checksum command verifies the saved figure in the complete package, which includes the manuscript directory.
