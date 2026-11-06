# [Experiment, Analysis, and Benchmark] Batch Fidelity Certificates for Semantic COUNT: Scope and Full Policy Cost

This artifact connects sampled batch-fidelity checks to complete semantic COUNT results and full policy costs, including calibration, fallback, and native cache counters. It provides actual execution evidence for 36 main configurations and 16 separately frozen budget-follow-up policies, together with CPU reconstruction of the reported decisions and bills.

[Current paper PDF](paper/current/ICDE2027_Semantic_COUNT.pdf) · [Editable source ZIP](paper/current/paper_source.zip) · [Active TeX](paper/current/source/draft/main.tex) · [Vector figures](paper/current/figures/) · [中文入口](START_HERE_zh.md) · [Asset map](ASSETS.md)

## Reproduce the results

Download and extract the repository archive for the exact published commit. This obtains all 177 parts in one request. Use Python 3.12, NumPy 2.2.6 and SciPy 1.16.2, then run from the repository root:

```bash
python3 -B repro/real_engine_20261106/public/replay_artifact.py --index repro/real_engine_20261106/public/artifact_index.json --parts-dir repro/real_engine_20261106/public/parts --out fresh_core
python3 -B repro/real_engine_20261106/public/replay_native_sensitivity.py --bundle fresh_core/expanded --supplement repro/real_engine_20261106/posthoc/supplement --out fresh_native_HG
python3 -B repro/real_engine_20261106/stopping_audit/analyze_observed_rejections.py --root fresh_core/expanded --main-summary fresh_core/cpu_replay/engine/summary.json --budget-summary fresh_core/cpu_replay/separate_author_budget_analysis/summary.json --output fresh_stopping --tex-output fresh_stopping/tex
python3 -B repro/real_engine_20261106/readability_evidence/recompute_frozen_evidence.py --root fresh_core/expanded --out fresh_readability_evidence
```

The first command validates all part, compressed-stream and member hashes, safely expands eight modules, and recomputes the main and follow-up mathematics. The next commands reproduce the post-hoc native-vector HG readout, the 154 observed candidate prefixes (106 original + 48 follow-up), and the explanatory stopped-chunk/cost pair. Each pair of stopping/readability CSVs must equal its supplied CSV bytes. These replays use the recorded responses and labels on CPU.

The [current scientific index](repro/real_engine_20261106/public/artifact_index.json) gives actual content-addressed Git blob URLs and SHA-256 values. The [readability evidence](repro/real_engine_20261106/readability_evidence/) explains how complete returned chunks are scored and billed. [Producer-side verification](verification/) and [paper source rebuild evidence](verification/readability_paper_source/) preserve their measured scopes; final anonymous acquisition is verified separately against the completed immutable commit.

## Rebuild the paper

`bash paper/current/source/build.sh` compiles the editable source with ordinary pdflatex and the included vector PDFs. The ten-author block and current captions are part of the frozen active TeX. The source ZIP has actually been unpacked, its four figures redrawn, and its TeX compiled; all 14 page texts/renders matched the frozen PDF, with the main body ending on page 12.

Optional `python3 -B paper/current/source/redraw_figures.py` reads the supplied aggregate CSVs. Generated table exports go to `regenerated/`, preserving the frozen editorial TeX. Plotting dependencies and source hashes are included in the clean source.

## Evidence scope and rights

Six workloads cover two text domains and two models. The follow-up contains four model/workload configurations and 16 physical policy executions; it was frozen after an initial main case had been observed. Native delivery, strict output format, named-reference fidelity, supplied-label COUNT, and complete calibration-plus-execution cost have separate readouts. The post-hoc HG analysis conditions on existing immutable vectors. Stopped prefixes describe issued comparisons and retain unreached candidates as unobserved.

The compact modules retain raw model completions, canonical responses, effective settings, partitions, labels, probes and token-length/hash receipts. Pinned acquisition recipes support separate exact source reconstruction. Full source news/review text, token arrays and model weights are omitted. AG News has unknown license as declared by its publisher; movie CC0 is recorded as the host declaration. See [rights/notices](repro/real_engine_20261106/public/RIGHTS_AND_NOTICES.md) and [model provenance](repro/real_engine_20261106/model_sources/MODEL_ACQUISITION.md). Built with glm-4.

## Current and historical indexes

Use [PUBLICATION_FILES_SHA256.json](PUBLICATION_FILES_SHA256.json) for this repository edition and `repro/real_engine_20261106/public/artifact_index.json` for its eight scientific modules. The historical root `asset_index.json` and older SHA/checksum lists describe v2 files. They remain historical provenance; their hashes do not validate the new README or current paper.

The [immutable v2 release](https://github.com/yzd2002/semantic-aggregation-audit-icde2027/releases/tag/v2026.11.06-final) supplies six legacy scientific archives. [ASSETS.md](ASSETS.md) distinguishes those archives from the current eight-module payload. The current paper and source are under `paper/current/`.
