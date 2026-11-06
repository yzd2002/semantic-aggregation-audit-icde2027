# Real-engine extension: compact CPU reproduction

The scientific payload contains the completed 36-configuration main matrix and all 16 policies in the separately frozen native-Boolean budget follow-up. The actual core archive has eight modules and 177 parts (90,066,144 bytes in total). Producer-side safe extraction and independent CPU mathematical replay passed; see [ONE_COMMAND_RESULT.json](../../../verification/core/ONE_COMMAND_RESULT.json) and the [verification scope](../../../verification/README.md). These producer-side checks do not establish an anonymous public download or replay.

Acquire the complete immutable repository commit (archive download or Git checkout) so its index and all parts are available locally. From this directory, replay with:

```bash
python3 -B replay_artifact.py --index artifact_index.json --parts-dir parts --out fresh_replay
```

The publisher supplies the final index and parts with the immutable commit; they are assembled separately from the source staging tree. `fresh_replay` must not already exist. The command verifies every part, joins the compressed streams, rejects unsafe archive entries, expands into a new directory, and replays the CPU mathematics. With `--parts-dir parts`, acquisition needs no per-part API calls. Anonymous requests for individual files may reach the GitHub API limit of about 60 requests per hour, so whole-commit acquisition is preferred.

Public part files default to 512 KiB for conservative GitHub App/Contents transport and always stay below the 8 MiB ceiling. Each part and joined stream has its own SHA-256, length and member inventory. The local joined streams are not additional upload assets. The optional network acquisition path uses a process-local direct anonymous HTTPS opener, default TLS validation, no token/cookie, no global proxy changes, and unique retained partial attempts. Signed redirect queries are not logged. An index marked `SYNTHETIC_CPU_FIXTURE_NOT_MODEL_EVIDENCE` is a software fixture, never scientific run evidence.

The mathematical replay needs Python 3.12, NumPy 2.2.6 and SciPy 1.16.2. It imports the unchanged original independent `analyze_study.py`, original `analyze_finite_audits.py` and `finite_audit.py`. Explicit read adapters supply compact actual-response, label, grouping and identity records. They do not change classification, CP inversion/selection, fixed-budget hypergeometric inversion or cost formulas. Results must match the same code's full-private-input analysis, not merely a copied summary. A runtime access guard forbids opening or listing the original scientific tree outside the expanded bundle, new output and portable tool directory; a leftover absolute dependency therefore fails even when replay is tested on the producer's server.

The compact projection retains actual returned raw completion strings, original canonical response bytes, durable-return input/output lengths and array hashes, original effective controls, cache/reset receipts, every physical partition, probe membership/position, CP decisions, native and strict labels, source-label vectors, completion and failure receipts. Unknown cached usage remains unknown. Native LOTUS output and additional strict-format completion are reported separately: `strict_count: null` does not imply that the native LOTUS query produced no result. Failed attempts stay in their own ledger and are not charged twice to successful configurations; no zero GPU-initialization cost is inferred from zero generation.

The later native-Boolean CP80/CP320 bounded follow-up has a separate family and output table. Its real design has four model/workload configurations and 16 physically independent policy jobs; it was prospectively frozen after an original case had been observed. It does not replace the original 36 cells, add independent domains, or certify a budget chosen after looking at outcomes. Its calibration uses new native-Boolean reference calls; source gold and the older complete S1/S2 are evaluation-only. Strict format remains a separate diagnostic. The one-command report records the original configuration count and the separate follow-up job count independently.

Full input text and token arrays are omitted from the public compact projection. The builder first verifies them in the untouched private tree and records exact source hashes. This CPU replay rechecks the public input-hash/length/native-ID/control chain and recomputes the mathematics. It does not retokenize omitted inputs or claim independent cleanroom GPU execution. Full original private evidence remains in the study root for the user's local delivery.

AG News's publisher card says **license unknown**. The separate `AGNEWS_FETCH_PLAN.json` gives the fixed revision, publisher and recorded mirror URLs, exact Parquet hash, feature ordering and source-text-free native formatter templates. Optional local input reconstruction is:

```bash
python3 -B fetch_agnews.py --plan expanded/AGNEWS_FETCH_PLAN.json --out fetched_local_inputs
```

`--from-file exact_pinned.parquet` checks already acquired bytes without network; `--mirror` selects the explicitly recorded public mirror. Complete 7,600-row records, labels and all four native request JSONLs must reproduce the frozen hashes exactly. No new license is assigned to news text or token IDs. Movie source acquisition uses the pinned Kaggle version-4 recipe and host-declared CC0 notice; see `RIGHTS_AND_NOTICES.md`.

Optional exact movie-input reconstruction uses `python3 -B fetch_movie_inputs.py --bundle expanded --out fetched_movies`. It explicitly maps the unchanged pinned movie fetch/preparation code to a new local output tree. Source-text-free formatter templates in `MOVIE_FETCH_PLAN.json` then regenerate both complete 8,192-row native request JSONLs; all eight records/labels/metadata/request files must match the frozen hashes. `--from-zip` can reuse an already acquired exact version-4 ZIP. No model is run and no original input is overwritten.

The original six scientific modules are reused through their immutable baseline-v2 release links in `BASELINE_V2_REFERENCES.json`; their bytes are not repackaged here. That baseline evidence does not certify this extension or its GLM-only v3 tokenizer guard. New v2 frozen sources, v3 sidecar/amendment, public byte-verification records and the applicable independent CPU checks are retained separately.

The separately disclosed post-hoc native-vector HG sensitivity has a seven-file source/results supplement at `../posthoc/`, including the actual producer-side CPU replay report. Its CPU entry is:

```bash
python3 -B replay_native_sensitivity.py --bundle fresh_replay/expanded --supplement ../posthoc/supplement --out fresh_native_HG_replay
```

This executes the unchanged writer's `analyze_native_audit_sensitivity.py` and original `finite_audit.py`, explicitly maps their input/output roots, verifies the supplied source/gold/policy fingerprints, and requires the recomputed JSON and CSV to equal the actual writer outputs. It uses the existing fixed SRS order with `m=min(1024,N)`, alpha and epsilon 0.05. It is a post-hoc readout over immutable native Boolean vectors for all 36 configurations, three existing plans and source gold/frozen native S1 targets; the canonical strict-format analysis stays separately identified. It adds no model calls and claims no future-reference or selected-budget joint coverage. The supplied supplement retains its source, complete actual outputs and CPU receipt separately from the core eight-module archive. This receipt also records producer-side replay, rather than an unperformed anonymous public acquisition.

Compact scientific archive membership is explicit. No virtual environment, model weights, runtime cache, credential/login material, unrelated job files, author/CMT fields, manuscript or user submission instructions are selected into that archive. The locked paper and its clean source are supplied separately under `paper/v3/` at the repository root. This tool does not publish, send messages or submit a paper.
