# Auditing Reference Fidelity and Checking Costs in Semantic COUNT

Verified public artifacts for **[Experiment, Analysis, and Benchmark] Auditing Reference Fidelity and Checking Costs in Semantic COUNT** (ICDE 2027 EAB).

Current version: [v2026.11.06-final](https://github.com/yzd2002/semantic-aggregation-audit-icde2027/releases/tag/v2026.11.06-final). All eight scientific/manuscript assets were downloaded without authentication or cookies and matched their SHA-256 digests; all315 files from the public scientific tag also matched. Complete CPU replay using the actual public downloads passed. See [download receipts](verification/PUBLIC_DOWNLOAD_VERIFICATION.json), [CPU replay](verification/PUBLIC_SOURCE_CPU_REPLAY.json), and [asset manifest](asset_index.json).

The manuscript is the final scientific v2 with author placeholders. This is an authors-pending submission draft, not an acceptance announcement or completed CMT submission. The historical recovered release remains separate. 请先读 [中文 START_HERE](START_HERE_zh.md)。

The final-paper artifacts are modular: fresh DBLP–ACM/BoolQ context experiments in three model families; fresh SNLI in the same families; new real transfer checking-cost runs; the original DEV-only 36-policy/calibration dependency; separate mathematical/agent audits of the nine fresh context cases; and the real-transfer independent mathematical audit. Complete selected frozen files, actual durable returns, analyses, controls, failed attempts and notices accompany each applicable module. Model weights, caches, credentials and unused historical pilots are excluded.

Use Python 3.12.11 with [the recorded CPU versions](requirements-cpu.txt). CPU outcome replay requires NumPy 2.2.6 and PyArrow 22.0.0; deterministic decision prompt reconstruction additionally uses the listed SciPy/Transformers/tokenizer packages. Do not change a working environment merely to run the replay; a separate environment is suitable when installation is needed.

The immutable `v2026.11.06-final` tag binds the scientific source snapshot. Its source tree may contain only `asset_index.template.json`: the verified `asset_index.json` is generated after actual publication and anonymous verification, then attached to that same GitHub release. **First download `asset_index.json` from the matching release's attachments into this tool directory**, then run:

```bash
python3 -B artifact_cli.py download --index asset_index.json --archives-dir downloaded
python3 -B artifact_cli.py verify --index asset_index.json --archives-dir downloaded
python3 -B artifact_cli.py extract --index asset_index.json --archives-dir downloaded --out expanded
python3 -B cpu_replay.py --bundle-root expanded --out replay_results
```

Both `expanded` and `replay_results` must be new directories. Extraction verifies archive hashes/sizes, rejects absolute paths, traversal, symbolic/hard links, special files and duplicate members, then verifies every expanded scientific file. Replay rechecks those inventories, hides GPU devices, uses offline inputs and writes only a new output tree. Frozen sources are never silently edited. The SNLI independent-audit adapter explicitly maps the original certificate's recorded absolute context path to the expanded context module.

For a local candidate, the same `verify`/`extract` commands accept `asset_index.template.json` and locally supplied archives. The `download` command deliberately refuses an unpublished template.

The main branch may additionally provide a convenience copy of the actual verified index and receipts. That convenience update does not rewrite the immutable scientific tag. A missing local `asset_index.json` means the release attachment must be obtained first; it does not justify treating the unpublished template as verified public metadata.

On this server, add `--direct` to the download command if its inherited proxy prevents anonymous access. This creates a process-local proxy-free opener and retains TLS certificate verification; it does not change system proxy settings. If the browser asset URL returns HTTP 403 or disconnects, a publisher-provided `public_asset_api_url` permits an anonymous GitHub asset-API fallback with `Accept: application/octet-stream`. No token or cookie is used. Every failed partial download is retained under a unique attempt name; signed redirect queries are never logged.

Raw outcomes are the original preserved actual model returns. CPU replay recomputes the paper's statistics and costs; it does not generate new model judgments. It is not a clean-machine GPU replication, outside human review, population-coverage certificate, SOTA claim or acceptance guarantee. Finite selected-target bootstrap intervals remain approximate; legacy 98.75% flags do not replace the final nine-case primary criterion.

Read [RIGHTS_AND_NOTICES.md](RIGHTS_AND_NOTICES.md) before reuse. Dataset licenses differ: DBLP benchmark CC BY 4.0 provenance; BoolQ CC BY-SA 3.0; SNLI CC BY-SA 4.0. Upstream model terms are retained separately. This repository makes no blanket MIT grant for project or third-party material.
