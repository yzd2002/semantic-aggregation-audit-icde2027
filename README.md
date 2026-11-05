# Semantic Aggregation Audit — ICDE 2027 EAB Artifacts

Reproduction entry point for the historical ICDE 2027 EAB review manuscript,
`[Experiment, Analysis, and Benchmark] Auditing LLM-Based Semantic Aggregation
over Fixed Candidate Pools`. The execution harness retains the name IntentSat-2S.

**Published snapshot:** [v2026.11.06-recovered](https://github.com/yzd2002/semantic-aggregation-audit-icde2027/releases/tag/v2026.11.06-recovered).
It matches the historical review manuscript retained by 2026-11-02, with renewed
archive checks and CPU replay. All ten selected files have been downloaded without credentials or cookies and
matched against their exact SHA-256 digests and byte counts. The public asset
API was used for this check; [verification receipts](verification/PUBLIC_DOWNLOADS.json)
record each route and result.

The [prospective follow-up analysis plan](prospective_20261106/README.md) records the new nine-case primary family before new GPU authorization. It does not imply observed effects or completed experiments.

The recovered context/decision extensions and new model replications are being
prepared as separately versioned additions. They are not yet claimed by this
snapshot. See [coverage and limits](docs/COVERAGE.md).

The historical study is an empirical audit of completion, semantic quality,
order/replay variation, checking cost, and conditional diagnostics over fixed
candidate pools. Aggregate ratio accuracy can hide candidate-label mistakes.
Saved-trace verification is distinct from rerunning model inference.

## Get the artifacts

Use the actual release links in [ASSETS.md](ASSETS.md). One main archive plus **all seven
supplements** is necessary for the historical manuscript. The selected main
variant omits sampled TREC article titles/abstracts and provides fixed-source
reconstruction. The full internal research workspace is not a release artifact.
Model weights and complete upstream datasets are obtained separately.

The repository contains readable scientific source in `science/`, reproduction
guides, source notices and hashes. Large fixed archives are intended for a
versioned GitHub Release, rather than Git history. GitHub's automatically
generated source ZIP is not a substitute for the separately attached artifacts.

```sh
# Run from this repository; choose new output directories.
python3 -B tools/download_assets.py --out /path/to/downloaded-assets
python3 -B tools/verify_assets.py --asset-dir /path/to/downloaded-assets
python3 -B tools/prepare_workspace.py \
  --asset-dir /path/to/downloaded-assets --out /path/to/new-workspace
```

The download, verification and extraction tools use only Python's standard library. The workspace tool rejects an
existing output directory and unsafe archive members. It does not load models,
use a GPU, contact a model service, or download source data.

## Recompute on CPU first

Use Python 3.12. [CPU_REPRODUCTION.md](docs/CPU_REPRODUCTION.md) supplies explicit
commands for the standard-library error-cancellation diagnostic, controlled-order
replay, consensus references, and count-envelope checks. These read the delivered
saved traces. Each result is written to a new path. ACORD and TREC are rebuilt
from the exact upstream revisions in [DATA.md](docs/DATA.md), without distributing
the omitted source text here. The historical detailed runbook is preserved as
[HISTORICAL_REPRODUCTION_GUIDE.md](docs/HISTORICAL_REPRODUCTION_GUIDE.md).

Environment requirements differ by historical analysis: the broad verification
suite and midpoint/lexical work record NumPy 1.26.4; controlled-order, consensus,
and routing work record NumPy 2.2.6. Keep those environments separate. Historical
GPU dependency conflicts and runtime amendments are disclosed in the artifacts.
No complete clean-machine GPU rerun is claimed.

## Manuscript and source provenance

The manuscript PDF and standalone LaTeX archive are identified by exact hashes
in `asset_index.json`; they retain their historical review status. The saved
manuscript has author placeholders. It is not a final single-blind submission.
Author identities, acceptance decisions and a DOI remain unset. The artifact URLs identify the actual published GitHub release.
`science/SOURCE_PROVENANCE.json` maps copied source bytes to archive members.
Copied historical code is unchanged; the wrapper tools are new publication aids.

## Rights and interpretation

Upstream license texts and notices are retained under [LICENSES](LICENSES/README.md)
and [NOTICE](NOTICE). No project-wide open-source license is fabricated. The
project-authored code's license status is explicit in
[PROJECT_LICENSE_STATUS.md](LICENSES/PROJECT_LICENSE_STATUS.md). Public access
does not change third-party rights or turn linkable ACORD pseudonyms into
anonymized data. The historical disclosure of AI assistance remains in the paper.

According to the [official ICDE 2027 CFP](https://icde2027.github.io/cf-research-papers),
EAB submissions must supply all necessary reproduction artifacts through an
openly accessible standard service. This public repository and its versioned release supply the selected snapshot.
Publication and verification are documented in [RELEASE_PLAN.md](docs/RELEASE_PLAN.md);
coverage claims must match the submitted manuscript revision.
