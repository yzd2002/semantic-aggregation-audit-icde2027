# Review-audit supplement for the ICDE 2027 EAB manuscript

This archive accompanies *Auditing LLM-Based Semantic Aggregation over Fixed Candidate Pools*. It adds the current audit to the historical main reproduction artifact. It is a local release candidate until the authors publish the verified files and complete their content/rights review. It contains no model weights or full upstream datasets.

Keep this archive, the main artifact, and the other supplements listed in the top-level English reproduction guide together. One main variant is sufficient; the variant without TREC text requires the documented reconstruction before TREC analyses. Extract this supplement into a directory with sufficient space; the complete wrapped request records alone occupy about 173 MiB uncompressed. Do not replace a previous result directory.

## Contents

- `controlled_order/`: fixed 128-pool ESCI selection, exposed WANDS52 inputs, full randomized schedule, wrapped requests for both models, frozen code and analysis, append-only amendments, failed setup attempts, controls, actual responses, and independent verification. Follow its final reproduction guide for portable CPU re-analysis and optional fresh inference. Setup attempts that produced no experimental judgments remain separate from the completed runs.
- `consensus/`: post-hoc AND/OR comparisons, class-composition attribution, exact expected uniform-checking references and optional oracle-checker diagnostic. These reuse 302 distinct pools for two models: 604 pool–model cases, not 604 independent pools.
- `envelope/`: reference-dependent attainable-count diagnostic, frozen calculation and independent position/DP verification, and a portable launcher. See `README_PORTABLE.txt`; it supersedes machine-specific paths in the historical README.
- `manuscript_revision/`: verified text blocks and figure input for the current manuscript. These support presentation; the underlying response records remain in the experimental folders.
- `source_notices/`: the relevant ESCI/WANDS source notices already included in the main artifact. New scripts and outputs require the authors' own publication review.

## Check the supplied files

From the extracted `review_audit/` directory, run:

```sh
sha256sum -c SHA256SUMS.txt
```

The checksum manifest identifies every supplied file. Internal experiment freezes have separate meanings and must not be rewritten: they establish the selected inputs, code versions, amendments and observed runs. Absolute historical source paths are provenance; portable launchers map the explicitly supplied files while checking their original hashes.

## Recompute consensus and count-envelope analyses without inference

Use Python 3.12 with NumPy 2.2.6 for the consensus analysis. The count-envelope calculation uses the Python standard library. Do not use optimized Python (`-O`), because the independent checks use assertions. Set the variables below to real extracted directories; no original server paths are required.

```sh
INTENTSAT_BASE=/path/to/main-artifact/reproducibility
INTENTSAT_AUDIT=/path/to/review_audit
INTENTSAT_NEW=/path/to/existing-parent/new-consensus-audit.json

OPENBLAS_NUM_THREADS=1 python3 -B "$INTENTSAT_AUDIT/consensus/audit_consensus_baselines.py" \
  --artifact "$INTENTSAT_BASE" --out "$INTENTSAT_NEW"
OPENBLAS_NUM_THREADS=1 python3 -B "$INTENTSAT_AUDIT/consensus/verify_consensus_audit.py" \
  --artifact "$INTENTSAT_BASE" --result "$INTENTSAT_NEW"

python3 -B "$INTENTSAT_AUDIT/envelope/portable_envelope.py" \
  --artifact "$INTENTSAT_BASE" \
  --consensus-audit "$INTENTSAT_AUDIT/consensus/consensus_audit.json" \
  --output /path/to/existing-parent/new-envelope-results
```

The consensus result is a new file; the envelope result is a new directory. The envelope check deliberately uses the delivered, hash-identified consensus JSON. A newly computed consensus JSON can differ in path metadata even when all numeric results match. The portable envelope guide also documents read-only verification of moved saved results.

## Interpretation

AND/OR preserve both keyed passes' agreements; they are reported on every setting, without choosing a winner using test labels. The actual sparse checker and the cached full-pointwise checker are different recorded executions on WANDS/Qwen; their routed MAEs are 0.120 and 0.118. The envelope is a diagnostic conditional on the known reference count, not an online selection rule or a bound on plans that reconsider agreeing labels. All negative and adverse settings are retained.

The controlled experiment is new inference on the same GPU and inherited software installation. Its initial Gloo setup failed because PCI enumeration was restricted. The documented process-local singleton-NCCL adapter changes communication setup for a single rank; it is not an unmodified clean installation. Both models use the same amended runtime, and grammar controls precede data inference. This evidence supports saved-run re-analysis, not another machine's installation, future-population guarantees or standalone latency comparisons.

To rerender the current paper, combine the main artifact's analysis data with `manuscript_revision/` copied to `02_reproducibility/review_revision/`, and use the latest renderer supplied with the standalone source package. The already rendered `main.tex` and bundled figures compile independently of the data and models. The top-level CMT copy text must remain synchronized with the final author-filled PDF.
