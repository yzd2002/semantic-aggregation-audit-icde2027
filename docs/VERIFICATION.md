# Verification performed on 2026-11-06

All verification below ran on CPU, with new outputs outside the immutable
historical package. No model inference or publication was performed in this step.

## Asset and content screening

- Read all members of both main variants, seven supplements and the standalone
  source archive; also inspected the historical manuscript PDF. The recursive
  scan covered 2,672 entries including archive containers and 27 PDF documents.
- Verified ZIP member CRCs and recorded each member's SHA-256. Recursed into the
  nested TAR/gzip records; parsed PDF text and metadata with pypdf 6.1.1.
- No nominated credential-shaped pattern, private-cache path, symlink/unsafe
  archive path, or encrypted/unreadable member was flagged.
- In the three ACORD reconstruction candidates, no exact match occurred for
  2,027 nominated quoted-query / 160-character clause-prefix patterns derived
  from the private reference selections. The reference text was read only in
  memory and is not copied to the public repository or scan summary.
- The manuscript author line remains a placeholder. The PDF has no Author
  metadata. The example author-metadata strings are empty. No private completed
  author record is included in this source staging.

These are bounded heuristic checks, not an exhaustive privacy, OCR, arbitrary
password, short-fragment, linkability, or copyright review. Pseudonyms remain
reversible from public source dictionaries. The full TREC-text main variant was
inspected for completeness comparison but is excluded from the selected set.

## Executed CPU recomputations

| Check | Observed outcome |
| --- | --- |
| Closed selected asset set | All 10 actual files matched recorded SHA-256/size and ZIP CRCs; total 88,954,821 bytes. |
| Browsable source provenance | All 554 copied source/protocol/notice files matched their original member hashes; 3,268,466 bytes. |
| Syntax | 343 Python source files compiled in memory without writing bytecode. |
| Error cancellation | Analyzer and independent verifier passed on the saved 52 WANDS + 250 ESCI Qwen pools. |
| Count envelope | Portable recomputation and independent verification passed; all 604 pool–model rows and baseline summary fields matched. |
| Controlled-order replay | Both models' primary, supplementary and execution-audit results matched saved results at tolerance `1e-12`; independent checks passed. |
| Consensus AND/OR | Recomputed after explicitly mapping the delivered routing-result dependency; independent verifier passed. |
| Equal-label-budget routing | Recomputed exact uniform-subset reference and query-bootstrap quantities; independent verifier passed. |

The first consensus attempt failed before producing a result because the
historical analyzer expects the supplemental routing JSON under
`reproducibility/posthoc_routing_budget/`. Its delivered bytes live in the midpoint
supplement. The workspace tool now checks the original expected SHA-256 and
copies those bytes to that explicit location, recording the mapping in its
receipt. Original archives, frozen scripts and the failed-attempt log remain
unchanged. This is a portable assembly fix, not a change to scientific results.

The staging session did not newly download/reconstruct every upstream dataset,
replay every historical verification suite, rerun models, or establish a clean
GPU environment elsewhere. The exact ACORD/TREC reconstruction steps, tokenizer
hashes and historical successful checks remain in the delivered archives. Later
lost transfer/GLM volumes are not restored by these checks.
