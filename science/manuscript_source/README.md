# Current ICDE 2027 EAB editable source

Title: [Experiment, Analysis, and Benchmark] Auditing LLM-Based Semantic Aggregation over Fixed Candidate Pools.

Open `01_manuscript/main.tex` as the main file in Overleaf, or run `bash build_pdf.sh` from that directory with a complete LaTeX installation. The rendered source, included IEEEtran class and figure PDFs are sufficient to compile; no model, original dataset or API is needed. The artifact/file prefix IntentSat2S is retained for continuity.

Author details are intentionally pending. Use `02_reproducibility/prepare_submission_authors.py` and the guide in `03_submission/03_作者信息最后填写.md` after obtaining real author information. Recheck the body-page limit and layout after author entry. Do not submit the placeholder version to a single-blind conference.

The package includes the template, renderer, current Word converter, field generator, both figure generators, vector/editable/preview figures and the new presentation data. Figure, table and equation references in the editing copy are dynamic; captions are retained explicitly because the converter does not reliably preserve IEEE table captions. The Word file is for editing, not the submission layout.

To rerender statistics rather than compile the already rendered source, put the main artifact's `reproducibility/` contents under `02_reproducibility/`, keeping the newer renderer, `review_revision/` and `posthoc_routing_budget/` supplied here. The review-audit supplement contains the underlying new experiment and CPU analyses. Full model weights and original upstream datasets are obtained separately from their fixed sources; they are not part of a paper-source archive.

The figure supplement documents the Matplotlib environment and complete regeneration commands. `review_revision/controlled_qwen_analysis.json` and `controlled_smol_analysis.json` regenerate the new controlled-order plot without inference. The current `main.tex` records which figures are actually used; other supplied figures are audit alternatives.

Keep PDF, Word and CMT title/abstract/disclosure synchronized. `generate_submission_fields.py` writes the five copyable TXT files and the CMT guide from the current rendered manuscript. This is an EAB preparation/review version, not an acceptance or camera-ready claim. The official requirements are:

- https://icde2027.github.io/cf-research-papers
- https://icde2027.github.io/submission-guidelines.html

EAB requires all necessary reproduction artifacts at a publicly accessible URL. The local candidate files do not themselves satisfy that publication step.
