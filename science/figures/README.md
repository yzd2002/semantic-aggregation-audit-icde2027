# Current original figures and reproduction

This supplement replaces the earlier figure supplement. It supplies publication PDF vectors, editable SVGs, PNG previews, data/provenance checks and two self-contained generators. The manuscript selects five figures; the two earlier alternative plots remain available for audit. No figure asserts that visual refinement establishes scientific novelty or acceptance.

Use Matplotlib 3.11.2 with DejaVu Sans for the recorded rendering. The audit environment additionally uses PyMuPDF and Pillow; those are not required merely to compile the already rendered paper. Fixed canvases preserve the intended type size: the redesigned plots are 7.10 inches wide with ordinary labels at least 8.2 pt nominal, about 8.08 pt at the manuscript's 0.98 text-width scale. Match rendering libraries and fonts if byte-identical images are required.

Verify `SHA256SUMS.txt` from this directory before using the files. `figures/FIGURE_MANIFEST.json` covers all current figure assets and records both generation paths. `figures/CONTROLLED_FIGURE_MANIFEST.json` separately identifies the later controlled-order figure and its inputs. The historical generator recreates six figures; the separate controlled generator recreates the seventh.

Regenerate the six historical-result/overview figures using the extracted main artifact and two-order supplement:

```sh
MPLCONFIGDIR=/path/to/writable/matplotlib-cache python3 -B make_icde_style_figures.py \
  --data-dir /path/to/main-artifact/reproducibility \
  --routing-result /path/to/midpoint_ratio_supplement/routing_budget_result.json \
  --out /path/to/new-six-figure-directory
```

Regenerate the controlled-order figure entirely from the two supplied completed analyses:

```sh
MPLCONFIGDIR=/path/to/writable/matplotlib-cache python3 -B make_controlled_order_figure.py \
  --qwen-analysis controlled_qwen_analysis.json \
  --smol-analysis controlled_smol_analysis.json \
  --out /path/to/new-controlled-figure-directory
```

The new plot shows excess row-order variation over identical-condition replay in percentage points, at both output-key orders. These are stability distances, not semantic accuracy. Negative values and intervals crossing zero are retained. The generator checks every plotted estimate/interval against the supplied analyses, per-pool mean arithmetic, sample sizes and source hashes. The experimental supplement retains the raw responses and the independent bootstrap verification behind those analyses.

The overview's four-candidate panel is an explicitly labeled illustration. Result plots use recorded observations, without selecting favorable settings or changing scales to conceal adverse intervals. All main-figure content and layouts are original. The design study consulted MICRO, identified by the ICDE 2026 awards page as Best Research Paper, for information organization rather than copying its figures or data:

- https://icde2026.github.io/awards.html
- https://adalabucsd.github.io/papers/2026_MICRO_ICDE.pdf

The main artifact's older figure files and descriptions are historical; use this supplement and the current standalone LaTeX package together.
