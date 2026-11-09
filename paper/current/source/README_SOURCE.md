# Frozen editable paper source

Start at `draft/main.tex`; the supplied eight-author block is `draft/authors.tex`. This clean snapshot contains the recursively active TeX/refs/class, the actual four vector figures (PDF/SVG), the original pure plotting sources and exact CSV inputs. It excludes old metadata, historical drafts, previews, auxiliary files and logs.

Run `bash build.sh` with an ordinary TeX installation providing pdflatex, amsmath/amssymb, graphicx, booktabs, array, url, xcolor and hyperref. IEEEtran.cls is included unchanged with its upstream notices. Compilation uses the supplied vector PDFs and needs no scientific run, model, private data root or plotting package. `PDFLATEX` can explicitly select another compatible pdflatex executable.

To redraw the four figures, use Python 3.12 and `requirements_plotting.txt`, then run `python3 -B redraw_figures.py`. This entry calls the unchanged pure plot functions and schematic source. It directs generated table exports to `regenerated/`, preserving the frozen editorial captions and author block in `draft/`. All computations use the supplied aggregate CSVs; no model call, new experiment or raw news/review text is involved. The separate public readability-evidence tool recomputes the two exact trace/cost CSVs from the expanded scientific artifact.

`MANUSCRIPT_CONTENT.json` records this freeze's title, abstract, eight authors and page roles. `SOURCE_MANIFEST.json` binds the included file bytes. PDF metadata and object ordering may differ after compilation; the separate delivery rebuild receipt records actual unpacking, figure redraw, TeX compilation, and page-text/render comparison with the frozen PDF. No blanket license is assigned to third-party data or model assets.
