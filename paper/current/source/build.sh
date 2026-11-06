#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/draft"
intent_pdflatex="${PDFLATEX:-pdflatex}"
command -v "$intent_pdflatex" >/dev/null 2>&1 || { echo "Install a standard TeX distribution with pdflatex and the packages listed in README_SOURCE.md." >&2; exit 1; }
"$intent_pdflatex" -interaction=nonstopmode -halt-on-error main.tex
"$intent_pdflatex" -interaction=nonstopmode -halt-on-error main.tex
