#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
RESULTS_DIR="${RESULTS_DIR:-results}"
FIGURE_DIR="${FIGURE_DIR:-results/figures}"

"$PYTHON_BIN" -m eviosahs.plot_paper_figures \
  --results-dir "$RESULTS_DIR" \
  --output-dir "$FIGURE_DIR"
