#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/run_matrix}"
RESULTS_DIR="${RESULTS_DIR:-results/visual_audit}"
F5_DIR="${F5_DIR:-$OUTPUT_ROOT/main/F5_qwen_two_stage_final_only_clinical}"

mkdir -p "$RESULTS_DIR/rows" "$RESULTS_DIR/quality" "$RESULTS_DIR/proxy"

"$PYTHON_BIN" -m eviosahs.visual_eval_prepare \
  --visual-jsonl "$F5_DIR/visual.jsonl" \
  --output-jsonl "$RESULTS_DIR/rows/f5_visual_rows.jsonl" \
  --output-csv "$RESULTS_DIR/rows/f5_visual_rows.csv"

"$PYTHON_BIN" -m eviosahs.visual_eval_quality \
  --rows-jsonl "$RESULTS_DIR/rows/f5_visual_rows.jsonl" \
  --output-csv "$RESULTS_DIR/quality/f5_quality_by_question.csv" \
  --output-json "$RESULTS_DIR/quality/f5_quality_summary.json"

"$PYTHON_BIN" -m eviosahs.visual_eval_proxy \
  --rows-jsonl "$RESULTS_DIR/rows/f5_visual_rows.jsonl" \
  --output-dir "$RESULTS_DIR/proxy"
