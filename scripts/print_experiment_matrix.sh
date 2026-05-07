#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
INPUT_PKL="${INPUT_PKL:?Set INPUT_PKL=/path/to/cohort.pkl}"
IMAGE_DIR="${IMAGE_DIR:?Set IMAGE_DIR=/path/to/images}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/run_matrix}"
GROUP="${GROUP:-main}"

cmd=(
  "$PYTHON_BIN" -m eviosahs.run_matrix
  --python-bin "$PYTHON_BIN"
  --input-pkl "$INPUT_PKL"
  --image-dir "$IMAGE_DIR"
  --output-root "$OUTPUT_ROOT"
  --group "$GROUP"
  --mode print
)

if [[ -n "${QWEN_ROOT:-}" ]]; then
  cmd+=(--qwen-root "$QWEN_ROOT")
fi
if [[ -n "${INSTRUCTBLIP_MODEL_PATH:-}" ]]; then
  cmd+=(--instructblip-model-path "$INSTRUCTBLIP_MODEL_PATH")
fi
if [[ -n "${LLAVA_MODEL_PATH:-}" ]]; then
  cmd+=(--llava-model-path "$LLAVA_MODEL_PATH")
fi
if [[ -n "${LLAMA_MODEL_PATH:-}" ]]; then
  cmd+=(--llama-model-path "$LLAMA_MODEL_PATH")
fi

"${cmd[@]}"
