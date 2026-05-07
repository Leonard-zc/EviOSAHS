#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
INPUT_PKL="${INPUT_PKL:?Set INPUT_PKL=/path/to/cohort.pkl}"
IMAGE_DIR="${IMAGE_DIR:?Set IMAGE_DIR=/path/to/images}"
QWEN_ROOT="${QWEN_ROOT:?Set QWEN_ROOT=/path/to/qwen_weights}"
LLAVA_MODEL_PATH="${LLAVA_MODEL_PATH:?Set LLAVA_MODEL_PATH=/path/to/llava}"
LLAMA_MODEL_PATH="${LLAMA_MODEL_PATH:?Set LLAMA_MODEL_PATH=/path/to/llama}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/run_matrix}"
VL_DEVICE="${VL_DEVICE:-cuda:0}"
REASONER_DEVICE="${REASONER_DEVICE:-cuda:1}"

cmd=(
  "$PYTHON_BIN" -m eviosahs.run_matrix
  --python-bin "$PYTHON_BIN"
  --input-pkl "$INPUT_PKL"
  --image-dir "$IMAGE_DIR"
  --qwen-root "$QWEN_ROOT"
  --llava-model-path "$LLAVA_MODEL_PATH"
  --llama-model-path "$LLAMA_MODEL_PATH"
  --output-root "$OUTPUT_ROOT"
  --vl-device "$VL_DEVICE"
  --reasoner-device "$REASONER_DEVICE"
  --group backbone
  --mode execute
)

if [[ "${RESUME:-0}" == "1" ]]; then
  cmd+=(--resume)
fi
if [[ "${OVERWRITE:-0}" == "1" ]]; then
  cmd+=(--overwrite)
fi

"${cmd[@]}"
