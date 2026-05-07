#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
INPUT_PKL="${INPUT_PKL:?Set INPUT_PKL=/path/to/cohort.pkl}"
IMAGE_DIR="${IMAGE_DIR:?Set IMAGE_DIR=/path/to/images}"
QWEN_ROOT="${QWEN_ROOT:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/run_matrix}"
DEVICE="${DEVICE:-cuda:0}"
VL_DEVICE="${VL_DEVICE:-cuda:0}"
REASONER_DEVICE="${REASONER_DEVICE:-cuda:1}"

cmd=(
  "$PYTHON_BIN" -m eviosahs.run_matrix
  --python-bin "$PYTHON_BIN"
  --input-pkl "$INPUT_PKL"
  --image-dir "$IMAGE_DIR"
  --output-root "$OUTPUT_ROOT"
  --device "$DEVICE"
  --vl-device "$VL_DEVICE"
  --reasoner-device "$REASONER_DEVICE"
  --experiment-id T5_clinical_only_qwen_text
  --experiment-id F1_instructblip_direct
  --experiment-id F2_llava16_direct
  --experiment-id F3_qwen_direct
  --experiment-id F4_qwen_two_stage_naive
  --experiment-id F5_qwen_two_stage_final_only_clinical
  --experiment-id F6_ours_eviosahs_balanced
  --mode execute
)

if [[ -n "$QWEN_ROOT" ]]; then
  cmd+=(--qwen-root "$QWEN_ROOT")
fi
if [[ -n "${INSTRUCTBLIP_MODEL_PATH:-}" ]]; then
  cmd+=(--instructblip-model-path "$INSTRUCTBLIP_MODEL_PATH")
fi
if [[ -n "${LLAVA_MODEL_PATH:-}" ]]; then
  cmd+=(--llava-model-path "$LLAVA_MODEL_PATH")
fi
if [[ "${RESUME:-0}" == "1" ]]; then
  cmd+=(--resume)
fi
if [[ "${OVERWRITE:-0}" == "1" ]]; then
  cmd+=(--overwrite)
fi

"${cmd[@]}"
