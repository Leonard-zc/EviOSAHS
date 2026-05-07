#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
INPUT_PKL="${INPUT_PKL:?Set INPUT_PKL=/path/to/cohort.pkl}"
IMAGE_DIR="${IMAGE_DIR:?Set IMAGE_DIR=/path/to/images}"
CONTROL_ROOT="${CONTROL_ROOT:-outputs/image_controls}"

"$PYTHON_BIN" -m eviosahs.visual_eval_v6_make_controls \
  --input-pkl "$INPUT_PKL" \
  --image-dir "$IMAGE_DIR" \
  --output-root "$CONTROL_ROOT" \
  --mode all \
  "$@"
