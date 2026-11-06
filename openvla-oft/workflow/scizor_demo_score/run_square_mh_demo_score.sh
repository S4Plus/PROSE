#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/workspace/liushuo/cupid/openvla-oft"
cd "${ROOT_DIR}"

CHECKPOINT_DIR="${CHECKPOINT_DIR:-${ROOT_DIR}/runs/square_mh_openvla_lora_full--10000_chkpt}"
SCORE_PATH="${SCORE_PATH:-${ROOT_DIR}/workflow/scizor_demo_score/input/square_mh/train_set_demo_scores.mmap}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/workflow/scizor_demo_score/output/square_mh_demo_score}"
EXP_NAME="${EXP_NAME:-square_mh_demo_score}"
RETRAIN_RUN_ROOT="${RETRAIN_RUN_ROOT:-${ROOT_DIR}/runs}"

mkdir -p "${OUTPUT_DIR}"

python "${ROOT_DIR}/workflow/scizor_demo_score/retrain_demo_score_openvla.py" \
  --score-path "${SCORE_PATH}" \
  --checkpoint-dir "${CHECKPOINT_DIR}" \
  --dataset-root /workspace/liushuo/cupid/datasets/lerobot/square_mh \
  --dataset-name square_mh \
  --output-dir "${OUTPUT_DIR}" \
  --exp-name "${EXP_NAME}" \
  --retrain-run-root "${RETRAIN_RUN_ROOT}" \
  "$@"
