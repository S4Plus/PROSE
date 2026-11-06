#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/workspace/liushuo/cupid/openvla-oft"
cd "${ROOT_DIR}"

CHECKPOINT_DIR="${CHECKPOINT_DIR:-${ROOT_DIR}/runs/square_mh_openvla_lora_full--10000_chkpt}"
SCORE_PKL="${SCORE_PKL:-/workspace/liushuo/SCIZOR/outputs/act_scizor/square_mh/scizor_joint_filter.pkl}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/workflow/scizor_demo_score/output/square_mh_scizor}"
EXP_NAME="${EXP_NAME:-square_mh_scizor}"
RETRAIN_RUN_ROOT="${RETRAIN_RUN_ROOT:-${ROOT_DIR}/runs}"

mkdir -p "${OUTPUT_DIR}"

python "${ROOT_DIR}/workflow/scizor_demo_score/retrain_scizor_openvla.py" \
  --score-pkl "${SCORE_PKL}" \
  --checkpoint-dir "${CHECKPOINT_DIR}" \
  --dataset-root /workspace/liushuo/cupid/datasets/lerobot/square_mh \
  --dataset-name square_mh \
  --output-dir "${OUTPUT_DIR}" \
  --exp-name "${EXP_NAME}" \
  --retrain-run-root "${RETRAIN_RUN_ROOT}" \
  "$@"
