#!/usr/bin/env bash
set -euo pipefail

ROOT="/workspace/liushuo/cupid/openpi"
RAW_DIR="/workspace/liushuo/data/right_box_left_object_right_lid/right_box_left_object_right_lid"
DATASET_NAME="right_box_left_object_right_lid"
LEROBOT_ROOT="/workspace/liushuo/data/dataset"
DATASET_DIR="${LEROBOT_ROOT}/${DATASET_NAME}"
CONFIG_NAME="pi05_right_box_left_object_right_lid_lora"
EXP_NAME="${EXP_NAME:-right_box_left_object_right_lid_pi05_30k}"
LOG_DIR="${ROOT}/logs/right_box_left_object_right_lid_pi05"
LOG_FILE="${LOG_DIR}/run.log"
GPU="${GPU:-0}"
UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache-openpi}"
WANDB_MODE="${WANDB_MODE:-offline}"
TASK_PROMPT="${TASK_PROMPT:-Use the right arm to stage the box and lid, the left arm to place the object, then close the lid.}"

mkdir -p "${LOG_DIR}" "${LEROBOT_ROOT}"
cd "${ROOT}"

{
  echo "[start] $(date '+%F %T') config=${CONFIG_NAME} gpu=${GPU}"
  echo "[convert] raw=${RAW_DIR} out=${DATASET_DIR}"
  UV_CACHE_DIR="${UV_CACHE_DIR}" \
    uv run python franka/convert_dual_arm_mp4_joint_state_to_lerobot.py \
      --raw-dir "${RAW_DIR}" \
      --repo-id "${DATASET_DIR}" \
      --lerobot-home "${DATASET_DIR}" \
      --task "${TASK_PROMPT}" \
      --overwrite

  echo "[norm] config=${CONFIG_NAME}"
  CUDA_VISIBLE_DEVICES="${GPU}" \
  UV_CACHE_DIR="${UV_CACHE_DIR}" \
    uv run scripts/compute_norm_stats.py \
      --config-name "${CONFIG_NAME}"

  echo "[train] config=${CONFIG_NAME} exp=${EXP_NAME}"
  CUDA_VISIBLE_DEVICES="${GPU}" \
  XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}" \
  UV_CACHE_DIR="${UV_CACHE_DIR}" \
  WANDB_MODE="${WANDB_MODE}" \
    uv run scripts/train.py "${CONFIG_NAME}" \
      --exp-name "${EXP_NAME}" \
      --overwrite

  echo "[done] $(date '+%F %T')"
} > "${LOG_FILE}" 2>&1
