#!/usr/bin/env bash
set -euo pipefail

OPENPI_ROOT="/workspace/liushuo/cupid/openpi"
ENTRY_SCRIPT="${OPENPI_ROOT}/myscripts/run_train_openpi05_from_act_filter.sh"

DATASET_NAME="${DATASET_NAME:-all}"
OUTPUT_BASE="${OUTPUT_BASE:-${OPENPI_ROOT}/myscripts/output/cupid_quality_openpi05}"

run_one() {
  local dataset_name="$1"
  local dataset_dir="$2"
  local train_config="$3"
  local act_filter_dir="$4"

  echo "[dataset] ${dataset_name}"
  DATASET_DIR="${dataset_dir}" \
  TRAIN_CONFIG="${train_config}" \
  ACT_FILTER_MODE="cupid_quality" \
  ACT_FILTER_DIR="${act_filter_dir}" \
  OUTPUT_BASE="${OUTPUT_BASE}" \
  RUN_NAME="${dataset_name}_cupid_quality_openpi05" \
  JOB_NAME="${dataset_name}_cupid_quality_openpi05" \
  CHECKPOINT_BASE_DIR="${CHECKPOINT_BASE_DIR:-${OPENPI_ROOT}/checkpoints}" \
  NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-10000}" \
  MAX_FRAMES_FOR_NORM_STATS="${MAX_FRAMES_FOR_NORM_STATS:-}" \
  WANDB_MODE="${WANDB_MODE:-disabled}" \
  UV_CACHE_DIR="${UV_CACHE_DIR:-${OPENPI_ROOT}/tmp/uv-cache}" \
  SKIP_TRAIN="${SKIP_TRAIN:-0}" \
  OVERWRITE="${OVERWRITE:-1}" \
  bash "${ENTRY_SCRIPT}"
}

case "${DATASET_NAME}" in
  can_mh)
    run_one \
      "can_mh" \
      "/workspace/liushuo/cupid/datasets/lerobot/can_mh" \
      "pi05_can_mh_lora" \
      "/workspace/liushuo/cupid/3_model/can_mh_cupid_quality_dropout"
    ;;
  square_mh)
    run_one \
      "square_mh" \
      "/workspace/liushuo/cupid/datasets/lerobot/square_mh" \
      "pi05_square_mh_lora" \
      "/workspace/liushuo/cupid/3_model/square_mh_cupid_quality"
    ;;
  transport_mh)
    run_one \
      "transport_mh" \
      "/workspace/liushuo/cupid/datasets/lerobot/transport_mh" \
      "pi05_transport_mh_lora" \
      "/workspace/liushuo/cupid/3_model/quality_eval/transport_mh_cupid_quality_joint_drop20"
    ;;
  all)
    run_one \
      "can_mh" \
      "/workspace/liushuo/cupid/datasets/lerobot/can_mh" \
      "pi05_can_mh_lora" \
      "/workspace/liushuo/cupid/3_model/can_mh_cupid_quality_dropout"
    run_one \
      "square_mh" \
      "/workspace/liushuo/cupid/datasets/lerobot/square_mh" \
      "pi05_square_mh_lora" \
      "/workspace/liushuo/cupid/3_model/square_mh_cupid_quality"
    run_one \
      "transport_mh" \
      "/workspace/liushuo/cupid/datasets/lerobot/transport_mh" \
      "pi05_transport_mh_lora" \
      "/workspace/liushuo/cupid/3_model/quality_eval/transport_mh_cupid_quality_joint_drop20"
    ;;
  *)
    echo "Unsupported dataset: ${DATASET_NAME}. Use can_mh, square_mh, transport_mh, or all." >&2
    exit 1
    ;;
esac
