#!/usr/bin/env bash
set -euo pipefail

ROOT="/workspace/liushuo/cupid/openpi"
RAW_ROOT="/workspace/liushuo/data/mp4"
LEROBOT_ROOT="/workspace/liushuo/data/dataset"
LOG_DIR="${ROOT}/logs/dataset_test_pi05"
UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache-openpi}"
WANDB_MODE="${WANDB_MODE:-offline}"

LEFT_GPU="${LEFT_GPU:-6}"
RIGHT_GPU="${RIGHT_GPU:-7}"
LEFT_EXP="${LEFT_EXP:-left_arm_grasp_object_pi05_30k}"
RIGHT_EXP="${RIGHT_EXP:-right_arm_grasp_box_pi05_30k}"

mkdir -p "${LOG_DIR}" "${LEROBOT_ROOT}"
cd "${ROOT}"

run_one_task() {
  set -euo pipefail

  local gpu="$1"
  local raw_name="$2"
  local repo_id="$3"
  local task_prompt="$4"
  local config_name="$5"
  local exp_name="$6"
  local log_file="${LOG_DIR}/${raw_name}.log"

  {
    echo "[start] $(date '+%F %T') task=${raw_name} gpu=${gpu}"
    echo "[convert] raw=${RAW_ROOT}/${raw_name}"
    UV_CACHE_DIR="${UV_CACHE_DIR}" \
      uv run python franka/convert_mp4_joint_state_to_lerobot.py \
        --raw-dir "${RAW_ROOT}/${raw_name}" \
        --repo-id "${repo_id}" \
        --lerobot-home "${LEROBOT_ROOT}/${raw_name}" \
        --task "${task_prompt}" \
        --overwrite

    echo "[norm] config=${config_name}"
    CUDA_VISIBLE_DEVICES="${gpu}" \
    UV_CACHE_DIR="${UV_CACHE_DIR}" \
      uv run scripts/compute_norm_stats.py \
        --config-name "${config_name}"

    echo "[train] config=${config_name} exp=${exp_name}"
    CUDA_VISIBLE_DEVICES="${gpu}" \
    XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}" \
    UV_CACHE_DIR="${UV_CACHE_DIR}" \
    WANDB_MODE="${WANDB_MODE}" \
      uv run scripts/train.py "${config_name}" \
        --exp-name "${exp_name}" \
        --overwrite

    echo "[done] $(date '+%F %T') task=${raw_name}"
  } > "${log_file}" 2>&1
}

nohup bash -c "$(declare -f run_one_task); ROOT='${ROOT}'; RAW_ROOT='${RAW_ROOT}'; LEROBOT_ROOT='${LEROBOT_ROOT}'; LOG_DIR='${LOG_DIR}'; UV_CACHE_DIR='${UV_CACHE_DIR}'; WANDB_MODE='${WANDB_MODE}'; cd '${ROOT}'; run_one_task '${LEFT_GPU}' 'left_arm_grasp_object_simple' 'local/left_arm_grasp_object_simple' 'Use the left arm to grasp the object.' 'pi05_left_arm_grasp_object_lora' '${LEFT_EXP}'" >/dev/null 2>&1 &
left_pid=$!

nohup bash -c "$(declare -f run_one_task); ROOT='${ROOT}'; RAW_ROOT='${RAW_ROOT}'; LEROBOT_ROOT='${LEROBOT_ROOT}'; LOG_DIR='${LOG_DIR}'; UV_CACHE_DIR='${UV_CACHE_DIR}'; WANDB_MODE='${WANDB_MODE}'; cd '${ROOT}'; run_one_task '${RIGHT_GPU}' 'right_arm_grasp_box_simple' 'local/right_arm_grasp_box_simple' 'Use the right arm to grasp the box.' 'pi05_right_arm_grasp_box_lora' '${RIGHT_EXP}'" >/dev/null 2>&1 &
right_pid=$!

echo "left_pid=${left_pid} log=${LOG_DIR}/left_arm_grasp_object_simple.log"
echo "right_pid=${right_pid} log=${LOG_DIR}/right_arm_grasp_box_simple.log"
