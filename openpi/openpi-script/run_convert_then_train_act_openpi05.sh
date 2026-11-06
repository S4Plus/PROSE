#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"


# export PYTHONPATH="${REPO_ROOT}/lerobot_2.1/src:${PYTHONPATH:-}"

RAW_DIR="${RAW_DIR:-/ext_workspace/liushuo/stack_cube}"
DATASET_ROOT="${DATASET_ROOT:-/workspace/liushuo/dataset/stack_cube_205}"
DATASET_REPO_ID="${DATASET_REPO_ID:-local/stack_cube_205}"
DATASET_NAME="${DATASET_NAME:-$(basename "${DATASET_ROOT}")}"
TASK_TEXT="${TASK_TEXT:-stack cube}"

# ACT_GPU="${ACT_GPU:-5}"
OPENPI_NORM_GPU="${OPENPI_NORM_GPU:-1}"
OPENPI_TRAIN_GPUS="${OPENPI_TRAIN_GPUS:-6,7}"

ACT_STEPS="${ACT_STEPS:-300000}"
OPENPI_STEPS="${OPENPI_STEPS:-30000}"

ACT_POLICY_REPO_ID="${ACT_POLICY_REPO_ID:-local/stack_cube_205_act}"
ACT_OUTPUT_DIR="${ACT_OUTPUT_DIR:-/workspace/liushuo/cupid/outputs/stack_cube_205_act}"
ACT_JOB_NAME="${ACT_JOB_NAME:-stack_cube_205_act}"

OPENPI_CONFIG_NAME="${OPENPI_CONFIG_NAME:-pi05_stack_cube_205_lora}"
OPENPI_EXP_NAME="${OPENPI_EXP_NAME:-stack_cube_205_pi05_30k}"
OPENPI_ASSETS_DIR="${OPENPI_ASSETS_DIR:-/workspace/liushuo/dataset}"
OPENPI_ASSET_ID="${OPENPI_ASSET_ID:-stack_cube_205}"
OPENPI_CHECKPOINT_BASE_DIR="${OPENPI_CHECKPOINT_BASE_DIR:-/workspace/liushuo/cupid/openpi/checkpoints}"
OPENPI_NORM_MAX_FRAMES="${OPENPI_NORM_MAX_FRAMES:-}"

ACT_LOG_DIR="${ACT_LOG_DIR:-/workspace/liushuo/cupid/act/logs}"
OPENPI_LOG_DIR="${OPENPI_LOG_DIR:-/workspace/liushuo/cupid/openpi/logs}"
# HF_CACHE_ROOT="${HF_CACHE_ROOT:-${REPO_ROOT}/.cache/huggingface}"
# HF_HOME="${HF_HOME:-${HF_CACHE_ROOT}/home}"
# HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_CACHE_ROOT}/datasets}"

# mkdir -p "${ACT_LOG_DIR}" "${OPENPI_LOG_DIR}" "${HF_HOME}" "${HF_DATASETS_CACHE}" "${OPENPI_CHECKPOINT_BASE_DIR}"

if [[ -z "${OPENPI_FSDP_DEVICES:-}" ]]; then
  IFS=',' read -r -a openpi_gpu_array <<< "${OPENPI_TRAIN_GPUS}"
  OPENPI_FSDP_DEVICES="${#openpi_gpu_array[@]}"
fi

timestamp="$(date +%Y%m%d_%H%M%S)"
act_log="${ACT_LOG_DIR}/${ACT_JOB_NAME}_${timestamp}.log"
openpi_log="${OPENPI_LOG_DIR}/${OPENPI_EXP_NAME}_${timestamp}.log"
convert_log="${ACT_LOG_DIR}/convert_${DATASET_NAME}_${timestamp}.log"
norm_log="${OPENPI_LOG_DIR}/${OPENPI_EXP_NAME}_norm_${timestamp}.log"
act_pid=""

# echo "[1/4] 开始转换数据集"
# echo "RAW_DIR=${RAW_DIR}"
# echo "DATASET_ROOT=${DATASET_ROOT}"
# echo "DATASET_REPO_ID=${DATASET_REPO_ID}"

# python "${SCRIPT_DIR}/convert_stack_cube_hdf5_to_lerobot.py" \
#   --raw-dir "${RAW_DIR}" \
#   --repo-id "${DATASET_REPO_ID}" \
#   --lerobot-home "${DATASET_ROOT}" \
#   --task "${TASK_TEXT}" \
#   --overwrite \
#   2>&1 | tee "${convert_log}"

# echo "[2/4] 启动 ACT 训练"
# nohup env \
#   CUDA_VISIBLE_DEVICES="${ACT_GPU}" \
#   HF_HOME="${HF_HOME}" \
#   HF_DATASETS_CACHE="${HF_DATASETS_CACHE}" \
#   PYTHONPATH="${PYTHONPATH}" \
#   lerobot-train \
#   --policy.type=act \
#   --policy.device=cuda \
#   --policy.push_to_hub=false \
#   --policy.repo_id="${ACT_POLICY_REPO_ID}" \
#   --dataset.repo_id="${DATASET_REPO_ID}" \
#   --dataset.root="${DATASET_ROOT}" \
#   --output_dir="${ACT_OUTPUT_DIR}" \
#   --job_name="${ACT_JOB_NAME}" \
#   --steps="${ACT_STEPS}" \
#   > "${act_log}" 2>&1 &
# act_pid=$!

# echo "[3/4] 计算 OpenPI05 norm_stats"
# (
#   cd /workspace/liushuo/cupid/openpi
#   CUDA_VISIBLE_DEVICES="${OPENPI_NORM_GPU}" \
#   WANDB_MODE=disabled \
#   uv run scripts/compute_norm_stats.py \
#     --config-name "${OPENPI_CONFIG_NAME}"
# ) 2>&1 | tee "${norm_log}"

echo "[4/4] 启动 OpenPI05 训练"
nohup bash -lc "
cd /workspace/liushuo/cupid/openpi
CUDA_VISIBLE_DEVICES=6,7 \
WANDB_MODE=disabled \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run scripts/train.py '${OPENPI_CONFIG_NAME}' \
  --exp-name '${OPENPI_EXP_NAME}' \
  --checkpoint-base-dir '${OPENPI_CHECKPOINT_BASE_DIR}' \
  --fsdp-devices 2 \
  --num-train-steps '${OPENPI_STEPS}' \
  --overwrite
" > "${openpi_log}" 2>&1 &
openpi_pid=$!

echo "完成："
echo "  转换日志: ${convert_log}"
echo "  ACT 日志: ${act_log}"
echo "  OpenPI norm_stats 日志: ${norm_log}"
echo "  OpenPI 训练日志: ${openpi_log}"
echo "  ACT PID: ${act_pid:-not_started}"
echo "  OpenPI PID: ${openpi_pid}"
