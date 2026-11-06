#!/usr/bin/env bash

# 使用本地 LeRobot v2.1 数据集训练 ACT。
# 默认数据集：
#   /workspace/liushuo/dataset/beat_block_hammer_ur5_randomized_500
#
# 用法示例：
#   bash robotwin/train_beat_block_hammer_ur5_randomized_500_act.sh
#
# 常见覆盖方式：
#   CUDA_VISIBLE_DEVICES=1 BATCH_SIZE=16 bash robotwin/train_beat_block_hammer_ur5_randomized_500_act.sh
#   LAUNCH_MODE=foreground CUDA_VISIBLE_DEVICES=0 bash robotwin/train_beat_block_hammer_ur5_randomized_500_act.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

DATASET_ROOT="${DATASET_ROOT:-/workspace/liushuo/dataset/beat_block_hammer_ur5_randomized_500}"
DATASET_REPO_ID="${DATASET_REPO_ID:-local/beat_block_hammer_ur5_randomized_500}"

RUN_NAME="${RUN_NAME:-beat_block_hammer_ur5_randomized_500_act}"
OUTPUT_BASE="${OUTPUT_BASE:-${PROJECT_ROOT}/outputs/robotwin}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_BASE}/${RUN_NAME}}"

LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs/robotwin}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/${RUN_NAME}.log}"

HF_CACHE_ROOT="${HF_CACHE_ROOT:-${PROJECT_ROOT}/.cache/huggingface}"
HF_HOME="${HF_HOME:-${HF_CACHE_ROOT}/home}"
HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_CACHE_ROOT}/datasets}"
PYTHON_BIN="${PYTHON_BIN:-python}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
DEVICE="${DEVICE:-cuda}"
USE_AMP="${USE_AMP:-true}"

# 这份数据集是 3 路 240x320 图像，默认 batch size 取偏稳妥值。
# 如果你是 24GB+ 显存，可以优先从 8 起步，再逐步加到 12/16。
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-4}"
STEPS="${STEPS:-100000}"
SAVE_FREQ="${SAVE_FREQ:-10000}"
LOG_FREQ="${LOG_FREQ:-100}"
EVAL_FREQ="${EVAL_FREQ:-0}"
SEED="${SEED:-1000}"

CHUNK_SIZE="${CHUNK_SIZE:-50}"
N_ACTION_STEPS="${N_ACTION_STEPS:-50}"
VISION_BACKBONE="${VISION_BACKBONE:-resnet18}"
PRETRAINED_BACKBONE="${PRETRAINED_BACKBONE:-ResNet18_Weights.IMAGENET1K_V1}"
DROPOUT="${DROPOUT:-0.1}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
BACKBONE_LR="${BACKBONE_LR:-1e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"

# `nohup` 适合服务器后台训练；`foreground` 适合现场观察日志。
LAUNCH_MODE="${LAUNCH_MODE:-nohup}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "未找到 Python 解释器: ${PYTHON_BIN}" >&2
  exit 1
fi

if [[ ! -f "${DATASET_ROOT}/meta/info.json" ]]; then
  echo "数据集目录不合法，缺少 meta/info.json: ${DATASET_ROOT}" >&2
  exit 1
fi

if [[ ! -d "${DATASET_ROOT}/data" ]]; then
  echo "数据集目录不合法，缺少 data/ 目录: ${DATASET_ROOT}" >&2
  exit 1
fi

if [[ -e "${OUTPUT_DIR}" ]]; then
  echo "输出目录已存在，当前脚本不处理 resume：${OUTPUT_DIR}" >&2
  echo "请修改 RUN_NAME 或 OUTPUT_DIR 后重新启动训练。" >&2
  exit 1
fi

mkdir -p "${OUTPUT_BASE}" "${LOG_DIR}" "${HF_HOME}" "${HF_DATASETS_CACHE}"

AMP_ARGS=()
if [[ "${USE_AMP}" == "true" ]]; then
  AMP_ARGS+=(--use-amp)
fi

cmd=(
  "${PYTHON_BIN}"
  "${PROJECT_ROOT}/robotwin/train_act.py"
  --dataset-root "${DATASET_ROOT}"
  --repo-id "${DATASET_REPO_ID}"
  --output-dir "${OUTPUT_DIR}"
  --device "${DEVICE}"
  --batch-size "${BATCH_SIZE}"
  --num-workers "${NUM_WORKERS}"
  --max-steps "${STEPS}"
  --save-every "${SAVE_FREQ}"
  --log-every "${LOG_FREQ}"
  --seed="${SEED}"
  --chunk-size "${CHUNK_SIZE}"
  --n-action-steps "${N_ACTION_STEPS}"
  --vision-backbone "${VISION_BACKBONE}"
  --pretrained-backbone "${PRETRAINED_BACKBONE}"
  --dropout "${DROPOUT}"
  --learning-rate "${LEARNING_RATE}"
  --backbone-lr "${BACKBONE_LR}"
  --weight-decay "${WEIGHT_DECAY}"
)
cmd+=("${AMP_ARGS[@]}")

echo "==================== ACT 训练配置 ===================="
echo "DATASET_ROOT=${DATASET_ROOT}"
echo "DATASET_REPO_ID=${DATASET_REPO_ID}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "LOG_FILE=${LOG_FILE}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "DEVICE=${DEVICE}"
echo "USE_AMP=${USE_AMP}"
echo "BATCH_SIZE=${BATCH_SIZE}"
echo "NUM_WORKERS=${NUM_WORKERS}"
echo "STEPS=${STEPS}"
echo "SAVE_FREQ=${SAVE_FREQ}"
echo "LOG_FREQ=${LOG_FREQ}"
echo "CHUNK_SIZE=${CHUNK_SIZE}"
echo "N_ACTION_STEPS=${N_ACTION_STEPS}"
echo "======================================================"

if [[ "${LAUNCH_MODE}" == "foreground" ]]; then
  env \
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
    HF_HOME="${HF_HOME}" \
    HF_DATASETS_CACHE="${HF_DATASETS_CACHE}" \
    "${cmd[@]}" 2>&1 | tee "${LOG_FILE}"
else
  nohup env \
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
    HF_HOME="${HF_HOME}" \
    HF_DATASETS_CACHE="${HF_DATASETS_CACHE}" \
    "${cmd[@]}" \
    > "${LOG_FILE}" 2>&1 &
  echo "训练已后台启动，PID=$!"
  echo "日志文件：${LOG_FILE}"
fi
