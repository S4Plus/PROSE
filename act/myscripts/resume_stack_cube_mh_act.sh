#!/usr/bin/env bash
set -euo pipefail

# 从 stack_cube_mh_act 的 100000 step checkpoint 继续训练到 200000 step。
# 用法：
#   CUDA_VISIBLE_DEVICES=0 bash /workspace/liushuo/cupid/act/myscripts/resume_stack_cube_mh_act.sh

REPO_ROOT="/workspace/liushuo/cupid/act"
OUTPUT_DIR="/workspace/liushuo/cupid/outputs/stack_cube_mh_act"
CONFIG_PATH="${OUTPUT_DIR}/checkpoints/last/pretrained_model/train_config.json"
LOG_DIR="${REPO_ROOT}/logs"
LOG_FILE="${LOG_DIR}/stack_cube_mh_act_resume_200k.log"

mkdir -p "${LOG_DIR}"

source /workspace/liushuo/miniconda3/bin/activate cupid

# 尽量复用本地缓存，避免在离线或受限环境下触发外部访问。
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"

cd "${REPO_ROOT}"

nohup lerobot-train \
    --config_path="${CONFIG_PATH}" \
    --resume=true \
    --steps=200000 \
    > "${LOG_FILE}" 2>&1 &

echo "已启动后台续训。"
echo "config: ${CONFIG_PATH}"
echo "log: ${LOG_FILE}"
