#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

source "${SCRIPT_DIR}/setup_env.sh"

export PYTHONPATH="/workspace/liushuo/cupid/lerobot_2.1/src:${PYTHONPATH:-}"

CHECKPOINT_DIR="${CHECKPOINT_DIR:-/workspace/liushuo/cupid/outputs/stack_cube_franka_joint_act/checkpoints/100000/pretrained_model}"
TOTAL_STEPS="${TOTAL_STEPS:-300000}"
CUDA_DEVICE="${CUDA_DEVICE:-1}"
PYTHON_BIN="${PYTHON_BIN:-python}"

echo "开始续训 ACT"
echo "CHECKPOINT_DIR=${CHECKPOINT_DIR}"
echo "TOTAL_STEPS=${TOTAL_STEPS}"
echo "CUDA_DEVICE=${CUDA_DEVICE}"
echo "PYTHON_BIN=${PYTHON_BIN}"

CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" \
"${PYTHON_BIN}" -m lerobot.scripts.train \
  --config_path="${CHECKPOINT_DIR}" \
  --resume=true \
  --steps="${TOTAL_STEPS}"
