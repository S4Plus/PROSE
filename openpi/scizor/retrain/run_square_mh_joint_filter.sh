#!/usr/bin/env bash


# nohup bash -lc '
#   cd /workspace/liushuo/cupid/openpi
#   CUDA_VISIBLE_DEVICES=4 \
#   XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
#   WANDB_MODE=disabled \
#   UV_CACHE_DIR=./tmp/uv-cache \
#   bash scizor/retrain/run_square_mh_joint_filter.sh \
#     --overwrite \
#     --exp-name square_mh_scizor_joint_filter \
#     --num-train-steps 10000
#   ' > /workspace/liushuo/cupid/openpi/scizor/retrain/output/pi05_square_mh_lora_scizor_joint_filter/train_openpi_scizor.log 2>&1 &

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

export HF_HOME="${HF_HOME:-${ROOT_DIR}/tmp/scizor-retrain-cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${ROOT_DIR}/tmp/scizor-retrain-cache/xdg}"
export WANDB_MODE="${WANDB_MODE:-disabled}"

UV_CACHE_DIR="${UV_CACHE_DIR:-${ROOT_DIR}/tmp/uv-cache}" \
uv run python scizor/retrain/retrain_scizor.py \
  --score-pkl /workspace/liushuo/SCIZOR/outputs/act_scizor/square_mh/scizor_joint_filter.pkl \
  --train-config pi05_square_mh_lora \
  --dataset-dir /workspace/liushuo/cupid/datasets/lerobot/square_mh \
  --output-dir "${ROOT_DIR}/scizor/retrain/output/pi05_square_mh_lora_scizor_joint_filter" \
  "$@"


#  CUDA_VISIBLE_DEVICES=0,1,2,3 nohup bash scizor/retrain/run_square_mh_joint_filter.sh \
#     --overwrite \
#     --exp-name square_mh_scizor_joint_filter \
#     --num-train-steps 10000 \
#     > scizor/retrain/output/pi05_square_mh_lora_scizor_joint_filter/train_openpi_scizor.log 2>&1 &
