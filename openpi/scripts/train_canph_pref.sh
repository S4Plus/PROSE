#!/usr/bin/env bash
set -euo pipefail

cd /workspace/liushuo/cupid/openpi

DATASET_DIR="${DATASET_DIR:-/workspace/liushuo/cupid/output/openpi05_canph_ablation/datasets/can_mh_twostage_plus_can_ph_pref_k054}"
ASSETS_DIR="${ASSETS_DIR:-/workspace/liushuo/cupid/output/openpi05_canph_ablation/datasets}"
ASSET_ID="${ASSET_ID:-can_mh_twostage_plus_can_ph_pref_k054}"
TRAIN_CONFIG="${TRAIN_CONFIG:-pi05_can_mh_lora}"
EXP_NAME="${EXP_NAME:-can_mh_twostage_plus_can_ph_pref_k054}"
NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-10000}"
SAMPLE_INDICES_PATH="${SAMPLE_INDICES_PATH:-/workspace/liushuo/cupid/output/openpi05_canph_ablation/datasets/can_mh_twostage_plus_can_ph_pref_k054/openpi_sample_indices.npy}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}"
WANDB_MODE="${WANDB_MODE:-disabled}"
UV_CACHE_DIR="${UV_CACHE_DIR:-/workspace/liushuo/cupid/openpi/tmp/uv-cache}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION}" \
WANDB_MODE="${WANDB_MODE}" \
UV_CACHE_DIR="${UV_CACHE_DIR}" \
uv run scripts/train.py "${TRAIN_CONFIG}" \
  --exp-name "${EXP_NAME}" \
  --data.repo-id "${DATASET_DIR}" \
  --data.assets.assets-dir "${ASSETS_DIR}" \
  --data.assets.asset-id "${ASSET_ID}" \
  --data.sample-indices-path "${SAMPLE_INDICES_PATH}" \
  --num-train-steps "${NUM_TRAIN_STEPS}" \
  --overwrite
