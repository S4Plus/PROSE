#!/usr/bin/env bash
# CUDA_VISIBLE_DEVICES=6 PREPARE_ONLY=0 bash scripts/retrain_random_mask_ratios.sh pi05_can_mh_lora
# nohup bash -c 'CUDA_VISIBLE_DEVICES=6 PREPARE_ONLY=0 bash scripts/retrain_random_mask_ratios.sh pi05_can_mh_lora' > logs/retrain_random_mask_ratios.log 2>&1 &
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <train_config> [dataset_dir]"
  echo "Example: $0 pi05_can_mh_lora /workspace/liushuo/cupid/datasets/lerobot/can_mh"
  exit 1
fi

TRAIN_CONFIG="$1"
DATASET_DIR="${2:-}"

SEED="${SEED:-42}"
OVERWRITE="${OVERWRITE:-1}"
PREPARE_ONLY="${PREPARE_ONLY:-1}"
ENABLE_WANDB="${ENABLE_WANDB:-0}"
CHECKPOINT_BASE_DIR="${CHECKPOINT_BASE_DIR:-./checkpoints}"
MAX_FRAMES_FOR_NORM_STATS="${MAX_FRAMES_FOR_NORM_STATS:-}"
UV_CACHE_DIR="${UV_CACHE_DIR:-./tmp/uv-cache}"

RATIOS=(0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1)

for ratio in "${RATIOS[@]}"; do
  cmd=(
    uv run python retrain_random_mask.py
    --train-config "$TRAIN_CONFIG"
    --keep-ratio "$ratio"
    --seed "$SEED"
    --checkpoint-base-dir "$CHECKPOINT_BASE_DIR"
  )

  if [[ -n "$DATASET_DIR" ]]; then
    cmd+=(--dataset-dir "$DATASET_DIR")
  fi

  if [[ -n "$MAX_FRAMES_FOR_NORM_STATS" ]]; then
    cmd+=(--max-frames-for-norm-stats "$MAX_FRAMES_FOR_NORM_STATS")
  fi

  if [[ "$OVERWRITE" == "1" ]]; then
    cmd+=(--overwrite)
  fi

  if [[ "$PREPARE_ONLY" == "1" ]]; then
    cmd+=(--prepare-only)
  fi

  if [[ "$ENABLE_WANDB" == "1" ]]; then
    cmd+=(--enable-wandb)
  fi

  echo "============================================================"
  echo "Running random-mask retrain pipeline for keep ratio: $ratio"
  echo "Command: ${cmd[*]}"
  echo "============================================================"

  UV_CACHE_DIR="$UV_CACHE_DIR" "${cmd[@]}"
done
