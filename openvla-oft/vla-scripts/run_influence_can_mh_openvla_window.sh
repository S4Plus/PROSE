#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/workspace/liushuo/cupid/openvla-oft"
cd "${ROOT_DIR}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

CHECKPOINT_DIR="${CHECKPOINT_DIR:-/workspace/liushuo/cupid/openvla-oft/runs/can_mh_openvla_lora_fixed_stats--10000_chkpt}"
EVAL_ROLLOUTS_DIR="${EVAL_ROLLOUTS_DIR:-/workspace/liushuo/cupid/mimicgen/eval/tmp/can_mh_openvla_eval/pkl}"
OUTPUT_PATH="${OUTPUT_PATH:-/workspace/liushuo/cupid/openvla-oft/tmp/can_mh_openvla_window_grass/influence_grass.pkl}"
CACHE_DIR="${CACHE_DIR:-/workspace/liushuo/cupid/openvla-oft/tmp/can_mh_openvla_window_grass/cache}"
LOG_DIR="${LOG_DIR:-/workspace/liushuo/cupid/openvla-oft/logs/can_mh_openvla_window_grass}"

mkdir -p "$(dirname "${OUTPUT_PATH}")" "${CACHE_DIR}" "${LOG_DIR}"

python vla-scripts/influence_openvla_lora_grass_window.py \
  --checkpoint-dir "${CHECKPOINT_DIR}" \
  --eval-rollouts-dir "${EVAL_ROLLOUTS_DIR}" \
  --output-path "${OUTPUT_PATH}" \
  --stage-cache-prefix "${CACHE_DIR}" \
  --window-size "${WINDOW_SIZE:-50}" \
  --window-stride "${WINDOW_STRIDE:-50}" \
  --window-aggregation "${WINDOW_AGGREGATION:-sum}" \
  --quality-score "${QUALITY_SCORE:-zscore_mean}" \
  --sparsification "${SPARSIFICATION:-random_mask-32*32}" \
  --projection "${PROJECTION:-sjlt-1024}" \
  --hessian "${HESSIAN:-eFIM}" \
  --damping "${DAMPING:-1e-4}" \
  --stage1-microbatch-size "${STAGE1_MICROBATCH_SIZE:-1}" \
  --stage2-batch-size "${STAGE2_BATCH_SIZE:-16}" \
  --progress-every "${PROGRESS_EVERY:-200}" \
  --seed "${SEED:-7}"
