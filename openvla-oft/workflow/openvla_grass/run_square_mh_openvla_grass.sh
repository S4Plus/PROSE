#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/workspace/liushuo/cupid/openvla-oft"
cd "${ROOT_DIR}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

CHECKPOINT_DIR="${CHECKPOINT_DIR:-${ROOT_DIR}/runs/square_mh_openvla_lora_full--10000_chkpt}"
DATASET_ROOT="${DATASET_ROOT:-/workspace/liushuo/cupid/datasets/lerobot/square_mh}"
EVAL_ROLLOUTS_DIR="${EVAL_ROLLOUTS_DIR:-/workspace/liushuo/cupid/mimicgen/eval/tmp/square_openvla_eval/pkl}"
OUTPUT_PATH="${OUTPUT_PATH:-${ROOT_DIR}/tmp/square_mh_openvla_window_grass/influence_grass.pkl}"
CACHE_DIR="${CACHE_DIR:-${ROOT_DIR}/tmp/square_mh_openvla_window_grass/cache}"
TASK_DESCRIPTION="${TASK_DESCRIPTION:-Insert the square peg into the square hole.}"

mkdir -p "$(dirname "${OUTPUT_PATH}")" "${CACHE_DIR}"

cmd=(
  python "${ROOT_DIR}/vla-scripts/influence_openvla_lora_grass_window.py"
  --checkpoint-dir "${CHECKPOINT_DIR}"
  --eval-rollouts-dir "${EVAL_ROLLOUTS_DIR}"
  --output-path "${OUTPUT_PATH}"
  --dataset-root "${DATASET_ROOT}"
  --task-description "${TASK_DESCRIPTION}"
  --stage-cache-prefix "${CACHE_DIR}"
  --window-size "${WINDOW_SIZE:-50}"
  --window-stride "${WINDOW_STRIDE:-50}"
  --window-aggregation "${WINDOW_AGGREGATION:-sum}"
  --quality-score "${QUALITY_SCORE:-zscore_mean}"
  --sparsification "${SPARSIFICATION:-random_mask-32*32}"
  --projection "${PROJECTION:-sjlt-1024}"
  --hessian "${HESSIAN:-eFIM}"
  --damping "${DAMPING:-1e-4}"
  --stage1-microbatch-size "${STAGE1_MICROBATCH_SIZE:-1}"
  --stage2-batch-size "${STAGE2_BATCH_SIZE:-16}"
  --progress-every "${PROGRESS_EVERY:-200}"
  --seed "${SEED:-7}"
)

if [[ -n "${MAX_TRAIN_SAMPLES:-}" ]]; then
  cmd+=(--max-train-samples "${MAX_TRAIN_SAMPLES}")
fi
if [[ -n "${MAX_EVAL_EPISODES:-}" ]]; then
  cmd+=(--max-eval-episodes "${MAX_EVAL_EPISODES}")
fi
if [[ -n "${MAX_EVAL_STEPS_PER_EPISODE:-}" ]]; then
  cmd+=(--max-eval-steps-per-episode "${MAX_EVAL_STEPS_PER_EPISODE}")
fi
if [[ -n "${SAVE_PRECONDITIONERS:-}" ]]; then
  cmd+=(--save-preconditioners)
fi

"${cmd[@]}"
