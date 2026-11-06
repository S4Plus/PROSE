#!/usr/bin/env bash
# Run window-level joint retraining serially from an already computed window GraSS score file.
#
# Example:
#   CUDA_VISIBLE_DEVICES=0 bash scripts/retrain_window_joint_serial.sh
#   PREPARE_ONLY=1 bash scripts/retrain_window_joint_serial.sh
#   QUALITY_SOURCE=trajectory_eval RATIOS="0.8 0.7" bash scripts/retrain_window_joint_serial.sh
#
# Tunable env vars:
#   SCORE_PKL
#   TRAIN_CONFIG
#   DATASET_DIR
#   RATIOS
#   NUM_TRAIN_STEPS
#   PREPARE_ONLY
#   OVERWRITE
#   WANDB_MODE
#   CHECKPOINT_BASE_DIR
#   OUTPUT_DIR_PARENT
#   OUTPUT_DIR_TAG
#   EXP_NAME_PREFIX
#   QUALITY_SOURCE
#   QUALITY_NORMALIZER
#   QUALITY_QUANTILE_LOW
#   QUALITY_QUANTILE_HIGH
#   JOINT_QUALITY_POWER
#   JOINT_INFLUENCE_POWER
#   EPISODE_ACTION_WEIGHT
#   EPISODE_STATE_WEIGHT
#   EPISODE_LENGTH_WEIGHT
#   EPISODE_ENERGY_WEIGHT
#   TRAJECTORY_TERMINAL_WEIGHT
#   TRAJECTORY_EFFICIENCY_WEIGHT
#   TRAJECTORY_PROGRESS_WEIGHT
#   TRAJECTORY_STATE_SMOOTH_WEIGHT
#   TRAJECTORY_ACTION_SMOOTH_WEIGHT
#   TRAJECTORY_LENGTH_WEIGHT
#   RANK_FUSION_EPISODE_WEIGHT
#   RANK_FUSION_TRAJECTORY_WEIGHT
#   MAX_FRAMES_FOR_NORM_STATS
#   UV_CACHE_DIR
#
# Example nohup:
  # CUDA_VISIBLE_DEVICES=5 nohup bash scripts/retrain_window_joint_serial.sh \
  #   > /workspace/liushuo/cupid/openpi/tmp/can_mh_grass_window/retrain_window_joint_serial_master.log 2>&1 &

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

TRAIN_CONFIG="${TRAIN_CONFIG:-pi05_can_mh_lora}"
SCORE_PKL="${SCORE_PKL:-/workspace/liushuo/cupid/openpi/tmp/can_mh_grass_window/score.pkl}"
DATASET_DIR="${DATASET_DIR:-/workspace/liushuo/cupid/datasets/lerobot/can_mh}"

RATIOS_STRING="${RATIOS:-0.9 0.8 0.7 0.6 0.5}"
read -r -a RATIOS <<< "$RATIOS_STRING"

JOINT_QUALITY_POWER="${JOINT_QUALITY_POWER:-1}"
JOINT_INFLUENCE_POWER="${JOINT_INFLUENCE_POWER:-1}"
QUALITY_SOURCE="${QUALITY_SOURCE:-rank_fusion}"
QUALITY_NORMALIZER="${QUALITY_NORMALIZER:-quantile}"
QUALITY_QUANTILE_LOW="${QUALITY_QUANTILE_LOW:-0.05}"
QUALITY_QUANTILE_HIGH="${QUALITY_QUANTILE_HIGH:-0.95}"
EPISODE_ACTION_WEIGHT="${EPISODE_ACTION_WEIGHT:-0.35}"
EPISODE_STATE_WEIGHT="${EPISODE_STATE_WEIGHT:-0.35}"
EPISODE_LENGTH_WEIGHT="${EPISODE_LENGTH_WEIGHT:-0.15}"
EPISODE_ENERGY_WEIGHT="${EPISODE_ENERGY_WEIGHT:-0.15}"
TRAJECTORY_TERMINAL_WEIGHT="${TRAJECTORY_TERMINAL_WEIGHT:-0.25}"
TRAJECTORY_EFFICIENCY_WEIGHT="${TRAJECTORY_EFFICIENCY_WEIGHT:-0.20}"
TRAJECTORY_PROGRESS_WEIGHT="${TRAJECTORY_PROGRESS_WEIGHT:-0.25}"
TRAJECTORY_STATE_SMOOTH_WEIGHT="${TRAJECTORY_STATE_SMOOTH_WEIGHT:-0.15}"
TRAJECTORY_ACTION_SMOOTH_WEIGHT="${TRAJECTORY_ACTION_SMOOTH_WEIGHT:-0.10}"
TRAJECTORY_LENGTH_WEIGHT="${TRAJECTORY_LENGTH_WEIGHT:-0.05}"
RANK_FUSION_EPISODE_WEIGHT="${RANK_FUSION_EPISODE_WEIGHT:-0.5}"
RANK_FUSION_TRAJECTORY_WEIGHT="${RANK_FUSION_TRAJECTORY_WEIGHT:-0.5}"

NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-10000}"
PREPARE_ONLY="${PREPARE_ONLY:-0}"
OVERWRITE="${OVERWRITE:-1}"
WANDB_MODE="${WANDB_MODE:-disabled}"
CHECKPOINT_BASE_DIR="${CHECKPOINT_BASE_DIR:-./checkpoints}"
OUTPUT_DIR_PARENT="${OUTPUT_DIR_PARENT:-/workspace/liushuo/cupid/openpi/tmp}"
OUTPUT_DIR_TAG="${OUTPUT_DIR_TAG:-can_mh_window_joint_keep}"
EXP_NAME_PREFIX="${EXP_NAME_PREFIX:-can_mh_window_joint_keep}"
MAX_FRAMES_FOR_NORM_STATS="${MAX_FRAMES_FOR_NORM_STATS:-}"
UV_CACHE_DIR="${UV_CACHE_DIR:-./tmp/uv-cache}"
RUN_LOG_DIR="${RUN_LOG_DIR:-/workspace/liushuo/cupid/openpi/tmp/can_mh_grass_window}"
OUTPUT_SUFFIX="${OUTPUT_SUFFIX:-window_joint}"

mkdir -p "$RUN_LOG_DIR"

run_ratio() {
  local ratio="$1"
  local keep_pct
  local ratio_tag
  local output_dir
  local exp_name
  local run_log
  local -a cmd

  keep_pct="$(python - "$ratio" <<'PY'
import sys
print(int(round(float(sys.argv[1]) * 100)))
PY
)"
  ratio_tag="keep${keep_pct}"
  output_dir="${OUTPUT_DIR_PARENT}/${OUTPUT_DIR_TAG}${keep_pct}_${QUALITY_SOURCE}"
  exp_name="${EXP_NAME_PREFIX}${keep_pct}_${QUALITY_SOURCE}"
  run_log="${RUN_LOG_DIR}/retrain_window_joint_${ratio_tag}.log"

  cmd=(
    uv run python retrain_window_joint.py
    --score-pkl "$SCORE_PKL"
    --train-config "$TRAIN_CONFIG"
    --dataset-dir "$DATASET_DIR"
    --keep-ratio "$ratio"
    --joint-quality-power "$JOINT_QUALITY_POWER"
    --joint-influence-power "$JOINT_INFLUENCE_POWER"
    --quality-source "$QUALITY_SOURCE"
    --quality-normalizer "$QUALITY_NORMALIZER"
    --quality-quantile-low "$QUALITY_QUANTILE_LOW"
    --quality-quantile-high "$QUALITY_QUANTILE_HIGH"
    --episode-action-weight "$EPISODE_ACTION_WEIGHT"
    --episode-state-weight "$EPISODE_STATE_WEIGHT"
    --episode-length-weight "$EPISODE_LENGTH_WEIGHT"
    --episode-energy-weight "$EPISODE_ENERGY_WEIGHT"
    --trajectory-terminal-weight "$TRAJECTORY_TERMINAL_WEIGHT"
    --trajectory-efficiency-weight "$TRAJECTORY_EFFICIENCY_WEIGHT"
    --trajectory-progress-weight "$TRAJECTORY_PROGRESS_WEIGHT"
    --trajectory-state-smooth-weight "$TRAJECTORY_STATE_SMOOTH_WEIGHT"
    --trajectory-action-smooth-weight "$TRAJECTORY_ACTION_SMOOTH_WEIGHT"
    --trajectory-length-weight "$TRAJECTORY_LENGTH_WEIGHT"
    --rank-fusion-episode-weight "$RANK_FUSION_EPISODE_WEIGHT"
    --rank-fusion-trajectory-weight "$RANK_FUSION_TRAJECTORY_WEIGHT"
    --output-dir "$output_dir"
    --exp-name "$exp_name"
    --checkpoint-base-dir "$CHECKPOINT_BASE_DIR"
  )

  if [[ -n "$NUM_TRAIN_STEPS" ]]; then
    cmd+=(--num-train-steps "$NUM_TRAIN_STEPS")
  fi
  if [[ -n "$MAX_FRAMES_FOR_NORM_STATS" ]]; then
    cmd+=(--max-frames-for-norm-stats "$MAX_FRAMES_FOR_NORM_STATS")
  fi
  if [[ "$PREPARE_ONLY" == "1" ]]; then
    cmd+=(--prepare-only)
  fi
  if [[ "$OVERWRITE" == "1" ]]; then
    cmd+=(--overwrite)
  fi

  echo "============================================================"
  echo "Running window joint retrain pipeline for keep ratio: $ratio"
  echo "Score file: $SCORE_PKL"
  echo "Dataset dir: $DATASET_DIR"
  echo "Quality source: $QUALITY_SOURCE"
  echo "Output dir: $output_dir"
  echo "Exp name: $exp_name"
  echo "Log: $run_log"
  echo "Command: ${cmd[*]}"
  echo "============================================================"

  UV_CACHE_DIR="$UV_CACHE_DIR" WANDB_MODE="$WANDB_MODE" "${cmd[@]}" 2>&1 | tee "$run_log"
}

for ratio in "${RATIOS[@]}"; do
  run_ratio "$ratio"
done

echo "All window joint retrain jobs finished."
