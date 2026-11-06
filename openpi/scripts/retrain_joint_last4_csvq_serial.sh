#!/usr/bin/env bash
# Run joint retraining serially from an already computed last-4-leaf GraSS score file.
#
# Example:
#   CUDA_VISIBLE_DEVICES=0 bash scripts/retrain_joint_last4_csvq_serial.sh
#   PREPARE_ONLY=1 bash scripts/retrain_joint_last4_csvq_serial.sh
#
# Tunable env vars:
#   SCORE_PKL
#   QUALITY_SCORE_CSV
#   TRAIN_CONFIG
#   RATIOS             (space-separated, default: "0.7 0.8 0.9")
#   NUM_TRAIN_STEPS
#   PREPARE_ONLY       (0 or 1)
#   OVERWRITE          (0 or 1)
#   WANDB_MODE
#   CHECKPOINT_BASE_DIR
#   OUTPUT_DATASET_PARENT
#   OUTPUT_DATASET_TAG
#   EXP_NAME_PREFIX
#   QUALITY_CSV_EPISODE_COLUMN
#   QUALITY_CSV_SCORE_COLUMN
#   EPISODE_INFLUENCE_FIELD
#   MAX_FRAMES_FOR_NORM_STATS
#   UV_CACHE_DIR


# CUDA_VISIBLE_DEVICES=2 nohup bash scripts/retrain_joint_last4_csvq_serial.sh \
#     > /workspace/liushuo/cupid/openpi/tmp/can_mh_grass_last4_strict/retrain_joint_serial_master_0.3.log 2>&1 &
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

TRAIN_CONFIG="${TRAIN_CONFIG:-pi05_can_mh_lora}"
SCORE_PKL="${SCORE_PKL:-/workspace/liushuo/cupid/openpi/tmp/can_mh_grass_last4_strict/influence_grass.pkl}"
QUALITY_SCORE_CSV="${QUALITY_SCORE_CSV:-/workspace/liushuo/cupid/3_model/quality_eval/can_mh_rank_fusion/quality_episodes.csv}"
QUALITY_CSV_EPISODE_COLUMN="${QUALITY_CSV_EPISODE_COLUMN:-episode}"
QUALITY_CSV_SCORE_COLUMN="${QUALITY_CSV_SCORE_COLUMN:-quality_score}"
EPISODE_INFLUENCE_FIELD="${EPISODE_INFLUENCE_FIELD:-episode_influence_scores}"

# RATIOS_STRING="${RATIOS:-0.9 0.8 0.7 0.6 0.5}"
RATIOS_STRING="${RATIOS:-0.3 0.2}"
read -r -a RATIOS <<< "$RATIOS_STRING"

JOINT_QUALITY_POWER="${JOINT_QUALITY_POWER:-1}"
JOINT_INFLUENCE_POWER="${JOINT_INFLUENCE_POWER:-1}"
NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-10000}"
PREPARE_ONLY="${PREPARE_ONLY:-0}"
OVERWRITE="${OVERWRITE:-1}"
WANDB_MODE="${WANDB_MODE:-disabled}"
CHECKPOINT_BASE_DIR="${CHECKPOINT_BASE_DIR:-./checkpoints}"
OUTPUT_DATASET_PARENT="${OUTPUT_DATASET_PARENT:-/workspace/liushuo/cupid/datasets/lerobot}"
OUTPUT_DATASET_TAG="${OUTPUT_DATASET_TAG:-can_mh_joint_keep}"
EXP_NAME_PREFIX="${EXP_NAME_PREFIX:-can_mh_joint_keep}"
MAX_FRAMES_FOR_NORM_STATS="${MAX_FRAMES_FOR_NORM_STATS:-}"
UV_CACHE_DIR="${UV_CACHE_DIR:-./tmp/uv-cache}"
RUN_LOG_DIR="${RUN_LOG_DIR:-/workspace/liushuo/cupid/openpi/tmp/can_mh_grass_last4_strict}"

mkdir -p "$RUN_LOG_DIR"

run_ratio() {
  local ratio="$1"
  local keep_pct
  local ratio_tag
  local output_dataset
  local exp_name
  local run_log
  local -a cmd

  keep_pct="$(python - "$ratio" <<'PY'
import sys
print(int(round(float(sys.argv[1]) * 100)))
PY
)"
  ratio_tag="keep${keep_pct}"
  output_dataset="${OUTPUT_DATASET_PARENT}/${OUTPUT_DATASET_TAG}${keep_pct}_last4_csvq"
  exp_name="${EXP_NAME_PREFIX}${keep_pct}_last4_csvq"
  run_log="${RUN_LOG_DIR}/retrain_joint_${ratio_tag}.log"

  cmd=(
    uv run python retrain_joint.py
    --score-pkl "$SCORE_PKL"
    --quality-score-csv "$QUALITY_SCORE_CSV"
    --quality-csv-episode-column "$QUALITY_CSV_EPISODE_COLUMN"
    --quality-csv-score-column "$QUALITY_CSV_SCORE_COLUMN"
    --episode-influence-field "$EPISODE_INFLUENCE_FIELD"
    --train-config "$TRAIN_CONFIG"
    --keep-ratio "$ratio"
    --joint-quality-power "$JOINT_QUALITY_POWER"
    --joint-influence-power "$JOINT_INFLUENCE_POWER"
    --output-dataset "$output_dataset"
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
  echo "Running joint retrain pipeline for keep ratio: $ratio"
  echo "Output dataset: $output_dataset"
  echo "Exp name: $exp_name"
  echo "Log: $run_log"
  echo "Command: ${cmd[*]}"
  echo "============================================================"

  UV_CACHE_DIR="$UV_CACHE_DIR" WANDB_MODE="$WANDB_MODE" "${cmd[@]}" 2>&1 | tee "$run_log"
}

for ratio in "${RATIOS[@]}"; do
  run_ratio "$ratio"
done

echo "All joint retrain jobs finished."
