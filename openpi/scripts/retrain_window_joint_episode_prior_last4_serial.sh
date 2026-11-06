#!/usr/bin/env bash
# Run window+episode-prior retraining serially from an existing last4 GraSS score file.
#
# Pipeline:
# 1. Aggregate sample-level last4 GraSS scores into a window-style pickle.
# 2. Run retrain_window_joint_episode_prior.py serially over multiple keep ratios.
#
# Example:
#   CUDA_VISIBLE_DEVICES=0 bash scripts/retrain_window_joint_episode_prior_last4_serial.sh
#   PREPARE_ONLY=1 RATIOS="0.5 0.3" bash scripts/retrain_window_joint_episode_prior_last4_serial.sh
#   EPISODE_PRIOR_WEIGHT=0.3 WINDOW_SCORE_WEIGHT=0.7 bash scripts/retrain_window_joint_episode_prior_last4_serial.sh
#   WINDOW_SCORE_WEIGHTS="0.9 0.8 0.7" EPISODE_PRIOR_WEIGHTS="0.1 0.2 0.3" \
#     RATIOS="0.9 0.8 0.6" bash scripts/retrain_window_joint_episode_prior_last4_serial.sh
#   ADAPTIVE_PRIOR_MODE=window_std ADAPTIVE_PRIOR_STRENGTH=0.5 INTERACTION_WEIGHT=0.2 \
#     bash scripts/retrain_window_joint_episode_prior_last4_serial.sh
#   ADAPTIVE_PRIOR_MODES="none window_std" ADAPTIVE_PRIOR_STRENGTHS="0.3 0.5" \
#     INTERACTION_WEIGHTS="0.0 0.1" bash scripts/retrain_window_joint_episode_prior_last4_serial.sh
#   MIN_WINDOWS_PER_EPISODE=2 EPISODE_BUDGET_RULE=episode_prior \
#     bash scripts/retrain_window_joint_episode_prior_last4_serial.sh
#
# Example nohup:
#   CUDA_VISIBLE_DEVICES=4 nohup bash scripts/retrain_window_joint_episode_prior_last4_serial.sh \
#     > /workspace/liushuo/cupid/openpi/tmp/can_mh_grass_last4_strict/retrain_window_joint_episode_prior_last4_serial_master.log 2>&1 &
# CUDA_VISIBLE_DEVICES=0 RATIOS="0.7 0.5 0.3" nohup \
#     bash scripts/retrain_window_joint_episode_prior_last4_serial.sh \
#     > /workspace/liushuo/cupid/openpi/tmp/can_mh_grass_last4_strict/retrain_window_joint_episode_prior_last4_serial_master.log 2>&1 &


# CUDA_VISIBLE_DEVICES=3 \
#   ADAPTIVE_PRIOR_MODE=window_std \
#   ADAPTIVE_PRIOR_STRENGTH=0.5 \
#   INTERACTION_WEIGHT=0.1 \
#   MIN_WINDOWS_PER_EPISODE=1 \
#   EPISODE_BUDGET_RULE=episode_prior \
#   nohup bash scripts/retrain_window_joint_episode_prior_last4_serial.sh\
#   > /workspace/liushuo/cupid/openpi/tmp/can_mh_grass_last4_strict/retrain_window_joint_episode_prior_last4_serial_master_3.log 2>&1 &

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

TRAIN_CONFIG="${TRAIN_CONFIG:-pi05_can_mh_lora}"
SCORE_PKL="${SCORE_PKL:-/workspace/liushuo/cupid/openpi/tmp/can_mh_grass_last4_strict/influence_grass.pkl}"
WINDOW_SCORE_PKL="${WINDOW_SCORE_PKL:-/workspace/liushuo/cupid/openpi/tmp/can_mh_grass_last4_strict/influence_grass_window.pkl}"
DATASET_DIR="${DATASET_DIR:-/workspace/liushuo/cupid/datasets/lerobot/can_mh}"

WINDOW_SIZE="${WINDOW_SIZE:-50}"
WINDOW_STRIDE="${WINDOW_STRIDE:-}"
WINDOW_AGGREGATION="${WINDOW_AGGREGATION:-sum}"
WINDOW_QUALITY_SCORE="${WINDOW_QUALITY_SCORE:-zscore_mean}"
AGGREGATE_OVERWRITE="${AGGREGATE_OVERWRITE:-1}"

# RATIOS_STRING="${RATIOS:-0.7 0.5 0.3}"
RATIOS_STRING="${RATIOS:-0.9 0.8 0.6}"
# RATIOS_STRING="${RATIOS:-0.05}"
read -r -a RATIOS <<< "$RATIOS_STRING"

JOINT_RULE="${JOINT_RULE:-sum}"
JOINT_QUALITY_POWER="${JOINT_QUALITY_POWER:-1}"
JOINT_INFLUENCE_POWER="${JOINT_INFLUENCE_POWER:-1}"
WINDOW_SCORE_WEIGHT="${WINDOW_SCORE_WEIGHT:-0.8}"
EPISODE_PRIOR_WEIGHT="${EPISODE_PRIOR_WEIGHT:-0.2}"
ADAPTIVE_PRIOR_MODE="${ADAPTIVE_PRIOR_MODE:-none}"
ADAPTIVE_PRIOR_STRENGTH="${ADAPTIVE_PRIOR_STRENGTH:-0.5}"
INTERACTION_WEIGHT="${INTERACTION_WEIGHT:-0.0}"
FUSION_NORMALIZER="${FUSION_NORMALIZER:-minmax}"
EPISODE_QUALITY_AGGREGATION="${EPISODE_QUALITY_AGGREGATION:-mean}"
MIN_WINDOWS_PER_EPISODE="${MIN_WINDOWS_PER_EPISODE:-0}"
EPISODE_BUDGET_RULE="${EPISODE_BUDGET_RULE:-global}"

WINDOW_SCORE_WEIGHTS_STRING="${WINDOW_SCORE_WEIGHTS:-$WINDOW_SCORE_WEIGHT}"
EPISODE_PRIOR_WEIGHTS_STRING="${EPISODE_PRIOR_WEIGHTS:-$EPISODE_PRIOR_WEIGHT}"
ADAPTIVE_PRIOR_MODES_STRING="${ADAPTIVE_PRIOR_MODES:-$ADAPTIVE_PRIOR_MODE}"
ADAPTIVE_PRIOR_STRENGTHS_STRING="${ADAPTIVE_PRIOR_STRENGTHS:-$ADAPTIVE_PRIOR_STRENGTH}"
INTERACTION_WEIGHTS_STRING="${INTERACTION_WEIGHTS:-$INTERACTION_WEIGHT}"
FUSION_NORMALIZERS_STRING="${FUSION_NORMALIZERS:-$FUSION_NORMALIZER}"
EPISODE_QUALITY_AGGREGATIONS_STRING="${EPISODE_QUALITY_AGGREGATIONS:-$EPISODE_QUALITY_AGGREGATION}"
MIN_WINDOWS_PER_EPISODES_STRING="${MIN_WINDOWS_PER_EPISODES:-$MIN_WINDOWS_PER_EPISODE}"
EPISODE_BUDGET_RULES_STRING="${EPISODE_BUDGET_RULES:-$EPISODE_BUDGET_RULE}"

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
QUALITY_SOURCES_STRING="${QUALITY_SOURCES:-$QUALITY_SOURCE}"

read -r -a WINDOW_SCORE_WEIGHTS <<< "$WINDOW_SCORE_WEIGHTS_STRING"
read -r -a EPISODE_PRIOR_WEIGHTS <<< "$EPISODE_PRIOR_WEIGHTS_STRING"
read -r -a ADAPTIVE_PRIOR_MODES <<< "$ADAPTIVE_PRIOR_MODES_STRING"
read -r -a ADAPTIVE_PRIOR_STRENGTHS <<< "$ADAPTIVE_PRIOR_STRENGTHS_STRING"
read -r -a INTERACTION_WEIGHTS <<< "$INTERACTION_WEIGHTS_STRING"
read -r -a FUSION_NORMALIZERS <<< "$FUSION_NORMALIZERS_STRING"
read -r -a EPISODE_QUALITY_AGGREGATIONS <<< "$EPISODE_QUALITY_AGGREGATIONS_STRING"
read -r -a MIN_WINDOWS_PER_EPISODES <<< "$MIN_WINDOWS_PER_EPISODES_STRING"
read -r -a EPISODE_BUDGET_RULES <<< "$EPISODE_BUDGET_RULES_STRING"
read -r -a QUALITY_SOURCES <<< "$QUALITY_SOURCES_STRING"

NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-10000}"
PREPARE_ONLY="${PREPARE_ONLY:-0}"
OVERWRITE="${OVERWRITE:-1}"
WANDB_MODE="${WANDB_MODE:-disabled}"
CHECKPOINT_BASE_DIR="${CHECKPOINT_BASE_DIR:-./checkpoints}"
OUTPUT_DIR_PARENT="${OUTPUT_DIR_PARENT:-/workspace/liushuo/cupid/openpi/tmp}"
OUTPUT_DIR_TAG="${OUTPUT_DIR_TAG:-can_mh_window_episode_prior_${JOINT_RULE}_keep}"
EXP_NAME_PREFIX="${EXP_NAME_PREFIX:-can_mh_window_episode_prior_${JOINT_RULE}_keep}"
MAX_FRAMES_FOR_NORM_STATS="${MAX_FRAMES_FOR_NORM_STATS:-}"
UV_CACHE_DIR="${UV_CACHE_DIR:-./tmp/uv-cache}"
RUN_LOG_DIR="${RUN_LOG_DIR:-/workspace/liushuo/cupid/openpi/tmp/can_mh_grass_last4_strict}"
RUN_COUNTER=0
TOTAL_RUNS=$(( \
  ${#RATIOS[@]} \
  * ${#WINDOW_SCORE_WEIGHTS[@]} \
  * ${#EPISODE_PRIOR_WEIGHTS[@]} \
  * ${#ADAPTIVE_PRIOR_MODES[@]} \
  * ${#ADAPTIVE_PRIOR_STRENGTHS[@]} \
  * ${#INTERACTION_WEIGHTS[@]} \
  * ${#FUSION_NORMALIZERS[@]} \
  * ${#EPISODE_QUALITY_AGGREGATIONS[@]} \
  * ${#MIN_WINDOWS_PER_EPISODES[@]} \
  * ${#EPISODE_BUDGET_RULES[@]} \
  * ${#QUALITY_SOURCES[@]} \
))

mkdir -p "$RUN_LOG_DIR"

to_pct_tag() {
  python - "$1" <<'PY'
import sys
print(int(round(float(sys.argv[1]) * 100)))
PY
}

is_zero_float() {
  python - "$1" <<'PY'
import sys
value = float(sys.argv[1])
print("1" if abs(value) < 1e-12 else "0")
PY
}

aggregate_windows() {
  local -a cmd
  cmd=(
    uv run python scripts/aggregate_window_from_grass_pkl.py
    --score-pkl "$SCORE_PKL"
    --output-path "$WINDOW_SCORE_PKL"
    --train-config "$TRAIN_CONFIG"
    --dataset-dir "$DATASET_DIR"
    --window-size "$WINDOW_SIZE"
    --window-aggregation "$WINDOW_AGGREGATION"
    --quality-score "$WINDOW_QUALITY_SCORE"
  )

  if [[ -n "$WINDOW_STRIDE" ]]; then
    cmd+=(--window-stride "$WINDOW_STRIDE")
  fi
  if [[ "$AGGREGATE_OVERWRITE" == "1" ]]; then
    cmd+=(--overwrite)
  fi

  echo "============================================================"
  echo "Aggregating last4 GraSS samples into windows"
  echo "Source score: $SCORE_PKL"
  echo "Window score: $WINDOW_SCORE_PKL"
  echo "Command: ${cmd[*]}"
  echo "============================================================"

  UV_CACHE_DIR="$UV_CACHE_DIR" "${cmd[@]}"
}

run_ratio() {
  local ratio="$1"
  local window_score_weight="$2"
  local episode_prior_weight="$3"
  local adaptive_prior_mode="$4"
  local adaptive_prior_strength="$5"
  local interaction_weight="$6"
  local fusion_normalizer="$7"
  local episode_quality_aggregation="$8"
  local min_windows_per_episode="$9"
  local episode_budget_rule="${10}"
  local quality_source="${11}"
  local keep_pct
  local window_weight_tag
  local episode_weight_tag
  local adaptive_strength_tag
  local interaction_weight_tag
  local ratio_tag
  local combo_tag
  local output_dir
  local exp_name
  local run_log
  local -a cmd

  keep_pct="$(to_pct_tag "$ratio")"
  window_weight_tag="$(to_pct_tag "$window_score_weight")"
  episode_weight_tag="$(to_pct_tag "$episode_prior_weight")"
  adaptive_strength_tag="$(to_pct_tag "$adaptive_prior_strength")"
  interaction_weight_tag="$(to_pct_tag "$interaction_weight")"
  ratio_tag="keep${keep_pct}"
  combo_tag="${ratio_tag}_${quality_source}_w${window_weight_tag}_e${episode_weight_tag}_ap${adaptive_prior_mode}_as${adaptive_strength_tag}_i${interaction_weight_tag}_fn${fusion_normalizer}_qa${episode_quality_aggregation}_mw${min_windows_per_episode}_br${episode_budget_rule}"
  output_dir="${OUTPUT_DIR_PARENT}/${OUTPUT_DIR_TAG}${keep_pct}_${quality_source}_w${window_weight_tag}_e${episode_weight_tag}_ap${adaptive_prior_mode}_as${adaptive_strength_tag}_i${interaction_weight_tag}_fn${fusion_normalizer}_qa${episode_quality_aggregation}_mw${min_windows_per_episode}_br${episode_budget_rule}"
  exp_name="${EXP_NAME_PREFIX}${keep_pct}_${quality_source}_w${window_weight_tag}_e${episode_weight_tag}_ap${adaptive_prior_mode}_as${adaptive_strength_tag}_i${interaction_weight_tag}_fn${fusion_normalizer}_qa${episode_quality_aggregation}_mw${min_windows_per_episode}_br${episode_budget_rule}"
  run_log="${RUN_LOG_DIR}/retrain_window_episode_prior_${combo_tag}.log"

  if [[ "$(is_zero_float "$window_score_weight")" == "1" && "$(is_zero_float "$episode_prior_weight")" == "1" ]]; then
    echo "Skipping invalid combo: window_score_weight=0 and episode_prior_weight=0"
    return
  fi

  cmd=(
    uv run python retrain_window_joint_episode_prior.py
    --score-pkl "$WINDOW_SCORE_PKL"
    --train-config "$TRAIN_CONFIG"
    --dataset-dir "$DATASET_DIR"
    --keep-ratio "$ratio"
    --joint-rule "$JOINT_RULE"
    --joint-quality-power "$JOINT_QUALITY_POWER"
    --joint-influence-power "$JOINT_INFLUENCE_POWER"
    --window-score-weight "$window_score_weight"
    --episode-prior-weight "$episode_prior_weight"
    --adaptive-prior-mode "$adaptive_prior_mode"
    --adaptive-prior-strength "$adaptive_prior_strength"
    --interaction-weight "$interaction_weight"
    --fusion-normalizer "$fusion_normalizer"
    --episode-quality-aggregation "$episode_quality_aggregation"
    --min-windows-per-episode "$min_windows_per_episode"
    --episode-budget-rule "$episode_budget_rule"
    --quality-source "$quality_source"
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

  RUN_COUNTER=$((RUN_COUNTER + 1))
  echo "============================================================"
  echo "Running window+episode-prior retrain pipeline [$RUN_COUNTER/$TOTAL_RUNS]"
  echo "Keep ratio: $ratio"
  echo "Window score file: $WINDOW_SCORE_PKL"
  echo "Dataset dir: $DATASET_DIR"
  echo "Joint rule: $JOINT_RULE"
  echo "Weights: window=$window_score_weight episode=$episode_prior_weight"
  echo "Adaptive prior: mode=$adaptive_prior_mode strength=$adaptive_prior_strength"
  echo "Interaction weight: $interaction_weight"
  echo "Fusion normalizer: $fusion_normalizer"
  echo "Episode quality aggregation: $episode_quality_aggregation"
  echo "Episode budget: min_windows=$min_windows_per_episode rule=$episode_budget_rule"
  echo "Quality source: $quality_source"
  echo "Output dir: $output_dir"
  echo "Exp name: $exp_name"
  echo "Log: $run_log"
  echo "Command: ${cmd[*]}"
  echo "============================================================"

  UV_CACHE_DIR="$UV_CACHE_DIR" WANDB_MODE="$WANDB_MODE" "${cmd[@]}" 2>&1 | tee "$run_log"
}

aggregate_windows

echo "Total hyperparameter combinations to consider: $TOTAL_RUNS"

for ratio in "${RATIOS[@]}"; do
  for window_score_weight in "${WINDOW_SCORE_WEIGHTS[@]}"; do
    for episode_prior_weight in "${EPISODE_PRIOR_WEIGHTS[@]}"; do
      for adaptive_prior_mode in "${ADAPTIVE_PRIOR_MODES[@]}"; do
        for adaptive_prior_strength in "${ADAPTIVE_PRIOR_STRENGTHS[@]}"; do
          for interaction_weight in "${INTERACTION_WEIGHTS[@]}"; do
            for fusion_normalizer in "${FUSION_NORMALIZERS[@]}"; do
              for episode_quality_aggregation in "${EPISODE_QUALITY_AGGREGATIONS[@]}"; do
                for min_windows_per_episode in "${MIN_WINDOWS_PER_EPISODES[@]}"; do
                  for episode_budget_rule in "${EPISODE_BUDGET_RULES[@]}"; do
                    for quality_source in "${QUALITY_SOURCES[@]}"; do
                      run_ratio \
                        "$ratio" \
                        "$window_score_weight" \
                        "$episode_prior_weight" \
                        "$adaptive_prior_mode" \
                        "$adaptive_prior_strength" \
                        "$interaction_weight" \
                        "$fusion_normalizer" \
                        "$episode_quality_aggregation" \
                        "$min_windows_per_episode" \
                        "$episode_budget_rule" \
                        "$quality_source"
                    done
                  done
                done
              done
            done
          done
        done
      done
    done
  done
done

echo "All window+episode-prior retrain jobs finished."
