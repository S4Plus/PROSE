#!/usr/bin/env bash
# Run joint episode-window filtering/norm_stats/training serially through
# openpi-script. It can optionally compute GraSS window influence first, then
# compute episode quality scores, and finally run filtering + prepare/train.
# CUDA_VISIBLE_DEVICES=5 \
# DATASETS=transport_mh \
#   INFLUENCE_ROOT=/workspace/liushuo/cupid/openpi/tmp \
#   nohup bash /workspace/liushuo/cupid/openpi/openpi-script/run_train_joint_episode_window_only.sh \
#   > logs/transport_mh_joint_episode_window.log 2>&1 &

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

DATASETS_STRING="${DATASETS:-can_mh square_mh}"
read -r -a DATASETS <<< "$DATASETS_STRING"

INFLUENCE_ROOT="${INFLUENCE_ROOT:-${ROOT_DIR}/tmp}"
DATASET_PARENT="${DATASET_PARENT:-/workspace/liushuo/cupid/datasets/lerobot}"
OUTPUT_BASE="${OUTPUT_BASE:-${ROOT_DIR}/openpi-script/output/joint_episode_window_serial}"
CHECKPOINT_BASE_DIR="${CHECKPOINT_BASE_DIR:-${ROOT_DIR}/checkpoints}"
NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-30000}"
MAX_FRAMES_FOR_NORM_STATS="${MAX_FRAMES_FOR_NORM_STATS:-}"
WANDB_MODE="${WANDB_MODE:-disabled}"
UV_CACHE_DIR="${UV_CACHE_DIR:-${ROOT_DIR}/tmp/uv-cache}"
RUN_INFLUENCE="${RUN_INFLUENCE:-0}"
INFLUENCE_OVERWRITE="${INFLUENCE_OVERWRITE:-0}"
INFLUENCE_SPARSIFICATION="${INFLUENCE_SPARSIFICATION:-random_mask-32*32}"
INFLUENCE_PROJECTION="${INFLUENCE_PROJECTION:-sjlt-1024}"
INFLUENCE_HESSIAN="${INFLUENCE_HESSIAN:-eFIM}"
INFLUENCE_DAMPING="${INFLUENCE_DAMPING:-1e-4}"
INFLUENCE_STAGE1_MICROBATCH_SIZE="${INFLUENCE_STAGE1_MICROBATCH_SIZE:-1}"
INFLUENCE_STAGE2_BATCH_SIZE="${INFLUENCE_STAGE2_BATCH_SIZE:-32}"
INFLUENCE_PROGRESS_EVERY="${INFLUENCE_PROGRESS_EVERY:-1}"
INFLUENCE_MAX_TRAIN_BATCHES="${INFLUENCE_MAX_TRAIN_BATCHES:-}"
INFLUENCE_MAX_EVAL_EPISODES="${INFLUENCE_MAX_EVAL_EPISODES:-}"
INFLUENCE_MAX_EVAL_STEPS_PER_EPISODE="${INFLUENCE_MAX_EVAL_STEPS_PER_EPISODE:-}"
XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.85}"
INFLUENCE_CUDA_VISIBLE_DEVICES="${INFLUENCE_CUDA_VISIBLE_DEVICES:-}"
TRAIN_CUDA_VISIBLE_DEVICES="${TRAIN_CUDA_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-}}"

EPISODE_QUALITY_SOURCE="${EPISODE_QUALITY_SOURCE:-trajectory_eval}"
EPISODE_DROP_RATIO="${EPISODE_DROP_RATIO:-0.10}"
EPISODE_INFLUENCE_WEIGHT="${EPISODE_INFLUENCE_WEIGHT:-0.5}"
EPISODE_QUALITY_WEIGHT="${EPISODE_QUALITY_WEIGHT:-0.5}"
WINDOW_SIZE="${WINDOW_SIZE:-50}"
WINDOW_STRIDE="${WINDOW_STRIDE:-50}"
WINDOW_LOCAL_EPISODE_SCORE_WEIGHT="${WINDOW_LOCAL_EPISODE_SCORE_WEIGHT:-0.4}"
WINDOW_LOCAL_QUALITY_WEIGHT="${WINDOW_LOCAL_QUALITY_WEIGHT:-0.4}"
WINDOW_LOCAL_EPISODE_JOINT_WEIGHT="${WINDOW_LOCAL_EPISODE_JOINT_WEIGHT:-0.2}"
MIN_WINDOWS_PER_EPISODE="${MIN_WINDOWS_PER_EPISODE:-1}"
MIN_COVERAGE_RATIO="${MIN_COVERAGE_RATIO:-0.5}"
MARGINAL_GAIN_THRESHOLD="${MARGINAL_GAIN_THRESHOLD:-0.15}"
RELATIVE_GAIN_THRESHOLD="${RELATIVE_GAIN_THRESHOLD:-0.5}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
OVERWRITE="${OVERWRITE:-1}"

resolve_train_config() {
  local dataset="$1"
  case "$dataset" in
    stack_cube_205) echo "pi05_stack_cube_205_lora" ;;
    can_mh) echo "pi05_can_mh_lora" ;;
    square_mh) echo "pi05_square_mh_lora" ;;
    transport_mh) echo "pi05_transport_mh_lora" ;;
    beat_block_hammer) echo "pi05_beat_block_hammer_lora" ;;
    *)
      echo "Unsupported dataset: $dataset. Supported: stack_cube_205, can_mh, square_mh, transport_mh, beat_block_hammer" >&2
      return 1
      ;;
  esac
}

resolve_dataset_dir() {
  local dataset="$1"
  case "$dataset" in
    stack_cube_205) echo "/workspace/liushuo/dataset/stack_cube_205" ;;
    beat_block_hammer) echo "/workspace/liushuo/dataset/beat_block_hammer_ur5_randomized_500" ;;
    *) echo "${DATASET_PARENT}/${dataset}" ;;
  esac
}

resolve_score_pkl() {
  local dataset="$1"
  echo "${INFLUENCE_ROOT}/${dataset}_grass_window/influence_grass_window.pkl"
}

resolve_checkpoint_dir() {
  local dataset="$1"
  case "$dataset" in
    stack_cube_205)
      echo "${CHECKPOINT_BASE_DIR}/pi05_stack_cube_205_lora/stack_cube_205_pi05_30k/29999"
      ;;
    beat_block_hammer)
      echo "${CHECKPOINT_BASE_DIR}/pi05_beat_block_hammer_lora/beat_block_hammer_pi05_prompt_arm/9999"
      ;;
    *)
      local train_config
      train_config="$(resolve_train_config "$dataset")"
      echo "${CHECKPOINT_BASE_DIR}/${train_config}/${train_config}/9999"
      ;;
  esac
}

resolve_eval_rollouts_dir() {
  local dataset="$1"
  case "$dataset" in
    stack_cube_205)
      echo "/workspace/liushuo/stack_cube_act"
      ;;
    beat_block_hammer)
      echo "/workspace/liushuo/RoboTwin/eval_result/beat_block_hammer/openpi05/demo_randomized_ur5_dual/beat_block_hammer_pi05_prompt_arm_9999/2026-04-01 11:29:34/episodes"
      ;;
    *)
      echo ""
      ;;
  esac
}

run_one() {
  local dataset="$1"
  local train_config
  local dataset_dir
  local score_pkl
  local checkpoint_dir
  local eval_rollouts_dir
  local output_dir
  local run_log
  local quality_dir
  local quality_csv
  local -a cmd
  local -a quality_cmd
  local -a influence_cmd

  run_with_stage_env() {
    local stage_cuda_visible_devices="$1"
    shift
    if [[ -n "$stage_cuda_visible_devices" ]]; then
      CUDA_VISIBLE_DEVICES="$stage_cuda_visible_devices" "$@"
    else
      "$@"
    fi
  }

  train_config="$(resolve_train_config "$dataset")"
  dataset_dir="$(resolve_dataset_dir "$dataset")"
  score_pkl="$(resolve_score_pkl "$dataset")"
  checkpoint_dir="$(resolve_checkpoint_dir "$dataset")"
  eval_rollouts_dir="${EVAL_ROLLOUTS_DIR:-$(resolve_eval_rollouts_dir "$dataset")}"
  output_dir="${OUTPUT_BASE}/${dataset}_joint_episode_window"
  run_log="${output_dir}/pipeline.log"
  quality_dir="${output_dir}/quality"
  quality_csv="${quality_dir}/quality_episodes.csv"

  mkdir -p "$output_dir"
  mkdir -p "$(dirname "$score_pkl")"

  if [[ ! -d "$dataset_dir" ]]; then
    echo "Missing dataset dir: $dataset_dir" >&2
    return 1
  fi
  if [[ "$RUN_INFLUENCE" == "1" ]]; then
    if [[ -z "$eval_rollouts_dir" || ! -d "$eval_rollouts_dir" ]]; then
      echo "Missing eval rollouts dir: $eval_rollouts_dir" >&2
      return 1
    fi
    if [[ ! -d "$checkpoint_dir" ]]; then
      echo "Missing checkpoint dir: $checkpoint_dir" >&2
      return 1
    fi
  elif [[ ! -f "$score_pkl" ]]; then
    echo "Missing score pkl: $score_pkl" >&2
    return 1
  fi

  influence_cmd=(
    uv run python "${ROOT_DIR}/scripts/influence_pi05_lora_grass_window.py"
    --config-name "$train_config"
    --checkpoint-dir "$checkpoint_dir"
    --eval-rollouts-dir "$eval_rollouts_dir"
    --output-path "$score_pkl"
    --sparsification "$INFLUENCE_SPARSIFICATION"
    --projection "$INFLUENCE_PROJECTION"
    --hessian "$INFLUENCE_HESSIAN"
    --damping "$INFLUENCE_DAMPING"
    --stage1-microbatch-size "$INFLUENCE_STAGE1_MICROBATCH_SIZE"
    --stage2-batch-size "$INFLUENCE_STAGE2_BATCH_SIZE"
    --window-size "$WINDOW_SIZE"
    --window-stride "$WINDOW_STRIDE"
    --progress-every "$INFLUENCE_PROGRESS_EVERY"
    --stage-cache-prefix "${score_pkl%.pkl}.grass_cache"
  )
  if [[ -n "$INFLUENCE_MAX_TRAIN_BATCHES" ]]; then
    influence_cmd+=(--max-train-batches "$INFLUENCE_MAX_TRAIN_BATCHES")
  fi
  if [[ -n "$INFLUENCE_MAX_EVAL_EPISODES" ]]; then
    influence_cmd+=(--max-eval-episodes "$INFLUENCE_MAX_EVAL_EPISODES")
  fi
  if [[ -n "$INFLUENCE_MAX_EVAL_STEPS_PER_EPISODE" ]]; then
    influence_cmd+=(--max-eval-steps-per-episode "$INFLUENCE_MAX_EVAL_STEPS_PER_EPISODE")
  fi

  quality_cmd=(
    uv run python "${ROOT_DIR}/openpi-script/eval_cupid_quality.py"
    --dataset.root "$dataset_dir"
    --quality-source "$EPISODE_QUALITY_SOURCE"
    --keep-ratio 1.0
    --output-dir "$quality_dir"
  )

  cmd=(
    uv run python "${ROOT_DIR}/openpi-script/retrain_joint_episode_window.py"
    --score-pkl "$score_pkl"
    --train-config "$train_config"
    --dataset-dir "$dataset_dir"
    --output-dir "$output_dir"
    --checkpoint-base-dir "$CHECKPOINT_BASE_DIR"
    --quality-source quality_csv
    --quality-score-csv "$quality_csv"
    --episode-influence-weight "$EPISODE_INFLUENCE_WEIGHT"
    --episode-quality-weight "$EPISODE_QUALITY_WEIGHT"
    --episode-drop-ratio "$EPISODE_DROP_RATIO"
    --window-size "$WINDOW_SIZE"
    --window-stride "$WINDOW_STRIDE"
    --window-local-episode-score-weight "$WINDOW_LOCAL_EPISODE_SCORE_WEIGHT"
    --window-local-quality-weight "$WINDOW_LOCAL_QUALITY_WEIGHT"
    --window-local-episode-joint-weight "$WINDOW_LOCAL_EPISODE_JOINT_WEIGHT"
    --budget-mode auto_marginal
    --min-windows-per-episode "$MIN_WINDOWS_PER_EPISODE"
    --min-coverage-ratio "$MIN_COVERAGE_RATIO"
    --marginal-gain-threshold "$MARGINAL_GAIN_THRESHOLD"
    --relative-gain-threshold "$RELATIVE_GAIN_THRESHOLD"
  )
  if [[ -n "$NUM_TRAIN_STEPS" ]]; then
    cmd+=(--num-train-steps "$NUM_TRAIN_STEPS")
  fi
  if [[ -n "$MAX_FRAMES_FOR_NORM_STATS" ]]; then
    cmd+=(--max-frames-for-norm-stats "$MAX_FRAMES_FOR_NORM_STATS")
  fi
  if [[ "$SKIP_TRAIN" == "1" ]]; then
    cmd+=(--prepare-only)
  fi
  if [[ "$OVERWRITE" == "1" ]]; then
    cmd+=(--overwrite)
  fi

  echo "============================================================"
  echo "Dataset: $dataset"
  echo "Mode: joint episode-window filtering"
  echo "Episode quality source: $EPISODE_QUALITY_SOURCE"
  echo "Quality csv: $quality_csv"
  echo "Score file: $score_pkl"
  echo "Dataset dir: $dataset_dir"
  echo "Output dir: $output_dir"
  echo "Log: $run_log"
  if [[ "$RUN_INFLUENCE" == "1" ]]; then
    echo "Checkpoint dir: $checkpoint_dir"
    echo "Eval rollouts dir: $eval_rollouts_dir"
    echo "Influence CUDA_VISIBLE_DEVICES: ${INFLUENCE_CUDA_VISIBLE_DEVICES:-<inherit>}"
  fi
  echo "Train CUDA_VISIBLE_DEVICES: ${TRAIN_CUDA_VISIBLE_DEVICES:-<inherit>}"
  echo "============================================================"

  {
    if [[ "$RUN_INFLUENCE" == "1" ]]; then
      if [[ "$INFLUENCE_OVERWRITE" == "1" || ! -f "$score_pkl" ]]; then
        echo "[stage] influence"
        printf ' %q' "${influence_cmd[@]}"
        printf '\n'
        run_with_stage_env "$INFLUENCE_CUDA_VISIBLE_DEVICES" \
          env \
          XLA_PYTHON_CLIENT_MEM_FRACTION="$XLA_PYTHON_CLIENT_MEM_FRACTION" \
          UV_CACHE_DIR="$UV_CACHE_DIR" \
          WANDB_MODE="$WANDB_MODE" \
          "${influence_cmd[@]}"
      else
        echo "[stage] influence"
        echo "Skip influence because score file already exists: $score_pkl"
      fi
    fi

    echo "[stage] quality"
    printf ' %q' "${quality_cmd[@]}"
    printf '\n'
    run_with_stage_env "$TRAIN_CUDA_VISIBLE_DEVICES" \
      env \
      UV_CACHE_DIR="$UV_CACHE_DIR" \
      WANDB_MODE="$WANDB_MODE" \
      "${quality_cmd[@]}"

    echo "[stage] episode_window_filter_train"
    printf ' %q' "${cmd[@]}"
    printf '\n'
    run_with_stage_env "$TRAIN_CUDA_VISIBLE_DEVICES" \
      env \
      UV_CACHE_DIR="$UV_CACHE_DIR" \
      WANDB_MODE="$WANDB_MODE" \
      "${cmd[@]}"
  } 2>&1 | tee "$run_log"
}

for dataset in "${DATASETS[@]}"; do
  run_one "$dataset"
done

echo "All episode-window retrain jobs finished."
