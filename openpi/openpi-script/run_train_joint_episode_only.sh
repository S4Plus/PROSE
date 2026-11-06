#!/usr/bin/env bash
# Run episode-only filtering/norm_stats/training serially for can_mh and square_mh.
#
# This pipeline only performs trajectory-level dropping:
# - no window construction
# - no window-level selection
# - drop 20% episodes by default, i.e. keep 80%

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

DATASETS_STRING="${DATASETS:-can_mh square_mh}"
read -r -a DATASETS <<< "$DATASETS_STRING"

INFLUENCE_ROOT="${INFLUENCE_ROOT:-/workspace/liushuo/cupid/openpi/1_influence}"
DATASET_PARENT="${DATASET_PARENT:-/workspace/liushuo/cupid/datasets/lerobot}"
OUTPUT_BASE="${OUTPUT_BASE:-${ROOT_DIR}/openpi-script/output/episode_joint_serial}"
CHECKPOINT_BASE_DIR="${CHECKPOINT_BASE_DIR:-${ROOT_DIR}/checkpoints}"
NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-10000}"
KEEP_RATIO="${KEEP_RATIO:-0.8}"
WANDB_MODE="${WANDB_MODE:-disabled}"
UV_CACHE_DIR="${UV_CACHE_DIR:-${ROOT_DIR}/tmp/uv-cache}"
PREPARE_ONLY="${PREPARE_ONLY:-0}"
OVERWRITE="${OVERWRITE:-1}"
EPISODE_QUALITY_SOURCE="${EPISODE_QUALITY_SOURCE:-rank_fusion}"

resolve_train_config() {
  local dataset="$1"
  case "$dataset" in
    can_mh) echo "pi05_can_mh_lora" ;;
    square_mh) echo "pi05_square_mh_lora" ;;
    *)
      echo "Unsupported dataset: $dataset" >&2
      return 1
      ;;
  esac
}

to_pct_tag() {
  python - "$1" <<'PY'
import sys
print(int(round(float(sys.argv[1]) * 100)))
PY
}

run_one() {
  local dataset="$1"
  local train_config
  local dataset_dir
  local score_pkl
  local keep_pct
  local output_dir
  local output_dataset
  local exp_name
  local run_log
  local quality_dir
  local quality_csv
  local -a cmd
  local -a quality_cmd

  train_config="$(resolve_train_config "$dataset")"
  dataset_dir="${DATASET_PARENT}/${dataset}"
  score_pkl="${INFLUENCE_ROOT}/${dataset}_grass_window/influence_grass_window.pkl"
  keep_pct="$(to_pct_tag "$KEEP_RATIO")"
  output_dir="${OUTPUT_BASE}/${dataset}_episode_only_keep${keep_pct}"
  output_dataset="${DATASET_PARENT}/${dataset}_joint_keep${keep_pct}_ep_joint_pkl"
  exp_name="${dataset}_joint_keep${keep_pct}_ep_joint_pkl"
  run_log="${output_dir}/pipeline.log"
  quality_dir="${output_dir}/quality"
  quality_csv="${quality_dir}/quality_episodes.csv"

  mkdir -p "$output_dir"

  if [[ ! -d "$dataset_dir" ]]; then
    echo "Missing dataset dir: $dataset_dir" >&2
    return 1
  fi
  if [[ ! -f "$score_pkl" ]]; then
    echo "Missing score pkl: $score_pkl" >&2
    return 1
  fi

  quality_cmd=(
    uv run python "${ROOT_DIR}/openpi-script/eval_cupid_quality.py"
    --dataset.root "$dataset_dir"
    --quality-source "$EPISODE_QUALITY_SOURCE"
    --keep-ratio 1.0
    --output-dir "$quality_dir"
  )

  cmd=(
    uv run python retrain_joint.py
    --score-pkl "$score_pkl"
    --quality-score-csv "$quality_csv"
    --episode-influence-field episode_influence_scores
    --train-config "$train_config"
    --dataset-dir "$dataset_dir"
    --keep-ratio "$KEEP_RATIO"
    --joint-quality-power 1
    --joint-influence-power 1
    --output-dataset "$output_dataset"
    --exp-name "$exp_name"
    --checkpoint-base-dir "$CHECKPOINT_BASE_DIR"
  )
  if [[ -n "$NUM_TRAIN_STEPS" ]]; then
    cmd+=(--num-train-steps "$NUM_TRAIN_STEPS")
  fi
  if [[ "$PREPARE_ONLY" == "1" ]]; then
    cmd+=(--prepare-only)
  fi
  if [[ "$OVERWRITE" == "1" ]]; then
    cmd+=(--overwrite)
  fi

  echo "============================================================"
  echo "Dataset: $dataset"
  echo "Mode: episode-only trajectory filtering"
  echo "Drop ratio: 0.2"
  echo "Keep ratio: $KEEP_RATIO"
  echo "Episode quality source: $EPISODE_QUALITY_SOURCE"
  echo "Score file: $score_pkl"
  echo "Dataset dir: $dataset_dir"
  echo "Quality csv: $quality_csv"
  echo "Output dataset: $output_dataset"
  echo "Log: $run_log"
  echo "============================================================"

  {
    echo "[stage] quality"
    printf ' %q' "${quality_cmd[@]}"
    printf '\n'
    UV_CACHE_DIR="$UV_CACHE_DIR" WANDB_MODE="$WANDB_MODE" "${quality_cmd[@]}"

    echo "[stage] episode_filter_train"
    printf ' %q' "${cmd[@]}"
    printf '\n'
    UV_CACHE_DIR="$UV_CACHE_DIR" WANDB_MODE="$WANDB_MODE" "${cmd[@]}"
  } 2>&1 | tee "$run_log"
}

for dataset in "${DATASETS[@]}"; do
  run_one "$dataset"
done

echo "All episode-only retrain jobs finished."
