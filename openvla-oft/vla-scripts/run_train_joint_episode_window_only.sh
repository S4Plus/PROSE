#!/usr/bin/env bash
# 串行运行 OpenVLA 联合片段-窗口（joint episode-window）筛选/归一化统计量计算/训练
# 通过 vla-scripts 处理 can_mh 和 square_mh 数据集。

set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

DATASETS_STRING="${DATASETS:-can_mh square_mh}"
read -r -a DATASETS <<< "$DATASETS_STRING"

INFLUENCE_ROOT="${INFLUENCE_ROOT:-${ROOT_DIR}/tmp}"
DATASET_PARENT="${DATASET_PARENT:-/workspace/liushuo/cupid/datasets/lerobot}"
OUTPUT_BASE="${OUTPUT_BASE:-${ROOT_DIR}/tmp/joint_episode_window_serial}"
CHECKPOINT_BASE_DIR="${CHECKPOINT_BASE_DIR:-${ROOT_DIR}/runs}"
RETRAIN_RUN_ROOT="${RETRAIN_RUN_ROOT:-${ROOT_DIR}/runs}"
NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-10000}"

EPISODE_QUALITY_SOURCE="${EPISODE_QUALITY_SOURCE:-rank_fusion}"
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
PREPARE_ONLY="${PREPARE_ONLY:-0}"
OVERWRITE="${OVERWRITE:-1}"

resolve_checkpoint_dir() {
  local dataset="$1"
  case "$dataset" in
    can_mh) echo "${CHECKPOINT_BASE_DIR}/can_mh_openvla_lora--10000_chkpt" ;;
    square_mh) echo "${CHECKPOINT_BASE_DIR}/square_mh_openvla_lora--10000_chkpt" ;;
    *)
      echo "不支持的数据集: $dataset" >&2
      return 1
      ;;
  esac
}

run_one() {
  local dataset="$1"
  local checkpoint_dir
  local dataset_dir
  local score_pkl
  local output_dir
  local run_log
  local quality_dir
  local quality_csv
  local -a quality_cmd
  local -a filter_cmd
  local -a prepare_cmd

  checkpoint_dir="$(resolve_checkpoint_dir "$dataset")"
  dataset_dir="${DATASET_PARENT}/${dataset}"
  score_pkl="${INFLUENCE_ROOT}/${dataset}_openvla_window_grass/influence_grass.pkl"
  output_dir="${OUTPUT_BASE}/${dataset}_joint_episode_window"
  selection_dir="${output_dir}/selection"
  train_dir="${output_dir}/train"
  run_log="${output_dir}/pipeline.log"
  quality_dir="${output_dir}/quality"
  quality_csv="${quality_dir}/quality_episodes.csv"

  mkdir -p "$output_dir" "$selection_dir" "$train_dir"

  if [[ ! -d "$dataset_dir" ]]; then
    echo "缺少数据集目录: $dataset_dir" >&2
    return 1
  fi
  if [[ ! -f "$score_pkl" ]]; then
    echo "缺少分数 pkl 文件: $score_pkl" >&2
    return 1
  fi
  if [[ ! -d "$checkpoint_dir" ]]; then
    echo "缺少检查点目录: $checkpoint_dir" >&2
    return 1
  fi

  quality_cmd=(
    python "${ROOT_DIR}/vla-scripts/eval_cupid_quality.py"
    --dataset.root "$dataset_dir"
    --quality-source "$EPISODE_QUALITY_SOURCE"
    --keep-ratio 1.0
    --output-dir "$quality_dir"
  )

  filter_cmd=(
    python "${ROOT_DIR}/vla-scripts/filter_joint_episode_window.py"
    --score-pkl "$score_pkl"
    --dataset-root "$dataset_dir"
    --output-dir "$selection_dir"
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

  prepare_cmd=(
    python "${ROOT_DIR}/vla-scripts/prepare_openvla_from_score_joint_episode_window.py"
    --score-pkl "$score_pkl"
    --selection-npz "${selection_dir}/joint_episode_window_selection.npz"
    --checkpoint-dir "$checkpoint_dir"
    --output-dir "$train_dir"
    --retrain-run-root "$RETRAIN_RUN_ROOT"
    --exp-name "${dataset}_joint_episode_window"
    --num-train-steps "$NUM_TRAIN_STEPS"
  )
  
  if [[ "$PREPARE_ONLY" == "1" ]]; then
    prepare_cmd+=(--prepare-only)
  fi
  if [[ "$OVERWRITE" == "1" ]]; then
    prepare_cmd+=(--overwrite)
  fi

  echo "============================================================"
  echo "数据集 (Dataset): $dataset"
  echo "模式 (Mode): OpenVLA 联合片段-窗口串行流水线"
  echo "片段质量来源 (Episode quality source): $EPISODE_QUALITY_SOURCE"
  echo "分数文件 (Score file): $score_pkl"
  echo "检查点目录 (Checkpoint dir): $checkpoint_dir"
  echo "输出目录 (Output dir): $output_dir"
  echo "============================================================"

  {
    echo "[阶段] 质量评估 (quality)"
    printf ' %q' "${quality_cmd[@]}"
    printf '\n'
    "${quality_cmd[@]}"

    echo "[阶段] 筛选 (filter)"
    printf ' %q' "${filter_cmd[@]}"
    printf '\n'
    "${filter_cmd[@]}"

    echo "[阶段] 准备/训练 (openvla prepare/train)"
    printf ' %q' "${prepare_cmd[@]}"
    printf '\n'
    "${prepare_cmd[@]}"
  } 2>&1 | tee "$run_log"
}

for dataset in "${DATASETS[@]}"; do
  run_one "$dataset"
done

echo "所有 OpenVLA 片段-窗口重新训练任务已完成。"
