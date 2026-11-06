#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${ROOT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

DATASET_ROOT="${DATASET_ROOT:-${REPO_ROOT}/datasets/lerobot/can_mh}"
SCORE_PKL="${SCORE_PKL:-${ROOT_DIR}/tmp/can_mh_openvla_window_grass/influence_grass.pkl}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${ROOT_DIR}/runs/can_mh_openvla_lora--10000_chkpt}"
OUTPUT_BASE="${OUTPUT_BASE:-${ROOT_DIR}/tmp/joint_episode_window}"
RUN_NAME="${RUN_NAME:-$(basename "${DATASET_ROOT}")_score_joint_episode_window}"
SELECTION_OUTPUT_DIR="${SELECTION_OUTPUT_DIR:-${OUTPUT_BASE}/${RUN_NAME}/selection}"
TRAIN_OUTPUT_DIR="${TRAIN_OUTPUT_DIR:-${OUTPUT_BASE}/${RUN_NAME}/train}"
RETRAIN_RUN_ROOT="${RETRAIN_RUN_ROOT:-${ROOT_DIR}/runs}"
EXP_NAME="${EXP_NAME:-${RUN_NAME}}"

QUALITY_SOURCE="${QUALITY_SOURCE:-trajectory_eval}"
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
QUALITY_CSV="${QUALITY_CSV:-}"
NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-10000}"
PREPARE_ONLY="${PREPARE_ONLY:-0}"

mkdir -p "${SELECTION_OUTPUT_DIR}" "${TRAIN_OUTPUT_DIR}"

filter_cmd=(
  python "${ROOT_DIR}/vla-scripts/filter_joint_episode_window.py"
  --score-pkl "${SCORE_PKL}"
  --dataset-root "${DATASET_ROOT}"
  --output-dir "${SELECTION_OUTPUT_DIR}"
  --quality-source "${QUALITY_SOURCE}"
  --episode-influence-weight "${EPISODE_INFLUENCE_WEIGHT}"
  --episode-quality-weight "${EPISODE_QUALITY_WEIGHT}"
  --episode-drop-ratio "${EPISODE_DROP_RATIO}"
  --window-size "${WINDOW_SIZE}"
  --window-stride "${WINDOW_STRIDE}"
  --window-local-episode-score-weight "${WINDOW_LOCAL_EPISODE_SCORE_WEIGHT}"
  --window-local-quality-weight "${WINDOW_LOCAL_QUALITY_WEIGHT}"
  --window-local-episode-joint-weight "${WINDOW_LOCAL_EPISODE_JOINT_WEIGHT}"
  --budget-mode auto_marginal
  --min-windows-per-episode "${MIN_WINDOWS_PER_EPISODE}"
  --min-coverage-ratio "${MIN_COVERAGE_RATIO}"
  --marginal-gain-threshold "${MARGINAL_GAIN_THRESHOLD}"
  --relative-gain-threshold "${RELATIVE_GAIN_THRESHOLD}"
)

if [[ "${QUALITY_SOURCE}" == "quality_csv" ]]; then
  QUALITY_CSV="${QUALITY_CSV:?Set QUALITY_CSV when QUALITY_SOURCE=quality_csv}"
  filter_cmd+=(--quality-score-csv "${QUALITY_CSV}")
fi

prepare_cmd=(
  python "${ROOT_DIR}/vla-scripts/prepare_openvla_from_score_joint_episode_window.py"
  --score-pkl "${SCORE_PKL}"
  --selection-npz "${SELECTION_OUTPUT_DIR}/joint_episode_window_selection.npz"
  --checkpoint-dir "${CHECKPOINT_DIR}"
  --output-dir "${TRAIN_OUTPUT_DIR}"
  --retrain-run-root "${RETRAIN_RUN_ROOT}"
  --exp-name "${EXP_NAME}"
  --num-train-steps "${NUM_TRAIN_STEPS}"
  --overwrite
)

if [[ "${PREPARE_ONLY}" == "1" ]]; then
  prepare_cmd+=(--prepare-only)
fi

echo "[run] filter stage"
printf ' %q' "${filter_cmd[@]}"
printf '\n'
"${filter_cmd[@]}"

echo "[run] openvla prepare/train stage"
printf ' %q' "${prepare_cmd[@]}"
printf '\n'
"${prepare_cmd[@]}"
