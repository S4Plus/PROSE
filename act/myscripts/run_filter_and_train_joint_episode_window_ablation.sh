#!/usr/bin/env bash
set -euo pipefail

ACT_ROOT="/workspace/liushuo/cupid/act"
ENTRY_SCRIPT="${ACT_ROOT}/myscripts/run_filter_and_train_joint_episode_window.sh"
PYTHON_BIN="${PYTHON_BIN:-python}"

resolve_eval_rollout_dir() {
  local requested="${1:-}"
  local default_root="/workspace/liushuo/cupid/mimicgen/eval/tmp"

  if [[ -n "${requested}" && -d "${requested}" ]]; then
    printf '%s\n' "${requested}"
    return 0
  fi

  local candidates=(
    "${default_root}/can_mh_act_63_73/episodes"
  )
  local path
  for path in "${candidates[@]}"; do
    if [[ -d "${path}" ]]; then
      printf '%s\n' "${path}"
      return 0
    fi
  done

  path="$(find "${default_root}" -maxdepth 3 -type d 2>/dev/null | rg 'can.*(episodes|pkl)$' | head -n 1 || true)"
  if [[ -n "${path}" && -d "${path}" ]]; then
    printf '%s\n' "${path}"
    return 0
  fi

  return 1
}

export BASE_SCORE_PKL="${BASE_SCORE_PKL:-/workspace/liushuo/cupid/1_infulenece_score/can_mh/score.pkl}"
if [[ ! -f "${BASE_SCORE_PKL}" ]]; then
  echo "BASE_SCORE_PKL not found: ${BASE_SCORE_PKL}" >&2
  exit 1
fi

readarray -t base_score_info < <("${PYTHON_BIN}" - <<'PY' "${BASE_SCORE_PKL}"
import pathlib
import pickle
import sys

base_score = pathlib.Path(sys.argv[1])
with base_score.open("rb") as f:
    payload = pickle.load(f)
cfg = payload.get("config", {})

pretrained_dir = cfg.get("pretrained_dir")
train_dataset_path = cfg.get("train_dataset_path")
eval_rollout_dir = cfg.get("eval_rollout_dir")
stage_cache_prefix = cfg.get("stage_cache_prefix") or str(base_score.with_suffix(""))
dataset_name = pathlib.Path(train_dataset_path).name if train_dataset_path else "dataset"

for item in [
    pretrained_dir or "",
    train_dataset_path or "",
    eval_rollout_dir or "",
    stage_cache_prefix,
    str(base_score.parent),
    dataset_name,
]:
    print(item)
PY
)

export PRETRAINED_DIR_FROM_SCORE="${base_score_info[0]}"
export DATASET_ROOT="${DATASET_ROOT:-${base_score_info[1]}}"
requested_eval_rollout="${EVAL_ROLLOUT_DIR:-${base_score_info[2]}}"
if ! resolved_eval_rollout="$(resolve_eval_rollout_dir "${requested_eval_rollout}")"; then
  echo "No valid eval rollout dir found." >&2
  echo "Requested: ${requested_eval_rollout}" >&2
  echo "Set EVAL_ROLLOUT_DIR manually." >&2
  exit 1
fi
export EVAL_ROLLOUT_DIR="${resolved_eval_rollout}"
export STAGE_CACHE_PREFIX="${STAGE_CACHE_PREFIX:-${base_score_info[3]}}"
export SCORE_DIR="${SCORE_DIR:-${base_score_info[4]}}"
dataset_name_from_score="${base_score_info[5]}"

if [[ -z "${DATASET_ROOT}" || -z "${PRETRAINED_DIR_FROM_SCORE}" ]]; then
  echo "Failed to resolve DATASET_ROOT or PRETRAINED_DIR from ${BASE_SCORE_PKL}" >&2
  exit 1
fi

export OUTPUT_BASE="${OUTPUT_BASE:-/workspace/liushuo/cupid/outputs/joint_episode_window_ablation_original}"
export QUALITY_SOURCE="${QUALITY_SOURCE:-trajectory_eval}"
export DATASET_REPO_ID="${DATASET_REPO_ID:-ls/${dataset_name_from_score}}"
export PRETRAINED_DIR="${PRETRAINED_DIR:-${PRETRAINED_DIR_FROM_SCORE}}"
export QUALITY_CSV="${QUALITY_CSV:-}"
export SKIP_TRAIN="${SKIP_TRAIN:-0}"

# Baseline defaults follow the original myscripts entry.
BASE_EPISODE_DROP_RATIO="${BASE_EPISODE_DROP_RATIO:-0.10}"
BASE_WINDOW_SIZE="${BASE_WINDOW_SIZE:-50}"
BASE_WINDOW_STRIDE="${BASE_WINDOW_STRIDE:-50}"
BASE_WINDOW_LOCAL_EPISODE_SCORE_WEIGHT="${BASE_WINDOW_LOCAL_EPISODE_SCORE_WEIGHT:-0.4}"
BASE_WINDOW_LOCAL_QUALITY_WEIGHT="${BASE_WINDOW_LOCAL_QUALITY_WEIGHT:-0.4}"
BASE_WINDOW_LOCAL_EPISODE_JOINT_WEIGHT="${BASE_WINDOW_LOCAL_EPISODE_JOINT_WEIGHT:-0.2}"
BASE_MIN_COVERAGE_RATIO="${BASE_MIN_COVERAGE_RATIO:-0.4}"
BASE_MARGINAL_GAIN_THRESHOLD="${BASE_MARGINAL_GAIN_THRESHOLD:-0.15}"
BASE_RELATIVE_GAIN_THRESHOLD="${BASE_RELATIVE_GAIN_THRESHOLD:-0.5}"

# Training-side model hyperparameters are intentionally not passed here.
# The train script / LeRobot-side defaults are used instead.
export INFLUENCE_DEVICE="${INFLUENCE_DEVICE:-cuda:0}"

find_existing_window_score() {
  local window_size="$1"
  local window_stride="$2"

  local candidates=(
    "${SCORE_DIR}/score_episode_window_ws${window_size}_st${window_stride}.pkl"
    "${SCORE_DIR}/score_episode_window_${window_size}_${window_stride}.pkl"
  )
  if [[ "${window_size}" == "50" && "${window_stride}" == "50" ]]; then
    candidates+=("${SCORE_DIR}/score_episode_window.pkl")
  fi

  local candidate
  for candidate in "${candidates[@]}"; do
    if [[ -f "${candidate}" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done
  return 1
}

ensure_stage_cache() {
  local stage_files=(
    "${STAGE_CACHE_PREFIX}.stage1_train_projections.npy"
    "${STAGE_CACHE_PREFIX}.stage2_return_gradient.pt"
    "${STAGE_CACHE_PREFIX}.stage3_inverse_hvp.pt"
  )
  local f
  for f in "${stage_files[@]}"; do
    if [[ ! -f "${f}" ]]; then
      echo "Required stage cache missing: ${f}" >&2
      exit 1
    fi
  done
}

ensure_window_score() {
  local window_size="$1"
  local window_stride="$2"
  local existing=""
  if existing="$(find_existing_window_score "${window_size}" "${window_stride}")"; then
    printf '%s\n' "${existing}"
    return 0
  fi

  ensure_stage_cache

  local target="${SCORE_DIR}/score_episode_window_ws${window_size}_st${window_stride}.pkl"
  echo "[score] build ${target}" >&2
  echo "[score] base=${BASE_SCORE_PKL} ws=${window_size} st=${window_stride}" >&2
  "${PYTHON_BIN}" "${ACT_ROOT}/myscripts/cupid_act_influence.py" \
    --pretrained_dir "${PRETRAINED_DIR_FROM_SCORE}" \
    --train_dataset_path "${DATASET_ROOT}" \
    --eval_rollout_dir "${EVAL_ROLLOUT_DIR}" \
    --output_path "${target}" \
    --stage_cache_prefix "${STAGE_CACHE_PREFIX}" \
    --aggregate_mode episode \
    --window_size "${window_size}" \
    --window_stride "${window_stride}" \
    --device "${INFLUENCE_DEVICE}"
  printf '%s\n' "${target}"
}

run_case() {
  local case_name="$1"
  local episode_drop_ratio="$2"
  local window_size="$3"
  local window_stride="$4"
  local weight_episode="$5"
  local weight_quality="$6"
  local weight_joint="$7"
  local min_coverage_ratio="$8"
  local marginal_gain_threshold="$9"
  local relative_gain_threshold="${10}"
  local score_pkl=""
  score_pkl="$(ensure_window_score "${window_size}" "${window_stride}")"

  echo "[ablation] case=${case_name}"
  echo "[ablation] score_pkl=${score_pkl}"
  echo "[ablation] quality_source=${QUALITY_SOURCE}"
  echo "[ablation] edr=${episode_drop_ratio} ws=${window_size} st=${window_stride}"
  echo "[ablation] w=${weight_episode},${weight_quality},${weight_joint}"
  echo "[ablation] cov=${min_coverage_ratio} mg=${marginal_gain_threshold} rg=${relative_gain_threshold}"

  OUTPUT_BASE="${OUTPUT_BASE}" \
  RUN_NAME="${case_name}" \
  JOB_NAME="${case_name}" \
  DATASET_ROOT="${DATASET_ROOT}" \
  SCORE_PKL="${score_pkl}" \
  DATASET_REPO_ID="${DATASET_REPO_ID}" \
  PRETRAINED_DIR="${PRETRAINED_DIR}" \
  QUALITY_SOURCE="${QUALITY_SOURCE}" \
  QUALITY_CSV="${QUALITY_CSV}" \
  EPISODE_DROP_RATIO="${episode_drop_ratio}" \
  WINDOW_SIZE="${window_size}" \
  WINDOW_STRIDE="${window_stride}" \
  WINDOW_LOCAL_EPISODE_SCORE_WEIGHT="${weight_episode}" \
  WINDOW_LOCAL_QUALITY_WEIGHT="${weight_quality}" \
  WINDOW_LOCAL_EPISODE_JOINT_WEIGHT="${weight_joint}" \
  MIN_COVERAGE_RATIO="${min_coverage_ratio}" \
  MARGINAL_GAIN_THRESHOLD="${marginal_gain_threshold}" \
  RELATIVE_GAIN_THRESHOLD="${relative_gain_threshold}" \
  SKIP_TRAIN="${SKIP_TRAIN}" \
  bash "${ENTRY_SCRIPT}"
}

run_case \
  "ablate_baseline" \
  "${BASE_EPISODE_DROP_RATIO}" \
  "${BASE_WINDOW_SIZE}" \
  "${BASE_WINDOW_STRIDE}" \
  "${BASE_WINDOW_LOCAL_EPISODE_SCORE_WEIGHT}" \
  "${BASE_WINDOW_LOCAL_QUALITY_WEIGHT}" \
  "${BASE_WINDOW_LOCAL_EPISODE_JOINT_WEIGHT}" \
  "${BASE_MIN_COVERAGE_RATIO}" \
  "${BASE_MARGINAL_GAIN_THRESHOLD}" \
  "${BASE_RELATIVE_GAIN_THRESHOLD}"

for value in 0.05 0.10 0.20; do
  run_case \
    "ablate_edr_${value//./p}" \
    "${value}" \
    "${BASE_WINDOW_SIZE}" \
    "${BASE_WINDOW_STRIDE}" \
    "${BASE_WINDOW_LOCAL_EPISODE_SCORE_WEIGHT}" \
    "${BASE_WINDOW_LOCAL_QUALITY_WEIGHT}" \
    "${BASE_WINDOW_LOCAL_EPISODE_JOINT_WEIGHT}" \
    "${BASE_MIN_COVERAGE_RATIO}" \
    "${BASE_MARGINAL_GAIN_THRESHOLD}" \
    "${BASE_RELATIVE_GAIN_THRESHOLD}"
done

for value in 25 50 75; do
  run_case \
    "ablate_ws_${value}" \
    "${BASE_EPISODE_DROP_RATIO}" \
    "${value}" \
    "${BASE_WINDOW_STRIDE}" \
    "${BASE_WINDOW_LOCAL_EPISODE_SCORE_WEIGHT}" \
    "${BASE_WINDOW_LOCAL_QUALITY_WEIGHT}" \
    "${BASE_WINDOW_LOCAL_EPISODE_JOINT_WEIGHT}" \
    "${BASE_MIN_COVERAGE_RATIO}" \
    "${BASE_MARGINAL_GAIN_THRESHOLD}" \
    "${BASE_RELATIVE_GAIN_THRESHOLD}"
done

for value in 25 50; do
  run_case \
    "ablate_st_${value}" \
    "${BASE_EPISODE_DROP_RATIO}" \
    "${BASE_WINDOW_SIZE}" \
    "${value}" \
    "${BASE_WINDOW_LOCAL_EPISODE_SCORE_WEIGHT}" \
    "${BASE_WINDOW_LOCAL_QUALITY_WEIGHT}" \
    "${BASE_WINDOW_LOCAL_EPISODE_JOINT_WEIGHT}" \
    "${BASE_MIN_COVERAGE_RATIO}" \
    "${BASE_MARGINAL_GAIN_THRESHOLD}" \
    "${BASE_RELATIVE_GAIN_THRESHOLD}"
done

for value in 0.2 0.4 0.6; do
  run_case \
    "ablate_w_episode_${value//./p}" \
    "${BASE_EPISODE_DROP_RATIO}" \
    "${BASE_WINDOW_SIZE}" \
    "${BASE_WINDOW_STRIDE}" \
    "${value}" \
    "${BASE_WINDOW_LOCAL_QUALITY_WEIGHT}" \
    "${BASE_WINDOW_LOCAL_EPISODE_JOINT_WEIGHT}" \
    "${BASE_MIN_COVERAGE_RATIO}" \
    "${BASE_MARGINAL_GAIN_THRESHOLD}" \
    "${BASE_RELATIVE_GAIN_THRESHOLD}"
done

for value in 0.2 0.4 0.6; do
  run_case \
    "ablate_w_quality_${value//./p}" \
    "${BASE_EPISODE_DROP_RATIO}" \
    "${BASE_WINDOW_SIZE}" \
    "${BASE_WINDOW_STRIDE}" \
    "${BASE_WINDOW_LOCAL_EPISODE_SCORE_WEIGHT}" \
    "${value}" \
    "${BASE_WINDOW_LOCAL_EPISODE_JOINT_WEIGHT}" \
    "${BASE_MIN_COVERAGE_RATIO}" \
    "${BASE_MARGINAL_GAIN_THRESHOLD}" \
    "${BASE_RELATIVE_GAIN_THRESHOLD}"
done

for value in 0.0 0.2 0.4; do
  run_case \
    "ablate_w_joint_${value//./p}" \
    "${BASE_EPISODE_DROP_RATIO}" \
    "${BASE_WINDOW_SIZE}" \
    "${BASE_WINDOW_STRIDE}" \
    "${BASE_WINDOW_LOCAL_EPISODE_SCORE_WEIGHT}" \
    "${BASE_WINDOW_LOCAL_QUALITY_WEIGHT}" \
    "${value}" \
    "${BASE_MIN_COVERAGE_RATIO}" \
    "${BASE_MARGINAL_GAIN_THRESHOLD}" \
    "${BASE_RELATIVE_GAIN_THRESHOLD}"
done

for value in 0.3 0.4 0.5; do
  run_case \
    "ablate_cov_${value//./p}" \
    "${BASE_EPISODE_DROP_RATIO}" \
    "${BASE_WINDOW_SIZE}" \
    "${BASE_WINDOW_STRIDE}" \
    "${BASE_WINDOW_LOCAL_EPISODE_SCORE_WEIGHT}" \
    "${BASE_WINDOW_LOCAL_QUALITY_WEIGHT}" \
    "${BASE_WINDOW_LOCAL_EPISODE_JOINT_WEIGHT}" \
    "${value}" \
    "${BASE_MARGINAL_GAIN_THRESHOLD}" \
    "${BASE_RELATIVE_GAIN_THRESHOLD}"
done

for value in 0.10 0.15 0.20; do
  run_case \
    "ablate_mg_${value//./p}" \
    "${BASE_EPISODE_DROP_RATIO}" \
    "${BASE_WINDOW_SIZE}" \
    "${BASE_WINDOW_STRIDE}" \
    "${BASE_WINDOW_LOCAL_EPISODE_SCORE_WEIGHT}" \
    "${BASE_WINDOW_LOCAL_QUALITY_WEIGHT}" \
    "${BASE_WINDOW_LOCAL_EPISODE_JOINT_WEIGHT}" \
    "${BASE_MIN_COVERAGE_RATIO}" \
    "${value}" \
    "${BASE_RELATIVE_GAIN_THRESHOLD}"
done

for value in 0.3 0.5 0.7; do
  run_case \
    "ablate_rg_${value//./p}" \
    "${BASE_EPISODE_DROP_RATIO}" \
    "${BASE_WINDOW_SIZE}" \
    "${BASE_WINDOW_STRIDE}" \
    "${BASE_WINDOW_LOCAL_EPISODE_SCORE_WEIGHT}" \
    "${BASE_WINDOW_LOCAL_QUALITY_WEIGHT}" \
    "${BASE_WINDOW_LOCAL_EPISODE_JOINT_WEIGHT}" \
    "${BASE_MIN_COVERAGE_RATIO}" \
    "${BASE_MARGINAL_GAIN_THRESHOLD}" \
    "${value}"
done
