#!/usr/bin/env bash
set -euo pipefail

# One-factor-at-a-time ablation for only three hyperparameters:
#   1) EPISODE_DROP_RATIO
#   2) WINDOW_SIZE
#   3) WINDOW_STRIDE
#
# Baseline is aligned to:
# outputs/joint_episode_window_tune/can_mh_joint_episode_window/sweep/
# edr0p10_ws50_st50_w0p4-0p4-0p2_cov0p5_mg0p15_rg0p5

  # CUDA_VISIBLE_DEVICES=5 \
  # BASE_SCORE_PKL=/workspace/liushuo/cupid/1_infulenece_score/can_mh/score.pkl \
  # nohup bash /workspace/liushuo/cupid/act/myscripts/run_filter_and_train_joint_episode_window_3param_ablation.sh \
  # > /workspace/liushuo/cupid/act/myscripts/output/logs/joint_episode_window_3param_ablation_gpu5.log 2>&1 &

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

sanitize_token() {
  printf '%s' "$1" | tr ' /,' '_' | tr '.' 'p'
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

export OUTPUT_BASE="${OUTPUT_BASE:-/workspace/liushuo/cupid/act/myscripts/output/joint_episode_window_ablation_3params}"
export QUALITY_SOURCE="${QUALITY_SOURCE:-trajectory_eval}"
export DATASET_REPO_ID="${DATASET_REPO_ID:-ls/${dataset_name_from_score}}"
export PRETRAINED_DIR="${PRETRAINED_DIR:-${PRETRAINED_DIR_FROM_SCORE}}"
export QUALITY_CSV="${QUALITY_CSV:-}"
export SKIP_TRAIN="${SKIP_TRAIN:-0}"
export RUN_BASELINE="${RUN_BASELINE:-1}"

# Fixed baseline from:
# edr0p10_ws50_st50_w0p4-0p4-0p2_cov0p5_mg0p15_rg0p5
BASE_EPISODE_DROP_RATIO="${BASE_EPISODE_DROP_RATIO:-0.10}"
BASE_WINDOW_SIZE="${BASE_WINDOW_SIZE:-50}"
BASE_WINDOW_STRIDE="${BASE_WINDOW_STRIDE:-50}"

# Fixed non-ablated selection params from the same baseline.
BASE_EPISODE_INFLUENCE_WEIGHT="${BASE_EPISODE_INFLUENCE_WEIGHT:-0.5}"
BASE_EPISODE_QUALITY_WEIGHT="${BASE_EPISODE_QUALITY_WEIGHT:-0.5}"
BASE_WINDOW_LOCAL_EPISODE_SCORE_WEIGHT="${BASE_WINDOW_LOCAL_EPISODE_SCORE_WEIGHT:-0.4}"
BASE_WINDOW_LOCAL_QUALITY_WEIGHT="${BASE_WINDOW_LOCAL_QUALITY_WEIGHT:-0.4}"
BASE_WINDOW_LOCAL_EPISODE_JOINT_WEIGHT="${BASE_WINDOW_LOCAL_EPISODE_JOINT_WEIGHT:-0.2}"
BASE_MIN_WINDOWS_PER_EPISODE="${BASE_MIN_WINDOWS_PER_EPISODE:-1}"
BASE_MIN_COVERAGE_RATIO="${BASE_MIN_COVERAGE_RATIO:-0.5}"
BASE_MARGINAL_GAIN_THRESHOLD="${BASE_MARGINAL_GAIN_THRESHOLD:-0.15}"
BASE_RELATIVE_GAIN_THRESHOLD="${BASE_RELATIVE_GAIN_THRESHOLD:-0.5}"

# Training-side model hyperparameters are intentionally not passed here.
# The train script / LeRobot-side defaults are used instead.
export INFLUENCE_DEVICE="${INFLUENCE_DEVICE:-cuda:0}"

# By default, EPISODE_DROP_RATIO is ablated independently.
# WINDOW_SIZE and WINDOW_STRIDE use an explicit combo list.
EPISODE_DROP_RATIO_SWEEP_VALUES="${EPISODE_DROP_RATIO_SWEEP_VALUES:-0.05 0.15 0.20}"
WINDOW_SIZE_STRIDE_COMBOS="${WINDOW_SIZE_STRIDE_COMBOS:-25,25 50,25 75,25 75,50 75,75 100,25 100,50 100,75 100,100}"

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
  local score_pkl=""
  score_pkl="$(ensure_window_score "${window_size}" "${window_stride}")"

  echo "[ablation] case=${case_name}"
  echo "[ablation] score_pkl=${score_pkl}"
  echo "[ablation] edr=${episode_drop_ratio} ws=${window_size} st=${window_stride}"

  OUTPUT_BASE="${OUTPUT_BASE}" \
  RUN_NAME="${case_name}" \
  JOB_NAME="${case_name}" \
  DATASET_ROOT="${DATASET_ROOT}" \
  SCORE_PKL="${score_pkl}" \
  DATASET_REPO_ID="${DATASET_REPO_ID}" \
  PRETRAINED_DIR="${PRETRAINED_DIR}" \
  QUALITY_SOURCE="${QUALITY_SOURCE}" \
  QUALITY_CSV="${QUALITY_CSV}" \
  EPISODE_INFLUENCE_WEIGHT="${BASE_EPISODE_INFLUENCE_WEIGHT}" \
  EPISODE_QUALITY_WEIGHT="${BASE_EPISODE_QUALITY_WEIGHT}" \
  EPISODE_DROP_RATIO="${episode_drop_ratio}" \
  WINDOW_SIZE="${window_size}" \
  WINDOW_STRIDE="${window_stride}" \
  WINDOW_LOCAL_EPISODE_SCORE_WEIGHT="${BASE_WINDOW_LOCAL_EPISODE_SCORE_WEIGHT}" \
  WINDOW_LOCAL_QUALITY_WEIGHT="${BASE_WINDOW_LOCAL_QUALITY_WEIGHT}" \
  WINDOW_LOCAL_EPISODE_JOINT_WEIGHT="${BASE_WINDOW_LOCAL_EPISODE_JOINT_WEIGHT}" \
  MIN_WINDOWS_PER_EPISODE="${BASE_MIN_WINDOWS_PER_EPISODE}" \
  MIN_COVERAGE_RATIO="${BASE_MIN_COVERAGE_RATIO}" \
  MARGINAL_GAIN_THRESHOLD="${BASE_MARGINAL_GAIN_THRESHOLD}" \
  RELATIVE_GAIN_THRESHOLD="${BASE_RELATIVE_GAIN_THRESHOLD}" \
  SKIP_TRAIN="${SKIP_TRAIN}" \
  bash "${ENTRY_SCRIPT}"
}

if [[ "${RUN_BASELINE}" == "1" ]]; then
  run_case \
    "baseline_edr$(sanitize_token "${BASE_EPISODE_DROP_RATIO}")_ws${BASE_WINDOW_SIZE}_st${BASE_WINDOW_STRIDE}" \
    "${BASE_EPISODE_DROP_RATIO}" \
    "${BASE_WINDOW_SIZE}" \
    "${BASE_WINDOW_STRIDE}"
fi

read -r -a episode_drop_ratio_cases <<< "${EPISODE_DROP_RATIO_SWEEP_VALUES}"
for value in "${episode_drop_ratio_cases[@]}"; do
  run_case \
    "ablate_edr_$(sanitize_token "${value}")" \
    "${value}" \
    "${BASE_WINDOW_SIZE}" \
    "${BASE_WINDOW_STRIDE}"
done

read -r -a window_size_stride_cases <<< "${WINDOW_SIZE_STRIDE_COMBOS}"
for combo in "${window_size_stride_cases[@]}"; do
  IFS=',' read -r window_size_value window_stride_value <<< "${combo}"
  if [[ -z "${window_size_value}" || -z "${window_stride_value}" ]]; then
    echo "Invalid WINDOW_SIZE_STRIDE_COMBOS entry: ${combo}" >&2
    exit 1
  fi
  if [[ "${window_size_value}" == "${BASE_WINDOW_SIZE}" && "${window_stride_value}" == "${BASE_WINDOW_STRIDE}" ]]; then
    continue
  fi
  run_case \
    "ablate_ws$(sanitize_token "${window_size_value}")_st$(sanitize_token "${window_stride_value}")" \
    "${BASE_EPISODE_DROP_RATIO}" \
    "${window_size_value}" \
    "${window_stride_value}"
done
