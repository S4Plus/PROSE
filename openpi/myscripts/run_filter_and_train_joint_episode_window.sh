#!/usr/bin/env bash
set -euo pipefail

OPENPI_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${OPENPI_ROOT}/.." && pwd)"
cd "${OPENPI_ROOT}"

DATASET_DIR="${DATASET_DIR:-${REPO_ROOT}/datasets/lerobot/can_mh}"
SCORE_PKL="${SCORE_PKL:-${OPENPI_ROOT}/tmp/can_mh_grass_all/influence_grass.pkl}"
OUTPUT_BASE="${OUTPUT_BASE:-${OPENPI_ROOT}/myscripts/output/joint_episode_window}"

RUN_NAME="${RUN_NAME:-$(basename "${DATASET_DIR}")_joint_episode_window}"
JOB_NAME="${JOB_NAME:-${RUN_NAME}}"
SELECTION_OUTPUT_DIR="${SELECTION_OUTPUT_DIR:-${OUTPUT_BASE}/${RUN_NAME}/selection}"
TRAIN_OUTPUT_DIR="${TRAIN_OUTPUT_DIR:-${OUTPUT_BASE}/${RUN_NAME}/train}"

TRAIN_CONFIG="${TRAIN_CONFIG:-pi05_can_mh_lora}"
NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-10000}"
CHECKPOINT_BASE_DIR="${CHECKPOINT_BASE_DIR:-${OPENPI_ROOT}/checkpoints}"
MAX_FRAMES_FOR_NORM_STATS="${MAX_FRAMES_FOR_NORM_STATS:-}"
WANDB_MODE="${WANDB_MODE:-disabled}"
UV_CACHE_DIR="${UV_CACHE_DIR:-${OPENPI_ROOT}/tmp/uv-cache}"

QUALITY_SOURCE="${QUALITY_SOURCE:-trajectory_eval}"
WINDOW_QUALITY_SOURCE="${WINDOW_QUALITY_SOURCE:-dataset_eval}"
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
SKIP_TRAIN="${SKIP_TRAIN:-0}"
OVERWRITE="${OVERWRITE:-1}"
MODE="${MODE:-single}"
SELECTION_NPZ="${SELECTION_NPZ:-${ACT_SELECTION_NPZ:-}}"

EPISODE_DROP_RATIO_SWEEP="${EPISODE_DROP_RATIO_SWEEP:-}"
WINDOW_WEIGHT_SWEEP="${WINDOW_WEIGHT_SWEEP:-}"
WINDOW_SIZE_SWEEP="${WINDOW_SIZE_SWEEP:-}"
WINDOW_STRIDE_SWEEP="${WINDOW_STRIDE_SWEEP:-}"
MIN_COVERAGE_RATIO_SWEEP="${MIN_COVERAGE_RATIO_SWEEP:-}"
MARGINAL_GAIN_THRESHOLD_SWEEP="${MARGINAL_GAIN_THRESHOLD_SWEEP:-}"
RELATIVE_GAIN_THRESHOLD_SWEEP="${RELATIVE_GAIN_THRESHOLD_SWEEP:-}"

mkdir -p "${OUTPUT_BASE}"

sanitize_token() {
  printf '%s' "$1" | tr ' /,' '_' | tr '.' 'p'
}

append_quality_csv_args() {
  local -n _filter_ref="$1"
  if [[ "${QUALITY_SOURCE}" == "quality_csv" ]]; then
    QUALITY_CSV="${QUALITY_CSV:?Set QUALITY_CSV when QUALITY_SOURCE=quality_csv}"
    _filter_ref+=(--quality-score-csv "${QUALITY_CSV}")
  fi
}

run_case() {
  local case_name="$1"
  local case_selection_dir="$2"
  local case_train_dir="$3"
  local case_job_name="$4"
  local case_episode_drop_ratio="$5"
  local case_window_size="$6"
  local case_window_stride="$7"
  local case_window_episode_weight="$8"
  local case_window_quality_weight="$9"
  local case_window_joint_weight="${10}"
  local case_min_coverage_ratio="${11}"
  local case_marginal_gain_threshold="${12}"
  local case_relative_gain_threshold="${13}"
  local case_selection_npz="${SELECTION_NPZ}"

  mkdir -p "${case_selection_dir}" "${case_train_dir}"

  local -a filter_cmd=()

  local prepare_cmd=(
    uv run python "${OPENPI_ROOT}/myscripts/prepare_openpi_from_joint_episode_window.py"
    --score-pkl "${SCORE_PKL}"
    --train-config "${TRAIN_CONFIG}"
    --dataset-dir "${DATASET_DIR}"
    --output-dir "${case_train_dir}"
    --exp-name "${case_job_name}"
    --checkpoint-base-dir "${CHECKPOINT_BASE_DIR}"
    --num-train-steps "${NUM_TRAIN_STEPS}"
  )

  if [[ -z "${case_selection_npz}" ]]; then
    filter_cmd=(
      uv run python "${OPENPI_ROOT}/myscripts/filter_joint_episode_window.py"
      --score-pkl "${SCORE_PKL}"
      --dataset-root "${DATASET_DIR}"
      --output-dir "${case_selection_dir}"
      --quality-source "${QUALITY_SOURCE}"
      --window-quality-source "${WINDOW_QUALITY_SOURCE}"
      --episode-influence-weight "${EPISODE_INFLUENCE_WEIGHT}"
      --episode-quality-weight "${EPISODE_QUALITY_WEIGHT}"
      --episode-drop-ratio "${case_episode_drop_ratio}"
      --window-size "${case_window_size}"
      --window-stride "${case_window_stride}"
      --window-local-episode-score-weight "${case_window_episode_weight}"
      --window-local-quality-weight "${case_window_quality_weight}"
      --window-local-episode-joint-weight "${case_window_joint_weight}"
      --budget-mode auto_marginal
      --min-windows-per-episode "${MIN_WINDOWS_PER_EPISODE}"
      --min-coverage-ratio "${case_min_coverage_ratio}"
      --marginal-gain-threshold "${case_marginal_gain_threshold}"
      --relative-gain-threshold "${case_relative_gain_threshold}"
    )
    append_quality_csv_args filter_cmd
    case_selection_npz="${case_selection_dir}/joint_episode_window_selection.npz"
  fi

  prepare_cmd+=(--selection-npz "${case_selection_npz}")

  if [[ -n "${MAX_FRAMES_FOR_NORM_STATS}" ]]; then
    prepare_cmd+=(--max-frames-for-norm-stats "${MAX_FRAMES_FOR_NORM_STATS}")
  fi
  if [[ "${SKIP_TRAIN}" == "1" ]]; then
    prepare_cmd+=(--prepare-only)
  fi
  if [[ "${OVERWRITE}" == "1" ]]; then
    prepare_cmd+=(--overwrite)
  fi

  echo "[case] ${case_name}"
  if [[ ${#filter_cmd[@]} -gt 0 ]]; then
    echo "[run] filter stage"
    printf ' %q' "${filter_cmd[@]}"
    printf '\n'
    UV_CACHE_DIR="${UV_CACHE_DIR}" WANDB_MODE="${WANDB_MODE}" "${filter_cmd[@]}"
  else
    echo "[skip] filter stage because SELECTION_NPZ is provided"
    echo "[use] existing selection: ${case_selection_npz}"
  fi

  echo "[run] openpi prepare/train stage"
  printf ' %q' "${prepare_cmd[@]}"
  printf '\n'
  UV_CACHE_DIR="${UV_CACHE_DIR}" WANDB_MODE="${WANDB_MODE}" "${prepare_cmd[@]}"
}

if [[ "${MODE}" == "single" ]]; then
  run_case \
    "${RUN_NAME}" \
    "${SELECTION_OUTPUT_DIR}" \
    "${TRAIN_OUTPUT_DIR}" \
    "${JOB_NAME}" \
    "${EPISODE_DROP_RATIO}" \
    "${WINDOW_SIZE}" \
    "${WINDOW_STRIDE}" \
    "${WINDOW_LOCAL_EPISODE_SCORE_WEIGHT}" \
    "${WINDOW_LOCAL_QUALITY_WEIGHT}" \
    "${WINDOW_LOCAL_EPISODE_JOINT_WEIGHT}" \
    "${MIN_COVERAGE_RATIO}" \
    "${MARGINAL_GAIN_THRESHOLD}" \
    "${RELATIVE_GAIN_THRESHOLD}"
  exit 0
fi

if [[ "${MODE}" != "sweep" ]]; then
  echo "Unsupported MODE=${MODE}. Choose single or sweep." >&2
  exit 1
fi

read -r -a episode_drop_ratio_cases <<< "${EPISODE_DROP_RATIO_SWEEP:-${EPISODE_DROP_RATIO}}"
read -r -a window_weight_cases <<< "${WINDOW_WEIGHT_SWEEP:-${WINDOW_LOCAL_EPISODE_SCORE_WEIGHT},${WINDOW_LOCAL_QUALITY_WEIGHT},${WINDOW_LOCAL_EPISODE_JOINT_WEIGHT}}"
read -r -a window_size_cases <<< "${WINDOW_SIZE_SWEEP:-${WINDOW_SIZE}}"
read -r -a window_stride_cases <<< "${WINDOW_STRIDE_SWEEP:-${WINDOW_STRIDE}}"
read -r -a min_coverage_cases <<< "${MIN_COVERAGE_RATIO_SWEEP:-${MIN_COVERAGE_RATIO}}"
read -r -a marginal_gain_cases <<< "${MARGINAL_GAIN_THRESHOLD_SWEEP:-${MARGINAL_GAIN_THRESHOLD}}"
read -r -a relative_gain_cases <<< "${RELATIVE_GAIN_THRESHOLD_SWEEP:-${RELATIVE_GAIN_THRESHOLD}}"

sweep_root="${OUTPUT_BASE}/${RUN_NAME}/sweep"
mkdir -p "${sweep_root}"
echo "[mode] sweep"
echo "[root] ${sweep_root}"

for episode_drop_ratio_case in "${episode_drop_ratio_cases[@]}"; do
  for weight_case in "${window_weight_cases[@]}"; do
    IFS=',' read -r weight_episode weight_quality weight_joint <<< "${weight_case}"
    if [[ -z "${weight_episode}" || -z "${weight_quality}" || -z "${weight_joint}" ]]; then
      echo "Invalid WINDOW_WEIGHT_SWEEP entry: ${weight_case}" >&2
      exit 1
    fi
    for window_size_case in "${window_size_cases[@]}"; do
      for window_stride_case in "${window_stride_cases[@]}"; do
        for min_coverage_case in "${min_coverage_cases[@]}"; do
          for marginal_gain_case in "${marginal_gain_cases[@]}"; do
            for relative_gain_case in "${relative_gain_cases[@]}"; do
              case_tag="edr$(sanitize_token "${episode_drop_ratio_case}")"
              case_tag+="_ws$(sanitize_token "${window_size_case}")"
              case_tag+="_st$(sanitize_token "${window_stride_case}")"
              case_tag+="_w$(sanitize_token "${weight_episode}")-$(sanitize_token "${weight_quality}")-$(sanitize_token "${weight_joint}")"
              case_tag+="_cov$(sanitize_token "${min_coverage_case}")"
              case_tag+="_mg$(sanitize_token "${marginal_gain_case}")"
              case_tag+="_rg$(sanitize_token "${relative_gain_case}")"
              case_dir="${sweep_root}/${case_tag}"
              run_case \
                "${case_tag}" \
                "${case_dir}/selection" \
                "${case_dir}/train" \
                "${JOB_NAME}_${case_tag}" \
                "${episode_drop_ratio_case}" \
                "${window_size_case}" \
                "${window_stride_case}" \
                "${weight_episode}" \
                "${weight_quality}" \
                "${weight_joint}" \
                "${min_coverage_case}" \
                "${marginal_gain_case}" \
                "${relative_gain_case}"
            done
          done
        done
      done
    done
  done
done
