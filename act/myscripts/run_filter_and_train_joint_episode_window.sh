#!/usr/bin/env bash
set -euo pipefail

ACT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${ACT_ROOT}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

DATASET_ROOT="${DATASET_ROOT:?Set DATASET_ROOT}"
OUTPUT_BASE="${OUTPUT_BASE:-${REPO_ROOT}/outputs/joint_episode_window}"

RUN_NAME="${RUN_NAME:-$(basename "${DATASET_ROOT}")_joint_episode_window}"
DATASET_REPO_ID="${DATASET_REPO_ID:-}"
JOB_NAME="${JOB_NAME:-${RUN_NAME}}"
TRAIN_OUTPUT_DIR="${TRAIN_OUTPUT_DIR:-${OUTPUT_BASE}/${RUN_NAME}/train}"
SELECTION_OUTPUT_DIR="${SELECTION_OUTPUT_DIR:-${OUTPUT_BASE}/${RUN_NAME}/selection}"
INFLUENCE_OUTPUT_DIR="${INFLUENCE_OUTPUT_DIR:-${OUTPUT_BASE}/${RUN_NAME}/score}"

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

PRETRAINED_DIR="${PRETRAINED_DIR:-}"
EVAL_ROLLOUT_DIR="${EVAL_ROLLOUT_DIR:-}"
STAGE_CACHE_PREFIX="${STAGE_CACHE_PREFIX:-${INFLUENCE_OUTPUT_DIR}/score}"
INFLUENCE_DEVICE="${INFLUENCE_DEVICE:-cuda:0}"
RECOMPUTE_SCORE="${RECOMPUTE_SCORE:-0}"

MODE="${MODE:-single}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"

TRAIN_DEVICE="${TRAIN_DEVICE:-cuda}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
TRAIN_NUM_WORKERS="${TRAIN_NUM_WORKERS:-4}"
TRAIN_MAX_STEPS="${TRAIN_MAX_STEPS:-100000}"
TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY:-200}"
TRAIN_SAVE_EVERY="${TRAIN_SAVE_EVERY:-20000}"
TRAIN_SEED="${TRAIN_SEED:-1000}"

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

resolve_score_pkl_for_case() {
  local window_size="$1"
  local window_stride="$2"

  if [[ -n "${SCORE_PKL:-}" ]]; then
    if [[ "${MODE}" == "single" || ( "${window_size}" == "${WINDOW_SIZE}" && "${window_stride}" == "${WINDOW_STRIDE}" ) ]]; then
      printf '%s\n' "${SCORE_PKL}"
      return 0
    fi

    local score_dir score_file score_stem score_ext
    score_dir="$(dirname "${SCORE_PKL}")"
    score_file="$(basename "${SCORE_PKL}")"
    if [[ "${score_file}" == *.* ]]; then
      score_ext=".${score_file##*.}"
      score_stem="${score_file%.*}"
    else
      score_ext=""
      score_stem="${score_file}"
    fi
    printf '%s/%s_ws%s_st%s%s\n' "${score_dir}" "${score_stem}" "${window_size}" "${window_stride}" "${score_ext}"
    return 0
  fi

  printf '%s/score_episode_window_ws%s_st%s.pkl\n' "${INFLUENCE_OUTPUT_DIR}" "${window_size}" "${window_stride}"
}

ensure_window_score() {
  local window_size="$1"
  local window_stride="$2"
  local target_score_pkl=""
  target_score_pkl="$(resolve_score_pkl_for_case "${window_size}" "${window_stride}")"

  if [[ "${RECOMPUTE_SCORE}" != "1" && -f "${target_score_pkl}" ]]; then
    echo "[score] reuse ${target_score_pkl}" >&2
    printf '%s\n' "${target_score_pkl}"
    return 0
  fi

  if [[ -z "${PRETRAINED_DIR}" ]]; then
    echo "Set PRETRAINED_DIR when SCORE_PKL is missing or RECOMPUTE_SCORE=1." >&2
    exit 1
  fi
  if [[ -z "${EVAL_ROLLOUT_DIR}" ]]; then
    echo "Set EVAL_ROLLOUT_DIR when SCORE_PKL is missing or RECOMPUTE_SCORE=1." >&2
    exit 1
  fi

  mkdir -p "$(dirname "${target_score_pkl}")" "${INFLUENCE_OUTPUT_DIR}"

  local influence_cmd=(
    "${PYTHON_BIN}" "${ACT_ROOT}/myscripts/cupid_act_influence.py"
    --pretrained_dir "${PRETRAINED_DIR}"
    --train_dataset_path "${DATASET_ROOT}"
    --eval_rollout_dir "${EVAL_ROLLOUT_DIR}"
    --output_path "${target_score_pkl}"
    --stage_cache_prefix "${STAGE_CACHE_PREFIX}"
    --aggregate_mode episode
    --window_size "${window_size}"
    --window_stride "${window_stride}"
    --device "${INFLUENCE_DEVICE}"
  )

  echo "[run] influence stage" >&2
  printf ' %q' "${influence_cmd[@]}" >&2
  printf '\n' >&2
  if ! "${influence_cmd[@]}" >&2; then
    echo "[error] influence stage failed for ${target_score_pkl}" >&2
    exit 1
  fi

  printf '%s\n' "${target_score_pkl}"
}

append_quality_csv_args() {
  local -n _filter_ref="$1"
  local -n _train_ref="$2"
  if [[ "${QUALITY_SOURCE}" == "quality_csv" ]]; then
    QUALITY_CSV="${QUALITY_CSV:?Set QUALITY_CSV when QUALITY_SOURCE=quality_csv}"
    _filter_ref+=(--quality-score-csv "${QUALITY_CSV}")
    _train_ref+=(--quality-score-csv "${QUALITY_CSV}")
  fi
}

append_optional_train_args() {
  local -n _train_ref="$1"
  if [[ -n "${DATASET_REPO_ID}" ]]; then
    _train_ref+=(--dataset.repo_id "${DATASET_REPO_ID}" --policy.repo_id "${DATASET_REPO_ID}")
  fi
  if [[ -n "${PRETRAINED_DIR}" ]]; then
    _train_ref+=(--pretrained-dir "${PRETRAINED_DIR}")
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
  local case_score_pkl=""
  case_score_pkl="$(ensure_window_score "${case_window_size}" "${case_window_stride}")"

  mkdir -p "${case_selection_dir}" "${case_train_dir}"

  local filter_cmd=(
    "${PYTHON_BIN}" "${ACT_ROOT}/myscripts/filter_joint_episode_window.py"
    --score-pkl "${case_score_pkl}"
    --dataset-root "${DATASET_ROOT}"
    --output-dir "${case_selection_dir}"
    --quality-source "${QUALITY_SOURCE}"
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

  local train_cmd=(
    "${PYTHON_BIN}" "${ACT_ROOT}/train_act_cupid_reweight_joint.py"
    --dataset.root "${DATASET_ROOT}"
    --score-pkl "${case_score_pkl}"
    --selection-npz "${case_selection_dir}/joint_episode_window_selection.npz"
    --selection-override-episodes
    --output_dir "${case_train_dir}"
    --job_name "${case_job_name}"
    --policy.type act
    --filter-score-mode joint
    --joint-drop-quantile 0
    --drop-weight-quantile 0
    --drop-bad-ratio 1.1
    --joint-quality-source "${QUALITY_SOURCE}"
    --weight-mode influence
    --use-weighted-sampler
    --segment-mask
    --segment-weight
    --device "${TRAIN_DEVICE}"
    --batch-size "${TRAIN_BATCH_SIZE}"
    --num-workers "${TRAIN_NUM_WORKERS}"
    --max-steps "${TRAIN_MAX_STEPS}"
    --log-every "${TRAIN_LOG_EVERY}"
    --save-every "${TRAIN_SAVE_EVERY}"
    --seed "${TRAIN_SEED}"
  )

  append_quality_csv_args filter_cmd train_cmd
  append_optional_train_args train_cmd

  echo "[case] ${case_name}"
  echo "[score] ${case_score_pkl}"
  echo "[run] filter stage"
  printf ' %q' "${filter_cmd[@]}"
  printf '\n'
  "${filter_cmd[@]}"

  if [[ "${SKIP_TRAIN}" == "1" ]]; then
    echo "[skip] train stage for ${case_name} because SKIP_TRAIN=1"
    return
  fi

  echo "[run] train stage"
  printf ' %q' "${train_cmd[@]}"
  printf '\n'
  "${train_cmd[@]}"
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
