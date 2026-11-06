#!/usr/bin/env bash
set -euo pipefail

# transport_mh 三路训练调度器：
# 1. 直接使用已完成的 score_episode_window.pkl
# 2. 在 GPU1 上训练 CUPID（仅 influence，去掉 20%）
# 3. 在 GPU2 上训练 CUPID-QUALITY（quality + influence 联合分，去掉 20%）
# 4. 在 GPU3 上执行 joint_episode_window（先筛选，再训练）

ACT_ROOT="/workspace/liushuo/cupid/act"
PYTHON_BIN="${PYTHON_BIN:-python}"

DATASET_ROOT="${DATASET_ROOT:-/workspace/liushuo/cupid/datasets/lerobot/transport_mh}"
DATASET_REPO_ID="${DATASET_REPO_ID:-ls/transport_mh}"

DROP_RATIO="${DROP_RATIO:-0.2}"
DROP_BAD_RATIO="${DROP_BAD_RATIO:-1.1}"
MAX_STEPS="${MAX_STEPS:-100000}"
NUM_WORKERS="${NUM_WORKERS:-0}"
LOG_EVERY="${LOG_EVERY:-20}"
SAVE_EVERY="${SAVE_EVERY:-20000}"
SEED="${SEED:-1000}"
BATCH_SIZE="${BATCH_SIZE:-32}"

CUPID_GPU="${CUPID_GPU:-1}"
CUPID_QUALITY_GPU="${CUPID_QUALITY_GPU:-2}"
JOINT_GPU="${JOINT_GPU:-3}"

SCORE_PKL="${SCORE_PKL:-/workspace/liushuo/cupid/1_infulenece_score/transport_mh/score_episode_window.pkl}"

MODEL_ROOT="${MODEL_ROOT:-/workspace/liushuo/cupid/3_model}"
LOG_ROOT="${LOG_ROOT:-${MODEL_ROOT}/logs/transport_mh_serial}"
QUALITY_OUTPUT_DIR="${QUALITY_OUTPUT_DIR:-${MODEL_ROOT}/quality_eval/transport_mh_cupid_quality_joint_drop20}"
JOINT_ROOT="${JOINT_ROOT:-${MODEL_ROOT}/joint_episode_window/transport_mh_joint_episode_window_drop20}"

CUPID_OUT_DIR="${CUPID_OUT_DIR:-${MODEL_ROOT}/transport_mh_cupid_drop20}"
CUPID_QUALITY_OUT_DIR="${CUPID_QUALITY_OUT_DIR:-${MODEL_ROOT}/transport_mh_cupid_quality_joint_drop20}"
JOINT_SELECTION_DIR="${JOINT_SELECTION_DIR:-${JOINT_ROOT}/selection}"
JOINT_TRAIN_OUT_DIR="${JOINT_TRAIN_OUT_DIR:-${JOINT_ROOT}/train}"

CUPID_JOB_NAME="${CUPID_JOB_NAME:-transport_mh_cupid_drop20}"
CUPID_QUALITY_JOB_NAME="${CUPID_QUALITY_JOB_NAME:-transport_mh_cupid_quality_joint_drop20}"
JOINT_JOB_NAME="${JOINT_JOB_NAME:-transport_mh_joint_episode_window_drop20}"

mkdir -p "${LOG_ROOT}" "${QUALITY_OUTPUT_DIR}" "${JOINT_SELECTION_DIR}" "${JOINT_TRAIN_OUT_DIR}"

timestamp() {
  date "+%Y-%m-%d %H:%M:%S"
}

log() {
  echo "[$(timestamp)] $*"
}

require_file() {
  local path="$1"
  local label="$2"

  if [[ -f "${path}" ]]; then
    return 0
  fi
  log "[fail] ${label} 文件不存在: ${path}"
  return 1
}

score_has_inline_train_projections() {
  local score_pkl="$1"
  "${PYTHON_BIN}" - "${score_pkl}" <<'PY'
import pickle
import sys
from pathlib import Path

path = Path(sys.argv[1])
with path.open("rb") as f:
    payload = pickle.load(f)

has_inline = isinstance(payload, dict) and payload.get("train_projections") is not None
print("1" if has_inline else "0")
PY
}

resolve_stage1_path() {
  local score_pkl="$1"
  local parent
  local stem
  parent="$(dirname "${score_pkl}")"
  stem="$(basename "${score_pkl}" .pkl)"

  local candidate_same_dir="${parent}/${stem}.stage1_train_projections.npy"
  local candidate_score_dir="${parent}/score/${stem}.stage1_train_projections.npy"

  if [[ "$(basename "${parent}")" == "score" ]]; then
    if [[ -f "${candidate_same_dir}" ]]; then
      echo "${candidate_same_dir}"
      return 0
    fi
    echo "${candidate_same_dir}"
    return 0
  fi

  if [[ -f "${candidate_score_dir}" ]]; then
    echo "${candidate_score_dir}"
    return 0
  fi
  if [[ -f "${candidate_same_dir}" ]]; then
    echo "${candidate_same_dir}"
    return 0
  fi
  echo "${candidate_score_dir}"
}

run_logged_command() {
  local run_name="$1"
  local log_file="$2"
  shift 2

  log "[start] ${run_name}"
  printf '[%s] [cmd] ' "$(timestamp)" | tee -a "${log_file}" >/dev/null
  printf '%q ' "$@" | tee -a "${log_file}" >/dev/null
  printf '\n' | tee -a "${log_file}" >/dev/null

  set +e
  "$@" >> "${log_file}" 2>&1
  local rc=$?
  set -e

  if [[ ${rc} -ne 0 ]]; then
    log "[fail] ${run_name} rc=${rc} log=${log_file}"
    return "${rc}"
  fi

  log "[done] ${run_name} log=${log_file}"
}

wait_for_named_job() {
  local job_name="$1"
  local pid="$2"

  if wait "${pid}"; then
    log "[done] 后台任务完成: ${job_name} pid=${pid}"
    return 0
  fi

  local rc=$?
  log "[fail] 后台任务失败: ${job_name} pid=${pid} rc=${rc}"
  return "${rc}"
}

main() {
  log "[info] 使用 GPU 分配: CUPID=${CUPID_GPU} CUPID_QUALITY=${CUPID_QUALITY_GPU} JOINT=${JOINT_GPU}"
  log "[info] DATASET_ROOT=${DATASET_ROOT}"
  log "[info] DROP_RATIO=${DROP_RATIO} MAX_STEPS=${MAX_STEPS} NUM_WORKERS=${NUM_WORKERS} BATCH_SIZE=${BATCH_SIZE}"

  local score_pkl="${SCORE_PKL}"
  require_file "${score_pkl}" "score_pkl"
  log "[info] 采用 score 文件: ${score_pkl}"

  local stage1_arg=()
  if [[ "$(score_has_inline_train_projections "${score_pkl}")" == "0" ]]; then
    local stage1_path
    stage1_path="$(resolve_stage1_path "${score_pkl}")"
    require_file "${stage1_path}" "stage1_train_projections"
    stage1_arg=(--stage1-train-projections "${stage1_path}")
    log "[info] 使用 stage1 缓存: ${stage1_path}"
  else
    log "[info] score_pkl 内含 train_projections，无需额外 stage1 文件"
  fi

  local cupid_log="${LOG_ROOT}/${CUPID_JOB_NAME}.log"
  local quality_log="${LOG_ROOT}/${CUPID_QUALITY_JOB_NAME}.log"
  local joint_filter_log="${LOG_ROOT}/${JOINT_JOB_NAME}_filter.log"
  local joint_train_log="${LOG_ROOT}/${JOINT_JOB_NAME}_train.log"

  local cupid_hf_home="/tmp/hf_home_${CUPID_JOB_NAME}"
  local cupid_hf_cache="/tmp/hf_datasets_cache_${CUPID_JOB_NAME}"
  mkdir -p "${cupid_hf_home}" "${cupid_hf_cache}"

  (
    run_logged_command "${CUPID_JOB_NAME}" "${cupid_log}" \
      env CUDA_VISIBLE_DEVICES="${CUPID_GPU}" \
        HF_HOME="${cupid_hf_home}" \
        HF_DATASETS_CACHE="${cupid_hf_cache}" \
        PYTHONUNBUFFERED=1 \
        "${PYTHON_BIN}" "${ACT_ROOT}/train_act_cupid_reweight.py" \
        --dataset.root "${DATASET_ROOT}" \
        --dataset.repo_id "${DATASET_REPO_ID}" \
        --policy.repo_id "${DATASET_REPO_ID}" \
        --score-pkl "${score_pkl}" \
        --output-dir "${CUPID_OUT_DIR}" \
        --job_name "${CUPID_JOB_NAME}" \
        --device cuda \
        --batch-size "${BATCH_SIZE}" \
        --num-workers "${NUM_WORKERS}" \
        --max-steps "${MAX_STEPS}" \
        --log-every "${LOG_EVERY}" \
        --save-every "${SAVE_EVERY}" \
        --seed "${SEED}" \
        --drop-bad-ratio "${DROP_BAD_RATIO}" \
        --drop-weight-quantile "${DROP_RATIO}" \
        --weight-mode influence \
        --use-weighted-sampler
  ) &
  local cupid_pid=$!
  log "[spawn] ${CUPID_JOB_NAME} pid=${cupid_pid} gpu=${CUPID_GPU}"

  local quality_hf_home="/tmp/hf_home_${CUPID_QUALITY_JOB_NAME}"
  local quality_hf_cache="/tmp/hf_datasets_cache_${CUPID_QUALITY_JOB_NAME}"
  mkdir -p "${quality_hf_home}" "${quality_hf_cache}"

  (
    run_logged_command "${CUPID_QUALITY_JOB_NAME}" "${quality_log}" \
      env CUDA_VISIBLE_DEVICES="${CUPID_QUALITY_GPU}" \
        HF_HOME="${quality_hf_home}" \
        HF_DATASETS_CACHE="${quality_hf_cache}" \
        PYTHONUNBUFFERED=1 \
        "${PYTHON_BIN}" "${ACT_ROOT}/cupid_quality/run_cupid_quality_train.py" \
        --score-pkl "${score_pkl}" \
        "${stage1_arg[@]}" \
        --quality-output-dir "${QUALITY_OUTPUT_DIR}" \
        --dataset.root "${DATASET_ROOT}" \
        --dataset.repo_id "${DATASET_REPO_ID}" \
        --train-output-dir "${CUPID_QUALITY_OUT_DIR}" \
        --job-name "${CUPID_QUALITY_JOB_NAME}" \
        --train-device cuda \
        --normalize-by-trajectory-length \
        --normalizer quantile \
        --quantile-low 0.05 \
        --quantile-high 0.95 \
        --joint-drop-quantile "${DROP_RATIO}" \
        --joint-quality-power 1.0 \
        --joint-influence-power 1.0 \
        --weight-mode influence \
        --use-weighted-sampler \
        --num-workers "${NUM_WORKERS}" \
        --max-steps "${MAX_STEPS}" \
        --log-every "${LOG_EVERY}" \
        --save-every "${SAVE_EVERY}" \
        --seed "${SEED}"
  ) &
  local quality_pid=$!
  log "[spawn] ${CUPID_QUALITY_JOB_NAME} pid=${quality_pid} gpu=${CUPID_QUALITY_GPU}"

  (
    run_logged_command "${JOINT_JOB_NAME}_filter" "${joint_filter_log}" \
      env CUDA_VISIBLE_DEVICES="${JOINT_GPU}" \
        PYTHONUNBUFFERED=1 \
        "${PYTHON_BIN}" "${ACT_ROOT}/myscripts/filter_joint_episode_window.py" \
        --score-pkl "${score_pkl}" \
        --dataset-root "${DATASET_ROOT}" \
        --output-dir "${JOINT_SELECTION_DIR}" \
        --quality-source trajectory_eval \
        --episode-influence-weight 0.5 \
        --episode-quality-weight 0.5 \
        --episode-drop-ratio "${DROP_RATIO}" \
        --window-size 50 \
        --window-stride 50 \
        --window-local-episode-score-weight 0.4 \
        --window-local-quality-weight 0.4 \
        --window-local-episode-joint-weight 0.2 \
        --budget-mode auto_marginal \
        --min-windows-per-episode 1 \
        --min-coverage-ratio 0.5 \
        --marginal-gain-threshold 0.15 \
        --relative-gain-threshold 0.5

    local joint_hf_home="/tmp/hf_home_${JOINT_JOB_NAME}"
    local joint_hf_cache="/tmp/hf_datasets_cache_${JOINT_JOB_NAME}"
    mkdir -p "${joint_hf_home}" "${joint_hf_cache}"

    run_logged_command "${JOINT_JOB_NAME}_train" "${joint_train_log}" \
      env CUDA_VISIBLE_DEVICES="${JOINT_GPU}" \
        HF_HOME="${joint_hf_home}" \
        HF_DATASETS_CACHE="${joint_hf_cache}" \
        PYTHONUNBUFFERED=1 \
        "${PYTHON_BIN}" "${ACT_ROOT}/train_act_cupid_reweight_joint.py" \
        --dataset.root "${DATASET_ROOT}" \
        --dataset.repo_id "${DATASET_REPO_ID}" \
        --policy.repo_id "${DATASET_REPO_ID}" \
        --score-pkl "${score_pkl}" \
        --selection-npz "${JOINT_SELECTION_DIR}/joint_episode_window_selection.npz" \
        --selection-override-episodes \
        --output-dir "${JOINT_TRAIN_OUT_DIR}" \
        --job_name "${JOINT_JOB_NAME}" \
        --device cuda \
        --batch-size "${BATCH_SIZE}" \
        --num-workers "${NUM_WORKERS}" \
        --max-steps "${MAX_STEPS}" \
        --log-every "${LOG_EVERY}" \
        --save-every "${SAVE_EVERY}" \
        --seed "${SEED}" \
        --filter-score-mode joint \
        --joint-drop-quantile 0 \
        --drop-weight-quantile 0 \
        --drop-bad-ratio "${DROP_BAD_RATIO}" \
        --joint-quality-source trajectory_eval \
        --weight-mode influence \
        --use-weighted-sampler \
        --segment-mask \
        --segment-weight
  ) &
  local joint_pid=$!
  log "[spawn] ${JOINT_JOB_NAME} pid=${joint_pid} gpu=${JOINT_GPU}"

  local rc=0
  wait_for_named_job "${CUPID_JOB_NAME}" "${cupid_pid}" || rc=$?
  wait_for_named_job "${CUPID_QUALITY_JOB_NAME}" "${quality_pid}" || rc=$?
  wait_for_named_job "${JOINT_JOB_NAME}" "${joint_pid}" || rc=$?

  if [[ ${rc} -ne 0 ]]; then
    log "[fail] 存在训练任务失败，请检查日志"
    return "${rc}"
  fi

  log "[all-done] 三路训练任务已全部完成"
}

main "$@"
