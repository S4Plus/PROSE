#!/usr/bin/env bash

set -euo pipefail

# 说明：
# - 当前默认使用 CUDA_VISIBLE_DEVICES=5。
# - 若 influence 产物不存在，会先串行补齐 influence。
# - 训练阶段严格串行执行：mysys -> cupid -> cupid_quality。

ACT_ROOT="/workspace/liushuo/cupid/act"
REPRO_SCRIPT="${ACT_ROOT}/myscripts/reproduce_stack_cube_205_mysys_and_cupid_quality.sh"

GPU_ID="${GPU_ID:-7}"
PYTHON_BIN="${PYTHON_BIN:-python}"

CHECKPOINT_DIR="${CHECKPOINT_DIR:-/workspace/liushuo/cupid/outputs/stack_cube_205_act/checkpoints/300000/pretrained_model}"
DATASET_ROOT="${DATASET_ROOT:-/workspace/liushuo/dataset/stack_cube_205}"
DATASET_REPO_ID="${DATASET_REPO_ID:-local/stack_cube_205}"
ROLLOUT_DIR="${ROLLOUT_DIR:-/workspace/liushuo/stack_cube_act}"

EXP_ROOT="${EXP_ROOT:-/workspace/liushuo/cupid/3_model/stack_cube_205_compare_gpu5}"
INFLUENCE_ROOT="${INFLUENCE_ROOT:-${EXP_ROOT}/influence}"
SCORE_PKL="${SCORE_PKL:-${INFLUENCE_ROOT}/score_episode_window.pkl}"
STAGE_CACHE_DIR="${STAGE_CACHE_DIR:-${INFLUENCE_ROOT}/score}"
STAGE1_TRAIN_PROJ="${STAGE1_TRAIN_PROJ:-${STAGE_CACHE_DIR}/score_episode_window.stage1_train_projections.npy}"

LOG_DIR="${LOG_DIR:-${EXP_ROOT}/logs}"

DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-4}"
MAX_STEPS="${MAX_STEPS:-300000}"
LOG_EVERY="${LOG_EVERY:-200}"
SAVE_EVERY="${SAVE_EVERY:-20000}"
SEED="${SEED:-1000}"

CUPID_DROP_QUANTILE="${CUPID_DROP_QUANTILE:-0.2}"
CUPID_QUALITY_DROP_QUANTILE="${CUPID_QUALITY_DROP_QUANTILE:-0.2}"
MYSYS_EPISODE_DROP_RATIO="${MYSYS_EPISODE_DROP_RATIO:-0.10}"
WINDOW_SIZE="${WINDOW_SIZE:-50}"
WINDOW_STRIDE="${WINDOW_STRIDE:-50}"
WINDOW_LOCAL_EPISODE_SCORE_WEIGHT="${WINDOW_LOCAL_EPISODE_SCORE_WEIGHT:-0.4}"
WINDOW_LOCAL_QUALITY_WEIGHT="${WINDOW_LOCAL_QUALITY_WEIGHT:-0.4}"
WINDOW_LOCAL_EPISODE_JOINT_WEIGHT="${WINDOW_LOCAL_EPISODE_JOINT_WEIGHT:-0.2}"
MIN_COVERAGE_RATIO="${MIN_COVERAGE_RATIO:-0.5}"
MARGINAL_GAIN_THRESHOLD="${MARGINAL_GAIN_THRESHOLD:-0.15}"
RELATIVE_GAIN_THRESHOLD="${RELATIVE_GAIN_THRESHOLD:-0.5}"

mkdir -p "${EXP_ROOT}" "${INFLUENCE_ROOT}" "${STAGE_CACHE_DIR}" "${LOG_DIR}"

run_sync() {
  echo
  echo "[run] $*"
  "$@"
}

run_method() {
  local method="$1"
  local log_file="${LOG_DIR}/${method}.log"
  local job_prefix="stack_cube_205_${method}_gpu${GPU_ID}"

  local -a cmd=(
    env
    CUDA_VISIBLE_DEVICES="${GPU_ID}"
    PYTHON_BIN="${PYTHON_BIN}"
    CHECKPOINT_DIR="${CHECKPOINT_DIR}"
    DATASET_ROOT="${DATASET_ROOT}"
    DATASET_REPO_ID="${DATASET_REPO_ID}"
    ROLLOUT_DIR="${ROLLOUT_DIR}"
    INFLUENCE_ROOT="${INFLUENCE_ROOT}"
    SCORE_PKL="${SCORE_PKL}"
    STAGE_CACHE_DIR="${STAGE_CACHE_DIR}"
    STAGE1_TRAIN_PROJ="${STAGE1_TRAIN_PROJ}"
    DEVICE="${DEVICE}"
    BATCH_SIZE="${BATCH_SIZE}"
    NUM_WORKERS="${NUM_WORKERS}"
    MAX_STEPS="${MAX_STEPS}"
    LOG_EVERY="${LOG_EVERY}"
    SAVE_EVERY="${SAVE_EVERY}"
    SEED="${SEED}"
    JOB_PREFIX="${job_prefix}"
    CUPID_DROP_QUANTILE="${CUPID_DROP_QUANTILE}"
    CUPID_QUALITY_DROP_QUANTILE="${CUPID_QUALITY_DROP_QUANTILE}"
    MYSYS_EPISODE_DROP_RATIO="${MYSYS_EPISODE_DROP_RATIO}"
    WINDOW_SIZE="${WINDOW_SIZE}"
    WINDOW_STRIDE="${WINDOW_STRIDE}"
    WINDOW_LOCAL_EPISODE_SCORE_WEIGHT="${WINDOW_LOCAL_EPISODE_SCORE_WEIGHT}"
    WINDOW_LOCAL_QUALITY_WEIGHT="${WINDOW_LOCAL_QUALITY_WEIGHT}"
    WINDOW_LOCAL_EPISODE_JOINT_WEIGHT="${WINDOW_LOCAL_EPISODE_JOINT_WEIGHT}"
    MIN_COVERAGE_RATIO="${MIN_COVERAGE_RATIO}"
    MARGINAL_GAIN_THRESHOLD="${MARGINAL_GAIN_THRESHOLD}"
    RELATIVE_GAIN_THRESHOLD="${RELATIVE_GAIN_THRESHOLD}"
  )

  case "${method}" in
    cupid)
      cmd+=(
        CUPID_OUTPUT_DIR="${EXP_ROOT}/cupid"
      )
      ;;
    mysys)
      cmd+=(
        MYSYS_OUTPUT_BASE="${EXP_ROOT}/mysys"
        MYSYS_RUN_NAME="stack_cube_205_mysys_gpu${GPU_ID}"
      )
      ;;
    cupid_quality)
      cmd+=(
        CUPID_QUALITY_DIR="${EXP_ROOT}/cupid_quality_artifacts"
        CUPID_QUALITY_TRAIN_DIR="${EXP_ROOT}/cupid_quality"
      )
      ;;
    *)
      echo "[error] unknown method: ${method}" >&2
      exit 1
      ;;
  esac

  cmd+=(bash "${REPRO_SCRIPT}" "${method}")

  echo "[run] ${method} -> ${log_file}"
  "${cmd[@]}" > "${log_file}" 2>&1
}

if [[ ! -f "${SCORE_PKL}" || ! -f "${STAGE1_TRAIN_PROJ}" ]]; then
  run_sync env \
    CUDA_VISIBLE_DEVICES="${GPU_ID}" \
    PYTHON_BIN="${PYTHON_BIN}" \
    CHECKPOINT_DIR="${CHECKPOINT_DIR}" \
    DATASET_ROOT="${DATASET_ROOT}" \
    DATASET_REPO_ID="${DATASET_REPO_ID}" \
    ROLLOUT_DIR="${ROLLOUT_DIR}" \
    INFLUENCE_ROOT="${INFLUENCE_ROOT}" \
    SCORE_PKL="${SCORE_PKL}" \
    STAGE_CACHE_DIR="${STAGE_CACHE_DIR}" \
    STAGE1_TRAIN_PROJ="${STAGE1_TRAIN_PROJ}" \
    DEVICE="${DEVICE}" \
    BATCH_SIZE="${BATCH_SIZE}" \
    NUM_WORKERS="${NUM_WORKERS}" \
    MAX_STEPS="${MAX_STEPS}" \
    LOG_EVERY="${LOG_EVERY}" \
    SAVE_EVERY="${SAVE_EVERY}" \
    SEED="${SEED}" \
    JOB_PREFIX="stack_cube_205_influence_gpu${GPU_ID}" \
    bash "${REPRO_SCRIPT}" influence
fi

run_method mysys
run_method cupid
run_method cupid_quality

echo
echo "[done] 已按顺序完成三条方法。"
echo "[logs] ${LOG_DIR}"
