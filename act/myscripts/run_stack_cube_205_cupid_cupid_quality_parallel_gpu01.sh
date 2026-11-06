#!/usr/bin/env bash



# nohup bash /workspace/liushuo/cupid/act/myscripts/run_stack_cube_205_cupid_cupid_quality_parallel_gpu01.sh \
#     > /workspace/liushuo/cupid/act/myscripts/output_run_stack_cube_205_cupid_cupid_quality_parallel_gpu01.log 2>&1 &


set -euo pipefail

# 说明：
# - 该脚本只训练 cupid 与 cupid_quality，不包含 mysys。
# - 若 influence 产物不存在，会先在 INFLUENCE_GPU 上补齐一次。
# - 两条方法并行执行：
#   - gpu0: cupid 的 ACT -> 用 kept episodes 训练 OpenPI0.5
#   - gpu1: cupid_quality 的 ACT -> 用 kept episodes 训练 OpenPI0.5
# - 原有 mysys 训练脚本保持不变，可继续单独使用。

ACT_ROOT="/workspace/liushuo/cupid/act"
OPENPI_ROOT="/workspace/liushuo/cupid/openpi"
REPRO_SCRIPT="${ACT_ROOT}/myscripts/reproduce_stack_cube_205_mysys_and_cupid_quality.sh"
OPENPI_PREP_SCRIPT="${OPENPI_ROOT}/myscripts/prepare_openpi_from_act_filter.py"

PYTHON_BIN="${PYTHON_BIN:-python}"
INFLUENCE_GPU="${INFLUENCE_GPU:-0}"
CUPID_GPU="${CUPID_GPU:-0}"
CUPID_QUALITY_GPU="${CUPID_QUALITY_GPU:-1}"

CHECKPOINT_DIR="${CHECKPOINT_DIR:-/workspace/liushuo/cupid/outputs/stack_cube_205_act/checkpoints/300000/pretrained_model}"
DATASET_ROOT="${DATASET_ROOT:-/workspace/liushuo/dataset/stack_cube_205}"
DATASET_REPO_ID="${DATASET_REPO_ID:-local/stack_cube_205}"
ROLLOUT_DIR="${ROLLOUT_DIR:-/workspace/liushuo/stack_cube_act}"

EXP_ROOT="${EXP_ROOT:-/workspace/liushuo/cupid/3_model/stack_cube_205_cupid_cupid_quality_parallel_gpu01}"
INFLUENCE_ROOT="${INFLUENCE_ROOT:-${EXP_ROOT}/influence}"
SCORE_PKL="${SCORE_PKL:-${INFLUENCE_ROOT}/score_episode_window.pkl}"
STAGE_CACHE_DIR="${STAGE_CACHE_DIR:-${INFLUENCE_ROOT}/score}"
STAGE1_TRAIN_PROJ="${STAGE1_TRAIN_PROJ:-${STAGE_CACHE_DIR}/score_episode_window.stage1_train_projections.npy}"

LOG_DIR="${LOG_DIR:-${EXP_ROOT}/logs}"

CUPID_OUTPUT_DIR="${CUPID_OUTPUT_DIR:-${EXP_ROOT}/cupid/act}"
CUPID_QUALITY_ARTIFACT_DIR="${CUPID_QUALITY_ARTIFACT_DIR:-${EXP_ROOT}/cupid_quality/artifacts}"
CUPID_QUALITY_TRAIN_DIR="${CUPID_QUALITY_TRAIN_DIR:-${EXP_ROOT}/cupid_quality/act}"

OPENPI_CONFIG_NAME="${OPENPI_CONFIG_NAME:-pi05_stack_cube_205_lora}"
OPENPI_STEPS="${OPENPI_STEPS:-30000}"
OPENPI_CHECKPOINT_BASE_DIR="${OPENPI_CHECKPOINT_BASE_DIR:-${EXP_ROOT}/openpi_checkpoints}"
OPENPI_MAX_FRAMES_FOR_NORM_STATS="${OPENPI_MAX_FRAMES_FOR_NORM_STATS:-}"
WANDB_MODE_VALUE="${WANDB_MODE_VALUE:-disabled}"
XLA_MEM_FRACTION="${XLA_MEM_FRACTION:-0.9}"

DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-4}"
MAX_STEPS="${MAX_STEPS:-300000}"
LOG_EVERY="${LOG_EVERY:-200}"
SAVE_EVERY="${SAVE_EVERY:-20000}"
SEED="${SEED:-1000}"

CUPID_DROP_QUANTILE="${CUPID_DROP_QUANTILE:-0.2}"
CUPID_QUALITY_DROP_QUANTILE="${CUPID_QUALITY_DROP_QUANTILE:-0.2}"
WINDOW_SIZE="${WINDOW_SIZE:-50}"
WINDOW_STRIDE="${WINDOW_STRIDE:-50}"

mkdir -p \
  "${EXP_ROOT}" \
  "${INFLUENCE_ROOT}" \
  "${STAGE_CACHE_DIR}" \
  "${LOG_DIR}" \
  "${CUPID_OUTPUT_DIR}" \
  "${CUPID_QUALITY_ARTIFACT_DIR}" \
  "${CUPID_QUALITY_TRAIN_DIR}" \
  "${OPENPI_CHECKPOINT_BASE_DIR}"

require_file() {
  local path="$1"
  if [[ ! -f "${path}" ]]; then
    echo "[error] 文件不存在: ${path}" >&2
    exit 1
  fi
}

run_sync() {
  echo
  echo "[run] $*"
  "$@"
}

run_influence_if_needed() {
  if [[ -f "${SCORE_PKL}" && -f "${STAGE1_TRAIN_PROJ}" ]]; then
    echo "[skip] influence 产物已存在，跳过补算。"
    return
  fi

  local job_prefix="stack_cube_205_influence_gpu${INFLUENCE_GPU}"
  local hf_home="/tmp/hf_home_${job_prefix}"
  local hf_cache="/tmp/hf_datasets_cache_${job_prefix}"
  mkdir -p "${hf_home}" "${hf_cache}"

  run_sync env \
    CUDA_VISIBLE_DEVICES="${INFLUENCE_GPU}" \
    HF_HOME="${hf_home}" \
    HF_DATASETS_CACHE="${hf_cache}" \
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
    WINDOW_SIZE="${WINDOW_SIZE}" \
    WINDOW_STRIDE="${WINDOW_STRIDE}" \
    JOB_PREFIX="${job_prefix}" \
    bash "${REPRO_SCRIPT}" influence
}

run_act_method() {
  local method="$1"
  local gpu_id="$2"
  local job_prefix="stack_cube_205_${method}_gpu${gpu_id}"
  local hf_home="/tmp/hf_home_${job_prefix}"
  local hf_cache="/tmp/hf_datasets_cache_${job_prefix}"
  mkdir -p "${hf_home}" "${hf_cache}"

  local -a cmd=(
    env
    CUDA_VISIBLE_DEVICES="${gpu_id}"
    HF_HOME="${hf_home}"
    HF_DATASETS_CACHE="${hf_cache}"
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
    WINDOW_SIZE="${WINDOW_SIZE}"
    WINDOW_STRIDE="${WINDOW_STRIDE}"
  )

  case "${method}" in
    cupid)
      cmd+=(
        CUPID_OUTPUT_DIR="${CUPID_OUTPUT_DIR}"
      )
      ;;
    cupid_quality)
      cmd+=(
        CUPID_QUALITY_DIR="${CUPID_QUALITY_ARTIFACT_DIR}"
        CUPID_QUALITY_TRAIN_DIR="${CUPID_QUALITY_TRAIN_DIR}"
      )
      ;;
    *)
      echo "[error] 未知方法: ${method}" >&2
      exit 1
      ;;
  esac

  cmd+=(bash "${REPRO_SCRIPT}" "${method}")

  echo "[run] ACT method=${method} gpu=${gpu_id}"
  "${cmd[@]}"
}

run_openpi_from_keep() {
  local method="$1"
  local gpu_id="$2"
  local keep_file="$3"
  local output_dir="$4"
  local exp_name="$5"

  require_file "${keep_file}"
  mkdir -p "${output_dir}"

  local -a cmd=(
    uv run python "${OPENPI_PREP_SCRIPT}"
    --train-config "${OPENPI_CONFIG_NAME}"
    --dataset-dir "${DATASET_ROOT}"
    --keep-episodes-txt "${keep_file}"
    --output-dir "${output_dir}"
    --source-tag "${method}"
    --checkpoint-base-dir "${OPENPI_CHECKPOINT_BASE_DIR}"
    --exp-name "${exp_name}"
    --num-train-steps "${OPENPI_STEPS}"
    --overwrite
  )

  if [[ -n "${OPENPI_MAX_FRAMES_FOR_NORM_STATS}" ]]; then
    cmd+=(
      --max-frames-for-norm-stats "${OPENPI_MAX_FRAMES_FOR_NORM_STATS}"
    )
  fi

  echo "[run] OpenPI method=${method} gpu=${gpu_id}"
  (
    cd "${OPENPI_ROOT}"
    env \
      CUDA_VISIBLE_DEVICES="${gpu_id}" \
      WANDB_MODE="${WANDB_MODE_VALUE}" \
      XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_MEM_FRACTION}" \
      "${cmd[@]}"
  )
}

run_method_pipeline() {
  local method="$1"
  local gpu_id="$2"
  local keep_file=""
  local openpi_output_dir=""
  local openpi_exp_name=""

  case "${method}" in
    cupid)
      keep_file="${CUPID_OUTPUT_DIR}/cupid_reweight_keep.txt"
      openpi_output_dir="${EXP_ROOT}/cupid/openpi"
      openpi_exp_name="stack_cube_205_cupid_pi05_gpu${gpu_id}"
      ;;
    cupid_quality)
      keep_file="${CUPID_QUALITY_TRAIN_DIR}/cupid_reweight_keep.txt"
      openpi_output_dir="${EXP_ROOT}/cupid_quality/openpi"
      openpi_exp_name="stack_cube_205_cupid_quality_pi05_gpu${gpu_id}"
      ;;
    *)
      echo "[error] 未知方法: ${method}" >&2
      exit 1
      ;;
  esac

  echo "[start] method=${method} gpu=${gpu_id}"
  run_act_method "${method}" "${gpu_id}"
  run_openpi_from_keep "${method}" "${gpu_id}" "${keep_file}" "${openpi_output_dir}" "${openpi_exp_name}"
  echo "[done] method=${method} gpu=${gpu_id}"
}

declare -a JOB_PIDS=()
declare -a JOB_NAMES=()

spawn_pipeline() {
  local method="$1"
  local gpu_id="$2"
  local log_file="${LOG_DIR}/${method}_gpu${gpu_id}.log"

  (
    run_method_pipeline "${method}" "${gpu_id}"
  ) > "${log_file}" 2>&1 &

  local pid=$!
  JOB_PIDS+=("${pid}")
  JOB_NAMES+=("${method}")
  echo "[spawn] method=${method} gpu=${gpu_id} pid=${pid} log=${log_file}"
}

wait_all_jobs() {
  local rc=0
  local idx
  for idx in "${!JOB_PIDS[@]}"; do
    local pid="${JOB_PIDS[$idx]}"
    local name="${JOB_NAMES[$idx]}"
    if wait "${pid}"; then
      echo "[ok] ${name} finished"
    else
      local status=$?
      echo "[error] ${name} failed with status=${status}" >&2
      rc=1
    fi
  done
  return "${rc}"
}

run_influence_if_needed
spawn_pipeline cupid "${CUPID_GPU}"
spawn_pipeline cupid_quality "${CUPID_QUALITY_GPU}"
wait_all_jobs

echo
echo "[done] cupid 与 cupid_quality 已并行完成。"
echo "[logs] ${LOG_DIR}"
