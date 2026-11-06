#!/usr/bin/env bash
set -euo pipefail

# Serial random ACT baseline sweep by state-action pair retention ratio.
#
# Default ratios:
#   0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1
#
# Example:
  # GPU=4 \
  # DATASET_ROOT=/workspace/liushuo/cupid/datasets/lerobot/square_mh \
  # DATASET_REPO_ID=ls/square_mh \
  # OUTPUT_ROOT=/workspace/liushuo/cupid/outputs/square_mh_random_sa_sweep \
  # LOG_DIR=/workspace/liushuo/cupid/logs/square_mh_random_sa_sweep \
  # nohup bash comparison_random_state_action_ratio/run_random_state_action_ratio_sweep.sh\
  # > logs/square_mh_random_sa_sweep.log 2>&1 &

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
GPU="${GPU:-7}"
DATASET_ROOT="${DATASET_ROOT:-/workspace/liushuo/cupid/datasets/lerobot/square_mh}"
DATASET_REPO_ID="${DATASET_REPO_ID:-ls/square_mh}"
LEROBOT_REPO="${LEROBOT_REPO:-/workspace/liushuo/cupid/lerobot_2.1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/workspace/liushuo/cupid/outputs/square_mh_random_sa_sweep}"
LOG_DIR="${LOG_DIR:-/workspace/liushuo/cupid/logs/square_mh_random_sa_sweep}"
RATIOS="${RATIOS:-0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1}"
SEED="${SEED:-0}"
RUN_PREFIX="${RUN_PREFIX:-square_mh_random_sa}"
DRY_RUN="${DRY_RUN:-0}"
EXTRA_TRAIN_ARGS="${EXTRA_TRAIN_ARGS:-}"

mkdir -p "${OUTPUT_ROOT}" "${LOG_DIR}"

worker_log="${LOG_DIR}/serial_worker_gpu${GPU}.log"

echo "[launch] gpu=${GPU} worker_log=${worker_log}"

nohup bash -c '
set -euo pipefail

PYTHON_BIN="$1"
GPU="$2"
SCRIPT_DIR="$3"
DATASET_ROOT="$4"
DATASET_REPO_ID="$5"
LEROBOT_REPO="$6"
OUTPUT_ROOT="$7"
LOG_DIR="$8"
RATIOS="$9"
SEED="${10}"
RUN_PREFIX="${11}"
DRY_RUN="${12}"
EXTRA_TRAIN_ARGS="${13}"

for ratio in ${RATIOS}; do
  ratio_tag="${ratio//./p}"
  run_name="${RUN_PREFIX}_${ratio_tag}_seed${SEED}"
  out_dir="${OUTPUT_ROOT}/${run_name}"
  log_file="${LOG_DIR}/${run_name}.log"
  selection_json="${OUTPUT_ROOT}/${run_name}_selection.json"

  echo "[$(date "+%Y-%m-%d %H:%M:%S")] [gpu=${GPU}] [start] ratio=${ratio} run=${run_name}"

  cmd=(
    "${PYTHON_BIN}" "${SCRIPT_DIR}/random_train_by_state_action_ratio.py"
    --sample-ratio "${ratio}"
    --sample-seed "${SEED}"
    --dataset-repo-id "${DATASET_REPO_ID}"
    --dataset-root "${DATASET_ROOT}"
    --lerobot-repo "${LEROBOT_REPO}"
    --selection-output "${selection_json}"
  )

  if [[ "${DRY_RUN}" == "1" ]]; then
    cmd+=(--dry-run)
  fi

  extra_train_args=()
  if [[ -n "${EXTRA_TRAIN_ARGS}" ]]; then
    read -r -a extra_train_args <<< "${EXTRA_TRAIN_ARGS}"
  fi

  cmd+=(
    --
    --output_dir "${out_dir}"
    --job_name "${run_name}"
    --policy.type act
    --policy.repo_id "${DATASET_REPO_ID}"
  )
  cmd+=("${extra_train_args[@]}")

  set +e
  CUDA_VISIBLE_DEVICES="${GPU}" PYTHONUNBUFFERED=1 "${cmd[@]}" > "${log_file}" 2>&1
  rc=$?
  set -e

  if [[ ${rc} -eq 0 ]]; then
    echo "[$(date "+%Y-%m-%d %H:%M:%S")] [gpu=${GPU}] [done] ratio=${ratio} run=${run_name}"
  else
    echo "[$(date "+%Y-%m-%d %H:%M:%S")] [gpu=${GPU}] [fail] ratio=${ratio} run=${run_name} rc=${rc}"
  fi
done

echo "[$(date "+%Y-%m-%d %H:%M:%S")] [gpu=${GPU}] queue finished."
' _ \
  "${PYTHON_BIN}" \
  "${GPU}" \
  "${SCRIPT_DIR}" \
  "${DATASET_ROOT}" \
  "${DATASET_REPO_ID}" \
  "${LEROBOT_REPO}" \
  "${OUTPUT_ROOT}" \
  "${LOG_DIR}" \
  "${RATIOS}" \
  "${SEED}" \
  "${RUN_PREFIX}" \
  "${DRY_RUN}" \
  "${EXTRA_TRAIN_ARGS}" \
  > "${worker_log}" 2>&1 &

echo "[launched] pid=$!"
echo "[logs] tail -f ${worker_log}"
