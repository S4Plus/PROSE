#!/usr/bin/env bash
set -euo pipefail

# Joint training scheduler:
# - one worker per GPU
# - each worker executes its queue sequentially
# - vary only score file + joint drop quantile
#
# Optional env overrides:
#   GPUS="7 6 5 4"
#   GPU=7
#   MAX_STEPS=100000
#   SCORES="/path/a.pkl /path/b.pkl"
#   RATIOS="0.1 0.2 0.3 0.4 0.5"
#   RUN_PREFIX="can_mh_jointsr"
#   OUT_ROOT="/workspace/liushuo/cupid/3_model/joint_by_score_ratio"
#   LOG_DIR="/workspace/liushuo/cupid/3_model/logs/joint_by_score_ratio"
#   QUALITY_CSV="/workspace/liushuo/cupid/3_model/quality_eval/can_mh_rank_fusion/quality_episodes.csv"

MAX_STEPS="${MAX_STEPS:-100000}"
RUN_PREFIX="${RUN_PREFIX:-can_mh_jointsr}"
OUT_ROOT="${OUT_ROOT:-/workspace/liushuo/cupid/3_model/joint_by_score_ratio}"
LOG_DIR="${LOG_DIR:-/workspace/liushuo/cupid/3_model/logs/joint_by_score_ratio}"
QUEUE_DIR="${QUEUE_DIR:-/workspace/liushuo/cupid/3_model/job_queues/joint_by_score_ratio}"

if [[ -n "${GPUS:-}" ]]; then
  read -r -a GPU_LIST <<< "${GPUS}"
elif [[ -n "${GPU:-}" ]]; then
  GPU_LIST=("${GPU}")
else
  GPU_LIST=(7 6 5 4)
fi
if [[ "${#GPU_LIST[@]}" -eq 0 ]]; then
  echo "[error] no GPUs provided"
  exit 1
fi

DATASET_ROOT="/workspace/liushuo/cupid/datasets/lerobot/can_mh"
DATASET_REPO_ID="ls/can_mh"
QUALITY_CSV="${QUALITY_CSV:-/workspace/liushuo/cupid/3_model/quality_eval/can_mh_rank_fusion/quality_episodes.csv}"
SCRIPT="/workspace/liushuo/cupid/act/train_act_cupid_reweight_joint.py"

mkdir -p "${OUT_ROOT}" "${LOG_DIR}" "${QUEUE_DIR}"

if [[ -n "${SCORES:-}" ]]; then
  read -r -a SCORE_LIST <<< "${SCORES}"
else
  SCORE_LIST=(
    "/workspace/liushuo/cupid/1_infulenece_score/can_mh/score_last4.pkl"
    "/workspace/liushuo/cupid/1_infulenece_score/can_mh/score_last8.pkl"
    "/workspace/liushuo/cupid/1_infulenece_score/can_mh/score_last12.pkl"
  )
fi

if [[ -n "${RATIOS:-}" ]]; then
  read -r -a RATIO_LIST <<< "${RATIOS}"
else
  RATIO_LIST=(0.1 0.2 0.3 0.4 0.5)
fi

# Reset queue files.
for gpu in "${GPU_LIST[@]}"; do
  : > "${QUEUE_DIR}/queue_gpu${gpu}.tsv"
done

# Build queues in round-robin.
idx=0
for score_pkl in "${SCORE_LIST[@]}"; do
  if [[ ! -f "${score_pkl}" ]]; then
    echo "[skip] missing score file: ${score_pkl}"
    continue
  fi
  score_tag="$(basename "${score_pkl}" .pkl)"

  for q in "${RATIO_LIST[@]}"; do
    gpu="${GPU_LIST[$((idx % ${#GPU_LIST[@]}))]}"
    qtag="$(echo "${q}" | tr -d '.')"
    run_name="${RUN_PREFIX}_${score_tag}_dq${qtag}"
    out_dir="${OUT_ROOT}/${run_name}"
    log_file="${LOG_DIR}/${run_name}_gpu${gpu}.log"
    printf "%s\t%s\t%s\t%s\t%s\n" \
      "${score_pkl}" "${q}" "${run_name}" "${out_dir}" "${log_file}" \
      >> "${QUEUE_DIR}/queue_gpu${gpu}.tsv"
    idx=$((idx + 1))
  done
done

echo "[info] queue files generated under ${QUEUE_DIR}"
for gpu in "${GPU_LIST[@]}"; do
  echo "----- GPU ${gpu} queue -----"
  cat "${QUEUE_DIR}/queue_gpu${gpu}.tsv"
done

# One serial worker per GPU.
for gpu in "${GPU_LIST[@]}"; do
  queue_file="${QUEUE_DIR}/queue_gpu${gpu}.tsv"
  worker_log="${LOG_DIR}/serial_worker_gpu${gpu}.log"

  nohup bash -c '
set -euo pipefail
QUEUE_FILE="$1"
GPU="$2"
SCRIPT="$3"
DATASET_ROOT="$4"
DATASET_REPO_ID="$5"
QUALITY_CSV="$6"
MAX_STEPS="$7"

while IFS=$'\''\t'\'' read -r SCORE_PKL Q RUN_NAME OUT_DIR LOG_FILE; do
  [[ -z "${SCORE_PKL}" ]] && continue
  START_TS="$(date "+%Y-%m-%d %H:%M:%S")"
  echo "[${START_TS}] [gpu=${GPU}] [start] ${RUN_NAME} q=${Q} score=$(basename "${SCORE_PKL}")"

  HF_HOME="/tmp/hf_home_${RUN_NAME}"
  HF_DATASETS_CACHE="/tmp/hf_datasets_cache_${RUN_NAME}"
  mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}"

  set +e
  CUDA_VISIBLE_DEVICES="${GPU}" HF_HOME="${HF_HOME}" HF_DATASETS_CACHE="${HF_DATASETS_CACHE}" PYTHONUNBUFFERED=1 \
  python "${SCRIPT}" \
    --dataset.root "${DATASET_ROOT}" \
    --dataset.repo_id "${DATASET_REPO_ID}" \
    --score-pkl "${SCORE_PKL}" \
    --output-dir "${OUT_DIR}" \
    --joint-quality-source quality_csv \
    --quality-score-csv "${QUALITY_CSV}" \
    --filter-score-mode joint \
    --joint-drop-quantile "${Q}" \
    --joint-quality-power 1.0 \
    --joint-influence-power 1.0 \
    --weight-mode influence \
    --use-weighted-sampler \
    --num-workers 0 \
    --log-every 20 \
    --max-steps "${MAX_STEPS}" \
    > "${LOG_FILE}" 2>&1
  RC=$?
  set -e

  END_TS="$(date "+%Y-%m-%d %H:%M:%S")"
  if [[ ${RC} -eq 0 ]]; then
    echo "[${END_TS}] [gpu=${GPU}] [done] ${RUN_NAME}"
  else
    echo "[${END_TS}] [gpu=${GPU}] [fail] ${RUN_NAME} rc=${RC}"
  fi
done < "${QUEUE_FILE}"

echo "[$(date "+%Y-%m-%d %H:%M:%S")] [gpu=${GPU}] queue finished."
' _ "${queue_file}" "${gpu}" "${SCRIPT}" "${DATASET_ROOT}" "${DATASET_REPO_ID}" "${QUALITY_CSV}" "${MAX_STEPS}" \
  > "${worker_log}" 2>&1 &

  echo "[launch-worker] gpu=${gpu} pid=$! worker_log=${worker_log}"
done

echo "[done] serial workers submitted."
echo "check:"
for gpu in "${GPU_LIST[@]}"; do
  echo "  tail -f ${LOG_DIR}/serial_worker_gpu${gpu}.log"
done
echo "  pgrep -af train_act_cupid_reweight_joint.py"
