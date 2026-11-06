#!/usr/bin/env bash
set -euo pipefail

# Serial scheduler: one training process per GPU, remaining jobs queued.
# Score files: score_last4/score_last8/score_last12
# Quantiles: 0.1/0.2/0.3/0.4/0.5
# GPUs: 7/6/5/4   只有influence score

DATASET_ROOT="/workspace/liushuo/cupid/datasets/lerobot/can_mh"
DATASET_REPO_ID="ls/can_mh"
TRAIN_SCRIPT="/workspace/liushuo/cupid/act/train_act_cupid_reweight.py"
LOG_DIR="/workspace/liushuo/cupid/3_model/logs"
OUT_ROOT="/workspace/liushuo/cupid/3_model"
QUEUE_DIR="/workspace/liushuo/cupid/3_model/job_queues"

SCORES=(
  "/workspace/liushuo/cupid/1_infulenece_score/can_mh/score_last4.pkl"
  "/workspace/liushuo/cupid/1_infulenece_score/can_mh/score_last8.pkl"
  "/workspace/liushuo/cupid/1_infulenece_score/can_mh/score_last12.pkl"
)
QUANTILES=(0.1 0.2 0.3 0.4 0.5)
GPUS=(7 6 5 4)

mkdir -p "${LOG_DIR}" "${QUEUE_DIR}"

# Clean old queue files.
for gpu in "${GPUS[@]}"; do
  : > "${QUEUE_DIR}/queue_gpu${gpu}.tsv"
done

# Build queues with round-robin assignment across GPUs.
idx=0
for score_pkl in "${SCORES[@]}"; do
  if [[ ! -f "${score_pkl}" ]]; then
    echo "[skip] score not found: ${score_pkl}"
    continue
  fi
  score_tag="$(basename "${score_pkl}" .pkl)"
  for q in "${QUANTILES[@]}"; do
    gpu="${GPUS[$((idx % ${#GPUS[@]}))]}"
    qtag="$(echo "${q}" | tr -d '.')"
    run_name="can_mh_${score_tag}_q${qtag}"
    out_dir="${OUT_ROOT}/${run_name}"
    log_file="${LOG_DIR}/${run_name}_gpu${gpu}.log"
    printf "%s\t%s\t%s\t%s\t%s\n" \
      "${score_pkl}" "${q}" "${run_name}" "${out_dir}" "${log_file}" \
      >> "${QUEUE_DIR}/queue_gpu${gpu}.tsv"
    idx=$((idx + 1))
  done
done

echo "[info] queue files generated under ${QUEUE_DIR}"
for gpu in "${GPUS[@]}"; do
  echo "----- GPU ${gpu} queue -----"
  cat "${QUEUE_DIR}/queue_gpu${gpu}.tsv"
done

# Start one worker per GPU; each worker runs queued jobs sequentially.
for gpu in "${GPUS[@]}"; do
  queue_file="${QUEUE_DIR}/queue_gpu${gpu}.tsv"
  worker_log="${LOG_DIR}/serial_worker_gpu${gpu}.log"

  nohup bash -c '
set -euo pipefail
QUEUE_FILE="$1"
GPU="$2"
TRAIN_SCRIPT="$3"
DATASET_ROOT="$4"
DATASET_REPO_ID="$5"

while IFS=$'\''\t'\'' read -r SCORE_PKL Q RUN_NAME OUT_DIR LOG_FILE; do
  [[ -z "${SCORE_PKL}" ]] && continue
  START_TS="$(date "+%Y-%m-%d %H:%M:%S")"
  echo "[${START_TS}] [gpu=${GPU}] [start] ${RUN_NAME} q=${Q} score=$(basename "${SCORE_PKL}")"

  HF_HOME="/tmp/hf_home_${RUN_NAME}"
  HF_DATASETS_CACHE="/tmp/hf_datasets_cache_${RUN_NAME}"
  mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}"

  set +e
  CUDA_VISIBLE_DEVICES="${GPU}" HF_HOME="${HF_HOME}" HF_DATASETS_CACHE="${HF_DATASETS_CACHE}" PYTHONUNBUFFERED=1 \
  python "${TRAIN_SCRIPT}" \
    --dataset.root "${DATASET_ROOT}" \
    --dataset.repo_id "${DATASET_REPO_ID}" \
    --output-dir "${OUT_DIR}" \
    --job_name "${RUN_NAME}" \
    --policy.type=act \
    --policy.repo_id "${DATASET_REPO_ID}" \
    --score-pkl "${SCORE_PKL}" \
    --drop-bad-ratio 1.1 \
    --drop-weight-quantile "${Q}" \
    --weight-mode influence \
    --use-weighted-sampler \
    --num-workers 0 \
    --log-every 20 \
    --max-steps 100000 \
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
' _ "${queue_file}" "${gpu}" "${TRAIN_SCRIPT}" "${DATASET_ROOT}" "${DATASET_REPO_ID}" \
  > "${worker_log}" 2>&1 &

  echo "[launch-worker] gpu=${gpu} pid=$! worker_log=${worker_log}"
done

echo "[done] serial workers submitted."
echo "Check worker logs:"
echo "  tail -f ${LOG_DIR}/serial_worker_gpu7.log"
echo "  tail -f ${LOG_DIR}/serial_worker_gpu6.log"
echo "  tail -f ${LOG_DIR}/serial_worker_gpu5.log"
echo "  tail -f ${LOG_DIR}/serial_worker_gpu4.log"
