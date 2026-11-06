set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-/workspace/liushuo/cupid/act/train_act_cupid_reweight_joint.py}"
DATASET_ROOT="${DATASET_ROOT:-/workspace/liushuo/cupid/datasets/lerobot/can_mh}"
DATASET_REPO_ID="${DATASET_REPO_ID:-ls/can_mh}"
SCORE_ROOT="${SCORE_ROOT:-/workspace/liushuo/cupid/1_infulenece_score/can_mh_strict}"
QUALITY_CSV="${QUALITY_CSV:-/workspace/liushuo/cupid/3_model/quality_eval/can_mh_rank_fusion/quality_episodes.csv}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/workspace/liushuo/cupid/3_model/can_mh_model/joint_strict}"
LOG_DIR="${LOG_DIR:-/workspace/liushuo/cupid/3_model/logs/can_mh_model/joint_strict}"
MAX_STEPS="${MAX_STEPS:-100000}"
DROP_BAD_RATIO="${DROP_BAD_RATIO:-1.1}"

mkdir -p "${OUTPUT_ROOT}" "${LOG_DIR}"

if [[ ! -f "${QUALITY_CSV}" ]]; then
    echo "Missing quality csv: ${QUALITY_CSV}" >&2
    exit 1
fi

SCORES=(
    # "${SCORE_ROOT}/score.pkl"
    "${SCORE_ROOT}/score_last4.pkl"
    "${SCORE_ROOT}/score_last8.pkl"
    "${SCORE_ROOT}/score_last12.pkl"
)
QUANTILES=(0.1 0.2 0.3 0.4 0.5)
GPUS=(7 6 5 4)

run_joint() {
    local score_pkl="$1"
    local q="$2"
    local gpu_id="$3"

    if [[ ! -f "${score_pkl}" ]]; then
        echo "Skip missing score file: ${score_pkl}"
        return
    fi

    local tag
    tag="$(basename "${score_pkl}" .pkl)"
    local qtag="${q//./}"
    local run_name="can_mh_joint_${tag}_q${qtag}"
    local out_dir="${OUTPUT_ROOT}/${run_name}"
    local log_file="${LOG_DIR}/${run_name}_gpu${gpu_id}.log"

    echo "Running score=${tag} q=${q} gpu=${gpu_id} -> ${log_file}"
    CUDA_VISIBLE_DEVICES="${gpu_id}" \
    HF_HOME="/tmp/hf_home_${run_name}" \
    HF_DATASETS_CACHE="/tmp/hf_datasets_cache_${run_name}" \
    PYTHONUNBUFFERED=1 \
    "${PYTHON_BIN}" "${TRAIN_SCRIPT}" \
        --dataset.root "${DATASET_ROOT}" \
        --dataset.repo_id "${DATASET_REPO_ID}" \
        --score-pkl "${score_pkl}" \
        --output-dir "${out_dir}" \
        --job_name "${run_name}" \
        --joint-quality-source quality_csv \
        --quality-score-csv "${QUALITY_CSV}" \
        --filter-score-mode joint \
        --joint-drop-quantile "${q}" \
        --joint-quality-power 1.0 \
        --joint-influence-power 1.0 \
        --drop-bad-ratio "${DROP_BAD_RATIO}" \
        --weight-mode influence \
        --use-weighted-sampler \
        --max-steps "${MAX_STEPS}" \
        > "${log_file}" 2>&1
}

# One queue file per GPU (round-robin assignment).
QUEUE_DIR="${LOG_DIR}/job_queues"
mkdir -p "${QUEUE_DIR}"
for gpu in "${GPUS[@]}"; do
    : > "${QUEUE_DIR}/queue_gpu${gpu}.tsv"
done

idx=0
for score_pkl in "${SCORES[@]}"; do
    for q in "${QUANTILES[@]}"; do
        gpu="${GPUS[$((idx % ${#GPUS[@]}))]}"
        printf "%s\t%s\n" "${score_pkl}" "${q}" >> "${QUEUE_DIR}/queue_gpu${gpu}.tsv"
        idx=$((idx + 1))
    done
done

echo "Queue files generated under ${QUEUE_DIR}"
for gpu in "${GPUS[@]}"; do
    echo "----- GPU ${gpu} queue -----"
    cat "${QUEUE_DIR}/queue_gpu${gpu}.tsv"
done

# Each GPU starts only one worker; worker executes remaining jobs serially.
for gpu in "${GPUS[@]}"; do
    queue_file="${QUEUE_DIR}/queue_gpu${gpu}.tsv"
    worker_log="${LOG_DIR}/serial_worker_gpu${gpu}.log"
    (
        set -euo pipefail
        while IFS=$'\t' read -r score_pkl q; do
            [[ -z "${score_pkl}" ]] && continue
            start_ts="$(date "+%Y-%m-%d %H:%M:%S")"
            echo "[${start_ts}] [gpu=${gpu}] [start] score=$(basename "${score_pkl}") q=${q}"
            set +e
            run_joint "${score_pkl}" "${q}" "${gpu}"
            rc=$?
            set -e
            end_ts="$(date "+%Y-%m-%d %H:%M:%S")"
            if [[ ${rc} -eq 0 ]]; then
                echo "[${end_ts}] [gpu=${gpu}] [done] score=$(basename "${score_pkl}") q=${q}"
            else
                echo "[${end_ts}] [gpu=${gpu}] [fail] score=$(basename "${score_pkl}") q=${q} rc=${rc}"
            fi
        done < "${queue_file}"
        echo "[$(date "+%Y-%m-%d %H:%M:%S")] [gpu=${gpu}] queue finished."
    ) > "${worker_log}" 2>&1 &
    echo "[launch-worker] gpu=${gpu} pid=$! worker_log=${worker_log}"
done

wait
echo "All serial workers finished. Logs: ${LOG_DIR}"
