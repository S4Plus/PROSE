set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-/workspace/liushuo/cupid/act/train_act_cupid_reweight_joint.py}"
DATASET_ROOT="${DATASET_ROOT:-/workspace/liushuo/cupid/datasets/lerobot/can_ph}"
DATASET_REPO_ID="${DATASET_REPO_ID:-ls/can_ph}"
SCORE_ROOT="${SCORE_ROOT:-/workspace/liushuo/cupid/1_infulenece_score/can_ph}"
QUALITY_CSV="${QUALITY_CSV:-/workspace/liushuo/cupid/3_model/quality_eval/can_ph_rank_fusion/quality_episodes.csv}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/workspace/liushuo/cupid/3_model/can_ph_model/joint}"
LOG_DIR="${LOG_DIR:-/workspace/liushuo/cupid/3_model/logs/can_ph_model/joint}"
MAX_STEPS="${MAX_STEPS:-100000}"
DROP_BAD_RATIO="${DROP_BAD_RATIO:-1.1}"

mkdir -p "${OUTPUT_ROOT}" "${LOG_DIR}"

if [[ ! -f "${QUALITY_CSV}" ]]; then
    echo "Missing quality csv: ${QUALITY_CSV}" >&2
    exit 1
fi

SCORES=(
    "${SCORE_ROOT}/score.pkl"
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
    local run_name="can_ph_joint_${tag}_q${qtag}"
    local out_dir="${OUTPUT_ROOT}/${run_name}"
    local log_file="${LOG_DIR}/${run_name}_gpu${gpu_id}.log"

    echo "Launching score=${tag} q=${q} gpu=${gpu_id} -> ${log_file}"
    CUDA_VISIBLE_DEVICES="${gpu_id}" \
    HF_HOME="/tmp/hf_home_${run_name}" \
    HF_DATASETS_CACHE="/tmp/hf_datasets_cache_${run_name}" nohup \
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
        > "${log_file}" 2>&1 &
}

idx=0
for score_pkl in "${SCORES[@]}"; do
    for q in "${QUANTILES[@]}"; do
        gpu="${GPUS[$((idx % ${#GPUS[@]}))]}"
        run_joint "${score_pkl}" "${q}" "${gpu}"
        idx=$((idx + 1))
    done
done

echo "Jobs started. Logs: ${LOG_DIR}"
