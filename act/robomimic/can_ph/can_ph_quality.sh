# 计算quality
# python /workspace/liushuo/cupid/act/eval_cupid_quality.py \
#   --dataset.root /workspace/liushuo/cupid/datasets/lerobot/can_ph \
#   --quality-source rank_fusion \
#   --rank-fusion-episode-weight 0.3 \
#   --rank-fusion-trajectory-weight 0.7 \
#   --drop-quantile 0.2 \
#   --output-dir /workspace/liushuo/cupid/3_model/quality_eval/can_ph_rank_fusion




set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-/workspace/liushuo/cupid/act/train_act_cupid_reweight_joint.py}"
DATASET_ROOT="${DATASET_ROOT:-/workspace/liushuo/cupid/datasets/lerobot/can_ph}"
DATASET_REPO_ID="${DATASET_REPO_ID:-ls/can_ph}"
QUALITY_DIR="${QUALITY_DIR:-/workspace/liushuo/cupid/3_model/quality_eval/can_ph_rank_fusion}"
QUALITY_CSV="${QUALITY_CSV:-${QUALITY_DIR}/quality_episodes.csv}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/workspace/liushuo/cupid/3_model/can_ph_model/quality}"
LOG_DIR="${LOG_DIR:-/workspace/liushuo/cupid/3_model/logs/can_ph_model/quality}"
MAX_STEPS="${MAX_STEPS:-100000}"
DROP_BAD_RATIO="${DROP_BAD_RATIO:-1.1}"
RUN_QUALITY_EVAL="${RUN_QUALITY_EVAL:-0}"

mkdir -p "${QUALITY_DIR}" "${OUTPUT_ROOT}" "${LOG_DIR}"
run_quality_only() {
    local q="$1"
    local gpu_id="$2"
    local qtag="${q//./}"
    local run_name="can_ph_quality_only_q${qtag}"
    local out_dir="${OUTPUT_ROOT}/${run_name}"
    local log_file="${LOG_DIR}/${run_name}_gpu${gpu_id}.log"

    echo "Launching q=${q} gpu=${gpu_id} -> ${log_file}"
    CUDA_VISIBLE_DEVICES="${gpu_id}" \
    HF_HOME="/tmp/hf_home_${run_name}" \
    HF_DATASETS_CACHE="/tmp/hf_datasets_cache_${run_name}" nohup \
    "${PYTHON_BIN}" "${TRAIN_SCRIPT}" \
        --dataset.root "${DATASET_ROOT}" \
        --dataset.repo_id "${DATASET_REPO_ID}" \
        --output-dir "${out_dir}" \
        --job_name "${run_name}" \
        --joint-quality-source quality_csv \
        --quality-score-csv "${QUALITY_CSV}" \
        --filter-score-mode joint \
        --joint-drop-quantile "${q}" \
        --joint-quality-power 1.0 \
        --joint-influence-power 0.0 \
        --drop-bad-ratio "${DROP_BAD_RATIO}" \
        --weight-mode none \
        --max-steps "${MAX_STEPS}" \
        > "${log_file}" 2>&1 &
}

if [[ "${RUN_QUALITY_EVAL}" == "1" ]]; then
    run_quality_eval
fi

if [[ ! -f "${QUALITY_CSV}" ]]; then
    echo "Missing quality csv: ${QUALITY_CSV}" >&2
    echo "Set RUN_QUALITY_EVAL=1 or provide QUALITY_CSV path." >&2
    exit 1
fi

QUANTILES=(0.1 0.2 0.3 0.4 0.5)
GPUS=(7 6 5 4)

for i in "${!QUANTILES[@]}"; do
    q="${QUANTILES[$i]}"
    gpu="${GPUS[$((i % ${#GPUS[@]}))]}"
    run_quality_only "${q}" "${gpu}"
done

echo "Jobs started. Logs: ${LOG_DIR}"
