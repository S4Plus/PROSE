# cd ~/cupid
# bash act/robomimic/can_ph/can_ph_random.sh
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
DATASET_ROOT="${DATASET_ROOT:-/workspace/liushuo/cupid/datasets/lerobot/can_ph}"
LEROBOT_REPO="${LEROBOT_REPO:-/workspace/liushuo/cupid/lerobot_2.1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/workspace/liushuo/cupid/3_model/can_ph_model/random}"
LOG_DIR="${LOG_DIR:-/workspace/liushuo/cupid/3_model/logs/can_ph_model/random}"

mkdir -p "${OUTPUT_ROOT}" "${LOG_DIR}"

run_random() {
    local ratio="$1"
    local seed="${2:-40}"
    local gpu_id="$3"
    local tag="${ratio//./_}"
    local out_dir="${OUTPUT_ROOT}/random42_${ratio}"
    local log_file="${LOG_DIR}/train2_can_ph_act_random_42_${ratio}_gpu${gpu_id}.log"

    echo "Launching ratio=${ratio} seed=${seed} gpu=${gpu_id} -> ${log_file}"
    CUDA_VISIBLE_DEVICES="${gpu_id}" nohup "${PYTHON_BIN}" /workspace/liushuo/cupid/scripts/lerobot/random_train.py \
        --sample-ratio "${ratio}" \
        --sample-seed "${seed}" \
        --dataset-repo-id ls/can_ph \
        --dataset-root "${DATASET_ROOT}" \
        --lerobot-repo "${LEROBOT_REPO}" \
        --policy.type act \
        --output_dir "${out_dir}" \
        --job_name "can_ph_random${tag}" \
        --policy.device cuda \
        --policy.repo_id ls/can_ph \
        > "${log_file}" 2>&1 &
}

# Example launches (comment out any you don't need).
RATIOS=(1.0 0.9 0.8 0.7 0.6 0.5)
GPUS=(7 6 5 4)
SEED=42

for i in "${!RATIOS[@]}"; do
    ratio="${RATIOS[$i]}"
    gpu="${GPUS[$((i % ${#GPUS[@]}))]}"
    run_random "${ratio}" "${SEED}" "${gpu}"
done

echo "Jobs started. Logs: ${LOG_DIR}"
