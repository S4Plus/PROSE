# #!/usr/bin/env bash
# # Quick wrapper to launch random ACT retraining jobs.
# set -euo pipefail

# PYTHON_BIN="${PYTHON_BIN:-python}"
# DATASET_ROOT="${DATASET_ROOT:-/workspace/liushuo/cupid/datasets/square_d1_joint}"
# LEROBOT_REPO="${LEROBOT_REPO:-/workspace/liushuo/cupid/lerobot_2.1}"
# OUTPUT_ROOT="${OUTPUT_ROOT:-/workspace/liushuo/cupid/mymimicgen/outputs/random_40act_joint}"
# LOG_DIR="${LOG_DIR:-/workspace/liushuo/cupid/mymimicgen/outputs/logs_new}"

# mkdir -p "${OUTPUT_ROOT}" "${LOG_DIR}"

# run_random() {
#     local ratio="$1"
#     local seed="${2:-40}"
#     local tag="${ratio//./_}"
#     local out_dir="${OUTPUT_ROOT}/retrain2_random_${ratio}"
#     local log_file="${LOG_DIR}/train_act_random_40act_abs_${ratio}.log"

#     echo "Launching ratio=${ratio} seed=${seed} -> ${log_file}"
#     nohup "${PYTHON_BIN}" scripts/lerobot/random_train.py \
#         --sample-ratio "${ratio}" \
#         --sample-seed "${seed}" \
#         --dataset-repo-id  ls/square_d1_joint \
#         --dataset-root "${DATASET_ROOT}" \
#         --lerobot-repo "${LEROBOT_REPO}" \
#         --policy.type act \
#         --output_dir "${out_dir}" \
#         --job_name " mimicgen/square_d1_joint_random${tag}" \
#         --policy.device cuda \
#         --policy.repo_id  mimicgen/square_d1_joint \
#         > "${log_file}" 2>&1 &
# }

# # Example launches (comment out any you don't need).
# run_random 1.0 40
# run_random 0.8 40
# run_random 0.5 40
# run_random 0.2 40

# echo "Jobs started. Logs: ${LOG_DIR}"




#!/usr/bin/env bash
# Quick wrapper to launch random ACT retraining jobs.
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
DATASET_ROOT="${DATASET_ROOT:-/workspace/liushuo/cupid/datasets/square_d1}"
LEROBOT_REPO="${LEROBOT_REPO:-/workspace/liushuo/cupid/lerobot_2.1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/workspace/liushuo/cupid/mymimicgen/outputs/random_40act}"
LOG_DIR="${LOG_DIR:-/workspace/liushuo/cupid/mymimicgen/outputs/logs}"

mkdir -p "${OUTPUT_ROOT}" "${LOG_DIR}"

run_random() {
    local ratio="$1"
    local seed="${2:-40}"
    local tag="${ratio//./_}"
    local out_dir="${OUTPUT_ROOT}/retrain2_random_${ratio}"
    local log_file="${LOG_DIR}/train_act_random_40act_${ratio}.log"

    echo "Launching ratio=${ratio} seed=${seed} -> ${log_file}"
    nohup "${PYTHON_BIN}" scripts/lerobot/random_train.py \
        --sample-ratio "${ratio}" \
        --sample-seed "${seed}" \
        --dataset-repo-id  ls/square_d1 \
        --dataset-root "${DATASET_ROOT}" \
        --lerobot-repo "${LEROBOT_REPO}" \
        --policy.type act \
        --output_dir "${out_dir}" \
        --job_name " mimicgen/square_d1_joint_random${tag}" \
        --policy.device cuda \
        --policy.repo_id  ls/square_d1 \
        > "${log_file}" 2>&1 &
}

# Example launches (comment out any you don't need).
run_random 1.0 40
run_random 0.8 40
run_random 0.5 40
run_random 0.2 40

echo "Jobs started. Logs: ${LOG_DIR}"
