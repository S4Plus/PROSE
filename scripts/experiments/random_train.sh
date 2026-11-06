# #!/usr/bin/env bash
# # Quick wrapper to launch random ACT retraining jobs.
# set -euo pipefail

# PYTHON_BIN="${PYTHON_BIN:-python}"
# DATASET_ROOT="${DATASET_ROOT:-/ext_workspace/liushuo/Static_Pick_Bottle_400}"
# LEROBOT_REPO="${LEROBOT_REPO:-/workspace/liushuo/cupid/lerobot_2.1}"
# OUTPUT_ROOT="${OUTPUT_ROOT:-/workspace/liushuo/cupid/outputs_new/random_40}"
# LOG_DIR="${LOG_DIR:-/workspace/liushuo/cupid/logs_new}"

# mkdir -p "${OUTPUT_ROOT}" "${LOG_DIR}"

# run_random() {
#     local ratio="$1"
#     local seed="${2:-40}"
#     local tag="${ratio//./_}"
#     local out_dir="${OUTPUT_ROOT}/retrain_random_${ratio}"
#     local log_file="${LOG_DIR}/train2_400_act_random_40_${ratio}.log"

#     echo "Launching ratio=${ratio} seed=${seed} -> ${log_file}"
#     nohup "${PYTHON_BIN}" scripts/lerobot/random_train.py \
#         --sample-ratio "${ratio}" \
#         --sample-seed "${seed}" \
#         --dataset-repo-id ls/Static_Pick_Bottle_400 \
#         --dataset-root "${DATASET_ROOT}" \
#         --lerobot-repo "${LEROBOT_REPO}" \
#         --policy.type act \
#         --output_dir "${out_dir}" \
#         --job_name "Belt_Pick_Bottle_abs_403_random${tag}" \
#         --policy.device cuda \
#         --policy.repo_id ls/Static_Pick_Bottle_400 \
#         > "${log_file}" 2>&1 &
# }

# # Example launches (comment out any you don't need).
# run_random 1.0 40
# run_random 0.8 40
# run_random 0.6 40
# run_random 0.4 40
# run_random 0.2 40

# echo "Jobs started. Logs: ${LOG_DIR}"




#!/usr/bin/env bash
# Quick wrapper to launch random ACT retraining jobs.
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
DATASET_ROOT="${DATASET_ROOT:-/workspace/liushuo/cupid/datasets/hammer_cleanup_d0_core}"
LEROBOT_REPO="${LEROBOT_REPO:-/workspace/liushuo/cupid/lerobot_2.1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/workspace/liushuo/cupid/act/mimicgen/hammer_cleanup_d0_core/retrain_cupid}"
LOG_DIR="${LOG_DIR:-/workspace/liushuo/cupid/logs_new}"

mkdir -p "${OUTPUT_ROOT}" "${LOG_DIR}"

run_random() {
    local ratio="$1"
    local seed="${2:-40}"
    local tag="${ratio//./_}"
    local out_dir="${OUTPUT_ROOT}/retrain_random42_${ratio}"
    local log_file="${LOG_DIR}/train2_hammer_cleanup_d0_core_act_random_40_${ratio}.log"

    echo "Launching ratio=${ratio} seed=${seed} -> ${log_file}"
    nohup "${PYTHON_BIN}" scripts/lerobot/random_train.py \
        --sample-ratio "${ratio}" \
        --sample-seed "${seed}" \
        --dataset-repo-id ls/hammer_cleanup_d0_core \
        --dataset-root "${DATASET_ROOT}" \
        --lerobot-repo "${LEROBOT_REPO}" \
        --policy.type act \
        --output_dir "${out_dir}" \
        --job_name "hammer_cleanup_d0_core_random${tag}" \
        --policy.device cuda \
        --policy.repo_id ls/hammer_cleanup_d0_core \
        > "${log_file}" 2>&1 &
}

# Example launches (comment out any you don't need).
run_random 1.0 42
run_random 0.8 42
run_random 0.6 42
run_random 0.4 42
run_random 0.2 42

echo "Jobs started. Logs: ${LOG_DIR}"
