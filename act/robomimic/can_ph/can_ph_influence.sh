# # GPU 1 (last 4 layers):
# CUDA_VISIBLE_DEVICES=1 nohup python /workspace/liushuo/cupid/act/cupid_act_influence_last_layers.py \
#   --pretrained_dir /workspace/liushuo/cupid/3_model/can_ph_model/random/random42_1.0/checkpoints/100000/pretrained_model\
#   --train_dataset_path /workspace/liushuo/cupid/datasets/lerobot/can_ph \
#   --eval_rollout_dir /workspace/liushuo/cupid/mimicgen/eval/tmp/can_ph/random/1.0/episodes \
#   --output_path /workspace/liushuo/cupid/1_infulenece_score/can_ph/score_last4 \
#   --param-scope last_n_layers \
#   --last-n-layers 4 \
#   > /workspace/liushuo/cupid/3_model/logs/cupid_act_influence_last4_ph.log 2>&1 &

# # GPU 2 (last 8 layers):
# CUDA_VISIBLE_DEVICES=2 nohup python /workspace/liushuo/cupid/act/cupid_act_influence_last_layers.py \
#   --pretrained_dir /workspace/liushuo/cupid/3_model/can_ph_model/random/random42_1.0/checkpoints/100000/pretrained_model \
#   --train_dataset_path /workspace/liushuo/cupid/datasets/lerobot/can_ph \
#   --eval_rollout_dir /workspace/liushuo/cupid/mimicgen/eval/tmp/can_ph/random/1.0/episodes \
#   --output_path /workspace/liushuo/cupid/1_infulenece_score/can_ph/score_last8 \
#   --param-scope last_n_layers \
#   --last-n-layers 8 \
#   > /workspace/liushuo/cupid/3_model/logs/cupid_act_influence_last8_ph.log 2>&1 &

# # GPU 3 (last 12 layers):
# CUDA_VISIBLE_DEVICES=3 nohup python /workspace/liushuo/cupid/act/cupid_act_influence_last_layers.py \
#   --pretrained_dir /workspace/liushuo/cupid/3_model/can_ph_model/random/random42_1.0/checkpoints/100000/pretrained_model \
#   --train_dataset_path /workspace/liushuo/cupid/datasets/lerobot/can_ph \
#   --eval_rollout_dir /workspace/liushuo/cupid/mimicgen/eval/tmp/can_ph/random/1.0/episodes \
#   --output_path /workspace/liushuo/cupid/1_infulenece_score/can_ph/score_last12 \
#   --param-scope last_n_layers \
#   --last-n-layers 12 \
#   > /workspace/liushuo/cupid/3_model/logs/cupid_act_influence_last12_ph.log 2>&1 &

# CUDA_VISIBLE_DEVICES=4 \
# nohup python /workspace/liushuo/cupid/act/cupid_act_influence.py \
#     --pretrained_dir /workspace/liushuo/cupid/3_model/can_ph_model/random/random42_1.0/checkpoints/100000/pretrained_model \
#     --train_dataset_path /workspace/liushuo/cupid/datasets/lerobot/can_ph\
#     --eval_rollout_dir /workspace/liushuo/cupid/mimicgen/eval/tmp/can_ph/random/1.0/episodes\
#     --output_path /workspace/liushuo/cupid/1_infulenece_score/can_ph/score \
#     > /workspace/liushuo/cupid/3_model/logs/cupid_act_influence_can_ph.log 2>&1 &

set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-/workspace/liushuo/cupid/act/train_act_cupid_reweight.py}"
DATASET_ROOT="${DATASET_ROOT:-/workspace/liushuo/cupid/datasets/lerobot/can_ph}"
DATASET_REPO_ID="${DATASET_REPO_ID:-ls/can_ph}"
SCORE_ROOT="${SCORE_ROOT:-/workspace/liushuo/cupid/1_infulenece_score/can_ph}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/workspace/liushuo/cupid/3_model/can_ph_model/influence}"
LOG_DIR="${LOG_DIR:-/workspace/liushuo/cupid/3_model/logs/can_ph_model/influence}"
MAX_STEPS="${MAX_STEPS:-100000}"
DROP_BAD_RATIO="${DROP_BAD_RATIO:-1.1}"

mkdir -p "${OUTPUT_ROOT}" "${LOG_DIR}"

SCORES=(
    "${SCORE_ROOT}/score.pkl"
    "${SCORE_ROOT}/score_last4.pkl"
    "${SCORE_ROOT}/score_last8.pkl"
    "${SCORE_ROOT}/score_last12.pkl"
)
QUANTILES=(0.1 0.2 0.3 0.4 0.5)
GPUS=(7 6 5 4)

run_influence() {
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
    local run_name="can_ph_influence_${tag}_q${qtag}"
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
        --policy.type act \
        --policy.repo_id "${DATASET_REPO_ID}" \
        --drop-bad-ratio "${DROP_BAD_RATIO}" \
        --drop-weight-quantile "${q}" \
        --weight-mode influence \
        --use-weighted-sampler \
        --max-steps "${MAX_STEPS}" \
        > "${log_file}" 2>&1 &
}

idx=0
for score_pkl in "${SCORES[@]}"; do
    for q in "${QUANTILES[@]}"; do
        gpu="${GPUS[$((idx % ${#GPUS[@]}))]}"
        run_influence "${score_pkl}" "${q}" "${gpu}"
        idx=$((idx + 1))
    done
done

echo "Jobs started. Logs: ${LOG_DIR}"
