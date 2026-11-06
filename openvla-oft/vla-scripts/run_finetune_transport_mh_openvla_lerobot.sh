#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/workspace/liushuo/cupid/openvla-oft"
cd "${ROOT_DIR}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

DATASET_ROOT="${DATASET_ROOT:-/workspace/liushuo/cupid/datasets/lerobot/transport_mh}"
VLA_PATH="${VLA_PATH:-openvla/openvla-7b}"
OUTPUT_DIR="${OUTPUT_DIR:-/workspace/liushuo/cupid/openvla-oft/runs/transport_mh_openvla_lora}"
RESUME_CHECKPOINT_DIR="${RESUME_CHECKPOINT_DIR:-}"
RESUME_STEP="${RESUME_STEP:-}"
LOG_DIR="${LOG_DIR:-/workspace/liushuo/cupid/openvla-oft/logs/transport_mh_openvla_lora}"
FIXED_NORM_STATS_PATH="${FIXED_NORM_STATS_PATH:-${OUTPUT_DIR}/norm_stats.fixed.json}"
RECOMPUTE_NORM_STATS="${RECOMPUTE_NORM_STATS:-1}"
CACHE_ROOT="${CACHE_ROOT:-/workspace/liushuo/.cache}"

export HF_HOME="${HF_HOME:-${CACHE_ROOT}/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export WANDB_DIR="${WANDB_DIR:-${CACHE_ROOT}/wandb}"

mkdir -p "${LOG_DIR}" "${HF_HOME}" "${HF_HUB_CACHE}" "${WANDB_DIR}"

if [[ "${RECOMPUTE_NORM_STATS}" == "1" ]]; then
  python vla-scripts/recompute_lerobot_norm_stats.py \
    --dataset_root "${DATASET_ROOT}" \
    --dataset_name transport_mh \
    --output "${FIXED_NORM_STATS_PATH}"
  NORM_STATS_ARG=(--norm_stats_path "${FIXED_NORM_STATS_PATH}")
else
  NORM_STATS_ARG=()
fi

TASK_DESCRIPTION="${TASK_DESCRIPTION:-}"
TASK_DESCRIPTION_ARG=()
if [[ -n "${TASK_DESCRIPTION}" ]]; then
  TASK_DESCRIPTION_ARG=(--task_description "${TASK_DESCRIPTION}")
fi
RESUME_ARG=()
if [[ -n "${RESUME_CHECKPOINT_DIR}" ]]; then
  RESUME_ARG=(--resume_checkpoint_dir "${RESUME_CHECKPOINT_DIR}")
fi
if [[ -n "${RESUME_STEP}" ]]; then
  RESUME_ARG+=(--resume_step "${RESUME_STEP}")
fi

python vla-scripts/finetune_openvla_lerobot.py \
  --vla_path "${VLA_PATH}" \
  --dataset_root "${DATASET_ROOT}" \
  --dataset_name transport_mh \
  --output_dir "${OUTPUT_DIR}" \
  "${RESUME_ARG[@]}" \
  "${NORM_STATS_ARG[@]}" \
  "${TASK_DESCRIPTION_ARG[@]}" \
  --batch_size "${BATCH_SIZE:-8}" \
  --grad_accumulation_steps "${GRAD_ACCUMULATION_STEPS:-1}" \
  --max_steps "${MAX_STEPS:-30000}" \
  --save_freq "${SAVE_FREQ:-5000}" \
  --learning_rate "${LEARNING_RATE:-5e-4}" \
  --num_steps_before_decay "${NUM_STEPS_BEFORE_DECAY:-20000}" \
  --action_chunk_size "${ACTION_CHUNK_SIZE:-8}" \
  --num_images_in_input "${NUM_IMAGES_IN_INPUT:-2}" \
  --use_proprio "${USE_PROPRIO:-True}" \
  --use_lora True \
  --lora_rank "${LORA_RANK:-32}" \
  --lora_dropout "${LORA_DROPOUT:-0.0}" \
  --merge_lora_during_training "${MERGE_LORA_DURING_TRAINING:-False}" \
  --num_workers "${NUM_WORKERS:-0}" \
  --seed "${SEED:-7}"
