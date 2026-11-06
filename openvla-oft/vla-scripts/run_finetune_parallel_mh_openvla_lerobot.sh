#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/workspace/liushuo/cupid/openvla-oft"
cd "${ROOT_DIR}"

LOG_ROOT="${LOG_ROOT:-${ROOT_DIR}/logs/parallel_openvla_lerobot}"
mkdir -p "${LOG_ROOT}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

start_job() {
  local dataset="$1"
  local gpu="$2"
  local script_path="$3"
  local output_dir="$4"
  local log_path="${LOG_ROOT}/${dataset}_${TIMESTAMP}.log"
  local norm_stats_path="${output_dir}/norm_stats.fixed.json"

  mkdir -p "${output_dir}"
  echo "[launch] dataset=${dataset} gpu=${gpu} log=${log_path} norm_stats=${norm_stats_path}"
  nohup env \
    CUDA_VISIBLE_DEVICES="${gpu}" \
    TOKENIZERS_PARALLELISM=false \
    OUTPUT_DIR="${output_dir}" \
    RECOMPUTE_NORM_STATS=1 \
    FIXED_NORM_STATS_PATH="${norm_stats_path}" \
    bash "${script_path}" > "${log_path}" 2>&1 &
  echo "[pid] dataset=${dataset} pid=$!"
}

start_job \
  "can_mh" \
  "1" \
  "${ROOT_DIR}/vla-scripts/run_finetune_can_mh_openvla_lerobot.sh" \
  "${ROOT_DIR}/runs/can_mh_openvla_lora"
start_job \
  "square_mh" \
  "0" \
  "${ROOT_DIR}/vla-scripts/run_finetune_square_mh_openvla_lerobot.sh" \
  "${ROOT_DIR}/runs/square_mh_openvla_lora"
start_job \
  "transport_mh" \
  "6" \
  "${ROOT_DIR}/vla-scripts/run_finetune_transport_mh_openvla_lerobot.sh" \
  "${ROOT_DIR}/runs/transport_mh_openvla_lora"

echo "[done] launched 3 parallel OpenVLA LeRobot fine-tuning jobs."
echo "[logs] ${LOG_ROOT}"
