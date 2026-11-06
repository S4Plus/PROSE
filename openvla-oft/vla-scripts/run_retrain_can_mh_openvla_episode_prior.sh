#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/workspace/liushuo/cupid/openvla-oft"
cd "${ROOT_DIR}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

SCORE_PKL="${SCORE_PKL:-/workspace/liushuo/cupid/openvla-oft/tmp/can_mh_openvla_window_grass/influence_grass.pkl}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/workspace/liushuo/cupid/openvla-oft/runs/can_mh_openvla_lora--10000_chkpt}"
OUTPUT_DIR="${OUTPUT_DIR:-/workspace/liushuo/cupid/openvla-oft/tmp/can_mh_openvla_window_episode_prior_keep90}"
RETRAIN_RUN_ROOT="${RETRAIN_RUN_ROOT:-/workspace/liushuo/cupid/openvla-oft/runs}"
EXP_NAME="${EXP_NAME:-can_mh_openvla_window_episode_prior_keep90}"
LOG_DIR="${LOG_DIR:-/workspace/liushuo/cupid/openvla-oft/logs/can_mh_openvla_window_episode_prior_keep90}"

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

python vla-scripts/retrain_window_joint_episode_prior_openvla.py \
  --score-pkl "${SCORE_PKL}" \
  --checkpoint-dir "${CHECKPOINT_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --retrain-run-root "${RETRAIN_RUN_ROOT}" \
  --exp-name "${EXP_NAME}" \
  --keep-ratio "${KEEP_RATIO:-0.9}" \
  --joint-rule "${JOINT_RULE:-sum}" \
  --joint-quality-power "${JOINT_QUALITY_POWER:-1.0}" \
  --joint-influence-power "${JOINT_INFLUENCE_POWER:-1.0}" \
  --window-score-weight "${WINDOW_SCORE_WEIGHT:-0.8}" \
  --episode-prior-weight "${EPISODE_PRIOR_WEIGHT:-0.2}" \
  --adaptive-prior-mode "${ADAPTIVE_PRIOR_MODE:-none}" \
  --adaptive-prior-strength "${ADAPTIVE_PRIOR_STRENGTH:-0.5}" \
  --interaction-weight "${INTERACTION_WEIGHT:-0.0}" \
  --fusion-normalizer "${FUSION_NORMALIZER:-minmax}" \
  --episode-quality-aggregation "${EPISODE_QUALITY_AGGREGATION:-mean}" \
  --episode-budget-rule "${EPISODE_BUDGET_RULE:-global}" \
  --quality-source "${QUALITY_SOURCE:-rank_fusion}" \
  --quality-normalizer "${QUALITY_NORMALIZER:-quantile}" \
  --quality-quantile-low "${QUALITY_QUANTILE_LOW:-0.05}" \
  --quality-quantile-high "${QUALITY_QUANTILE_HIGH:-0.95}" \
  --num-train-steps "${NUM_TRAIN_STEPS:-10000}" \
  --overwrite
