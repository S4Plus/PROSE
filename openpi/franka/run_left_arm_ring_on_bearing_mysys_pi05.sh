#!/usr/bin/env bash
# 复现 left_arm_ring_on_bearing 的 MYSYS joint episode-window 筛选和 PI05 训练。
# 流程：
# 1. 用 29999 checkpoint 和 rollout pkl 计算 window-level GraSS/MYSYS influence 分数。
# 2. 基于 influence + 轨迹质量做 gain+coverage joint episode-window 筛选。
# 3. 用筛选出的 sample indices 与对应 norm_stats 重新训练 PI05 LoRA。

set -euo pipefail

ROOT="${ROOT:-/workspace/liushuo/cupid/openpi}"
CONFIG_NAME="${CONFIG_NAME:-pi05_left_arm_ring_on_bearing_lora}"
DATASET_DIR="${DATASET_DIR:-/workspace/liushuo/data/dataset/left_arm_ring_on_bearing}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${ROOT}/checkpoints/pi05_left_arm_ring_on_bearing_lora/left_arm_ring_on_bearing_pi05_30k_delta_bs16_chunk50_gpu6/29999}"
ROLLOUT_DIR="${ROLLOUT_DIR:-/workspace/liushuo/ring_on_bearing_pi05_rollout}"

WORK_DIR="${WORK_DIR:-${ROOT}/tmp/left_arm_ring_on_bearing_mysys_from_29999}"
SCORE_PKL="${SCORE_PKL:-${WORK_DIR}/influence_grass_window.pkl}"
STAGE_CACHE_PREFIX="${STAGE_CACHE_PREFIX:-${WORK_DIR}/influence_grass_window.grass_cache}"
SELECTION_DIR="${SELECTION_DIR:-${WORK_DIR}/selection_gaincov}"
CHECKPOINT_BASE_DIR="${CHECKPOINT_BASE_DIR:-${ROOT}/checkpoints}"
EXP_NAME="${EXP_NAME:-left_arm_ring_on_bearing_pi05_30k_delta_bs16_chunk50_mysys_from_29999}"

LOG_DIR="${LOG_DIR:-${ROOT}/logs/left_arm_ring_on_bearing_mysys_pi05}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/full_pipeline_$(date '+%Y%m%d_%H%M%S').log}"
UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache-openpi}"
WANDB_MODE="${WANDB_MODE:-offline}"

# stage 可选：all / score / select / train
STAGE="${STAGE:-all}"
DRY_RUN="${DRY_RUN:-0}"

# GPU 策略：score 单卡；train 两卡 FSDP，避免 80G 单卡压力过大。
GPU_SCORE="${GPU_SCORE:-1}"
GPU_TRAIN="${GPU_TRAIN:-1,4}"
FSDP_DEVICES="${FSDP_DEVICES:-2}"
XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}"

# influence/GraSS 参数。
MAX_TRAIN_BATCHES="${MAX_TRAIN_BATCHES:-}"
MAX_EVAL_EPISODES="${MAX_EVAL_EPISODES:-}"
MAX_EVAL_STEPS_PER_EPISODE="${MAX_EVAL_STEPS_PER_EPISODE:-}"
STAGE1_MICROBATCH_SIZE="${STAGE1_MICROBATCH_SIZE:-1}"
STAGE2_BATCH_SIZE="${STAGE2_BATCH_SIZE:-16}"
SPARSIFICATION="${SPARSIFICATION:-random_mask-32*32}"
PROJECTION="${PROJECTION:-sjlt-1024}"
HESSIAN="${HESSIAN:-eFIM}"
DAMPING="${DAMPING:-1e-4}"
WINDOW_SIZE="${WINDOW_SIZE:-50}"
WINDOW_STRIDE="${WINDOW_STRIDE:-}"
WINDOW_AGGREGATION="${WINDOW_AGGREGATION:-sum}"
WINDOW_QUALITY_SCORE="${WINDOW_QUALITY_SCORE:-zscore_mean}"
FORCE_SCORE="${FORCE_SCORE:-0}"

# MYSYS joint episode-window 筛选参数。
JOINT_RULE="${JOINT_RULE:-sum}"
JOINT_QUALITY_POWER="${JOINT_QUALITY_POWER:-1}"
JOINT_INFLUENCE_POWER="${JOINT_INFLUENCE_POWER:-1}"
WINDOW_SCORE_WEIGHT="${WINDOW_SCORE_WEIGHT:-0.8}"
EPISODE_PRIOR_WEIGHT="${EPISODE_PRIOR_WEIGHT:-0.2}"
ADAPTIVE_PRIOR_MODE="${ADAPTIVE_PRIOR_MODE:-none}"
ADAPTIVE_PRIOR_STRENGTH="${ADAPTIVE_PRIOR_STRENGTH:-0.5}"
INTERACTION_WEIGHT="${INTERACTION_WEIGHT:-0.0}"
FUSION_NORMALIZER="${FUSION_NORMALIZER:-minmax}"
EPISODE_QUALITY_AGGREGATION="${EPISODE_QUALITY_AGGREGATION:-mean}"
MIN_WINDOWS_PER_EPISODE="${MIN_WINDOWS_PER_EPISODE:-0}"
EPISODE_BUDGET_RULE="${EPISODE_BUDGET_RULE:-global}"
QUALITY_SOURCE="${QUALITY_SOURCE:-rank_fusion}"
QUALITY_NORMALIZER="${QUALITY_NORMALIZER:-quantile}"
QUALITY_QUANTILE_LOW="${QUALITY_QUANTILE_LOW:-0.05}"
QUALITY_QUANTILE_HIGH="${QUALITY_QUANTILE_HIGH:-0.95}"
MIN_COVERAGE_RATIO="${MIN_COVERAGE_RATIO:-0.9}"
RELATIVE_GAIN_THRESHOLD="${RELATIVE_GAIN_THRESHOLD:-0.02}"
MARGINAL_GAIN_THRESHOLD="${MARGINAL_GAIN_THRESHOLD:-}"
MAX_FRAMES_FOR_NORM_STATS="${MAX_FRAMES_FOR_NORM_STATS:-}"

# 训练参数。
NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-30000}"
OVERWRITE_TRAIN="${OVERWRITE_TRAIN:-1}"

mkdir -p "${WORK_DIR}" "${LOG_DIR}"
cd "${ROOT}"
exec > >(tee -a "${LOG_FILE}") 2>&1

run_cmd() {
  echo
  echo "[cmd] $*"
  if [[ "${DRY_RUN}" == "1" ]]; then
    return 0
  fi
  "$@"
}

run_score() {
  if [[ -f "${SCORE_PKL}" && "${FORCE_SCORE}" != "1" ]]; then
    echo "[skip] score 已存在：${SCORE_PKL}"
    return 0
  fi

  local -a cmd=(
    uv run python scripts/influence_pi05_lora_grass_window.py
    --config-name "${CONFIG_NAME}"
    --checkpoint-dir "${CHECKPOINT_DIR}"
    --eval-rollouts-dir "${ROLLOUT_DIR}"
    --output-path "${SCORE_PKL}"
    --stage-cache-prefix "${STAGE_CACHE_PREFIX}"
    --stage1-microbatch-size "${STAGE1_MICROBATCH_SIZE}"
    --stage2-batch-size "${STAGE2_BATCH_SIZE}"
    --sparsification "${SPARSIFICATION}"
    --projection "${PROJECTION}"
    --hessian "${HESSIAN}"
    --damping "${DAMPING}"
    --window-size "${WINDOW_SIZE}"
    --window-aggregation "${WINDOW_AGGREGATION}"
    --quality-score "${WINDOW_QUALITY_SCORE}"
  )
  if [[ -n "${WINDOW_STRIDE}" ]]; then
    cmd+=(--window-stride "${WINDOW_STRIDE}")
  fi
  if [[ -n "${MAX_TRAIN_BATCHES}" ]]; then
    cmd+=(--max-train-batches "${MAX_TRAIN_BATCHES}")
  fi
  if [[ -n "${MAX_EVAL_EPISODES}" ]]; then
    cmd+=(--max-eval-episodes "${MAX_EVAL_EPISODES}")
  fi
  if [[ -n "${MAX_EVAL_STEPS_PER_EPISODE}" ]]; then
    cmd+=(--max-eval-steps-per-episode "${MAX_EVAL_STEPS_PER_EPISODE}")
  fi

  echo "[stage] score"
  echo "[info] GPU_SCORE=${GPU_SCORE}"
  CUDA_VISIBLE_DEVICES="${GPU_SCORE}" \
  UV_CACHE_DIR="${UV_CACHE_DIR}" \
    run_cmd "${cmd[@]}"
}

run_select() {
  local -a cmd=(
    uv run python retrain_window_joint_episode_prior_gain_coverage.py
    --score-pkl "${SCORE_PKL}"
    --train-config "${CONFIG_NAME}"
    --dataset-dir "${DATASET_DIR}"
    --output-dir "${SELECTION_DIR}"
    --exp-name "${EXP_NAME}"
    --checkpoint-base-dir "${CHECKPOINT_BASE_DIR}"
    --joint-rule "${JOINT_RULE}"
    --joint-quality-power "${JOINT_QUALITY_POWER}"
    --joint-influence-power "${JOINT_INFLUENCE_POWER}"
    --window-score-weight "${WINDOW_SCORE_WEIGHT}"
    --episode-prior-weight "${EPISODE_PRIOR_WEIGHT}"
    --adaptive-prior-mode "${ADAPTIVE_PRIOR_MODE}"
    --adaptive-prior-strength "${ADAPTIVE_PRIOR_STRENGTH}"
    --interaction-weight "${INTERACTION_WEIGHT}"
    --fusion-normalizer "${FUSION_NORMALIZER}"
    --episode-quality-aggregation "${EPISODE_QUALITY_AGGREGATION}"
    --min-windows-per-episode "${MIN_WINDOWS_PER_EPISODE}"
    --episode-budget-rule "${EPISODE_BUDGET_RULE}"
    --quality-source "${QUALITY_SOURCE}"
    --quality-normalizer "${QUALITY_NORMALIZER}"
    --quality-quantile-low "${QUALITY_QUANTILE_LOW}"
    --quality-quantile-high "${QUALITY_QUANTILE_HIGH}"
    --min-coverage-ratio "${MIN_COVERAGE_RATIO}"
    --relative-gain-threshold "${RELATIVE_GAIN_THRESHOLD}"
    --prepare-only
    --overwrite
  )
  if [[ -n "${MARGINAL_GAIN_THRESHOLD}" ]]; then
    cmd+=(--marginal-gain-threshold "${MARGINAL_GAIN_THRESHOLD}")
  fi
  if [[ -n "${MAX_FRAMES_FOR_NORM_STATS}" ]]; then
    cmd+=(--max-frames-for-norm-stats "${MAX_FRAMES_FOR_NORM_STATS}")
  fi

  echo "[stage] select"
  UV_CACHE_DIR="${UV_CACHE_DIR}" \
  WANDB_MODE="${WANDB_MODE}" \
    run_cmd "${cmd[@]}"
}

run_train() {
  local sample_indices="${SELECTION_DIR}/kept_sample_indices.npy"
  if [[ "${DRY_RUN}" != "1" && ! -f "${sample_indices}" ]]; then
    echo "[error] 缺少筛选结果：${sample_indices}"
    echo "[hint] 先执行 STAGE=select，或直接执行 STAGE=all。"
    exit 1
  fi

  local -a cmd=(
    uv run scripts/train.py "${CONFIG_NAME}"
    --exp-name "${EXP_NAME}"
    --checkpoint-base-dir "${CHECKPOINT_BASE_DIR}"
    --num-train-steps "${NUM_TRAIN_STEPS}"
    --fsdp-devices "${FSDP_DEVICES}"
    --data.sample-indices-path "${sample_indices}"
    --data.assets.assets-dir "${WORK_DIR}"
    --data.assets.asset-id "$(basename "${SELECTION_DIR}")"
  )
  if [[ "${OVERWRITE_TRAIN}" == "1" ]]; then
    cmd+=(--overwrite)
  fi

  echo "[stage] train"
  echo "[info] GPU_TRAIN=${GPU_TRAIN} FSDP_DEVICES=${FSDP_DEVICES}"
  CUDA_VISIBLE_DEVICES="${GPU_TRAIN}" \
  XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION}" \
  UV_CACHE_DIR="${UV_CACHE_DIR}" \
  WANDB_MODE="${WANDB_MODE}" \
    run_cmd "${cmd[@]}"
}

echo "[start] $(date '+%F %T')"
echo "[config] CONFIG_NAME=${CONFIG_NAME}"
echo "[config] DATASET_DIR=${DATASET_DIR}"
echo "[config] CHECKPOINT_DIR=${CHECKPOINT_DIR}"
echo "[config] ROLLOUT_DIR=${ROLLOUT_DIR}"
echo "[config] WORK_DIR=${WORK_DIR}"
echo "[config] STAGE=${STAGE} DRY_RUN=${DRY_RUN}"
echo "[log] ${LOG_FILE}"

case "${STAGE}" in
  all)
    run_score
    run_select
    run_train
    ;;
  score)
    run_score
    ;;
  select)
    run_select
    ;;
  train)
    run_train
    ;;
  *)
    echo "[error] STAGE 只能是 all / score / select / train，当前为：${STAGE}"
    exit 2
    ;;
esac

echo "[done] $(date '+%F %T')"
