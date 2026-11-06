#!/usr/bin/env bash

set -euo pipefail

# 复现入口：
# 1. influence      -> 先用同一批 rollout 计算 score_episode_window.pkl
# 2. cupid          -> 仅用 influence score 去掉 20% 轨迹
# 3. mysys          -> 轨迹级先去掉 10%，再做 window 级筛选
# 4. cupid_quality  -> 用 influence score + influence-derived quality 联合去掉 20% 轨迹
# 5. all            -> 顺序执行上述四步

METHOD="${1:-all}"

: "${PYTHON_BIN:=python}"
: "${CHECKPOINT_DIR:=/workspace/liushuo/cupid/outputs/stack_cube_205_act/checkpoints/300000/pretrained_model}"
: "${DATASET_ROOT:=/workspace/liushuo/dataset/stack_cube_205}"
: "${DATASET_REPO_ID:=local/stack_cube_205}"
: "${ROLLOUT_DIR:=/workspace/liushuo/stack_cube_act}"

: "${INFLUENCE_ROOT:=/workspace/liushuo/cupid/1_infulenece_score/stack_cube_205}"
: "${SCORE_PKL:=${INFLUENCE_ROOT}/score_episode_window.pkl}"
: "${STAGE_CACHE_DIR:=${INFLUENCE_ROOT}/score}"
: "${STAGE1_TRAIN_PROJ:=${STAGE_CACHE_DIR}/score_episode_window.stage1_train_projections.npy}"

: "${CUPID_QUALITY_DIR:=/workspace/liushuo/cupid/act/cupid_quality/outputs/stack_cube_205}"
: "${CUPID_OUTPUT_DIR:=/workspace/liushuo/cupid/3_model/stack_cube_205_cupid}"
: "${MYSYS_OUTPUT_BASE:=/workspace/liushuo/cupid/3_model/joint_episode_window}"
: "${MYSYS_RUN_NAME:=stack_cube_205_mysys}"
: "${CUPID_QUALITY_TRAIN_DIR:=/workspace/liushuo/cupid/3_model/stack_cube_205_cupid_quality}"

: "${JOB_PREFIX:=stack_cube_205}"
: "${DEVICE:=cuda}"
: "${BATCH_SIZE:=32}"
: "${NUM_WORKERS:=4}"
: "${MAX_STEPS:=300000}"
: "${LOG_EVERY:=200}"
: "${SAVE_EVERY:=20000}"
: "${SEED:=1000}"

: "${CUPID_DROP_QUANTILE:=0.2}"
: "${CUPID_QUALITY_DROP_QUANTILE:=0.2}"
: "${MYSYS_EPISODE_DROP_RATIO:=0.10}"
: "${WINDOW_SIZE:=50}"
: "${WINDOW_STRIDE:=50}"
: "${WINDOW_LOCAL_EPISODE_SCORE_WEIGHT:=0.4}"
: "${WINDOW_LOCAL_QUALITY_WEIGHT:=0.4}"
: "${WINDOW_LOCAL_EPISODE_JOINT_WEIGHT:=0.2}"
: "${MIN_COVERAGE_RATIO:=0.5}"
: "${MARGINAL_GAIN_THRESHOLD:=0.15}"
: "${RELATIVE_GAIN_THRESHOLD:=0.5}"

: "${HF_HOME:=/tmp/hf_home_${JOB_PREFIX}}"
: "${HF_DATASETS_CACHE:=/tmp/hf_datasets_cache_${JOB_PREFIX}}"

mkdir -p \
  "${INFLUENCE_ROOT}" \
  "${STAGE_CACHE_DIR}" \
  "${CUPID_QUALITY_DIR}" \
  "${MYSYS_OUTPUT_BASE}" \
  "$(dirname "${CUPID_QUALITY_TRAIN_DIR}")" \
  "${HF_HOME}" \
  "${HF_DATASETS_CACHE}"

run() {
  echo
  echo "[run] $*"
  "$@"
}

require_path() {
  local path="$1"
  if [[ ! -e "${path}" ]]; then
    echo "[error] 路径不存在: ${path}" >&2
    exit 1
  fi
}

run_influence() {
  require_path "${CHECKPOINT_DIR}"
  require_path "${DATASET_ROOT}"
  require_path "${ROLLOUT_DIR}"

  run env \
    HF_HOME="${HF_HOME}" \
    HF_DATASETS_CACHE="${HF_DATASETS_CACHE}" \
    "${PYTHON_BIN}" /workspace/liushuo/cupid/act/myscripts/cupid_act_influence.py \
      --pretrained_dir "${CHECKPOINT_DIR}" \
      --train_dataset_path "${DATASET_ROOT}" \
      --eval_rollout_dir "${ROLLOUT_DIR}" \
      --output_path "${SCORE_PKL}" \
      --stage_cache_prefix "${STAGE_CACHE_DIR}" \
      --aggregate_mode episode \
      --window_size "${WINDOW_SIZE}" \
      --window_stride "${WINDOW_STRIDE}"
}

run_cupid() {
  require_path "${DATASET_ROOT}"
  require_path "${SCORE_PKL}"

  run env \
    HF_HOME="${HF_HOME}" \
    HF_DATASETS_CACHE="${HF_DATASETS_CACHE}" \
    "${PYTHON_BIN}" /workspace/liushuo/cupid/act/train_act_cupid_reweight.py \
      --dataset.root "${DATASET_ROOT}" \
      --dataset.repo_id "${DATASET_REPO_ID}" \
      --score-pkl "${SCORE_PKL}" \
      --output-dir "${CUPID_OUTPUT_DIR}" \
      --job_name "${JOB_PREFIX}_cupid" \
      --device "${DEVICE}" \
      --batch-size "${BATCH_SIZE}" \
      --num-workers "${NUM_WORKERS}" \
      --max-steps "${MAX_STEPS}" \
      --log-every "${LOG_EVERY}" \
      --save-every "${SAVE_EVERY}" \
      --seed "${SEED}" \
      --drop-bad-ratio 1.1 \
      --drop-weight-quantile "${CUPID_DROP_QUANTILE}" \
      --weight-mode influence \
      --use-weighted-sampler
}

run_mysys() {
  require_path "${DATASET_ROOT}"
  require_path "${SCORE_PKL}"

  run env \
    HF_HOME="${HF_HOME}" \
    HF_DATASETS_CACHE="${HF_DATASETS_CACHE}" \
    PYTHON_BIN="${PYTHON_BIN}" \
    DATASET_ROOT="${DATASET_ROOT}" \
    DATASET_REPO_ID="${DATASET_REPO_ID}" \
    SCORE_PKL="${SCORE_PKL}" \
    OUTPUT_BASE="${MYSYS_OUTPUT_BASE}" \
    RUN_NAME="${MYSYS_RUN_NAME}" \
    JOB_NAME="${MYSYS_RUN_NAME}" \
    QUALITY_SOURCE="trajectory_eval" \
    EPISODE_DROP_RATIO="${MYSYS_EPISODE_DROP_RATIO}" \
    TRAIN_DEVICE="${DEVICE}" \
    TRAIN_BATCH_SIZE="${BATCH_SIZE}" \
    TRAIN_NUM_WORKERS="${NUM_WORKERS}" \
    TRAIN_MAX_STEPS="${MAX_STEPS}" \
    TRAIN_LOG_EVERY="${LOG_EVERY}" \
    TRAIN_SAVE_EVERY="${SAVE_EVERY}" \
    TRAIN_SEED="${SEED}" \
    WINDOW_SIZE="${WINDOW_SIZE}" \
    WINDOW_STRIDE="${WINDOW_STRIDE}" \
    WINDOW_LOCAL_EPISODE_SCORE_WEIGHT="${WINDOW_LOCAL_EPISODE_SCORE_WEIGHT}" \
    WINDOW_LOCAL_QUALITY_WEIGHT="${WINDOW_LOCAL_QUALITY_WEIGHT}" \
    WINDOW_LOCAL_EPISODE_JOINT_WEIGHT="${WINDOW_LOCAL_EPISODE_JOINT_WEIGHT}" \
    MIN_COVERAGE_RATIO="${MIN_COVERAGE_RATIO}" \
    MARGINAL_GAIN_THRESHOLD="${MARGINAL_GAIN_THRESHOLD}" \
    RELATIVE_GAIN_THRESHOLD="${RELATIVE_GAIN_THRESHOLD}" \
    bash /workspace/liushuo/cupid/act/myscripts/run_filter_and_train_joint_episode_window.sh
}

run_cupid_quality() {
  require_path "${DATASET_ROOT}"
  require_path "${SCORE_PKL}"
  require_path "${STAGE1_TRAIN_PROJ}"

  run env \
    HF_HOME="${HF_HOME}" \
    HF_DATASETS_CACHE="${HF_DATASETS_CACHE}" \
    "${PYTHON_BIN}" /workspace/liushuo/cupid/act/cupid_quality/run_cupid_quality_train.py \
      --score-pkl "${SCORE_PKL}" \
      --stage1-train-projections "${STAGE1_TRAIN_PROJ}" \
      --quality-output-dir "${CUPID_QUALITY_DIR}" \
      --dataset.root "${DATASET_ROOT}" \
      --dataset.repo_id "${DATASET_REPO_ID}" \
      --train-output-dir "${CUPID_QUALITY_TRAIN_DIR}" \
      --job-name "${JOB_PREFIX}_cupid_quality" \
      --reconstruct-device "${DEVICE}" \
      --train-device "${DEVICE}" \
      --batch-size "${BATCH_SIZE}" \
      --normalize-by-trajectory-length \
      --normalizer quantile \
      --quantile-low 0.05 \
      --quantile-high 0.95 \
      --joint-drop-quantile "${CUPID_QUALITY_DROP_QUANTILE}" \
      --joint-quality-power 1.0 \
      --joint-influence-power 1.0 \
      --weight-mode influence \
      --use-weighted-sampler \
      --num-workers "${NUM_WORKERS}" \
      --max-steps "${MAX_STEPS}" \
      --log-every "${LOG_EVERY}" \
      --save-every "${SAVE_EVERY}" \
      --seed "${SEED}"
}

case "${METHOD}" in
  influence)
    run_influence
    ;;
  cupid)
    run_cupid
    ;;
  mysys)
    run_mysys
    ;;
  cupid_quality)
    run_cupid_quality
    ;;
  all)
    run_influence
    run_cupid
    run_mysys
    run_cupid_quality
    ;;
  *)
    echo "用法: $0 {influence|cupid|mysys|cupid_quality|all}" >&2
    exit 1
    ;;
esac
