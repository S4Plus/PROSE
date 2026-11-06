#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/workspace/liushuo/cupid/openvla-oft"
cd "${ROOT_DIR}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

SERVER_PID=""

cleanup() {
  if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" >/dev/null 2>&1; then
    kill "${SERVER_PID}" >/dev/null 2>&1 || true
    wait "${SERVER_PID}" >/dev/null 2>&1 || true
  fi
}

trap cleanup EXIT

resolve_conda_sh() {
  if [[ -n "${CONDA_SH:-}" ]] && [[ -f "${CONDA_SH}" ]]; then
    printf '%s\n' "${CONDA_SH}"
    return 0
  fi
  if [[ -n "${CONDA_EXE:-}" ]]; then
    local conda_base
    conda_base="$(cd "$(dirname "${CONDA_EXE}")/.." && pwd)"
    if [[ -f "${conda_base}/etc/profile.d/conda.sh" ]]; then
      printf '%s\n' "${conda_base}/etc/profile.d/conda.sh"
      return 0
    fi
  fi
  for candidate in \
    "${HOME}/miniconda3/etc/profile.d/conda.sh" \
    "${HOME}/anaconda3/etc/profile.d/conda.sh" \
    "/opt/conda/etc/profile.d/conda.sh"; do
    if [[ -f "${candidate}" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done
  return 1
}

wait_for_healthz() {
  local url="${1}"
  local max_attempts="${2:-60}"
  local sleep_s="${3:-2}"
  local attempt
  for ((attempt = 1; attempt <= max_attempts; attempt++)); do
    if OPENVLA_HEALTH_URL="${url}" python - <<'PY'
import os
import sys
import urllib.error
import urllib.request

url = os.environ["OPENVLA_HEALTH_URL"]
try:
    with urllib.request.urlopen(url, timeout=2.0) as resp:
        ok = resp.status == 200 and resp.read().decode("utf-8", errors="ignore").strip() == "OK"
except (urllib.error.URLError, TimeoutError, OSError):
    ok = False
sys.exit(0 if ok else 1)
PY
    then
      return 0
    fi
    sleep "${sleep_s}"
  done
  return 1
}

# ===== Common paths =====
DATASET_ROOT="${DATASET_ROOT:-/workspace/liushuo/cupid/datasets/lerobot/can_mh}"
VLA_PATH="${VLA_PATH:-openvla/openvla-7b}"
TASK_DESCRIPTION="${TASK_DESCRIPTION:-Place a coke can from a large bin into a smaller target bin.}"

# ===== Stage 1: full-data training =====
TRAIN_OUTPUT_DIR="${TRAIN_OUTPUT_DIR:-/workspace/liushuo/cupid/openvla-oft/runs/can_mh_openvla_lora_full}"
TRAIN_LOG_DIR="${TRAIN_LOG_DIR:-/workspace/liushuo/cupid/openvla-oft/logs/can_mh_openvla_lora_full}"
MAX_STEPS="${MAX_STEPS:-10000}"
SAVE_FREQ="${SAVE_FREQ:-1000}"
FINAL_CHECKPOINT_STEP="${FINAL_CHECKPOINT_STEP:-${MAX_STEPS}}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${TRAIN_OUTPUT_DIR}--${FINAL_CHECKPOINT_STEP}_chkpt}"

# ===== Stage 2: eval rollout collection =====
EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-/workspace/liushuo/cupid/mimicgen/eval/tmp/can_mh_openvla_eval}"
EVAL_ROLLOUTS_DIR="${EVAL_ROLLOUTS_DIR:-${EVAL_OUTPUT_DIR}/pkl}"
EVAL_VIDEO_DIR="${EVAL_VIDEO_DIR:-${EVAL_OUTPUT_DIR}/videos}"
EVAL_SUMMARY_PATH="${EVAL_SUMMARY_PATH:-${EVAL_OUTPUT_DIR}/summary.json}"
EVAL_LOG_DIR="${EVAL_LOG_DIR:-/workspace/liushuo/cupid/openvla-oft/logs/can_mh_openvla_eval}"
EVAL_SERVER_LOG="${EVAL_SERVER_LOG:-${EVAL_LOG_DIR}/server.log}"
EVAL_CLIENT_LOG="${EVAL_CLIENT_LOG:-${EVAL_LOG_DIR}/client.log}"
EVAL_SERVER_HOST="${EVAL_SERVER_HOST:-127.0.0.1}"
EVAL_SERVER_PORT="${EVAL_SERVER_PORT:-8000}"
EVAL_ENV_NAME="${EVAL_ENV_NAME:-can}"
EVAL_NUM_EPISODES="${EVAL_NUM_EPISODES:-100}"
EVAL_TIMEOUT_S="${EVAL_TIMEOUT_S:-15}"
EVAL_TEST_INIT_MODE="${EVAL_TEST_INIT_MODE:-auto}"
EVAL_EXTRA_ARGS="${EVAL_EXTRA_ARGS:-}"
ROBO_CONDA_ENV="${ROBO_CONDA_ENV:-robomimic_venv}"
SKIP_EVAL_IF_EXISTS="${SKIP_EVAL_IF_EXISTS:-1}"

# ===== Stage 3: score computation =====
SCORE_OUTPUT_DIR="${SCORE_OUTPUT_DIR:-/workspace/liushuo/cupid/openvla-oft/tmp/can_mh_openvla_window_grass}"
SCORE_OUTPUT_PATH="${SCORE_OUTPUT_PATH:-${SCORE_OUTPUT_DIR}/influence_grass.pkl}"
SCORE_CACHE_DIR="${SCORE_CACHE_DIR:-${SCORE_OUTPUT_DIR}/cache}"
SCORE_LOG_DIR="${SCORE_LOG_DIR:-/workspace/liushuo/cupid/openvla-oft/logs/can_mh_openvla_window_grass}"

# ===== Stage 4: filter + retrain =====
RETRAIN_PREP_DIR="${RETRAIN_PREP_DIR:-/workspace/liushuo/cupid/openvla-oft/tmp/can_mh_openvla_window_episode_prior_keep90}"
RETRAIN_RUN_ROOT="${RETRAIN_RUN_ROOT:-/workspace/liushuo/cupid/openvla-oft/runs}"
RETRAIN_EXP_NAME="${RETRAIN_EXP_NAME:-can_mh_openvla_window_episode_prior_keep90}"
RETRAIN_LOG_DIR="${RETRAIN_LOG_DIR:-/workspace/liushuo/cupid/openvla-oft/logs/can_mh_openvla_window_episode_prior_keep90}"
FILTER_METHOD="${FILTER_METHOD:-episode_prior}"
SCORE_SELECTION_DIR="${SCORE_SELECTION_DIR:-${RETRAIN_PREP_DIR}/selection}"

mkdir -p \
  "${TRAIN_LOG_DIR}" \
  "${EVAL_OUTPUT_DIR}" \
  "${EVAL_VIDEO_DIR}" \
  "${EVAL_LOG_DIR}" \
  "${SCORE_OUTPUT_DIR}" \
  "${SCORE_CACHE_DIR}" \
  "${SCORE_LOG_DIR}" \
  "${RETRAIN_PREP_DIR}" \
  "${RETRAIN_LOG_DIR}"

echo "[1/4] Full-data LoRA finetuning on ${DATASET_ROOT}"
python vla-scripts/finetune_openvla_lerobot.py \
  --vla_path "${VLA_PATH}" \
  --dataset_root "${DATASET_ROOT}" \
  --dataset_name can_mh \
  --task_description "${TASK_DESCRIPTION}" \
  --output_dir "${TRAIN_OUTPUT_DIR}" \
  --batch_size "${BATCH_SIZE:-4}" \
  --grad_accumulation_steps "${GRAD_ACCUMULATION_STEPS:-1}" \
  --max_steps "${MAX_STEPS}" \
  --save_freq "${SAVE_FREQ}" \
  --learning_rate "${LEARNING_RATE:-5e-4}" \
  --num_steps_before_decay "${NUM_STEPS_BEFORE_DECAY:-8000}" \
  --action_chunk_size "${ACTION_CHUNK_SIZE:-8}" \
  --num_images_in_input "${NUM_IMAGES_IN_INPUT:-2}" \
  --use_proprio "${USE_PROPRIO:-True}" \
  --use_lora True \
  --lora_rank "${LORA_RANK:-32}" \
  --lora_dropout "${LORA_DROPOUT:-0.0}" \
  --merge_lora_during_training "${MERGE_LORA_DURING_TRAINING:-False}" \
  --num_workers "${NUM_WORKERS:-0}" \
  --seed "${SEED:-7}"

echo "[2/4] Eval rollout collection"
if compgen -G "${EVAL_ROLLOUTS_DIR}/*.pkl" > /dev/null && [[ -f "${EVAL_SUMMARY_PATH}" ]] && [[ "${SKIP_EVAL_IF_EXISTS}" == "1" ]]; then
  echo "Existing eval summary and rollout PKLs found; reusing ${EVAL_SUMMARY_PATH} and ${EVAL_ROLLOUTS_DIR}."
else
  CONDA_SH_PATH="$(resolve_conda_sh)" || {
    echo "Could not locate conda.sh. Set CONDA_SH to your conda profile script path."
    exit 1
  }

  mkdir -p "${EVAL_ROLLOUTS_DIR}" "${EVAL_VIDEO_DIR}"

  echo "Starting OpenVLA websocket server for eval..."
  python vla-scripts/serve_policy_openvla.py \
    --checkpoint-dir "${CHECKPOINT_DIR}" \
    --dataset-root "${DATASET_ROOT}" \
    --host "${EVAL_SERVER_HOST}" \
    --port "${EVAL_SERVER_PORT}" \
    --default-prompt "${TASK_DESCRIPTION}" \
    > "${EVAL_SERVER_LOG}" 2>&1 &
  SERVER_PID=$!

  if ! wait_for_healthz "http://${EVAL_SERVER_HOST}:${EVAL_SERVER_PORT}/healthz" 90 2; then
    echo "OpenVLA eval server failed to become healthy. Check ${EVAL_SERVER_LOG}"
    exit 1
  fi

  export CONDA_SH_PATH
  export ROBO_CONDA_ENV
  export EVAL_SERVER_HOST
  export EVAL_SERVER_PORT
  export EVAL_ENV_NAME
  export EVAL_VIDEO_DIR
  export EVAL_ROLLOUTS_DIR
  export EVAL_SUMMARY_PATH
  export EVAL_NUM_EPISODES
  export EVAL_TIMEOUT_S
  export EVAL_TEST_INIT_MODE
  export EVAL_EXTRA_ARGS

  echo "Running robomimic eval client and collecting summary + rollout PKLs..."
  bash -lc '
    set -euo pipefail
    source "${CONDA_SH_PATH}"
    conda activate "${ROBO_CONDA_ENV}"
    cd /workspace/liushuo/cupid/mimicgen/eval
    OPENVLA_SERVER_URL="ws://${EVAL_SERVER_HOST}:${EVAL_SERVER_PORT}" \
    python client_robomimic_openvla.py \
      --env_name "${EVAL_ENV_NAME}" \
      --n_rollouts "${EVAL_NUM_EPISODES}" \
      --timeout_s "${EVAL_TIMEOUT_S}" \
      --test_init_mode "${EVAL_TEST_INIT_MODE}" \
      --video_dir "${EVAL_VIDEO_DIR}" \
      --save_rollouts_dir "${EVAL_ROLLOUTS_DIR}" \
      --summary_path "${EVAL_SUMMARY_PATH}" \
      ${EVAL_EXTRA_ARGS}
  ' > "${EVAL_CLIENT_LOG}" 2>&1

  cleanup
  SERVER_PID=""
fi

if ! compgen -G "${EVAL_ROLLOUTS_DIR}/*.pkl" > /dev/null; then
  echo "Eval rollout collection did not produce any PKL files in ${EVAL_ROLLOUTS_DIR}."
  exit 1
fi

if [[ ! -f "${EVAL_SUMMARY_PATH}" ]]; then
  echo "Eval did not produce summary json at ${EVAL_SUMMARY_PATH}."
  exit 1
fi

echo "[3/4] Window-level GraSS score computation"
python vla-scripts/influence_openvla_lora_grass_window.py \
  --checkpoint-dir "${CHECKPOINT_DIR}" \
  --eval-rollouts-dir "${EVAL_ROLLOUTS_DIR}" \
  --output-path "${SCORE_OUTPUT_PATH}" \
  --stage-cache-prefix "${SCORE_CACHE_DIR}" \
  --task-description "${TASK_DESCRIPTION}" \
  --window-size "${WINDOW_SIZE:-50}" \
  --window-stride "${WINDOW_STRIDE:-50}" \
  --window-aggregation "${WINDOW_AGGREGATION:-sum}" \
  --quality-score "${QUALITY_SCORE:-zscore_mean}" \
  --sparsification "${SPARSIFICATION:-random_mask-32*32}" \
  --projection "${PROJECTION:-sjlt-1024}" \
  --hessian "${HESSIAN:-eFIM}" \
  --damping "${DAMPING:-1e-4}" \
  --stage1-microbatch-size "${STAGE1_MICROBATCH_SIZE:-1}" \
  --stage2-batch-size "${STAGE2_BATCH_SIZE:-16}" \
  --progress-every "${PROGRESS_EVERY:-200}" \
  --seed "${SEED:-7}"

echo "[4/4] Filter + retrain (method=${FILTER_METHOD})"
if [[ "${FILTER_METHOD}" == "score_joint_episode_window" ]]; then
  mkdir -p "${SCORE_SELECTION_DIR}"

  filter_cmd=(
    python vla-scripts/filter_joint_episode_window.py
    --score-pkl "${SCORE_OUTPUT_PATH}"
    --dataset-root "${DATASET_ROOT}"
    --output-dir "${SCORE_SELECTION_DIR}"
    --quality-source "${SCORE_JOINT_QUALITY_SOURCE:-trajectory_eval}"
    --episode-influence-weight "${SCORE_EPISODE_INFLUENCE_WEIGHT:-0.5}"
    --episode-quality-weight "${SCORE_EPISODE_QUALITY_WEIGHT:-0.5}"
    --episode-drop-ratio "${SCORE_EPISODE_DROP_RATIO:-0.10}"
    --window-size "${SCORE_WINDOW_SIZE:-${WINDOW_SIZE:-50}}"
    --window-stride "${SCORE_WINDOW_STRIDE:-${WINDOW_STRIDE:-50}}"
    --window-local-episode-score-weight "${SCORE_WINDOW_LOCAL_EPISODE_SCORE_WEIGHT:-0.4}"
    --window-local-quality-weight "${SCORE_WINDOW_LOCAL_QUALITY_WEIGHT:-0.4}"
    --window-local-episode-joint-weight "${SCORE_WINDOW_LOCAL_EPISODE_JOINT_WEIGHT:-0.2}"
    --budget-mode auto_marginal
    --min-windows-per-episode "${SCORE_MIN_WINDOWS_PER_EPISODE:-1}"
    --min-coverage-ratio "${SCORE_MIN_COVERAGE_RATIO:-0.5}"
    --marginal-gain-threshold "${SCORE_MARGINAL_GAIN_THRESHOLD:-0.15}"
    --relative-gain-threshold "${SCORE_RELATIVE_GAIN_THRESHOLD:-0.5}"
  )

  if [[ "${SCORE_JOINT_QUALITY_SOURCE:-trajectory_eval}" == "quality_csv" ]]; then
    SCORE_QUALITY_CSV="${SCORE_QUALITY_CSV:?Set SCORE_QUALITY_CSV when SCORE_JOINT_QUALITY_SOURCE=quality_csv}"
    filter_cmd+=(--quality-score-csv "${SCORE_QUALITY_CSV}")
  fi

  prepare_cmd=(
    python vla-scripts/prepare_openvla_from_score_joint_episode_window.py
    --score-pkl "${SCORE_OUTPUT_PATH}"
    --selection-npz "${SCORE_SELECTION_DIR}/joint_episode_window_selection.npz"
    --checkpoint-dir "${CHECKPOINT_DIR}"
    --output-dir "${RETRAIN_PREP_DIR}"
    --retrain-run-root "${RETRAIN_RUN_ROOT}"
    --exp-name "${RETRAIN_EXP_NAME}"
    --num-train-steps "${RETRAIN_STEPS:-10000}"
    --overwrite
  )

  if [[ "${RETRAIN_PREPARE_ONLY:-0}" == "1" ]]; then
    prepare_cmd+=(--prepare-only)
  fi

  printf ' %q' "${filter_cmd[@]}"
  printf '\n'
  "${filter_cmd[@]}"

  printf ' %q' "${prepare_cmd[@]}"
  printf '\n'
  "${prepare_cmd[@]}"
else
  python vla-scripts/retrain_window_joint_episode_prior_openvla.py \
    --score-pkl "${SCORE_OUTPUT_PATH}" \
    --checkpoint-dir "${CHECKPOINT_DIR}" \
    --output-dir "${RETRAIN_PREP_DIR}" \
    --retrain-run-root "${RETRAIN_RUN_ROOT}" \
    --exp-name "${RETRAIN_EXP_NAME}" \
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
    --num-train-steps "${RETRAIN_STEPS:-10000}" \
    --overwrite
fi

echo "Pipeline finished."
