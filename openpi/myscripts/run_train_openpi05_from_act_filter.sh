#!/usr/bin/env bash
set -euo pipefail

OPENPI_ROOT="/workspace/liushuo/cupid/openpi"

DATASET_DIR="${DATASET_DIR:-/workspace/liushuo/cupid/datasets/lerobot/can_mh}"
TRAIN_CONFIG="${TRAIN_CONFIG:-pi05_can_mh_lora}"

ACT_FILTER_MODE="${ACT_FILTER_MODE:-cupid}"
ACT_FILTER_DIR="${ACT_FILTER_DIR:-}"
SELECTION_NPZ="${SELECTION_NPZ:-}"
KEEP_EPISODES_TXT="${KEEP_EPISODES_TXT:-}"
EPISODES_CSV="${EPISODES_CSV:-}"
CSV_KEEP_COLUMN="${CSV_KEEP_COLUMN:-}"
CSV_DROP_COLUMN="${CSV_DROP_COLUMN:-}"

OUTPUT_BASE="${OUTPUT_BASE:-${OPENPI_ROOT}/myscripts/output/act_filter_to_openpi}"
RUN_NAME="${RUN_NAME:-$(basename "${DATASET_DIR}")_${ACT_FILTER_MODE}_openpi05}"
JOB_NAME="${JOB_NAME:-${RUN_NAME}}"
TRAIN_OUTPUT_DIR="${TRAIN_OUTPUT_DIR:-${OUTPUT_BASE}/${RUN_NAME}/train}"

CHECKPOINT_BASE_DIR="${CHECKPOINT_BASE_DIR:-${OPENPI_ROOT}/checkpoints}"
NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-10000}"
MAX_FRAMES_FOR_NORM_STATS="${MAX_FRAMES_FOR_NORM_STATS:-}"
WANDB_MODE="${WANDB_MODE:-disabled}"
UV_CACHE_DIR="${UV_CACHE_DIR:-${OPENPI_ROOT}/tmp/uv-cache}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
OVERWRITE="${OVERWRITE:-1}"

mkdir -p "${OUTPUT_BASE}" "${TRAIN_OUTPUT_DIR}"

resolve_inputs_from_dir() {
  local root="$1"
  if [[ -n "${SELECTION_NPZ}" || -n "${KEEP_EPISODES_TXT}" || -n "${EPISODES_CSV}" ]]; then
    return
  fi
  if [[ -z "${root}" ]]; then
    return
  fi

  if [[ -f "${root}/joint_episode_window_selection.npz" ]]; then
    SELECTION_NPZ="${root}/joint_episode_window_selection.npz"
    return
  fi
  if [[ -f "${root}/cupid_reweight_keep.txt" ]]; then
    KEEP_EPISODES_TXT="${root}/cupid_reweight_keep.txt"
    return
  fi
  if [[ -f "${root}/quality_keep.txt" ]]; then
    KEEP_EPISODES_TXT="${root}/quality_keep.txt"
    return
  fi
  if [[ -f "${root}/cupid_reweight_episodes.csv" ]]; then
    EPISODES_CSV="${root}/cupid_reweight_episodes.csv"
    return
  fi
  if [[ -f "${root}/quality_episodes.csv" ]]; then
    EPISODES_CSV="${root}/quality_episodes.csv"
    return
  fi
}

resolve_inputs_from_dir "${ACT_FILTER_DIR}"

if [[ -n "${ACT_FILTER_DIR}" && ! -d "${ACT_FILTER_DIR}" ]]; then
  echo "ACT filter directory does not exist: ${ACT_FILTER_DIR}" >&2
  exit 1
fi

cmd=(
  uv run python "${OPENPI_ROOT}/myscripts/prepare_openpi_from_act_filter.py"
  --train-config "${TRAIN_CONFIG}"
  --dataset-dir "${DATASET_DIR}"
  --output-dir "${TRAIN_OUTPUT_DIR}"
  --exp-name "${JOB_NAME}"
  --checkpoint-base-dir "${CHECKPOINT_BASE_DIR}"
  --num-train-steps "${NUM_TRAIN_STEPS}"
  --source-tag "${ACT_FILTER_MODE}"
)

if [[ -n "${SELECTION_NPZ}" ]]; then
  cmd+=(--selection-npz "${SELECTION_NPZ}")
elif [[ -n "${KEEP_EPISODES_TXT}" ]]; then
  cmd+=(--keep-episodes-txt "${KEEP_EPISODES_TXT}")
elif [[ -n "${EPISODES_CSV}" ]]; then
  cmd+=(--episodes-csv "${EPISODES_CSV}")
else
  if [[ -n "${ACT_FILTER_DIR}" ]]; then
    echo "No ACT filter artifact resolved under ${ACT_FILTER_DIR}." >&2
    echo "Expected one of: joint_episode_window_selection.npz, cupid_reweight_keep.txt, quality_keep.txt, cupid_reweight_episodes.csv, quality_episodes.csv" >&2
  else
    echo "No ACT filter artifact resolved. Set one of SELECTION_NPZ / KEEP_EPISODES_TXT / EPISODES_CSV, or ACT_FILTER_DIR." >&2
  fi
  exit 1
fi

if [[ -n "${CSV_KEEP_COLUMN}" ]]; then
  cmd+=(--csv-keep-column "${CSV_KEEP_COLUMN}")
fi
if [[ -n "${CSV_DROP_COLUMN}" ]]; then
  cmd+=(--csv-drop-column "${CSV_DROP_COLUMN}")
fi

if [[ -n "${MAX_FRAMES_FOR_NORM_STATS}" ]]; then
  cmd+=(--max-frames-for-norm-stats "${MAX_FRAMES_FOR_NORM_STATS}")
fi
if [[ "${SKIP_TRAIN}" == "1" ]]; then
  cmd+=(--prepare-only)
fi
if [[ "${OVERWRITE}" == "1" ]]; then
  cmd+=(--overwrite)
fi

echo "[run] train openpi05 from ACT filter"
printf ' %q' "${cmd[@]}"
printf '\n'
UV_CACHE_DIR="${UV_CACHE_DIR}" WANDB_MODE="${WANDB_MODE}" "${cmd[@]}"
