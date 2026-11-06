#!/usr/bin/env bash
set -euo pipefail

DATASET_ROOT="${1:-/ext_workspace/liushuo/swk/CB81}"
OUT_DIR="${2:-libero/checkpoints/cb81_act}"

MAX_STEPS="${MAX_STEPS:-100000}"
BATCH_SIZE="${BATCH_SIZE:-32}"
DEVICE="${DEVICE:-cuda}"
CHUNK_SIZE="${CHUNK_SIZE:-50}"
N_ACTION_STEPS="${N_ACTION_STEPS:-50}"

if [[ -f "${DATASET_ROOT}/meta/stats.json" ]]; then
  REPO_ID="$(basename "${DATASET_ROOT}")"
  LEROBOT_HOME="$(dirname "${DATASET_ROOT}")"
elif [[ -f "${DATASET_ROOT}/local/cb81/meta/stats.json" ]]; then
  REPO_ID="local/cb81"
  LEROBOT_HOME="${DATASET_ROOT}"
else
  echo "ERROR: cannot find meta/stats.json under ${DATASET_ROOT}" >&2
  echo "Please edit train_cb81_act.sh to set REPO_ID and LEROBOT_HOME." >&2
  exit 1
fi

python libero/train_act.py \
  --repo-id "${REPO_ID}" \
  --lerobot-home "${LEROBOT_HOME}" \
  --output-dir "${OUT_DIR}" \
  --device "${DEVICE}" \
  --batch-size "${BATCH_SIZE}" \
  --max-steps "${MAX_STEPS}" \
  --chunk-size "${CHUNK_SIZE}" \
  --n-action-steps "${N_ACTION_STEPS}"
