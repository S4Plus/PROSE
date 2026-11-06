#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source "${SCRIPT_DIR}/setup_env.sh"

SERVER_URL="${SERVER_URL:-http://127.0.0.1:8002}"
FRAME_RATE="${FRAME_RATE:-10}"
CHUNK_SIZE="${CHUNK_SIZE:-1}"
CHUNK_EXEC_STEPS="${CHUNK_EXEC_STEPS:-1}"
SAVE_ROLLOUTS_DIR="${SAVE_ROLLOUTS_DIR:-${SCRIPT_DIR}/rollouts}"

python "${SCRIPT_DIR}/client_act_new.py" \
  --server-url "${SERVER_URL}" \
  --frame-rate "${FRAME_RATE}" \
  --chunk-size "${CHUNK_SIZE}" \
  --chunk-exec-steps "${CHUNK_EXEC_STEPS}" \
  --save-rollouts-dir "${SAVE_ROLLOUTS_DIR}" \
  --save-video
