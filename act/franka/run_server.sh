#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

source "${SCRIPT_DIR}/setup_env.sh"

CHECKPOINT="${CHECKPOINT:-/ext_workspace/liushuo/ring_on_bearing_act}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8002}"
DEVICE="${DEVICE:-cuda:0}"

python "${SCRIPT_DIR}/server_act.py" \
  --checkpoint "${CHECKPOINT}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --device "${DEVICE}"
