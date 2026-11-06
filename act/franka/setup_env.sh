#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ -f /opt/ros/noetic/setup.bash ]]; then
  # shellcheck disable=SC1091
  source /opt/ros/noetic/setup.bash
fi

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

echo "环境已初始化"
echo "REPO_ROOT=${REPO_ROOT}"
echo "PYTHONPATH=${PYTHONPATH}"

