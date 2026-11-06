#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/workspace/liushuo/cupid/openvla-oft"
cd "${ROOT_DIR}"

OPENVLA_ENV_NAME="${OPENVLA_ENV_NAME:-openvla-oft}"
ROBO_CONDA_ENV="${ROBO_CONDA_ENV:-robomimic_venv}"
CONDA_SH="${CONDA_SH:-}"
SKIP_PIPELINE="${SKIP_PIPELINE:-0}"

resolve_conda_sh() {
  if [[ -n "${CONDA_SH}" ]] && [[ -f "${CONDA_SH}" ]]; then
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

CONDA_SH_PATH="$(resolve_conda_sh)" || {
  echo "Could not locate conda.sh. Set CONDA_SH to your conda profile script path."
  exit 1
}

bash -lc "
set -euo pipefail
source '${CONDA_SH_PATH}'

if ! conda env list | awk '{print \$1}' | grep -Fxq '${OPENVLA_ENV_NAME}'; then
  echo 'Required training environment ${OPENVLA_ENV_NAME} not found.'
  echo 'Create it manually, then rerun this script.'
  exit 1
fi

conda activate '${OPENVLA_ENV_NAME}'

if ! conda env list | awk '{print \$1}' | grep -Fxq '${ROBO_CONDA_ENV}'; then
  echo 'Required eval environment ${ROBO_CONDA_ENV} not found.'
  echo 'Create it first, then rerun this script.'
  exit 1
fi

if [[ '${SKIP_PIPELINE}' != '1' ]]; then
  bash '${ROOT_DIR}/vla-scripts/run_full_can_mh_openvla_pipeline.sh'
fi
"
