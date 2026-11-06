#!/usr/bin/env bash
set -euo pipefail

# 基于当前 can_mh 两阶段筛选结果，构造并训练 ACT 补充实验数据集：
# 1. mh_random_ph: 在基础集上追加随机 ph 补充集
# 2. mh_select_ph: 在基础集上追加按 OpenPI05 偏好原型挑选的 ph 补充集
# 如需重跑 only_mh 基线，可额外把 RUN_VARIANTS 设为 "base random pref"。

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
LEROBOT_TRAIN_BIN="${LEROBOT_TRAIN_BIN:-lerobot-train}"
BUILD_SCRIPT="${BUILD_SCRIPT:-${PROJECT_ROOT}/scripts/build_openpi05_canph_ablation.py}"

OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/output/openpi05_canph_ablation}"
MANIFEST_SUMMARY="${MANIFEST_SUMMARY:-${OUTPUT_ROOT}/manifests/build_summary.json}"
MODEL_OUTPUT_ROOT="${MODEL_OUTPUT_ROOT:-${PROJECT_ROOT}/3_model/can_mh_ph_ablation/act}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/3_model/logs/can_mh_ph_ablation/act}"

HF_CACHE_ROOT="${HF_CACHE_ROOT:-${PROJECT_ROOT}/.cache/huggingface}"
HF_HOME_BASE="${HF_HOME_BASE:-${HF_CACHE_ROOT}/home}"
HF_DATASETS_CACHE_BASE="${HF_DATASETS_CACHE_BASE:-${HF_CACHE_ROOT}/datasets}"

BUILD_DATASETS="${BUILD_DATASETS:-0}"
BUILD_OVERWRITE="${BUILD_OVERWRITE:-0}"
RUN_VARIANTS="${RUN_VARIANTS:-random pref}"
FOREGROUND="${FOREGROUND:-0}"

MAX_STEPS="${MAX_STEPS:-100000}"
CUDA_BASE="${CUDA_BASE:-7}"
CUDA_RANDOM="${CUDA_RANDOM:-6}"
CUDA_PREF="${CUDA_PREF:-5}"

SUPPLEMENT_COUNT="${SUPPLEMENT_COUNT:-}"
SUPPLEMENT_RATIO="${SUPPLEMENT_RATIO:-}"
RANDOM_SEED="${RANDOM_SEED:-42}"

DATASET_BASE="${DATASET_BASE:-}"
DATASET_RANDOM="${DATASET_RANDOM:-}"
DATASET_PREF="${DATASET_PREF:-}"

mkdir -p "${MODEL_OUTPUT_ROOT}" "${LOG_DIR}" "${HF_HOME_BASE}" "${HF_DATASETS_CACHE_BASE}"

if [[ "${BUILD_DATASETS}" == "1" ]]; then
  build_cmd=("${PYTHON_BIN}" "${BUILD_SCRIPT}")
  if [[ -n "${SUPPLEMENT_COUNT}" ]]; then
    build_cmd+=(--supplement-count "${SUPPLEMENT_COUNT}")
  elif [[ -n "${SUPPLEMENT_RATIO}" ]]; then
    build_cmd+=(--supplement-ratio "${SUPPLEMENT_RATIO}")
  fi
  if [[ -n "${RANDOM_SEED}" ]]; then
    build_cmd+=(--random-seed "${RANDOM_SEED}")
  fi
  if [[ "${BUILD_OVERWRITE}" == "1" ]]; then
    build_cmd+=(--overwrite)
  fi

  echo "[build] ${build_cmd[*]}"
  "${build_cmd[@]}"
fi

if [[ ! -f "${MANIFEST_SUMMARY}" ]]; then
  echo "缺少构建摘要文件: ${MANIFEST_SUMMARY}" >&2
  echo "可先设置 BUILD_DATASETS=1，或手动运行 ${BUILD_SCRIPT}" >&2
  exit 1
fi

mapfile -t summary_fields < <(
  "${PYTHON_BIN}" - "${MANIFEST_SUMMARY}" <<'PY'
import json
import sys

summary_path = sys.argv[1]
with open(summary_path, "r", encoding="utf-8") as f:
    summary = json.load(f)

outputs = summary["outputs"]
print(outputs["can_mh_twostage_dataset"])
print(outputs["merged_random_dataset"])
print(outputs["merged_preference_dataset"])
print(summary["supplement_count"])
PY
)

summary_base="${summary_fields[0]}"
summary_random="${summary_fields[1]}"
summary_pref="${summary_fields[2]}"
summary_k="${summary_fields[3]}"

if [[ -z "${DATASET_BASE}" ]]; then
  DATASET_BASE="${summary_base}"
fi
if [[ -z "${DATASET_RANDOM}" ]]; then
  DATASET_RANDOM="${summary_random}"
fi
if [[ -z "${DATASET_PREF}" ]]; then
  DATASET_PREF="${summary_pref}"
fi

launch_train() {
  local variant="$1"
  local dataset_dir="$2"
  local gpu_id="$3"

  if [[ ! -d "${dataset_dir}" ]]; then
    echo "数据集不存在，跳过 ${variant}: ${dataset_dir}" >&2
    return 0
  fi

  local dataset_name
  dataset_name="$(basename "${dataset_dir}")"
  local run_name="act_${dataset_name}"
  local output_dir="${MODEL_OUTPUT_ROOT}/${dataset_name}"
  local log_file="${LOG_DIR}/${run_name}.log"
  local hf_home="${HF_HOME_BASE}/${run_name}"
  local hf_datasets_cache="${HF_DATASETS_CACHE_BASE}/${run_name}"

  mkdir -p "${output_dir}" "${hf_home}" "${hf_datasets_cache}"

  local cmd=(
    "${LEROBOT_TRAIN_BIN}"
    --policy.type=act
    --policy.device=cuda
    --policy.push_to_hub=false
    --policy.repo_id="ls/${dataset_name}"
    --dataset.repo_id="ls/${dataset_name}"
    --dataset.root="${dataset_dir}"
    --output_dir="${output_dir}"
    --job_name="${run_name}"
    --steps="${MAX_STEPS}"
  )

  echo "[train:${variant}] dataset=${dataset_dir}"
  echo "[train:${variant}] output=${output_dir}"
  echo "[train:${variant}] log=${log_file}"

  if [[ "${FOREGROUND}" == "1" ]]; then
    env \
      CUDA_VISIBLE_DEVICES="${gpu_id}" \
      HF_HOME="${hf_home}" \
      HF_DATASETS_CACHE="${hf_datasets_cache}" \
      "${cmd[@]}" | tee "${log_file}"
  else
    nohup env \
      CUDA_VISIBLE_DEVICES="${gpu_id}" \
      HF_HOME="${hf_home}" \
      HF_DATASETS_CACHE="${hf_datasets_cache}" \
      "${cmd[@]}" \
      > "${log_file}" 2>&1 &
    echo "[train:${variant}] pid=$!"
  fi
}

for variant in ${RUN_VARIANTS}; do
  case "${variant}" in
    base)
      launch_train "base" "${DATASET_BASE}" "${CUDA_BASE}"
      ;;
    random)
      launch_train "random" "${DATASET_RANDOM}" "${CUDA_RANDOM}"
      ;;
    pref)
      launch_train "pref" "${DATASET_PREF}" "${CUDA_PREF}"
      ;;
    *)
      echo "不支持的 variant: ${variant}" >&2
      exit 1
      ;;
  esac
done

cat <<EOF
[summary]
supplement_count=${summary_k}
base_dataset=${DATASET_BASE}
random_dataset=${DATASET_RANDOM}
pref_dataset=${DATASET_PREF}
model_output_root=${MODEL_OUTPUT_ROOT}
log_dir=${LOG_DIR}
EOF
