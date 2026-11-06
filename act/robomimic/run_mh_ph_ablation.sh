#!/usr/bin/env bash
set -euo pipefail

# 三任务通用 ACT 补充实验入口：
# 1. 从 mh 两阶段筛选基础集出发
# 2. 追加随机 ph 补充集，得到 mh_random_ph
# 3. 追加按 ACT 偏好原型挑选的 ph 补充集，得到 mh_select_ph
#
# 用法示例：
#   TASK=can BUILD_DATASETS=1 bash act/robomimic/run_mh_ph_ablation.sh
#   TASK=square BUILD_DATASETS=1 bash act/robomimic/run_mh_ph_ablation.sh
#   TASK=transport BUILD_DATASETS=1 bash act/robomimic/run_mh_ph_ablation.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

TASK="${TASK:?请设置 TASK=can|square|transport}"
case "${TASK}" in
  can|square|transport)
    ;;
  *)
    echo "不支持的 TASK=${TASK}，仅支持 can|square|transport" >&2
    exit 1
    ;;
esac

PYTHON_BIN="${PYTHON_BIN:-python}"
LEROBOT_TRAIN_BIN="${LEROBOT_TRAIN_BIN:-lerobot-train}"
BUILD_SCRIPT="${BUILD_SCRIPT:-${PROJECT_ROOT}/scripts/build_openpi05_canph_ablation.py}"

DATASET_MH="${DATASET_MH:-${PROJECT_ROOT}/datasets/lerobot/${TASK}_mh}"
DATASET_PH="${DATASET_PH:-/workspace/liushuo/robomimic/robomimic/datasets/lerobot/${TASK}_ph}"
KEEP_LIST="${KEEP_LIST:-${PROJECT_ROOT}/openpi/openpi-script/output/joint_episode_window_serial/${TASK}_mh_joint_episode_window/selection/kept_episodes.txt}"
COMPARE_CSV="${COMPARE_CSV:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/output/act_${TASK}ph_ablation}"
MANIFEST_SUMMARY="${MANIFEST_SUMMARY:-${OUTPUT_ROOT}/manifests/build_summary.json}"

MODEL_OUTPUT_ROOT="${MODEL_OUTPUT_ROOT:-${PROJECT_ROOT}/3_model/mh_ph_ablation/act/${TASK}}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/3_model/logs/mh_ph_ablation/act/${TASK}}"

HF_CACHE_ROOT="${HF_CACHE_ROOT:-${PROJECT_ROOT}/.cache/huggingface}"
HF_HOME_BASE="${HF_HOME_BASE:-${HF_CACHE_ROOT}/home}"
HF_DATASETS_CACHE_BASE="${HF_DATASETS_CACHE_BASE:-${HF_CACHE_ROOT}/datasets}"

BUILD_DATASETS="${BUILD_DATASETS:-0}"
BUILD_OVERWRITE="${BUILD_OVERWRITE:-0}"
TRAIN_OVERWRITE="${TRAIN_OVERWRITE:-1}"
RUN_VARIANTS="${RUN_VARIANTS:-random pref}"
FOREGROUND="${FOREGROUND:-0}"

PROTOTYPE_SOURCE="${PROTOTYPE_SOURCE:-act_keep}"
PREFERENCE_COHORT="${PREFERENCE_COHORT:-pi05_only}"
PROTOTYPE_TOP_K="${PROTOTYPE_TOP_K:-}"
PROTOTYPE_TOP_RATIO="${PROTOTYPE_TOP_RATIO:-0.2}"
SUPPLEMENT_COUNT="${SUPPLEMENT_COUNT:-}"
SUPPLEMENT_RATIO="${SUPPLEMENT_RATIO:-}"
RANDOM_SEED="${RANDOM_SEED:-42}"

MAX_STEPS="${MAX_STEPS:-100000}"
CUDA_RANDOM="${CUDA_RANDOM:-6}"
CUDA_PREF="${CUDA_PREF:-5}"

mkdir -p "${MODEL_OUTPUT_ROOT}" "${LOG_DIR}" "${HF_HOME_BASE}" "${HF_DATASETS_CACHE_BASE}"

if [[ -z "${COMPARE_CSV}" ]]; then
  if [[ -f "${PROJECT_ROOT}/output/compare_act_openpi_${TASK}_mh_retest/episode_comparison.csv" ]]; then
    COMPARE_CSV="${PROJECT_ROOT}/output/compare_act_openpi_${TASK}_mh_retest/episode_comparison.csv"
  else
    COMPARE_CSV="${PROJECT_ROOT}/output/compare_act_openpi_${TASK}_mh/episode_comparison.csv"
  fi
fi

if [[ "${BUILD_DATASETS}" == "1" ]]; then
  build_cmd=(
    "${PYTHON_BIN}" "${BUILD_SCRIPT}"
    --can-mh-dataset "${DATASET_MH}"
    --can-mh-keep-list "${KEEP_LIST}"
    --can-ph-dataset "${DATASET_PH}"
    --compare-csv "${COMPARE_CSV}"
    --output-root "${OUTPUT_ROOT}"
    --prototype-source "${PROTOTYPE_SOURCE}"
    --cohort "${PREFERENCE_COHORT}"
    --random-seed "${RANDOM_SEED}"
  )
  if [[ -n "${PROTOTYPE_TOP_K}" ]]; then
    build_cmd+=(--prototype-top-k "${PROTOTYPE_TOP_K}")
  elif [[ -n "${PROTOTYPE_TOP_RATIO}" ]]; then
    build_cmd+=(--prototype-top-ratio "${PROTOTYPE_TOP_RATIO}")
  fi
  if [[ -n "${SUPPLEMENT_COUNT}" ]]; then
    build_cmd+=(--supplement-count "${SUPPLEMENT_COUNT}")
  elif [[ -n "${SUPPLEMENT_RATIO}" ]]; then
    build_cmd+=(--supplement-ratio "${SUPPLEMENT_RATIO}")
  fi
  if [[ "${BUILD_OVERWRITE}" == "1" ]]; then
    build_cmd+=(--overwrite)
  fi

  echo "[build:${TASK}] ${build_cmd[*]}"
  "${build_cmd[@]}"
fi

if [[ ! -f "${MANIFEST_SUMMARY}" ]]; then
  echo "缺少构建摘要文件: ${MANIFEST_SUMMARY}" >&2
  echo "可先设置 BUILD_DATASETS=1 重新构建 ${TASK} 的补充数据集。" >&2
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
print(outputs["merged_random_dataset"])
print(outputs["merged_preference_dataset"])
print(summary["supplement_count"])
print(summary.get("preference_prototype_source", ""))
print(summary.get("preference_cohort", ""))
PY
)

DATASET_RANDOM="${summary_fields[0]}"
DATASET_PREF="${summary_fields[1]}"
SUMMARY_K="${summary_fields[2]}"
SUMMARY_SOURCE="${summary_fields[3]}"
SUMMARY_COHORT="${summary_fields[4]}"

launch_train() {
  local variant="$1"
  local dataset_dir="$2"
  local gpu_id="$3"

  if [[ ! -d "${dataset_dir}" ]]; then
    echo "数据集不存在，跳过 ${variant}: ${dataset_dir}" >&2
    return 0
  fi

  local run_name="${TASK}_mh_${variant}_ph_k${SUMMARY_K}_actpref"
  local output_dir="${MODEL_OUTPUT_ROOT}/${run_name}"
  local log_file="${LOG_DIR}/${run_name}.log"
  local hf_home="${HF_HOME_BASE}/${run_name}"
  local hf_datasets_cache="${HF_DATASETS_CACHE_BASE}/${run_name}"

  if [[ -d "${output_dir}" ]]; then
    if [[ "${TRAIN_OVERWRITE}" == "1" ]]; then
      echo "[train:${TASK}:${variant}] remove existing output_dir=${output_dir}"
      rm -rf "${output_dir}"
      rm -f "${log_file}"
    else
      echo "[train:${TASK}:${variant}] output_dir exists, skip because TRAIN_OVERWRITE=${TRAIN_OVERWRITE}: ${output_dir}"
      return 0
    fi
  fi

  mkdir -p "$(dirname "${output_dir}")" "${hf_home}" "${hf_datasets_cache}"

  local cmd=(
    "${LEROBOT_TRAIN_BIN}"
    --policy.type=act
    --policy.device=cuda
    --policy.push_to_hub=false
    --policy.repo_id="ls/${run_name}"
    --dataset.repo_id="ls/${run_name}"
    --dataset.root="${dataset_dir}"
    --output_dir="${output_dir}"
    --job_name="${run_name}"
    --steps="${MAX_STEPS}"
  )

  echo "[train:${TASK}:${variant}] dataset=${dataset_dir}"
  echo "[train:${TASK}:${variant}] output=${output_dir}"
  echo "[train:${TASK}:${variant}] log=${log_file}"

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
    echo "[train:${TASK}:${variant}] pid=$!"
  fi
}

for variant in ${RUN_VARIANTS}; do
  case "${variant}" in
    random)
      launch_train "random" "${DATASET_RANDOM}" "${CUDA_RANDOM}"
      ;;
    pref)
      launch_train "select" "${DATASET_PREF}" "${CUDA_PREF}"
      ;;
    *)
      echo "不支持的 variant: ${variant}" >&2
      exit 1
      ;;
  esac
done

cat <<EOF
[summary]
task=${TASK}
preference_source=${SUMMARY_SOURCE}
preference_cohort=${SUMMARY_COHORT}
supplement_count=${SUMMARY_K}
compare_csv=${COMPARE_CSV}
keep_list=${KEEP_LIST}
random_dataset=${DATASET_RANDOM}
pref_dataset=${DATASET_PREF}
model_output_root=${MODEL_OUTPUT_ROOT}
log_dir=${LOG_DIR}
EOF
