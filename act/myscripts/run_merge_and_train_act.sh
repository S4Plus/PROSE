#!/usr/bin/env bash
# 合并两个 LeRobot 数据集，并基于合并后的数据集启动 ACT 训练。
# 设计目标：
# 1. 复用仓库内的 merge_lerobot_datasets.py，自动处理 episode 重编号与特征对齐。
# 2. 显式设置 HF 缓存目录，避免训练时写到只读目录。
# 3. 支持后台 nohup 训练，减少手工拼接命令的出错概率。

set -euo pipefail

usage() {
  cat <<'EOF'
用法:
  bash act/myscripts/run_merge_and_train_act.sh \
    --dataset-a /path/to/dataset_a \
    --dataset-b /path/to/dataset_b \
    --merged-root /path/to/merged_dataset \
    [--state-layout ee_pose_joint_angles14] \
    [--dataset-repo-id ls/merged_dataset] \
    [--policy-repo-id ls/merged_dataset] \
    [--output-dir /path/to/output_dir] \
    [--job-name merged_dataset_act] \
    [--cuda 0] \
    [--overwrite] \
    [--foreground] \
    [--skip-merge] \
    [--skip-train] \
    [-- extra lerobot-train args]

示例:
  bash act/myscripts/run_merge_and_train_act.sh \
    --dataset-a /workspace/liushuo/dataset/stack_cube_50 \
    --dataset-b /workspace/liushuo/dataset/stack_cube_69 \
    --merged-root /workspace/liushuo/dataset/stack_cube_119 \
    --state-layout ee_pose_joint_angles14 \
    --dataset-repo-id ls/stack_cube_119 \
    --policy-repo-id ls/stack_cube_119 \
    --output-dir /workspace/liushuo/cupid/outputs/stack_cube_119_act \
    --job-name stack_cube_119_act \
    --cuda 7 \
    --overwrite
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
MERGE_SCRIPT="${PROJECT_ROOT}/act/mimicgen/merge_lerobot_datasets.py"

PYTHON_BIN="${PYTHON_BIN:-python}"
LEROBOT_TRAIN_BIN="${LEROBOT_TRAIN_BIN:-lerobot-train}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/act/logs}"
HF_CACHE_ROOT="${HF_CACHE_ROOT:-${PROJECT_ROOT}/.cache/huggingface}"
HF_HOME="${HF_HOME:-${HF_CACHE_ROOT}/home}"
HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_CACHE_ROOT}/datasets}"

DATASET_A=""
DATASET_B=""
MERGED_ROOT=""
DATASET_REPO_ID=""
POLICY_REPO_ID=""
OUTPUT_DIR=""
JOB_NAME=""
CUDA_DEVICE="${CUDA_VISIBLE_DEVICES:-0}"
OVERWRITE=false
RUN_IN_FOREGROUND=false
SKIP_MERGE=false
SKIP_TRAIN=false
STATE_LAYOUT=""
EXTRA_TRAIN_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset-a)
      DATASET_A="$2"
      shift 2
      ;;
    --dataset-b)
      DATASET_B="$2"
      shift 2
      ;;
    --merged-root)
      MERGED_ROOT="$2"
      shift 2
      ;;
    --dataset-repo-id)
      DATASET_REPO_ID="$2"
      shift 2
      ;;
    --state-layout)
      STATE_LAYOUT="$2"
      shift 2
      ;;
    --policy-repo-id)
      POLICY_REPO_ID="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --job-name)
      JOB_NAME="$2"
      shift 2
      ;;
    --cuda)
      CUDA_DEVICE="$2"
      shift 2
      ;;
    --overwrite)
      OVERWRITE=true
      shift
      ;;
    --foreground)
      RUN_IN_FOREGROUND=true
      shift
      ;;
    --skip-merge)
      SKIP_MERGE=true
      shift
      ;;
    --skip-train)
      SKIP_TRAIN=true
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    --)
      shift
      EXTRA_TRAIN_ARGS=("$@")
      break
      ;;
    *)
      echo "未知参数: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "${MERGED_ROOT}" ]]; then
  echo "--merged-root 是必填参数" >&2
  usage >&2
  exit 1
fi

if [[ "${SKIP_MERGE}" == false ]]; then
  if [[ -z "${DATASET_A}" || -z "${DATASET_B}" ]]; then
    echo "执行合并时，--dataset-a 和 --dataset-b 都是必填参数" >&2
    usage >&2
    exit 1
  fi
fi

MERGED_BASENAME="$(basename "${MERGED_ROOT}")"
if [[ -z "${JOB_NAME}" ]]; then
  JOB_NAME="${MERGED_BASENAME}_act"
fi
if [[ -z "${DATASET_REPO_ID}" ]]; then
  DATASET_REPO_ID="ls/${MERGED_BASENAME}"
fi
if [[ -z "${POLICY_REPO_ID}" ]]; then
  POLICY_REPO_ID="${DATASET_REPO_ID}"
fi
if [[ -z "${OUTPUT_DIR}" ]]; then
  OUTPUT_DIR="${PROJECT_ROOT}/outputs/${JOB_NAME}"
fi

mkdir -p "${LOG_DIR}" "${HF_HOME}" "${HF_DATASETS_CACHE}"
mkdir -p "$(dirname "${OUTPUT_DIR}")"

if [[ -d "${OUTPUT_DIR}" ]]; then
  if [[ -z "$(find "${OUTPUT_DIR}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    rmdir "${OUTPUT_DIR}"
  fi
fi

if [[ "${SKIP_MERGE}" == false ]]; then
  MERGE_CMD=(
    "${PYTHON_BIN}" "${MERGE_SCRIPT}"
    --dataset-a "${DATASET_A}"
    --dataset-b "${DATASET_B}"
    --output-dataset "${MERGED_ROOT}"
  )
  if [[ "${OVERWRITE}" == true ]]; then
    MERGE_CMD+=(--overwrite)
  fi
  if [[ -n "${STATE_LAYOUT}" ]]; then
    MERGE_CMD+=(--state-layout "${STATE_LAYOUT}")
  fi

  echo "[merge] ${MERGE_CMD[*]}"
  "${MERGE_CMD[@]}"
fi

if [[ "${SKIP_TRAIN}" == false ]]; then
  LOG_PATH="${LOG_DIR}/${JOB_NAME}.log"
  TRAIN_CMD=(
    "${LEROBOT_TRAIN_BIN}"
    --policy.type=act
    --policy.device=cuda
    --policy.push_to_hub=false
    --policy.repo_id="${POLICY_REPO_ID}"
    --dataset.repo_id="${DATASET_REPO_ID}"
    --dataset.root="${MERGED_ROOT}"
    --output_dir="${OUTPUT_DIR}"
    --job_name="${JOB_NAME}"
  )
  if [[ ${#EXTRA_TRAIN_ARGS[@]} -gt 0 ]]; then
    TRAIN_CMD+=("${EXTRA_TRAIN_ARGS[@]}")
  fi

  echo "[train] CUDA_VISIBLE_DEVICES=${CUDA_DEVICE}"
  echo "[train] log -> ${LOG_PATH}"

  if [[ "${RUN_IN_FOREGROUND}" == true ]]; then
    env \
      CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" \
      HF_HOME="${HF_HOME}" \
      HF_DATASETS_CACHE="${HF_DATASETS_CACHE}" \
      "${TRAIN_CMD[@]}" | tee "${LOG_PATH}"
  else
    nohup env \
      CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" \
      HF_HOME="${HF_HOME}" \
      HF_DATASETS_CACHE="${HF_DATASETS_CACHE}" \
      "${TRAIN_CMD[@]}" \
      > "${LOG_PATH}" 2>&1 &
    echo "[train] 已后台启动，PID=$!"
  fi
fi
