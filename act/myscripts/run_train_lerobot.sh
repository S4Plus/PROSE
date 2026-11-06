# 用 LeRobot 数据集训练 transport_mh 的 ACT 模型。
# 关键兼容项：
# 1. 显式指定 HF 缓存目录，避免默认落到只读路径导致 parquet lock 失败。
# 2. 关闭 push_to_hub，避免本地训练时误触发 Hub 上传逻辑。
# 3. 预创建日志目录，避免重定向失败。

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOG_DIR="${LOG_DIR:-/workspace/liushuo/cupid/act/logs}"
HF_CACHE_ROOT="${HF_CACHE_ROOT:-${PROJECT_ROOT}/.cache/huggingface}"
HF_HOME="${HF_HOME:-${HF_CACHE_ROOT}/home}"
HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_CACHE_ROOT}/datasets}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"

mkdir -p "${LOG_DIR}" "${HF_HOME}" "${HF_DATASETS_CACHE}"

nohup env \
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
  HF_HOME="${HF_HOME}" \
  HF_DATASETS_CACHE="${HF_DATASETS_CACHE}" \
  lerobot-train \
  --policy.type=act \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --policy.repo_id=ls/transport_mh \
  --dataset.repo_id=ls/transport_mh \
  --dataset.root=/workspace/liushuo/cupid/datasets/lerobot/transport_mh \
  --output_dir=/workspace/liushuo/cupid/outputs/transport_mh_full_act \
  --job_name=transport_mh_full_act \
  > "${LOG_DIR}/transport_mh_full_act.log" 2>&1 &
