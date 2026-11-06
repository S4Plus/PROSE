#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

source "${SCRIPT_DIR}/setup_env.sh"

export PYTHONPATH="${REPO_ROOT}/lerobot_2.1/src:${PYTHONPATH:-}"

RAW_DIR="${RAW_DIR:-/ext_workspace/liushuo/stack_cube}"
DATASET_ROOT="${DATASET_ROOT:-/workspace/liushuo/dataset/stack_cube_102}"
DATASET_REPO_ID="${DATASET_REPO_ID:-local/stack_cube_102}"
DATASET_NAME="${DATASET_NAME:-$(basename "${DATASET_ROOT}")}"
TASK_TEXT="${TASK_TEXT:-stack cube}"

ACT_GPU="${ACT_GPU:-5}"
OPENPI_NORM_GPU="${OPENPI_NORM_GPU:-1}"
OPENPI_TRAIN_GPUS="${OPENPI_TRAIN_GPUS:-0,1}"

ACT_STEPS="${ACT_STEPS:-300000}"
OPENPI_STEPS="${OPENPI_STEPS:-30000}"

ACT_POLICY_REPO_ID="${ACT_POLICY_REPO_ID:-local/stack_cube_102_act}"
ACT_OUTPUT_DIR="${ACT_OUTPUT_DIR:-/workspace/liushuo/cupid/outputs/stack_cube_102_act}"
ACT_JOB_NAME="${ACT_JOB_NAME:-stack_cube_102_act}"

OPENPI_CONFIG_NAME="${OPENPI_CONFIG_NAME:-pi05_stack_cube_franka_lora}"
OPENPI_EXP_NAME="${OPENPI_EXP_NAME:-stack_cube_102_pi05_30k}"
OPENPI_ASSETS_DIR="${OPENPI_ASSETS_DIR:-/workspace/liushuo/dataset}"
OPENPI_ASSET_ID="${OPENPI_ASSET_ID:-stack_cube_102}"
OPENPI_CHECKPOINT_BASE_DIR="${OPENPI_CHECKPOINT_BASE_DIR:-/workspace/liushuo/cupid/openpi/checkpoints}"
OPENPI_NORM_MAX_FRAMES="${OPENPI_NORM_MAX_FRAMES:-}"

ACT_LOG_DIR="${ACT_LOG_DIR:-/workspace/liushuo/cupid/act/logs}"
OPENPI_LOG_DIR="${OPENPI_LOG_DIR:-/workspace/liushuo/cupid/openpi/logs}"
HF_CACHE_ROOT="${HF_CACHE_ROOT:-${REPO_ROOT}/.cache/huggingface}"
HF_HOME="${HF_HOME:-${HF_CACHE_ROOT}/home}"
HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_CACHE_ROOT}/datasets}"

mkdir -p "${ACT_LOG_DIR}" "${OPENPI_LOG_DIR}" "${HF_HOME}" "${HF_DATASETS_CACHE}" "${OPENPI_CHECKPOINT_BASE_DIR}"

if [[ -z "${OPENPI_FSDP_DEVICES:-}" ]]; then
  IFS=',' read -r -a openpi_gpu_array <<< "${OPENPI_TRAIN_GPUS}"
  OPENPI_FSDP_DEVICES="${#openpi_gpu_array[@]}"
fi

timestamp="$(date +%Y%m%d_%H%M%S)"
act_log="${ACT_LOG_DIR}/${ACT_JOB_NAME}_${timestamp}.log"
openpi_log="${OPENPI_LOG_DIR}/${OPENPI_EXP_NAME}_${timestamp}.log"
convert_log="${ACT_LOG_DIR}/convert_${DATASET_NAME}_${timestamp}.log"
norm_log="${OPENPI_LOG_DIR}/${OPENPI_EXP_NAME}_norm_${timestamp}.log"
act_pid=""

# echo "[1/4] 开始转换数据集"
# echo "RAW_DIR=${RAW_DIR}"
# echo "DATASET_ROOT=${DATASET_ROOT}"
# echo "DATASET_REPO_ID=${DATASET_REPO_ID}"

# python "${SCRIPT_DIR}/convert_stack_cube_hdf5_to_lerobot.py" \
#   --raw-dir "${RAW_DIR}" \
#   --repo-id "${DATASET_REPO_ID}" \
#   --lerobot-home "${DATASET_ROOT}" \
#   --task "${TASK_TEXT}" \
#   --overwrite \
#   2>&1 | tee "${convert_log}"

# echo "[2/4] 启动 ACT 训练"
# nohup env \
#   CUDA_VISIBLE_DEVICES="${ACT_GPU}" \
#   HF_HOME="${HF_HOME}" \
#   HF_DATASETS_CACHE="${HF_DATASETS_CACHE}" \
#   PYTHONPATH="${PYTHONPATH}" \
#   lerobot-train \
#   --policy.type=act \
#   --policy.device=cuda \
#   --policy.push_to_hub=false \
#   --policy.repo_id="${ACT_POLICY_REPO_ID}" \
#   --dataset.repo_id="${DATASET_REPO_ID}" \
#   --dataset.root="${DATASET_ROOT}" \
#   --output_dir="${ACT_OUTPUT_DIR}" \
#   --job_name="${ACT_JOB_NAME}" \
#   --steps="${ACT_STEPS}" \
#   > "${act_log}" 2>&1 &
# act_pid=$!

echo "[3/4] 计算 OpenPI05 norm_stats"
(
  cd /workspace/liushuo/cupid/openpi
  CUDA_VISIBLE_DEVICES="${OPENPI_NORM_GPU}" \
  WANDB_MODE=disabled \
  OPENPI_CONFIG_NAME="${OPENPI_CONFIG_NAME}" \
  OPENPI_DATASET_ROOT="${DATASET_ROOT}" \
  OPENPI_ASSETS_DIR="${OPENPI_ASSETS_DIR}" \
  OPENPI_ASSET_ID="${OPENPI_ASSET_ID}" \
  OPENPI_NORM_MAX_FRAMES="${OPENPI_NORM_MAX_FRAMES}" \
  uv run python - <<'PY'
import dataclasses
import os
from pathlib import Path

import numpy as np
import tqdm

import openpi.shared.normalize as normalize
import openpi.training.config as training_config
import scripts.compute_norm_stats as compute_norm_stats


def main() -> None:
    config_name = os.environ["OPENPI_CONFIG_NAME"]
    dataset_root = os.environ["OPENPI_DATASET_ROOT"]
    assets_dir = os.environ["OPENPI_ASSETS_DIR"]
    asset_id = os.environ["OPENPI_ASSET_ID"]
    max_frames_raw = os.environ.get("OPENPI_NORM_MAX_FRAMES", "").strip()
    max_frames = int(max_frames_raw) if max_frames_raw else None

    config = training_config.get_config(config_name)
    if config.data.assets is None:
        raise ValueError(f"{config_name} 缺少 assets 配置，无法为自定义数据集计算 norm_stats。")

    config = dataclasses.replace(
        config,
        data=dataclasses.replace(
            config.data,
            repo_id=dataset_root,
            assets=dataclasses.replace(
                config.data.assets,
                assets_dir=assets_dir,
                asset_id=asset_id,
            ),
        ),
    )

    data_config = config.data.create(config.assets_dirs, config.model)
    if data_config.rlds_data_dir is not None:
        data_loader, num_batches = compute_norm_stats.create_rlds_dataloader(
            data_config,
            config.model.action_horizon,
            config.batch_size,
            max_frames,
        )
    else:
        data_loader, num_batches = compute_norm_stats.create_torch_dataloader(
            data_config,
            config.model.action_horizon,
            config.batch_size,
            config.model,
            config.num_workers,
            max_frames,
        )

    stats = {key: normalize.RunningStats() for key in ("state", "actions")}
    for batch in tqdm.tqdm(data_loader, total=num_batches, desc="Computing stats"):
        for key in stats:
            stats[key].update(np.asarray(batch[key]))

    output_path = Path(dataset_root)
    print(f"Writing stats to: {output_path}")
    normalize.save(output_path, {key: value.get_statistics() for key, value in stats.items()})


if __name__ == "__main__":
    main()
PY
) 2>&1 | tee "${norm_log}"

echo "[4/4] 启动 OpenPI05 训练"
nohup bash -lc "
cd /workspace/liushuo/cupid/openpi
CUDA_VISIBLE_DEVICES='${OPENPI_TRAIN_GPUS}' \
WANDB_MODE=disabled \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run scripts/train.py '${OPENPI_CONFIG_NAME}' \
  --exp-name '${OPENPI_EXP_NAME}' \
  --data.repo-id '${DATASET_ROOT}' \
  --data.assets.assets-dir '${OPENPI_ASSETS_DIR}' \
  --data.assets.asset-id '${OPENPI_ASSET_ID}' \
  --checkpoint-base-dir '${OPENPI_CHECKPOINT_BASE_DIR}' \
  --fsdp-devices '${OPENPI_FSDP_DEVICES}' \
  --num-train-steps '${OPENPI_STEPS}' \
  --overwrite
" > "${openpi_log}" 2>&1 &
openpi_pid=$!

echo "完成："
echo "  转换日志: ${convert_log}"
echo "  ACT 日志: ${act_log}"
echo "  OpenPI norm_stats 日志: ${norm_log}"
echo "  OpenPI 训练日志: ${openpi_log}"
echo "  ACT PID: ${act_pid:-not_started}"
echo "  OpenPI PID: ${openpi_pid}"
