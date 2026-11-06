#!/usr/bin/env bash
set -euo pipefail

cd /workspace/liushuo/cupid/openpi

DEFAULT_DATASET_DIR="/workspace/liushuo/cupid/output/openpi05_canph_ablation/datasets/can_mh_twostage_plus_can_ph_random_k054_seed42"
DEFAULT_TRAIN_CONFIG="pi05_can_mh_lora"
DEFAULT_SAMPLE_INDICES_PATH="${DEFAULT_DATASET_DIR}/openpi_sample_indices.npy"

SAMPLE_INDICES_PATH="${SAMPLE_INDICES_PATH:-${DEFAULT_SAMPLE_INDICES_PATH}}"

if [[ -n "${DATASET_DIR:-}" ]]; then
  DATASET_DIR="${DATASET_DIR}"
elif [[ -f "${SAMPLE_INDICES_PATH}" && -f "$(dirname "${SAMPLE_INDICES_PATH}")/meta/info.json" ]]; then
  DATASET_DIR="$(dirname "${SAMPLE_INDICES_PATH}")"
else
  DATASET_DIR="${DEFAULT_DATASET_DIR}"
fi

if [[ -n "${TRAIN_CONFIG:-}" ]]; then
  TRAIN_CONFIG="${TRAIN_CONFIG}"
elif [[ "${DATASET_DIR}" != "${DEFAULT_DATASET_DIR}" ]]; then
  echo "[error] 未显式设置 TRAIN_CONFIG，但根据 SAMPLE_INDICES_PATH 推断出的 DATASET_DIR 为非默认数据集：" >&2
  echo "        DATASET_DIR=${DATASET_DIR}" >&2
  echo "        SAMPLE_INDICES_PATH=${SAMPLE_INDICES_PATH}" >&2
  echo "        请显式传入对应的 TRAIN_CONFIG，例如 transport 数据集使用 pi05_transport_mh_lora。" >&2
  exit 2
else
  TRAIN_CONFIG="${DEFAULT_TRAIN_CONFIG}"
fi

echo "[config] DATASET_DIR=${DATASET_DIR}"
echo "[config] TRAIN_CONFIG=${TRAIN_CONFIG}"
echo "[config] SAMPLE_INDICES_PATH=${SAMPLE_INDICES_PATH}"
UV_CACHE_DIR="${UV_CACHE_DIR:-/workspace/liushuo/cupid/openpi/tmp/uv-cache}"

DATASET_DIR="${DATASET_DIR}" TRAIN_CONFIG="${TRAIN_CONFIG}" SAMPLE_INDICES_PATH="${SAMPLE_INDICES_PATH}" \
UV_CACHE_DIR="${UV_CACHE_DIR}" uv run python - <<'PY'
import dataclasses
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/workspace/liushuo/cupid/openpi")
sys.path.insert(0, "/workspace/liushuo/cupid/openpi/src")

import openpi.training.config as _config
import retrain_window_joint

dataset_dir = Path(os.environ["DATASET_DIR"])
train_config = os.environ["TRAIN_CONFIG"]
sample_indices_path = Path(os.environ["SAMPLE_INDICES_PATH"])

cfg = _config.get_config(train_config)
cfg = dataclasses.replace(
    cfg,
    data=dataclasses.replace(
        cfg.data,
        repo_id=str(dataset_dir),
        sample_indices_path=str(sample_indices_path),
        assets=_config.AssetsConfig(
            assets_dir=str(dataset_dir.parent),
            asset_id=dataset_dir.name,
        ),
    ),
)

meta_info_path = dataset_dir / "meta" / "info.json"
if not meta_info_path.exists():
    raise FileNotFoundError(f"Dataset metadata not found: {meta_info_path}")
if not sample_indices_path.exists():
    raise FileNotFoundError(f"Sample indices file not found: {sample_indices_path}")

meta_info = json.loads(meta_info_path.read_text())
dataset_total_frames = int(meta_info["total_frames"])
sample_indices = np.load(sample_indices_path).astype(np.int64).reshape(-1)
if sample_indices.size == 0:
    raise ValueError(f"Sample indices file is empty: {sample_indices_path}")

sample_parent = sample_indices_path.parent.resolve()
dataset_dir_resolved = dataset_dir.resolve()
if sample_parent != dataset_dir_resolved:
    print(
        "[warn] SAMPLE_INDICES_PATH 不在 DATASET_DIR 目录内，请确认两者属于同一数据集："
        f" dataset_dir={dataset_dir_resolved}, sample_parent={sample_parent}"
    )

max_sample_index = int(sample_indices.max())
if max_sample_index >= dataset_total_frames:
    raise ValueError(
        "sample_indices 与 DATASET_DIR 不匹配："
        f" max_sample_index={max_sample_index}, dataset_total_frames={dataset_total_frames}, "
        f"dataset_dir={dataset_dir_resolved}, sample_indices_path={sample_indices_path}. "
        "通常是因为命令行里的环境变量续行写法有空行，导致 DATASET_DIR 或 TRAIN_CONFIG 没有真正传进脚本。"
    )

print(
    "[preflight] dataset_total_frames="
    f"{dataset_total_frames}, kept_samples={sample_indices.shape[0]}, max_sample_index={max_sample_index}"
)

try:
    retrain_window_joint._recompute_norm_stats_for_sample_subset(
        cfg,
        sample_indices_path=sample_indices_path,
        output_dir=dataset_dir,
        max_frames=None,
    )
except PermissionError:
    cfg = dataclasses.replace(cfg, num_workers=0)
    retrain_window_joint._recompute_norm_stats_for_sample_subset(
        cfg,
        sample_indices_path=sample_indices_path,
        output_dir=dataset_dir,
        max_frames=None,
    )

print(f"[done] norm_stats -> {dataset_dir} with {sample_indices_path}")
PY
