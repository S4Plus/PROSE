#!/usr/bin/env bash
set -euo pipefail

cd /workspace/liushuo/cupid

python /workspace/liushuo/cupid/scripts/build_openpi05_canph_ablation.py \
  --prototype-source pi05_keep \
  --overwrite

cd /workspace/liushuo/cupid/openpi

UV_CACHE_DIR="${UV_CACHE_DIR:-/workspace/liushuo/cupid/openpi/tmp/uv-cache}"
KEEP_LIST="/workspace/liushuo/cupid/openpi/openpi-script/output/joint_episode_window_serial/can_mh_joint_episode_window/selection/kept_episodes.txt"
SELECTION_NPZ="/workspace/liushuo/cupid/openpi/openpi-script/output/joint_episode_window_serial/can_mh_joint_episode_window/selection/joint_episode_window_selection.npz"

UV_CACHE_DIR="${UV_CACHE_DIR}" uv run python /workspace/liushuo/cupid/openpi/scripts/build_canph_openpi05_sample_indices.py \
  --dataset-dir /workspace/liushuo/cupid/output/openpi05_canph_ablation/datasets/can_mh_twostage_plus_can_ph_pref_k054 \
  --selection-npz "${SELECTION_NPZ}" \
  --can-mh-keep-list "${KEEP_LIST}" \
  --output-indices /workspace/liushuo/cupid/output/openpi05_canph_ablation/datasets/can_mh_twostage_plus_can_ph_pref_k054/openpi_sample_indices.npy \
  --output-summary /workspace/liushuo/cupid/output/openpi05_canph_ablation/datasets/can_mh_twostage_plus_can_ph_pref_k054/openpi_sample_indices_summary.json

UV_CACHE_DIR="${UV_CACHE_DIR}" uv run python /workspace/liushuo/cupid/openpi/scripts/build_canph_openpi05_sample_indices.py \
  --dataset-dir /workspace/liushuo/cupid/output/openpi05_canph_ablation/datasets/can_mh_twostage_plus_can_ph_random_k054_seed42 \
  --selection-npz "${SELECTION_NPZ}" \
  --can-mh-keep-list "${KEEP_LIST}" \
  --output-indices /workspace/liushuo/cupid/output/openpi05_canph_ablation/datasets/can_mh_twostage_plus_can_ph_random_k054_seed42/openpi_sample_indices.npy \
  --output-summary /workspace/liushuo/cupid/output/openpi05_canph_ablation/datasets/can_mh_twostage_plus_can_ph_random_k054_seed42/openpi_sample_indices_summary.json
