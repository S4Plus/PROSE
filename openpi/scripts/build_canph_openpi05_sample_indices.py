#!/usr/bin/env python3
"""为 can_mh 两阶段 window 过滤 + can_ph 补充数据构建 OpenPI sample_indices。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import tqdm.auto as tqdm

OPENPI_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = OPENPI_ROOT / "src"
if str(OPENPI_ROOT) not in sys.path:
    sys.path.insert(0, str(OPENPI_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from lerobot.common.datasets import lerobot_dataset
import openpi.training.config as _config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="构建 merged dataset 的 OpenPI sample_indices。")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--selection-npz", type=Path, required=True)
    parser.add_argument("--can-mh-keep-list", type=Path, required=True)
    parser.add_argument("--train-config", type=str, default="pi05_can_mh_lora")
    parser.add_argument("--output-indices", type=Path, required=True)
    parser.add_argument("--output-summary", type=Path, required=True)
    return parser.parse_args()


def load_keep_list(path: Path) -> list[int]:
    return [int(x.strip()) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not intervals:
        return []
    intervals = sorted(intervals)
    merged = [intervals[0]]
    for start, end in intervals[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def build_window_map(selection_npz: Path, keep_list: list[int]) -> dict[int, list[tuple[int, int]]]:
    keep_pos = {old_ep: new_ep for new_ep, old_ep in enumerate(keep_list)}
    window_map: dict[int, list[tuple[int, int]]] = {new_ep: [] for new_ep in range(len(keep_list))}

    with np.load(selection_npz) as data:
        window_episode = np.asarray(data["window_episode"], dtype=np.int64)
        window_start = np.asarray(data["window_start"], dtype=np.int64)
        window_end = np.asarray(data["window_end"], dtype=np.int64)
        window_keep = np.asarray(data["window_keep"], dtype=np.int64).astype(bool)

    for old_ep, start, end, keep in zip(
        window_episode.tolist(),
        window_start.tolist(),
        window_end.tolist(),
        window_keep.tolist(),
        strict=True,
    ):
        if not keep:
            continue
        if old_ep not in keep_pos:
            continue
        if end <= start:
            continue
        new_ep = keep_pos[old_ep]
        window_map[new_ep].append((int(start), int(end)))

    return {ep: merge_intervals(intervals) for ep, intervals in window_map.items()}


def in_intervals(index: int, intervals: list[tuple[int, int]]) -> bool:
    for start, end in intervals:
        if start <= index < end:
            return True
    return False


def main() -> None:
    args = parse_args()

    keep_list = load_keep_list(args.can_mh_keep_list)
    num_can_mh_episodes = len(keep_list)
    window_map = build_window_map(args.selection_npz, keep_list)

    train_cfg = _config.get_config(args.train_config)
    data_cfg = train_cfg.data.create(train_cfg.assets_dirs, train_cfg.model)

    dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(str(args.dataset_dir))
    dataset = lerobot_dataset.LeRobotDataset(
        str(args.dataset_dir),
        delta_timestamps={
            key: [t / dataset_meta.fps for t in range(train_cfg.model.action_horizon)]
            for key in data_cfg.action_sequence_keys
        },
    )
    print(
        f"[build_sample_indices] dataset={args.dataset_dir} "
        f"episodes={dataset_meta.total_episodes} samples={len(dataset)}"
    )

    local_positions: dict[int, int] = {}
    kept_indices: list[int] = []
    kept_can_mh = 0
    kept_can_ph = 0

    for sample_idx in tqdm.trange(len(dataset), desc="Building OpenPI sample indices", dynamic_ncols=True):
        item = dataset[sample_idx]
        episode_index = int(item["episode_index"])
        local_pos = local_positions.get(episode_index, 0)
        local_positions[episode_index] = local_pos + 1

        if episode_index < num_can_mh_episodes:
            if in_intervals(local_pos, window_map.get(episode_index, [])):
                kept_indices.append(sample_idx)
                kept_can_mh += 1
        else:
            kept_indices.append(sample_idx)
            kept_can_ph += 1

    args.output_indices.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary.parent.mkdir(parents=True, exist_ok=True)
    kept_indices_arr = np.asarray(kept_indices, dtype=np.int64)
    can_mh_indices_arr = kept_indices_arr[:kept_can_mh]
    can_ph_indices_arr = kept_indices_arr[kept_can_mh:]
    np.save(args.output_indices, kept_indices_arr)
    np.savez_compressed(
        args.output_indices.with_suffix(".parts.npz"),
        can_mh_sample_indices=can_mh_indices_arr,
        can_ph_sample_indices=can_ph_indices_arr,
    )

    summary = {
        "dataset_dir": str(args.dataset_dir),
        "train_config": args.train_config,
        "can_mh_keep_list": str(args.can_mh_keep_list),
        "selection_npz": str(args.selection_npz),
        "num_can_mh_episodes": num_can_mh_episodes,
        "num_total_episodes": int(dataset_meta.total_episodes),
        "dataset_num_samples": int(len(dataset)),
        "kept_total_samples": int(len(kept_indices)),
        "kept_can_mh_samples": int(kept_can_mh),
        "kept_can_ph_samples": int(kept_can_ph),
        "parts_npz": str(args.output_indices.with_suffix(".parts.npz")),
    }
    args.output_summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
