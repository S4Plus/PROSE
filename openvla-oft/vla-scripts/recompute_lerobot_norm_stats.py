#!/usr/bin/env python3
"""
Recompute LeRobot state / action normalization statistics from parquet data.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from prismatic.vla.lerobot_openvla import (
    LeRobotDatasetMetadata,
    recompute_norm_stats_for_indices,
    save_lerobot_norm_stats,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recompute LeRobot norm_stats.json from parquet data.")
    parser.add_argument("--dataset_root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--sample_indices_path", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = LeRobotDatasetMetadata(args.dataset_root, dataset_name=args.dataset_name)

    if args.sample_indices_path is not None:
        sample_indices = np.load(args.sample_indices_path).astype(np.int64).reshape(-1)
        recompute_norm_stats_for_indices(metadata, sample_indices, args.output)
    else:
        states = []
        actions = []
        for row in metadata.episodes:
            payload = metadata.load_episode_payload(int(row["episode_index"]))
            states.append(np.asarray(payload.states, dtype=np.float32))
            actions.append(np.asarray(payload.actions, dtype=np.float32))
        save_lerobot_norm_stats(
            args.output,
            np.concatenate(states, axis=0),
            np.concatenate(actions, axis=0),
        )

    print(f"Wrote corrected norm stats to: {args.output}")


if __name__ == "__main__":
    main()
