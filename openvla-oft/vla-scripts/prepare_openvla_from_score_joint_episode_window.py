#!/usr/bin/env python3
"""Convert score joint episode-window selection into OpenVLA sample indices and retrain."""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
from typing import Any

import numpy as np

from prismatic.vla.lerobot_openvla import (
    LeRobotDatasetMetadata,
    finetune_openvla_lerobot,
    load_finetune_config,
    recompute_norm_stats_for_indices,
)


def _load_selection(selection_npz: Path) -> dict[str, np.ndarray]:
    with np.load(selection_npz) as data:
        required = ("episode_keep_mask", "window_episode", "window_start", "window_end", "window_keep")
        missing = [key for key in required if key not in data]
        if missing:
            raise ValueError(f"{selection_npz} is missing required keys: {missing}")
        return {key: np.asarray(data[key]) for key in data.files}


def _materialize_kept_sample_indices(
    selection: dict[str, np.ndarray],
    metadata: LeRobotDatasetMetadata,
) -> tuple[np.ndarray, dict[str, int]]:
    window_episode = np.asarray(selection["window_episode"], dtype=np.int64).reshape(-1)
    window_start = np.asarray(selection["window_start"], dtype=np.int64).reshape(-1)
    window_end = np.asarray(selection["window_end"], dtype=np.int64).reshape(-1)
    window_keep = np.asarray(selection["window_keep"], dtype=np.int64).reshape(-1).astype(bool)

    if not (window_episode.shape[0] == window_start.shape[0] == window_end.shape[0] == window_keep.shape[0]):
        raise ValueError("Selection NPZ window arrays have inconsistent lengths.")

    kept_sample_indices: list[np.ndarray] = []
    kept_windows = 0
    covered_episodes: set[int] = set()
    episode_cache: dict[int, np.ndarray] = {}
    for episode, start, end, keep in zip(
        window_episode.tolist(),
        window_start.tolist(),
        window_end.tolist(),
        window_keep.tolist(),
        strict=True,
    ):
        if not keep:
            continue
        episode = int(episode)
        start = int(start)
        end = int(end)
        if end <= start:
            continue
        if episode not in episode_cache:
            payload = metadata.load_episode_payload(episode)
            episode_cache[episode] = np.asarray(payload.global_indices, dtype=np.int64).reshape(-1)
        global_indices = episode_cache[episode]
        if start >= global_indices.shape[0]:
            continue
        slice_end = min(end, global_indices.shape[0])
        indices = global_indices[start:slice_end]
        if indices.size == 0:
            continue
        kept_sample_indices.append(indices)
        kept_windows += 1
        covered_episodes.add(episode)

    if kept_sample_indices:
        unique_indices = np.unique(np.concatenate(kept_sample_indices).astype(np.int64, copy=False))
    else:
        unique_indices = np.empty((0,), dtype=np.int64)

    return unique_indices, {
        "kept_windows_with_samples": int(kept_windows),
        "covered_episodes_from_windows": int(len(covered_episodes)),
    }


def _write_summary(
    output_dir: Path,
    *,
    args: argparse.Namespace,
    selection: dict[str, np.ndarray],
    metadata: LeRobotDatasetMetadata,
    kept_sample_indices: np.ndarray,
    mapping_stats: dict[str, int],
) -> None:
    summary = {
        "method": "score_joint_episode_window_to_openvla",
        "score_pkl": str(args.score_pkl),
        "selection_npz": str(args.selection_npz),
        "checkpoint_dir": str(args.checkpoint_dir),
        "dataset_root": str(metadata.dataset_root),
        "dataset_name": metadata.dataset_name,
        "kept_samples": int(kept_sample_indices.shape[0]),
        "kept_windows": int(np.count_nonzero(np.asarray(selection["window_keep"], dtype=np.int64))),
        "num_windows": int(np.asarray(selection["window_keep"]).shape[0]),
        "kept_episodes": int(np.count_nonzero(np.asarray(selection["episode_keep_mask"], dtype=np.int64))),
        "num_episodes": int(np.asarray(selection["episode_keep_mask"]).shape[0]),
        **mapping_stats,
    }
    (output_dir / "openvla_selection_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert score joint episode-window selection into OpenVLA sample indices and retrain."
    )
    parser.add_argument("--score-pkl", type=Path, required=True, help="Influence score pickle used to produce the selection.")
    parser.add_argument(
        "--selection-npz",
        type=Path,
        required=True,
        help="Selection NPZ produced by vla-scripts/filter_joint_episode_window.py.",
    )
    parser.add_argument("--checkpoint-dir", type=Path, required=True, help="Existing OpenVLA finetune run directory.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory for indices, summary, and norm_stats.")
    parser.add_argument("--output-indices", type=Path, default=None, help="Optional explicit kept_sample_indices.npy path.")
    parser.add_argument("--exp-name", type=str, default=None, help="Experiment name for retraining.")
    parser.add_argument("--retrain-run-root", type=Path, default=Path("runs"))
    parser.add_argument("--num-train-steps", type=int, default=None)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_cfg = load_finetune_config(args.checkpoint_dir)
    metadata = LeRobotDatasetMetadata(
        base_cfg.dataset_root,
        dataset_name=base_cfg.dataset_name,
        task_description_override=base_cfg.task_description,
        norm_stats_path=base_cfg.norm_stats_path,
    )
    selection = _load_selection(args.selection_npz)
    kept_sample_indices, mapping_stats = _materialize_kept_sample_indices(selection, metadata)
    if kept_sample_indices.size == 0:
        raise ValueError("Selection produced zero OpenVLA samples after window-to-sample mapping.")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    output_indices = args.output_indices or (output_dir / "kept_sample_indices.npy")
    if output_indices.exists() and not args.overwrite:
        raise FileExistsError(f"{output_indices} already exists. Use --overwrite to replace it.")

    np.save(output_indices, kept_sample_indices)
    np.savez_compressed(
        output_dir / "joint_episode_window_selection_openvla.npz",
        kept_sample_indices=kept_sample_indices,
    )
    _write_summary(
        output_dir,
        args=args,
        selection=selection,
        metadata=metadata,
        kept_sample_indices=kept_sample_indices,
        mapping_stats=mapping_stats,
    )

    subset_norm_stats_path = output_dir / "norm_stats.json"
    print(f"Selection file: {args.selection_npz}")
    print(f"Checkpoint dir: {args.checkpoint_dir}")
    print(f"Dataset: {metadata.dataset_root}")
    print(f"Kept samples: {kept_sample_indices.shape[0]}")
    print(f"Output dir: {output_dir}")
    print(f"Sample indices: {output_indices}")
    print("Recomputing norm_stats.json for selected sample subset...")
    recompute_norm_stats_for_indices(metadata, kept_sample_indices, subset_norm_stats_path)
    print(f"Subset norm_stats written to: {subset_norm_stats_path}")

    if args.prepare_only:
        return

    exp_name = args.exp_name or f"{base_cfg.dataset_name}_score_joint_episode_window"
    retrain_output_dir = args.retrain_run_root / exp_name
    retrain_cfg = dataclasses.replace(
        base_cfg,
        output_dir=retrain_output_dir,
        run_root_dir=args.retrain_run_root,
        sample_indices_path=output_indices,
        norm_stats_path=subset_norm_stats_path,
        max_steps=args.num_train_steps or base_cfg.max_steps,
        run_id_note=exp_name,
    )
    finetune_openvla_lerobot(retrain_cfg)


if __name__ == "__main__":
    main()
