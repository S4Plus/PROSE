#!/usr/bin/env python3
"""Convert Demo-SCORE sample scores into OpenVLA sample indices and retrain."""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from prismatic.vla.lerobot_openvla import (
    LeRobotDatasetMetadata,
    OpenVLALeRobotConfig,
    finetune_openvla_lerobot,
    load_finetune_config,
    recompute_norm_stats_for_indices,
)


def _load_score_vector(path: Path, *, total_frames: int, dtype: str, key: str | None) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".mmap":
        score_dtype = np.dtype(dtype)
        total_values = path.stat().st_size // score_dtype.itemsize
        scores = np.memmap(path, mode="r", dtype=score_dtype, shape=(total_values,))
        vector = np.asarray(scores, dtype=np.float32).reshape(-1)
    elif suffix == ".npy":
        vector = np.asarray(np.load(path), dtype=np.float32).reshape(-1)
    elif suffix == ".npz":
        with np.load(path) as data:
            chosen_key = key or data.files[0]
            if chosen_key not in data:
                raise KeyError(f"{path} does not contain key {chosen_key!r}. Available keys: {sorted(data.files)}")
            vector = np.asarray(data[chosen_key], dtype=np.float32).reshape(-1)
    elif suffix == ".json":
        payload = json.loads(path.read_text())
        value = payload[key] if key else payload
        vector = np.asarray(value, dtype=np.float32).reshape(-1)
    elif suffix == ".csv":
        rows: list[float] = []
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if key is not None:
                if reader.fieldnames is None or key not in reader.fieldnames:
                    raise KeyError(f"{path} does not contain CSV column {key!r}.")
                for row in reader:
                    rows.append(float(row[key]))
            else:
                for row in reader:
                    if not row:
                        continue
                    rows.append(float(next(iter(row.values()))))
        vector = np.asarray(rows, dtype=np.float32).reshape(-1)
    else:
        raise ValueError(f"Unsupported score file type: {path}")

    if vector.shape[0] < total_frames:
        raise ValueError(f"Loaded {vector.shape[0]} scores, expected at least {total_frames}.")
    if vector.shape[0] > total_frames:
        vector = vector[:total_frames]
    return vector


def _episode_lengths(metadata: LeRobotDatasetMetadata) -> list[tuple[int, int]]:
    return [(int(row["episode_index"]), int(row["length"])) for row in metadata.episodes]


def _aggregate_episode_scores(
    sample_scores: np.ndarray,
    metadata: LeRobotDatasetMetadata,
    *,
    aggregation: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    aggregators = {
        "mean": np.mean,
        "sum": np.sum,
        "median": np.median,
        "min": np.min,
        "max": np.max,
    }
    if aggregation not in aggregators:
        raise ValueError(f"Unsupported aggregation {aggregation!r}. Choose from {sorted(aggregators)}.")
    aggr_fn = aggregators[aggregation]

    episode_ids: list[int] = []
    episode_lengths: list[int] = []
    episode_scores: list[float] = []
    cursor = 0
    for episode_id, length in _episode_lengths(metadata):
        next_cursor = cursor + length
        if next_cursor > sample_scores.shape[0]:
            raise ValueError("Sample score vector is shorter than dataset total frames.")
        chunk = sample_scores[cursor:next_cursor]
        episode_ids.append(episode_id)
        episode_lengths.append(length)
        episode_scores.append(float(aggr_fn(chunk)))
        cursor = next_cursor

    if cursor != sample_scores.shape[0]:
        raise ValueError("Sample score vector is longer than dataset total frames.")

    return (
        np.asarray(episode_ids, dtype=np.int64),
        np.asarray(episode_lengths, dtype=np.int64),
        np.asarray(episode_scores, dtype=np.float32),
    )


def _select_episodes(
    episode_scores: np.ndarray,
    *,
    keep_ratio: float,
    higher_is_better: bool,
) -> tuple[np.ndarray, np.ndarray]:
    num_episodes = int(episode_scores.shape[0])
    if not 0.0 < keep_ratio <= 1.0:
        raise ValueError(f"keep_ratio must be in (0, 1], got {keep_ratio}")
    num_keep = max(1, int(math.ceil(num_episodes * keep_ratio)))
    order = np.argsort(episode_scores)
    if higher_is_better:
        order = order[::-1]
    keep_order = order[:num_keep]
    keep_mask = np.zeros(num_episodes, dtype=bool)
    keep_mask[keep_order] = True
    return keep_mask, order


def _episode_mask_to_sample_indices(
    metadata: LeRobotDatasetMetadata,
    episode_keep_mask: np.ndarray,
) -> np.ndarray:
    kept_sample_indices: list[np.ndarray] = []
    for keep, row in zip(episode_keep_mask.tolist(), metadata.episodes, strict=True):
        if not keep:
            continue
        payload = metadata.load_episode_payload(int(row["episode_index"]))
        kept_sample_indices.append(np.asarray(payload.global_indices, dtype=np.int64).reshape(-1))
    if not kept_sample_indices:
        return np.empty((0,), dtype=np.int64)
    return np.unique(np.concatenate(kept_sample_indices).astype(np.int64, copy=False))


def _build_retrain_cfg(
    base_cfg: OpenVLALeRobotConfig,
    *,
    dataset_root: Path,
    dataset_name: str,
    output_indices: Path,
    subset_norm_stats_path: Path,
    retrain_run_root: Path,
    exp_name: str,
    num_train_steps: int | None,
) -> OpenVLALeRobotConfig:
    retrain_output_dir = retrain_run_root / exp_name
    return dataclasses.replace(
        base_cfg,
        dataset_root=dataset_root,
        dataset_name=dataset_name,
        output_dir=retrain_output_dir,
        run_root_dir=retrain_run_root,
        sample_indices_path=output_indices,
        norm_stats_path=subset_norm_stats_path,
        max_steps=num_train_steps or base_cfg.max_steps,
        run_id_note=exp_name,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Demo-SCORE train-set sample scores into OpenVLA sample_indices, recompute norm_stats, and retrain."
    )
    parser.add_argument("--score-path", type=Path, required=True)
    parser.add_argument("--score-dtype", type=str, default="float16")
    parser.add_argument("--score-key", type=str, default=None)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--dataset-name", type=str, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--exp-name", type=str, default=None)
    parser.add_argument("--retrain-run-root", type=Path, default=Path("runs"))
    parser.add_argument("--num-train-steps", type=int, default=None)
    parser.add_argument("--keep-ratio", type=float, default=0.9)
    parser.add_argument("--drop-ratio", type=float, default=None)
    parser.add_argument("--aggregation", type=str, default="mean")
    parser.add_argument("--lower-is-better", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_cfg = load_finetune_config(args.checkpoint_dir)
    dataset_root = args.dataset_root or base_cfg.dataset_root
    dataset_name = args.dataset_name or base_cfg.dataset_name

    metadata = LeRobotDatasetMetadata(
        dataset_root,
        dataset_name=dataset_name,
        task_description_override=base_cfg.task_description,
        norm_stats_path=base_cfg.norm_stats_path,
    )
    sample_scores = _load_score_vector(
        args.score_path,
        total_frames=metadata.total_frames,
        dtype=args.score_dtype,
        key=args.score_key,
    )

    keep_ratio = 1.0 - args.drop_ratio if args.drop_ratio is not None else args.keep_ratio
    episode_ids, episode_lengths, episode_scores = _aggregate_episode_scores(
        sample_scores,
        metadata,
        aggregation=args.aggregation,
    )
    episode_keep_mask, ranking = _select_episodes(
        episode_scores,
        keep_ratio=keep_ratio,
        higher_is_better=not args.lower_is_better,
    )
    kept_sample_indices = _episode_mask_to_sample_indices(metadata, episode_keep_mask)
    if kept_sample_indices.size == 0:
        raise ValueError("Demo-SCORE selection produced zero kept samples.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_indices = args.output_dir / "kept_sample_indices.npy"
    if output_indices.exists() and not args.overwrite:
        raise FileExistsError(f"{output_indices} already exists. Use --overwrite to replace it.")
    np.save(output_indices, kept_sample_indices)

    np.savez_compressed(
        args.output_dir / "demo_score_selection_artifacts.npz",
        episode_ids=episode_ids,
        episode_lengths=episode_lengths,
        episode_scores=episode_scores,
        episode_keep_mask=episode_keep_mask.astype(np.int64),
        ranking=ranking.astype(np.int64),
        kept_sample_indices=kept_sample_indices,
    )

    summary = {
        "method": "demo_score_episode_filter_to_openvla",
        "score_path": str(args.score_path),
        "aggregation": args.aggregation,
        "higher_is_better": not args.lower_is_better,
        "keep_ratio": float(keep_ratio),
        "total_episodes": int(episode_ids.shape[0]),
        "kept_episodes": int(np.count_nonzero(episode_keep_mask)),
        "dropped_episodes": int(np.count_nonzero(~episode_keep_mask)),
        "total_samples": int(metadata.total_frames),
        "kept_samples": int(kept_sample_indices.shape[0]),
        "dropped_samples": int(metadata.total_frames - kept_sample_indices.shape[0]),
        "score_min": float(np.min(episode_scores)),
        "score_max": float(np.max(episode_scores)),
        "score_mean": float(np.mean(episode_scores)),
    }
    (args.output_dir / "demo_score_selection_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )

    subset_norm_stats_path = args.output_dir / "norm_stats.json"
    recompute_norm_stats_for_indices(metadata, kept_sample_indices, subset_norm_stats_path)

    if args.prepare_only:
        return

    exp_name = args.exp_name or f"{dataset_name}_demo_score"
    retrain_cfg = _build_retrain_cfg(
        base_cfg,
        dataset_root=dataset_root,
        dataset_name=dataset_name,
        output_indices=output_indices,
        subset_norm_stats_path=subset_norm_stats_path,
        retrain_run_root=args.retrain_run_root,
        exp_name=exp_name,
        num_train_steps=args.num_train_steps,
    )
    finetune_openvla_lerobot(retrain_cfg)


if __name__ == "__main__":
    main()
