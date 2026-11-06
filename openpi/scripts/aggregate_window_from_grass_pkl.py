#!/usr/bin/env python3
"""Aggregate an existing sample-level GraSS pickle into window-level scores.

This is a pure post-processing script:
- It does not recompute GraSS gradients or HVPs.
- It does not use `.grass_cache`.
- It reads `sample_influence_scores` from an existing score pickle, reconstructs
  sample metadata from the train dataset, and emits a window-style pickle.

Typical use:
  uv run python scripts/aggregate_window_from_grass_pkl.py \
    --score-pkl /workspace/liushuo/cupid/openpi/tmp/can_mh_grass_last4_strict/influence_grass.pkl \
    --output-path /workspace/liushuo/cupid/openpi/tmp/can_mh_grass_last4_strict/influence_grass_window.pkl \
    --window-size 50
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import retrain as _retrain

LOGGER = logging.getLogger("aggregate_window_from_grass_pkl")


@dataclass
class AggregateWindowConfig:
    score_pkl: Path
    output_path: Path
    train_config: str | None = None
    dataset_dir: Path | None = None
    window_size: int = 50
    window_stride: int | None = None
    window_aggregation: str = "sum"
    quality_score: str = "zscore_mean"
    overwrite: bool = False


def _to_scalar_int(value: Any) -> int:
    return int(np.asarray(value).reshape(()))


def _zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return values
    mean = float(np.mean(values))
    std = float(np.std(values))
    if std < 1e-12:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - mean) / std).astype(np.float32)


def _reduce(values: np.ndarray, mode: str) -> float:
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        raise ValueError("Cannot reduce an empty window.")
    if mode == "sum":
        return float(np.sum(values))
    if mode == "mean":
        return float(np.mean(values))
    raise ValueError(f"Unsupported aggregation mode: {mode}")


def _infer_train_config_name(payload: dict[str, Any], override: str | None) -> str:
    config_name = override or _retrain._extract_score_config_name(payload)  # noqa: SLF001
    if config_name is None:
        raise ValueError("--train-config is required because the score pickle does not include config.config_name.")
    return config_name


def _load_sample_scores(payload: dict[str, Any], score_pkl: Path) -> np.ndarray:
    if "sample_influence_scores" not in payload:
        raise ValueError(f"{score_pkl} does not contain sample_influence_scores.")
    sample_scores = np.asarray(payload["sample_influence_scores"], dtype=np.float32).reshape(-1)
    if sample_scores.size == 0:
        raise ValueError(f"{score_pkl} contains empty sample_influence_scores.")
    return sample_scores


def _resolve_train_cfg(train_config_name: str, dataset_dir: Path | None):
    train_cfg = _config.get_config(train_config_name)
    if dataset_dir is not None:
        train_cfg = dataclasses.replace(
            train_cfg,
            data=dataclasses.replace(train_cfg.data, repo_id=str(dataset_dir)),
        )
    return train_cfg


def _load_sample_metadata_from_payload(payload: dict[str, Any], total_samples: int) -> dict[str, np.ndarray | int] | None:
    meta = payload.get("metadata", {})
    if not isinstance(meta, dict):
        return None
    required = ("train_sample_episode_indices", "train_dataset_sample_indices")
    if any(key not in meta for key in required):
        return None

    episode_indices = np.asarray(meta["train_sample_episode_indices"], dtype=np.int64).reshape(-1)
    dataset_sample_indices = np.asarray(meta["train_dataset_sample_indices"], dtype=np.int64).reshape(-1)
    if episode_indices.shape[0] != total_samples or dataset_sample_indices.shape[0] != total_samples:
        raise ValueError("Sample metadata shape mismatch in score pickle.")

    if "train_dataset_positions" in meta:
        dataset_positions = np.asarray(meta["train_dataset_positions"], dtype=np.int64).reshape(-1)
        if dataset_positions.shape[0] != total_samples:
            raise ValueError("train_dataset_positions shape mismatch in score pickle.")
    else:
        dataset_positions = np.arange(total_samples, dtype=np.int64)

    # The existing GraSS pickle stores samples in stable dataset order.
    # Recover per-episode frame indices by counting occurrences within each episode.
    frame_indices = np.empty((total_samples,), dtype=np.int64)
    counters: dict[int, int] = {}
    for i, episode_idx in enumerate(episode_indices.tolist()):
        frame_indices[i] = counters.get(episode_idx, 0)
        counters[episode_idx] = int(frame_indices[i] + 1)

    total_dataset_samples = int(meta.get("total_dataset_samples", int(np.max(dataset_positions)) + 1))
    return {
        "dataset_positions": dataset_positions,
        "dataset_sample_indices": dataset_sample_indices,
        "episode_indices": episode_indices,
        "frame_indices": frame_indices,
        "total_dataset_samples": total_dataset_samples,
    }


def _load_dataset_sample_metadata(train_cfg, payload: dict[str, Any], total_samples: int) -> dict[str, np.ndarray | int]:
    data_cfg = train_cfg.data.create(train_cfg.assets_dirs, train_cfg.model)
    if data_cfg.rlds_data_dir is not None:
        raise NotImplementedError("Window aggregation is not supported for RLDS datasets.")

    dataset = _data_loader.create_torch_dataset(
        data_cfg,
        train_cfg.model.action_horizon,
        train_cfg.model,
    )
    total_dataset_samples = int(len(dataset))

    meta = payload.get("metadata", {})
    if isinstance(meta, dict) and "train_dataset_positions" in meta:
        dataset_positions = np.asarray(meta["train_dataset_positions"], dtype=np.int64).reshape(-1)
    else:
        dataset_positions = np.arange(total_samples, dtype=np.int64)

    if dataset_positions.shape[0] != total_samples:
        raise ValueError(
            f"dataset position count {dataset_positions.shape[0]} does not match sample score count {total_samples}."
        )
    if np.any(dataset_positions < 0) or np.any(dataset_positions >= total_dataset_samples):
        raise ValueError("dataset positions in score pickle are out of range for the resolved dataset.")

    dataset_sample_indices = np.empty((total_samples,), dtype=np.int64)
    episode_indices = np.empty((total_samples,), dtype=np.int64)
    frame_indices = np.empty((total_samples,), dtype=np.int64)

    for i, dataset_pos in enumerate(dataset_positions.tolist()):
        item = dataset[int(dataset_pos)]
        if "episode_index" not in item:
            raise KeyError("Train dataset item does not contain episode_index.")
        if "frame_index" not in item:
            raise KeyError("Train dataset item does not contain frame_index.")
        episode_indices[i] = _to_scalar_int(item["episode_index"])
        frame_indices[i] = _to_scalar_int(item["frame_index"])
        if "index" in item:
            dataset_sample_indices[i] = _to_scalar_int(item["index"])
        else:
            dataset_sample_indices[i] = int(dataset_pos)

    return {
        "dataset_positions": dataset_positions,
        "dataset_sample_indices": dataset_sample_indices,
        "episode_indices": episode_indices,
        "frame_indices": frame_indices,
        "total_dataset_samples": total_dataset_samples,
    }


def _aggregate_windows(
    sample_scores: np.ndarray,
    sample_meta: dict[str, np.ndarray | int],
    *,
    window_size: int,
    window_stride: int,
    window_aggregation: str,
    quality_score: str,
) -> dict[str, np.ndarray]:
    episode_indices = np.asarray(sample_meta["episode_indices"], dtype=np.int64)
    frame_indices = np.asarray(sample_meta["frame_indices"], dtype=np.int64)
    dataset_sample_indices = np.asarray(sample_meta["dataset_sample_indices"], dtype=np.int64)
    dataset_positions = np.asarray(sample_meta["dataset_positions"], dtype=np.int64)

    sample_window_indices = np.full(sample_scores.shape[0], -1, dtype=np.int64)
    sample_window_local_indices = np.full(sample_scores.shape[0], -1, dtype=np.int64)

    window_episode_indices: list[int] = []
    window_local_indices: list[int] = []
    window_frame_starts: list[int] = []
    window_frame_ends: list[int] = []
    window_sample_counts: list[int] = []
    window_is_partial: list[bool] = []
    window_influence_scores: list[float] = []
    window_sum_scores: list[float] = []
    window_mean_scores: list[float] = []

    global_window_idx = 0
    for episode_idx in sorted(np.unique(episode_indices).tolist()):
        episode_sample_ids = np.where(episode_indices == episode_idx)[0]
        order = np.argsort(frame_indices[episode_sample_ids], kind="stable")
        ordered_sample_ids = episode_sample_ids[order]

        local_window_idx = 0
        for start in range(0, len(ordered_sample_ids), window_stride):
            end = min(start + window_size, len(ordered_sample_ids))
            if end <= start:
                continue
            current_sample_ids = ordered_sample_ids[start:end]
            current_scores = np.asarray(sample_scores[current_sample_ids], dtype=np.float32)
            current_frames = frame_indices[current_sample_ids]

            sample_window_indices[current_sample_ids] = global_window_idx
            sample_window_local_indices[current_sample_ids] = local_window_idx

            window_episode_indices.append(int(episode_idx))
            window_local_indices.append(local_window_idx)
            window_frame_starts.append(int(np.min(current_frames)))
            window_frame_ends.append(int(np.max(current_frames)))
            window_sample_counts.append(int(current_sample_ids.size))
            window_is_partial.append(int(current_sample_ids.size) < window_size)
            window_sum_scores.append(float(np.sum(current_scores)))
            window_mean_scores.append(float(np.mean(current_scores)))
            window_influence_scores.append(_reduce(current_scores, window_aggregation))

            global_window_idx += 1
            local_window_idx += 1

    window_sum_scores_np = np.asarray(window_sum_scores, dtype=np.float32)
    window_mean_scores_np = np.asarray(window_mean_scores, dtype=np.float32)
    if quality_score == "mean":
        window_quality_scores = window_mean_scores_np
    elif quality_score == "zscore_mean":
        window_quality_scores = _zscore(window_mean_scores_np)
    elif quality_score == "sum":
        window_quality_scores = window_sum_scores_np
    elif quality_score == "zscore_sum":
        window_quality_scores = _zscore(window_sum_scores_np)
    else:
        raise ValueError(f"Unsupported quality score mode: {quality_score}")

    return {
        "window_indices": np.arange(len(window_episode_indices), dtype=np.int64),
        "window_episode_indices": np.asarray(window_episode_indices, dtype=np.int64),
        "window_local_indices": np.asarray(window_local_indices, dtype=np.int64),
        "window_frame_starts": np.asarray(window_frame_starts, dtype=np.int64),
        "window_frame_ends": np.asarray(window_frame_ends, dtype=np.int64),
        "window_sample_counts": np.asarray(window_sample_counts, dtype=np.int64),
        "window_is_partial": np.asarray(window_is_partial, dtype=np.bool_),
        "window_influence_scores": np.asarray(window_influence_scores, dtype=np.float32),
        "window_sum_influence_scores": window_sum_scores_np,
        "window_mean_influence_scores": window_mean_scores_np,
        "window_quality_scores": np.asarray(window_quality_scores, dtype=np.float32),
        "sample_window_indices": sample_window_indices,
        "sample_window_local_indices": sample_window_local_indices,
        "sample_episode_indices": episode_indices,
        "sample_frame_indices": frame_indices,
        "sample_dataset_sample_indices": dataset_sample_indices,
        "sample_dataset_positions": dataset_positions,
    }


def _aggregate_episodes(sample_scores: np.ndarray, window_data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    episode_indices = np.asarray(window_data["sample_episode_indices"], dtype=np.int64)
    unique_episode_indices = np.asarray(sorted(np.unique(episode_indices).tolist()), dtype=np.int64)

    episode_influence_scores = np.zeros((len(unique_episode_indices),), dtype=np.float32)
    episode_mean_influence_scores = np.zeros((len(unique_episode_indices),), dtype=np.float32)
    episode_quality_scores = np.zeros((len(unique_episode_indices),), dtype=np.float32)
    episode_sample_counts = np.zeros((len(unique_episode_indices),), dtype=np.int64)
    episode_window_counts = np.zeros((len(unique_episode_indices),), dtype=np.int64)

    window_episode_indices = np.asarray(window_data["window_episode_indices"], dtype=np.int64)
    window_quality_scores = np.asarray(window_data["window_quality_scores"], dtype=np.float32)

    for out_idx, episode_idx in enumerate(unique_episode_indices.tolist()):
        sample_ids = np.where(episode_indices == episode_idx)[0]
        values = np.asarray(sample_scores[sample_ids], dtype=np.float32)
        episode_influence_scores[out_idx] = float(np.sum(values))
        episode_mean_influence_scores[out_idx] = float(np.mean(values))
        episode_sample_counts[out_idx] = int(sample_ids.size)

        window_ids = np.where(window_episode_indices == episode_idx)[0]
        episode_window_counts[out_idx] = int(window_ids.size)
        if window_ids.size > 0:
            episode_quality_scores[out_idx] = float(np.mean(window_quality_scores[window_ids]))

    return {
        "episode_indices": unique_episode_indices,
        "episode_influence_scores": episode_influence_scores,
        "episode_mean_influence_scores": episode_mean_influence_scores,
        "episode_quality_scores": _zscore(episode_quality_scores),
        "episode_quality_raw_scores": episode_quality_scores,
        "episode_sample_counts": episode_sample_counts,
        "episode_window_counts": episode_window_counts,
    }


def _build_output_payload(
    cfg: AggregateWindowConfig,
    source_payload: dict[str, Any],
    sample_scores: np.ndarray,
    sample_meta: dict[str, np.ndarray | int],
    window_data: dict[str, np.ndarray],
    episode_data: dict[str, np.ndarray],
) -> dict[str, Any]:
    source_meta = source_payload.get("metadata", {})
    meta: dict[str, Any] = {
        "leaf_names": source_meta.get("leaf_names", np.asarray([], dtype=object)),
        "leaf_shapes": source_meta.get("leaf_shapes", np.asarray([], dtype=object)),
        "episode_rewards": source_meta.get("episode_rewards", np.asarray([], dtype=np.float32)),
        "episode_lengths": source_meta.get("episode_lengths", np.asarray([], dtype=np.int64)),
        "train_window_indices": np.asarray(window_data["window_indices"], dtype=np.int64),
        "train_window_episode_indices": np.asarray(window_data["window_episode_indices"], dtype=np.int64),
        "train_window_local_indices": np.asarray(window_data["window_local_indices"], dtype=np.int64),
        "train_window_frame_starts": np.asarray(window_data["window_frame_starts"], dtype=np.int64),
        "train_window_frame_ends": np.asarray(window_data["window_frame_ends"], dtype=np.int64),
        "train_window_sample_counts": np.asarray(window_data["window_sample_counts"], dtype=np.int64),
        "train_window_is_partial": np.asarray(window_data["window_is_partial"], dtype=np.bool_),
        "train_sample_window_indices": np.asarray(window_data["sample_window_indices"], dtype=np.int64),
        "train_sample_window_local_indices": np.asarray(window_data["sample_window_local_indices"], dtype=np.int64),
        "train_sample_episode_indices": np.asarray(window_data["sample_episode_indices"], dtype=np.int64),
        "train_sample_frame_indices": np.asarray(window_data["sample_frame_indices"], dtype=np.int64),
        "train_dataset_sample_indices": np.asarray(window_data["sample_dataset_sample_indices"], dtype=np.int64),
        "train_dataset_positions": np.asarray(window_data["sample_dataset_positions"], dtype=np.int64),
        "train_episode_indices": np.asarray(episode_data["episode_indices"], dtype=np.int64),
        "train_episode_sample_counts": np.asarray(episode_data["episode_sample_counts"], dtype=np.int64),
        "train_episode_window_counts": np.asarray(episode_data["episode_window_counts"], dtype=np.int64),
        "train_score_aggregation": cfg.window_aggregation,
        "total_dataset_samples": int(sample_meta["total_dataset_samples"]),
        "source_score_pkl": str(cfg.score_pkl),
    }
    for key in ("selected_leaf_names", "selected_leaf_indices"):
        if key in source_meta:
            meta[key] = source_meta[key]

    result: dict[str, Any] = {
        "config": {
            "source_config": source_payload.get("config"),
            "aggregation": dataclasses.asdict(cfg),
        },
        "score_level": "window",
        "window_size": int(cfg.window_size),
        "window_stride": int(cfg.window_stride if cfg.window_stride is not None else cfg.window_size),
        "window_aggregation": cfg.window_aggregation,
        "quality_score": cfg.quality_score,
        "influence_scores": np.asarray(window_data["window_influence_scores"], dtype=np.float32),
        "sample_influence_scores": np.asarray(sample_scores, dtype=np.float32),
        "window_influence_scores": np.asarray(window_data["window_influence_scores"], dtype=np.float32),
        "window_sum_influence_scores": np.asarray(window_data["window_sum_influence_scores"], dtype=np.float32),
        "window_mean_influence_scores": np.asarray(window_data["window_mean_influence_scores"], dtype=np.float32),
        "window_quality_scores": np.asarray(window_data["window_quality_scores"], dtype=np.float32),
        "episode_influence_scores": np.asarray(episode_data["episode_influence_scores"], dtype=np.float32),
        "episode_mean_influence_scores": np.asarray(episode_data["episode_mean_influence_scores"], dtype=np.float32),
        "episode_quality_scores": np.asarray(episode_data["episode_quality_scores"], dtype=np.float32),
        "episode_quality_raw_scores": np.asarray(episode_data["episode_quality_raw_scores"], dtype=np.float32),
        "leaf_dims": source_payload.get("leaf_dims", {}),
        "metadata": meta,
    }
    for key in (
        "selected_leaf_names",
        "selected_leaf_shapes",
        "selected_leaf_indices",
        "all_leaf_names",
        "all_leaf_shapes",
    ):
        if key in source_payload:
            result[key] = source_payload[key]
    return result


def parse_args() -> AggregateWindowConfig:
    parser = argparse.ArgumentParser(
        description="Aggregate sample-level GraSS scores into window-level scores without recomputing GraSS.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--score-pkl", type=Path, required=True, help="Existing GraSS pickle with sample_influence_scores.")
    parser.add_argument("--output-path", type=Path, required=True, help="Output pickle path.")
    parser.add_argument("--train-config", type=str, default=None, help="Override train config if not stored in score pickle.")
    parser.add_argument("--dataset-dir", type=Path, default=None, help="Optional dataset root override.")
    parser.add_argument("--window-size", type=int, default=50, help="Window size within each train episode.")
    parser.add_argument(
        "--window-stride",
        type=int,
        default=None,
        help="Window stride within each train episode. Default uses non-overlapping windows.",
    )
    parser.add_argument(
        "--window-aggregation",
        choices=("sum", "mean"),
        default="sum",
        help="How to aggregate sample influence scores inside each window.",
    )
    parser.add_argument(
        "--quality-score",
        choices=("mean", "zscore_mean", "sum", "zscore_sum"),
        default="zscore_mean",
        help="How to derive window quality scores from sample influence scores.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output file if it exists.")

    args = parser.parse_args()
    return AggregateWindowConfig(
        score_pkl=args.score_pkl,
        output_path=args.output_path,
        train_config=args.train_config,
        dataset_dir=args.dataset_dir,
        window_size=args.window_size,
        window_stride=args.window_stride,
        window_aggregation=args.window_aggregation,
        quality_score=args.quality_score,
        overwrite=args.overwrite,
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", force=True)
    cfg = parse_args()
    if cfg.window_size <= 0:
        raise ValueError("--window-size must be > 0")
    if cfg.window_stride is None:
        cfg.window_stride = cfg.window_size
    if int(cfg.window_stride) <= 0:
        raise ValueError("--window-stride must be > 0")
    if cfg.output_path.exists() and not cfg.overwrite:
        raise FileExistsError(f"{cfg.output_path} already exists; pass --overwrite to replace it.")

    payload = _retrain._load_score_payload(cfg.score_pkl)  # noqa: SLF001
    if not isinstance(payload, dict):
        raise ValueError(f"{cfg.score_pkl} does not contain a dict payload.")
    sample_scores = _load_sample_scores(payload, cfg.score_pkl)
    train_config_name = _infer_train_config_name(payload, cfg.train_config)
    sample_meta = _load_sample_metadata_from_payload(payload, int(sample_scores.shape[0]))
    if sample_meta is None:
        train_cfg = _resolve_train_cfg(train_config_name, cfg.dataset_dir)
        sample_meta = _load_dataset_sample_metadata(train_cfg, payload, int(sample_scores.shape[0]))
    else:
        LOGGER.info("Using sample metadata already stored in score pickle; no dataset scan needed.")
    window_data = _aggregate_windows(
        sample_scores,
        sample_meta,
        window_size=int(cfg.window_size),
        window_stride=int(cfg.window_stride),
        window_aggregation=cfg.window_aggregation,
        quality_score=cfg.quality_score,
    )
    episode_data = _aggregate_episodes(sample_scores, window_data)
    result = _build_output_payload(cfg, payload, sample_scores, sample_meta, window_data, episode_data)

    cfg.output_path.parent.mkdir(parents=True, exist_ok=True)
    with cfg.output_path.open("wb") as f:
        pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)

    LOGGER.info(
        "Saved aggregated window result to %s (samples=%d, windows=%d, episodes=%d).",
        cfg.output_path,
        int(sample_scores.shape[0]),
        int(result["window_influence_scores"].shape[0]),
        int(result["episode_influence_scores"].shape[0]),
    )


if __name__ == "__main__":
    main()
