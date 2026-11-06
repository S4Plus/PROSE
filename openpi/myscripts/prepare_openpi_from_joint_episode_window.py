#!/usr/bin/env python3
"""Convert joint episode-window selection into OpenPI sample indices and retrain."""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

OPENPI_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = OPENPI_ROOT / "src"
if str(OPENPI_ROOT) not in sys.path:
    sys.path.insert(0, str(OPENPI_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import openpi.training.config as _config
import retrain as _retrain
import retrain_window_joint as _window_base
from scripts import train as train_script


def _resolve_train_context(args: argparse.Namespace, payload: dict[str, Any]) -> tuple[str, Path, str]:
    train_config_name = args.train_config or _retrain._extract_score_config_name(payload)  # noqa: SLF001
    if train_config_name is None:
        raise ValueError("--train-config is required because the score pickle does not include config.config_name.")

    if args.dataset_dir is None:
        dataset_dir, default_exp_name = _retrain._infer_defaults_from_train_config(train_config_name)  # noqa: SLF001
    else:
        dataset_dir = args.dataset_dir
        default_exp_name = train_config_name
    return train_config_name, dataset_dir, default_exp_name


def _build_episode_sample_lookup(metadata: dict[str, Any]) -> dict[int, dict[str, np.ndarray]]:
    if "train_sample_episode_indices" not in metadata or "train_dataset_sample_indices" not in metadata:
        raise ValueError(
            "Score pickle metadata must contain train_sample_episode_indices and train_dataset_sample_indices."
        )

    sample_episode_indices = np.asarray(metadata["train_sample_episode_indices"], dtype=np.int64).reshape(-1)
    dataset_sample_indices = np.asarray(metadata["train_dataset_sample_indices"], dtype=np.int64).reshape(-1)
    if sample_episode_indices.shape[0] != dataset_sample_indices.shape[0]:
        raise ValueError("Sample metadata shape mismatch in score pickle.")

    local_frame_indices = np.empty_like(sample_episode_indices, dtype=np.int64)
    counters: dict[int, int] = {}
    for idx, episode in enumerate(sample_episode_indices.tolist()):
        episode = int(episode)
        frame_idx = counters.get(episode, 0)
        local_frame_indices[idx] = frame_idx
        counters[episode] = frame_idx + 1

    episode_lookup: dict[int, dict[str, np.ndarray]] = {}
    for episode in sorted(np.unique(sample_episode_indices).tolist()):
        episode = int(episode)
        mask = sample_episode_indices == episode
        episode_lookup[episode] = {
            "dataset_sample_indices": dataset_sample_indices[mask].astype(np.int64, copy=False),
            "local_frame_indices": local_frame_indices[mask].astype(np.int64, copy=False),
        }
    return episode_lookup


def _load_selection(selection_npz: Path) -> dict[str, np.ndarray]:
    with np.load(selection_npz) as data:
        required = ("episode_keep_mask", "window_episode", "window_start", "window_end", "window_keep")
        missing = [key for key in required if key not in data]
        if missing:
            raise ValueError(f"{selection_npz} is missing required keys: {missing}")
        return {key: np.asarray(data[key]) for key in data.files}


def _materialize_kept_sample_indices(
    selection: dict[str, np.ndarray],
    episode_sample_lookup: dict[int, dict[str, np.ndarray]],
) -> tuple[np.ndarray, dict[str, int]]:
    window_episode = np.asarray(selection["window_episode"], dtype=np.int64).reshape(-1)
    window_start = np.asarray(selection["window_start"], dtype=np.int64).reshape(-1)
    window_end = np.asarray(selection["window_end"], dtype=np.int64).reshape(-1)
    window_keep = np.asarray(selection["window_keep"], dtype=np.int64).reshape(-1).astype(bool)

    if not (
        window_episode.shape[0]
        == window_start.shape[0]
        == window_end.shape[0]
        == window_keep.shape[0]
    ):
        raise ValueError("Selection NPZ window arrays have inconsistent lengths.")

    kept_sample_indices: list[np.ndarray] = []
    kept_windows = 0
    covered_episodes: set[int] = set()
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
        lookup = episode_sample_lookup.get(episode)
        if lookup is None:
            continue
        frame_indices = lookup["local_frame_indices"]
        sample_indices = lookup["dataset_sample_indices"]
        mask = (frame_indices >= start) & (frame_indices < end)
        if np.any(mask):
            kept_sample_indices.append(sample_indices[mask])
            kept_windows += 1
            covered_episodes.add(episode)

    if kept_sample_indices:
        unique_indices = np.unique(np.concatenate(kept_sample_indices).astype(np.int64, copy=False))
    else:
        unique_indices = np.empty((0,), dtype=np.int64)

    stats = {
        "kept_windows_with_samples": int(kept_windows),
        "covered_episodes_from_windows": int(len(covered_episodes)),
    }
    return unique_indices, stats


def _write_summary(
    output_dir: Path,
    *,
    args: argparse.Namespace,
    selection_npz: Path,
    score_pkl: Path,
    dataset_dir: Path,
    kept_sample_indices: np.ndarray,
    selection: dict[str, np.ndarray],
    mapping_stats: dict[str, int],
) -> None:
    summary = {
        "method": "score_joint_episode_window_to_openpi",
        "score_pkl": str(score_pkl),
        "selection_npz": str(selection_npz),
        "dataset_dir": str(dataset_dir),
        "train_config": args.train_config,
        "kept_samples": int(kept_sample_indices.shape[0]),
        "kept_windows": int(np.count_nonzero(np.asarray(selection["window_keep"], dtype=np.int64))),
        "num_windows": int(np.asarray(selection["window_keep"]).shape[0]),
        "kept_episodes": int(np.count_nonzero(np.asarray(selection["episode_keep_mask"], dtype=np.int64))),
        "num_episodes": int(np.asarray(selection["episode_keep_mask"]).shape[0]),
        **mapping_stats,
    }
    (output_dir / "openpi_selection_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


def _recompute_norm_stats_with_fallback(
    train_cfg: _config.TrainConfig,
    *,
    sample_indices_path: Path,
    output_dir: Path,
    max_frames: int | None,
) -> tuple[Path, bool]:
    try:
        _window_base._recompute_norm_stats_for_sample_subset(  # noqa: SLF001
            train_cfg,
            sample_indices_path=sample_indices_path,
            output_dir=output_dir,
            max_frames=max_frames,
        )
        return output_dir / "norm_stats.json", False
    except PermissionError:
        single_worker_cfg = dataclasses.replace(train_cfg, num_workers=0)
        _window_base._recompute_norm_stats_for_sample_subset(  # noqa: SLF001
            single_worker_cfg,
            sample_indices_path=sample_indices_path,
            output_dir=output_dir,
            max_frames=max_frames,
        )
        return output_dir / "norm_stats.json", True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert joint episode-window selection into OpenPI sample indices and retrain."
    )
    parser.add_argument("--score-pkl", type=Path, required=True, help="OpenPI influence score pickle.")
    parser.add_argument(
        "--selection-npz",
        type=Path,
        required=True,
        help="Selection NPZ produced by myscripts/filter_joint_episode_window.py.",
    )
    parser.add_argument("--train-config", type=str, default=None, help="OpenPI train config name.")
    parser.add_argument("--dataset-dir", type=Path, default=None, help="LeRobot dataset root. If omitted, infer from train config.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory for indices, summary, and norm_stats.")
    parser.add_argument("--output-indices", type=Path, default=None, help="Optional explicit kept_sample_indices.npy path.")
    parser.add_argument("--exp-name", type=str, default=None, help="Experiment name for retraining.")
    parser.add_argument("--checkpoint-base-dir", type=Path, default=Path("./checkpoints"))
    parser.add_argument("--num-train-steps", type=int, default=None)
    parser.add_argument("--max-frames-for-norm-stats", type=int, default=None)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = _window_base._load_window_payload(args.score_pkl) if args.score_pkl.name.endswith("_window.pkl") else _retrain._load_score_payload(args.score_pkl)  # noqa: SLF001
    if not isinstance(payload, dict) or "metadata" not in payload or not isinstance(payload["metadata"], dict):
        raise ValueError(f"{args.score_pkl} does not contain the expected dict metadata payload.")

    train_config_name, dataset_dir, default_exp_name = _resolve_train_context(args, payload)
    selection = _load_selection(args.selection_npz)
    episode_sample_lookup = _build_episode_sample_lookup(payload["metadata"])
    kept_sample_indices, mapping_stats = _materialize_kept_sample_indices(selection, episode_sample_lookup)
    if kept_sample_indices.size == 0:
        raise ValueError("Selection produced zero OpenPI samples after window-to-sample mapping.")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    output_indices = args.output_indices or (output_dir / "kept_sample_indices.npy")
    _retrain._save_sample_indices(output_indices, kept_sample_indices, overwrite=args.overwrite)  # noqa: SLF001
    np.savez_compressed(output_dir / "joint_episode_window_selection_openpi.npz", kept_sample_indices=kept_sample_indices)
    _write_summary(
        output_dir,
        args=args,
        selection_npz=args.selection_npz,
        score_pkl=args.score_pkl,
        dataset_dir=dataset_dir,
        kept_sample_indices=kept_sample_indices,
        selection=selection,
        mapping_stats=mapping_stats,
    )

    train_cfg = _config.get_config(train_config_name)
    print(f"Score file: {args.score_pkl}")
    print(f"Selection file: {args.selection_npz}")
    print(f"Train config: {train_config_name}")
    print(f"Dataset: {dataset_dir}")
    print(f"Kept samples: {kept_sample_indices.shape[0]}")
    print(f"Output dir: {output_dir}")
    print(f"Sample indices: {output_indices}")
    print("Recomputing norm_stats.json for selected sample subset...")
    norm_stats_path, used_single_worker_fallback = _recompute_norm_stats_with_fallback(
        train_cfg,
        sample_indices_path=output_indices,
        output_dir=output_dir,
        max_frames=args.max_frames_for_norm_stats,
    )
    if used_single_worker_fallback:
        print("norm_stats fallback: retrying with num_workers=0 due to multiprocessing permission limits.")
    print(f"Subset norm_stats written to: {norm_stats_path}")

    if args.prepare_only:
        return

    exp_name = args.exp_name or f"{default_exp_name}_score_joint_episode_window"
    retrain_cfg = _window_base._build_train_config(  # noqa: SLF001
        train_cfg,
        sample_indices_path=output_indices,
        output_dir=output_dir,
        exp_name=exp_name,
        checkpoint_base_dir=args.checkpoint_base_dir,
        overwrite=args.overwrite,
        num_train_steps=args.num_train_steps,
    )
    print(
        f"Starting retraining: exp_name={retrain_cfg.exp_name}, "
        f"checkpoint_dir={retrain_cfg.checkpoint_dir}, sample_indices={output_indices}"
    )
    train_script.main(retrain_cfg)


if __name__ == "__main__":
    main()
