#!/usr/bin/env python3
"""Convert ACT filter artifacts into OpenPI sample indices and optionally train."""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pyarrow.parquet as pq

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


def _resolve_train_context(args: argparse.Namespace) -> tuple[str, Path, str]:
    if args.train_config is None:
        raise ValueError("--train-config is required.")

    if args.dataset_dir is None:
        dataset_dir, default_exp_name = _retrain._infer_defaults_from_train_config(args.train_config)  # noqa: SLF001
    else:
        dataset_dir = args.dataset_dir
        default_exp_name = args.train_config
    return args.train_config, dataset_dir, default_exp_name


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


def _parse_episode_id_from_path(path: Path) -> int:
    stem = path.stem
    if not stem.startswith("episode_"):
        raise ValueError(f"Unexpected episode filename: {path}")
    episode_str = stem.split("_")[-1]
    if not episode_str.isdigit():
        raise ValueError(f"Failed to parse episode id from: {path}")
    return int(episode_str)


def _index_episode_files(dataset_dir: Path) -> dict[int, dict[str, Any]]:
    episode_paths = sorted((dataset_dir / "data").glob("chunk-*/episode_*.parquet"))
    if not episode_paths:
        raise FileNotFoundError(f"No episode parquet files found under {dataset_dir / 'data'}")

    episode_index: dict[int, dict[str, Any]] = {}
    global_offset = 0
    for path in episode_paths:
        metadata = pq.read_metadata(path)
        num_rows = int(metadata.num_rows)
        episode = _parse_episode_id_from_path(path)
        episode_index[episode] = {
            "path": path,
            "num_rows": num_rows,
            "offset": global_offset,
        }
        global_offset += num_rows
    return episode_index


def _load_episode_sample_data(
    episode: int,
    episode_info: dict[str, Any],
    *,
    cache: dict[int, dict[str, np.ndarray]],
) -> dict[str, np.ndarray]:
    if episode in cache:
        return cache[episode]

    table = pq.read_table(episode_info["path"])
    column_names = set(table.column_names)
    num_rows = int(table.num_rows)

    if "index" in column_names:
        sample_indices = np.asarray(table["index"].to_pylist(), dtype=np.int64).reshape(-1)
    else:
        start = int(episode_info["offset"])
        sample_indices = np.arange(start, start + num_rows, dtype=np.int64)

    if "frame_index" in column_names:
        frame_indices = np.asarray(table["frame_index"].to_pylist(), dtype=np.int64).reshape(-1)
    else:
        frame_indices = np.arange(num_rows, dtype=np.int64)

    payload = {
        "sample_indices": sample_indices,
        "frame_indices": frame_indices,
    }
    cache[episode] = payload
    return payload


def _parse_bool_like(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "t", "yes", "y"}:
        return True
    if normalized in {"0", "false", "f", "no", "n"}:
        return False
    raise ValueError(f"Failed to parse boolean value: {value!r}")


def _load_keep_episodes_from_txt(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Keep-episodes file not found: {path}")
    episodes: list[int] = []
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line:
            continue
        episodes.append(int(line))
    if not episodes:
        raise ValueError(f"No kept episodes found in {path}")
    return np.asarray(sorted(set(episodes)), dtype=np.int64)


def _load_keep_episodes_from_csv(
    path: Path,
    *,
    episode_column: str,
    keep_column: str | None,
    drop_column: str | None,
) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Episodes CSV not found: {path}")

    keep_eps: list[int] = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        if episode_column not in reader.fieldnames:
            raise ValueError(f"{path} is missing episode column: {episode_column}")

        resolved_keep_column = keep_column
        resolved_drop_column = drop_column
        if resolved_keep_column == "auto":
            resolved_keep_column = "keep" if "keep" in reader.fieldnames else None
        if resolved_drop_column == "auto":
            resolved_drop_column = "drop" if "drop" in reader.fieldnames else None

        if resolved_keep_column is None and resolved_drop_column is None:
            raise ValueError(
                f"{path} must contain either a keep column or a drop column. "
                "Pass --csv-keep-column/--csv-drop-column if the names differ."
            )
        if resolved_keep_column is not None and resolved_keep_column not in reader.fieldnames:
            raise ValueError(f"{path} is missing keep column: {resolved_keep_column}")
        if resolved_drop_column is not None and resolved_drop_column not in reader.fieldnames:
            raise ValueError(f"{path} is missing drop column: {resolved_drop_column}")

        for row in reader:
            episode = int(row[episode_column])
            keep = True
            if resolved_keep_column is not None:
                keep = _parse_bool_like(row[resolved_keep_column])
            elif resolved_drop_column is not None:
                keep = not _parse_bool_like(row[resolved_drop_column])
            if keep:
                keep_eps.append(episode)

    if not keep_eps:
        raise ValueError(f"No kept episodes were parsed from {path}")
    return np.asarray(sorted(set(keep_eps)), dtype=np.int64)


def _load_selection(selection_npz: Path) -> dict[str, np.ndarray]:
    with np.load(selection_npz) as data:
        required = ("episode_keep_mask", "window_episode", "window_start", "window_end", "window_keep")
        missing = [key for key in required if key not in data]
        if missing:
            raise ValueError(f"{selection_npz} is missing required keys: {missing}")
        return {key: np.asarray(data[key]) for key in data.files}


def _materialize_samples_from_kept_episodes(
    keep_episode_ids: np.ndarray,
    episode_index: dict[int, dict[str, Any]],
) -> tuple[np.ndarray, dict[str, int]]:
    cache: dict[int, dict[str, np.ndarray]] = {}
    kept_chunks: list[np.ndarray] = []
    missing_episodes: list[int] = []
    for episode in np.asarray(keep_episode_ids, dtype=np.int64).reshape(-1).tolist():
        info = episode_index.get(int(episode))
        if info is None:
            missing_episodes.append(int(episode))
            continue
        episode_data = _load_episode_sample_data(int(episode), info, cache=cache)
        kept_chunks.append(episode_data["sample_indices"])

    if not kept_chunks:
        raise ValueError("Episode filter produced zero OpenPI samples.")

    kept_sample_indices = np.unique(np.concatenate(kept_chunks).astype(np.int64, copy=False))
    stats = {
        "kept_episodes_requested": int(np.asarray(keep_episode_ids).shape[0]),
        "kept_episodes_found": int(len(kept_chunks)),
        "missing_episodes": int(len(missing_episodes)),
    }
    return kept_sample_indices, stats


def _materialize_samples_from_selection_npz(
    selection: dict[str, np.ndarray],
    episode_index: dict[int, dict[str, Any]],
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

    cache: dict[int, dict[str, np.ndarray]] = {}
    kept_chunks: list[np.ndarray] = []
    covered_episodes: set[int] = set()
    missing_episodes: set[int] = set()
    kept_windows = 0

    for episode, start, end, keep in zip(
        window_episode.tolist(),
        window_start.tolist(),
        window_end.tolist(),
        window_keep.tolist(),
        strict=True,
    ):
        if not keep:
            continue
        if end <= start:
            continue
        info = episode_index.get(int(episode))
        if info is None:
            missing_episodes.add(int(episode))
            continue
        episode_data = _load_episode_sample_data(int(episode), info, cache=cache)
        mask = (episode_data["frame_indices"] >= int(start)) & (episode_data["frame_indices"] < int(end))
        if not np.any(mask):
            continue
        kept_chunks.append(episode_data["sample_indices"][mask])
        covered_episodes.add(int(episode))
        kept_windows += 1

    if not kept_chunks:
        raise ValueError("Selection NPZ produced zero OpenPI samples.")

    kept_sample_indices = np.unique(np.concatenate(kept_chunks).astype(np.int64, copy=False))
    stats = {
        "kept_windows_with_samples": int(kept_windows),
        "covered_episodes_from_windows": int(len(covered_episodes)),
        "missing_episodes": int(len(missing_episodes)),
    }
    return kept_sample_indices, stats


def _write_summary(
    output_dir: Path,
    *,
    args: argparse.Namespace,
    dataset_dir: Path,
    input_kind: str,
    input_path: Path,
    kept_sample_indices: np.ndarray,
    extra_stats: dict[str, int],
) -> None:
    summary = {
        "method": "act_filter_to_openpi",
        "source_tag": args.source_tag,
        "input_kind": input_kind,
        "input_path": str(input_path),
        "dataset_dir": str(dataset_dir),
        "train_config": args.train_config,
        "kept_samples": int(kept_sample_indices.shape[0]),
        **extra_stats,
    }
    (output_dir / "openpi_selection_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert ACT filter artifacts into OpenPI sample indices and retrain."
    )
    parser.add_argument("--train-config", type=str, required=True, help="OpenPI train config name.")
    parser.add_argument("--dataset-dir", type=Path, default=None, help="LeRobot dataset root. If omitted, infer from train config.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory for indices, summary, and norm_stats.")
    parser.add_argument("--output-indices", type=Path, default=None, help="Optional explicit kept_sample_indices.npy path.")
    parser.add_argument("--source-tag", type=str, default="act_filter", help="Short tag recorded in summary and defaults.")

    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--selection-npz", type=Path, default=None, help="ACT joint episode-window selection NPZ.")
    source_group.add_argument("--keep-episodes-txt", type=Path, default=None, help="Text file with one kept episode id per line.")
    source_group.add_argument("--episodes-csv", type=Path, default=None, help="CSV with episode keep/drop info.")

    parser.add_argument("--csv-episode-column", type=str, default="episode")
    parser.add_argument(
        "--csv-keep-column",
        type=str,
        default="auto",
        help="Keep column name, or 'auto' to prefer 'keep'. Use 'none' to disable keep-column parsing.",
    )
    parser.add_argument(
        "--csv-drop-column",
        type=str,
        default="auto",
        help="Drop column name, or 'auto' to prefer 'drop'. Use 'none' to disable drop-column parsing.",
    )

    parser.add_argument("--exp-name", type=str, default=None, help="Experiment name for retraining.")
    parser.add_argument("--checkpoint-base-dir", type=Path, default=Path("./checkpoints"))
    parser.add_argument("--num-train-steps", type=int, default=None)
    parser.add_argument("--max-frames-for-norm-stats", type=int, default=None)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_config_name, dataset_dir, default_exp_name = _resolve_train_context(args)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    episode_index = _index_episode_files(dataset_dir)
    input_kind: str
    input_path: Path
    if args.selection_npz is not None:
        selection = _load_selection(args.selection_npz)
        kept_sample_indices, extra_stats = _materialize_samples_from_selection_npz(selection, episode_index)
        input_kind = "selection_npz"
        input_path = args.selection_npz
    else:
        keep_column = None if args.csv_keep_column == "none" else args.csv_keep_column
        drop_column = None if args.csv_drop_column == "none" else args.csv_drop_column
        if args.keep_episodes_txt is not None:
            keep_episode_ids = _load_keep_episodes_from_txt(args.keep_episodes_txt)
            input_kind = "keep_episodes_txt"
            input_path = args.keep_episodes_txt
        elif args.episodes_csv is not None:
            keep_episode_ids = _load_keep_episodes_from_csv(
                args.episodes_csv,
                episode_column=args.csv_episode_column,
                keep_column=keep_column,
                drop_column=drop_column,
            )
            input_kind = "episodes_csv"
            input_path = args.episodes_csv
        else:
            raise AssertionError("One input source must be provided.")
        kept_sample_indices, extra_stats = _materialize_samples_from_kept_episodes(keep_episode_ids, episode_index)
        extra_stats["kept_episodes"] = int(np.asarray(keep_episode_ids).shape[0])

    output_indices = args.output_indices or (output_dir / "kept_sample_indices.npy")
    _retrain._save_sample_indices(output_indices, kept_sample_indices, overwrite=args.overwrite)  # noqa: SLF001
    np.savez_compressed(output_dir / "act_filter_selection_openpi.npz", kept_sample_indices=kept_sample_indices)
    _write_summary(
        output_dir,
        args=args,
        dataset_dir=dataset_dir,
        input_kind=input_kind,
        input_path=input_path,
        kept_sample_indices=kept_sample_indices,
        extra_stats=extra_stats,
    )

    train_cfg = _config.get_config(train_config_name)
    print(f"Input kind: {input_kind}")
    print(f"Input path: {input_path}")
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

    exp_name = args.exp_name or f"{default_exp_name}_{args.source_tag}"
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
