#!/usr/bin/env python3
"""Reproduce OpenPI retraining from a SCIZOR filter pickle.

The recommended path is sample-level reproduction:
1. Read SCIZOR's per-frame drop mask from `metadata.window_drop_mask`.
2. Convert the kept frames into OpenPI `sample_indices`.
3. Recompute `norm_stats.json` on the kept subset.
4. Launch OpenPI training with `data.sample_indices_path`.

This is closer to the original SCIZOR training semantics than dropping full
episodes by `influence_scores`, because the SCIZOR payload already contains
frame-level keep/drop decisions.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

def _ensure_local_cache_dirs() -> None:
    cache_root = REPO_ROOT / "tmp" / "scizor-retrain-cache"
    hf_home = cache_root / "huggingface"
    datasets_cache = hf_home / "datasets"
    xdg_cache = cache_root / "xdg"
    for path in (cache_root, hf_home, datasets_cache, xdg_cache):
        path.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(hf_home))
    os.environ.setdefault("HF_DATASETS_CACHE", str(datasets_cache))
    os.environ.setdefault("XDG_CACHE_HOME", str(xdg_cache))


_ensure_local_cache_dirs()

import openpi.training.config as _config
import retrain as _retrain
import retrain_grass
import retrain_window_joint as base


DEFAULT_SCORE_PKL = Path("/workspace/liushuo/SCIZOR/outputs/act_scizor/can_mh/scizor_joint_filter.pkl")
DEFAULT_TRAIN_CONFIG = "pi05_can_mh_lora"


def _default_output_dir(score_pkl: Path, train_config_name: str) -> Path:
    return REPO_ROOT / "scizor" / "retrain" / "output" / f"{train_config_name}_{score_pkl.stem}"


def _load_scizor_payload(score_pkl: Path) -> dict[str, Any]:
    payload = _retrain._load_score_payload(score_pkl)  # noqa: SLF001
    if not isinstance(payload, dict):
        raise ValueError(f"{score_pkl} does not contain a dict payload.")
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError(f"{score_pkl} is missing dict metadata.")
    return payload


def _episode_offsets(episode_lengths: dict[int, int]) -> tuple[dict[int, int], np.ndarray]:
    ordered_episode_ids = np.asarray(sorted(episode_lengths), dtype=np.int64)
    offsets: dict[int, int] = {}
    current = 0
    for episode_id in ordered_episode_ids.tolist():
        offsets[int(episode_id)] = current
        current += int(episode_lengths[episode_id])
    return offsets, ordered_episode_ids


def _load_drop_mask(metadata: dict[str, Any], total_windows: int) -> np.ndarray:
    if "window_drop_mask" in metadata:
        return np.asarray(metadata["window_drop_mask"], dtype=bool).reshape(-1)
    if "window_weights" in metadata:
        weights = np.asarray(metadata["window_weights"], dtype=np.float32).reshape(-1)
        return weights <= 0
    raise ValueError("SCIZOR metadata must contain window_drop_mask or window_weights.")


def _compute_kept_sample_indices(
    payload: dict[str, Any],
    dataset_dir: Path,
    *,
    strict_validation: bool,
) -> tuple[np.ndarray, dict[str, Any], dict[str, np.ndarray]]:
    metadata = payload["metadata"]
    episode_lengths = retrain_grass._load_episode_lengths(dataset_dir)  # noqa: SLF001
    offsets, ordered_episode_ids = _episode_offsets(episode_lengths)
    total_frames = int(sum(episode_lengths.values()))

    window_episode_index = np.asarray(metadata["window_episode_index"], dtype=np.int64).reshape(-1)
    window_start = np.asarray(metadata["window_start"], dtype=np.int64).reshape(-1)
    window_end = np.asarray(metadata["window_end"], dtype=np.int64).reshape(-1)
    window_drop_mask = _load_drop_mask(metadata, window_episode_index.shape[0])

    if not (
        window_episode_index.shape[0]
        == window_start.shape[0]
        == window_end.shape[0]
        == window_drop_mask.shape[0]
    ):
        raise ValueError("SCIZOR window metadata shape mismatch.")

    keep_mask = np.ones(total_frames, dtype=bool)
    coverage = np.zeros(total_frames, dtype=np.int16)

    for episode_id, start, end, is_drop in zip(
        window_episode_index.tolist(),
        window_start.tolist(),
        window_end.tolist(),
        window_drop_mask.tolist(),
        strict=True,
    ):
        episode_id = int(episode_id)
        start = int(start)
        end = int(end)
        if episode_id not in offsets:
            raise ValueError(f"Window metadata references unknown episode {episode_id}.")
        episode_length = int(episode_lengths[episode_id])
        if start < 0 or end < start or end > episode_length:
            raise ValueError(
                f"Invalid window range for episode {episode_id}: start={start}, end={end}, length={episode_length}"
            )
        global_start = offsets[episode_id] + start
        global_end = offsets[episode_id] + end
        coverage[global_start:global_end] += 1
        if is_drop:
            keep_mask[global_start:global_end] = False

    covered_frames = int(np.count_nonzero(coverage > 0))
    overlapping_frames = int(np.count_nonzero(coverage > 1))
    uncovered_frames = int(total_frames - covered_frames)

    if strict_validation and uncovered_frames != 0:
        raise ValueError(f"SCIZOR windows do not cover the full dataset: uncovered_frames={uncovered_frames}")

    kept_sample_indices = np.flatnonzero(keep_mask).astype(np.int64)
    dropped_sample_indices = np.flatnonzero(~keep_mask).astype(np.int64)

    metadata_total_frames = metadata.get("total_frames")
    if metadata_total_frames is not None and int(metadata_total_frames) != total_frames:
        raise ValueError(
            f"Dataset total_frames={total_frames} does not match SCIZOR metadata.total_frames={metadata_total_frames}"
        )

    metadata_kept_frames = metadata.get("kept_frames")
    if metadata_kept_frames is not None and int(metadata_kept_frames) != int(kept_sample_indices.shape[0]):
        raise ValueError(
            "Computed kept sample count does not match SCIZOR metadata.kept_frames: "
            f"{kept_sample_indices.shape[0]} vs {metadata_kept_frames}"
        )

    per_episode_kept = np.asarray(
        [int(np.count_nonzero(keep_mask[offsets[int(ep)] : offsets[int(ep)] + episode_lengths[int(ep)]])) for ep in ordered_episode_ids],
        dtype=np.int64,
    )
    per_episode_lengths = np.asarray([episode_lengths[int(ep)] for ep in ordered_episode_ids], dtype=np.int64)
    per_episode_keep_ratio = per_episode_kept / np.maximum(per_episode_lengths, 1)

    score_vector = np.asarray(payload.get("influence_scores", []), dtype=np.float32).reshape(-1)
    if score_vector.size not in {0, ordered_episode_ids.shape[0]}:
        raise ValueError(
            f"SCIZOR influence_scores has {score_vector.size} entries, expected 0 or {ordered_episode_ids.shape[0]}"
        )

    summary = {
        "score_pkl": str(DEFAULT_SCORE_PKL),
        "dataset_dir": str(dataset_dir),
        "source": metadata.get("source"),
        "curation_config": metadata.get("curation_config"),
        "total_episodes": int(ordered_episode_ids.shape[0]),
        "total_samples": int(total_frames),
        "kept_samples": int(kept_sample_indices.shape[0]),
        "dropped_samples": int(dropped_sample_indices.shape[0]),
        "keep_ratio": float(kept_sample_indices.shape[0] / max(total_frames, 1)),
        "covered_samples": covered_frames,
        "uncovered_samples": uncovered_frames,
        "overlapping_samples": overlapping_frames,
        "partially_kept_episodes": int(np.count_nonzero((per_episode_kept > 0) & (per_episode_kept < per_episode_lengths))),
        "fully_dropped_episodes": int(np.count_nonzero(per_episode_kept == 0)),
        "fully_kept_episodes": int(np.count_nonzero(per_episode_kept == per_episode_lengths)),
        "metadata_total_frames": None if metadata_total_frames is None else int(metadata_total_frames),
        "metadata_kept_frames": None if metadata_kept_frames is None else int(metadata_kept_frames),
        "metadata_union_drop_frames": None
        if metadata.get("union_drop_frames") is None
        else int(metadata["union_drop_frames"]),
    }

    artifacts = {
        "episode_ids": ordered_episode_ids,
        "episode_lengths": per_episode_lengths,
        "episode_kept_counts": per_episode_kept,
        "episode_keep_ratio_from_mask": per_episode_keep_ratio.astype(np.float32),
        "kept_sample_indices": kept_sample_indices,
        "dropped_sample_indices": dropped_sample_indices,
    }
    if score_vector.size:
        artifacts["episode_scores"] = score_vector.astype(np.float32)
    if "episode_keep_ratio" in metadata:
        artifacts["episode_keep_ratio_from_metadata"] = np.asarray(metadata["episode_keep_ratio"], dtype=np.float32).reshape(-1)
    if "episode_bad_ratio" in metadata:
        artifacts["episode_bad_ratio_from_metadata"] = np.asarray(metadata["episode_bad_ratio"], dtype=np.float32).reshape(-1)
    return kept_sample_indices, summary, artifacts


def _write_artifacts(
    output_dir: Path,
    *,
    score_pkl: Path,
    kept_sample_indices: np.ndarray,
    summary: dict[str, Any],
    artifacts: dict[str, np.ndarray],
    overwrite: bool,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_indices = output_dir / "kept_sample_indices.npy"
    _retrain._save_sample_indices(output_indices, kept_sample_indices, overwrite=overwrite)  # noqa: SLF001
    np.savez_compressed(output_dir / "scizor_selection_artifacts.npz", **artifacts)
    summary = dict(summary)
    summary["score_pkl"] = str(score_pkl)
    (output_dir / "scizor_selection_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return output_indices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a SCIZOR filter pickle into OpenPI sample_indices, recompute norm_stats, and retrain."
    )
    parser.add_argument("--score-pkl", type=Path, default=DEFAULT_SCORE_PKL)
    parser.add_argument("--train-config", type=str, default=DEFAULT_TRAIN_CONFIG)
    parser.add_argument("--dataset-dir", type=Path, default=None, help="Override dataset root. Defaults from train config.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output dir for sample subset and norm_stats.")
    parser.add_argument("--exp-name", type=str, default=None, help="Experiment name for retraining.")
    parser.add_argument("--checkpoint-base-dir", type=Path, default=REPO_ROOT / "checkpoints")
    parser.add_argument("--num-train-steps", type=int, default=None)
    parser.add_argument("--max-frames-for-norm-stats", type=int, default=None)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--skip-norm-stats", action="store_true", help="Skip norm_stats recomputation for fast debugging.")
    parser.add_argument("--no-strict-validation", action="store_true", help="Allow uncovered frames in SCIZOR mask metadata.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = _load_scizor_payload(args.score_pkl)

    if args.dataset_dir is None:
        dataset_dir, default_exp_name = _retrain._infer_defaults_from_train_config(args.train_config)  # noqa: SLF001
    else:
        dataset_dir = args.dataset_dir
        default_exp_name = args.train_config

    output_dir = args.output_dir or _default_output_dir(args.score_pkl, args.train_config)
    strict_validation = not args.no_strict_validation
    train_cfg = _config.get_config(args.train_config)

    kept_sample_indices, summary, artifacts = _compute_kept_sample_indices(
        payload,
        dataset_dir,
        strict_validation=strict_validation,
    )
    output_indices = _write_artifacts(
        output_dir,
        score_pkl=args.score_pkl,
        kept_sample_indices=kept_sample_indices,
        summary=summary,
        artifacts=artifacts,
        overwrite=args.overwrite,
    )

    print(f"Score file: {args.score_pkl}")
    print(f"Train config: {args.train_config}")
    print(f"Dataset: {dataset_dir}")
    print(
        "SCIZOR selection summary: "
        f"kept_samples={summary['kept_samples']}/{summary['total_samples']}, "
        f"fully_kept_episodes={summary['fully_kept_episodes']}, "
        f"partially_kept_episodes={summary['partially_kept_episodes']}, "
        f"fully_dropped_episodes={summary['fully_dropped_episodes']}"
    )
    print(f"Output dir: {output_dir}")
    print(f"Sample indices: {output_indices}")

    if not args.skip_norm_stats:
        print("Recomputing norm_stats.json for SCIZOR-selected sample subset...")
        train_cfg_for_norm = dataclasses.replace(train_cfg, num_workers=0)
        base._recompute_norm_stats_for_sample_subset(  # noqa: SLF001
            train_cfg_for_norm,
            sample_indices_path=output_indices,
            output_dir=output_dir,
            max_frames=args.max_frames_for_norm_stats,
        )
        print(f"Subset norm_stats written to: {output_dir / 'norm_stats.json'}")

    if args.prepare_only:
        return

    exp_name = args.exp_name or f"{default_exp_name}_scizor_{args.score_pkl.stem}"
    retrain_cfg = base._build_train_config(  # noqa: SLF001
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
    from scripts import train as train_script

    train_script.main(retrain_cfg)


if __name__ == "__main__":
    main()
