#!/usr/bin/env python3
"""Convert a SCIZOR filter pickle into OpenVLA sample indices and retrain."""

from __future__ import annotations

import argparse
import dataclasses
import json
import pickle
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


def _load_payload(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        payload = pickle.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} does not contain a dict payload.")
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError(f"{path} is missing dict metadata.")
    return payload


def _episode_lengths(metadata: LeRobotDatasetMetadata) -> dict[int, int]:
    return {int(row["episode_index"]): int(row["length"]) for row in metadata.episodes}


def _episode_offsets(episode_lengths: dict[int, int]) -> tuple[dict[int, int], np.ndarray]:
    ordered_episode_ids = np.asarray(sorted(episode_lengths), dtype=np.int64)
    offsets: dict[int, int] = {}
    current = 0
    for episode_id in ordered_episode_ids.tolist():
        offsets[int(episode_id)] = current
        current += int(episode_lengths[episode_id])
    return offsets, ordered_episode_ids


def _load_drop_mask(payload_meta: dict[str, Any], total_windows: int) -> np.ndarray:
    if "window_drop_mask" in payload_meta:
        return np.asarray(payload_meta["window_drop_mask"], dtype=bool).reshape(-1)
    if "window_weights" in payload_meta:
        weights = np.asarray(payload_meta["window_weights"], dtype=np.float32).reshape(-1)
        return weights <= 0.0
    raise ValueError("SCIZOR metadata must contain window_drop_mask or window_weights.")


def _compute_kept_sample_indices(
    payload: dict[str, Any],
    metadata: LeRobotDatasetMetadata,
    *,
    strict_validation: bool,
) -> tuple[np.ndarray, dict[str, Any], dict[str, np.ndarray]]:
    payload_meta = payload["metadata"]
    episode_lengths = _episode_lengths(metadata)
    offsets, ordered_episode_ids = _episode_offsets(episode_lengths)
    total_frames = int(sum(episode_lengths.values()))

    window_episode_index = np.asarray(payload_meta["window_episode_index"], dtype=np.int64).reshape(-1)
    window_start = np.asarray(payload_meta["window_start"], dtype=np.int64).reshape(-1)
    window_end = np.asarray(payload_meta["window_end"], dtype=np.int64).reshape(-1)
    window_drop_mask = _load_drop_mask(payload_meta, window_episode_index.shape[0])

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

    meta_total_frames = payload_meta.get("total_frames")
    if meta_total_frames is not None and int(meta_total_frames) != total_frames:
        raise ValueError(
            f"Dataset total_frames={total_frames} does not match SCIZOR metadata.total_frames={meta_total_frames}"
        )

    meta_kept_frames = payload_meta.get("kept_frames")
    if meta_kept_frames is not None and int(meta_kept_frames) != int(kept_sample_indices.shape[0]):
        raise ValueError(
            "Computed kept sample count does not match SCIZOR metadata.kept_frames: "
            f"{kept_sample_indices.shape[0]} vs {meta_kept_frames}"
        )

    per_episode_lengths = np.asarray([episode_lengths[int(ep)] for ep in ordered_episode_ids], dtype=np.int64)
    per_episode_kept = np.asarray(
        [
            int(np.count_nonzero(keep_mask[offsets[int(ep)] : offsets[int(ep)] + episode_lengths[int(ep)]]))
            for ep in ordered_episode_ids
        ],
        dtype=np.int64,
    )
    per_episode_keep_ratio = per_episode_kept / np.maximum(per_episode_lengths, 1)

    summary = {
        "method": "scizor_frame_mask_to_openvla",
        "source": payload_meta.get("source"),
        "curation_config": payload_meta.get("curation_config"),
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
        "metadata_total_frames": None if meta_total_frames is None else int(meta_total_frames),
        "metadata_kept_frames": None if meta_kept_frames is None else int(meta_kept_frames),
    }

    artifacts = {
        "episode_ids": ordered_episode_ids,
        "episode_lengths": per_episode_lengths,
        "episode_kept_counts": per_episode_kept,
        "episode_keep_ratio_from_mask": per_episode_keep_ratio.astype(np.float32),
        "kept_sample_indices": kept_sample_indices,
        "dropped_sample_indices": dropped_sample_indices,
    }
    episode_scores = np.asarray(payload.get("influence_scores", []), dtype=np.float32).reshape(-1)
    if episode_scores.size in {0, ordered_episode_ids.shape[0]}:
        if episode_scores.size:
            artifacts["episode_scores"] = episode_scores.astype(np.float32)
    else:
        raise ValueError(
            f"SCIZOR influence_scores has {episode_scores.size} entries, expected 0 or {ordered_episode_ids.shape[0]}"
        )
    return kept_sample_indices, summary, artifacts


def _save_outputs(
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
    if output_indices.exists() and not overwrite:
        raise FileExistsError(f"{output_indices} already exists. Use --overwrite to replace it.")
    np.save(output_indices, kept_sample_indices)
    np.savez_compressed(output_dir / "scizor_selection_artifacts.npz", **artifacts)
    payload = dict(summary)
    payload["score_pkl"] = str(score_pkl)
    (output_dir / "scizor_selection_summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return output_indices


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
        description="Convert a SCIZOR filter pickle into OpenVLA sample_indices, recompute norm_stats, and retrain."
    )
    parser.add_argument("--score-pkl", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--dataset-name", type=str, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--exp-name", type=str, default=None)
    parser.add_argument("--retrain-run-root", type=Path, default=Path("runs"))
    parser.add_argument("--num-train-steps", type=int, default=None)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--no-strict-validation", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = _load_payload(args.score_pkl)
    base_cfg = load_finetune_config(args.checkpoint_dir)

    dataset_root = args.dataset_root or base_cfg.dataset_root
    dataset_name = args.dataset_name or base_cfg.dataset_name
    metadata = LeRobotDatasetMetadata(
        dataset_root,
        dataset_name=dataset_name,
        task_description_override=base_cfg.task_description,
        norm_stats_path=base_cfg.norm_stats_path,
    )

    kept_sample_indices, summary, artifacts = _compute_kept_sample_indices(
        payload,
        metadata,
        strict_validation=not args.no_strict_validation,
    )
    if kept_sample_indices.size == 0:
        raise ValueError("SCIZOR selection produced zero kept samples.")

    output_indices = _save_outputs(
        args.output_dir,
        score_pkl=args.score_pkl,
        kept_sample_indices=kept_sample_indices,
        summary=summary,
        artifacts=artifacts,
        overwrite=args.overwrite,
    )

    subset_norm_stats_path = args.output_dir / "norm_stats.json"
    recompute_norm_stats_for_indices(metadata, kept_sample_indices, subset_norm_stats_path)

    if args.prepare_only:
        return

    exp_name = args.exp_name or f"{dataset_name}_scizor"
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
