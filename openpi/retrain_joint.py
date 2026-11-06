#!/usr/bin/env python3
"""Retrain OpenPI with joint episode filtering from influence and quality scores.

Joint scoring rule:
    joint_score_i = signed_power(quality_score_i, joint_quality_power)
                    * minmax_norm(influence_i) ** joint_influence_power

Defaults:
    joint_quality_power = 1
    joint_influence_power = 1
    joint_score = quality_score * normalized_influence

Notes:
- `normalized_influence` is min-max normalization over the available episode
  influence scores in the score file.
- If all episode influence scores are identical, the normalized influence is set
  to 1 for every scored episode so filtering falls back to quality only.

Example: prepare filtered datasets only
  UV_CACHE_DIR=./tmp/uv-cache uv run python retrain_joint.py \
    --score-pkl /workspace/liushuo/cupid/openpi/tmp/can_mh_grass_last4_strict/influence_grass.pkl \
    --quality-score-csv /workspace/liushuo/cupid/3_model/quality_eval/can_mh_rank_fusion/quality_episodes.csv \
    --episode-influence-field episode_influence_scores \
    --train-config pi05_can_mh_lora \
    --keep-ratio 0.8 \
    --joint-quality-power 1 \
    --joint-influence-power 1 \
    --output-dataset /workspace/liushuo/cupid/datasets/lerobot/can_mh_joint_keep80_last4_csvq \
    --prepare-only \
    --overwrite

Examples: keep 70/80/90 and retrain
  CUDA_VISIBLE_DEVICES=0 UV_CACHE_DIR=./tmp/uv-cache nohup uv run python retrain_joint.py \
    --score-pkl /workspace/liushuo/cupid/openpi/tmp/can_mh_grass_last4_strict/influence_grass.pkl \
    --quality-score-csv /workspace/liushuo/cupid/3_model/quality_eval/can_mh_rank_fusion/quality_episodes.csv \
    --episode-influence-field episode_influence_scores \
    --train-config pi05_can_mh_lora \
    --keep-ratio 0.7 \
    --joint-quality-power 1 \
    --joint-influence-power 1 \
    --output-dataset /workspace/liushuo/cupid/datasets/lerobot/can_mh_joint_keep70_last4_csvq \
    --exp-name can_mh_joint_keep70_last4_csvq \
    --num-train-steps 10000 \
    --overwrite \
    > /workspace/liushuo/cupid/openpi/tmp/can_mh_grass_last4_strict/retrain_joint_keep70.log 2>&1 &

  CUDA_VISIBLE_DEVICES=1 UV_CACHE_DIR=./tmp/uv-cache nohup uv run python retrain_joint.py \
    --score-pkl /workspace/liushuo/cupid/openpi/tmp/can_mh_grass_last4_strict/influence_grass.pkl \
    --quality-score-csv /workspace/liushuo/cupid/3_model/quality_eval/can_mh_rank_fusion/quality_episodes.csv \
    --episode-influence-field episode_influence_scores \
    --train-config pi05_can_mh_lora \
    --keep-ratio 0.8 \
    --joint-quality-power 1 \
    --joint-influence-power 1 \
    --output-dataset /workspace/liushuo/cupid/datasets/lerobot/can_mh_joint_keep80_last4_csvq \
    --exp-name can_mh_joint_keep80_last4_csvq \
    --num-train-steps 10000 \
    --overwrite \
    > /workspace/liushuo/cupid/openpi/tmp/can_mh_grass_last4_strict/retrain_joint_keep80.log 2>&1 &

  CUDA_VISIBLE_DEVICES=2 UV_CACHE_DIR=./tmp/uv-cache nohup uv run python retrain_joint.py \
    --score-pkl /workspace/liushuo/cupid/openpi/tmp/can_mh_grass_last4_strict/influence_grass.pkl \
    --quality-score-csv /workspace/liushuo/cupid/3_model/quality_eval/can_mh_rank_fusion/quality_episodes.csv \
    --episode-influence-field episode_influence_scores \
    --train-config pi05_can_mh_lora \
    --keep-ratio 0.9 \
    --joint-quality-power 1 \
    --joint-influence-power 1 \
    --output-dataset /workspace/liushuo/cupid/datasets/lerobot/can_mh_joint_keep90_last4_csvq \
    --exp-name can_mh_joint_keep90_last4_csvq \
    --num-train-steps 10000 \
    --overwrite \
    > /workspace/liushuo/cupid/openpi/tmp/can_mh_grass_last4_strict/retrain_joint_keep90.log 2>&1 &
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
from pathlib import Path

import numpy as np

import openpi.training.config as _config
import retrain as _retrain
import retrain_grass as retrain_grass


def _default_output_dataset(
    dataset_dir: Path,
    keep_ratio: float,
    influence_score_pkl: Path,
    quality_power: float,
    influence_power: float,
    quality_score_pkl: Path | None = None,
) -> Path:
    keep_pct = int(round(keep_ratio * 100))
    score_tag = influence_score_pkl.stem
    if quality_score_pkl is not None and quality_score_pkl != influence_score_pkl:
        score_tag = f"{influence_score_pkl.stem}__qfrom_{quality_score_pkl.stem}"
    return dataset_dir.parent / (
        f"{dataset_dir.name}_joint_keep{keep_pct}_"
        f"q{quality_power:g}_i{influence_power:g}_{score_tag}"
    )


def _signed_power(values: np.ndarray, power: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if power < 0:
        raise ValueError("joint powers must be >= 0")
    return np.sign(values) * (np.abs(values) ** power)


def _minmax_normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        raise ValueError("Cannot normalize an empty influence array.")
    vmin = float(np.min(values))
    vmax = float(np.max(values))
    if abs(vmax - vmin) < 1e-12:
        return np.ones_like(values, dtype=np.float32)
    return ((values - vmin) / (vmax - vmin)).astype(np.float32)


def _write_selection_artifacts(
    output_dataset: Path,
    *,
    episode_indices: np.ndarray,
    influence_scores: np.ndarray,
    quality_scores: np.ndarray,
    normalized_influence: np.ndarray,
    joint_scores: np.ndarray,
    kept_episode_indices: np.ndarray,
    summary: dict[str, object],
) -> None:
    output_dataset.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dataset / "joint_selection_scores.npz",
        episode_indices=np.asarray(episode_indices, dtype=np.int64),
        influence_scores=np.asarray(influence_scores, dtype=np.float32),
        quality_scores=np.asarray(quality_scores, dtype=np.float32),
        normalized_influence=np.asarray(normalized_influence, dtype=np.float32),
        joint_scores=np.asarray(joint_scores, dtype=np.float32),
        kept_episode_indices=np.asarray(kept_episode_indices, dtype=np.int64),
    )
    (output_dataset / "joint_selection_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )


def _load_quality_csv(
    csv_path: Path,
    *,
    episode_column: str,
    quality_column: str,
) -> tuple[np.ndarray, np.ndarray]:
    episode_indices: list[int] = []
    quality_scores: list[float] = []
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"Quality CSV has no header: {csv_path}")
        if episode_column not in reader.fieldnames:
            raise ValueError(f"Quality CSV {csv_path} is missing episode column: {episode_column}")
        if quality_column not in reader.fieldnames:
            raise ValueError(f"Quality CSV {csv_path} is missing quality column: {quality_column}")
        for row in reader:
            episode_indices.append(int(row[episode_column]))
            quality_scores.append(float(row[quality_column]))
    if not episode_indices:
        raise ValueError(f"Quality CSV is empty: {csv_path}")
    return (
        np.asarray(episode_indices, dtype=np.int64),
        np.asarray(quality_scores, dtype=np.float32),
    )


def _align_episode_vectors(
    influence_episode_indices: np.ndarray,
    influence_scores: np.ndarray,
    quality_episode_indices: np.ndarray,
    quality_scores: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    influence_map = {
        int(ep): float(score)
        for ep, score in zip(influence_episode_indices, influence_scores, strict=True)
    }
    quality_map = {
        int(ep): float(score)
        for ep, score in zip(quality_episode_indices, quality_scores, strict=True)
    }
    common_episode_indices = np.asarray(
        sorted(set(influence_map).intersection(quality_map)),
        dtype=np.int64,
    )
    if common_episode_indices.size == 0:
        raise ValueError("Influence scores and quality scores have no overlapping episode indices.")
    aligned_influence = np.asarray([influence_map[int(ep)] for ep in common_episode_indices], dtype=np.float32)
    aligned_quality = np.asarray([quality_map[int(ep)] for ep in common_episode_indices], dtype=np.float32)
    return common_episode_indices, aligned_influence, aligned_quality


def _extract_episode_vector(payload: dict, field: str, dataset_dir: Path | None = None) -> tuple[np.ndarray, np.ndarray]:
    if field not in payload:
        raise ValueError(f"Score file does not contain field: {field}")
    values = np.asarray(payload[field], dtype=np.float32).reshape(-1)
    if values.size == 0:
        raise ValueError(f"Field {field} is empty.")

    metadata = payload.get("metadata", {})
    if isinstance(metadata, dict) and "train_episode_indices" in metadata:
        episode_indices = np.asarray(metadata["train_episode_indices"], dtype=np.int64).reshape(-1)
    elif dataset_dir is not None:
        total = retrain_grass._load_episode_lengths(dataset_dir)  # noqa: SLF001
        if len(values) == len(total):
            episode_indices = np.arange(len(values), dtype=np.int64)
        else:
            raise ValueError(
                f"Field {field} has {len(values)} entries, but metadata.train_episode_indices is missing."
            )
    else:
        raise ValueError(
            f"Field {field} requires metadata.train_episode_indices, or pass --dataset-dir for full-length inference."
        )

    if len(episode_indices) != len(values):
        raise ValueError(
            f"Field {field} has {len(values)} values but train_episode_indices has {len(episode_indices)} entries."
        )
    return episode_indices, values


def _select_episodes_to_keep(
    dataset_dir: Path,
    episode_indices: np.ndarray,
    influence_scores: np.ndarray,
    quality_scores: np.ndarray,
    *,
    keep_ratio: float,
    joint_quality_power: float,
    joint_influence_power: float,
    unscored_episode_policy: str,
) -> tuple[np.ndarray, dict[str, object], dict[str, np.ndarray]]:
    episode_lengths = retrain_grass._load_episode_lengths(dataset_dir)  # noqa: SLF001
    total_episodes = len(episode_lengths)

    episode_indices = np.asarray(episode_indices, dtype=np.int64).reshape(-1)
    influence_scores = np.asarray(influence_scores, dtype=np.float32).reshape(-1)
    quality_scores = np.asarray(quality_scores, dtype=np.float32).reshape(-1)
    if not (len(episode_indices) == len(influence_scores) == len(quality_scores)):
        raise ValueError("Episode indices, influence scores, and quality scores must have the same length.")

    norm_influence = _minmax_normalize(influence_scores)
    joint_scores = _signed_power(quality_scores, joint_quality_power) * (
        norm_influence ** joint_influence_power
    )

    candidate_positions = _retrain._select_keep_indices(joint_scores.astype(float), keep_ratio)  # noqa: SLF001
    kept_scored_episodes = set(episode_indices[candidate_positions].tolist())
    scored_episodes = set(episode_indices.tolist())
    unscored_episodes = set(episode_lengths) - scored_episodes

    final_keep = set(kept_scored_episodes)
    if unscored_episode_policy == "keep":
        final_keep.update(unscored_episodes)
    elif unscored_episode_policy != "drop":
        raise ValueError(f"Unsupported unscored episode policy: {unscored_episode_policy}")

    summary = {
        "total_dataset_episodes": total_episodes,
        "scored_episodes": len(scored_episodes),
        "unscored_episodes": len(unscored_episodes),
        "kept_scored_episodes": len(kept_scored_episodes),
        "final_kept_episodes": len(final_keep),
        "joint_quality_power": joint_quality_power,
        "joint_influence_power": joint_influence_power,
        "joint_score_min": float(np.min(joint_scores)),
        "joint_score_max": float(np.max(joint_scores)),
        "normalized_influence_min": float(np.min(norm_influence)),
        "normalized_influence_max": float(np.max(norm_influence)),
        "scored_episode_indices": episode_indices.astype(np.int64).tolist(),
        "kept_scored_episode_indices": sorted(kept_scored_episodes),
        "final_kept_episode_indices": sorted(final_keep),
    }
    artifacts = {
        "normalized_influence": norm_influence.astype(np.float32),
        "joint_scores": joint_scores.astype(np.float32),
    }
    return np.asarray(sorted(final_keep), dtype=np.int64), summary, artifacts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter episodes by joint influence/quality score, recompute norm_stats, and retrain OpenPI."
    )
    parser.add_argument(
        "--score-pkl",
        type=Path,
        required=True,
        help="Path to the influence score pickle. Can also contain quality scores if --quality-score-pkl is omitted.",
    )
    parser.add_argument(
        "--quality-score-pkl",
        type=Path,
        default=None,
        help="Optional separate pickle providing episode quality scores.",
    )
    parser.add_argument(
        "--quality-score-csv",
        type=Path,
        default=None,
        help="Optional CSV providing episode quality scores.",
    )
    parser.add_argument(
        "--quality-csv-episode-column",
        type=str,
        default="episode",
        help="Episode index column in --quality-score-csv.",
    )
    parser.add_argument(
        "--quality-csv-score-column",
        type=str,
        default="quality_score",
        help="Quality score column in --quality-score-csv.",
    )
    parser.add_argument(
        "--episode-influence-field",
        type=str,
        default="episode_influence_scores",
        help="Field name for episode-level influence scores in the score pickle.",
    )
    parser.add_argument(
        "--episode-quality-field",
        type=str,
        default="episode_quality_scores",
        help="Field name for episode-level quality scores in the score pickle.",
    )
    parser.add_argument(
        "--joint-quality-power",
        type=float,
        default=1.0,
        help="Exponent applied to quality scores in the joint rule.",
    )
    parser.add_argument(
        "--joint-influence-power",
        type=float,
        default=1.0,
        help="Exponent applied to min-max normalized influence scores in the joint rule.",
    )
    parser.add_argument(
        "--keep-ratio",
        type=float,
        default=None,
        help="Fraction of scored episodes to keep.",
    )
    parser.add_argument(
        "--drop-ratio",
        type=float,
        default=None,
        help="Fraction of scored episodes to drop from the bottom.",
    )
    parser.add_argument(
        "--unscored-episode-policy",
        choices=("keep", "drop"),
        default="keep",
        help="How to treat dataset episodes that are not present in the score file.",
    )
    parser.add_argument(
        "--train-config",
        type=str,
        default=None,
        help="Training config name. If omitted, infer from the score pickle metadata.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=None,
        help="Original dataset root. If omitted, infer from the training config.",
    )
    parser.add_argument(
        "--output-dataset",
        type=Path,
        default=None,
        help="Where to write the filtered dataset.",
    )
    parser.add_argument("--exp-name", type=str, default=None, help="Experiment name for retraining.")
    parser.add_argument(
        "--checkpoint-base-dir",
        type=Path,
        default=Path("./checkpoints"),
        help="Base directory for checkpoints.",
    )
    parser.add_argument(
        "--num-train-steps",
        type=int,
        default=None,
        help="Override the train step count from the base config.",
    )
    parser.add_argument(
        "--max-frames-for-norm-stats",
        type=int,
        default=None,
        help="Optionally subsample frames when recomputing norm_stats.",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Only build the filtered dataset and recompute norm_stats, skip training.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing filtered dataset and checkpoint directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    keep_ratio = retrain_grass._resolve_keep_ratio(args.keep_ratio, args.drop_ratio)  # noqa: SLF001
    if args.quality_score_pkl is not None and args.quality_score_csv is not None:
        raise ValueError("Only one of --quality-score-pkl or --quality-score-csv can be set.")

    influence_payload = _retrain._load_score_payload(args.score_pkl)  # noqa: SLF001
    quality_payload = (
        _retrain._load_score_payload(args.quality_score_pkl)  # noqa: SLF001
        if args.quality_score_pkl is not None
        else influence_payload
    )

    train_config_name = args.train_config or _retrain._extract_score_config_name(influence_payload)  # noqa: SLF001
    if train_config_name is None and args.quality_score_pkl is not None:
        train_config_name = _retrain._extract_score_config_name(quality_payload)  # noqa: SLF001
    if train_config_name is None:
        raise ValueError("--train-config is required because the score pickle does not include config.config_name.")

    if args.dataset_dir is None:
        dataset_dir, default_exp_name = _retrain._infer_defaults_from_train_config(train_config_name)  # noqa: SLF001
    else:
        dataset_dir = args.dataset_dir
        default_exp_name = train_config_name

    episode_indices, influence_scores = _extract_episode_vector(
        influence_payload,
        args.episode_influence_field,
        dataset_dir=dataset_dir,
    )
    if args.quality_score_csv is not None:
        quality_episode_indices, quality_scores = _load_quality_csv(
            args.quality_score_csv,
            episode_column=args.quality_csv_episode_column,
            quality_column=args.quality_csv_score_column,
        )
    else:
        quality_episode_indices, quality_scores = _extract_episode_vector(
            quality_payload,
            args.episode_quality_field,
            dataset_dir=dataset_dir,
        )
    episode_indices, influence_scores, quality_scores = _align_episode_vectors(
        episode_indices,
        influence_scores,
        quality_episode_indices,
        quality_scores,
    )

    output_dataset = args.output_dataset or _default_output_dataset(
        dataset_dir,
        keep_ratio,
        args.score_pkl,
        args.joint_quality_power,
        args.joint_influence_power,
        args.quality_score_pkl or args.quality_score_csv,
    )
    keep_episode_indices, summary, artifacts = _select_episodes_to_keep(
        dataset_dir,
        episode_indices,
        influence_scores,
        quality_scores,
        keep_ratio=keep_ratio,
        joint_quality_power=args.joint_quality_power,
        joint_influence_power=args.joint_influence_power,
        unscored_episode_policy=args.unscored_episode_policy,
    )

    exp_name = args.exp_name or (
        f"{default_exp_name}_joint_keep{int(round(keep_ratio * 100))}"
        f"_q{args.joint_quality_power:g}_i{args.joint_influence_power:g}"
    )
    train_cfg = _config.get_config(train_config_name)

    print(f"Influence score file: {args.score_pkl}")
    print(f"Quality score file: {args.quality_score_csv or args.quality_score_pkl or args.score_pkl}")
    print(f"Train config: {train_config_name}")
    print(f"Dataset: {dataset_dir}")
    print(
        "Joint selection summary: "
        f"scored_episodes={summary['scored_episodes']}, "
        f"unscored_episodes={summary['unscored_episodes']}, "
        f"kept_scored_episodes={summary['kept_scored_episodes']}, "
        f"final_kept_episodes={summary['final_kept_episodes']}/{summary['total_dataset_episodes']}, "
        f"joint_score_range=[{summary['joint_score_min']:.6f}, {summary['joint_score_max']:.6f}]"
    )
    print(f"Writing filtered dataset to: {output_dataset}")

    _retrain._materialize_subset(  # noqa: SLF001
        dataset_dir,
        output_dataset,
        keep_episode_indices,
        overwrite=args.overwrite,
    )
    print("Filtered dataset materialized.")

    _write_selection_artifacts(
        output_dataset,
        episode_indices=episode_indices,
        influence_scores=influence_scores,
        quality_scores=quality_scores,
        normalized_influence=artifacts["normalized_influence"],
        joint_scores=artifacts["joint_scores"],
        kept_episode_indices=keep_episode_indices,
        summary=summary,
    )
    print(f"Joint score artifacts written to: {output_dataset / 'joint_selection_scores.npz'}")

    print("Recomputing norm_stats.json for filtered dataset...")
    retrain_grass._recompute_norm_stats(  # noqa: SLF001
        train_cfg,
        output_dataset,
        args.max_frames_for_norm_stats,
    )
    print(f"Filtered norm_stats written to: {output_dataset / 'norm_stats.json'}")

    if args.prepare_only:
        return

    retrain_cfg = retrain_grass._build_train_config(  # noqa: SLF001
        train_cfg,
        output_dataset,
        exp_name=exp_name,
        checkpoint_base_dir=args.checkpoint_base_dir,
        overwrite=args.overwrite,
        num_train_steps=args.num_train_steps,
    )
    print(
        f"Starting retraining: exp_name={retrain_cfg.exp_name}, "
        f"checkpoint_dir={retrain_cfg.checkpoint_dir}, "
        f"dataset={output_dataset}"
    )
    from scripts import train as train_script

    train_script.main(retrain_cfg)


if __name__ == "__main__":
    main()
