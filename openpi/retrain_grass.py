#!/usr/bin/env python3
"""Retrain OpenPI from GraSS influence scores with episode-level filtering.

This script is tailored for GraSS outputs like:
`/workspace/liushuo/cupid/openpi/tmp/can_mh_grass/influence_grass.pkl`

Workflow:
1. Load GraSS sample-level influence scores.
2. Map scored samples back to dataset episodes and aggregate scores per episode.
3. Drop the lowest-scoring episodes by a target ratio.
4. Materialize a filtered LeRobot dataset.
5. Recompute `norm_stats.json` for the filtered dataset.
6. Launch training on the filtered dataset with the filtered dataset as both
   `repo_id` and norm-stats asset source.
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import json
from pathlib import Path

import numpy as np
import torch
import tqdm

import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
import openpi.shared.normalize as normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import retrain as _retrain
from scripts import compute_norm_stats as norm_stats_script
from scripts import train as train_script


DEFAULT_SCORE_PKL = Path("/workspace/liushuo/cupid/openpi/tmp/can_mh_grass/influence_grass.pkl")


def _resolve_keep_ratio(keep_ratio: float | None, drop_ratio: float | None) -> float:
    if keep_ratio is not None and drop_ratio is not None:
        raise ValueError("Only one of --keep-ratio or --drop-ratio can be set.")
    if keep_ratio is not None:
        return keep_ratio
    if drop_ratio is not None:
        if not 0.0 <= drop_ratio < 1.0:
            raise ValueError("--drop-ratio must be in [0,1).")
        return 1.0 - drop_ratio
    return 0.8


def _default_output_dataset(dataset_dir: Path, keep_ratio: float, score_pkl: Path, aggregation: str) -> Path:
    keep_pct = int(round(keep_ratio * 100))
    return dataset_dir.parent / f"{dataset_dir.name}_grass_keep{keep_pct}_{aggregation}_{score_pkl.stem}"


def _load_episode_lengths(dataset_dir: Path) -> dict[int, int]:
    episodes_path = dataset_dir / "meta" / "episodes.jsonl"
    if not episodes_path.exists():
        raise FileNotFoundError(f"Missing episodes metadata: {episodes_path}")

    lengths: dict[int, int] = {}
    with episodes_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            lengths[int(record["episode_index"])] = int(record["length"])
    return lengths


def _load_sample_episode_indices(
    train_cfg: _config.TrainConfig,
    sample_indices: np.ndarray,
) -> np.ndarray:
    data_cfg = train_cfg.data.create(train_cfg.assets_dirs, train_cfg.model)
    if data_cfg.repo_id is None:
        raise ValueError("Training config does not define a dataset repo_id.")

    metadata = lerobot_dataset.LeRobotDatasetMetadata(data_cfg.repo_id)
    dataset = lerobot_dataset.LeRobotDataset(
        data_cfg.repo_id,
        delta_timestamps={
            key: [t / metadata.fps for t in range(train_cfg.model.action_horizon)]
            for key in data_cfg.action_sequence_keys
        },
    )

    sample_indices = np.asarray(sample_indices, dtype=np.int64).reshape(-1)
    if sample_indices.size == 0:
        raise ValueError("No sample indices to map.")
    if np.any(sample_indices < 0):
        raise ValueError("Sample indices must be non-negative.")
    if int(sample_indices.max()) >= len(dataset):
        raise ValueError(
            f"Sample index {int(sample_indices.max())} exceeds dataset size {len(dataset)}."
        )

    episode_indices = np.empty(sample_indices.shape[0], dtype=np.int64)
    for i, sample_idx in enumerate(tqdm.tqdm(sample_indices, desc="Mapping sample -> episode")):
        item = dataset[int(sample_idx)]
        episode_indices[i] = int(item["episode_index"])
    return episode_indices


def _aggregate_episode_scores(
    scores: np.ndarray,
    sample_episode_indices: np.ndarray,
    episode_lengths: dict[int, int],
    aggregation: str,
) -> tuple[dict[int, float], dict[int, int], set[int]]:
    buckets: dict[int, list[float]] = collections.defaultdict(list)
    unique_sample_counts: collections.Counter[int] = collections.Counter()

    for score, episode_idx in zip(scores, sample_episode_indices, strict=True):
        episode_idx = int(episode_idx)
        buckets[episode_idx].append(float(score))
        unique_sample_counts[episode_idx] += 1

    if aggregation == "mean":
        reducer = np.mean
    elif aggregation == "sum":
        reducer = np.sum
    elif aggregation == "min":
        reducer = np.min
    elif aggregation == "max":
        reducer = np.max
    else:
        raise ValueError(f"Unsupported aggregation: {aggregation}")

    episode_scores = {episode_idx: float(reducer(values)) for episode_idx, values in buckets.items()}
    fully_scored = {
        episode_idx
        for episode_idx, count in unique_sample_counts.items()
        if count >= episode_lengths.get(episode_idx, count + 1)
    }
    return episode_scores, dict(unique_sample_counts), fully_scored


def _select_episodes_to_keep(
    dataset_dir: Path,
    scores: np.ndarray,
    sample_indices: np.ndarray,
    train_cfg: _config.TrainConfig,
    *,
    keep_ratio: float,
    aggregation: str,
    partial_episode_policy: str,
    unscored_episode_policy: str,
) -> tuple[np.ndarray, dict[str, object]]:
    episode_lengths = _load_episode_lengths(dataset_dir)
    total_episodes = len(episode_lengths)
    sample_episode_indices = _load_sample_episode_indices(train_cfg, sample_indices)
    episode_scores, sample_counts, fully_scored = _aggregate_episode_scores(
        scores,
        sample_episode_indices,
        episode_lengths,
        aggregation,
    )

    scored_episodes = set(episode_scores)
    partial_episodes = scored_episodes - fully_scored
    unscored_episodes = set(episode_lengths) - scored_episodes

    if partial_episode_policy == "drop":
        candidate_episodes = sorted(fully_scored)
        always_keep_partial = sorted(partial_episodes)
    elif partial_episode_policy == "keep":
        candidate_episodes = sorted(scored_episodes)
        always_keep_partial = []
    else:
        raise ValueError(f"Unsupported partial episode policy: {partial_episode_policy}")

    if not candidate_episodes:
        raise ValueError(
            "No candidate episodes are available for filtering. "
            "Try --partial-episode-policy keep or use a score file that covers full episodes."
        )

    candidate_scores = np.asarray([episode_scores[ep] for ep in candidate_episodes], dtype=float)
    kept_candidate_positions = _retrain._select_keep_indices(candidate_scores, keep_ratio)  # noqa: SLF001
    kept_candidate_episodes = {candidate_episodes[pos] for pos in kept_candidate_positions.tolist()}

    final_keep = set(kept_candidate_episodes)
    final_keep.update(always_keep_partial)

    if unscored_episode_policy == "keep":
        final_keep.update(unscored_episodes)
    elif unscored_episode_policy != "drop":
        raise ValueError(f"Unsupported unscored episode policy: {unscored_episode_policy}")

    summary = {
        "total_dataset_episodes": total_episodes,
        "scored_samples": int(len(scores)),
        "scored_episodes": len(scored_episodes),
        "fully_scored_episodes": len(fully_scored),
        "partial_episodes": len(partial_episodes),
        "unscored_episodes": len(unscored_episodes),
        "candidate_episodes": len(candidate_episodes),
        "kept_candidate_episodes": len(kept_candidate_episodes),
        "final_kept_episodes": len(final_keep),
        "aggregation": aggregation,
        "partial_episode_policy": partial_episode_policy,
        "unscored_episode_policy": unscored_episode_policy,
        "sample_counts_by_episode": sample_counts,
    }
    return np.asarray(sorted(final_keep), dtype=np.int64), summary


def _build_filtered_data_factory(
    data_factory: _config.DataConfigFactory,
    filtered_repo: Path,
) -> _config.DataConfigFactory:
    return dataclasses.replace(
        data_factory,
        repo_id=str(filtered_repo),
        sample_indices_path=None,
        assets=_config.AssetsConfig(
            assets_dir=str(filtered_repo.parent),
            asset_id=filtered_repo.name,
        ),
    )


def _recompute_norm_stats(
    train_cfg: _config.TrainConfig,
    filtered_repo: Path,
    max_frames: int | None,
) -> None:
    data_factory = _build_filtered_data_factory(train_cfg.data, filtered_repo)
    data_cfg = data_factory.create(train_cfg.assets_dirs, train_cfg.model)

    if data_cfg.rlds_data_dir is not None:
        data_loader, num_batches = norm_stats_script.create_rlds_dataloader(
            data_cfg,
            train_cfg.model.action_horizon,
            train_cfg.batch_size,
            max_frames=max_frames,
        )
    else:
        dataset = _data_loader.create_torch_dataset(
            data_cfg,
            train_cfg.model.action_horizon,
            train_cfg.model,
        )
        dataset = _data_loader.TransformedDataset(
            dataset,
            [
                *data_cfg.repack_transforms.inputs,
                *data_cfg.data_transforms.inputs,
                norm_stats_script.RemoveStrings(),
            ],
        )
        if max_frames is not None and max_frames < len(dataset):
            num_batches = max(1, max_frames // train_cfg.batch_size)
            shuffle = True
        else:
            num_batches = max(1, len(dataset) // train_cfg.batch_size)
            shuffle = False
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=train_cfg.batch_size,
            shuffle=shuffle,
            num_workers=train_cfg.num_workers,
            collate_fn=_data_loader._collate_fn,  # noqa: SLF001
            drop_last=True,
        )

    stats = {key: normalize.RunningStats() for key in ("state", "actions")}
    for batch in tqdm.tqdm(data_loader, total=num_batches, desc="Computing norm_stats"):
        for key in stats:
            stats[key].update(np.asarray(batch[key]))

    norm_stats = {key: running.get_statistics() for key, running in stats.items()}
    normalize.save(filtered_repo, norm_stats)


def _build_train_config(
    base_train_cfg: _config.TrainConfig,
    filtered_repo: Path,
    *,
    exp_name: str,
    checkpoint_base_dir: Path,
    overwrite: bool,
    num_train_steps: int | None,
) -> _config.TrainConfig:
    filtered_data = _build_filtered_data_factory(base_train_cfg.data, filtered_repo)
    kwargs = {
        "data": filtered_data,
        "exp_name": exp_name,
        "checkpoint_base_dir": str(checkpoint_base_dir),
        "overwrite": overwrite,
        "resume": False,
    }
    if num_train_steps is not None:
        kwargs["num_train_steps"] = int(num_train_steps)
    return dataclasses.replace(base_train_cfg, **kwargs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter GraSS low-score episodes, recompute norm_stats, and retrain OpenPI."
    )
    parser.add_argument(
        "--score-pkl",
        type=Path,
        default=DEFAULT_SCORE_PKL,
        help="Path to the GraSS output pickle.",
    )
    parser.add_argument(
        "--keep-ratio",
        type=float,
        default=None,
        help="Fraction of candidate episodes to keep.",
    )
    parser.add_argument(
        "--drop-ratio",
        type=float,
        default=None,
        help="Fraction of candidate episodes to drop from the bottom.",
    )
    parser.add_argument(
        "--aggregation",
        choices=("mean", "sum", "min", "max"),
        default="mean",
        help="How sample-level GraSS scores are aggregated into an episode score.",
    )
    parser.add_argument(
        "--partial-episode-policy",
        choices=("drop", "keep"),
        default="drop",
        help="Whether partially scored episodes participate in filtering. Default drops them from scoring and keeps them.",
    )
    parser.add_argument(
        "--unscored-episode-policy",
        choices=("keep", "drop"),
        default="keep",
        help="How to treat episodes with no GraSS score coverage.",
    )
    parser.add_argument(
        "--train-config",
        type=str,
        default=None,
        help="Training config name. If omitted, infer from the GraSS pickle metadata.",
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
    parser.add_argument(
        "--exp-name",
        type=str,
        default=None,
        help="Experiment name for retraining.",
    )
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
    keep_ratio = _resolve_keep_ratio(args.keep_ratio, args.drop_ratio)

    payload = _retrain._load_score_payload(args.score_pkl)  # noqa: SLF001
    scores = _retrain._load_scores(payload, args.score_pkl)  # noqa: SLF001
    train_config_name = args.train_config or _retrain._extract_score_config_name(payload)  # noqa: SLF001
    if train_config_name is None:
        raise ValueError("--train-config is required because the score pickle does not include config.config_name.")

    if args.dataset_dir is None:
        dataset_dir, default_exp_name = _retrain._infer_defaults_from_train_config(train_config_name)  # noqa: SLF001
    else:
        dataset_dir = args.dataset_dir
        default_exp_name = train_config_name

    train_cfg = _config.get_config(train_config_name)
    output_dataset = args.output_dataset or _default_output_dataset(
        dataset_dir,
        keep_ratio,
        args.score_pkl,
        args.aggregation,
    )

    sample_indices = _retrain._extract_scored_sample_indices(payload)  # noqa: SLF001
    if sample_indices is None:
        sample_indices = np.arange(len(scores), dtype=np.int64)
        print(
            "GraSS pickle does not contain explicit sample indices. "
            f"Assuming sequential training samples [0, {len(scores)})."
        )
    if len(sample_indices) != len(scores):
        raise ValueError(f"sample index count {len(sample_indices)} does not match score count {len(scores)}")

    keep_episode_indices, summary = _select_episodes_to_keep(
        dataset_dir,
        scores,
        sample_indices,
        train_cfg,
        keep_ratio=keep_ratio,
        aggregation=args.aggregation,
        partial_episode_policy=args.partial_episode_policy,
        unscored_episode_policy=args.unscored_episode_policy,
    )

    exp_name = args.exp_name or f"{default_exp_name}_grass_keep{int(round(keep_ratio * 100))}"

    print(f"Score file: {args.score_pkl}")
    print(f"Train config: {train_config_name}")
    print(f"Dataset: {dataset_dir}")
    print(
        "Episode selection summary: "
        f"scored_samples={summary['scored_samples']}, "
        f"scored_episodes={summary['scored_episodes']}, "
        f"fully_scored_episodes={summary['fully_scored_episodes']}, "
        f"partial_episodes={summary['partial_episodes']}, "
        f"unscored_episodes={summary['unscored_episodes']}, "
        f"final_kept_episodes={summary['final_kept_episodes']}/{summary['total_dataset_episodes']}"
    )
    print(f"Writing filtered dataset to: {output_dataset}")

    _retrain._materialize_subset(  # noqa: SLF001
        dataset_dir,
        output_dataset,
        keep_episode_indices,
        overwrite=args.overwrite,
    )
    print("Filtered dataset materialized.")

    print("Recomputing norm_stats.json for filtered dataset...")
    _recompute_norm_stats(train_cfg, output_dataset, args.max_frames_for_norm_stats)
    print(f"Filtered norm_stats written to: {output_dataset / 'norm_stats.json'}")

    if args.prepare_only:
        return

    retrain_cfg = _build_train_config(
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
    train_script.main(retrain_cfg)


if __name__ == "__main__":
    main()
