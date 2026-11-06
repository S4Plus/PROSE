#!/usr/bin/env python3
"""Randomly keep a ratio of dataset episodes, recompute norm stats, and retrain.

Workflow:
1. Resolve the base training config and dataset.
2. Randomly sample a subset of episodes with a reproducible RNG seed.
3. Materialize a filtered LeRobot dataset with contiguous episode indices.
4. Recompute `norm_stats.json` on the filtered dataset.
5. Launch training against the filtered dataset.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import numpy as np

import openpi.training.config as _config
import retrain as _retrain
import retrain_grass as _retrain_grass
from scripts import train as train_script


def _resolve_keep_ratio(keep_ratio: float | None, drop_ratio: float | None) -> float:
    if keep_ratio is not None and drop_ratio is not None:
        raise ValueError("Only one of --keep-ratio or --drop-ratio can be set.")
    if keep_ratio is not None:
        if not 0.0 < keep_ratio <= 1.0:
            raise ValueError("--keep-ratio must be in (0,1].")
        return keep_ratio
    if drop_ratio is not None:
        if not 0.0 <= drop_ratio < 1.0:
            raise ValueError("--drop-ratio must be in [0,1).")
        return 1.0 - drop_ratio
    return 0.8


def _default_output_dataset(dataset_dir: Path, keep_ratio: float, seed: int) -> Path:
    keep_pct = int(round(keep_ratio * 100))
    random_root = dataset_dir.parent / f"{dataset_dir.name}_random"
    return random_root / f"{dataset_dir.name}_random_keep{keep_pct}_seed{seed}"


def _load_episode_indices(dataset_dir: Path) -> np.ndarray:
    episodes_path = dataset_dir / "meta" / "episodes.jsonl"
    if episodes_path.exists():
        episode_indices: list[int] = []
        with episodes_path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                episode_indices.append(int(record["episode_index"]))
        if episode_indices:
            return np.asarray(sorted(episode_indices), dtype=np.int64)

    num_episodes = _retrain._count_dataset_episodes(dataset_dir)  # noqa: SLF001
    return np.arange(num_episodes, dtype=np.int64)


def _select_random_episode_indices(dataset_dir: Path, keep_ratio: float, seed: int) -> np.ndarray:
    episode_indices = _load_episode_indices(dataset_dir)
    if episode_indices.size == 0:
        raise ValueError(f"No episodes found in dataset: {dataset_dir}")

    keep_count = max(1, int(episode_indices.size * keep_ratio))
    rng = np.random.default_rng(seed)
    selected = rng.choice(episode_indices, size=keep_count, replace=False)
    return np.asarray(sorted(selected.tolist()), dtype=np.int64)


def _write_selection_summary(
    output_dataset: Path,
    *,
    source_dataset: Path,
    keep_ratio: float,
    seed: int,
    selected_episode_indices: np.ndarray,
    total_episodes: int,
) -> None:
    payload = {
        "source_dataset": str(source_dataset),
        "keep_ratio": float(keep_ratio),
        "seed": int(seed),
        "kept_episodes": int(selected_episode_indices.size),
        "total_episodes": int(total_episodes),
        "selected_episode_indices": [int(x) for x in selected_episode_indices.tolist()],
    }
    summary_path = output_dataset / "meta" / "random_selection.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(payload, indent=2) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Randomly sample dataset episodes, recompute norm_stats, and retrain OpenPI."
    )
    parser.add_argument(
        "--train-config",
        type=str,
        required=True,
        help="Training config name defined in openpi.training.config.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=None,
        help="Original dataset root. If omitted, infer from the training config.",
    )
    parser.add_argument(
        "--keep-ratio",
        type=float,
        default=None,
        help="Fraction of episodes to keep.",
    )
    parser.add_argument(
        "--drop-ratio",
        type=float,
        default=None,
        help="Fraction of episodes to drop at random.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed used for episode sampling.",
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
    parser.add_argument(
        "--enable-wandb",
        action="store_true",
        help="Enable Weights & Biases logging. Default is disabled.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    keep_ratio = _resolve_keep_ratio(args.keep_ratio, args.drop_ratio)

    if args.dataset_dir is None:
        dataset_dir, default_exp_name = _retrain._infer_defaults_from_train_config(args.train_config)  # noqa: SLF001
    else:
        dataset_dir = args.dataset_dir
        default_exp_name = args.train_config

    train_cfg = _config.get_config(args.train_config)
    output_dataset = args.output_dataset or _default_output_dataset(dataset_dir, keep_ratio, args.seed)
    keep_episode_indices = _select_random_episode_indices(dataset_dir, keep_ratio, args.seed)
    total_episodes = len(_load_episode_indices(dataset_dir))

    exp_name = args.exp_name or f"{default_exp_name}_random_keep{int(round(keep_ratio * 100))}_seed{args.seed}"

    print(f"Train config: {args.train_config}")
    print(f"Dataset: {dataset_dir}")
    print(
        f"Random episode selection: seed={args.seed}, "
        f"keep_ratio={keep_ratio:.4f}, kept={len(keep_episode_indices)}/{total_episodes}"
    )
    print(f"Writing filtered dataset to: {output_dataset}")

    _retrain._materialize_subset(  # noqa: SLF001
        dataset_dir,
        output_dataset,
        keep_episode_indices,
        overwrite=args.overwrite,
    )
    _write_selection_summary(
        output_dataset,
        source_dataset=dataset_dir,
        keep_ratio=keep_ratio,
        seed=args.seed,
        selected_episode_indices=keep_episode_indices,
        total_episodes=total_episodes,
    )
    print("Filtered dataset materialized.")

    print("Recomputing norm_stats.json for filtered dataset...")
    _retrain_grass._recompute_norm_stats(train_cfg, output_dataset, args.max_frames_for_norm_stats)  # noqa: SLF001
    print(f"Filtered norm_stats written to: {output_dataset / 'norm_stats.json'}")

    if args.prepare_only:
        return

    retrain_cfg = _retrain_grass._build_train_config(  # noqa: SLF001
        train_cfg,
        output_dataset,
        exp_name=exp_name,
        checkpoint_base_dir=args.checkpoint_base_dir,
        overwrite=args.overwrite,
        num_train_steps=args.num_train_steps,
    )
    retrain_cfg = dataclasses.replace(retrain_cfg, wandb_enabled=args.enable_wandb)
    print(
        f"Starting retraining: exp_name={retrain_cfg.exp_name}, "
        f"checkpoint_dir={retrain_cfg.checkpoint_dir}, "
        f"dataset={output_dataset}, "
        f"wandb_enabled={retrain_cfg.wandb_enabled}"
    )
    train_script.main(retrain_cfg)


if __name__ == "__main__":
    main()
