#!/usr/bin/env python3
"""Randomly mask sample-level state-action pairs, recompute norm_stats, and retrain.

This script keeps the existing episode-level random retraining flow untouched and
adds a separate sample-level baseline:

1. Resolve the base training config and effective dataset.
2. Collect candidate sample indices from the active training dataset.
3. Randomly mask/drop a target ratio of state-action pairs with a reproducible RNG seed.
4. Write comparison artifacts under a dedicated compare directory.
5. Recompute `norm_stats.json` on the kept sample subset.
6. Launch training against the original dataset using the kept sample indices.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import numpy as np

import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import retrain as _retrain
import retrain_window_joint as _sample_subset_retrain
from scripts import train as train_script


def _resolve_keep_ratio(
    keep_ratio: float | None,
    mask_ratio: float | None,
    drop_ratio: float | None,
) -> float:
    specified = [value is not None for value in (keep_ratio, mask_ratio, drop_ratio)]
    if sum(specified) > 1:
        raise ValueError("Only one of --keep-ratio, --mask-ratio, or --drop-ratio can be set.")
    if keep_ratio is not None:
        if not 0.0 < keep_ratio <= 1.0:
            raise ValueError("--keep-ratio must be in (0,1].")
        return float(keep_ratio)
    effective_drop_ratio = mask_ratio if mask_ratio is not None else drop_ratio
    if effective_drop_ratio is not None:
        if not 0.0 <= effective_drop_ratio < 1.0:
            raise ValueError("--mask-ratio/--drop-ratio must be in [0,1).")
        return 1.0 - float(effective_drop_ratio)
    return 0.8


def _default_output_dir(dataset_dir: Path, keep_ratio: float, seed: int) -> Path:
    keep_pct = int(round(keep_ratio * 100))
    compare_root = dataset_dir.parent / f"{dataset_dir.name}_compare"
    return compare_root / f"{dataset_dir.name}_random_mask_keep{keep_pct}_seed{seed}"


def _default_output_indices(output_dir: Path) -> Path:
    return output_dir / "kept_sample_indices.npy"


def _override_dataset_dir(train_cfg: _config.TrainConfig, dataset_dir: Path) -> _config.TrainConfig:
    if getattr(train_cfg.data, "repo_id", None) == str(dataset_dir):
        return train_cfg
    return dataclasses.replace(train_cfg, data=dataclasses.replace(train_cfg.data, repo_id=str(dataset_dir)))


def _load_candidate_sample_indices(
    train_cfg: _config.TrainConfig,
) -> tuple[np.ndarray, int, str | None]:
    data_cfg = train_cfg.data.create(train_cfg.assets_dirs, train_cfg.model)
    if data_cfg.rlds_data_dir is not None:
        raise ValueError("Random sample masking currently supports LeRobot datasets only.")
    if data_cfg.repo_id is None:
        raise ValueError("Training config does not define a dataset repo_id.")

    raw_data_cfg = dataclasses.replace(data_cfg, sample_indices_path=None)
    raw_dataset = _data_loader.create_torch_dataset(
        raw_data_cfg,
        train_cfg.model.action_horizon,
        train_cfg.model,
    )
    total_dataset_samples = len(raw_dataset)

    if data_cfg.sample_indices_path is not None:
        candidate_sample_indices = np.asarray(
            _data_loader._load_sample_indices(data_cfg.sample_indices_path),  # noqa: SLF001
            dtype=np.int64,
        ).reshape(-1)
        source_indices_path = data_cfg.sample_indices_path
    else:
        candidate_sample_indices = np.arange(total_dataset_samples, dtype=np.int64)
        source_indices_path = None

    if candidate_sample_indices.size == 0:
        raise ValueError("No candidate samples found for random masking.")
    return candidate_sample_indices, total_dataset_samples, source_indices_path


def _select_random_sample_subset(
    candidate_sample_indices: np.ndarray,
    keep_ratio: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    candidate_sample_indices = np.asarray(candidate_sample_indices, dtype=np.int64).reshape(-1)
    if candidate_sample_indices.size == 0:
        raise ValueError("Cannot sample from an empty candidate set.")

    keep_count = max(1, int(candidate_sample_indices.size * keep_ratio))
    rng = np.random.default_rng(seed)
    kept_positions = np.sort(rng.choice(candidate_sample_indices.size, size=keep_count, replace=False))
    keep_mask = np.zeros(candidate_sample_indices.shape[0], dtype=np.bool_)
    keep_mask[kept_positions] = True

    kept_sample_indices = np.sort(candidate_sample_indices[keep_mask])
    dropped_sample_indices = np.sort(candidate_sample_indices[~keep_mask])
    return kept_sample_indices, dropped_sample_indices, keep_mask


def _write_compare_artifacts(
    output_dir: Path,
    *,
    source_dataset: Path,
    keep_ratio: float,
    seed: int,
    total_dataset_samples: int,
    candidate_sample_indices: np.ndarray,
    kept_sample_indices: np.ndarray,
    dropped_sample_indices: np.ndarray,
    keep_mask: np.ndarray,
    source_indices_path: str | None,
    output_indices: Path,
    overwrite: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_paths = [
        output_dir / "candidate_sample_indices.npy",
        output_dir / "kept_sample_indices.npy",
        output_dir / "dropped_sample_indices.npy",
        output_dir / "keep_mask.npy",
        output_dir / "random_mask_selection_summary.json",
    ]
    if not overwrite:
        existing = [path for path in artifact_paths if path.exists()]
        if existing:
            raise FileExistsError(
                "Comparison artifacts already exist; pass --overwrite to reuse them: "
                + ", ".join(str(path) for path in existing)
            )

    np.save(output_dir / "candidate_sample_indices.npy", np.asarray(candidate_sample_indices, dtype=np.int64))
    np.save(output_dir / "kept_sample_indices.npy", np.asarray(kept_sample_indices, dtype=np.int64))
    np.save(output_dir / "dropped_sample_indices.npy", np.asarray(dropped_sample_indices, dtype=np.int64))
    np.save(output_dir / "keep_mask.npy", np.asarray(keep_mask, dtype=np.bool_))

    payload = {
        "selection_type": "random_sample_mask",
        "unit": "state_action_pair",
        "source_dataset": str(source_dataset),
        "source_sample_indices_path": source_indices_path,
        "output_dir": str(output_dir),
        "output_indices": str(output_indices),
        "seed": int(seed),
        "keep_ratio": float(keep_ratio),
        "mask_ratio": float(1.0 - keep_ratio),
        "total_dataset_samples": int(total_dataset_samples),
        "candidate_samples": int(candidate_sample_indices.size),
        "kept_samples": int(kept_sample_indices.size),
        "dropped_samples": int(dropped_sample_indices.size),
    }
    (output_dir / "random_mask_selection_summary.json").write_text(json.dumps(payload, indent=2) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Randomly mask a ratio of sample-level state-action pairs, recompute norm_stats, "
            "and retrain OpenPI without changing the existing episode-level random pipeline."
        )
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
        help="Fraction of sample-level state-action pairs to keep.",
    )
    parser.add_argument(
        "--mask-ratio",
        type=float,
        default=None,
        help="Fraction of sample-level state-action pairs to mask/drop.",
    )
    parser.add_argument(
        "--drop-ratio",
        type=float,
        default=None,
        help="Alias of --mask-ratio.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed used for sample masking.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Comparison output directory for mask artifacts and recomputed norm_stats.",
    )
    parser.add_argument(
        "--output-indices",
        type=Path,
        default=None,
        help="Optional output `.npy` path for the kept sample indices.",
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
        help="Only build the compare artifacts and recompute norm_stats, skip training.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing compare artifacts and checkpoint directory.",
    )
    parser.add_argument(
        "--enable-wandb",
        action="store_true",
        help="Enable Weights & Biases logging. Default is disabled.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    keep_ratio = _resolve_keep_ratio(args.keep_ratio, args.mask_ratio, args.drop_ratio)

    if args.dataset_dir is None:
        dataset_dir, default_exp_name = _retrain._infer_defaults_from_train_config(args.train_config)  # noqa: SLF001
    else:
        dataset_dir = args.dataset_dir
        default_exp_name = args.train_config

    base_train_cfg = _config.get_config(args.train_config)
    train_cfg = _override_dataset_dir(base_train_cfg, dataset_dir)

    candidate_sample_indices, total_dataset_samples, source_indices_path = _load_candidate_sample_indices(train_cfg)
    kept_sample_indices, dropped_sample_indices, keep_mask = _select_random_sample_subset(
        candidate_sample_indices,
        keep_ratio,
        args.seed,
    )
    if kept_sample_indices.size == 0:
        raise ValueError("Random masking resulted in zero kept samples.")

    output_dir = args.output_dir or _default_output_dir(dataset_dir, keep_ratio, args.seed)
    output_indices = args.output_indices or _default_output_indices(output_dir)
    exp_name = args.exp_name or f"{default_exp_name}_random_mask_keep{int(round(keep_ratio * 100))}_seed{args.seed}"
    if output_indices != output_dir / "kept_sample_indices.npy" and output_indices.exists() and not args.overwrite:
        raise FileExistsError(f"{output_indices} already exists; pass --overwrite to reuse it")

    _write_compare_artifacts(
        output_dir,
        source_dataset=dataset_dir,
        keep_ratio=keep_ratio,
        seed=args.seed,
        total_dataset_samples=total_dataset_samples,
        candidate_sample_indices=candidate_sample_indices,
        kept_sample_indices=kept_sample_indices,
        dropped_sample_indices=dropped_sample_indices,
        keep_mask=keep_mask,
        source_indices_path=source_indices_path,
        output_indices=output_indices,
        overwrite=args.overwrite,
    )
    if output_indices != output_dir / "kept_sample_indices.npy":
        _retrain._save_sample_indices(output_indices, kept_sample_indices, overwrite=args.overwrite)  # noqa: SLF001

    print(f"Train config: {args.train_config}")
    print(f"Dataset: {dataset_dir}")
    print(
        "Random sample masking: "
        f"seed={args.seed}, keep_ratio={keep_ratio:.4f}, mask_ratio={1.0 - keep_ratio:.4f}, "
        f"kept={kept_sample_indices.size}/{candidate_sample_indices.size}"
    )
    print(f"Compare dir: {output_dir}")
    print(f"Sample indices: {output_indices}")

    print("Recomputing norm_stats.json for randomly kept sample subset...")
    _sample_subset_retrain._recompute_norm_stats_for_sample_subset(  # noqa: SLF001
        train_cfg,
        sample_indices_path=output_indices,
        output_dir=output_dir,
        max_frames=args.max_frames_for_norm_stats,
    )
    print(f"Subset norm_stats written to: {output_dir / 'norm_stats.json'}")

    if args.prepare_only:
        return

    retrain_cfg = _sample_subset_retrain._build_train_config(  # noqa: SLF001
        train_cfg,
        sample_indices_path=output_indices,
        output_dir=output_dir,
        exp_name=exp_name,
        checkpoint_base_dir=args.checkpoint_base_dir,
        overwrite=args.overwrite,
        num_train_steps=args.num_train_steps,
    )
    retrain_cfg = dataclasses.replace(retrain_cfg, wandb_enabled=args.enable_wandb)
    print(
        f"Starting retraining: exp_name={retrain_cfg.exp_name}, "
        f"checkpoint_dir={retrain_cfg.checkpoint_dir}, "
        f"sample_indices={output_indices}"
    )
    train_script.main(retrain_cfg)


if __name__ == "__main__":
    main()
