#!/usr/bin/env python3
"""GraSS influence attribution with window-level and episode-level aggregation.

This script does not modify the existing GraSS implementation. It reuses the
sample-level GraSS pipeline from `scripts/influence_pi05_lora_grass.py`, then
adds fixed-size window aggregation on top of the sample scores.



cd /workspace/liushuo/cupid/openpi

  mkdir -p /workspace/liushuo/cupid/openpi/tmp/transport_mh_grass_window

  CUDA_VISIBLE_DEVICES=5 nohup bash -lc '
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 WANDB_MODE=disabled \
  uv run python scripts/influence_pi05_lora_grass_window.py \
    --config-name pi05_transport_mh_lora \
    --checkpoint-dir /workspace/liushuo/cupid/openpi/checkpoints/pi05_transport_mh_lora/transport_mh_lora/9999 \
    --eval-rollouts-dir /workspace/liushuo/robomimic/robomimic/eval/tmp/transport_openpi05_24.0/episodes\
    --output-path /workspace/liushuo/cupid/openpi/tmp/transport_mh_grass_window/influence_grass_window.pkl \
    --sparsification random_mask-32*32 \
    --projection sjlt-1024 \
    --hessian eFIM \
    --damping 1e-4 \
    --stage1-microbatch-size 1 \
    --stage2-batch-size 32 \
    --window-size 50 \
    --window-stride 50 \
    --progress-every 1
  ' > /workspace/liushuo/cupid/openpi/tmp/transport_mh_grass_window/influence_grass_window.log 2>&1 &

Default behavior:
1. Compute GraSS sample-level train influences.
2. Split each train episode into non-overlapping windows with `T=50`.
3. Keep the last incomplete window of each episode.
4. Aggregate sample scores into:
   - window-level influence scores
   - window-level quality scores
   - episode-level influence scores
   - episode-level quality scores

By default:
- `influence_scores` is the window-level influence score.
- `window_influence_scores` uses sum aggregation.
- `window_quality_scores` uses z-score normalization over window mean scores.
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import logging
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

import openpi.training.data_loader as _data_loader

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts import influence_pi05_lora_grass as grass_script

LOGGER = logging.getLogger("influence_pi05_lora_grass_window")


@dataclass
class WindowGrassConfig(grass_script.GrassConfig):
    window_size: int = 50
    window_stride: int | None = None
    window_aggregation: str = "sum"
    quality_score: str = "zscore_mean"


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


class WindowGrassOpenPIInfluence(grass_script.GrassOpenPIInfluence):
    def __init__(self, cfg: WindowGrassConfig):
        cfg.score_level = "sample"
        if cfg.window_size <= 0:
            raise ValueError("--window-size must be > 0")
        if cfg.window_stride is None:
            cfg.window_stride = cfg.window_size
        if int(cfg.window_stride) <= 0:
            raise ValueError("--window-stride must be > 0")
        if cfg.window_aggregation not in {"sum", "mean"}:
            raise ValueError("--window-aggregation must be one of: sum, mean")
        if cfg.quality_score not in {"mean", "zscore_mean", "sum", "zscore_sum"}:
            raise ValueError("--quality-score must be one of: mean, zscore_mean, sum, zscore_sum")
        super().__init__(cfg)
        self.cfg = cfg

    def _get_train_sample_metadata_full(self, total_samples: int) -> dict[str, np.ndarray | int]:
        data_cfg = self.train_loader.data_config()
        if data_cfg.rlds_data_dir is not None:
            raise NotImplementedError("Window-level GraSS aggregation is not supported for RLDS datasets.")

        dataset = _data_loader.create_torch_dataset(
            data_cfg,
            self.train_cfg.model.action_horizon,
            self.train_cfg.model,
        )
        total_dataset_samples = int(len(dataset))
        if total_samples > total_dataset_samples:
            raise ValueError(
                f"Scored sample count {total_samples} exceeds dataset size {total_dataset_samples}. "
                "This script expects at most one full sequential dataset pass."
            )

        dataset_positions = np.arange(total_samples, dtype=np.int64)
        dataset_sample_indices = np.empty((total_samples,), dtype=np.int64)
        episode_indices = np.empty((total_samples,), dtype=np.int64)
        frame_indices = np.empty((total_samples,), dtype=np.int64)

        for i, dataset_pos in enumerate(dataset_positions.tolist()):
            item = dataset[dataset_pos]
            if "episode_index" not in item:
                raise KeyError("Train dataset item does not contain episode_index.")
            if "frame_index" not in item:
                raise KeyError("Train dataset item does not contain frame_index.")
            episode_indices[i] = _to_scalar_int(item["episode_index"])
            frame_indices[i] = _to_scalar_int(item["frame_index"])
            if "index" in item:
                dataset_sample_indices[i] = _to_scalar_int(item["index"])
            else:
                dataset_sample_indices[i] = dataset_pos

        return {
            "dataset_positions": dataset_positions,
            "dataset_sample_indices": dataset_sample_indices,
            "episode_indices": episode_indices,
            "frame_indices": frame_indices,
            "total_dataset_samples": total_dataset_samples,
        }

    def _aggregate_windows(
        self,
        sample_scores: np.ndarray,
        sample_meta: dict[str, np.ndarray | int],
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

        stride = int(self.cfg.window_stride)
        window_size = int(self.cfg.window_size)
        global_window_idx = 0

        for episode_idx in sorted(np.unique(episode_indices).tolist()):
            episode_sample_ids = np.where(episode_indices == episode_idx)[0]
            order = np.argsort(frame_indices[episode_sample_ids], kind="stable")
            ordered_sample_ids = episode_sample_ids[order]

            local_window_idx = 0
            for start in range(0, len(ordered_sample_ids), stride):
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
                window_influence_scores.append(_reduce(current_scores, self.cfg.window_aggregation))

                global_window_idx += 1
                local_window_idx += 1

        window_sum_scores_np = np.asarray(window_sum_scores, dtype=np.float32)
        window_mean_scores_np = np.asarray(window_mean_scores, dtype=np.float32)
        if self.cfg.quality_score == "mean":
            window_quality_scores = window_mean_scores_np
        elif self.cfg.quality_score == "zscore_mean":
            window_quality_scores = _zscore(window_mean_scores_np)
        elif self.cfg.quality_score == "sum":
            window_quality_scores = window_sum_scores_np
        else:
            window_quality_scores = _zscore(window_sum_scores_np)

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

    def _aggregate_episodes(
        self,
        sample_scores: np.ndarray,
        window_data: dict[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
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

    def run(self) -> dict[str, Any]:
        cache_train = self.cache_dir / f"{self.output_path.stem}_stage1_train_grads.pkl"
        cache_precond = self.cache_dir / f"{self.output_path.stem}_stage2_preconditioners.pkl"
        cache_eval = self.cache_dir / f"{self.output_path.stem}_stage3_eval_return_grad.pkl"

        if cache_train.exists():
            with open(cache_train, "rb") as f:
                train_grads = pickle.load(f)
            LOGGER.info("Loaded cached train grads from %s", cache_train)
        else:
            train_grads = self._cache_train_grads()
            with open(cache_train, "wb") as f:
                pickle.dump(train_grads, f, protocol=pickle.HIGHEST_PROTOCOL)

        if cache_precond.exists():
            with open(cache_precond, "rb") as f:
                preconditioners = pickle.load(f)
            LOGGER.info("Loaded cached preconditioners from %s", cache_precond)
        else:
            preconditioners = self._compute_preconditioners(train_grads)
            with open(cache_precond, "wb") as f:
                pickle.dump(preconditioners, f, protocol=pickle.HIGHEST_PROTOCOL)

        if cache_eval.exists():
            with open(cache_eval, "rb") as f:
                eval_grad = pickle.load(f)
            LOGGER.info("Loaded cached eval return gradient from %s", cache_eval)
        else:
            eval_grad = self._compute_eval_return_grad()
            with open(cache_eval, "wb") as f:
                pickle.dump(eval_grad, f, protocol=pickle.HIGHEST_PROTOCOL)

        sample_scores = self._score(train_grads, preconditioners, eval_grad)
        sample_meta = self._get_train_sample_metadata_full(int(sample_scores.shape[0]))
        window_data = self._aggregate_windows(sample_scores, sample_meta)
        episode_data = self._aggregate_episodes(sample_scores, window_data)

        result = {
            "config": dataclasses.asdict(self.cfg),
            "score_level": "window",
            "window_size": int(self.cfg.window_size),
            "window_stride": int(self.cfg.window_stride),
            "window_aggregation": self.cfg.window_aggregation,
            "quality_score": self.cfg.quality_score,
            "influence_scores": np.asarray(window_data["window_influence_scores"], dtype=np.float32),
            "sample_influence_scores": np.asarray(sample_scores, dtype=np.float32),
            "window_influence_scores": np.asarray(window_data["window_influence_scores"], dtype=np.float32),
            "window_sum_influence_scores": np.asarray(window_data["window_sum_influence_scores"], dtype=np.float32),
            "window_mean_influence_scores": np.asarray(window_data["window_mean_influence_scores"], dtype=np.float32),
            "window_quality_scores": np.asarray(window_data["window_quality_scores"], dtype=np.float32),
            "episode_influence_scores": np.asarray(episode_data["episode_influence_scores"], dtype=np.float32),
            "episode_mean_influence_scores": np.asarray(
                episode_data["episode_mean_influence_scores"], dtype=np.float32
            ),
            "episode_quality_scores": np.asarray(episode_data["episode_quality_scores"], dtype=np.float32),
            "episode_quality_raw_scores": np.asarray(episode_data["episode_quality_raw_scores"], dtype=np.float32),
            "train_grads": train_grads,
            "eval_return_grad": eval_grad,
            "leaf_dims": {name: state.output_dim for name, state in self._leaf_state_map().items()},
            "metadata": {
                "leaf_names": np.asarray(self.leaf_names, dtype=object),
                "leaf_shapes": np.asarray(self.leaf_shapes, dtype=object),
                "episode_rewards": self.eval_dataset.episode_rewards,
                "episode_lengths": self.eval_dataset.episode_lengths,
                "train_window_indices": np.asarray(window_data["window_indices"], dtype=np.int64),
                "train_window_episode_indices": np.asarray(window_data["window_episode_indices"], dtype=np.int64),
                "train_window_local_indices": np.asarray(window_data["window_local_indices"], dtype=np.int64),
                "train_window_frame_starts": np.asarray(window_data["window_frame_starts"], dtype=np.int64),
                "train_window_frame_ends": np.asarray(window_data["window_frame_ends"], dtype=np.int64),
                "train_window_sample_counts": np.asarray(window_data["window_sample_counts"], dtype=np.int64),
                "train_window_is_partial": np.asarray(window_data["window_is_partial"], dtype=np.bool_),
                "train_sample_window_indices": np.asarray(window_data["sample_window_indices"], dtype=np.int64),
                "train_sample_window_local_indices": np.asarray(
                    window_data["sample_window_local_indices"], dtype=np.int64
                ),
                "train_sample_episode_indices": np.asarray(window_data["sample_episode_indices"], dtype=np.int64),
                "train_sample_frame_indices": np.asarray(window_data["sample_frame_indices"], dtype=np.int64),
                "train_dataset_sample_indices": np.asarray(
                    window_data["sample_dataset_sample_indices"], dtype=np.int64
                ),
                "train_dataset_positions": np.asarray(window_data["sample_dataset_positions"], dtype=np.int64),
                "train_episode_indices": np.asarray(episode_data["episode_indices"], dtype=np.int64),
                "train_episode_sample_counts": np.asarray(episode_data["episode_sample_counts"], dtype=np.int64),
                "train_episode_window_counts": np.asarray(episode_data["episode_window_counts"], dtype=np.int64),
                "train_score_aggregation": self.cfg.window_aggregation,
                "total_dataset_samples": int(sample_meta["total_dataset_samples"]),
            },
        }
        if self.cfg.save_preconditioners:
            result["preconditioners"] = preconditioners

        with open(self.output_path, "wb") as f:
            pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
        LOGGER.info(
            "Saved window-level GraSS result to %s (samples=%d, windows=%d, episodes=%d)",
            self.output_path,
            int(sample_scores.shape[0]),
            int(result["window_influence_scores"].shape[0]),
            int(result["episode_influence_scores"].shape[0]),
        )
        return result


def parse_args() -> WindowGrassConfig:
    parser = argparse.ArgumentParser(
        description="GraSS-style OpenPI05 LoRA influence attribution with window aggregation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config-name", required=True, help="OpenPI training config name")
    parser.add_argument("--checkpoint-dir", required=True, help="Checkpoint directory containing params/")
    parser.add_argument("--eval-rollouts-dir", required=True, help="Directory containing eval rollout PKLs")
    parser.add_argument("--output-path", required=True, help="Output pickle path")

    parser.add_argument("--batch-size", type=int, default=8, help="Unused placeholder for parity with other scripts")
    parser.add_argument("--num-workers", type=int, default=0, help="Unused placeholder for parity with other scripts")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--max-train-batches", type=int, default=None, help="Max Stage1 train batches")
    parser.add_argument("--max-eval-episodes", type=int, default=None, help="Max eval episodes for Stage3")
    parser.add_argument(
        "--max-eval-steps-per-episode",
        type=int,
        default=None,
        help="Max steps per eval episode for Stage3",
    )
    parser.add_argument(
        "--stage1-microbatch-size",
        type=int,
        default=1,
        help="Per-sample grad microbatch size for Stage1; lower uses less memory",
    )
    parser.add_argument("--stage2-batch-size", type=int, default=32, help="Eval rollout chunk size")
    parser.add_argument("--progress-every", type=int, default=1, help="Progress logging interval")

    parser.add_argument(
        "--sparsification",
        type=str,
        default="random_mask-32*32",
        help="GraSS stage-1 factorized compression, e.g. random_mask-32*32",
    )
    parser.add_argument(
        "--projection",
        type=str,
        default="sjlt-1024",
        help="GraSS stage-2 projection, e.g. sjlt-1024 or identity",
    )
    parser.add_argument(
        "--hessian",
        type=str,
        default="eFIM",
        choices=("eFIM", "Identity"),
        help="Compressed-space Hessian approximation",
    )
    parser.add_argument("--damping", type=float, default=1e-4, help="Preconditioner damping")
    parser.add_argument("--save-preconditioners", action="store_true", help="Include preconditioners in final output")
    parser.add_argument("--stage-cache-prefix", type=str, default=None, help="Optional stage cache directory")

    parser.add_argument("--window-size", type=int, default=50, help="Window size within each train episode")
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
        help="How to compute quality scores for windows before episode aggregation.",
    )

    args = parser.parse_args()
    return WindowGrassConfig(
        config_name=args.config_name,
        checkpoint_dir=args.checkpoint_dir,
        eval_rollouts_dir=args.eval_rollouts_dir,
        output_path=args.output_path,
        score_level="sample",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        max_train_batches=args.max_train_batches,
        max_eval_episodes=args.max_eval_episodes,
        max_eval_steps_per_episode=args.max_eval_steps_per_episode,
        stage1_microbatch_size=args.stage1_microbatch_size,
        stage2_batch_size=args.stage2_batch_size,
        progress_every=args.progress_every,
        sparsification=args.sparsification,
        projection=args.projection,
        hessian=args.hessian,
        damping=args.damping,
        save_preconditioners=args.save_preconditioners,
        stage_cache_prefix=args.stage_cache_prefix,
        window_size=args.window_size,
        window_stride=args.window_stride,
        window_aggregation=args.window_aggregation,
        quality_score=args.quality_score,
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        force=True,
    )
    cfg = parse_args()
    runner = WindowGrassOpenPIInfluence(cfg)
    runner.run()


if __name__ == "__main__":
    main()
