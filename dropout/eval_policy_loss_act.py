from __future__ import annotations

"""Evaluate ACT/LeRobot diffusion-policy losses and derive influence scores.

This script mirrors ``eval_policy_loss_lerobot.py`` for computing per-sample
losses, but extends it with three lightweight influence heuristics:

1. ``train_loss_only`` – ranks demonstrations using only their own losses.
2. ``train_test_loss`` – ranks demos by average loss discrepancy to each test rollout.
3. ``reward_weighted`` – weights the discrepancy by rollout rewards (J).

Both dataset and policy loading follow the LeRobot helpers used elsewhere in
CUPID.  Influence scores are stored under the same evaluation directory as the
loss artifacts, so downstream pipelines can reuse them.
"""

# Allow execution from any working directory without installing this checkout.
import sys as _prose_sys
from pathlib import Path as _ProsePath
_prose_sys.path.insert(0, str(_ProsePath(__file__).resolve().parents[1]))


import sys

# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode="w", buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode="w", buffering=1)

import pathlib
import pickle
import random
from typing import Any, Dict, List, Optional

import click
import numpy as np
import torch
from numpy.lib.format import open_memmap
from torch.utils.data import DataLoader

from act import scoring_utils as scoring
from scripts.lerobot.train_trak_lerobot import (
    EvalPklEpisodeDataset,
    IMAGE_KEYS,
    LerobotDiffusionModelOutput,
    LerobotTraKDataset,
    load_lerobot_policy,
    build_lerobot_dataset,
    prepare_batch,
)


def _build_dataloader(
    dataset: LerobotTraKDataset,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        persistent_workers=False,
        pin_memory=True,
        shuffle=False,
    )


def _format_timesteps(num_samples: int, num_timesteps: int, max_steps: int, device: torch.device) -> torch.Tensor:
    return torch.randint(
        low=0,
        high=max_steps,
        size=(num_samples, num_timesteps),
        device=device,
    )


def _build_dataset_metadata(dataset_path: Optional[str]) -> Dict[str, Any]:
    if not dataset_path:
        return {"num_eps": 0, "num_samples": 0, "ep_lens": np.array([], dtype=np.int64), "ep_idxs": []}
    lengths = scoring.read_episode_lengths(pathlib.Path(dataset_path))
    return scoring.build_metadata_from_lengths(lengths)


def _sample_to_trajectory_scores(
    sample_scores: np.ndarray,
    metadata: Dict[str, Any],
) -> np.ndarray:
    """Aggregate per-sample scores into per-trajectory scores via averaging."""

    if metadata["num_eps"] == 0:
        return np.zeros(0, dtype=sample_scores.dtype)

    assert len(sample_scores) == metadata["ep_lens"].sum(), "Sample count mismatch for aggregation."

    traj_scores = np.zeros(metadata["num_eps"], dtype=np.float64)
    for idx, ep_idx in enumerate(metadata["ep_idxs"]):
        traj_scores[idx] = float(sample_scores[ep_idx].mean())
    return traj_scores.astype(sample_scores.dtype)


def _compute_reward_weights(test_dataset: EvalPklEpisodeDataset) -> np.ndarray:
    """Return normalized per-episode rewards (J) to weight influence scores."""

    rewards: List[float] = []
    for episode in test_dataset._episodes:
        reward = 0.0
        if hasattr(episode, "columns") and "reward" in episode.columns:
            reward = float(np.asarray(episode["reward"], dtype=np.float64).sum())
        elif hasattr(episode, "columns") and "success" in episode.columns:
            reward = float(episode["success"].iloc[0])
        else:
            reward = float(episode.iloc[0].to_dict().get("success", 0.0))
        rewards.append(reward)

    weights = np.asarray(rewards, dtype=np.float64)
    if weights.size == 0:
        return weights
    max_abs = np.max(np.abs(weights))
    if max_abs == 0:
        return np.ones_like(weights)
    return weights / max_abs


def _compute_influence_variants(
    demo_losses: np.ndarray,
    test_losses: np.ndarray,
    reward_weights: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Compute the three requested influence variants for a dataset split."""

    if demo_losses.size == 0:
        return {
            "train_loss_only": np.zeros(0, dtype=np.float32),
            "train_test_loss": np.zeros(0, dtype=np.float32),
            "reward_weighted": np.zeros(0, dtype=np.float32),
        }

    train_loss_only = (-demo_losses).astype(np.float32)

    if test_losses.size == 0:
        zero_like = np.zeros_like(train_loss_only)
        return {
            "train_loss_only": train_loss_only,
            "train_test_loss": zero_like,
            "reward_weighted": zero_like,
        }

    pairwise_diff = np.abs(demo_losses[:, None] - test_losses[None, :])
    train_test_loss = (-pairwise_diff.mean(axis=1)).astype(np.float32)

    weights = reward_weights
    if weights.size == 0:
        weights = np.ones(test_losses.shape[0], dtype=np.float64)
    if weights.shape[0] != test_losses.shape[0]:
        raise ValueError("Reward weights must align with test episodes.")

    norm = np.linalg.norm(weights, ord=1)
    if norm == 0:
        norm = 1.0
    normalized = weights / norm
    reward_weighted = (-pairwise_diff @ normalized).astype(np.float32)

    return {
        "train_loss_only": train_loss_only,
        "train_test_loss": train_test_loss,
        "reward_weighted": reward_weighted,
    }


@click.command()
@click.option("--exp_name", type=str, required=True, help="Name of the loss evaluation experiment.")
@click.option(
    "--eval_dir",
    type=str,
    default=None,
    help="Directory containing rollout episodes under `episodes/` (required unless --train_only).",
)
@click.option(
    "--output_dir",
    type=str,
    default=None,
    help="Directory to store loss artifacts (defaults to eval_dir).",
)
@click.option("--pretrained_dir", type=str, required=True, help="Directory with `config.json` and `model.safetensors`.")
@click.option("--train_dataset_path", type=str, required=True, help="Path to the LeRobot training dataset.")
@click.option("--holdout_dataset_path", type=str, default=None, help="Optional path to the holdout dataset.")
@click.option("--batch_size", type=int, default=4)
@click.option("--num_timesteps", type=int, default=8, help="Number of diffusion timesteps to average per sample.")
@click.option(
    "--loss_fn",
    type=click.Choice(tuple(LerobotDiffusionModelOutput.LOSS_FNS)),
    default="ddpm",
    help="Loss variant applied when evaluating the diffusion policy.",
)
@click.option("--use_half_precision", type=bool, default=True)
@click.option("--compute_holdout", type=bool, default=False)
@click.option("--num_workers", type=int, default=4, help="Number of workers for train/holdout dataloaders.")
@click.option("--device", type=str, default="cuda:0")
@click.option("--seed", type=int, default=0)
@click.option("--overwrite", type=bool, default=False)
@click.option("--train_only", type=bool, default=False, help="Skip rollout evaluation and compute train losses only.")
def main(
    exp_name: str,
    eval_dir: Optional[str],
    output_dir: Optional[str],
    pretrained_dir: str,
    train_dataset_path: str,
    holdout_dataset_path: Optional[str],
    batch_size: int,
    num_timesteps: int,
    loss_fn: str,
    use_half_precision: bool,
    compute_holdout: bool,
    num_workers: int,
    device: str,
    seed: int,
    overwrite: bool,
    train_only: bool,
) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    device_t = torch.device(device)

    pretrained_path = pathlib.Path(pretrained_dir).expanduser()
    if pretrained_path.is_file():
        pretrained_path = pretrained_path.parent

    policy, cfg = load_lerobot_policy(pretrained_path, device_t)
    noise_steps = policy.diffusion.noise_scheduler.config.num_train_timesteps
    loss_model = LerobotDiffusionModelOutput(loss_fn=loss_fn, image_keys=IMAGE_KEYS)

    def _prepare_dict_loader(dataset_root: str) -> tuple[LerobotTraKDataset, DataLoader]:
        base_dataset = build_lerobot_dataset(pathlib.Path(dataset_root), cfg)
        wrapped = LerobotTraKDataset(base_dataset)
        return wrapped, _build_dataloader(wrapped, batch_size, num_workers)

    train_dataset, train_loader = _prepare_dict_loader(train_dataset_path)
    train_set_size = len(train_dataset)

    if compute_holdout:
        if not holdout_dataset_path:
            raise ValueError("Provide --holdout_dataset_path when compute_holdout=True")
        holdout_dataset, holdout_loader = _prepare_dict_loader(holdout_dataset_path)
        holdout_set_size = len(holdout_dataset)
    else:
        holdout_dataset = None
        holdout_loader = None
        holdout_set_size = 0

    if not train_only and not eval_dir:
        raise ValueError("Provide --eval_dir unless --train_only=True")
    if not (output_dir or eval_dir):
        raise ValueError("Provide --output_dir or --eval_dir to store loss artifacts.")
    output_root = pathlib.Path(output_dir or eval_dir).expanduser()

    if train_only:
        test_set = None
        test_loader = None
        test_set_size = 0
    else:
        eval_path = pathlib.Path(eval_dir).expanduser() / "episodes"
        test_set = EvalPklEpisodeDataset(
            dataset_path=eval_path,
            exec_horizon=1,
            sample_history=0,
        )
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)
        test_set_size = len(test_set)

    save_dir = output_root / exp_name
    if save_dir.exists():
        if not overwrite:
            raise ValueError(f"Output path {save_dir} already exists.")
        import shutil

        shutil.rmtree(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    dtype = np.float16 if use_half_precision else np.float32
    train_set_loss = open_memmap(
        filename=save_dir / "train_set_loss.mmap",
        mode="w+",
        shape=(train_set_size + holdout_set_size, 1),
        dtype=dtype,
    )
    test_set_loss = None
    if not train_only:
        test_set_loss = open_memmap(
            filename=save_dir / "test_set_loss.mmap",
            mode="w+",
            shape=(test_set_size, 1),
            dtype=dtype,
        )

    @torch.no_grad()
    def compute_dataset_losses(
        dataloader: DataLoader,
        loss_store: np.memmap,
        start_idx: int,
        dataset_name: str,
    ) -> int:
        last_idx = start_idx
        for batch in dataloader:
            formatted = prepare_batch(batch, cfg.n_obs_steps, cfg.horizon, device_t)
            num_samples = formatted["action"].shape[0]
            formatted["timesteps"] = _format_timesteps(num_samples, num_timesteps, noise_steps, device_t)
            losses = loss_model.get_output(policy, None, None, formatted)
            loss_store[last_idx:last_idx + num_samples, 0] = losses.detach().cpu().numpy().astype(dtype)
            last_idx += num_samples
        return last_idx

    last_index = compute_dataset_losses(train_loader, train_set_loss, 0, "train")
    assert last_index == train_set_size, f"Expected {train_set_size} samples, got {last_index}"
    print(f"Completed computing losses for train set ({train_set_size} samples).")

    if compute_holdout and holdout_dataset is not None and holdout_loader is not None:
        last_index = compute_dataset_losses(holdout_loader, train_set_loss, last_index, "holdout")
        assert last_index == train_set_size + holdout_set_size, "Mismatch in holdout sample count."
        print(f"Completed computing losses for holdout set ({holdout_set_size} samples).")

    if not train_only and test_loader is not None and test_set_loss is not None:
        last_index = compute_dataset_losses(test_loader, test_set_loss, 0, "test")
        assert last_index == test_set_size, f"Expected {test_set_size} test samples, got {last_index}"
        print(f"Completed computing losses for test set ({test_set_size} samples).")

    # ----------------------------------------------------------------------
    # Influence computation (train/holdout) using precomputed losses.
    # ----------------------------------------------------------------------
    train_metadata = _build_dataset_metadata(train_dataset_path)
    holdout_metadata = _build_dataset_metadata(holdout_dataset_path) if compute_holdout else _build_dataset_metadata(
        None
    )
    if train_only or test_set is None:
        test_metadata = _build_dataset_metadata(None)
    else:
        test_metadata = scoring.build_metadata_from_lengths(
            test_set.episode_lengths,
            success_mask=test_set.episode_successes,
        )

    train_holdout_losses = np.array(train_set_loss[:, 0], dtype=np.float32)
    train_losses = train_holdout_losses[: train_metadata["num_samples"]]
    holdout_losses = (
        train_holdout_losses[train_metadata["num_samples"] :]
        if holdout_metadata["num_samples"] > 0
        else np.zeros(0, dtype=np.float32)
    )
    if train_only or test_set_loss is None:
        test_losses = np.zeros(0, dtype=np.float32)
    else:
        test_losses = np.array(test_set_loss[:, 0], dtype=np.float32)

    train_demo_losses = _sample_to_trajectory_scores(train_losses, train_metadata)
    holdout_demo_losses = _sample_to_trajectory_scores(holdout_losses, holdout_metadata)
    test_demo_losses = _sample_to_trajectory_scores(test_losses, test_metadata)
    reward_weights = np.zeros(0, dtype=np.float64) if train_only or test_set is None else _compute_reward_weights(test_set)

    influence_results: Dict[str, Dict[str, np.ndarray]] = {}
    influence_results["train"] = _compute_influence_variants(train_demo_losses, test_demo_losses, reward_weights)
    if holdout_metadata["num_eps"] > 0:
        influence_results["holdout"] = _compute_influence_variants(
            holdout_demo_losses,
            test_demo_losses,
            reward_weights,
        )

    with open(save_dir / "influence_scores.pkl", "wb") as f:
        pickle.dump(influence_results, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Influence scores saved to {save_dir / 'influence_scores.pkl'}")
    print(f"Results saved to {save_dir}")


if __name__ == "__main__":
    main()
