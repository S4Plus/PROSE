from __future__ import annotations

"""Compute LeRobot diffusion-policy losses over train/holdout/test datasets.

This script mirrors ``eval_policy_loss_diffusion.py`` but operates on LeRobot
checkpoints saved as ``model.safetensors``.  The train/holdout datasets are
loaded directly from the provided Hugging Face cache (or local directory),
while the test set corresponds to pickled rollout episodes stored under
``<eval_dir>/episodes`` (as produced by ``eval_save_episodes.py``).

Example:
    python eval_policy_loss_lerobot.py \
        --exp_name default_policy_loss \
        --eval_dir /root/2_250_pkl/train_trak_lerobot_dp \
        --pretrained_dir /root/lerobot_2/outputs2/pretrained_model \
        --train_dataset_path /root/.cache/huggingface/lerobot/ls2/sim_Belt_Pick_Bottle_lerobot \
        --batch_size 4 \
        --num_timesteps 8 \
        --device cuda:0 \
        --compute_holdout False 
"""

# Allow execution from any working directory without installing this checkout.
import sys as _prose_sys
from pathlib import Path as _ProsePath
_prose_sys.path.insert(0, str(_ProsePath(__file__).resolve().parents[2]))


import sys
from typing import TextIO


def _enable_line_buffering(stream: TextIO) -> None:
    """Best-effort line buffering without touching raw file descriptors."""

    try:
        stream.reconfigure(line_buffering=True)
    except (AttributeError, ValueError):  # pragma: no cover - depends on host Python
        pass


_enable_line_buffering(sys.stdout)
_enable_line_buffering(sys.stderr)

import pathlib
import random
import shutil
from typing import Optional

import click
import numpy as np
import torch
from numpy.lib.format import open_memmap
from torch.utils.data import DataLoader

from scripts.lerobot.eval_demonstration_lerobot import load_lerobot_policy
from scripts.lerobot.train_trak_lerobot import (
    EvalPklEpisodeDataset,
    IMAGE_KEYS,
    LerobotDiffusionModelOutput,
    LerobotTraKDataset,
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


@click.command()
@click.option("--exp_name", type=str, required=True, help="Name of the loss evaluation experiment.")
@click.option("--eval_dir", type=str, required=True, help="Directory containing rollout episodes under `episodes/`.")
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
def main(
    exp_name: str,
    eval_dir: str,
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

    eval_path = pathlib.Path(eval_dir).expanduser() / "episodes"
    test_set = EvalPklEpisodeDataset(
        dataset_path=eval_path,
        exec_horizon=1,
        sample_history=0,
    )
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)
    test_set_size = len(test_set)

    save_dir = pathlib.Path(eval_dir).expanduser() / exp_name
    if save_dir.exists():
        if not overwrite:
            raise ValueError(f"Output path {save_dir} already exists.")
        shutil.rmtree(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    dtype = np.float16 if use_half_precision else np.float32
    train_set_loss = open_memmap(
        filename=save_dir / "train_set_loss.mmap",
        mode="w+",
        shape=(train_set_size + holdout_set_size, 1),
        dtype=dtype,
    )
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

    last_index = compute_dataset_losses(test_loader, test_set_loss, 0, "test")
    assert last_index == test_set_size, f"Expected {test_set_size} test samples, got {last_index}"
    print(f"Completed computing losses for test set ({test_set_size} samples).")

    print(f"Results saved to {save_dir}")


if __name__ == "__main__":
    main()
