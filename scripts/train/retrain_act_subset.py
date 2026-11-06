#!/usr/bin/env python
from __future__ import annotations

"""Helper to retrain ACT policies after dropping a fraction of the dataset.

The script wraps Lerobot's ``retrain.py`` utility and computes ``--retain-drop-k``
from a user provided ratio (e.g., drop half of the episodes).  All additional
arguments are forwarded to Lerobot's retraining entrypoint, so you can pass the
usual ``--policy.*`` or ``--output_dir`` flags directly.
"""

import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Optional

import click

from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata


def _ensure_hf_cache_env(root: Path) -> None:
    """Keep HF caches under the dataset root (avoid permission issues)."""

    os.environ.setdefault("HF_HOME", str(root))
    os.environ.setdefault("HF_DATASETS_CACHE", str(root / "hf_datasets_cache"))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(root / "hub"))


def _resolve_total_episodes(
    repo_id: str,
    root: Path,
    revision: Optional[str],
) -> int:
    root = root.expanduser()
    if root.is_file():
        root = root.parent
    _ensure_hf_cache_env(root)
    metadata = LeRobotDatasetMetadata(repo_id=repo_id, root=root, revision=revision)
    return metadata.total_episodes


def _validate_ratio(name: str, value: float) -> float:
    if not 0.0 <= value <= 1.0:
        raise click.BadParameter(f"{name} must be within [0, 1], received {value}.")
    return value


def _format_cmd(args: Iterable[str]) -> str:
    return shlex.join(list(args))


@click.command(context_settings={"ignore_unknown_options": True})
@click.option("--retain-trak-pkl", type=click.Path(path_type=Path), required=True, help="Path to `online_trak_influence.pkl`.")
@click.option(
    "--retain-metric",
    type=str,
    default="mean_of_mean-succ-all",
    show_default=True,
    help="Metric key inside the TRAK pickle used for ranking.",
)
@click.option(
    "--retain-split",
    type=str,
    default="train",
    show_default=True,
    help="Split name inside the TRAK pickle (e.g., train, holdout).",
)
@click.option(
    "--drop-ratio",
    type=float,
    default=None,
    help="Fraction of episodes to drop (e.g., 0.5 removes half).",
)
@click.option(
    "--keep-ratio",
    type=float,
    default=None,
    help="Optional fraction of episodes to keep instead of dropping.",
)
@click.option(
    "--drop-k",
    type=int,
    default=None,
    help="Override: drop exactly this many episodes (skips ratio computation).",
)
@click.option(
    "--dataset-repo-id",
    type=str,
    required=True,
    help="Hugging Face repo_id for the Lerobot dataset (e.g., ls/Static_Pick_Bottle_400).",
)
@click.option(
    "--dataset-root",
    type=click.Path(path_type=Path),
    required=True,
    help="Local path to the Lerobot dataset cache.",
)
@click.option(
    "--dataset-revision",
    type=str,
    default=None,
    help="Optional dataset revision (forwarded to Lerobot).",
)
@click.option(
    "--lerobot-repo",
    type=click.Path(path_type=Path),
    default=lambda: os.environ.get("LEROBOT_REPO", "/root/lerobot_2"),
    show_default="env LEROBOT_REPO or /root/lerobot_2",
    help="Path to the Lerobot repository that hosts `retrain.py`.",
)
@click.option(
    "--python-bin",
    type=click.Path(path_type=Path),
    default=Path(sys.executable),
    show_default="current interpreter",
    help="Python executable used to invoke Lerobot's retrain script.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Print the resolved command without executing it.",
)
@click.argument("train_args", nargs=-1, type=click.UNPROCESSED)
def main(
    retain_trak_pkl: Path,
    retain_metric: str,
    retain_split: str,
    drop_ratio: Optional[float],
    keep_ratio: Optional[float],
    drop_k: Optional[int],
    dataset_repo_id: str,
    dataset_root: Path,
    dataset_revision: Optional[str],
    lerobot_repo: Path,
    python_bin: Path,
    dry_run: bool,
    train_args: tuple[str, ...],
) -> None:
    """Launch Lerobot retraining after filtering a ratio of ACT training data."""

    lerobot_repo = lerobot_repo.expanduser()
    retrain_script = lerobot_repo / "retrain.py"
    if not retrain_script.exists():
        raise click.ClickException(f"Could not find retrain.py under {lerobot_repo}.")

    total_eps = _resolve_total_episodes(dataset_repo_id, dataset_root, dataset_revision)
    if total_eps <= 0:
        raise click.ClickException("Dataset appears to have zero episodes.")

    if drop_k is not None and (drop_ratio is not None or keep_ratio is not None):
        raise click.ClickException("Provide either --drop-k or a ratio (--drop-ratio/--keep-ratio), not both.")
    if drop_ratio is not None and keep_ratio is not None:
        raise click.ClickException("Only one of --drop-ratio or --keep-ratio can be set.")

    resolved_drop_k: int
    if drop_k is not None:
        if not 0 <= drop_k < total_eps:
            raise click.BadParameter(
                f"--drop-k must be within [0, {total_eps - 1}], received {drop_k}.", param_hint="--drop-k"
            )
        resolved_drop_k = drop_k
    elif keep_ratio is not None:
        keep_ratio = _validate_ratio("--keep-ratio", keep_ratio)
        keep = max(1, int(round(total_eps * keep_ratio)))
        keep = min(keep, total_eps)
        resolved_drop_k = max(0, total_eps - keep)
    elif drop_ratio is not None:
        drop_ratio = _validate_ratio("--drop-ratio", drop_ratio)
        resolved_drop_k = int(round(total_eps * drop_ratio))
        resolved_drop_k = min(resolved_drop_k, total_eps - 1)
    else:
        raise click.ClickException("Please specify --drop-k, --drop-ratio, or --keep-ratio.")

    keep_after = total_eps - resolved_drop_k
    if keep_after <= 0:
        raise click.ClickException(
            f"Cannot drop {resolved_drop_k} episodes because only {total_eps} are available."
        )

    cmd = [
        str(python_bin),
        str(retrain_script),
        f"--retain-trak-pkl={retain_trak_pkl}",
        f"--retain-metric={retain_metric}",
        f"--retain-split={retain_split}",
        f"--retain-drop-k={resolved_drop_k}",
        f"--dataset.repo_id={dataset_repo_id}",
        f"--dataset.root={dataset_root}",
    ]
    if dataset_revision:
        cmd.append(f"--dataset.revision={dataset_revision}")
    cmd.extend(train_args)

    print(
        f"[subset] Total episodes: {total_eps} | "
        f"Dropping: {resolved_drop_k} | Remaining: {keep_after}"
    )
    print(f"[subset] Launching: {_format_cmd(cmd)}")

    if dry_run:
        return

    completed = subprocess.run(cmd, check=False)
    if completed.returncode != 0:
        raise click.ClickException(f"Lerobot retrain command exited with code {completed.returncode}.")


if __name__ == "__main__":
    main()
