#!/usr/bin/env python
from __future__ import annotations

"""Randomly sample a fraction of episodes and launch ACT training."""

# Allow execution from any working directory without installing this checkout.
import sys as _prose_sys
from pathlib import Path as _ProsePath
_prose_sys.path.insert(0, str(_ProsePath(__file__).resolve().parents[2]))


import json
import os
import random
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Optional

import click

from prose.compat.hf_datasets import ensure_hf_list_feature


def _ensure_hf_cache_env(root: Path, env: dict[str, str]) -> None:
    """Keep Hugging Face caches under the dataset root to avoid permission issues."""

    root = root.expanduser()
    env.setdefault("HF_HOME", str(root))
    env.setdefault("HF_DATASETS_CACHE", str(root / "hf_datasets_cache"))
    env.setdefault("HUGGINGFACE_HUB_CACHE", str(root / "hub"))


def _extend_pythonpath(lerobot_repo: Path, env: dict[str, str]) -> None:
    """Prepend lerobot's source tree to PYTHONPATH for the training subprocess."""

    src_dir = lerobot_repo.expanduser() / "src"
    if not src_dir.exists():
        return
    existing = env.get("PYTHONPATH")
    paths = [str(src_dir)]
    if existing:
        paths.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(paths)


def _format_cmd(args: Iterable[str]) -> str:
    return shlex.join(list(args))


def _validate_ratio(name: str, value: float) -> float:
    if not 0.0 < value <= 1.0:
        raise click.BadParameter(f"{name} must be within (0, 1], received {value}.")
    return value


def _load_metadata_class(lerobot_repo: Path):
    """Import LeRobotDatasetMetadata, falling back to the provided source tree."""

    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata  # type: ignore
        return LeRobotDatasetMetadata
    except Exception:
        src_dir = lerobot_repo.expanduser() / "src"
        if src_dir.exists() and str(src_dir) not in sys.path:
            sys.path.insert(0, str(src_dir))
        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata  # type: ignore
            return LeRobotDatasetMetadata
        except Exception as exc:  # pragma: no cover - defensive path
            raise click.ClickException(
                f"Could not import LeRobotDatasetMetadata. "
                f"Ensure lerobot is installed or available under {lerobot_repo}."
            ) from exc


@click.command(context_settings={"ignore_unknown_options": True})
@click.option(
    "--sample-ratio",
    type=float,
    required=True,
    help="Fraction of episodes to keep (0, 1].",
)
@click.option(
    "--sample-seed",
    type=int,
    default=0,
    show_default=True,
    help="Seed used for episode sampling.",
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
    help="Optional dataset revision (forwarded to lerobot-train).",
)
@click.option(
    "--lerobot-repo",
    type=click.Path(path_type=Path),
    default=lambda: os.environ.get("LEROBOT_REPO", str(Path(__file__).resolve().parents[2] / "lerobot_2.1")),
    show_default="env LEROBOT_REPO or bundled lerobot_2.1",
    help="Path to the lerobot repository (used to resolve train.py if needed).",
)
@click.option(
    "--python-bin",
    type=click.Path(path_type=Path),
    default=Path(sys.executable),
    show_default="current interpreter",
    help="Python executable used to invoke lerobot training.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Print the resolved command without executing it.",
)
@click.argument("train_args", nargs=-1, type=click.UNPROCESSED)
def main(
    sample_ratio: float,
    sample_seed: int,
    dataset_repo_id: str,
    dataset_root: Path,
    dataset_revision: Optional[str],
    lerobot_repo: Path,
    python_bin: Path,
    dry_run: bool,
    train_args: tuple[str, ...],
) -> None:
    """Launch lerobot ACT training on a random subset of episodes."""

    sample_ratio = _validate_ratio("--sample-ratio", sample_ratio)

    dataset_root = dataset_root.expanduser()
    if dataset_root.is_file():
        dataset_root = dataset_root.parent
    os.makedirs(dataset_root, exist_ok=True)

    ensure_hf_list_feature()
    _ensure_hf_cache_env(dataset_root, os.environ)

    metadata_cls = _load_metadata_class(lerobot_repo)
    metadata = metadata_cls(repo_id=dataset_repo_id, root=dataset_root, revision=dataset_revision)
    total_eps = metadata.total_episodes
    if total_eps <= 0:
        raise click.ClickException("Dataset appears to have zero episodes.")

    keep = max(1, min(total_eps, int(round(total_eps * sample_ratio))))
    rng = random.Random(sample_seed)
    episodes = sorted(rng.sample(range(total_eps), k=keep))
    episodes_arg = json.dumps(episodes)

    lerobot_repo = lerobot_repo.expanduser()
    train_script = lerobot_repo / "src" / "lerobot" / "scripts" / "train.py"

    if train_script.exists():
        runner = (
            "import runpy; from prose.compat import random_subset as random_train_patch; "
            "random_train_patch.apply(); "
            f"runpy.run_path('{train_script.as_posix()}', run_name='__main__')"
        )
    else:
        runner = (
            "import runpy; from prose.compat import random_subset as random_train_patch; "
            "random_train_patch.apply(); "
            "runpy.run_module('lerobot.scripts.train', run_name='__main__')"
        )

    cmd = [str(python_bin), "-c", runner]

    cmd.extend(
        [
            f"--dataset.repo_id={dataset_repo_id}",
            f"--dataset.root={dataset_root}",
            f"--dataset.episodes={episodes_arg}",
        ]
    )
    if dataset_revision:
        cmd.append(f"--dataset.revision={dataset_revision}")
    cmd.extend(train_args)

    # Detect Git LFS placeholders early to avoid confusing parquet errors.
    parquet_files = sorted((dataset_root / "data").glob("chunk-*/episode_*.parquet"))
    if parquet_files:
        sample_file = parquet_files[0]
        try:
            with open(sample_file, "rb") as f:
                header = f.read(64)
            if b"git-lfs" in header:
                raise click.ClickException(
                    f"Dataset appears to contain Git LFS pointers (e.g., {sample_file}). "
                    "Please download the actual parquet shards to the dataset root before training."
                )
        except OSError:
            pass

    print(
        f"[random] Total episodes: {total_eps} | Keeping: {keep} "
        f"({keep/total_eps:.2%}) | Seed: {sample_seed}"
    )
    print(f"[random] Sampled episodes (first 20): {episodes[:20]}")
    print(f"[random] Launching: {_format_cmd(cmd)}")

    if dry_run:
        return

    env = os.environ.copy()
    repo_root = str(Path(__file__).resolve().parents[2])
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [repo_root, env.get("PYTHONPATH")]))
    _extend_pythonpath(lerobot_repo, env)
    _ensure_hf_cache_env(dataset_root, env)

    completed = subprocess.run(cmd, env=env, check=False)
    if completed.returncode != 0:
        raise click.ClickException(f"lerobot training exited with code {completed.returncode}.")


if __name__ == "__main__":
    main()
