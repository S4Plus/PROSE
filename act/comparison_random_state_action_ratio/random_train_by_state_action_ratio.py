#!/usr/bin/env python
from __future__ import annotations

"""Randomly keep a target ratio of frame indices and train ACT with a frame mask."""

import bisect
import json
import os
import random
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Iterable, Optional

import click


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hf_datasets_compat import ensure_hf_list_feature


def _ensure_hf_cache_env(root: Path, env: dict[str, str]) -> None:
    """Keep Hugging Face caches under the dataset root to avoid permission issues."""

    root = root.expanduser()
    env.setdefault("HF_HOME", str(root))
    env.setdefault("HF_DATASETS_CACHE", str(root / "hf_datasets_cache"))
    env.setdefault("HUGGINGFACE_HUB_CACHE", str(root / "hub"))


def _extend_pythonpath(repo_root: Path, script_dir: Path, lerobot_repo: Path, env: dict[str, str]) -> None:
    """Prepend project, local comparison scripts, and lerobot source to PYTHONPATH."""

    paths = [str(repo_root.expanduser()), str(script_dir.expanduser())]
    src_dir = lerobot_repo.expanduser() / "src"
    if src_dir.exists():
        paths.append(str(src_dir))

    existing = env.get("PYTHONPATH")
    if existing:
        paths.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(paths)


def _format_cmd(args: Iterable[str]) -> str:
    return shlex.join(list(args))


def _validate_ratio(name: str, value: float) -> float:
    if not 0.0 < value <= 1.0:
        raise click.BadParameter(f"{name} must be within (0, 1], received {value}.")
    return value


def _load_episode_lengths(dataset_root: Path) -> dict[int, int]:
    """Read per-episode frame counts from a LeRobot dataset."""

    meta_path = dataset_root / "meta" / "episodes.jsonl"
    if not meta_path.exists():
        raise click.ClickException(f"Missing episode metadata: {meta_path}")

    lengths: dict[int, int] = {}
    with meta_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                episode = int(row["episode_index"])
                length = int(row["length"])
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise click.ClickException(
                    f"Failed to parse {meta_path}:{line_no} as episode metadata."
                ) from exc
            if length > 0:
                lengths[episode] = length

    if not lengths:
        raise click.ClickException(f"No valid episode lengths found in {meta_path}")
    return dict(sorted(lengths.items()))


def _sample_frame_indices(total_pairs: int, target_pairs: int, rng: random.Random) -> list[int]:
    """Sample exact frame indices without replacement."""

    if target_pairs >= total_pairs:
        return list(range(total_pairs))
    return sorted(rng.sample(range(total_pairs), k=target_pairs))


def _frame_indices_to_episode_ids(frame_indices: list[int], episode_lengths: dict[int, int]) -> list[int]:
    """Map flattened frame indices back to episode ids."""

    episode_ids = list(episode_lengths.keys())
    cumulative: list[int] = []
    running = 0
    for ep in episode_ids:
        running += int(episode_lengths[ep])
        cumulative.append(running)

    selected_eps: set[int] = set()
    for idx in frame_indices:
        ep_pos = bisect.bisect_right(cumulative, idx)
        if ep_pos >= len(episode_ids):
            raise click.ClickException(f"Frame index {idx} exceeds dataset range.")
        selected_eps.add(int(episode_ids[ep_pos]))
    return sorted(selected_eps)


@click.command(context_settings={"ignore_unknown_options": True})
@click.option(
    "--sample-ratio",
    "--pair-ratio",
    type=float,
    required=True,
    help="Target fraction of state-action pairs / frames to keep (0, 1].",
)
@click.option(
    "--sample-seed",
    type=int,
    default=0,
    show_default=True,
    help="Seed used for frame sampling.",
)
@click.option(
    "--dataset-repo-id",
    type=str,
    required=True,
    help="Hugging Face repo_id for the LeRobot dataset (e.g., ls/square_mh).",
)
@click.option(
    "--dataset-root",
    type=click.Path(path_type=Path),
    required=True,
    help="Local path to the LeRobot dataset root.",
)
@click.option(
    "--dataset-revision",
    type=str,
    default=None,
    help="Optional dataset revision forwarded to lerobot training.",
)
@click.option(
    "--lerobot-repo",
    type=click.Path(path_type=Path),
    default=lambda: os.environ.get("LEROBOT_REPO", "/workspace/liushuo/cupid/lerobot_2.1"),
    show_default="env LEROBOT_REPO or /workspace/liushuo/cupid/lerobot_2.1",
    help="Path to the lerobot repository used to resolve train.py.",
)
@click.option(
    "--python-bin",
    type=click.Path(path_type=Path),
    default=Path(sys.executable),
    show_default="current interpreter",
    help="Python executable used to invoke lerobot training.",
)
@click.option(
    "--selection-output",
    type=click.Path(path_type=Path),
    default=None,
    help="Optional JSON file to save the chosen frame mask and summary.",
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
    selection_output: Optional[Path],
    dry_run: bool,
    train_args: tuple[str, ...],
) -> None:
    """Launch ACT training with a random frame-level keep mask."""

    sample_ratio = _validate_ratio("--sample-ratio", sample_ratio)

    dataset_root = dataset_root.expanduser()
    if dataset_root.is_file():
        dataset_root = dataset_root.parent
    os.makedirs(dataset_root, exist_ok=True)

    ensure_hf_list_feature()
    _ensure_hf_cache_env(dataset_root, os.environ)

    episode_lengths = _load_episode_lengths(dataset_root)
    total_episodes = len(episode_lengths)
    total_pairs = sum(episode_lengths.values())
    if total_episodes <= 0 or total_pairs <= 0:
        raise click.ClickException("Dataset appears to have zero valid episodes or samples.")

    target_pairs = max(1, min(total_pairs, int(round(total_pairs * sample_ratio))))
    rng = random.Random(sample_seed)
    selected_frame_indices = _sample_frame_indices(total_pairs, target_pairs, rng)
    kept_pairs = len(selected_frame_indices)
    actual_ratio = kept_pairs / total_pairs
    pair_gap = kept_pairs - target_pairs
    selected_episode_ids = _frame_indices_to_episode_ids(selected_frame_indices, episode_lengths)

    if selection_output is None:
        fd, tmp_path = tempfile.mkstemp(prefix="cupid_random_frame_mask_", suffix=".json")
        os.close(fd)
        selection_output = Path(tmp_path)
    else:
        selection_output = selection_output.expanduser()
        selection_output.parent.mkdir(parents=True, exist_ok=True)

    summary = {
        "dataset_root": str(dataset_root.resolve()),
        "dataset_repo_id": dataset_repo_id,
        "sample_seed": sample_seed,
        "target_ratio": sample_ratio,
        "target_pairs": target_pairs,
        "kept_pairs": kept_pairs,
        "actual_ratio": actual_ratio,
        "pair_gap": pair_gap,
        "total_pairs": total_pairs,
        "kept_episodes": len(selected_episode_ids),
        "total_episodes": total_episodes,
        "selected_episode_ids": selected_episode_ids,
        "selected_frame_indices": selected_frame_indices,
    }
    selection_output.write_text(json.dumps(summary), encoding="utf-8")

    lerobot_repo = lerobot_repo.expanduser()
    train_script = lerobot_repo / "src" / "lerobot" / "scripts" / "train.py"
    repo_root_str = str(REPO_ROOT)
    script_dir_str = str(SCRIPT_DIR)

    if train_script.exists():
        runner = (
            "import sys, runpy; "
            f"sys.path.insert(0, {repo_root_str!r}); "
            f"sys.path.insert(0, {script_dir_str!r}); "
            "import random_frame_mask_patch; "
            "random_frame_mask_patch.apply(); "
            f"runpy.run_path({train_script.as_posix()!r}, run_name='__main__')"
        )
    else:
        runner = (
            "import sys, runpy; "
            f"sys.path.insert(0, {repo_root_str!r}); "
            f"sys.path.insert(0, {script_dir_str!r}); "
            "import random_frame_mask_patch; "
            "random_frame_mask_patch.apply(); "
            "runpy.run_module('lerobot.scripts.train', run_name='__main__')"
        )

    cmd = [str(python_bin), "-c", runner]
    cmd.extend(
        [
            f"--dataset.repo_id={dataset_repo_id}",
            f"--dataset.root={dataset_root}",
        ]
    )
    if dataset_revision:
        cmd.append(f"--dataset.revision={dataset_revision}")
    cmd.extend(train_args)

    parquet_files = sorted((dataset_root / "data").glob("chunk-*/episode_*.parquet"))
    if parquet_files:
        sample_file = parquet_files[0]
        try:
            with sample_file.open("rb") as f:
                header = f.read(64)
            if b"git-lfs" in header:
                raise click.ClickException(
                    f"Dataset appears to contain Git LFS pointers (e.g., {sample_file}). "
                    "Please download the actual parquet shards to the dataset root before training."
                )
        except OSError:
            pass

    print(
        f"[random-frame-mask] Total episodes: {total_episodes} | Frames: {total_pairs} "
        f"| Kept episodes: {len(selected_episode_ids)}"
    )
    print(
        f"[random-frame-mask] Target frames: {target_pairs} / {total_pairs} ({sample_ratio:.2%}) "
        f"| Actual: {kept_pairs} / {total_pairs} ({actual_ratio:.2%})"
    )
    print(f"[random-frame-mask] Pair gap: {pair_gap:+d} | Seed: {sample_seed}")
    print(f"[random-frame-mask] Sampled frame indices (first 20): {selected_frame_indices[:20]}")
    print(f"[random-frame-mask] Selection file: {selection_output}")
    print(f"[random-frame-mask] Launching: {_format_cmd(cmd)}")

    if dry_run:
        return

    env = os.environ.copy()
    env["CUPID_FRAME_SELECTION_PATH"] = str(selection_output)
    _extend_pythonpath(REPO_ROOT, SCRIPT_DIR, lerobot_repo, env)
    _ensure_hf_cache_env(dataset_root, env)

    completed = subprocess.run(cmd, env=env, check=False)
    if completed.returncode != 0:
        raise click.ClickException(f"lerobot training exited with code {completed.returncode}.")


if __name__ == "__main__":
    main()
