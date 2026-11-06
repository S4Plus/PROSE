#!/usr/bin/env python
"""Utility to run lerobot training after filtering out the lowest-scoring episodes."""

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Sequence

import numpy as np

from lerobot.configs import parser as config_parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.scripts import train as train_module
from lerobot.utils.utils import init_logging


def _ensure_numpy_core_aliases() -> None:
    """
    Some legacy pickle files were created when numpy exposed modules under ``numpy._core``.
    Newer numpy installs only expose ``numpy.core.*``. Register aliases so pickle loading
    succeeds without pinning numpy to an old version.
    """

    core = getattr(np, "core", None)
    if core is None:
        return
    sys.modules.setdefault("numpy._core", core)
    if hasattr(core, "numeric"):
        sys.modules.setdefault("numpy._core.numeric", core.numeric)
    if hasattr(core, "multiarray"):
        sys.modules.setdefault("numpy._core.multiarray", core.multiarray)


def _read_local_episode_ids(root: Path) -> list[int]:
    episodes_file = root.expanduser() / "meta" / "episodes.jsonl"
    if not episodes_file.exists():
        raise FileNotFoundError(f"Local episodes file not found at {episodes_file}.")
    episode_ids = []
    with episodes_file.open("r") as f:
        for line in f:
            entry = json.loads(line)
            episode_ids.append(int(entry["episode_index"]))
    return episode_ids


def _read_local_episode_ids(root: Path) -> list[int]:
    episodes_file = root.expanduser() / "meta" / "episodes.jsonl"
    if not episodes_file.exists():
        raise FileNotFoundError(f"Local episodes file not found at {episodes_file}.")
    episode_ids = []
    with episodes_file.open("r") as f:
        for line in f:
            entry = json.loads(line)
            episode_ids.append(int(entry["episode_index"]))
    return episode_ids


def _ensure_hf_list_feature() -> None:
    """Datasets prior to HF 4.40 don't register ``List`` feature aliases."""

    try:
        from datasets.features import features as hf_features  # type: ignore
    except Exception:  # pragma: no cover - defensive import guard
        return

    hf_features._FEATURE_TYPES.setdefault("List", hf_features.Sequence)


def _register_retain_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--retain-trak-pkl",
        type=Path,
        help="Path to online_trak_influence.pkl (defaults to required argument).",
    )
    parser.add_argument(
        "--retain-metric",
        type=str,
        default="mean_of_mean-succ-all",
        help="Metric key from the online TRAK pickle to use for ranking episodes.",
    )
    parser.add_argument(
        "--retain-split",
        type=str,
        default="train",
        help="Split to read from the online TRAK pickle (e.g., train, holdout).",
    )
    parser.add_argument(
        "--retain-drop-k",
        type=int,
        default=6,
        help="Number of lowest-scoring episodes to remove.",
    )
    parser.add_argument(
        "--retain-help",
        action="store_true",
        help="Show this help message and exit.",
    )


def _build_arg_parser(add_help: bool = False) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Filter out the lowest-ranking episodes (based on online TRAK influence) before training.",
        add_help=add_help,
    )
    _register_retain_args(parser)
    return parser


def _parse_retain_args(argv: Sequence[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = _build_arg_parser()
    args, remaining = parser.parse_known_args(argv)
    if args.retain_help:
        _build_arg_parser(add_help=True).print_help()
        sys.exit(0)
    if args.retain_trak_pkl is None:
        parser.error("--retain-trak-pkl is required unless --retain-help is provided.")
    return args, remaining


@config_parser.wrap()
def _parse_train_cfg(cfg: TrainPipelineConfig) -> TrainPipelineConfig:
    return cfg


def _load_metric_values(pkl_path: Path, split: str, metric: str) -> np.ndarray:
    _ensure_numpy_core_aliases()
    with pkl_path.expanduser().open("rb") as f:
        data = pickle.load(f)
    if split not in data:
        raise KeyError(f"Split '{split}' not found in {pkl_path}. Available splits: {list(data.keys())}")
    split_payload = data[split]
    if metric not in split_payload:
        raise KeyError(f"Metric '{metric}' not found under split '{split}'. Available metrics: {list(split_payload)}")
    values = np.asarray(split_payload[metric]).reshape(-1)
    return values


def _resolve_episode_ids(cfg: TrainPipelineConfig) -> list[int]:
    if cfg.dataset.root is not None:
        return _read_local_episode_ids(Path(cfg.dataset.root))
    meta = LeRobotDatasetMetadata(
        repo_id=cfg.dataset.repo_id,
        root=cfg.dataset.root,
        revision=cfg.dataset.revision,
    )
    if cfg.dataset.episodes is None:
        return list(range(meta.total_episodes))
    return list(cfg.dataset.episodes)


def _filter_indices(values: np.ndarray, drop_k: int) -> tuple[list[int], list[int]]:
    num_eps = values.shape[0]
    if drop_k <= 0:
        keep = list(range(num_eps))
        return keep, []
    if drop_k >= num_eps:
        raise ValueError(f"Cannot drop {drop_k} episodes when only {num_eps} are available.")
    order = np.argsort(values)  # ascending -> lowest scores first
    drop_idx = order[:drop_k].tolist()
    keep = [idx for idx in range(num_eps) if idx not in drop_idx]
    return keep, drop_idx


def main() -> None:
    retain_args, remaining = _parse_retain_args(sys.argv[1:])
    sys.argv = [sys.argv[0]] + remaining

    # Ensure logging behaves like lerobot-train so progress is visible.
    init_logging()
    cfg = _parse_train_cfg()
    values = _load_metric_values(
        pkl_path=retain_args.retain_trak_pkl,
        split=retain_args.retain_split,
        metric=retain_args.retain_metric,
    )
    episode_ids = _resolve_episode_ids(cfg)
    if len(values) != len(episode_ids):
        raise ValueError(
            "Mismatch between TRAK influence entries and dataset episodes: "
            f"{len(values)=} vs {len(episode_ids)=}. "
            "Ensure the pickle file matches the dataset selection."
        )

    keep_idx, drop_idx = _filter_indices(values, retain_args.retain_drop_k)
    dropped = [(episode_ids[i], float(values[i])) for i in drop_idx]
    cfg.dataset.episodes = [episode_ids[i] for i in keep_idx]

    print(
        f"[retain] Filtering {len(drop_idx)} episode(s). Remaining: {len(cfg.dataset.episodes)} / {len(episode_ids)}."
    )
    if dropped:
        print("[retain] Dropped episodes (episode_index -> score):")
        for episode_index, score in dropped:
            print(f"  - {episode_index}: {score:.6e}")

    _ensure_hf_list_feature()
    # breakpoint()
    train_module.train(cfg)


if __name__ == "__main__":
    main()
