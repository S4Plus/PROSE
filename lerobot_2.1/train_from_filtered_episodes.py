#!/usr/bin/env python
"""Train a LeRobot v2.1 policy using episode indices listed in a filtered datasets file."""

import argparse
import sys
from pathlib import Path

from lerobot.configs import parser as config_parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.scripts import train as train_module
from lerobot.utils.utils import init_logging


def parse_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Train using a filtered episode list exported by Demo-SCORE."
    )
    parser.add_argument(
        "--filtered-datasets-file",
        required=True,
        type=Path,
        help="Path to a datasets_file line like '<abs_dataset_path> 1,2,3,...'.",
    )
    args, remaining = parser.parse_known_args(argv)
    return args, remaining


def load_filtered_episodes(path: Path) -> tuple[str, list[int]]:
    line = path.read_text().strip()
    if not line:
        raise ValueError(f"Filtered datasets file is empty: {path}")
    dataset_path, episode_csv = line.split(" ", 1)
    episode_ids = [int(x) for x in episode_csv.split(",") if x]
    if not episode_ids:
        raise ValueError(f"No episode ids found in {path}")
    return dataset_path, episode_ids


@config_parser.wrap()
def parse_train_cfg(cfg: TrainPipelineConfig) -> TrainPipelineConfig:
    return cfg


def main() -> None:
    args, remaining = parse_args(sys.argv[1:])
    dataset_path, episode_ids = load_filtered_episodes(args.filtered_datasets_file)

    sys.argv = [sys.argv[0]] + remaining
    init_logging()
    cfg = parse_train_cfg()

    cfg.dataset.root = dataset_path
    cfg.dataset.repo_id = Path(dataset_path).name
    cfg.dataset.episodes = episode_ids

    print(
        f"[filtered-train] Using {len(episode_ids)} filtered episodes from {dataset_path}"
    )
    train_module.train(cfg)


if __name__ == "__main__":
    main()
