#!/usr/bin/env python3
"""Run ACT-style joint episode-window selection, then OpenPI prepare/norm_stats/train."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
SRC_ROOT = REPO_ROOT / "src"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import retrain as _retrain


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Retrain OpenPI using the ACT joint_episode_window method ported into openpi-script."
    )
    parser.add_argument("--score-pkl", type=Path, required=True, help="Influence score pickle.")
    parser.add_argument("--train-config", type=str, default=None, help="Training config name.")
    parser.add_argument("--dataset-dir", type=Path, default=None, help="Dataset root. If omitted, infer from train config.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory for selection and training artifacts.")
    parser.add_argument("--output-indices", type=Path, default=None, help="Optional explicit sample-index file.")
    parser.add_argument("--exp-name", type=str, default=None, help="Experiment name for retraining.")
    parser.add_argument("--checkpoint-base-dir", type=Path, default=Path("./checkpoints"))
    parser.add_argument("--num-train-steps", type=int, default=None)
    parser.add_argument("--max-frames-for-norm-stats", type=int, default=None)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_known_args()


def main() -> None:
    args, selection_args = parse_args()
    payload = _retrain._load_score_payload(args.score_pkl)  # noqa: SLF001
    train_config_name = args.train_config or _retrain._extract_score_config_name(payload)  # noqa: SLF001
    if train_config_name is None:
        raise ValueError("--train-config is required because the score pickle does not include config.config_name.")

    if args.dataset_dir is None:
        dataset_dir, default_exp_name = _retrain._infer_defaults_from_train_config(train_config_name)  # noqa: SLF001
    else:
        dataset_dir = args.dataset_dir
        default_exp_name = train_config_name

    output_dir = args.output_dir
    selection_dir = output_dir / "selection"
    train_dir = output_dir / "train"
    selection_dir.mkdir(parents=True, exist_ok=True)
    train_dir.mkdir(parents=True, exist_ok=True)
    selection_npz = selection_dir / "joint_episode_window_selection.npz"

    select_cmd = [
        sys.executable,
        str(SCRIPT_DIR / "select_joint_episode_window.py"),
        "--score-pkl",
        str(args.score_pkl),
        "--dataset-root",
        str(dataset_dir),
        "--output-dir",
        str(selection_dir),
        *selection_args,
    ]

    prepare_cmd = [
        sys.executable,
        str(SCRIPT_DIR / "prepare_openpi_from_joint_episode_window.py"),
        "--score-pkl",
        str(args.score_pkl),
        "--selection-npz",
        str(selection_npz),
        "--train-config",
        train_config_name,
        "--dataset-dir",
        str(dataset_dir),
        "--output-dir",
        str(train_dir),
        "--checkpoint-base-dir",
        str(args.checkpoint_base_dir),
        "--exp-name",
        args.exp_name or default_exp_name,
    ]
    if args.output_indices is not None:
        prepare_cmd.extend(["--output-indices", str(args.output_indices)])
    if args.num_train_steps is not None:
        prepare_cmd.extend(["--num-train-steps", str(args.num_train_steps)])
    if args.max_frames_for_norm_stats is not None:
        prepare_cmd.extend(["--max-frames-for-norm-stats", str(args.max_frames_for_norm_stats)])
    if args.prepare_only:
        prepare_cmd.append("--prepare-only")
    if args.overwrite:
        prepare_cmd.append("--overwrite")

    print("[stage] selection")
    print("Command:", " ".join(select_cmd))
    subprocess.run(select_cmd, check=True)

    print("[stage] prepare/train")
    print("Command:", " ".join(prepare_cmd))
    subprocess.run(prepare_cmd, check=True)


if __name__ == "__main__":
    main()
