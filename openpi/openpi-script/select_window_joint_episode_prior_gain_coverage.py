#!/usr/bin/env python3
"""Select a window+episode-prior subset with a gain+coverage heuristic."""

from __future__ import annotations

import argparse
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

import numpy as np

import openpi.training.config as _config
import retrain as _retrain
import retrain_window_joint as base
from window_episode_prior_gain_coverage_common import (
    build_context_from_args,
    build_selection_summary,
    default_output_dir,
    materialize_selection,
    recompute_norm_stats_with_fallback,
    required_episode_count,
    search_keep_ratio_by_gain_and_coverage,
    validate_common_args,
    write_gain_coverage_report,
    write_selection_artifacts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select OpenPI training data with a gain+coverage heuristic over window+episode-prior scores."
    )
    parser.add_argument("--score-pkl", type=Path, required=True, help="Window GraSS pickle.")
    parser.add_argument("--train-config", type=str, default=None, help="Training config name. If omitted, infer from score pickle.")
    parser.add_argument("--dataset-dir", type=Path, default=None, help="Dataset root. If omitted, infer from train config.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory for subset, scores, reports, and norm_stats.")
    parser.add_argument("--output-indices", type=Path, default=None, help="Output sample-index file.")

    parser.add_argument("--joint-rule", choices=("product", "sum"), default="sum")
    parser.add_argument("--joint-quality-power", type=float, default=1.0)
    parser.add_argument("--joint-influence-power", type=float, default=1.0)
    parser.add_argument("--window-score-weight", type=float, default=0.8)
    parser.add_argument("--episode-prior-weight", type=float, default=0.2)
    parser.add_argument(
        "--adaptive-prior-mode",
        choices=("none", "window_std", "inverse_episode_length", "quality_std", "hybrid"),
        default="none",
    )
    parser.add_argument("--adaptive-prior-strength", type=float, default=0.5)
    parser.add_argument("--interaction-weight", type=float, default=0.0)
    parser.add_argument("--fusion-normalizer", choices=("minmax", "percentile", "none"), default="minmax")
    parser.add_argument("--episode-quality-aggregation", choices=("mean", "max"), default="mean")
    parser.add_argument("--min-windows-per-episode", type=int, default=0)
    parser.add_argument("--episode-budget-rule", choices=("global", "uniform_episode", "episode_prior"), default="global")
    parser.add_argument(
        "--quality-source",
        choices=("episode_eval", "trajectory_eval", "rank_fusion"),
        default="rank_fusion",
    )
    parser.add_argument("--quality-normalizer", choices=("quantile", "minmax"), default="quantile")
    parser.add_argument("--quality-quantile-low", type=float, default=0.05)
    parser.add_argument("--quality-quantile-high", type=float, default=0.95)
    parser.add_argument("--episode-action-weight", type=float, default=0.35)
    parser.add_argument("--episode-state-weight", type=float, default=0.35)
    parser.add_argument("--episode-length-weight", type=float, default=0.15)
    parser.add_argument("--episode-energy-weight", type=float, default=0.15)
    parser.add_argument("--trajectory-terminal-weight", type=float, default=0.25)
    parser.add_argument("--trajectory-efficiency-weight", type=float, default=0.20)
    parser.add_argument("--trajectory-progress-weight", type=float, default=0.25)
    parser.add_argument("--trajectory-state-smooth-weight", type=float, default=0.15)
    parser.add_argument("--trajectory-action-smooth-weight", type=float, default=0.10)
    parser.add_argument("--trajectory-length-weight", type=float, default=0.05)
    parser.add_argument("--rank-fusion-episode-weight", type=float, default=0.5)
    parser.add_argument("--rank-fusion-trajectory-weight", type=float, default=0.5)

    parser.add_argument("--min-episode-coverage-ratio", type=float, default=0.9)
    parser.add_argument("--max-relative-gain", type=float, default=0.02)

    parser.add_argument("--max-frames-for-norm-stats", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_common_args(args)
    context = build_context_from_args(args)
    heuristic = search_keep_ratio_by_gain_and_coverage(
        context,
        min_episode_coverage_ratio=args.min_episode_coverage_ratio,
        max_relative_gain=args.max_relative_gain,
    )
    selected = materialize_selection(context, keep_ratio=heuristic["recommended_keep_ratio"])

    output_dir = args.output_dir or default_output_dir(
        dataset_dir=context["dataset_dir"],
        score_pkl=args.score_pkl,
        keep_ratio=heuristic["recommended_keep_ratio"],
        joint_rule=args.joint_rule,
        quality_source=args.quality_source,
        window_score_weight=args.window_score_weight,
        episode_prior_weight=args.episode_prior_weight,
        adaptive_prior_mode=args.adaptive_prior_mode,
        interaction_weight=args.interaction_weight,
    )
    output_indices = args.output_indices or (output_dir / "kept_sample_indices.npy")
    output_dir.mkdir(parents=True, exist_ok=True)
    _retrain._save_sample_indices(output_indices, selected["kept_sample_indices"], overwrite=args.overwrite)  # noqa: SLF001

    summary = build_selection_summary(context, selected, heuristic)
    write_selection_artifacts(output_dir, context=context, selected=selected, summary=summary)
    write_gain_coverage_report(output_dir, heuristic=heuristic)
    np.savez_compressed(output_dir / "window_quality_components.npz", **context["quality_components"])

    train_cfg = _config.get_config(context["train_config_name"])
    print(f"Score file: {args.score_pkl}")
    print(f"Train config: {context['train_config_name']}")
    print(f"Dataset: {context['dataset_dir']}")
    print(
        "Gain+Coverage selection summary: "
        f"windows={summary['num_windows']}, kept_windows={summary['kept_windows']}, "
        f"kept_samples={summary['kept_samples']}, keep_ratio={summary['keep_ratio']:.6f}"
    )
    print(
        "Heuristic constraints: "
        f"gain_delta={heuristic['gain_delta']}, max_relative_gain={heuristic['max_relative_gain']:.6f}, "
        f"coverage>={heuristic['min_episode_coverage_ratio']:.4f}, "
        f"min_feasible_keep={heuristic['min_feasible_keep_ratio']:.4f}, "
        f"episodes_with_min_windows>={heuristic['min_episode_ratio_with_min_windows']:.4f}"
    )
    print(
        "Recommended selection stats: "
        f"covered_episodes={heuristic['recommended_covered_episodes']}/{summary['num_episodes']} "
        f"(target>={required_episode_count(summary['num_episodes'], heuristic['min_episode_coverage_ratio'])}), "
        f"episodes_meeting_min_windows={heuristic['recommended_episodes_meeting_min_windows']}/{summary['num_episodes']}"
    )
    print(f"Output dir: {output_dir}")
    print(f"Sample indices: {output_indices}")
    print("Recomputing norm_stats.json for selected sample subset...")
    norm_stats_path, used_single_worker_fallback = recompute_norm_stats_with_fallback(
        train_cfg,
        sample_indices_path=output_indices,
        output_dir=output_dir,
        max_frames=args.max_frames_for_norm_stats,
    )
    if used_single_worker_fallback:
        print("norm_stats fallback: retrying with num_workers=0 due to multiprocessing permission limits.")
    print(f"Subset norm_stats written to: {norm_stats_path}")
    print(f"Gain+coverage report: {output_dir / 'gain_coverage_selection_report.json'}")


if __name__ == "__main__":
    main()
