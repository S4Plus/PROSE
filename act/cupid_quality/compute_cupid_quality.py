#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from common import PairScoreData, higher_better_to_score, load_pair_score_data, summarize, to_json_ready  # type: ignore
else:
    from .common import PairScoreData, higher_better_to_score, load_pair_score_data, summarize, to_json_ready


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute standalone CUPID-QUALITY scores from state-action pair scores.")
    parser.add_argument("--input-path", type=Path, required=True, help="Pair-score file (.npz/.pkl/.csv).")
    parser.add_argument("--input-format", type=str, default="auto", choices=("auto", "npz", "pkl", "csv"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--reduce-rollouts",
        type=str,
        default="mean",
        choices=("mean", "sum"),
        help="How to aggregate per-rollout contributions over D_tau.",
    )
    parser.add_argument(
        "--normalize-by-trajectory-length",
        action="store_true",
        help="Divide each rollout contribution by its trajectory length m.",
    )
    parser.add_argument(
        "--normalizer",
        type=str,
        default="quantile",
        choices=("quantile", "minmax"),
        help="How to map raw quality values to [0,1].",
    )
    parser.add_argument("--quantile-low", type=float, default=0.05)
    parser.add_argument("--quantile-high", type=float, default=0.95)
    parser.add_argument("--drop-quantile", type=float, default=0.0, help="Optional lowest quantile to mark as drop.")
    return parser.parse_args()


def _group_indices(ids: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    unique_ids = np.unique(ids.astype(np.int64, copy=False))
    groups = [np.flatnonzero(ids == episode_id) for episode_id in unique_ids]
    return unique_ids, groups


def compute_cupid_quality(
    data: PairScoreData,
    *,
    reduce_rollouts: str,
    normalize_by_trajectory_length: bool,
    normalizer: str,
    quantile_low: float,
    quantile_high: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pair_scores = np.asarray(data.pair_scores, dtype=np.float64)
    eval_episode_ids = np.asarray(data.eval_episode_ids, dtype=np.int64)
    train_episode_ids = np.asarray(data.train_episode_ids, dtype=np.int64)

    eval_eps, eval_groups = _group_indices(eval_episode_ids)
    train_eps, train_groups = _group_indices(train_episode_ids)

    raw_sum = np.zeros(train_eps.shape[0], dtype=np.float64)
    raw_count = np.zeros(train_eps.shape[0], dtype=np.int64)

    for eval_rows in eval_groups:
        tau_scores = pair_scores[eval_rows, :]
        tau_len = int(tau_scores.shape[0])
        if tau_len <= 0:
            continue
        for train_idx, train_cols in enumerate(train_groups):
            sub = tau_scores[:, train_cols]
            if sub.size == 0:
                continue
            step_min = np.nanmin(sub, axis=1)
            step_max = np.nanmax(sub, axis=1)
            if not np.isfinite(step_min).any() or not np.isfinite(step_max).any():
                continue
            left = float(np.nanmax(step_min))
            right = float(np.nanmin(step_max))
            contrib = left - right
            if normalize_by_trajectory_length:
                contrib /= float(max(tau_len, 1))
            raw_sum[train_idx] += contrib
            raw_count[train_idx] += 1

    raw_quality = np.full(train_eps.shape[0], np.nan, dtype=np.float64)
    valid = raw_count > 0
    if reduce_rollouts == "sum":
        raw_quality[valid] = raw_sum[valid]
    else:
        raw_quality[valid] = raw_sum[valid] / raw_count[valid]

    quality_score = higher_better_to_score(
        raw_quality,
        normalizer=normalizer,
        quantile_low=quantile_low,
        quantile_high=quantile_high,
    )
    return train_eps, raw_quality, quality_score, raw_count


def _write_ranked_list(path: Path, episode_ids: np.ndarray, values: np.ndarray, *, reverse: bool, topk: int = 50) -> None:
    ranked = sorted(
        ((int(ep), float(val)) for ep, val in zip(episode_ids, values, strict=True) if np.isfinite(val)),
        key=lambda item: item[1],
        reverse=reverse,
    )
    with path.open("w") as f:
        for ep, val in ranked[:topk]:
            f.write(f"{ep},{val:.8f}\n")


def write_outputs(
    args: argparse.Namespace,
    data: PairScoreData,
    episode_ids: np.ndarray,
    raw_quality: np.ndarray,
    quality_score: np.ndarray,
    rollout_count: np.ndarray,
) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)

    finite_scores = np.isfinite(quality_score)
    rank_order = np.argsort(np.where(finite_scores, quality_score, -np.inf))[::-1]
    ranks = np.full(episode_ids.shape[0], -1, dtype=np.int64)
    for pos, idx in enumerate(rank_order, start=1):
        if finite_scores[idx]:
            ranks[idx] = pos

    drop_mask = np.zeros(episode_ids.shape[0], dtype=bool)
    finite_idx = np.flatnonzero(finite_scores)
    if float(args.drop_quantile) > 0.0 and finite_idx.size > 0:
        threshold = float(np.nanquantile(quality_score[finite_scores], float(args.drop_quantile)))
        drop_mask = finite_scores & (quality_score <= threshold)

    csv_path = args.output_dir / "quality_episodes.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "quality_score", "quality_raw", "rollout_count", "rank_desc", "drop"])
        for idx, episode_id in enumerate(episode_ids):
            writer.writerow(
                [
                    int(episode_id),
                    float(quality_score[idx]) if np.isfinite(quality_score[idx]) else float("nan"),
                    float(raw_quality[idx]) if np.isfinite(raw_quality[idx]) else float("nan"),
                    int(rollout_count[idx]),
                    int(ranks[idx]),
                    int(bool(drop_mask[idx])),
                ]
            )

    keep_path = args.output_dir / "quality_keep.txt"
    with keep_path.open("w") as f:
        for episode_id in episode_ids[~drop_mask]:
            f.write(f"{int(episode_id)}\n")

    drop_path = args.output_dir / "quality_drop.txt"
    with drop_path.open("w") as f:
        for episode_id in episode_ids[drop_mask]:
            f.write(f"{int(episode_id)}\n")

    _write_ranked_list(args.output_dir / "quality_top_50.txt", episode_ids, quality_score, reverse=True)
    _write_ranked_list(args.output_dir / "quality_bottom_50.txt", episode_ids, quality_score, reverse=False)

    report = {
        "method": "cupid_quality",
        "formula": "mean_tau[max_{(s',a') in tau} min_{(s,a) in xi} psi_a_inf - min_{(s',a') in tau} max_{(s,a) in xi} psi_a_inf]",
        "input_path": data.source_path,
        "num_eval_state_actions": int(data.pair_scores.shape[0]) if data.pair_scores.ndim == 2 else 0,
        "num_train_state_actions": int(data.pair_scores.shape[1]) if data.pair_scores.ndim == 2 else 0,
        "num_eval_episodes": int(np.unique(data.eval_episode_ids).shape[0]) if data.eval_episode_ids.size > 0 else 0,
        "num_train_episodes": int(episode_ids.shape[0]),
        "normalize_by_trajectory_length": bool(args.normalize_by_trajectory_length),
        "reduce_rollouts": args.reduce_rollouts,
        "quality_raw_stats": summarize(raw_quality),
        "quality_score_stats": summarize(quality_score),
        "top_10": [
            [int(episode_ids[idx]), float(quality_score[idx]), float(raw_quality[idx])]
            for idx in rank_order[:10]
            if np.isfinite(quality_score[idx])
        ],
        "config": vars(args),
    }
    report_path = args.output_dir / "quality_report.json"
    report_path.write_text(json.dumps(to_json_ready(report), indent=2) + "\n")


def main() -> None:
    args = parse_args()
    data = load_pair_score_data(args.input_path, fmt=args.input_format)
    episode_ids, raw_quality, quality_score, rollout_count = compute_cupid_quality(
        data,
        reduce_rollouts=args.reduce_rollouts,
        normalize_by_trajectory_length=bool(args.normalize_by_trajectory_length),
        normalizer=args.normalizer,
        quantile_low=float(args.quantile_low),
        quantile_high=float(args.quantile_high),
    )
    write_outputs(args, data, episode_ids, raw_quality, quality_score, rollout_count)
    print(f"[cupid_quality] train_episodes={len(episode_ids)}")
    print(f"[cupid_quality] outputs -> {args.output_dir}")


if __name__ == "__main__":
    main()
