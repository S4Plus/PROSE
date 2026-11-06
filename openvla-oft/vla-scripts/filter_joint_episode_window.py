#!/usr/bin/env python3
"""Two-stage joint episode-window selection for OpenVLA.

Stage 1:
- score each episode with
    episode_score = a * norm(influence) + b * norm(quality)
- drop the bottom episode fraction (default 10%)


CUDA_VISIBLE_DEVICES=1 nohup python /workspace/liushuo/cupid/openvla-oft/vla-scripts/filter_joint_episode_window.py \
    --score-pkl /workspace/liushuo/cupid/1_infulenece_score/can_mh/score_episode_window.pkl \
    --dataset-root /workspace/liushuo/cupid/datasets/lerobot/can_mh \
    --output-dir /workspace/liushuo/cupid/openvla-oft/tmp/can_mh/filter_episode10_window_joint_traj_eval \
    --quality-source trajectory_eval \
    --episode-influence-weight 0.5 \
    --episode-quality-weight 0.5 \
    --episode-drop-ratio 0.10 \
    --window-size 50 \
    --window-stride 50 \
    --window-local-episode-score-weight 0.4 \
    --window-local-quality-weight 0.4 \
    --window-local-episode-joint-weight 0.2\
    > logs/filter_joint_episode_window.log 2>&1 &

Stage 2:
- generate per-episode windows from the dataset
- score each window with a joint contribution term that combines:
  - local window value
  - episode-level global prior
- select windows greedily while balancing contribution and coverage
- support automatic budget stopping based on marginal gain + coverage constraints

This script is designed for LeRobot datasets where score.pkl often contains
episode-level influence only. Window-level influence is optional; if missing,
the episode prior is reused for window scoring.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from eval_cupid_quality import (  # noqa: E402
    _lower_better_to_score,
    _quality_from_episode_eval,
    _quality_from_trajectory_eval,
)


@dataclass
class WindowRecord:
    episode: int
    start: int
    end: int
    center: float
    feature: np.ndarray
    raw_action_jitter: float
    raw_state_jitter: float
    raw_value: float
    window_episode_score: float = 0.0
    window_quality_score: float = 0.0
    value_score: float = 0.0
    prior_score: float = 0.0
    contribution_score: float = 0.0
    keep: bool = False
    keep_rank: int = -1
    keep_gain: float = float("nan")
    keep_novelty: float = float("nan")


def _normalize_to_unit_interval(values: np.ndarray) -> np.ndarray:
    vals = np.asarray(values, dtype=np.float64)
    out = np.zeros_like(vals, dtype=np.float64)
    finite = np.isfinite(vals)
    if not finite.any():
        return out
    v = vals[finite]
    lo = float(np.min(v))
    hi = float(np.max(v))
    if np.isclose(lo, hi):
        out[finite] = 1.0
        return out
    out[finite] = (v - lo) / (hi - lo)
    return np.clip(out, 0.0, 1.0)


def _quality_from_bad_ratio(ep_bad: np.ndarray) -> np.ndarray:
    return np.clip(1.0 - np.asarray(ep_bad, dtype=np.float64), 0.0, None)


def _quality_from_quality_csv(csv_path: Path, num_episodes: int, default_value: float) -> np.ndarray:
    if not csv_path.exists():
        raise FileNotFoundError(f"quality score csv not found: {csv_path}")

    quality = np.full((num_episodes,), np.nan, dtype=np.float64)
    parsed = 0
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        required_cols = {"episode", "quality_score"}
        if not required_cols.issubset(set(reader.fieldnames or [])):
            raise ValueError(f"{csv_path} must contain columns {sorted(required_cols)}, got {reader.fieldnames}")
        for row in reader:
            try:
                episode = int(row.get("episode", "-1"))
                score = float(row.get("quality_score", "nan"))
            except (TypeError, ValueError):
                continue
            if 0 <= episode < num_episodes and np.isfinite(score):
                quality[episode] = score
                parsed += 1
    if parsed == 0:
        raise ValueError(f"No valid rows parsed from {csv_path}")
    missing = ~np.isfinite(quality)
    if missing.any():
        quality[missing] = float(default_value)
    return np.clip(quality, 0.0, None)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Two-stage OpenVLA selection with episode filtering and coverage-aware window selection."
    )
    parser.add_argument("--score-pkl", type=Path, required=True, help="OpenVLA score.pkl path.")
    parser.add_argument("--dataset-root", type=Path, required=True, help="LeRobot dataset root.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for selection artifacts.")

    parser.add_argument(
        "--quality-source",
        type=str,
        default="trajectory_eval",
        choices=("quality_csv", "episode_eval", "trajectory_eval", "bad_ratio"),
        help="How to compute episode-level quality.",
    )
    parser.add_argument("--quality-score-csv", type=Path, default=None, help="quality_episodes.csv for quality_csv mode.")
    parser.add_argument("--quality-score-default", type=float, default=0.5)

    parser.add_argument("--episode-influence-weight", type=float, default=0.5)
    parser.add_argument("--episode-quality-weight", type=float, default=0.5)
    parser.add_argument("--episode-drop-ratio", type=float, default=0.10)

    parser.add_argument("--window-size", type=int, default=50)
    parser.add_argument("--window-stride", type=int, default=50)
    parser.add_argument("--window-value-action-weight", type=float, default=0.5)
    parser.add_argument("--window-value-state-weight", type=float, default=0.5)
    parser.add_argument(
        "--window-local-episode-score-weight",
        type=float,
        default=0.4,
        help="Weight of window_episode_score in window_local_value.",
    )
    parser.add_argument(
        "--window-local-quality-weight",
        type=float,
        default=0.4,
        help="Weight of window_quality_score in window_local_value.",
    )
    parser.add_argument(
        "--window-local-episode-joint-weight",
        type=float,
        default=0.2,
        help="Weight of episode_score in window_local_value.",
    )
    parser.add_argument("--window-contribution-weight", type=float, default=0.7)
    parser.add_argument("--window-coverage-weight", type=float, default=0.3)

    parser.add_argument(
        "--budget-mode",
        type=str,
        default="auto_marginal",
        choices=("auto_marginal", "fixed_ratio"),
        help="Window budget policy. auto_marginal is the default recommended mode.",
    )
    parser.add_argument("--window-keep-ratio", type=float, default=0.5, help="Used only in fixed_ratio mode.")
    parser.add_argument("--min-windows-per-episode", type=int, default=1)
    parser.add_argument(
        "--min-coverage-ratio",
        type=float,
        default=0.5,
        help="Minimum temporal coverage before auto_marginal is allowed to stop.",
    )
    parser.add_argument(
        "--marginal-gain-threshold",
        type=float,
        default=0.15,
        help="Stop auto_marginal selection when best marginal gain drops below this value.",
    )
    parser.add_argument(
        "--relative-gain-threshold",
        type=float,
        default=0.5,
        help="Stop auto_marginal when gain drops below this fraction of the first selected gain.",
    )

    parser.add_argument(
        "--coverage-feature",
        type=str,
        default="auto",
        choices=("auto", "state_mean", "temporal_center"),
        help="Window coverage representation.",
    )
    parser.add_argument("--quality-eval-action-weight", type=float, default=0.5)
    parser.add_argument("--quality-eval-state-weight", type=float, default=0.3)
    parser.add_argument("--quality-eval-length-weight", type=float, default=0.2)
    parser.add_argument("--quality-eval-energy-weight", type=float, default=0.0)
    parser.add_argument("--quality-eval-length-scale", type=float, default=1.0)
    parser.add_argument("--quality-eval-default", type=float, default=0.5)
    parser.add_argument("--quality-eval-jitter-mode", type=str, default="absolute", choices=("relative", "absolute"))
    parser.add_argument("--quality-eval-normalizer", type=str, default="quantile", choices=("quantile", "minmax"))
    parser.add_argument("--quality-eval-quantile-low", type=float, default=0.05)
    parser.add_argument("--quality-eval-quantile-high", type=float, default=0.95)
    parser.add_argument("--traj-eval-terminal-weight", type=float, default=0.25)
    parser.add_argument("--traj-eval-efficiency-weight", type=float, default=0.20)
    parser.add_argument("--traj-eval-progress-weight", type=float, default=0.25)
    parser.add_argument("--traj-eval-state-smooth-weight", type=float, default=0.15)
    parser.add_argument("--traj-eval-action-smooth-weight", type=float, default=0.10)
    parser.add_argument("--traj-eval-length-weight", type=float, default=0.05)
    parser.add_argument("--traj-eval-terminal-knn-k", type=int, default=5)
    parser.add_argument("--traj-eval-monotonic-weight", type=float, default=0.15)
    parser.add_argument("--traj-eval-backtrack-weight", type=float, default=0.10)
    parser.add_argument("--traj-eval-stall-weight", type=float, default=0.10)
    parser.add_argument("--traj-eval-fusion", type=str, default="weighted_sum", choices=("weighted_sum", "weighted_geomean"))
    parser.add_argument("--traj-eval-min-terminal-score", type=float, default=-1.0)
    parser.add_argument("--traj-eval-min-progress-score", type=float, default=-1.0)
    parser.add_argument("--traj-eval-gate-quality-cap", type=float, default=-1.0)
    parser.add_argument("--traj-eval-terminal-distance-metric", type=str, default="l2", choices=("l2", "l1", "cosine"))
    parser.add_argument("--traj-eval-terminal-normalize", type=str, default="none", choices=("none", "zscore", "robust_zscore"))
    parser.add_argument("--traj-eval-terminal-agg", type=str, default="mean", choices=("mean", "median", "trimmed_mean"))
    parser.add_argument("--traj-eval-terminal-trim-ratio", type=float, default=0.1)
    return parser.parse_args()


def _load_pickle(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        return pickle.load(f)


def _list_episode_files(dataset_root: Path) -> dict[int, Path]:
    episode_files: dict[int, Path] = {}
    for path in sorted((dataset_root / "data").glob("chunk-*/episode_*.parquet")):
        stem = path.stem
        if not stem.startswith("episode_"):
            continue
        ep_str = stem.split("_")[-1]
        if ep_str.isdigit():
            episode_files[int(ep_str)] = path
    return episode_files


def _load_episode_lengths_from_meta(dataset_root: Path) -> dict[int, int]:
    path = dataset_root / "meta" / "episodes.jsonl"
    lengths: dict[int, int] = {}
    if not path.exists():
        return lengths
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                episode = int(row["episode_index"])
                length = int(row["length"])
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                continue
            lengths[episode] = length
    return lengths


def _resolve_all_episodes(
    metadata: dict[str, Any],
    episode_files: dict[int, Path],
    meta_episode_lengths: dict[int, int],
) -> list[int]:
    if episode_files:
        return sorted(episode_files.keys())
    if meta_episode_lengths:
        return sorted(meta_episode_lengths.keys())
    episode_lengths = np.asarray(metadata.get("episode_lengths", []), dtype=np.int64).reshape(-1)
    if episode_lengths.size > 0:
        return list(range(int(episode_lengths.shape[0])))
    raise ValueError("Unable to determine episode ids from score metadata or dataset_root/data.")


def _compute_episode_bad_ratio(metadata: dict[str, Any], num_episodes: int) -> np.ndarray:
    window_drop = metadata.get("window_drop_mask")
    window_ep = metadata.get("window_episode_index")
    if window_drop is None or window_ep is None:
        return np.zeros((num_episodes,), dtype=np.float64)
    drop = np.asarray(window_drop, dtype=np.float64).reshape(-1)
    episode = np.asarray(window_ep, dtype=np.int64).reshape(-1)
    bad = np.zeros((num_episodes,), dtype=np.float64)
    counts = np.zeros((num_episodes,), dtype=np.float64)
    for ep, val in zip(episode.tolist(), drop.tolist(), strict=True):
        if 0 <= ep < num_episodes:
            bad[ep] += float(val)
            counts[ep] += 1.0
    valid = counts > 0
    bad[valid] = bad[valid] / counts[valid]
    return bad


def _resolve_episode_influence(payload: dict[str, Any], all_eps: list[int]) -> np.ndarray:
    metadata = payload.get("metadata", {})
    num_episodes = len(all_eps)
    if "episode_influence_scores" in metadata:
        scores = np.asarray(metadata["episode_influence_scores"], dtype=np.float64).reshape(-1)
        if scores.shape[0] >= num_episodes:
            return scores[:num_episodes]
    if "influence_scores" in payload:
        scores = np.asarray(payload["influence_scores"], dtype=np.float64).reshape(-1)
        if scores.shape[0] == num_episodes:
            return scores
    raise ValueError(
        "Could not resolve episode-level influence. "
        "Expected metadata.episode_influence_scores or influence_scores aligned with episodes."
    )


def _compute_episode_quality(args: argparse.Namespace, metadata: dict[str, Any], all_eps: list[int]) -> np.ndarray:
    num_episodes = len(all_eps)
    if args.quality_source == "quality_csv":
        if args.quality_score_csv is None:
            raise ValueError("--quality-score-csv is required when --quality-source=quality_csv")
        return _quality_from_quality_csv(
            csv_path=args.quality_score_csv,
            num_episodes=num_episodes,
            default_value=args.quality_score_default,
        )
    if args.quality_source == "episode_eval":
        return _quality_from_episode_eval(
            dataset_root=args.dataset_root,
            all_eps=all_eps,
            action_weight=args.quality_eval_action_weight,
            state_weight=args.quality_eval_state_weight,
            length_weight=args.quality_eval_length_weight,
            energy_weight=args.quality_eval_energy_weight,
            length_scale=args.quality_eval_length_scale,
            default_value=args.quality_eval_default,
            jitter_mode=args.quality_eval_jitter_mode,
            normalizer=args.quality_eval_normalizer,
            quantile_low=args.quality_eval_quantile_low,
            quantile_high=args.quality_eval_quantile_high,
        )
    if args.quality_source == "trajectory_eval":
        return _quality_from_trajectory_eval(
            dataset_root=args.dataset_root,
            all_eps=all_eps,
            terminal_weight=args.traj_eval_terminal_weight,
            efficiency_weight=args.traj_eval_efficiency_weight,
            progress_weight=args.traj_eval_progress_weight,
            state_smooth_weight=args.traj_eval_state_smooth_weight,
            action_smooth_weight=args.traj_eval_action_smooth_weight,
            length_weight=args.traj_eval_length_weight,
            terminal_knn_k=args.traj_eval_terminal_knn_k,
            monotonic_weight=args.traj_eval_monotonic_weight,
            backtrack_weight=args.traj_eval_backtrack_weight,
            stall_weight=args.traj_eval_stall_weight,
            fusion=args.traj_eval_fusion,
            min_terminal_score=args.traj_eval_min_terminal_score,
            min_progress_score=args.traj_eval_min_progress_score,
            gate_quality_cap=args.traj_eval_gate_quality_cap,
            length_scale=args.quality_eval_length_scale,
            default_value=args.quality_eval_default,
            normalizer=args.quality_eval_normalizer,
            quantile_low=args.quality_eval_quantile_low,
            quantile_high=args.quality_eval_quantile_high,
            terminal_distance_metric=args.traj_eval_terminal_distance_metric,
            terminal_normalize=args.traj_eval_terminal_normalize,
            terminal_agg=args.traj_eval_terminal_agg,
            terminal_trim_ratio=args.traj_eval_terminal_trim_ratio,
        )
    ep_bad = _compute_episode_bad_ratio(metadata, num_episodes)
    return _quality_from_bad_ratio(ep_bad)


def _drop_bottom_fraction(scores: np.ndarray, drop_ratio: float) -> tuple[np.ndarray, np.ndarray]:
    n = int(scores.shape[0])
    drop_n = int(np.floor(np.clip(drop_ratio, 0.0, 1.0) * n))
    order = np.argsort(scores, kind="mergesort")
    keep_mask = np.ones((n,), dtype=bool)
    if drop_n > 0:
        keep_mask[order[:drop_n]] = False
    return keep_mask, order[:drop_n]


def _load_episode_arrays(path: Path) -> tuple[np.ndarray | None, np.ndarray | None]:
    try:
        import pyarrow.parquet as pq
    except Exception:
        return None, None

    schema = pq.read_schema(path)
    available = set(schema.names)
    cols = [col for col in ("action", "observation.state") if col in available]
    table = pq.read_table(path, columns=cols if cols else None)

    action = None
    state = None
    if "action" in table.column_names:
        action = np.asarray(table["action"].to_pylist(), dtype=np.float64)
    if "observation.state" in table.column_names:
        state = np.asarray(table["observation.state"].to_pylist(), dtype=np.float64)
    return action, state


def _window_ranges(length: int, window_size: int, stride: int) -> list[tuple[int, int]]:
    if length <= 0:
        return []
    if length <= window_size:
        return [(0, length)]
    out: list[tuple[int, int]] = []
    stride = max(int(stride), 1)
    for start in range(0, length - window_size + 1, stride):
        out.append((start, start + window_size))
    if out and out[-1][1] < length:
        out.append((length - window_size, length))
    return out


def _mean_diff_norm(values: np.ndarray | None) -> float:
    if values is None or values.ndim != 2 or values.shape[0] < 2:
        return float("nan")
    return float(np.linalg.norm(np.diff(values, axis=0), axis=1).mean())


def _build_window_feature(
    state_slice: np.ndarray | None,
    start: int,
    end: int,
    length: int,
    feature_mode: str,
) -> np.ndarray:
    if feature_mode == "state_mean" and state_slice is not None and state_slice.ndim == 2 and state_slice.size > 0:
        return np.asarray(state_slice.mean(axis=0), dtype=np.float64)
    center = 0.5 * (float(start) + float(end))
    denom = max(float(length), 1.0)
    return np.asarray([center / denom], dtype=np.float64)


def _minmax(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return arr
    lo = float(np.min(arr))
    hi = float(np.max(arr))
    if np.isclose(lo, hi):
        return np.ones_like(arr)
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0)


def _build_windows_for_episode(
    episode: int,
    path: Path | None,
    window_size: int,
    stride: int,
    feature_mode: str,
    fallback_length: int | None = None,
) -> list[WindowRecord]:
    action = None
    state = None
    if path is not None:
        action, state = _load_episode_arrays(path)
    length = 0
    if action is not None:
        length = int(action.shape[0])
    elif state is not None:
        length = int(state.shape[0])
    elif fallback_length is not None:
        length = int(fallback_length)
    ranges = _window_ranges(length, window_size, stride)
    windows: list[WindowRecord] = []
    for start, end in ranges:
        action_slice = None if action is None else action[start:end]
        state_slice = None if state is None else state[start:end]
        raw_action_jitter = _mean_diff_norm(action_slice)
        raw_state_jitter = _mean_diff_norm(state_slice)
        feature = _build_window_feature(state_slice, start, end, length, feature_mode)
        center = 0.5 * (float(start) + float(end))
        windows.append(
            WindowRecord(
                episode=episode,
                start=int(start),
                end=int(end),
                center=center,
                feature=feature,
                raw_action_jitter=raw_action_jitter,
                raw_state_jitter=raw_state_jitter,
                raw_value=float("nan"),
            )
        )
    return windows


def _assign_window_value_scores(
    windows_by_episode: dict[int, list[WindowRecord]],
    *,
    action_weight: float,
    state_weight: float,
) -> None:
    all_windows = [window for windows in windows_by_episode.values() for window in windows]
    action_raw = np.asarray([window.raw_action_jitter for window in all_windows], dtype=np.float64)
    state_raw = np.asarray([window.raw_state_jitter for window in all_windows], dtype=np.float64)
    action_score = _lower_better_to_score(action_raw, normalizer="quantile", quantile_low=0.05, quantile_high=0.95)
    state_score = _lower_better_to_score(state_raw, normalizer="quantile", quantile_low=0.05, quantile_high=0.95)

    for idx, window in enumerate(all_windows):
        num = 0.0
        den = 0.0
        if action_weight > 0.0 and np.isfinite(action_score[idx]):
            num += float(action_weight) * float(action_score[idx])
            den += float(action_weight)
        if state_weight > 0.0 and np.isfinite(state_score[idx]):
            num += float(state_weight) * float(state_score[idx])
            den += float(state_weight)
        window.raw_value = num / den if den > 0 else 0.5

    for windows in windows_by_episode.values():
        values = np.asarray([window.raw_value for window in windows], dtype=np.float64)
        normalized = _minmax(values)
        for window, score in zip(windows, normalized.tolist(), strict=True):
            window.window_quality_score = float(score)


def _build_window_episode_score_lookup(
    metadata: dict[str, Any],
) -> dict[tuple[int, int, int], float]:
    raw_scores = metadata.get("window_influence_scores")
    if raw_scores is None:
        raw_scores = metadata.get("window_weights")
    window_episode = metadata.get("window_episode_index")
    window_start = metadata.get("window_start")
    window_end = metadata.get("window_end")
    if raw_scores is None or window_episode is None or window_start is None or window_end is None:
        return {}

    scores = np.asarray(raw_scores, dtype=np.float64).reshape(-1)
    episodes = np.asarray(window_episode, dtype=np.int64).reshape(-1)
    starts = np.asarray(window_start, dtype=np.int64).reshape(-1)
    ends = np.asarray(window_end, dtype=np.int64).reshape(-1)
    if not (scores.shape[0] == episodes.shape[0] == starts.shape[0] == ends.shape[0]):
        return {}

    normalized = _minmax(scores)
    lookup: dict[tuple[int, int, int], float] = {}
    for ep, start, end, score in zip(episodes.tolist(), starts.tolist(), ends.tolist(), normalized.tolist(), strict=True):
        lookup[(int(ep), int(start), int(end))] = float(score)
    return lookup


def _assign_window_local_values(
    windows_by_episode: dict[int, list[WindowRecord]],
    *,
    window_episode_score_lookup: dict[tuple[int, int, int], float],
    norm_episode_influence: np.ndarray,
    episode_joint_scores: np.ndarray,
    local_episode_score_weight: float,
    local_quality_weight: float,
    local_episode_joint_weight: float,
) -> None:
    for ep, windows in windows_by_episode.items():
        episode_score = float(episode_joint_scores[ep]) if ep < len(episode_joint_scores) else 0.0
        episode_influence_score = float(norm_episode_influence[ep]) if ep < len(norm_episode_influence) else 0.0
        for window in windows:
            key = (int(window.episode), int(window.start), int(window.end))
            window.window_episode_score = float(window_episode_score_lookup.get(key, episode_influence_score))
            window.prior_score = episode_score
            num = (
                float(local_episode_score_weight) * float(window.window_episode_score)
                + float(local_quality_weight) * float(window.window_quality_score)
                + float(local_episode_joint_weight) * float(window.prior_score)
            )
            den = float(local_episode_score_weight) + float(local_quality_weight) + float(local_episode_joint_weight)
            window.value_score = num / max(den, 1e-8)
            window.contribution_score = float(window.value_score)


def _window_novelty_score(
    candidate: WindowRecord,
    selected: list[WindowRecord],
    max_distance: float,
) -> float:
    if not selected:
        return 1.0
    if max_distance <= 1e-8:
        return 0.0
    selected_features = np.stack([window.feature for window in selected], axis=0)
    dist = np.linalg.norm(selected_features - candidate.feature[None, :], axis=1)
    return float(np.clip(np.min(dist) / max_distance, 0.0, 1.0))


def _episode_temporal_coverage(selected: list[WindowRecord], episode_length: int) -> float:
    if episode_length <= 0 or not selected:
        return 0.0
    covered = np.zeros((episode_length,), dtype=bool)
    for window in selected:
        covered[window.start : window.end] = True
    return float(np.mean(covered))


def _pairwise_max_distance(windows: list[WindowRecord]) -> float:
    if len(windows) <= 1:
        return 1.0
    feats = np.stack([window.feature for window in windows], axis=0)
    max_dist = 0.0
    for idx in range(feats.shape[0]):
        dist = np.linalg.norm(feats[idx + 1 :] - feats[idx], axis=1)
        if dist.size > 0:
            max_dist = max(max_dist, float(np.max(dist)))
    return max(max_dist, 1.0)


def _select_windows_for_episode(
    windows: list[WindowRecord],
    *,
    episode_length: int,
    budget_mode: str,
    window_keep_ratio: float,
    min_windows_per_episode: int,
    min_coverage_ratio: float,
    marginal_gain_threshold: float,
    relative_gain_threshold: float,
    contribution_weight: float,
    coverage_weight: float,
) -> dict[str, float | int]:
    if not windows:
        return {
            "selected_count": 0,
            "coverage_ratio": 0.0,
            "selection_ratio": 0.0,
            "mean_selected_gain": float("nan"),
        }

    selected: list[WindowRecord] = []
    remaining = list(range(len(windows)))
    max_distance = _pairwise_max_distance(windows)
    max_keep = len(windows)
    if budget_mode == "fixed_ratio":
        target_keep = max(min_windows_per_episode, int(round(window_keep_ratio * len(windows))))
        target_keep = min(max(target_keep, 1), len(windows))
    else:
        target_keep = len(windows)

    rank = 0
    first_gain = None
    while remaining and len(selected) < target_keep:
        best_idx = -1
        best_gain = -1.0
        best_novelty = -1.0
        for idx in remaining:
            window = windows[idx]
            novelty = _window_novelty_score(window, selected, max_distance)
            gain = float(contribution_weight) * float(window.contribution_score) + float(coverage_weight) * novelty
            if gain > best_gain:
                best_idx = idx
                best_gain = gain
                best_novelty = novelty

        chosen = windows[best_idx]
        chosen.keep = True
        chosen.keep_rank = rank
        chosen.keep_gain = float(best_gain)
        chosen.keep_novelty = float(best_novelty)
        rank += 1
        selected.append(chosen)
        remaining.remove(best_idx)
        if first_gain is None:
            first_gain = float(best_gain)

        if budget_mode == "auto_marginal":
            coverage = _episode_temporal_coverage(selected, episode_length)
            adaptive_threshold = float(marginal_gain_threshold)
            if first_gain is not None:
                adaptive_threshold = max(
                    adaptive_threshold,
                    float(relative_gain_threshold) * float(first_gain),
                )
            if (
                len(selected) >= max(int(min_windows_per_episode), 1)
                and coverage >= float(min_coverage_ratio)
                and best_gain < adaptive_threshold
            ):
                break

    coverage = _episode_temporal_coverage(selected, episode_length)
    gains = np.asarray([window.keep_gain for window in selected if np.isfinite(window.keep_gain)], dtype=np.float64)
    return {
        "selected_count": int(len(selected)),
        "coverage_ratio": float(coverage),
        "selection_ratio": float(len(selected) / max(len(windows), 1)),
        "mean_selected_gain": float(np.mean(gains)) if gains.size > 0 else float("nan"),
    }


def _write_outputs(
    output_dir: Path,
    *,
    args: argparse.Namespace,
    all_eps: list[int],
    keep_episode_mask: np.ndarray,
    dropped_episode_indices: np.ndarray,
    episode_influence: np.ndarray,
    episode_quality: np.ndarray,
    norm_episode_influence: np.ndarray,
    norm_episode_quality: np.ndarray,
    episode_joint_scores: np.ndarray,
    windows_by_episode: dict[int, list[WindowRecord]],
    episode_lengths: dict[int, int],
    episode_window_stats: dict[int, dict[str, float | int]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    kept_episodes = [int(ep) for ep in all_eps if keep_episode_mask[ep]]
    dropped_episodes = [int(ep) for ep in dropped_episode_indices.tolist()]

    with (output_dir / "kept_episodes.txt").open("w") as f:
        f.write("\n".join(str(ep) for ep in kept_episodes))
    with (output_dir / "dropped_episodes.txt").open("w") as f:
        f.write("\n".join(str(ep) for ep in dropped_episodes))

    with (output_dir / "episode_scores.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "episode",
                "keep_episode",
                "episode_influence",
                "episode_quality",
                "norm_episode_influence",
                "norm_episode_quality",
                "episode_joint_score",
                "episode_length",
                "num_windows",
                "selected_windows",
                "window_selection_ratio",
                "window_coverage_ratio",
                "mean_selected_gain",
            ]
        )
        for ep in all_eps:
            stats = episode_window_stats.get(int(ep), {})
            windows = windows_by_episode.get(int(ep), [])
            writer.writerow(
                [
                    int(ep),
                    int(bool(keep_episode_mask[ep])),
                    float(episode_influence[ep]),
                    float(episode_quality[ep]),
                    float(norm_episode_influence[ep]),
                    float(norm_episode_quality[ep]),
                    float(episode_joint_scores[ep]),
                    int(episode_lengths.get(int(ep), 0)),
                    int(len(windows)),
                    int(stats.get("selected_count", 0)),
                    float(stats.get("selection_ratio", 0.0)),
                    float(stats.get("coverage_ratio", 0.0)),
                    float(stats.get("mean_selected_gain", float("nan"))),
                ]
            )

    with (output_dir / "window_scores.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "episode",
                "start",
                "end",
                "center",
                "keep_window",
                "keep_rank",
                "keep_gain",
                "keep_novelty",
                "window_episode_score",
                "window_quality_score",
                "window_local_value",
                "episode_score",
                "window_contribution_score",
                "raw_action_jitter",
                "raw_state_jitter",
            ]
        )
        for ep in sorted(windows_by_episode):
            for window in windows_by_episode[ep]:
                writer.writerow(
                    [
                        int(window.episode),
                        int(window.start),
                        int(window.end),
                        float(window.center),
                        int(bool(window.keep)),
                        int(window.keep_rank),
                        float(window.keep_gain),
                        float(window.keep_novelty),
                        float(window.window_episode_score),
                        float(window.window_quality_score),
                        float(window.value_score),
                        float(window.prior_score),
                        float(window.contribution_score),
                        float(window.raw_action_jitter),
                        float(window.raw_state_jitter),
                    ]
                )

    flat_episodes: list[int] = []
    flat_start: list[int] = []
    flat_end: list[int] = []
    flat_keep: list[int] = []
    flat_contribution: list[float] = []
    flat_window_episode_score: list[float] = []
    flat_window_quality_score: list[float] = []
    flat_window_local_value: list[float] = []
    for ep in sorted(windows_by_episode):
        for window in windows_by_episode[ep]:
            flat_episodes.append(int(window.episode))
            flat_start.append(int(window.start))
            flat_end.append(int(window.end))
            flat_keep.append(int(bool(window.keep)))
            flat_contribution.append(float(window.contribution_score))
            flat_window_episode_score.append(float(window.window_episode_score))
            flat_window_quality_score.append(float(window.window_quality_score))
            flat_window_local_value.append(float(window.value_score))
    np.savez_compressed(
        output_dir / "joint_episode_window_selection.npz",
        episode_keep_mask=np.asarray(keep_episode_mask, dtype=np.int64),
        episode_joint_scores=np.asarray(episode_joint_scores, dtype=np.float32),
        window_episode=np.asarray(flat_episodes, dtype=np.int64),
        window_start=np.asarray(flat_start, dtype=np.int64),
        window_end=np.asarray(flat_end, dtype=np.int64),
        window_keep=np.asarray(flat_keep, dtype=np.int64),
        window_episode_score=np.asarray(flat_window_episode_score, dtype=np.float32),
        window_quality_score=np.asarray(flat_window_quality_score, dtype=np.float32),
        window_local_value=np.asarray(flat_window_local_value, dtype=np.float32),
        window_contribution=np.asarray(flat_contribution, dtype=np.float32),
    )

    total_windows = sum(len(windows) for windows in windows_by_episode.values())
    kept_windows = sum(int(window.keep) for windows in windows_by_episode.values() for window in windows)
    summary = {
        "method": "joint_episode_window_selection",
        "episode_stage": {
            "drop_ratio": float(args.episode_drop_ratio),
            "kept_episodes": int(sum(bool(keep_episode_mask[ep]) for ep in all_eps)),
            "dropped_episodes": int(len(dropped_episode_indices)),
        },
        "window_stage": {
            "budget_mode": args.budget_mode,
            "total_windows": int(total_windows),
            "kept_windows": int(kept_windows),
            "kept_window_ratio": float(kept_windows / max(total_windows, 1)),
            "min_coverage_ratio": float(args.min_coverage_ratio),
            "marginal_gain_threshold": float(args.marginal_gain_threshold),
            "relative_gain_threshold": float(args.relative_gain_threshold),
        },
    }
    (output_dir / "selection_summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.episode_drop_ratio < 1.0:
        raise ValueError("--episode-drop-ratio must be in [0, 1).")
    if args.window_size <= 0 or args.window_stride <= 0:
        raise ValueError("--window-size and --window-stride must be > 0.")
    if args.episode_influence_weight < 0 or args.episode_quality_weight < 0:
        raise ValueError("Episode weights must be >= 0.")
    if np.isclose(args.episode_influence_weight + args.episode_quality_weight, 0.0):
        raise ValueError("At least one episode score weight must be > 0.")
    if (
        args.window_local_episode_score_weight < 0
        or args.window_local_quality_weight < 0
        or args.window_local_episode_joint_weight < 0
    ):
        raise ValueError("Window local-value weights must be >= 0.")
    if np.isclose(
        args.window_local_episode_score_weight
        + args.window_local_quality_weight
        + args.window_local_episode_joint_weight,
        0.0,
    ):
        raise ValueError("At least one window local-value weight must be > 0.")

    payload = _load_pickle(args.score_pkl)
    metadata = payload.get("metadata", {})
    episode_files = _list_episode_files(args.dataset_root)
    meta_episode_lengths = _load_episode_lengths_from_meta(args.dataset_root)
    all_eps = _resolve_all_episodes(metadata, episode_files, meta_episode_lengths)
    episode_influence = _resolve_episode_influence(payload, all_eps)
    episode_quality = _compute_episode_quality(args, metadata, all_eps)

    norm_episode_influence = _normalize_to_unit_interval(episode_influence)
    norm_episode_quality = _normalize_to_unit_interval(episode_quality)
    denom = float(args.episode_influence_weight) + float(args.episode_quality_weight)
    episode_joint_scores = (
        float(args.episode_influence_weight) * norm_episode_influence
        + float(args.episode_quality_weight) * norm_episode_quality
    ) / max(denom, 1e-8)

    keep_episode_mask, dropped_episode_indices = _drop_bottom_fraction(episode_joint_scores, args.episode_drop_ratio)

    windows_by_episode: dict[int, list[WindowRecord]] = {}
    episode_lengths: dict[int, int] = {}
    feature_mode = args.coverage_feature
    for ep in all_eps:
        if not keep_episode_mask[ep]:
            continue
        path = episode_files.get(int(ep))
        fallback_length = meta_episode_lengths.get(int(ep), 0)
        if path is None and fallback_length <= 0:
            continue
        mode = feature_mode
        if mode == "auto":
            mode = "state_mean"
        windows = _build_windows_for_episode(
            episode=int(ep),
            path=path,
            window_size=int(args.window_size),
            stride=int(args.window_stride),
            feature_mode=mode,
            fallback_length=fallback_length,
        )
        if mode == "state_mean" and any(window.feature.size == 1 for window in windows):
            windows = _build_windows_for_episode(
                episode=int(ep),
                path=path,
                window_size=int(args.window_size),
                stride=int(args.window_stride),
                feature_mode="temporal_center",
                fallback_length=fallback_length,
            )
        windows_by_episode[int(ep)] = windows
        if windows:
            episode_lengths[int(ep)] = max(window.end for window in windows)
        else:
            episode_lengths[int(ep)] = int(fallback_length)

    _assign_window_value_scores(
        windows_by_episode,
        action_weight=max(float(args.window_value_action_weight), 0.0),
        state_weight=max(float(args.window_value_state_weight), 0.0),
    )
    window_episode_score_lookup = _build_window_episode_score_lookup(metadata)
    _assign_window_local_values(
        windows_by_episode,
        window_episode_score_lookup=window_episode_score_lookup,
        norm_episode_influence=norm_episode_influence,
        episode_joint_scores=episode_joint_scores,
        local_episode_score_weight=max(float(args.window_local_episode_score_weight), 0.0),
        local_quality_weight=max(float(args.window_local_quality_weight), 0.0),
        local_episode_joint_weight=max(float(args.window_local_episode_joint_weight), 0.0),
    )

    episode_window_stats: dict[int, dict[str, float | int]] = {}
    for ep, windows in windows_by_episode.items():
        stats = _select_windows_for_episode(
            windows,
            episode_length=episode_lengths.get(ep, 0),
            budget_mode=args.budget_mode,
            window_keep_ratio=float(args.window_keep_ratio),
            min_windows_per_episode=int(args.min_windows_per_episode),
            min_coverage_ratio=float(args.min_coverage_ratio),
            marginal_gain_threshold=float(args.marginal_gain_threshold),
            relative_gain_threshold=float(args.relative_gain_threshold),
            contribution_weight=float(args.window_contribution_weight),
            coverage_weight=float(args.window_coverage_weight),
        )
        episode_window_stats[ep] = stats

    _write_outputs(
        args.output_dir,
        args=args,
        all_eps=all_eps,
        keep_episode_mask=keep_episode_mask,
        dropped_episode_indices=dropped_episode_indices,
        episode_influence=episode_influence,
        episode_quality=episode_quality,
        norm_episode_influence=norm_episode_influence,
        norm_episode_quality=norm_episode_quality,
        episode_joint_scores=episode_joint_scores,
        windows_by_episode=windows_by_episode,
        episode_lengths=episode_lengths,
        episode_window_stats=episode_window_stats,
    )

    kept_eps = int(np.sum(keep_episode_mask))
    kept_windows = sum(int(window.keep) for windows in windows_by_episode.values() for window in windows)
    total_windows = sum(len(windows) for windows in windows_by_episode.values())
    print(
        f"[done] kept episodes {kept_eps}/{len(all_eps)}, "
        f"kept windows {kept_windows}/{max(total_windows, 1)} "
        f"(budget_mode={args.budget_mode})"
    )


if __name__ == "__main__":
    main()
