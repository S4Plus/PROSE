#!/usr/bin/env python3
"""Window-level joint filtering with an episode-level prior.

This script extends the existing window-level joint filtering pipeline by
adding a soft episode prior. The base score is:

    final_window_score =
        window_score_weight * normalize(window_joint_score)
        + episode_prior_weight * normalize(episode_joint_score_of_its_episode)

It also supports three incremental extensions:

1. Adaptive episode prior weights
2. A window-episode interaction term
3. Episode-aware budgeted window retention

Compared with hard episode gating, this preserves fine-grained window
selection while still preferring windows from globally better episodes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

import openpi.training.config as _config
import retrain as _retrain
import retrain_grass as retrain_grass
import retrain_window_joint as base


def _aggregate_episode_quality(
    window_episode_indices: np.ndarray,
    window_quality_scores: np.ndarray,
    *,
    agg: str,
) -> tuple[np.ndarray, np.ndarray]:
    unique_episode_indices = np.asarray(sorted(np.unique(window_episode_indices).tolist()), dtype=np.int64)
    episode_quality_scores = np.zeros((len(unique_episode_indices),), dtype=np.float32)
    for out_idx, episode_idx in enumerate(unique_episode_indices.tolist()):
        values = np.asarray(window_quality_scores[window_episode_indices == episode_idx], dtype=np.float32)
        if values.size == 0:
            episode_quality_scores[out_idx] = 0.0
        elif agg == "max":
            episode_quality_scores[out_idx] = float(np.max(values))
        else:
            episode_quality_scores[out_idx] = float(np.mean(values))
    return unique_episode_indices, episode_quality_scores


def _align_episode_influence(
    payload: dict[str, Any],
    target_episode_indices: np.ndarray,
) -> np.ndarray:
    if "episode_influence_scores" not in payload:
        raise ValueError(f"{payload} does not contain episode_influence_scores.")

    meta = payload["metadata"]
    if "train_episode_indices" not in meta:
        raise ValueError("Window score pickle is missing metadata.train_episode_indices.")

    score_episode_indices = np.asarray(meta["train_episode_indices"], dtype=np.int64).reshape(-1)
    influence_scores = np.asarray(payload["episode_influence_scores"], dtype=np.float32).reshape(-1)
    if score_episode_indices.shape[0] != influence_scores.shape[0]:
        raise ValueError("Episode influence score shape mismatch in score pickle.")

    influence_map = {
        int(ep): float(score)
        for ep, score in zip(score_episode_indices.tolist(), influence_scores.tolist(), strict=True)
    }
    missing = [int(ep) for ep in target_episode_indices.tolist() if int(ep) not in influence_map]
    if missing:
        raise ValueError(f"Episode prior is missing influence scores for episodes: {missing[:10]}")
    return np.asarray([influence_map[int(ep)] for ep in target_episode_indices.tolist()], dtype=np.float32)


def _fuse_scores(
    window_joint_scores: np.ndarray,
    episode_joint_per_window: np.ndarray,
    *,
    window_score_weight: float,
    episode_prior_weights: np.ndarray,
    interaction_weight: float,
    fusion_normalizer: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if fusion_normalizer == "minmax":
        window_term = base._minmax_normalize(window_joint_scores)  # noqa: SLF001
        episode_term = base._minmax_normalize(episode_joint_per_window)  # noqa: SLF001
    elif fusion_normalizer == "percentile":
        window_term = base._to_percentile_score(window_joint_scores)  # noqa: SLF001
        episode_term = base._to_percentile_score(episode_joint_per_window)  # noqa: SLF001
    else:
        window_term = np.asarray(window_joint_scores, dtype=np.float32)
        episode_term = np.asarray(episode_joint_per_window, dtype=np.float32)

    interaction_term = np.asarray(window_term, dtype=np.float32) * np.asarray(episode_term, dtype=np.float32)
    episode_prior_weights = np.asarray(episode_prior_weights, dtype=np.float32)
    denom = np.maximum(
        float(window_score_weight) + episode_prior_weights + float(interaction_weight),
        1e-8,
    )
    fused = (
        float(window_score_weight) * np.asarray(window_term, dtype=np.float32)
        + episode_prior_weights * np.asarray(episode_term, dtype=np.float32)
        + float(interaction_weight) * interaction_term
    ) / denom
    return (
        np.asarray(window_term, dtype=np.float32),
        np.asarray(episode_term, dtype=np.float32),
        np.asarray(interaction_term, dtype=np.float32),
        np.asarray(fused, dtype=np.float32),
    )


def _build_episode_value_map(
    episode_indices: np.ndarray,
    episode_values: np.ndarray,
) -> dict[int, float]:
    return {
        int(ep): float(value)
        for ep, value in zip(episode_indices.tolist(), episode_values.tolist(), strict=True)
    }


def _map_episode_values_to_windows(
    window_episode_indices: np.ndarray,
    episode_value_map: dict[int, float],
) -> np.ndarray:
    return np.asarray(
        [episode_value_map[int(ep)] for ep in window_episode_indices.tolist()],
        dtype=np.float32,
    )


def _compute_episode_window_signals(
    window_episode_indices: np.ndarray,
    window_joint_scores: np.ndarray,
    window_quality_scores: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    unique_episode_indices = np.asarray(sorted(np.unique(window_episode_indices).tolist()), dtype=np.int64)
    window_std = np.zeros((len(unique_episode_indices),), dtype=np.float32)
    quality_std = np.zeros((len(unique_episode_indices),), dtype=np.float32)
    window_counts = np.zeros((len(unique_episode_indices),), dtype=np.float32)
    for out_idx, episode_idx in enumerate(unique_episode_indices.tolist()):
        mask = window_episode_indices == episode_idx
        window_values = np.asarray(window_joint_scores[mask], dtype=np.float32)
        quality_values = np.asarray(window_quality_scores[mask], dtype=np.float32)
        window_std[out_idx] = float(np.std(window_values)) if window_values.size > 0 else 0.0
        quality_std[out_idx] = float(np.std(quality_values)) if quality_values.size > 0 else 0.0
        window_counts[out_idx] = float(window_values.size)
    return unique_episode_indices, window_std, quality_std, window_counts


def _compute_adaptive_episode_prior_weights(
    window_episode_indices: np.ndarray,
    *,
    base_weight: float,
    adaptive_prior_mode: str,
    adaptive_prior_strength: float,
    window_joint_scores: np.ndarray,
    window_quality_scores: np.ndarray,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    base_weights = np.full(window_episode_indices.shape, float(base_weight), dtype=np.float32)
    debug = {
        "adaptive_prior_signal_per_window": np.zeros(window_episode_indices.shape, dtype=np.float32),
        "adaptive_prior_weight_per_window": base_weights.copy(),
    }
    if adaptive_prior_mode == "none" or adaptive_prior_strength == 0:
        return base_weights, debug

    episode_indices, window_std, quality_std, window_counts = _compute_episode_window_signals(
        window_episode_indices,
        window_joint_scores,
        window_quality_scores,
    )
    window_std_signal = base._minmax_normalize(window_std)  # noqa: SLF001
    quality_std_signal = base._minmax_normalize(quality_std)  # noqa: SLF001
    inverse_length_signal = 1.0 - base._minmax_normalize(window_counts)  # noqa: SLF001

    if adaptive_prior_mode == "window_std":
        episode_signal = window_std_signal
    elif adaptive_prior_mode == "inverse_episode_length":
        episode_signal = inverse_length_signal
    elif adaptive_prior_mode == "quality_std":
        episode_signal = quality_std_signal
    else:
        episode_signal = (window_std_signal + inverse_length_signal + quality_std_signal) / 3.0

    centered_signal = 2.0 * np.asarray(episode_signal, dtype=np.float32) - 1.0
    episode_weights = np.clip(
        float(base_weight) * (1.0 + float(adaptive_prior_strength) * centered_signal),
        0.0,
        1.0,
    ).astype(np.float32)

    signal_map = _build_episode_value_map(episode_indices, episode_signal)
    weight_map = _build_episode_value_map(episode_indices, episode_weights)
    debug = {
        "adaptive_prior_signal_per_window": _map_episode_values_to_windows(window_episode_indices, signal_map),
        "adaptive_prior_weight_per_window": _map_episode_values_to_windows(window_episode_indices, weight_map),
    }
    return debug["adaptive_prior_weight_per_window"], debug


def _allocate_remaining_budget(
    weights: np.ndarray,
    capacities: np.ndarray,
    remaining: int,
) -> np.ndarray:
    allocations = np.zeros_like(capacities, dtype=np.int64)
    if remaining <= 0 or np.sum(capacities) <= 0:
        return allocations

    positive_weights = np.maximum(np.asarray(weights, dtype=np.float64), 0.0)
    if float(np.sum(positive_weights)) <= 0:
        positive_weights = np.ones_like(positive_weights, dtype=np.float64)
    shares = positive_weights / float(np.sum(positive_weights))
    raw = shares * float(remaining)
    allocations = np.minimum(np.floor(raw).astype(np.int64), capacities.astype(np.int64))
    left = int(remaining - np.sum(allocations))
    if left <= 0:
        return allocations

    fractional = raw - np.floor(raw)
    for idx in np.argsort(-fractional):
        if left <= 0:
            break
        if allocations[idx] >= capacities[idx]:
            continue
        allocations[idx] += 1
        left -= 1

    if left > 0:
        for idx in np.argsort(-positive_weights):
            if left <= 0:
                break
            extra_capacity = int(capacities[idx] - allocations[idx])
            if extra_capacity <= 0:
                continue
            step = min(left, extra_capacity)
            allocations[idx] += step
            left -= step
    return allocations


def _select_keep_positions_with_episode_budget(
    final_window_scores: np.ndarray,
    window_episode_indices: np.ndarray,
    *,
    keep_ratio: float,
    min_windows_per_episode: int,
    episode_budget_rule: str,
    episode_budget_scores: np.ndarray,
) -> tuple[np.ndarray, dict[str, int | str]]:
    total_windows = int(final_window_scores.shape[0])
    total_keep = max(1, int(round(float(keep_ratio) * total_windows)))
    if min_windows_per_episode <= 0 and episode_budget_rule == "global":
        keep_positions = _retrain._select_keep_indices(final_window_scores.astype(float), keep_ratio)  # noqa: SLF001
        return np.asarray(keep_positions, dtype=np.int64), {
            "selection_rule": "global",
            "total_keep_target": total_keep,
            "base_reserved_windows": 0,
        }

    unique_episode_indices = np.asarray(sorted(np.unique(window_episode_indices).tolist()), dtype=np.int64)
    positions_by_episode = [np.flatnonzero(window_episode_indices == ep) for ep in unique_episode_indices.tolist()]
    counts = np.asarray([len(pos) for pos in positions_by_episode], dtype=np.int64)
    base_budget = np.minimum(counts, int(min_windows_per_episode))
    reserved = int(np.sum(base_budget))
    if reserved > total_keep:
        min_keep_ratio = reserved / float(total_windows) if total_windows > 0 else 1.0
        raise ValueError(
            "Episode budget is infeasible: "
            f"min_windows_per_episode={min_windows_per_episode} reserves {reserved} windows, "
            f"but keep_ratio={keep_ratio:.6f} only allows {total_keep} out of {total_windows}. "
            f"With the current dataset, keep_ratio must be at least {min_keep_ratio:.6f} "
            f"to satisfy the per-episode minimum. "
            "Increase keep_ratio, reduce min_windows_per_episode, or set "
            "min_windows_per_episode=0 if you want a purely global budget."
        )

    selected_positions: list[int] = []
    remaining_positions: list[np.ndarray] = []
    for positions, budget in zip(positions_by_episode, base_budget.tolist(), strict=True):
        episode_scores = final_window_scores[positions]
        order = positions[np.argsort(-episode_scores)]
        if budget > 0:
            selected_positions.extend(order[:budget].tolist())
        remaining_positions.append(order[budget:])

    remaining_keep = total_keep - reserved
    if remaining_keep > 0:
        if episode_budget_rule == "global":
            pooled = np.concatenate([pos for pos in remaining_positions if pos.size > 0], axis=0)
            if pooled.size > 0:
                pooled_scores = final_window_scores[pooled]
                pooled_order = pooled[np.argsort(-pooled_scores)]
                selected_positions.extend(pooled_order[:remaining_keep].tolist())
        else:
            capacities = np.asarray([len(pos) for pos in remaining_positions], dtype=np.int64)
            if episode_budget_rule == "episode_prior":
                weights = np.asarray(episode_budget_scores, dtype=np.float32)
            else:
                weights = np.ones_like(capacities, dtype=np.float32)
            extra_budget = _allocate_remaining_budget(weights, capacities, remaining_keep)
            for positions, budget in zip(remaining_positions, extra_budget.tolist(), strict=True):
                if budget > 0:
                    selected_positions.extend(positions[:budget].tolist())

    keep_positions = np.asarray(sorted(set(selected_positions)), dtype=np.int64)
    if keep_positions.shape[0] > total_keep:
        keep_positions = keep_positions[np.argsort(-final_window_scores[keep_positions])[:total_keep]]
    elif keep_positions.shape[0] < total_keep:
        remaining_mask = np.ones((total_windows,), dtype=bool)
        remaining_mask[keep_positions] = False
        leftover_positions = np.flatnonzero(remaining_mask)
        leftover_order = leftover_positions[np.argsort(-final_window_scores[leftover_positions])]
        fill_count = total_keep - keep_positions.shape[0]
        keep_positions = np.concatenate([keep_positions, leftover_order[:fill_count]])
    return np.asarray(np.sort(keep_positions), dtype=np.int64), {
        "selection_rule": episode_budget_rule,
        "total_keep_target": total_keep,
        "base_reserved_windows": reserved,
    }


def _default_output_dir(
    dataset_dir: Path,
    keep_ratio: float,
    score_pkl: Path,
    quality_source: str,
    joint_rule: str,
    window_score_weight: float,
    episode_prior_weight: float,
    adaptive_prior_mode: str,
    interaction_weight: float,
) -> Path:
    keep_pct = int(round(keep_ratio * 100))
    w_tag = int(round(window_score_weight * 100))
    e_tag = int(round(episode_prior_weight * 100))
    interaction_tag = int(round(interaction_weight * 100))
    return dataset_dir.parent / (
        f"{dataset_dir.name}_window_episode_prior_{joint_rule}_keep{keep_pct}_"
        f"w{w_tag}_e{e_tag}_i{interaction_tag}_{adaptive_prior_mode}_{quality_source}_{score_pkl.stem}"
    )


def _write_selection_artifacts(
    output_dir: Path,
    *,
    window_indices: np.ndarray,
    window_influence_scores: np.ndarray,
    window_quality_scores: np.ndarray,
    window_normalized_influence: np.ndarray,
    window_joint_scores: np.ndarray,
    episode_indices: np.ndarray,
    episode_influence_scores: np.ndarray,
    episode_quality_scores: np.ndarray,
    episode_normalized_influence: np.ndarray,
    episode_joint_scores: np.ndarray,
    episode_joint_per_window: np.ndarray,
    normalized_window_joint_scores: np.ndarray,
    normalized_episode_joint_scores: np.ndarray,
    interaction_scores: np.ndarray,
    adaptive_prior_weight_per_window: np.ndarray,
    adaptive_prior_signal_per_window: np.ndarray,
    final_window_scores: np.ndarray,
    kept_window_indices: np.ndarray,
    kept_sample_indices: np.ndarray,
    summary: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "joint_window_episode_prior_scores.npz",
        window_indices=np.asarray(window_indices, dtype=np.int64),
        influence_scores=np.asarray(window_influence_scores, dtype=np.float32),
        quality_scores=np.asarray(window_quality_scores, dtype=np.float32),
        normalized_influence=np.asarray(window_normalized_influence, dtype=np.float32),
        window_joint_scores=np.asarray(window_joint_scores, dtype=np.float32),
        episode_indices=np.asarray(episode_indices, dtype=np.int64),
        episode_influence_scores=np.asarray(episode_influence_scores, dtype=np.float32),
        episode_quality_scores=np.asarray(episode_quality_scores, dtype=np.float32),
        episode_normalized_influence=np.asarray(episode_normalized_influence, dtype=np.float32),
        episode_joint_scores=np.asarray(episode_joint_scores, dtype=np.float32),
        episode_joint_scores_per_window=np.asarray(episode_joint_per_window, dtype=np.float32),
        normalized_window_joint_scores=np.asarray(normalized_window_joint_scores, dtype=np.float32),
        normalized_episode_joint_scores=np.asarray(normalized_episode_joint_scores, dtype=np.float32),
        interaction_scores=np.asarray(interaction_scores, dtype=np.float32),
        adaptive_prior_weight_per_window=np.asarray(adaptive_prior_weight_per_window, dtype=np.float32),
        adaptive_prior_signal_per_window=np.asarray(adaptive_prior_signal_per_window, dtype=np.float32),
        final_window_scores=np.asarray(final_window_scores, dtype=np.float32),
        kept_window_indices=np.asarray(kept_window_indices, dtype=np.int64),
        kept_sample_indices=np.asarray(kept_sample_indices, dtype=np.int64),
    )
    (output_dir / "joint_window_episode_prior_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter OpenPI training data by window-level joint score with an episode-level prior.",
    )
    parser.add_argument("--score-pkl", type=Path, required=True, help="Window GraSS pickle.")
    parser.add_argument("--train-config", type=str, default=None, help="Training config name. If omitted, infer from score pickle.")
    parser.add_argument("--dataset-dir", type=Path, default=None, help="Dataset root. If omitted, infer from train config.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory for subset, scores, and norm_stats.")
    parser.add_argument("--output-indices", type=Path, default=None, help="Output sample-index file.")
    parser.add_argument("--exp-name", type=str, default=None, help="Experiment name for retraining.")
    parser.add_argument("--checkpoint-base-dir", type=Path, default=Path("./checkpoints"))
    parser.add_argument("--num-train-steps", type=int, default=None)
    parser.add_argument("--keep-ratio", type=float, default=None)
    parser.add_argument("--drop-ratio", type=float, default=None)
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
        help="Window-level quality source adapted from eval_cupid_quality.py.",
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
    parser.add_argument("--max-frames-for-norm-stats", type=int, default=None)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    keep_ratio = retrain_grass._resolve_keep_ratio(args.keep_ratio, args.drop_ratio)  # noqa: SLF001
    if args.window_score_weight < 0 or args.episode_prior_weight < 0:
        raise ValueError("Fusion weights must be >= 0.")
    if args.interaction_weight < 0:
        raise ValueError("--interaction-weight must be >= 0.")
    if not 0 <= args.adaptive_prior_strength <= 1:
        raise ValueError("--adaptive-prior-strength must be in [0, 1].")
    if args.min_windows_per_episode < 0:
        raise ValueError("--min-windows-per-episode must be >= 0.")
    if args.window_score_weight == 0 and args.episode_prior_weight == 0:
        raise ValueError("At least one of --window-score-weight or --episode-prior-weight must be > 0.")

    payload = base._load_window_payload(args.score_pkl)  # noqa: SLF001
    train_config_name = args.train_config or _retrain._extract_score_config_name(payload)  # noqa: SLF001
    if train_config_name is None:
        raise ValueError("--train-config is required because the score pickle does not include config.config_name.")

    if args.dataset_dir is None:
        dataset_dir, default_exp_name = _retrain._infer_defaults_from_train_config(train_config_name)  # noqa: SLF001
    else:
        dataset_dir = args.dataset_dir
        default_exp_name = train_config_name

    metadata = payload["metadata"]
    window_indices = np.asarray(metadata["train_window_indices"], dtype=np.int64).reshape(-1)
    window_episode_indices = np.asarray(metadata["train_window_episode_indices"], dtype=np.int64).reshape(-1)
    sample_window_indices = np.asarray(metadata["train_sample_window_indices"], dtype=np.int64).reshape(-1)
    dataset_sample_indices = np.asarray(metadata["train_dataset_sample_indices"], dtype=np.int64).reshape(-1)
    window_influence_scores = np.asarray(payload["window_influence_scores"], dtype=np.float32).reshape(-1)
    if not (
        window_influence_scores.shape[0] == window_indices.shape[0] == window_episode_indices.shape[0]
        and sample_window_indices.shape[0] == dataset_sample_indices.shape[0]
    ):
        raise ValueError("Window metadata shape mismatch in score pickle.")

    window_quality_scores, quality_components = base._compute_window_quality_scores(  # noqa: SLF001
        payload,
        dataset_dir,
        quality_source=args.quality_source,
        normalizer=args.quality_normalizer,
        quantile_low=args.quality_quantile_low,
        quantile_high=args.quality_quantile_high,
        episode_action_weight=args.episode_action_weight,
        episode_state_weight=args.episode_state_weight,
        episode_length_weight=args.episode_length_weight,
        episode_energy_weight=args.episode_energy_weight,
        trajectory_terminal_weight=args.trajectory_terminal_weight,
        trajectory_efficiency_weight=args.trajectory_efficiency_weight,
        trajectory_progress_weight=args.trajectory_progress_weight,
        trajectory_state_smooth_weight=args.trajectory_state_smooth_weight,
        trajectory_action_smooth_weight=args.trajectory_action_smooth_weight,
        trajectory_length_weight=args.trajectory_length_weight,
        rank_fusion_episode_weight=args.rank_fusion_episode_weight,
        rank_fusion_trajectory_weight=args.rank_fusion_trajectory_weight,
    )

    window_normalized_influence = base._minmax_normalize(window_influence_scores)  # noqa: SLF001
    window_joint_scores = base._combine_joint_scores(  # noqa: SLF001
        window_quality_scores,
        window_normalized_influence,
        joint_rule=args.joint_rule,
        joint_quality_power=args.joint_quality_power,
        joint_influence_power=args.joint_influence_power,
    )

    episode_indices, episode_quality_scores = _aggregate_episode_quality(
        window_episode_indices,
        window_quality_scores,
        agg=args.episode_quality_aggregation,
    )
    episode_influence_scores = _align_episode_influence(payload, episode_indices)
    episode_normalized_influence = base._minmax_normalize(episode_influence_scores)  # noqa: SLF001
    episode_joint_scores = base._combine_joint_scores(  # noqa: SLF001
        episode_quality_scores,
        episode_normalized_influence,
        joint_rule=args.joint_rule,
        joint_quality_power=args.joint_quality_power,
        joint_influence_power=args.joint_influence_power,
    )

    episode_joint_map = _build_episode_value_map(episode_indices, episode_joint_scores)
    episode_joint_per_window = _map_episode_values_to_windows(window_episode_indices, episode_joint_map)
    adaptive_prior_weight_per_window, adaptive_prior_debug = _compute_adaptive_episode_prior_weights(
        window_episode_indices,
        base_weight=args.episode_prior_weight,
        adaptive_prior_mode=args.adaptive_prior_mode,
        adaptive_prior_strength=args.adaptive_prior_strength,
        window_joint_scores=window_joint_scores,
        window_quality_scores=window_quality_scores,
    )
    (
        normalized_window_joint_scores,
        normalized_episode_joint_scores,
        interaction_scores,
        final_window_scores,
    ) = _fuse_scores(
        window_joint_scores,
        episode_joint_per_window,
        window_score_weight=args.window_score_weight,
        episode_prior_weights=adaptive_prior_weight_per_window,
        interaction_weight=args.interaction_weight,
        fusion_normalizer=args.fusion_normalizer,
    )
    normalized_episode_joint_scores_by_episode = np.asarray(
        [float(np.max(normalized_episode_joint_scores[window_episode_indices == ep])) for ep in episode_indices.tolist()],
        dtype=np.float32,
    )
    keep_positions, selection_debug = _select_keep_positions_with_episode_budget(
        final_window_scores,
        window_episode_indices,
        keep_ratio=keep_ratio,
        min_windows_per_episode=args.min_windows_per_episode,
        episode_budget_rule=args.episode_budget_rule,
        episode_budget_scores=normalized_episode_joint_scores_by_episode,
    )
    kept_window_indices = window_indices[keep_positions]
    keep_sample_mask = np.isin(sample_window_indices, kept_window_indices)
    kept_sample_indices = np.unique(dataset_sample_indices[keep_sample_mask]).astype(np.int64)
    if kept_sample_indices.size == 0:
        raise ValueError("Window selection resulted in zero kept samples.")

    output_dir = args.output_dir or _default_output_dir(
        dataset_dir,
        keep_ratio,
        args.score_pkl,
        args.quality_source,
        args.joint_rule,
        args.window_score_weight,
        args.episode_prior_weight,
        args.adaptive_prior_mode,
        args.interaction_weight,
    )
    output_indices = args.output_indices or (output_dir / "kept_sample_indices.npy")
    output_dir.mkdir(parents=True, exist_ok=True)
    _retrain._save_sample_indices(output_indices, kept_sample_indices, overwrite=args.overwrite)  # noqa: SLF001

    summary = {
        "score_pkl": str(args.score_pkl),
        "dataset_dir": str(dataset_dir),
        "quality_source": args.quality_source,
        "joint_rule": args.joint_rule,
        "fusion_normalizer": args.fusion_normalizer,
        "episode_quality_aggregation": args.episode_quality_aggregation,
        "window_score_weight": float(args.window_score_weight),
        "episode_prior_weight": float(args.episode_prior_weight),
        "adaptive_prior_mode": args.adaptive_prior_mode,
        "adaptive_prior_strength": float(args.adaptive_prior_strength),
        "adaptive_prior_weight_min": float(np.min(adaptive_prior_weight_per_window)),
        "adaptive_prior_weight_max": float(np.max(adaptive_prior_weight_per_window)),
        "interaction_weight": float(args.interaction_weight),
        "min_windows_per_episode": int(args.min_windows_per_episode),
        "episode_budget_rule": args.episode_budget_rule,
        "selection_rule": str(selection_debug["selection_rule"]),
        "selection_total_keep_target": int(selection_debug["total_keep_target"]),
        "selection_base_reserved_windows": int(selection_debug["base_reserved_windows"]),
        "num_windows": int(window_indices.shape[0]),
        "num_episodes": int(episode_indices.shape[0]),
        "kept_windows": int(kept_window_indices.shape[0]),
        "kept_samples": int(kept_sample_indices.shape[0]),
        "keep_ratio": float(keep_ratio),
        "window_joint_score_min": float(np.min(window_joint_scores)),
        "window_joint_score_max": float(np.max(window_joint_scores)),
        "episode_joint_score_min": float(np.min(episode_joint_scores)),
        "episode_joint_score_max": float(np.max(episode_joint_scores)),
        "final_window_score_min": float(np.min(final_window_scores)),
        "final_window_score_max": float(np.max(final_window_scores)),
    }

    _write_selection_artifacts(
        output_dir,
        window_indices=window_indices,
        window_influence_scores=window_influence_scores,
        window_quality_scores=window_quality_scores,
        window_normalized_influence=window_normalized_influence,
        window_joint_scores=window_joint_scores,
        episode_indices=episode_indices,
        episode_influence_scores=episode_influence_scores,
        episode_quality_scores=episode_quality_scores,
        episode_normalized_influence=episode_normalized_influence,
        episode_joint_scores=episode_joint_scores,
        episode_joint_per_window=episode_joint_per_window,
        normalized_window_joint_scores=normalized_window_joint_scores,
        normalized_episode_joint_scores=normalized_episode_joint_scores,
        interaction_scores=interaction_scores,
        adaptive_prior_weight_per_window=adaptive_prior_weight_per_window,
        adaptive_prior_signal_per_window=adaptive_prior_debug["adaptive_prior_signal_per_window"],
        final_window_scores=final_window_scores,
        kept_window_indices=kept_window_indices,
        kept_sample_indices=kept_sample_indices,
        summary=summary,
    )
    np.savez_compressed(output_dir / "window_quality_components.npz", **quality_components)

    train_cfg = _config.get_config(train_config_name)
    print(f"Score file: {args.score_pkl}")
    print(f"Train config: {train_config_name}")
    print(f"Dataset: {dataset_dir}")
    print(
        "Window+Episode prior selection summary: "
        f"windows={window_indices.shape[0]}, kept_windows={kept_window_indices.shape[0]}, "
        f"kept_samples={kept_sample_indices.shape[0]}, "
        f"selection_rule={summary['selection_rule']}, "
        f"final_score_range=[{summary['final_window_score_min']:.6f}, {summary['final_window_score_max']:.6f}]"
    )
    print(f"Output dir: {output_dir}")
    print(f"Sample indices: {output_indices}")

    print("Recomputing norm_stats.json for selected sample subset...")
    base._recompute_norm_stats_for_sample_subset(  # noqa: SLF001
        train_cfg,
        sample_indices_path=output_indices,
        output_dir=output_dir,
        max_frames=args.max_frames_for_norm_stats,
    )
    print(f"Subset norm_stats written to: {output_dir / 'norm_stats.json'}")

    if args.prepare_only:
        return

    keep_pct = int(round(keep_ratio * 100))
    w_tag = int(round(args.window_score_weight * 100))
    e_tag = int(round(args.episode_prior_weight * 100))
    interaction_tag = int(round(args.interaction_weight * 100))
    exp_name = args.exp_name or (
        f"{default_exp_name}_window_episode_prior_{args.joint_rule}_"
        f"keep{keep_pct}_w{w_tag}_e{e_tag}_i{interaction_tag}_{args.adaptive_prior_mode}_{args.quality_source}"
    )
    retrain_cfg = base._build_train_config(  # noqa: SLF001
        train_cfg,
        sample_indices_path=output_indices,
        output_dir=output_dir,
        exp_name=exp_name,
        checkpoint_base_dir=args.checkpoint_base_dir,
        overwrite=args.overwrite,
        num_train_steps=args.num_train_steps,
    )
    print(
        f"Starting retraining: exp_name={retrain_cfg.exp_name}, "
        f"checkpoint_dir={retrain_cfg.checkpoint_dir}, sample_indices={output_indices}"
    )
    from scripts import train as train_script

    train_script.main(retrain_cfg)


if __name__ == "__main__":
    main()
