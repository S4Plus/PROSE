#!/usr/bin/env python3
"""Common helpers for gain+coverage-based window episode-prior filtering."""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

import openpi.training.config as _config
import retrain as _retrain
import retrain_window_joint as base
import retrain_window_joint_episode_prior as episode_prior

DEFAULT_GAIN_DELTA_RATIO = 0.02


def validate_common_args(args: Any) -> None:
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


def build_context_from_args(args: Any) -> dict[str, Any]:
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

    episode_indices, episode_quality_scores = episode_prior._aggregate_episode_quality(  # noqa: SLF001
        window_episode_indices,
        window_quality_scores,
        agg=args.episode_quality_aggregation,
    )
    episode_influence_scores = episode_prior._align_episode_influence(payload, episode_indices)  # noqa: SLF001
    episode_normalized_influence = base._minmax_normalize(episode_influence_scores)  # noqa: SLF001
    episode_joint_scores = base._combine_joint_scores(  # noqa: SLF001
        episode_quality_scores,
        episode_normalized_influence,
        joint_rule=args.joint_rule,
        joint_quality_power=args.joint_quality_power,
        joint_influence_power=args.joint_influence_power,
    )

    episode_joint_map = episode_prior._build_episode_value_map(episode_indices, episode_joint_scores)  # noqa: SLF001
    episode_joint_per_window = episode_prior._map_episode_values_to_windows(  # noqa: SLF001
        window_episode_indices,
        episode_joint_map,
    )
    adaptive_prior_weight_per_window, adaptive_prior_debug = episode_prior._compute_adaptive_episode_prior_weights(  # noqa: SLF001
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
    ) = episode_prior._fuse_scores(  # noqa: SLF001
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

    return {
        "args": args,
        "payload": payload,
        "train_config_name": train_config_name,
        "dataset_dir": dataset_dir,
        "default_exp_name": default_exp_name,
        "window_indices": window_indices,
        "window_episode_indices": window_episode_indices,
        "sample_window_indices": sample_window_indices,
        "dataset_sample_indices": dataset_sample_indices,
        "window_influence_scores": window_influence_scores,
        "window_quality_scores": window_quality_scores,
        "quality_components": quality_components,
        "window_normalized_influence": window_normalized_influence,
        "window_joint_scores": window_joint_scores,
        "episode_indices": episode_indices,
        "episode_influence_scores": episode_influence_scores,
        "episode_quality_scores": episode_quality_scores,
        "episode_normalized_influence": episode_normalized_influence,
        "episode_joint_scores": episode_joint_scores,
        "episode_joint_per_window": episode_joint_per_window,
        "adaptive_prior_weight_per_window": adaptive_prior_weight_per_window,
        "adaptive_prior_signal_per_window": adaptive_prior_debug["adaptive_prior_signal_per_window"],
        "normalized_window_joint_scores": normalized_window_joint_scores,
        "normalized_episode_joint_scores": normalized_episode_joint_scores,
        "interaction_scores": interaction_scores,
        "final_window_scores": final_window_scores,
        "normalized_episode_joint_scores_by_episode": normalized_episode_joint_scores_by_episode,
    }


def _resolve_gain_delta(num_windows: int) -> int:
    return max(1, min(int(round(num_windows * DEFAULT_GAIN_DELTA_RATIO)), max(num_windows - 1, 1)))


def compute_gain_curve(mean_score_curve: np.ndarray, *, gain_delta: int, score_range: float) -> tuple[np.ndarray, np.ndarray]:
    mean_scores = np.asarray(mean_score_curve, dtype=np.float64).reshape(-1)
    if mean_scores.size == 0:
        raise ValueError("Cannot compute gain curve from an empty score array.")

    gain_abs = np.zeros(mean_scores.size, dtype=np.float64)
    for idx in range(mean_scores.size):
        right_idx = min(idx + gain_delta, mean_scores.size - 1)
        gain_abs[idx] = max(0.0, float(mean_scores[idx] - mean_scores[right_idx]))

    gain_rel = gain_abs / max(float(score_range), 1e-8)
    return gain_abs, gain_rel


def _evaluate_episode_stats(
    kept_window_episode_indices: np.ndarray,
    *,
    episode_indices: np.ndarray,
    min_windows_per_episode: int,
) -> dict[str, float | int]:
    episode_to_pos = {int(ep): idx for idx, ep in enumerate(episode_indices.tolist())}
    counts = np.zeros((len(episode_indices),), dtype=np.int64)
    for ep in kept_window_episode_indices.tolist():
        counts[episode_to_pos[int(ep)]] += 1

    total_episodes = int(len(episode_indices))
    covered_episodes = int(np.count_nonzero(counts > 0))
    if min_windows_per_episode <= 0:
        episodes_meeting_min_windows = total_episodes
    else:
        episodes_meeting_min_windows = int(np.count_nonzero(counts >= int(min_windows_per_episode)))

    return {
        "covered_episodes": covered_episodes,
        "covered_episode_ratio": covered_episodes / float(max(total_episodes, 1)),
        "episodes_meeting_min_windows": episodes_meeting_min_windows,
        "episodes_meeting_min_windows_ratio": episodes_meeting_min_windows / float(max(total_episodes, 1)),
    }


def search_keep_ratio_by_gain_and_coverage(
    context: dict[str, Any],
    *,
    min_episode_coverage_ratio: float,
    max_relative_gain: float,
) -> dict[str, Any]:
    args = context["args"]
    final_window_scores = np.asarray(context["final_window_scores"], dtype=np.float64).reshape(-1)
    window_episode_indices = np.asarray(context["window_episode_indices"], dtype=np.int64).reshape(-1)
    episode_indices = np.asarray(context["episode_indices"], dtype=np.int64).reshape(-1)
    episode_budget_scores = np.asarray(context["normalized_episode_joint_scores_by_episode"], dtype=np.float32).reshape(-1)

    total_windows = int(final_window_scores.size)
    if total_windows == 0:
        raise ValueError("No windows available for gain+coverage search.")
    if not 0 < min_episode_coverage_ratio <= 1:
        raise ValueError("--min-episode-coverage-ratio must be in (0, 1].")
    if max_relative_gain < 0:
        raise ValueError("--max-relative-gain must be >= 0.")

    min_episode_ratio_with_min_windows = float(min_episode_coverage_ratio)

    if args.min_windows_per_episode > 0:
        episode_to_pos = {int(ep): idx for idx, ep in enumerate(episode_indices.tolist())}
        total_counts = np.zeros((len(episode_indices),), dtype=np.int64)
        for ep in window_episode_indices.tolist():
            total_counts[episode_to_pos[int(ep)]] += 1
        min_feasible_keep_count = int(np.sum(np.minimum(total_counts, int(args.min_windows_per_episode))))
        max_supported_ratio = float(np.count_nonzero(total_counts >= int(args.min_windows_per_episode))) / float(
            max(len(episode_indices), 1)
        )
        if min_episode_ratio_with_min_windows > max_supported_ratio + 1e-12:
            raise ValueError(
                "The min-window constraint is infeasible: "
                f"min_windows_per_episode={args.min_windows_per_episode} can only be satisfied by "
                f"{max_supported_ratio:.4f} of episodes at most, but "
                f"min_episode_ratio_with_min_windows={min_episode_ratio_with_min_windows:.4f}."
            )
    else:
        min_feasible_keep_count = 1

    min_feasible_keep_count = max(1, min(min_feasible_keep_count, total_windows))

    resolved_gain_delta = _resolve_gain_delta(total_windows)

    keep_counts: list[int] = []
    keep_ratios: list[float] = []
    covered_episode_counts: list[int] = []
    covered_episode_ratios: list[float] = []
    min_window_episode_counts: list[int] = []
    min_window_episode_ratios: list[float] = []
    selection_base_reserved_windows: list[int] = []
    kept_mean_score_curve: list[float] = []

    for keep_count in range(min_feasible_keep_count, total_windows + 1):
        keep_ratio = keep_count / float(total_windows)
        keep_positions, selection_debug = episode_prior._select_keep_positions_with_episode_budget(  # noqa: SLF001
            final_window_scores.astype(np.float32),
            window_episode_indices,
            keep_ratio=keep_ratio,
            min_windows_per_episode=args.min_windows_per_episode,
            episode_budget_rule=args.episode_budget_rule,
            episode_budget_scores=episode_budget_scores,
        )
        stats = _evaluate_episode_stats(
            window_episode_indices[keep_positions],
            episode_indices=episode_indices,
            min_windows_per_episode=args.min_windows_per_episode,
        )

        keep_counts.append(int(keep_count))
        keep_ratios.append(float(keep_ratio))
        covered_episode_counts.append(int(stats["covered_episodes"]))
        covered_episode_ratios.append(float(stats["covered_episode_ratio"]))
        min_window_episode_counts.append(int(stats["episodes_meeting_min_windows"]))
        min_window_episode_ratios.append(float(stats["episodes_meeting_min_windows_ratio"]))
        selection_base_reserved_windows.append(int(selection_debug["base_reserved_windows"]))
        kept_mean_score_curve.append(float(np.mean(final_window_scores[keep_positions])))

    gain_abs_curve, gain_rel_curve = compute_gain_curve(
        np.asarray(kept_mean_score_curve, dtype=np.float64),
        gain_delta=resolved_gain_delta,
        score_range=float(np.max(final_window_scores) - np.min(final_window_scores)),
    )

    recommended_keep_count: int | None = None
    recommendation_reason = "full_keep_fallback"

    for idx, keep_count in enumerate(keep_counts):
        coverage_ok = covered_episode_ratios[idx] >= float(min_episode_coverage_ratio) - 1e-12
        min_windows_ok = min_window_episode_ratios[idx] >= float(min_episode_ratio_with_min_windows) - 1e-12
        gain_ok = float(gain_rel_curve[idx]) <= float(max_relative_gain) + 1e-12
        if coverage_ok and min_windows_ok and gain_ok:
            recommended_keep_count = keep_count
            recommendation_reason = "first_candidate_meeting_gain_and_coverage"
            break

    if recommended_keep_count is None:
        for idx, keep_count in enumerate(keep_counts):
            coverage_ok = covered_episode_ratios[idx] >= float(min_episode_coverage_ratio) - 1e-12
            min_windows_ok = min_window_episode_ratios[idx] >= float(min_episode_ratio_with_min_windows) - 1e-12
            if coverage_ok and min_windows_ok:
                recommended_keep_count = keep_count
                recommendation_reason = "first_candidate_meeting_coverage_only"
                break

    if recommended_keep_count is None:
        recommended_keep_count = total_windows

    recommended_idx = recommended_keep_count - 1
    recommended_keep_ratio = recommended_keep_count / float(total_windows)
    keep_positions, selection_debug = episode_prior._select_keep_positions_with_episode_budget(  # noqa: SLF001
        final_window_scores.astype(np.float32),
        window_episode_indices,
        keep_ratio=recommended_keep_ratio,
        min_windows_per_episode=args.min_windows_per_episode,
        episode_budget_rule=args.episode_budget_rule,
        episode_budget_scores=episode_budget_scores,
    )
    recommended_stats = _evaluate_episode_stats(
        window_episode_indices[keep_positions],
        episode_indices=episode_indices,
        min_windows_per_episode=args.min_windows_per_episode,
    )

    return {
        "recommended_keep_count": int(recommended_keep_count),
        "recommended_keep_ratio": float(recommended_keep_ratio),
        "recommended_keep_positions": np.asarray(keep_positions, dtype=np.int64),
        "recommendation_reason": recommendation_reason,
        "min_feasible_keep_count": int(min_feasible_keep_count),
        "min_feasible_keep_ratio": float(min_feasible_keep_count / float(total_windows)),
        "gain_delta": int(resolved_gain_delta),
        "max_relative_gain": float(max_relative_gain),
        "min_episode_coverage_ratio": float(min_episode_coverage_ratio),
        "min_episode_ratio_with_min_windows": float(min_episode_ratio_with_min_windows),
        "recommended_gain_abs": float(gain_abs_curve[recommended_idx]),
        "recommended_gain_rel": float(gain_rel_curve[recommended_idx]),
        "recommended_covered_episodes": int(recommended_stats["covered_episodes"]),
        "recommended_covered_episode_ratio": float(recommended_stats["covered_episode_ratio"]),
        "recommended_episodes_meeting_min_windows": int(recommended_stats["episodes_meeting_min_windows"]),
        "recommended_episodes_meeting_min_windows_ratio": float(
            recommended_stats["episodes_meeting_min_windows_ratio"]
        ),
        "recommended_selection_rule": str(selection_debug["selection_rule"]),
        "recommended_selection_base_reserved_windows": int(selection_debug["base_reserved_windows"]),
        "gain_abs_curve": np.asarray(gain_abs_curve, dtype=np.float32),
        "gain_rel_curve": np.asarray(gain_rel_curve, dtype=np.float32),
        "keep_counts": np.asarray(keep_counts, dtype=np.int64),
        "keep_ratios": np.asarray(keep_ratios, dtype=np.float32),
        "covered_episode_counts": np.asarray(covered_episode_counts, dtype=np.int64),
        "covered_episode_ratios": np.asarray(covered_episode_ratios, dtype=np.float32),
        "min_window_episode_counts": np.asarray(min_window_episode_counts, dtype=np.int64),
        "min_window_episode_ratios": np.asarray(min_window_episode_ratios, dtype=np.float32),
        "selection_base_reserved_windows": np.asarray(selection_base_reserved_windows, dtype=np.int64),
        "kept_mean_score_curve": np.asarray(kept_mean_score_curve, dtype=np.float32),
    }


def materialize_selection(context: dict[str, Any], *, keep_ratio: float) -> dict[str, Any]:
    keep_positions, selection_debug = episode_prior._select_keep_positions_with_episode_budget(  # noqa: SLF001
        np.asarray(context["final_window_scores"], dtype=np.float32),
        np.asarray(context["window_episode_indices"], dtype=np.int64),
        keep_ratio=keep_ratio,
        min_windows_per_episode=context["args"].min_windows_per_episode,
        episode_budget_rule=context["args"].episode_budget_rule,
        episode_budget_scores=np.asarray(context["normalized_episode_joint_scores_by_episode"], dtype=np.float32),
    )
    kept_window_indices = np.asarray(context["window_indices"], dtype=np.int64)[keep_positions]
    keep_sample_mask = np.isin(np.asarray(context["sample_window_indices"], dtype=np.int64), kept_window_indices)
    kept_sample_indices = np.unique(np.asarray(context["dataset_sample_indices"], dtype=np.int64)[keep_sample_mask]).astype(
        np.int64
    )
    if kept_sample_indices.size == 0:
        raise ValueError("Window selection resulted in zero kept samples.")
    return {
        "keep_positions": np.asarray(keep_positions, dtype=np.int64),
        "kept_window_indices": kept_window_indices,
        "kept_sample_indices": kept_sample_indices,
        "selection_debug": selection_debug,
    }


def default_output_dir(
    *,
    dataset_dir: Path,
    score_pkl: Path,
    keep_ratio: float,
    joint_rule: str,
    quality_source: str,
    window_score_weight: float,
    episode_prior_weight: float,
    adaptive_prior_mode: str,
    interaction_weight: float,
) -> Path:
    keep_pct = int(round(float(keep_ratio) * 100))
    w_tag = int(round(float(window_score_weight) * 100))
    e_tag = int(round(float(episode_prior_weight) * 100))
    interaction_tag = int(round(float(interaction_weight) * 100))
    return dataset_dir.parent / (
        f"{dataset_dir.name}_window_episode_prior_{joint_rule}_gaincov_keep{keep_pct}_"
        f"w{w_tag}_e{e_tag}_i{interaction_tag}_{adaptive_prior_mode}_{quality_source}_{score_pkl.stem}"
    )


def recommended_exp_name(default_exp_name: str, args: Any, keep_ratio: float) -> str:
    keep_pct = int(round(float(keep_ratio) * 100))
    w_tag = int(round(float(args.window_score_weight) * 100))
    e_tag = int(round(float(args.episode_prior_weight) * 100))
    interaction_tag = int(round(float(args.interaction_weight) * 100))
    return (
        f"{default_exp_name}_window_episode_prior_{args.joint_rule}_gaincov_"
        f"keep{keep_pct}_w{w_tag}_e{e_tag}_i{interaction_tag}_{args.adaptive_prior_mode}_{args.quality_source}"
    )


def build_selection_summary(context: dict[str, Any], selected: dict[str, Any], heuristic: dict[str, Any]) -> dict[str, Any]:
    args = context["args"]
    return {
        "score_pkl": str(args.score_pkl),
        "dataset_dir": str(context["dataset_dir"]),
        "quality_source": args.quality_source,
        "joint_rule": args.joint_rule,
        "fusion_normalizer": args.fusion_normalizer,
        "episode_quality_aggregation": args.episode_quality_aggregation,
        "window_score_weight": float(args.window_score_weight),
        "episode_prior_weight": float(args.episode_prior_weight),
        "adaptive_prior_mode": args.adaptive_prior_mode,
        "adaptive_prior_strength": float(args.adaptive_prior_strength),
        "adaptive_prior_weight_min": float(np.min(context["adaptive_prior_weight_per_window"])),
        "adaptive_prior_weight_max": float(np.max(context["adaptive_prior_weight_per_window"])),
        "interaction_weight": float(args.interaction_weight),
        "min_windows_per_episode": int(args.min_windows_per_episode),
        "episode_budget_rule": args.episode_budget_rule,
        "selection_rule": str(selected["selection_debug"]["selection_rule"]),
        "selection_total_keep_target": int(selected["selection_debug"]["total_keep_target"]),
        "selection_base_reserved_windows": int(selected["selection_debug"]["base_reserved_windows"]),
        "num_windows": int(len(context["window_indices"])),
        "num_episodes": int(len(context["episode_indices"])),
        "kept_windows": int(len(selected["kept_window_indices"])),
        "kept_samples": int(len(selected["kept_sample_indices"])),
        "keep_ratio": float(heuristic["recommended_keep_ratio"]),
        "window_joint_score_min": float(np.min(context["window_joint_scores"])),
        "window_joint_score_max": float(np.max(context["window_joint_scores"])),
        "episode_joint_score_min": float(np.min(context["episode_joint_scores"])),
        "episode_joint_score_max": float(np.max(context["episode_joint_scores"])),
        "final_window_score_min": float(np.min(context["final_window_scores"])),
        "final_window_score_max": float(np.max(context["final_window_scores"])),
        "heuristic_method": "gain_plus_coverage",
        "heuristic_min_feasible_keep_count": int(heuristic["min_feasible_keep_count"]),
        "heuristic_min_feasible_keep_ratio": float(heuristic["min_feasible_keep_ratio"]),
        "heuristic_gain_delta": int(heuristic["gain_delta"]),
        "heuristic_max_relative_gain": float(heuristic["max_relative_gain"]),
        "heuristic_min_episode_coverage_ratio": float(heuristic["min_episode_coverage_ratio"]),
        "heuristic_min_episode_ratio_with_min_windows": float(heuristic["min_episode_ratio_with_min_windows"]),
        "heuristic_recommendation_reason": str(heuristic["recommendation_reason"]),
        "heuristic_recommended_gain_abs": float(heuristic["recommended_gain_abs"]),
        "heuristic_recommended_gain_rel": float(heuristic["recommended_gain_rel"]),
        "heuristic_recommended_covered_episodes": int(heuristic["recommended_covered_episodes"]),
        "heuristic_recommended_covered_episode_ratio": float(heuristic["recommended_covered_episode_ratio"]),
        "heuristic_recommended_episodes_meeting_min_windows": int(heuristic["recommended_episodes_meeting_min_windows"]),
        "heuristic_recommended_episodes_meeting_min_windows_ratio": float(
            heuristic["recommended_episodes_meeting_min_windows_ratio"]
        ),
    }


def write_selection_artifacts(
    output_dir: Path,
    *,
    context: dict[str, Any],
    selected: dict[str, Any],
    summary: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "joint_window_episode_prior_scores.npz",
        window_indices=np.asarray(context["window_indices"], dtype=np.int64),
        window_episode_indices=np.asarray(context["window_episode_indices"], dtype=np.int64),
        influence_scores=np.asarray(context["window_influence_scores"], dtype=np.float32),
        quality_scores=np.asarray(context["window_quality_scores"], dtype=np.float32),
        normalized_influence=np.asarray(context["window_normalized_influence"], dtype=np.float32),
        window_joint_scores=np.asarray(context["window_joint_scores"], dtype=np.float32),
        episode_indices=np.asarray(context["episode_indices"], dtype=np.int64),
        episode_influence_scores=np.asarray(context["episode_influence_scores"], dtype=np.float32),
        episode_quality_scores=np.asarray(context["episode_quality_scores"], dtype=np.float32),
        episode_normalized_influence=np.asarray(context["episode_normalized_influence"], dtype=np.float32),
        episode_joint_scores=np.asarray(context["episode_joint_scores"], dtype=np.float32),
        episode_joint_scores_per_window=np.asarray(context["episode_joint_per_window"], dtype=np.float32),
        normalized_window_joint_scores=np.asarray(context["normalized_window_joint_scores"], dtype=np.float32),
        normalized_episode_joint_scores=np.asarray(context["normalized_episode_joint_scores"], dtype=np.float32),
        interaction_scores=np.asarray(context["interaction_scores"], dtype=np.float32),
        adaptive_prior_weight_per_window=np.asarray(context["adaptive_prior_weight_per_window"], dtype=np.float32),
        adaptive_prior_signal_per_window=np.asarray(context["adaptive_prior_signal_per_window"], dtype=np.float32),
        final_window_scores=np.asarray(context["final_window_scores"], dtype=np.float32),
        keep_positions=np.asarray(selected["keep_positions"], dtype=np.int64),
        kept_window_indices=np.asarray(selected["kept_window_indices"], dtype=np.int64),
        kept_sample_indices=np.asarray(selected["kept_sample_indices"], dtype=np.int64),
    )
    (output_dir / "joint_window_episode_prior_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )


def write_gain_coverage_report(output_dir: Path, *, heuristic: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "heuristic_method": "gain_plus_coverage",
        "recommendation_reason": heuristic["recommendation_reason"],
        "recommended_keep_count": int(heuristic["recommended_keep_count"]),
        "recommended_keep_ratio": float(heuristic["recommended_keep_ratio"]),
        "min_feasible_keep_count": int(heuristic["min_feasible_keep_count"]),
        "min_feasible_keep_ratio": float(heuristic["min_feasible_keep_ratio"]),
        "gain_delta": int(heuristic["gain_delta"]),
        "max_relative_gain": float(heuristic["max_relative_gain"]),
        "min_episode_coverage_ratio": float(heuristic["min_episode_coverage_ratio"]),
        "min_episode_ratio_with_min_windows": float(heuristic["min_episode_ratio_with_min_windows"]),
        "recommended_gain_abs": float(heuristic["recommended_gain_abs"]),
        "recommended_gain_rel": float(heuristic["recommended_gain_rel"]),
        "recommended_covered_episodes": int(heuristic["recommended_covered_episodes"]),
        "recommended_covered_episode_ratio": float(heuristic["recommended_covered_episode_ratio"]),
        "recommended_episodes_meeting_min_windows": int(heuristic["recommended_episodes_meeting_min_windows"]),
        "recommended_episodes_meeting_min_windows_ratio": float(
            heuristic["recommended_episodes_meeting_min_windows_ratio"]
        ),
        "recommended_selection_rule": heuristic["recommended_selection_rule"],
        "recommended_selection_base_reserved_windows": int(heuristic["recommended_selection_base_reserved_windows"]),
    }
    (output_dir / "gain_coverage_selection_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    np.savez_compressed(
        output_dir / "gain_coverage_selection_curves.npz",
        keep_counts=np.asarray(heuristic["keep_counts"], dtype=np.int64),
        keep_ratios=np.asarray(heuristic["keep_ratios"], dtype=np.float32),
        kept_mean_score_curve=np.asarray(heuristic["kept_mean_score_curve"], dtype=np.float32),
        gain_abs_curve=np.asarray(heuristic["gain_abs_curve"], dtype=np.float32),
        gain_rel_curve=np.asarray(heuristic["gain_rel_curve"], dtype=np.float32),
        covered_episode_counts=np.asarray(heuristic["covered_episode_counts"], dtype=np.int64),
        covered_episode_ratios=np.asarray(heuristic["covered_episode_ratios"], dtype=np.float32),
        min_window_episode_counts=np.asarray(heuristic["min_window_episode_counts"], dtype=np.int64),
        min_window_episode_ratios=np.asarray(heuristic["min_window_episode_ratios"], dtype=np.float32),
        selection_base_reserved_windows=np.asarray(heuristic["selection_base_reserved_windows"], dtype=np.int64),
    )


def required_episode_count(total_episodes: int, ratio: float) -> int:
    return int(math.ceil(float(total_episodes) * float(ratio) - 1e-12))


def recompute_norm_stats_with_fallback(
    train_cfg: _config.TrainConfig,
    *,
    sample_indices_path: Path,
    output_dir: Path,
    max_frames: int | None,
) -> tuple[Path, bool]:
    try:
        base._recompute_norm_stats_for_sample_subset(  # noqa: SLF001
            train_cfg,
            sample_indices_path=sample_indices_path,
            output_dir=output_dir,
            max_frames=max_frames,
        )
        return output_dir / "norm_stats.json", False
    except PermissionError:
        single_worker_cfg = dataclasses.replace(train_cfg, num_workers=0)
        base._recompute_norm_stats_for_sample_subset(  # noqa: SLF001
            single_worker_cfg,
            sample_indices_path=sample_indices_path,
            output_dir=output_dir,
            max_frames=max_frames,
        )
        return output_dir / "norm_stats.json", True
