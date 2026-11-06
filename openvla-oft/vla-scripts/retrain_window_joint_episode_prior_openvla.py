#!/usr/bin/env python3
"""
Window-level joint filtering with an episode-level prior for OpenVLA LeRobot training.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from prismatic.vla.lerobot_openvla import (
    LeRobotDatasetMetadata,
    OpenVLALeRobotConfig,
    finetune_openvla_lerobot,
    load_finetune_config,
    recompute_norm_stats_for_indices,
)


def _signed_power(values: np.ndarray, power: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if power < 0:
        raise ValueError("joint powers must be >= 0")
    return np.sign(values) * (np.abs(values) ** power)


def _minmax_normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if values.size == 0:
        raise ValueError("Cannot normalize an empty array.")
    vmin = float(np.min(values))
    vmax = float(np.max(values))
    if abs(vmax - vmin) < 1e-12:
        return np.ones_like(values, dtype=np.float32)
    return ((values - vmin) / (vmax - vmin)).astype(np.float32)


def _to_percentile_score(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if values.size <= 1:
        return np.ones_like(values, dtype=np.float32)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float32)
    ranks[order] = np.arange(values.size, dtype=np.float32)
    return (ranks / float(values.size - 1)).astype(np.float32)


def _combine_joint_scores(
    quality_scores: np.ndarray,
    normalized_influence: np.ndarray,
    *,
    joint_rule: str,
    joint_quality_power: float,
    joint_influence_power: float,
) -> np.ndarray:
    quality_term = _signed_power(quality_scores, joint_quality_power)
    influence_term = normalized_influence ** joint_influence_power
    if joint_rule == "sum":
        return quality_term + influence_term
    if joint_rule == "product":
        return quality_term * influence_term
    raise ValueError(f"Unsupported joint rule: {joint_rule}")


def _lower_better_to_score(
    values: np.ndarray,
    normalizer: str,
    quantile_low: float,
    quantile_high: float,
) -> np.ndarray:
    vals = np.asarray(values, dtype=np.float32)
    out = np.full_like(vals, np.nan, dtype=np.float32)
    finite = np.isfinite(vals)
    if not finite.any():
        return out
    v = vals[finite]
    if normalizer == "quantile":
        v_min = float(np.quantile(v, min(quantile_low, quantile_high)))
        v_max = float(np.quantile(v, max(quantile_low, quantile_high)))
    else:
        v_min = float(np.min(v))
        v_max = float(np.max(v))
    if np.isclose(v_min, v_max):
        out[finite] = 1.0
        return out
    out[finite] = 1.0 - (v - v_min) / (v_max - v_min)
    return np.clip(out, 0.0, 1.0)


def _higher_better_to_score(
    values: np.ndarray,
    normalizer: str,
    quantile_low: float,
    quantile_high: float,
) -> np.ndarray:
    vals = np.asarray(values, dtype=np.float32)
    out = np.full_like(vals, np.nan, dtype=np.float32)
    finite = np.isfinite(vals)
    if not finite.any():
        return out
    v = vals[finite]
    if normalizer == "quantile":
        v_min = float(np.quantile(v, min(quantile_low, quantile_high)))
        v_max = float(np.quantile(v, max(quantile_low, quantile_high)))
    else:
        v_min = float(np.min(v))
        v_max = float(np.max(v))
    if np.isclose(v_min, v_max):
        out[finite] = 1.0
        return out
    out[finite] = (v - v_min) / (v_max - v_min)
    return np.clip(out, 0.0, 1.0)


def _pairwise_distance(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=np.float32)
    return np.linalg.norm(x[:, None, :] - x[None, :, :], axis=2)


def _load_window_payload(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        payload = pickle.load(f)
    if not isinstance(payload, dict) or "window_influence_scores" not in payload:
        raise ValueError(f"{path} does not contain a valid window influence payload.")
    return payload


def _load_window_sequences(payload: dict[str, Any], metadata: LeRobotDatasetMetadata) -> dict[str, np.ndarray]:
    window_episode_indices = np.asarray(payload["metadata"]["train_window_episode_indices"], dtype=np.int64)
    window_frame_starts = np.asarray(payload["metadata"]["train_window_frame_starts"], dtype=np.int64)
    window_frame_ends = np.asarray(payload["metadata"]["train_window_frame_ends"], dtype=np.int64)

    num_windows = int(window_episode_indices.shape[0])
    action_jitter = np.full(num_windows, np.nan, dtype=np.float32)
    state_jitter = np.full(num_windows, np.nan, dtype=np.float32)
    action_energy = np.full(num_windows, np.nan, dtype=np.float32)
    state_energy = np.full(num_windows, np.nan, dtype=np.float32)
    state_efficiency = np.full(num_windows, np.nan, dtype=np.float32)
    state_progress = np.full(num_windows, np.nan, dtype=np.float32)
    action_rel_jitter = np.full(num_windows, np.nan, dtype=np.float32)
    state_rel_jitter = np.full(num_windows, np.nan, dtype=np.float32)
    window_lengths = np.full(num_windows, np.nan, dtype=np.float32)
    terminal_states: list[np.ndarray | None] = [None] * num_windows

    for window_idx in range(num_windows):
        episode_index = int(window_episode_indices[window_idx])
        payload_ep = metadata.load_episode_payload(episode_index)
        start = int(window_frame_starts[window_idx])
        end = int(window_frame_ends[window_idx])
        if end < start:
            continue
        sl = slice(start, end + 1)
        state_window = payload_ep.states[sl]
        action_window = payload_ep.actions[sl]

        if state_window.shape[0] > 0:
            window_lengths[window_idx] = float(state_window.shape[0])
            terminal_states[window_idx] = np.asarray(state_window[-1], dtype=np.float32)
        if state_window.shape[0] >= 2:
            ds = np.diff(state_window, axis=0)
            ds_norm = np.linalg.norm(ds, axis=1)
            path = float(ds_norm.sum())
            disp = float(np.linalg.norm(state_window[-1] - state_window[0]))
            state_energy[window_idx] = float(np.linalg.norm(state_window, axis=1).mean())
            state_jitter[window_idx] = float(ds_norm.mean())
            state_progress[window_idx] = disp
            state_efficiency[window_idx] = disp / (path + 1e-8)
            state_rel_jitter[window_idx] = float(ds_norm.mean() / (state_energy[window_idx] + 1e-8))

        if action_window.shape[0] >= 1:
            action_energy[window_idx] = float(np.linalg.norm(action_window, axis=1).mean())
        if action_window.shape[0] >= 2:
            da = np.diff(action_window, axis=0)
            da_norm = np.linalg.norm(da, axis=1)
            action_jitter[window_idx] = float(da_norm.mean())
            action_rel_jitter[window_idx] = float(da_norm.mean() / (action_energy[window_idx] + 1e-8))

    terminal_dist = np.full(num_windows, np.nan, dtype=np.float32)
    valid_terminal_ids = [idx for idx, state in enumerate(terminal_states) if state is not None]
    if valid_terminal_ids:
        stacked = np.stack([terminal_states[idx] for idx in valid_terminal_ids], axis=0)
        if len(valid_terminal_ids) == 1:
            terminal_dist[valid_terminal_ids[0]] = 0.0
        else:
            k = max(1, min(15, len(valid_terminal_ids) - 1))
            dist_mat = _pairwise_distance(stacked)
            np.fill_diagonal(dist_mat, np.inf)
            nearest = np.partition(dist_mat, kth=k - 1, axis=1)[:, :k]
            agg = nearest.mean(axis=1)
            for idx, dist in zip(valid_terminal_ids, agg, strict=True):
                terminal_dist[idx] = float(dist)

    return {
        "window_lengths": window_lengths,
        "action_jitter": action_jitter,
        "state_jitter": state_jitter,
        "action_energy": action_energy,
        "state_energy": state_energy,
        "state_efficiency": state_efficiency,
        "state_progress": state_progress,
        "action_rel_jitter": action_rel_jitter,
        "state_rel_jitter": state_rel_jitter,
        "terminal_dist": terminal_dist,
    }


def _compute_window_quality_scores(
    payload: dict[str, Any],
    metadata: LeRobotDatasetMetadata,
    *,
    quality_source: str,
    normalizer: str,
    quantile_low: float,
    quantile_high: float,
    episode_action_weight: float,
    episode_state_weight: float,
    episode_length_weight: float,
    episode_energy_weight: float,
    trajectory_terminal_weight: float,
    trajectory_efficiency_weight: float,
    trajectory_progress_weight: float,
    trajectory_state_smooth_weight: float,
    trajectory_action_smooth_weight: float,
    trajectory_length_weight: float,
    rank_fusion_episode_weight: float,
    rank_fusion_trajectory_weight: float,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    metrics = _load_window_sequences(payload, metadata)
    window_lengths = metrics["window_lengths"]
    finite_len = np.isfinite(window_lengths)
    length_score = np.full_like(window_lengths, np.nan, dtype=np.float32)
    if finite_len.any():
        lens = window_lengths[finite_len]
        med = float(np.median(lens))
        mad = float(np.median(np.abs(lens - med)))
        denom = max(1e-6, mad * 3.0)
        length_score[finite_len] = np.clip(np.exp(-np.abs(lens - med) / denom), 0.0, 1.0)

    episode_quality = np.full_like(window_lengths, np.nan, dtype=np.float32)
    trajectory_quality = np.full_like(window_lengths, np.nan, dtype=np.float32)

    if quality_source in {"episode_eval", "rank_fusion"}:
        action_metric = metrics["action_jitter"] / np.maximum(metrics["action_energy"], 1e-8)
        state_metric = metrics["state_jitter"] / np.maximum(metrics["state_energy"], 1e-8)
        action_score = _lower_better_to_score(action_metric, normalizer, quantile_low, quantile_high)
        state_score = _lower_better_to_score(state_metric, normalizer, quantile_low, quantile_high)
        energy_score = _lower_better_to_score(metrics["action_energy"], normalizer, quantile_low, quantile_high)
        pairs = [
            (action_score, max(episode_action_weight, 0.0)),
            (state_score, max(episode_state_weight, 0.0)),
            (length_score, max(episode_length_weight, 0.0)),
            (energy_score, max(episode_energy_weight, 0.0)),
        ]
        for idx in range(episode_quality.shape[0]):
            num = 0.0
            den = 0.0
            for comp, weight in pairs:
                value = comp[idx]
                if weight > 0.0 and np.isfinite(value):
                    num += weight * float(value)
                    den += weight
            if den > 0.0:
                episode_quality[idx] = float(num / den)

    if quality_source in {"trajectory_eval", "rank_fusion"}:
        terminal_score = _lower_better_to_score(metrics["terminal_dist"], normalizer, quantile_low, quantile_high)
        efficiency_score = _higher_better_to_score(metrics["state_efficiency"], normalizer, quantile_low, quantile_high)
        progress_score = _higher_better_to_score(metrics["state_progress"], normalizer, quantile_low, quantile_high)
        state_smooth_score = _lower_better_to_score(metrics["state_rel_jitter"], normalizer, quantile_low, quantile_high)
        action_smooth_score = _lower_better_to_score(metrics["action_rel_jitter"], normalizer, quantile_low, quantile_high)
        pairs = [
            (terminal_score, max(trajectory_terminal_weight, 0.0)),
            (efficiency_score, max(trajectory_efficiency_weight, 0.0)),
            (progress_score, max(trajectory_progress_weight, 0.0)),
            (state_smooth_score, max(trajectory_state_smooth_weight, 0.0)),
            (action_smooth_score, max(trajectory_action_smooth_weight, 0.0)),
            (length_score, max(trajectory_length_weight, 0.0)),
        ]
        for idx in range(trajectory_quality.shape[0]):
            num = 0.0
            den = 0.0
            for comp, weight in pairs:
                value = comp[idx]
                if weight > 0.0 and np.isfinite(value):
                    num += weight * float(value)
                    den += weight
            if den > 0.0:
                trajectory_quality[idx] = float(num / den)

    if quality_source == "episode_eval":
        quality = episode_quality
    elif quality_source == "trajectory_eval":
        quality = trajectory_quality
    else:
        episode_pct = _to_percentile_score(np.nan_to_num(episode_quality, nan=0.0))
        trajectory_pct = _to_percentile_score(np.nan_to_num(trajectory_quality, nan=0.0))
        denom = max(rank_fusion_episode_weight + rank_fusion_trajectory_weight, 1e-8)
        quality = (
            rank_fusion_episode_weight * episode_pct + rank_fusion_trajectory_weight * trajectory_pct
        ) / denom

    if np.any(~np.isfinite(quality)):
        fill_value = float(np.nanmean(quality)) if np.isfinite(quality).any() else 0.0
        quality = np.where(np.isfinite(quality), quality, fill_value)

    components = {
        "window_lengths": window_lengths.astype(np.float32),
        "episode_eval_quality": np.asarray(episode_quality, dtype=np.float32),
        "trajectory_eval_quality": np.asarray(trajectory_quality, dtype=np.float32),
        "final_window_quality": np.asarray(quality, dtype=np.float32),
    }
    components.update({key: np.asarray(value, dtype=np.float32) for key, value in metrics.items()})
    return np.asarray(quality, dtype=np.float32), components


def _aggregate_episode_quality(
    window_episode_indices: np.ndarray,
    window_quality_scores: np.ndarray,
    *,
    agg: str,
) -> tuple[np.ndarray, np.ndarray]:
    unique_episode_indices = np.asarray(sorted(np.unique(window_episode_indices).tolist()), dtype=np.int64)
    episode_quality_scores = np.zeros((len(unique_episode_indices),), dtype=np.float32)
    for out_idx, episode_index in enumerate(unique_episode_indices.tolist()):
        values = np.asarray(window_quality_scores[window_episode_indices == episode_index], dtype=np.float32)
        if values.size == 0:
            episode_quality_scores[out_idx] = 0.0
        elif agg == "max":
            episode_quality_scores[out_idx] = float(np.max(values))
        else:
            episode_quality_scores[out_idx] = float(np.mean(values))
    return unique_episode_indices, episode_quality_scores


def _align_episode_influence(payload: dict[str, Any], target_episode_indices: np.ndarray) -> np.ndarray:
    score_episode_indices = np.asarray(payload["metadata"]["train_episode_indices"], dtype=np.int64).reshape(-1)
    influence_scores = np.asarray(payload["episode_influence_scores"], dtype=np.float32).reshape(-1)
    influence_map = {int(idx): float(score) for idx, score in zip(score_episode_indices.tolist(), influence_scores.tolist(), strict=True)}
    return np.asarray([influence_map[int(idx)] for idx in target_episode_indices.tolist()], dtype=np.float32)


def _build_episode_value_map(episode_indices: np.ndarray, episode_values: np.ndarray) -> dict[int, float]:
    return {int(idx): float(value) for idx, value in zip(episode_indices.tolist(), episode_values.tolist(), strict=True)}


def _map_episode_values_to_windows(window_episode_indices: np.ndarray, episode_value_map: dict[int, float]) -> np.ndarray:
    return np.asarray([episode_value_map[int(idx)] for idx in window_episode_indices.tolist()], dtype=np.float32)


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
        window_term = _minmax_normalize(window_joint_scores)
        episode_term = _minmax_normalize(episode_joint_per_window)
    elif fusion_normalizer == "percentile":
        window_term = _to_percentile_score(window_joint_scores)
        episode_term = _to_percentile_score(episode_joint_per_window)
    else:
        window_term = np.asarray(window_joint_scores, dtype=np.float32)
        episode_term = np.asarray(episode_joint_per_window, dtype=np.float32)

    interaction_term = np.asarray(window_term, dtype=np.float32) * np.asarray(episode_term, dtype=np.float32)
    denom = np.maximum(float(window_score_weight) + episode_prior_weights + float(interaction_weight), 1e-8)
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

    unique_episode_indices = np.asarray(sorted(np.unique(window_episode_indices).tolist()), dtype=np.int64)
    window_std = np.zeros((len(unique_episode_indices),), dtype=np.float32)
    quality_std = np.zeros((len(unique_episode_indices),), dtype=np.float32)
    window_counts = np.zeros((len(unique_episode_indices),), dtype=np.float32)
    for out_idx, episode_index in enumerate(unique_episode_indices.tolist()):
        mask = window_episode_indices == episode_index
        window_std[out_idx] = float(np.std(window_joint_scores[mask])) if mask.any() else 0.0
        quality_std[out_idx] = float(np.std(window_quality_scores[mask])) if mask.any() else 0.0
        window_counts[out_idx] = float(np.sum(mask))

    window_std_signal = _minmax_normalize(window_std)
    quality_std_signal = _minmax_normalize(quality_std)
    inverse_length_signal = 1.0 - _minmax_normalize(window_counts)
    if adaptive_prior_mode == "window_std":
        episode_signal = window_std_signal
    elif adaptive_prior_mode == "inverse_episode_length":
        episode_signal = inverse_length_signal
    elif adaptive_prior_mode == "quality_std":
        episode_signal = quality_std_signal
    else:
        episode_signal = (window_std_signal + quality_std_signal + inverse_length_signal) / 3.0

    centered = 2.0 * np.asarray(episode_signal, dtype=np.float32) - 1.0
    episode_weights = np.clip(float(base_weight) * (1.0 + float(adaptive_prior_strength) * centered), 0.0, 1.0)
    signal_map = _build_episode_value_map(unique_episode_indices, episode_signal)
    weight_map = _build_episode_value_map(unique_episode_indices, episode_weights)
    debug = {
        "adaptive_prior_signal_per_window": _map_episode_values_to_windows(window_episode_indices, signal_map),
        "adaptive_prior_weight_per_window": _map_episode_values_to_windows(window_episode_indices, weight_map),
    }
    return debug["adaptive_prior_weight_per_window"], debug


def _allocate_remaining_budget(weights: np.ndarray, capacities: np.ndarray, remaining: int) -> np.ndarray:
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
    if left > 0:
        fractional = raw - np.floor(raw)
        for idx in np.argsort(-fractional):
            if left <= 0:
                break
            if allocations[idx] >= capacities[idx]:
                continue
            allocations[idx] += 1
            left -= 1
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
        keep_positions = np.argsort(-final_window_scores)[:total_keep]
        return np.sort(keep_positions).astype(np.int64), {
            "selection_rule": "global",
            "total_keep_target": total_keep,
            "base_reserved_windows": 0,
        }

    unique_episode_indices = np.asarray(sorted(np.unique(window_episode_indices).tolist()), dtype=np.int64)
    positions_by_episode = [np.flatnonzero(window_episode_indices == idx) for idx in unique_episode_indices.tolist()]
    counts = np.asarray([len(pos) for pos in positions_by_episode], dtype=np.int64)
    base_budget = np.minimum(counts, int(min_windows_per_episode))
    reserved = int(np.sum(base_budget))
    if reserved > total_keep:
        raise ValueError("Episode budget is infeasible for the requested keep ratio.")

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
            selected_positions.extend(pooled[np.argsort(-final_window_scores[pooled])][:remaining_keep].tolist())
        else:
            capacities = np.asarray([len(pos) for pos in remaining_positions], dtype=np.int64)
            weights = episode_budget_scores if episode_budget_rule == "episode_prior" else np.ones_like(capacities)
            extra_budget = _allocate_remaining_budget(weights, capacities, remaining_keep)
            for positions, budget in zip(remaining_positions, extra_budget.tolist(), strict=True):
                if budget > 0:
                    selected_positions.extend(positions[:budget].tolist())

    keep_positions = np.asarray(sorted(set(selected_positions)), dtype=np.int64)
    if keep_positions.shape[0] < total_keep:
        remaining_mask = np.ones((total_windows,), dtype=bool)
        remaining_mask[keep_positions] = False
        leftover_positions = np.flatnonzero(remaining_mask)
        keep_positions = np.concatenate(
            [keep_positions, leftover_positions[np.argsort(-final_window_scores[leftover_positions])][: total_keep - keep_positions.shape[0]]]
        )
    return np.sort(keep_positions).astype(np.int64), {
        "selection_rule": episode_budget_rule,
        "total_keep_target": total_keep,
        "base_reserved_windows": reserved,
    }


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
    (output_dir / "joint_window_episode_prior_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter OpenVLA LeRobot training data by window-level joint score with an episode-level prior."
    )
    parser.add_argument("--score-pkl", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--output-indices", type=Path, default=None)
    parser.add_argument("--exp-name", type=str, default=None)
    parser.add_argument("--retrain-run-root", type=Path, default=Path("runs"))
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
    parser.add_argument("--quality-source", choices=("episode_eval", "trajectory_eval", "rank_fusion"), default="rank_fusion")
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
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    keep_ratio = args.keep_ratio
    if keep_ratio is None:
        if args.drop_ratio is None:
            raise ValueError("Either --keep-ratio or --drop-ratio must be provided.")
        keep_ratio = 1.0 - float(args.drop_ratio)

    payload = _load_window_payload(args.score_pkl)
    base_cfg = load_finetune_config(args.checkpoint_dir)
    metadata = LeRobotDatasetMetadata(
        base_cfg.dataset_root,
        dataset_name=base_cfg.dataset_name,
        task_description_override=base_cfg.task_description,
        norm_stats_path=base_cfg.norm_stats_path,
    )

    meta = payload["metadata"]
    window_indices = np.asarray(meta["train_window_indices"], dtype=np.int64).reshape(-1)
    window_episode_indices = np.asarray(meta["train_window_episode_indices"], dtype=np.int64).reshape(-1)
    sample_window_indices = np.asarray(meta["train_sample_window_indices"], dtype=np.int64).reshape(-1)
    dataset_sample_indices = np.asarray(meta["train_dataset_sample_indices"], dtype=np.int64).reshape(-1)
    window_influence_scores = np.asarray(payload["window_influence_scores"], dtype=np.float32).reshape(-1)

    window_quality_scores, quality_components = _compute_window_quality_scores(
        payload,
        metadata,
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

    window_normalized_influence = _minmax_normalize(window_influence_scores)
    window_joint_scores = _combine_joint_scores(
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
    episode_normalized_influence = _minmax_normalize(episode_influence_scores)
    episode_joint_scores = _combine_joint_scores(
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
    normalized_window_joint_scores, normalized_episode_joint_scores, interaction_scores, final_window_scores = _fuse_scores(
        window_joint_scores,
        episode_joint_per_window,
        window_score_weight=args.window_score_weight,
        episode_prior_weights=adaptive_prior_weight_per_window,
        interaction_weight=args.interaction_weight,
        fusion_normalizer=args.fusion_normalizer,
    )
    normalized_episode_joint_scores_by_episode = np.asarray(
        [float(np.max(normalized_episode_joint_scores[window_episode_indices == idx])) for idx in episode_indices.tolist()],
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

    keep_pct = int(round(keep_ratio * 100))
    output_dir = args.output_dir or (
        metadata.dataset_root.parent
        / f"{metadata.dataset_root.name}_window_episode_prior_{args.joint_rule}_keep{keep_pct}_{args.quality_source}_{args.score_pkl.stem}"
    )
    output_indices = args.output_indices or (output_dir / "kept_sample_indices.npy")
    output_dir.mkdir(parents=True, exist_ok=True)
    if output_indices.exists() and not args.overwrite:
        raise FileExistsError(f"{output_indices} already exists. Use --overwrite to replace it.")
    np.save(output_indices, kept_sample_indices)

    summary = {
        "score_pkl": str(args.score_pkl),
        "checkpoint_dir": str(args.checkpoint_dir),
        "dataset_root": str(metadata.dataset_root),
        "quality_source": args.quality_source,
        "joint_rule": args.joint_rule,
        "fusion_normalizer": args.fusion_normalizer,
        "window_score_weight": float(args.window_score_weight),
        "episode_prior_weight": float(args.episode_prior_weight),
        "adaptive_prior_mode": args.adaptive_prior_mode,
        "adaptive_prior_strength": float(args.adaptive_prior_strength),
        "interaction_weight": float(args.interaction_weight),
        "selection_rule": str(selection_debug["selection_rule"]),
        "selection_total_keep_target": int(selection_debug["total_keep_target"]),
        "selection_base_reserved_windows": int(selection_debug["base_reserved_windows"]),
        "num_windows": int(window_indices.shape[0]),
        "num_episodes": int(episode_indices.shape[0]),
        "kept_windows": int(kept_window_indices.shape[0]),
        "kept_samples": int(kept_sample_indices.shape[0]),
        "keep_ratio": float(keep_ratio),
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

    subset_norm_stats_path = output_dir / "norm_stats.json"
    recompute_norm_stats_for_indices(metadata, kept_sample_indices, subset_norm_stats_path)

    if args.prepare_only:
        return

    exp_name = args.exp_name or (
        f"{base_cfg.dataset_name}_window_episode_prior_{args.joint_rule}_keep{keep_pct}_{args.quality_source}"
    )
    retrain_output_dir = args.retrain_run_root / exp_name
    retrain_cfg = dataclasses.replace(
        base_cfg,
        output_dir=retrain_output_dir,
        run_root_dir=args.retrain_run_root,
        sample_indices_path=output_indices,
        norm_stats_path=subset_norm_stats_path,
        max_steps=args.num_train_steps or base_cfg.max_steps,
        run_id_note=exp_name,
    )
    finetune_openvla_lerobot(retrain_cfg)


if __name__ == "__main__":
    main()
