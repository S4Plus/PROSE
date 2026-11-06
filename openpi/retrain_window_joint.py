#!/usr/bin/env python3
"""Retrain OpenPI from window-level joint influence/quality scores.

This script is independent from the existing episode-level retraining scripts.
It consumes the output of `scripts/influence_pi05_lora_grass_window.py`,
recomputes window-level quality scores inspired by
`/workspace/liushuo/cupid/act/eval_cupid_quality.py`, combines them with
window-level influence scores, maps selected windows back to dataset samples,
recomputes norm stats on the kept sample subset, and optionally launches
training with `data.sample_indices_path`.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import tqdm

import openpi.shared.normalize as normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import retrain as _retrain
import retrain_grass as retrain_grass
from scripts import compute_norm_stats as norm_stats_script


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


def _zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return values
    mean = float(np.mean(values))
    std = float(np.std(values))
    if std < 1e-8:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - mean) / std).astype(np.float32)


def _to_percentile_score(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if values.size == 0:
        return values
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float32)
    ranks[order] = np.arange(values.size, dtype=np.float32)
    if values.size == 1:
        return np.ones_like(values, dtype=np.float32)
    return (ranks / float(values.size - 1)).astype(np.float32)


def _lower_better_to_score(
    values: np.ndarray,
    normalizer: str = "quantile",
    quantile_low: float = 0.05,
    quantile_high: float = 0.95,
) -> np.ndarray:
    vals = np.asarray(values, dtype=np.float32)
    out = np.full_like(vals, np.nan, dtype=np.float32)
    finite = np.isfinite(vals)
    if not finite.any():
        return out
    v = vals[finite]
    if normalizer == "quantile":
        q_low = float(np.clip(quantile_low, 0.0, 1.0))
        q_high = float(np.clip(quantile_high, 0.0, 1.0))
        if q_high < q_low:
            q_low, q_high = q_high, q_low
        v_min = float(np.quantile(v, q_low))
        v_max = float(np.quantile(v, q_high))
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
    normalizer: str = "quantile",
    quantile_low: float = 0.05,
    quantile_high: float = 0.95,
) -> np.ndarray:
    vals = np.asarray(values, dtype=np.float32)
    out = np.full_like(vals, np.nan, dtype=np.float32)
    finite = np.isfinite(vals)
    if not finite.any():
        return out
    v = vals[finite]
    if normalizer == "quantile":
        q_low = float(np.clip(quantile_low, 0.0, 1.0))
        q_high = float(np.clip(quantile_high, 0.0, 1.0))
        if q_high < q_low:
            q_low, q_high = q_high, q_low
        v_min = float(np.quantile(v, q_low))
        v_max = float(np.quantile(v, q_high))
    else:
        v_min = float(np.min(v))
        v_max = float(np.max(v))
    if np.isclose(v_min, v_max):
        out[finite] = 1.0
        return out
    out[finite] = (v - v_min) / (v_max - v_min)
    return np.clip(out, 0.0, 1.0)


def _pairwise_distance(values: np.ndarray, metric: str) -> np.ndarray:
    x = np.asarray(values, dtype=np.float32)
    if metric == "l1":
        return np.sum(np.abs(x[:, None, :] - x[None, :, :]), axis=2)
    if metric == "cosine":
        denom = np.linalg.norm(x, axis=1, keepdims=True)
        denom = np.where(denom > 1e-8, denom, 1.0)
        y = x / denom
        sim = y @ y.T
        return np.clip(1.0 - sim, 0.0, None)
    return np.linalg.norm(x[:, None, :] - x[None, :, :], axis=2)


def _aggregate_knn_distance(nearest: np.ndarray, agg: str, trim_ratio: float) -> np.ndarray:
    if agg == "median":
        return np.median(nearest, axis=1)
    if agg == "trimmed_mean":
        k = int(nearest.shape[1])
        if k <= 2:
            return nearest.mean(axis=1)
        ratio = float(np.clip(trim_ratio, 0.0, 0.49))
        cut = int(np.floor(k * ratio))
        if cut <= 0 or cut * 2 >= k:
            return nearest.mean(axis=1)
        sorted_nearest = np.sort(nearest, axis=1)
        return sorted_nearest[:, cut : k - cut].mean(axis=1)
    return nearest.mean(axis=1)


def _load_window_payload(score_pkl: Path) -> dict[str, Any]:
    payload = _retrain._load_score_payload(score_pkl)  # noqa: SLF001
    if not isinstance(payload, dict):
        raise ValueError(f"{score_pkl} does not contain a dict payload.")
    if "window_influence_scores" not in payload:
        raise ValueError(f"{score_pkl} does not contain window_influence_scores.")
    if "metadata" not in payload or not isinstance(payload["metadata"], dict):
        raise ValueError(f"{score_pkl} is missing metadata.")
    return payload


def _collect_episode_files(dataset_root: Path) -> dict[int, Path]:
    episode_files: dict[int, Path] = {}
    for p in sorted((dataset_root / "data").glob("chunk-*/episode_*.parquet")):
        stem = p.stem
        if not stem.startswith("episode_"):
            continue
        ep_str = stem.split("_")[-1]
        if ep_str.isdigit():
            episode_files[int(ep_str)] = p
    return episode_files


def _load_window_sequences(payload: dict[str, Any], dataset_root: Path) -> dict[str, np.ndarray]:
    try:
        import pyarrow.parquet as pq
    except Exception as e:  # pragma: no cover - environment dependent
        raise RuntimeError("Window quality evaluation requires pyarrow.") from e

    meta = payload["metadata"]
    window_episode_indices = np.asarray(meta["train_window_episode_indices"], dtype=np.int64)
    window_frame_starts = np.asarray(meta["train_window_frame_starts"], dtype=np.int64)
    window_frame_ends = np.asarray(meta["train_window_frame_ends"], dtype=np.int64)

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

    episode_files = _collect_episode_files(dataset_root)
    cached_episode_data: dict[int, tuple[np.ndarray | None, np.ndarray | None]] = {}

    for window_idx in range(num_windows):
        episode_idx = int(window_episode_indices[window_idx])
        if episode_idx not in cached_episode_data:
            episode_path = episode_files.get(episode_idx)
            if episode_path is None:
                cached_episode_data[episode_idx] = (None, None)
            else:
                schema = pq.read_schema(episode_path)
                available = set(schema.names)
                cols = [c for c in ("action", "observation.state") if c in available]
                table = pq.read_table(episode_path, columns=cols if cols else None)
                actions = None
                states = None
                if "action" in table.column_names:
                    actions = np.asarray(table["action"].to_pylist(), dtype=np.float32)
                if "observation.state" in table.column_names:
                    states = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
                cached_episode_data[episode_idx] = (actions, states)

        actions, states = cached_episode_data[episode_idx]
        start = int(window_frame_starts[window_idx])
        end = int(window_frame_ends[window_idx])
        if end < start:
            continue
        sl = slice(start, end + 1)

        if states is not None and states.ndim == 2 and end < states.shape[0]:
            state_window = states[sl]
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

        if actions is not None and actions.ndim == 2 and end < actions.shape[0]:
            action_window = actions[sl]
            if not np.isfinite(window_lengths[window_idx]) and action_window.shape[0] > 0:
                window_lengths[window_idx] = float(action_window.shape[0])
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
            dist_mat = _pairwise_distance(stacked, metric="l2")
            np.fill_diagonal(dist_mat, np.inf)
            nearest = np.partition(dist_mat, kth=k - 1, axis=1)[:, :k]
            agg = _aggregate_knn_distance(nearest, agg="mean", trim_ratio=0.1)
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
    dataset_root: Path,
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
    metrics = _load_window_sequences(payload, dataset_root)

    window_lengths = metrics["window_lengths"]
    finite_len = np.isfinite(window_lengths)
    length_score = np.full_like(window_lengths, np.nan, dtype=np.float32)
    if finite_len.any():
        lens = window_lengths[finite_len]
        med = float(np.median(lens))
        mad = float(np.median(np.abs(lens - med)))
        denom = max(1e-6, mad * 3.0)
        length_score[finite_len] = np.exp(-np.abs(lens - med) / denom)
        length_score = np.clip(length_score, 0.0, 1.0)

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
                if weight <= 0.0:
                    continue
                value = comp[idx]
                if np.isfinite(value):
                    num += weight * float(value)
                    den += weight
            if den > 0.0:
                episode_quality[idx] = float(num / den)

    if quality_source in {"trajectory_eval", "rank_fusion"}:
        terminal_score = _lower_better_to_score(metrics["terminal_dist"], normalizer, quantile_low, quantile_high)
        efficiency_score = _higher_better_to_score(metrics["state_efficiency"], normalizer, quantile_low, quantile_high)
        progress_score = _higher_better_to_score(metrics["state_progress"], normalizer, quantile_low, quantile_high)
        state_smooth_score = _lower_better_to_score(
            metrics["state_rel_jitter"], normalizer, quantile_low, quantile_high
        )
        action_smooth_score = _lower_better_to_score(
            metrics["action_rel_jitter"], normalizer, quantile_low, quantile_high
        )
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
                if weight <= 0.0:
                    continue
                value = comp[idx]
                if np.isfinite(value):
                    num += weight * float(value)
                    den += weight
            if den > 0.0:
                trajectory_quality[idx] = float(num / den)

    if quality_source == "episode_eval":
        quality = episode_quality
    elif quality_source == "trajectory_eval":
        quality = trajectory_quality
    else:
        ep_pct = _to_percentile_score(np.nan_to_num(episode_quality, nan=0.0))
        traj_pct = _to_percentile_score(np.nan_to_num(trajectory_quality, nan=0.0))
        w_ep = max(float(rank_fusion_episode_weight), 0.0)
        w_traj = max(float(rank_fusion_trajectory_weight), 0.0)
        denom = max(w_ep + w_traj, 1e-8)
        quality = ((w_ep * ep_pct) + (w_traj * traj_pct)) / denom

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


def _write_selection_artifacts(
    output_dir: Path,
    *,
    window_indices: np.ndarray,
    influence_scores: np.ndarray,
    quality_scores: np.ndarray,
    normalized_influence: np.ndarray,
    joint_scores: np.ndarray,
    kept_window_indices: np.ndarray,
    kept_sample_indices: np.ndarray,
    summary: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "joint_window_selection_scores.npz",
        window_indices=np.asarray(window_indices, dtype=np.int64),
        influence_scores=np.asarray(influence_scores, dtype=np.float32),
        quality_scores=np.asarray(quality_scores, dtype=np.float32),
        normalized_influence=np.asarray(normalized_influence, dtype=np.float32),
        joint_scores=np.asarray(joint_scores, dtype=np.float32),
        kept_window_indices=np.asarray(kept_window_indices, dtype=np.int64),
        kept_sample_indices=np.asarray(kept_sample_indices, dtype=np.int64),
    )
    (output_dir / "joint_window_selection_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )


def _build_sample_subset_data_factory(
    data_factory: _config.DataConfigFactory,
    *,
    sample_indices_path: Path,
    output_dir: Path,
) -> _config.DataConfigFactory:
    return dataclasses.replace(
        data_factory,
        sample_indices_path=str(sample_indices_path),
        assets=_config.AssetsConfig(
            assets_dir=str(output_dir.parent),
            asset_id=output_dir.name,
        ),
    )


def _recompute_norm_stats_for_sample_subset(
    train_cfg: _config.TrainConfig,
    *,
    sample_indices_path: Path,
    output_dir: Path,
    max_frames: int | None,
) -> None:
    data_factory = _build_sample_subset_data_factory(
        train_cfg.data,
        sample_indices_path=sample_indices_path,
        output_dir=output_dir,
    )
    data_cfg = data_factory.create(train_cfg.assets_dirs, train_cfg.model)

    if data_cfg.rlds_data_dir is not None:
        data_loader, num_batches = norm_stats_script.create_rlds_dataloader(
            data_cfg,
            train_cfg.model.action_horizon,
            train_cfg.batch_size,
            max_frames=max_frames,
        )
    else:
        dataset = _data_loader.create_torch_dataset(
            data_cfg,
            train_cfg.model.action_horizon,
            train_cfg.model,
        )
        dataset = _data_loader.TransformedDataset(
            dataset,
            [
                *data_cfg.repack_transforms.inputs,
                *data_cfg.data_transforms.inputs,
                norm_stats_script.RemoveStrings(),
            ],
        )
        if max_frames is not None and max_frames < len(dataset):
            num_batches = max(1, max_frames // train_cfg.batch_size)
            shuffle = True
        else:
            num_batches = max(1, len(dataset) // train_cfg.batch_size)
            shuffle = False
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=train_cfg.batch_size,
            shuffle=shuffle,
            num_workers=train_cfg.num_workers,
            collate_fn=_data_loader._collate_fn,  # noqa: SLF001
            drop_last=True,
        )

    stats = {key: normalize.RunningStats() for key in ("state", "actions")}
    for batch in tqdm.tqdm(data_loader, total=num_batches, desc="Computing subset norm_stats"):
        for key in stats:
            stats[key].update(np.asarray(batch[key]))

    norm_stats = {key: running.get_statistics() for key, running in stats.items()}
    normalize.save(output_dir, norm_stats)


def _build_train_config(
    base_train_cfg: _config.TrainConfig,
    *,
    sample_indices_path: Path,
    output_dir: Path,
    exp_name: str,
    checkpoint_base_dir: Path,
    overwrite: bool,
    num_train_steps: int | None,
) -> _config.TrainConfig:
    filtered_data = _build_sample_subset_data_factory(
        base_train_cfg.data,
        sample_indices_path=sample_indices_path,
        output_dir=output_dir,
    )
    kwargs: dict[str, Any] = {
        "data": filtered_data,
        "exp_name": exp_name,
        "checkpoint_base_dir": str(checkpoint_base_dir),
        "overwrite": overwrite,
        "resume": False,
    }
    if num_train_steps is not None:
        kwargs["num_train_steps"] = int(num_train_steps)
    return dataclasses.replace(base_train_cfg, **kwargs)


def _default_output_dir(
    dataset_dir: Path,
    keep_ratio: float,
    score_pkl: Path,
    quality_source: str,
    joint_rule: str,
) -> Path:
    keep_pct = int(round(keep_ratio * 100))
    return dataset_dir.parent / (
        f"{dataset_dir.name}_window_joint_{joint_rule}_keep{keep_pct}_{quality_source}_{score_pkl.stem}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter OpenPI training data by window-level joint influence/quality score, recompute norm_stats, and retrain."
    )
    parser.add_argument("--score-pkl", type=Path, required=True, help="Window GraSS pickle from influence_pi05_lora_grass_window.py")
    parser.add_argument("--train-config", type=str, default=None, help="Training config name. If omitted, infer from score pickle.")
    parser.add_argument("--dataset-dir", type=Path, default=None, help="Dataset root. If omitted, infer from train config.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory for sample subset, scores, and norm_stats.")
    parser.add_argument("--output-indices", type=Path, default=None, help="Output sample-index file. Defaults under output-dir.")
    parser.add_argument("--exp-name", type=str, default=None, help="Experiment name for retraining.")
    parser.add_argument("--checkpoint-base-dir", type=Path, default=Path("./checkpoints"))
    parser.add_argument("--num-train-steps", type=int, default=None)
    parser.add_argument("--keep-ratio", type=float, default=None)
    parser.add_argument("--drop-ratio", type=float, default=None)
    parser.add_argument("--joint-rule", choices=("product", "sum"), default="product")
    parser.add_argument("--joint-quality-power", type=float, default=1.0)
    parser.add_argument("--joint-influence-power", type=float, default=1.0)
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

    payload = _load_window_payload(args.score_pkl)
    train_config_name = args.train_config or _retrain._extract_score_config_name(payload)  # noqa: SLF001
    if train_config_name is None:
        raise ValueError("--train-config is required because the score pickle does not include config.config_name.")

    if args.dataset_dir is None:
        dataset_dir, default_exp_name = _retrain._infer_defaults_from_train_config(train_config_name)  # noqa: SLF001
    else:
        dataset_dir = args.dataset_dir
        default_exp_name = train_config_name

    influence_scores = np.asarray(payload["window_influence_scores"], dtype=np.float32).reshape(-1)
    metadata = payload["metadata"]
    window_indices = np.asarray(metadata["train_window_indices"], dtype=np.int64).reshape(-1)
    sample_window_indices = np.asarray(metadata["train_sample_window_indices"], dtype=np.int64).reshape(-1)
    dataset_sample_indices = np.asarray(metadata["train_dataset_sample_indices"], dtype=np.int64).reshape(-1)
    if not (
        influence_scores.shape[0] == window_indices.shape[0]
        and sample_window_indices.shape[0] == dataset_sample_indices.shape[0]
    ):
        raise ValueError("Window metadata shape mismatch in score pickle.")

    quality_scores, quality_components = _compute_window_quality_scores(
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

    normalized_influence = _minmax_normalize(influence_scores)
    joint_scores = _combine_joint_scores(
        quality_scores,
        normalized_influence,
        joint_rule=args.joint_rule,
        joint_quality_power=args.joint_quality_power,
        joint_influence_power=args.joint_influence_power,
    )
    keep_positions = _retrain._select_keep_indices(joint_scores.astype(float), keep_ratio)  # noqa: SLF001
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
    )
    output_indices = args.output_indices or (output_dir / "kept_sample_indices.npy")
    if output_dir.exists() and args.overwrite:
        # preserve overwrite semantics but do not recursively delete arbitrary dirs
        pass
    output_dir.mkdir(parents=True, exist_ok=True)
    _retrain._save_sample_indices(output_indices, kept_sample_indices, overwrite=args.overwrite)  # noqa: SLF001

    summary = {
        "score_pkl": str(args.score_pkl),
        "dataset_dir": str(dataset_dir),
        "quality_source": args.quality_source,
        "num_windows": int(window_indices.shape[0]),
        "kept_windows": int(kept_window_indices.shape[0]),
        "kept_samples": int(kept_sample_indices.shape[0]),
        "keep_ratio": float(keep_ratio),
        "joint_rule": args.joint_rule,
        "joint_quality_power": float(args.joint_quality_power),
        "joint_influence_power": float(args.joint_influence_power),
        "joint_score_min": float(np.min(joint_scores)),
        "joint_score_max": float(np.max(joint_scores)),
        "quality_score_min": float(np.min(quality_scores)),
        "quality_score_max": float(np.max(quality_scores)),
        "normalized_influence_min": float(np.min(normalized_influence)),
        "normalized_influence_max": float(np.max(normalized_influence)),
    }

    _write_selection_artifacts(
        output_dir,
        window_indices=window_indices,
        influence_scores=influence_scores,
        quality_scores=quality_scores,
        normalized_influence=normalized_influence,
        joint_scores=joint_scores,
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
        "Window joint selection summary: "
        f"windows={window_indices.shape[0]}, kept_windows={kept_window_indices.shape[0]}, "
        f"kept_samples={kept_sample_indices.shape[0]}, "
        f"joint_score_range=[{summary['joint_score_min']:.6f}, {summary['joint_score_max']:.6f}]"
    )
    print(f"Output dir: {output_dir}")
    print(f"Sample indices: {output_indices}")

    print("Recomputing norm_stats.json for selected sample subset...")
    _recompute_norm_stats_for_sample_subset(
        train_cfg,
        sample_indices_path=output_indices,
        output_dir=output_dir,
        max_frames=args.max_frames_for_norm_stats,
    )
    print(f"Subset norm_stats written to: {output_dir / 'norm_stats.json'}")

    if args.prepare_only:
        return

    keep_pct = int(round(keep_ratio * 100))
    exp_name = args.exp_name or f"{default_exp_name}_window_joint_{args.joint_rule}_keep{keep_pct}_{args.quality_source}"
    retrain_cfg = _build_train_config(
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
