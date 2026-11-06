#!/usr/bin/env python
"""Standalone CUPID quality evaluation (no training, no prior CSV).

Directly reads lerobot dataset parquet trajectories to compute quality scores.

Quality source notes:
- `episode_eval`: smoothness-centric. Uses action/state jitter, length consistency,
  and optional action-energy penalty. Good for checking kinematic regularity.
- `trajectory_eval`: trajectory-structure-centric. Uses terminal-state consistency
  (kNN terminal distance), path efficiency, progress (state displacement),
  relative smoothness, and length consistency. Better for task-level trajectory quality.

Outputs:
- quality_report.json
- quality_consistency_report.json
- quality_episodes.csv
- quality_drop.txt / quality_keep.txt
- quality_top_50.txt / quality_bottom_50.txt

Example:
 python /workspace/liushuo/cupid/openpi/myscripts/eval_cupid_quality.py \
    --dataset.root /workspace/liushuo/cupid/datasets/lerobot/can_mh \
    --quality-source trajectory_eval \
    --drop-quantile 0.2 \
    --quality-eval-normalizer quantile \
    --quality-eval-quantile-low 0.05 \
    --quality-eval-quantile-high 0.95 \
    --traj-eval-terminal-weight 0.25 \
    --traj-eval-efficiency-weight 0.2 \
    --traj-eval-progress-weight 0.25 \
    --traj-eval-state-smooth-weight 0.15 \
    --traj-eval-action-smooth-weight 0.1 \
    --traj-eval-length-weight 0.05 \
    --traj-eval-terminal-knn-k 15 \
    --output-dir /workspace/liushuo/cupid/3_model/quality_eval/can_mh_traj_eval_q20_noprior
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def _collect_episodes(dataset_root: Path) -> list[int]:
    episodes: list[int] = []
    for p in sorted((dataset_root / "data").glob("chunk-*/episode_*.parquet")):
        ep_str = p.stem.split("_")[-1]
        if ep_str.isdigit():
            episodes.append(int(ep_str))
    return sorted(set(episodes))


def _collect_episode_files(dataset_root: Path) -> Dict[int, Path]:
    episode_files: Dict[int, Path] = {}
    for p in sorted((dataset_root / "data").glob("chunk-*/episode_*.parquet")):
        stem = p.stem
        if not stem.startswith("episode_"):
            continue
        ep_str = stem.split("_")[-1]
        if ep_str.isdigit():
            episode_files[int(ep_str)] = p
    return episode_files


# ---------------------------------------------------------------------------
# Score normalization helpers
# ---------------------------------------------------------------------------

def _lower_better_to_score(
    values: np.ndarray,
    normalizer: str = "quantile",
    quantile_low: float = 0.05,
    quantile_high: float = 0.95,
) -> np.ndarray:
    vals = np.asarray(values, dtype=float)
    out = np.full_like(vals, np.nan, dtype=float)
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
    vals = np.asarray(values, dtype=float)
    out = np.full_like(vals, np.nan, dtype=float)
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


def _normalize_states(values: np.ndarray, mode: str) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    if x.ndim != 2 or x.shape[0] == 0:
        return x
    if mode == "none":
        return x
    if mode == "zscore":
        mu = np.mean(x, axis=0, keepdims=True)
        sd = np.std(x, axis=0, keepdims=True)
        sd = np.where(sd > 1e-8, sd, 1.0)
        return (x - mu) / sd
    # robust_zscore: median + MAD scaling, less sensitive to outliers.
    med = np.median(x, axis=0, keepdims=True)
    mad = np.median(np.abs(x - med), axis=0, keepdims=True)
    scale = np.where(mad > 1e-8, 1.4826 * mad, 1.0)
    return (x - med) / scale


def _pairwise_distance(values: np.ndarray, metric: str) -> np.ndarray:
    x = np.asarray(values, dtype=float)
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
        r = float(np.clip(trim_ratio, 0.0, 0.49))
        cut = int(np.floor(k * r))
        if cut <= 0:
            return nearest.mean(axis=1)
        if cut * 2 >= k:
            return nearest.mean(axis=1)
        sorted_nearest = np.sort(nearest, axis=1)
        return sorted_nearest[:, cut:k - cut].mean(axis=1)
    return nearest.mean(axis=1)


# ---------------------------------------------------------------------------
# Quality computation: episode_eval (kinematic smoothness)
# ---------------------------------------------------------------------------

def _quality_from_episode_eval(
    dataset_root: Path,
    all_eps: list[int],
    action_weight: float,
    state_weight: float,
    length_weight: float,
    energy_weight: float,
    length_scale: float,
    default_value: float,
    jitter_mode: str,
    normalizer: str,
    quantile_low: float,
    quantile_high: float,
) -> np.ndarray:
    try:
        import pyarrow.parquet as pq
    except Exception as e:
        raise RuntimeError("episode_eval requires pyarrow to read parquet episodes") from e

    num_episodes = max(all_eps) + 1 if all_eps else 0
    action_jitter = np.full(num_episodes, np.nan, dtype=float)
    state_jitter = np.full(num_episodes, np.nan, dtype=float)
    action_energy = np.full(num_episodes, np.nan, dtype=float)
    state_energy = np.full(num_episodes, np.nan, dtype=float)
    episode_len = np.full(num_episodes, np.nan, dtype=float)

    episode_files = _collect_episode_files(dataset_root)

    for ep in all_eps:
        p = episode_files.get(int(ep))
        if p is None:
            continue
        schema = pq.read_schema(p)
        available = set(schema.names)
        cols = [c for c in ("action", "observation.state") if c in available]
        table = pq.read_table(p, columns=cols if cols else None)
        n = int(table.num_rows)
        episode_len[ep] = float(n)
        if n < 2:
            continue

        if "action" in table.column_names:
            a = np.asarray(table["action"].to_pylist(), dtype=float)
            if a.ndim == 2 and a.shape[0] >= 2:
                action_jitter[ep] = float(np.linalg.norm(np.diff(a, axis=0), axis=1).mean())
                action_energy[ep] = float(np.linalg.norm(a, axis=1).mean())

        if "observation.state" in table.column_names:
            s = np.asarray(table["observation.state"].to_pylist(), dtype=float)
            if s.ndim == 2 and s.shape[0] >= 2:
                state_jitter[ep] = float(np.linalg.norm(np.diff(s, axis=0), axis=1).mean())
                state_energy[ep] = float(np.linalg.norm(s, axis=1).mean())

    if jitter_mode == "relative":
        action_metric = action_jitter / np.maximum(action_energy, 1e-8)
        state_metric = state_jitter / np.maximum(state_energy, 1e-8)
    else:
        action_metric = action_jitter
        state_metric = state_jitter

    action_score = _lower_better_to_score(action_metric, normalizer=normalizer,
                                          quantile_low=quantile_low, quantile_high=quantile_high)
    state_score = _lower_better_to_score(state_metric, normalizer=normalizer,
                                         quantile_low=quantile_low, quantile_high=quantile_high)
    energy_score = _lower_better_to_score(action_energy, normalizer=normalizer,
                                          quantile_low=quantile_low, quantile_high=quantile_high)

    length_score = np.full(num_episodes, np.nan, dtype=float)
    finite_len = np.isfinite(episode_len)
    if finite_len.any():
        lens = episode_len[finite_len]
        med = float(np.median(lens))
        mad = float(np.median(np.abs(lens - med)))
        denom = max(mad * max(length_scale, 1e-6), 1e-6)
        length_score[finite_len] = np.exp(-np.abs(lens - med) / denom)
        length_score = np.clip(length_score, 0.0, 1.0)

    comp_pairs = [
        (action_score, max(action_weight, 0.0)),
        (state_score, max(state_weight, 0.0)),
        (length_score, max(length_weight, 0.0)),
        (energy_score, max(energy_weight, 0.0)),
    ]
    quality = np.full(num_episodes, np.nan, dtype=float)
    for ep in all_eps:
        num = 0.0
        den = 0.0
        for comp, w in comp_pairs:
            if w <= 0.0 or ep >= len(comp):
                continue
            val = comp[ep]
            if np.isfinite(val):
                num += w * float(val)
                den += w
        if den > 0:
            quality[ep] = num / den

    missing = ~np.isfinite(quality)
    if missing.any():
        quality[missing] = float(default_value)
    return np.clip(quality, 0.0, None)


# ---------------------------------------------------------------------------
# Quality computation: trajectory_eval (task-level trajectory structure)
# ---------------------------------------------------------------------------

def _quality_from_trajectory_eval(
    dataset_root: Path,
    all_eps: list[int],
    terminal_weight: float,
    efficiency_weight: float,
    progress_weight: float,
    state_smooth_weight: float,
    action_smooth_weight: float,
    length_weight: float,
    terminal_knn_k: int,
    monotonic_weight: float,
    backtrack_weight: float,
    stall_weight: float,
    fusion: str,
    min_terminal_score: float,
    min_progress_score: float,
    gate_quality_cap: float,
    length_scale: float,
    default_value: float,
    normalizer: str,
    quantile_low: float,
    quantile_high: float,
    terminal_distance_metric: str,
    terminal_normalize: str,
    terminal_agg: str,
    terminal_trim_ratio: float,
) -> np.ndarray:
    try:
        import pyarrow.parquet as pq
    except Exception as e:
        raise RuntimeError("trajectory_eval requires pyarrow to read parquet episodes") from e

    num_episodes = max(all_eps) + 1 if all_eps else 0
    terminal_dist = np.full(num_episodes, np.nan, dtype=float)
    state_efficiency = np.full(num_episodes, np.nan, dtype=float)
    state_progress = np.full(num_episodes, np.nan, dtype=float)
    state_rel_jitter = np.full(num_episodes, np.nan, dtype=float)
    state_monotonic = np.full(num_episodes, np.nan, dtype=float)
    state_backtrack = np.full(num_episodes, np.nan, dtype=float)
    state_stall = np.full(num_episodes, np.nan, dtype=float)
    action_rel_jitter = np.full(num_episodes, np.nan, dtype=float)
    episode_len = np.full(num_episodes, np.nan, dtype=float)
    final_state_map: Dict[int, np.ndarray] = {}

    episode_files = _collect_episode_files(dataset_root)

    for ep in all_eps:
        p = episode_files.get(int(ep))
        if p is None:
            continue
        schema = pq.read_schema(p)
        available = set(schema.names)
        cols = [c for c in ("action", "observation.state") if c in available]
        table = pq.read_table(p, columns=cols if cols else None)
        n = int(table.num_rows)
        episode_len[ep] = float(n)
        if n < 2:
            continue

        if "observation.state" in table.column_names:
            s = np.asarray(table["observation.state"].to_pylist(), dtype=float)
            if s.ndim == 2 and s.shape[0] >= 2:
                ds = np.diff(s, axis=0)
                ds_norm = np.linalg.norm(ds, axis=1)
                path = float(ds_norm.sum())
                disp = float(np.linalg.norm(s[-1] - s[0]))
                mag = float(np.linalg.norm(s, axis=1).mean())
                state_efficiency[ep] = disp / (path + 1e-8)
                state_progress[ep] = disp
                state_rel_jitter[ep] = float(ds_norm.mean() / (mag + 1e-8))
                if disp > 1e-8:
                    direction = (s[-1] - s[0]) / (disp + 1e-8)
                    proj = ds @ direction
                    forward = float(np.sum(np.clip(proj, 0.0, None)))
                    backward = float(np.sum(np.clip(-proj, 0.0, None)))
                    total_dir = forward + backward
                    if total_dir > 1e-8:
                        state_monotonic[ep] = forward / total_dir
                        state_backtrack[ep] = backward / total_dir
                    else:
                        state_monotonic[ep] = 0.0
                        state_backtrack[ep] = 1.0
                else:
                    state_monotonic[ep] = 0.0
                    state_backtrack[ep] = 1.0

                stall_thr = max(0.1 * disp / max(int(s.shape[0]) - 1, 1), 1e-6)
                state_stall[ep] = float(np.mean(ds_norm <= stall_thr))
                final_state_map[ep] = s[-1]

        if "action" in table.column_names:
            a = np.asarray(table["action"].to_pylist(), dtype=float)
            if a.ndim == 2 and a.shape[0] >= 2:
                da = np.diff(a, axis=0)
                mag = float(np.linalg.norm(a, axis=1).mean())
                action_rel_jitter[ep] = float(np.linalg.norm(da, axis=1).mean() / (mag + 1e-8))

    # kNN terminal distance (supports multi-modal terminal states)
    if final_state_map:
        valid_eps = sorted(final_state_map.keys())
        final_states = np.stack([final_state_map[ep] for ep in valid_eps], axis=0)
        final_states = _normalize_states(final_states, terminal_normalize)
        if len(valid_eps) == 1:
            terminal_dist[valid_eps[0]] = 0.0
        else:
            k = int(max(1, min(int(terminal_knn_k), len(valid_eps) - 1)))
            dist_mat = _pairwise_distance(final_states, metric=terminal_distance_metric)
            np.fill_diagonal(dist_mat, np.inf)
            nearest = np.partition(dist_mat, kth=k - 1, axis=1)[:, :k]
            d = _aggregate_knn_distance(nearest, agg=terminal_agg, trim_ratio=terminal_trim_ratio)
            for ep, dist in zip(valid_eps, d):
                terminal_dist[ep] = float(dist)

    terminal_score = _lower_better_to_score(terminal_dist, normalizer=normalizer,
                                            quantile_low=quantile_low, quantile_high=quantile_high)
    efficiency_score = _higher_better_to_score(state_efficiency, normalizer=normalizer,
                                               quantile_low=quantile_low, quantile_high=quantile_high)
    progress_score = _higher_better_to_score(state_progress, normalizer=normalizer,
                                             quantile_low=quantile_low, quantile_high=quantile_high)
    monotonic_score = _higher_better_to_score(state_monotonic, normalizer=normalizer,
                                              quantile_low=quantile_low, quantile_high=quantile_high)
    backtrack_score = _lower_better_to_score(state_backtrack, normalizer=normalizer,
                                             quantile_low=quantile_low, quantile_high=quantile_high)
    stall_score = _lower_better_to_score(state_stall, normalizer=normalizer,
                                         quantile_low=quantile_low, quantile_high=quantile_high)
    state_smooth_score = _lower_better_to_score(state_rel_jitter, normalizer=normalizer,
                                                quantile_low=quantile_low, quantile_high=quantile_high)
    action_smooth_score = _lower_better_to_score(action_rel_jitter, normalizer=normalizer,
                                                 quantile_low=quantile_low, quantile_high=quantile_high)

    length_score = np.full(num_episodes, np.nan, dtype=float)
    finite_len = np.isfinite(episode_len)
    if finite_len.any():
        lens = episode_len[finite_len]
        med = float(np.median(lens))
        mad = float(np.median(np.abs(lens - med)))
        denom = max(mad * max(length_scale, 1e-6), 1e-6)
        length_score[finite_len] = np.exp(-np.abs(lens - med) / denom)
        length_score = np.clip(length_score, 0.0, 1.0)

    comp_pairs = [
        (terminal_score,      max(terminal_weight, 0.0)),
        (efficiency_score,    max(efficiency_weight, 0.0)),
        (progress_score,      max(progress_weight, 0.0)),
        (monotonic_score,     max(monotonic_weight, 0.0)),
        (backtrack_score,     max(backtrack_weight, 0.0)),
        (stall_score,         max(stall_weight, 0.0)),
        (state_smooth_score,  max(state_smooth_weight, 0.0)),
        (action_smooth_score, max(action_smooth_weight, 0.0)),
        (length_score,        max(length_weight, 0.0)),
    ]
    quality = np.full(num_episodes, np.nan, dtype=float)
    for ep in all_eps:
        den = 0.0
        if fusion == "weighted_geomean":
            num_log = 0.0
            for comp, w in comp_pairs:
                if w <= 0.0 or ep >= len(comp):
                    continue
                val = comp[ep]
                if np.isfinite(val):
                    num_log += w * float(np.log(max(float(val), 1e-8)))
                    den += w
            if den > 0:
                quality[ep] = float(np.exp(num_log / den))
        else:
            num = 0.0
            for comp, w in comp_pairs:
                if w <= 0.0 or ep >= len(comp):
                    continue
                val = comp[ep]
                if np.isfinite(val):
                    num += w * float(val)
                    den += w
            if den > 0:
                quality[ep] = num / den

    # Gate: episodes with too-low terminal or progress score receive a quality cap
    if gate_quality_cap >= 0.0:
        gate_mask = np.zeros(num_episodes, dtype=bool)
        if min_terminal_score >= 0.0:
            gate_mask |= np.isfinite(terminal_score) & (terminal_score < float(min_terminal_score))
        if min_progress_score >= 0.0:
            gate_mask |= np.isfinite(progress_score) & (progress_score < float(min_progress_score))
        quality[gate_mask] = np.minimum(quality[gate_mask], float(gate_quality_cap))

    missing = ~np.isfinite(quality)
    if missing.any():
        quality[missing] = float(default_value)
    return np.clip(quality, 0.0, None)


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

def _summarize(values: np.ndarray) -> Dict[str, float]:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return {
            "count": 0,
            "min": float("nan"),
            "max": float("nan"),
            "mean": float("nan"),
            "std": float("nan"),
            "p05": float("nan"),
            "p10": float("nan"),
            "p20": float("nan"),
            "p50": float("nan"),
            "p80": float("nan"),
            "p90": float("nan"),
            "p95": float("nan"),
        }
    q = np.quantile(vals, [0.05, 0.1, 0.2, 0.5, 0.8, 0.9, 0.95])
    return {
        "count": int(vals.size),
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
        "mean": float(np.mean(vals)),
        "std": float(np.std(vals)),
        "p05": float(q[0]),
        "p10": float(q[1]),
        "p20": float(q[2]),
        "p50": float(q[3]),
        "p80": float(q[4]),
        "p90": float(q[5]),
        "p95": float(q[6]),
    }


def _to_percentile_score(values: np.ndarray) -> np.ndarray:
    vals = np.asarray(values, dtype=float)
    out = np.full_like(vals, np.nan, dtype=float)
    finite = np.isfinite(vals)
    idx = np.where(finite)[0]
    if idx.size == 0:
        return out

    v = vals[idx]
    n = int(v.size)
    if n == 1:
        out[idx] = 1.0
        return out

    order = np.argsort(v, kind="mergesort")
    sorted_v = v[order]
    ranks = np.empty(n, dtype=float)
    i = 0
    while i < n:
        j = i + 1
        while j < n and sorted_v[j] == sorted_v[i]:
            j += 1
        # Average rank for tied values.
        avg_rank = 0.5 * (i + j - 1)
        ranks[order[i:j]] = avg_rank
        i = j
    out[idx] = ranks / max(n - 1, 1)
    return out


def _spearman_corr(a: np.ndarray, b: np.ndarray) -> float:
    ra = _to_percentile_score(a)
    rb = _to_percentile_score(b)
    finite = np.isfinite(ra) & np.isfinite(rb)
    if finite.sum() < 2:
        return float("nan")
    x = ra[finite]
    y = rb[finite]
    sx = float(np.std(x))
    sy = float(np.std(y))
    if sx <= 1e-12 or sy <= 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _load_terminal_states(dataset_root: Path, all_eps: list[int]) -> tuple[list[int], np.ndarray]:
    try:
        import pyarrow.parquet as pq
    except Exception as e:
        raise RuntimeError("consistency report requires pyarrow to read parquet episodes") from e

    episode_files = _collect_episode_files(dataset_root)
    eps: list[int] = []
    states: list[np.ndarray] = []
    for ep in all_eps:
        p = episode_files.get(int(ep))
        if p is None:
            continue
        schema = pq.read_schema(p)
        if "observation.state" not in set(schema.names):
            continue
        table = pq.read_table(p, columns=["observation.state"])
        if int(table.num_rows) <= 0:
            continue
        s = np.asarray(table["observation.state"].to_pylist(), dtype=float)
        if s.ndim != 2 or s.shape[0] == 0:
            continue
        eps.append(ep)
        states.append(s[-1])
    if not states:
        return [], np.empty((0, 0), dtype=float)
    return eps, np.stack(states, axis=0)


def _cluster_by_radius(states: np.ndarray, radius: float) -> np.ndarray:
    n = int(states.shape[0])
    if n == 0:
        return np.empty((0,), dtype=int)
    if n == 1:
        return np.zeros((1,), dtype=int)

    dist_mat = np.linalg.norm(states[:, None, :] - states[None, :, :], axis=2)
    labels = np.full(n, -1, dtype=int)
    unassigned = np.ones(n, dtype=bool)
    cluster_id = 0

    while np.any(unassigned):
        candidates = np.where(unassigned)[0]
        if candidates.size == 1:
            center = int(candidates[0])
        else:
            counts = np.array(
                [np.sum(unassigned & (dist_mat[i] <= radius)) for i in candidates],
                dtype=int,
            )
            center = int(candidates[int(np.argmax(counts))])
        members = unassigned & (dist_mat[center] <= radius)
        labels[members] = cluster_id
        unassigned[members] = False
        cluster_id += 1

    return labels


def _score_pileup(values: np.ndarray, decimals: int = 6, top_k: int = 10) -> list[dict[str, float]]:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return []
    rounded = np.round(vals, decimals=decimals)
    uniq, counts = np.unique(rounded, return_counts=True)
    order = np.argsort(counts)[::-1]
    out: list[dict[str, float]] = []
    for idx in order[:top_k]:
        out.append(
            {
                "score": float(uniq[idx]),
                "count": int(counts[idx]),
                "ratio": float(counts[idx] / vals.size),
            }
        )
    return out


def _compute_consistency_report(
    dataset_root: Path,
    all_eps: list[int],
    quality: np.ndarray,
    drop_set: set[int],
    component_scores: Dict[str, np.ndarray] | None,
    drop_quantile: float,
    drop_threshold: float | None,
    bootstrap_rounds: int,
    bootstrap_seed: int,
    block_size: int,
    terminal_radius_quantile: float,
) -> Dict[str, Any]:
    eps = [ep for ep in all_eps if ep < len(quality)]
    q_vals = np.asarray([quality[ep] for ep in eps], dtype=float)

    corr: Dict[str, float] = {}
    if component_scores:
        for key, comp in component_scores.items():
            c_vals = np.asarray([comp[ep] for ep in eps if ep < len(comp)], dtype=float)
            if c_vals.size == q_vals.size:
                corr[f"spearman_fused_vs_{key}"] = _spearman_corr(q_vals, c_vals)

    base_drop = set(int(ep) for ep in drop_set)
    jaccards: list[float] = []
    rounds = max(int(bootstrap_rounds), 0)
    if rounds > 0 and eps:
        rng = np.random.default_rng(int(bootstrap_seed))
        eps_arr = np.asarray(eps, dtype=int)
        for _ in range(rounds):
            sample_idx = rng.integers(0, len(eps_arr), size=len(eps_arr))
            sampled_eps = eps_arr[sample_idx]
            sampled_vals = np.asarray([quality[int(ep)] for ep in sampled_eps], dtype=float)
            sample_drop: set[int] = set()
            if drop_quantile > 0:
                thr = float(np.quantile(sampled_vals, drop_quantile))
                sample_drop |= {int(ep) for ep in sampled_eps if quality[int(ep)] <= thr}
            if drop_threshold is not None:
                sample_drop |= {int(ep) for ep in sampled_eps if quality[int(ep)] <= float(drop_threshold)}
            union = base_drop | sample_drop
            if union:
                jaccards.append(len(base_drop & sample_drop) / len(union))
            else:
                jaccards.append(1.0)

    block_rows: list[dict[str, float]] = []
    if block_size <= 0:
        block_size = 50
    for start in range(min(all_eps), max(all_eps) + 1, int(block_size)):
        end = start + int(block_size) - 1
        block_eps = [ep for ep in eps if start <= ep <= end]
        if not block_eps:
            continue
        arr = np.asarray([quality[ep] for ep in block_eps], dtype=float)
        drop_n = sum(1 for ep in block_eps if ep in base_drop)
        block_rows.append(
            {
                "start": int(start),
                "end": int(end),
                "count": int(len(block_eps)),
                "drop_count": int(drop_n),
                "drop_rate": float(drop_n / max(len(block_eps), 1)),
                "quality_mean": float(np.mean(arr)),
                "quality_median": float(np.median(arr)),
            }
        )

    terminal_mode: Dict[str, Any] = {
        "num_modes": 0,
        "coverage_keep": float("nan"),
        "coverage_drop": float("nan"),
        "radius": float("nan"),
        "mode_sizes_top10": [],
    }
    try:
        term_eps, term_states = _load_terminal_states(dataset_root, all_eps)
        if len(term_eps) >= 1:
            if len(term_eps) == 1:
                labels = np.zeros((1,), dtype=int)
                radius = 0.0
            else:
                dist_mat = np.linalg.norm(term_states[:, None, :] - term_states[None, :, :], axis=2)
                np.fill_diagonal(dist_mat, np.inf)
                nearest = np.min(dist_mat, axis=1)
                q = float(np.clip(terminal_radius_quantile, 0.0, 1.0))
                radius = float(np.quantile(nearest, q))
                radius = max(radius, 1e-8)
                labels = _cluster_by_radius(term_states, radius)
            mode_sizes = np.bincount(labels)
            ep_to_mode = {int(ep): int(mode) for ep, mode in zip(term_eps, labels)}
            keep_modes = {ep_to_mode[ep] for ep in term_eps if ep not in base_drop}
            drop_modes = {ep_to_mode[ep] for ep in term_eps if ep in base_drop}
            n_modes = int(len(mode_sizes))
            terminal_mode = {
                "num_modes": n_modes,
                "coverage_keep": float(len(keep_modes) / max(n_modes, 1)),
                "coverage_drop": float(len(drop_modes) / max(n_modes, 1)),
                "radius": float(radius),
                "mode_sizes_top10": [int(x) for x in sorted(mode_sizes.tolist(), reverse=True)[:10]],
            }
    except Exception as e:
        terminal_mode = {
            "error": str(e),
            "num_modes": 0,
            "coverage_keep": float("nan"),
            "coverage_drop": float("nan"),
            "radius": float("nan"),
            "mode_sizes_top10": [],
        }

    drop_rates = [row["drop_rate"] for row in block_rows]
    bootstrap_stats = {
        "rounds": int(rounds),
        "mean_jaccard": float(np.mean(jaccards)) if jaccards else float("nan"),
        "std_jaccard": float(np.std(jaccards)) if jaccards else float("nan"),
        "p10_jaccard": float(np.quantile(jaccards, 0.1)) if jaccards else float("nan"),
        "p50_jaccard": float(np.quantile(jaccards, 0.5)) if jaccards else float("nan"),
        "p90_jaccard": float(np.quantile(jaccards, 0.9)) if jaccards else float("nan"),
    }

    return {
        "score_distribution": _summarize(q_vals),
        "component_correlation": corr,
        "bootstrap_drop_stability": bootstrap_stats,
        "drop_bias_by_episode_block": {
            "block_size": int(block_size),
            "drop_rate_std": float(np.std(drop_rates)) if drop_rates else float("nan"),
            "drop_rate_min": float(np.min(drop_rates)) if drop_rates else float("nan"),
            "drop_rate_max": float(np.max(drop_rates)) if drop_rates else float("nan"),
            "blocks": block_rows,
        },
        "terminal_mode_coverage": terminal_mode,
        "score_pileup_top10": _score_pileup(q_vals, decimals=6, top_k=10),
    }


def _compute_episode_quality(args: argparse.Namespace, dataset_root: Path, all_eps: list[int]) -> np.ndarray:
    return _quality_from_episode_eval(
        dataset_root=dataset_root,
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


def _compute_trajectory_quality(args: argparse.Namespace, dataset_root: Path, all_eps: list[int]) -> np.ndarray:
    return _quality_from_trajectory_eval(
        dataset_root=dataset_root,
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


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Standalone quality evaluation for CUPID data (no prior CSV required)."
    )
    parser.add_argument("--dataset-path", type=Path, default=None)
    parser.add_argument("--dataset.root", dest="dataset_root", type=Path, default=None)
    parser.add_argument(
        "--quality-source",
        type=str,
        default="trajectory_eval",
        choices=("episode_eval", "trajectory_eval", "rank_fusion"),
        help=(
            "episode_eval=kinematic smoothness; "
            "trajectory_eval=task-trajectory structure/progress; "
            "rank_fusion=percentile-rank fusion of episode_eval and trajectory_eval."
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--keep-ratio",
        type=float,
        default=None,
        help=(
            "Keep top-K episodes by quality only, where K=floor(total*keep_ratio). "
            "When set, this overrides drop-quantile/drop-threshold."
        ),
    )
    parser.add_argument("--drop-quantile", type=float, default=0.2,
                        help="Drop episodes below this quantile of quality score (0 to disable).")
    parser.add_argument("--drop-threshold", type=float, default=None,
                        help="Additionally drop episodes with quality <= this absolute threshold.")
    # episode_eval weights
    parser.add_argument("--quality-eval-action-weight", type=float, default=0.5)
    parser.add_argument("--quality-eval-state-weight", type=float, default=0.3)
    parser.add_argument("--quality-eval-length-weight", type=float, default=0.2)
    parser.add_argument("--quality-eval-energy-weight", type=float, default=0.0)
    parser.add_argument("--quality-eval-length-scale", type=float, default=1.0)
    parser.add_argument("--quality-eval-default", type=float, default=0.5)
    parser.add_argument("--quality-eval-jitter-mode", type=str, default="absolute",
                        choices=("relative", "absolute"))
    parser.add_argument("--quality-eval-normalizer", type=str, default="quantile",
                        choices=("quantile", "minmax"))
    parser.add_argument("--quality-eval-quantile-low", type=float, default=0.05)
    parser.add_argument("--quality-eval-quantile-high", type=float, default=0.95)
    # trajectory_eval weights
    parser.add_argument("--traj-eval-terminal-weight", type=float, default=0.25)
    parser.add_argument("--traj-eval-efficiency-weight", type=float, default=0.2)
    parser.add_argument("--traj-eval-progress-weight", type=float, default=0.25)
    parser.add_argument("--traj-eval-state-smooth-weight", type=float, default=0.15)
    parser.add_argument("--traj-eval-action-smooth-weight", type=float, default=0.1)
    parser.add_argument("--traj-eval-length-weight", type=float, default=0.05)
    parser.add_argument("--traj-eval-terminal-knn-k", type=int, default=15)
    parser.add_argument(
        "--traj-eval-terminal-distance-metric",
        type=str,
        default="l2",
        choices=("l2", "l1", "cosine"),
        help="Distance metric for terminal-state kNN consistency.",
    )
    parser.add_argument(
        "--traj-eval-terminal-normalize",
        type=str,
        default="robust_zscore",
        choices=("none", "zscore", "robust_zscore"),
        help="Per-dimension normalization before terminal-state distance computation.",
    )
    parser.add_argument(
        "--traj-eval-terminal-agg",
        type=str,
        default="trimmed_mean",
        choices=("mean", "median", "trimmed_mean"),
        help="How to aggregate each sample's kNN distances into terminal distance.",
    )
    parser.add_argument(
        "--traj-eval-terminal-trim-ratio",
        type=float,
        default=0.2,
        help="Trim ratio used when traj_eval_terminal_agg=trimmed_mean.",
    )
    parser.add_argument("--traj-eval-monotonic-weight", type=float, default=0.2)
    parser.add_argument("--traj-eval-backtrack-weight", type=float, default=0.1)
    parser.add_argument("--traj-eval-stall-weight", type=float, default=0.1)
    parser.add_argument("--traj-eval-fusion", type=str, default="weighted_geomean",
                        choices=("weighted_mean", "weighted_geomean"))
    parser.add_argument("--traj-eval-min-terminal-score", type=float, default=0.35)
    parser.add_argument("--traj-eval-min-progress-score", type=float, default=0.45)
    parser.add_argument("--traj-eval-gate-quality-cap", type=float, default=0.15)
    # rank_fusion settings
    parser.add_argument("--rank-fusion-episode-weight", type=float, default=0.3)
    parser.add_argument("--rank-fusion-trajectory-weight", type=float, default=0.7)
    # unsupervised consistency report settings
    parser.add_argument("--consistency-bootstrap-rounds", type=int, default=200)
    parser.add_argument("--consistency-bootstrap-seed", type=int, default=42)
    parser.add_argument("--consistency-episode-block-size", type=int, default=50)
    parser.add_argument("--consistency-terminal-radius-quantile", type=float, default=0.3)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    if args.keep_ratio is not None and not (0.0 < float(args.keep_ratio) <= 1.0):
        raise ValueError("--keep-ratio must be in (0, 1].")

    dataset_root = args.dataset_root or args.dataset_path
    if dataset_root is None:
        raise ValueError("Provide --dataset-path or --dataset.root.")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_eps = _collect_episodes(dataset_root)
    if not all_eps:
        raise RuntimeError(f"No episode parquet files found under {dataset_root / 'data'}")

    print(f"[quality] dataset={dataset_root}  episodes={len(all_eps)}")

    # ------------------------------------------------------------------
    # Compute quality scores
    # ------------------------------------------------------------------
    component_scores: Dict[str, np.ndarray] = {}
    if args.quality_source == "episode_eval":
        quality = _compute_episode_quality(args, dataset_root, all_eps)
        component_scores["episode_eval"] = quality.copy()
    elif args.quality_source == "trajectory_eval":
        quality = _compute_trajectory_quality(args, dataset_root, all_eps)
        component_scores["trajectory_eval"] = quality.copy()
    elif args.quality_source == "rank_fusion":
        episode_quality = _compute_episode_quality(args, dataset_root, all_eps)
        trajectory_quality = _compute_trajectory_quality(args, dataset_root, all_eps)
        ep_pct = _to_percentile_score(episode_quality)
        traj_pct = _to_percentile_score(trajectory_quality)

        w_ep = max(float(args.rank_fusion_episode_weight), 0.0)
        w_traj = max(float(args.rank_fusion_trajectory_weight), 0.0)
        w_sum = w_ep + w_traj
        if w_sum <= 1e-12:
            raise ValueError("rank_fusion requires positive fusion weights.")
        w_ep /= w_sum
        w_traj /= w_sum

        num_episodes = max(all_eps) + 1 if all_eps else 0
        quality = np.full(num_episodes, np.nan, dtype=float)
        for ep in all_eps:
            v_ep = ep_pct[ep] if ep < len(ep_pct) else np.nan
            v_traj = traj_pct[ep] if ep < len(traj_pct) else np.nan
            if np.isfinite(v_ep) and np.isfinite(v_traj):
                quality[ep] = w_ep * float(v_ep) + w_traj * float(v_traj)
            elif np.isfinite(v_ep):
                quality[ep] = float(v_ep)
            elif np.isfinite(v_traj):
                quality[ep] = float(v_traj)
            else:
                quality[ep] = float(args.quality_eval_default)
        quality = np.clip(quality, 0.0, None)
        component_scores["episode_eval"] = episode_quality
        component_scores["trajectory_eval"] = trajectory_quality
        component_scores["episode_rank_pct"] = ep_pct
        component_scores["trajectory_rank_pct"] = traj_pct
    else:
        raise ValueError(f"Unsupported quality source: {args.quality_source}")

    # ------------------------------------------------------------------
    # Drop / keep split
    # ------------------------------------------------------------------
    sorted_desc = sorted(
        [(ep, float(quality[ep])) for ep in all_eps if ep < len(quality)],
        key=lambda x: (-x[1], x[0]),
    )
    quality_vals = np.asarray([score for _, score in sorted_desc], dtype=float)
    threshold_quantile = None
    drop_set: set[int] = set()
    effective_drop_quantile = float(args.drop_quantile)
    effective_drop_threshold = args.drop_threshold

    if args.keep_ratio is not None:
        keep_count = int(np.floor(len(sorted_desc) * float(args.keep_ratio)))
        if keep_count == 0 and len(sorted_desc) > 0:
            keep_count = 1
        keep_set = {ep for ep, _ in sorted_desc[:keep_count]}
        drop_set = {ep for ep, _ in sorted_desc if ep not in keep_set}
        keep_eps = [ep for ep in all_eps if ep in keep_set]
        effective_drop_quantile = max(0.0, 1.0 - float(args.keep_ratio))
        effective_drop_threshold = None
    else:
        if args.drop_quantile > 0:
            threshold_quantile = float(np.quantile(quality_vals, args.drop_quantile))
            drop_set |= {ep for ep in all_eps if ep < len(quality) and quality[ep] <= threshold_quantile}
        if args.drop_threshold is not None:
            drop_set |= {ep for ep in all_eps if ep < len(quality) and quality[ep] <= float(args.drop_threshold)}
        keep_eps = [ep for ep in all_eps if ep not in drop_set]

    rank_map = {ep: idx + 1 for idx, (ep, _) in enumerate(sorted_desc)}

    # ------------------------------------------------------------------
    # Write outputs
    # ------------------------------------------------------------------
    report = {
        "dataset_root": str(dataset_root),
        "quality_source": args.quality_source,
        "total_episodes": len(all_eps),
        "quality_stats": _summarize(quality_vals),
        "selection_mode": "top_keep_ratio" if args.keep_ratio is not None else "drop_threshold",
        "keep_ratio": float(args.keep_ratio) if args.keep_ratio is not None else None,
        "drop_quantile": args.drop_quantile,
        "drop_threshold": args.drop_threshold,
        "quantile_threshold_value": threshold_quantile,
        "dropped_episodes": len(drop_set),
        "kept_episodes": len(keep_eps),
    }
    if args.quality_source == "rank_fusion":
        w_ep = max(float(args.rank_fusion_episode_weight), 0.0)
        w_traj = max(float(args.rank_fusion_trajectory_weight), 0.0)
        w_sum = max(w_ep + w_traj, 1e-12)
        report["rank_fusion"] = {
            "episode_weight": float(w_ep / w_sum),
            "trajectory_weight": float(w_traj / w_sum),
            "method": "weighted_percentile_rank",
        }

    (args.output_dir / "quality_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    (args.output_dir / "quality_drop.txt").write_text("\n".join(str(ep) for ep in sorted(drop_set)))
    (args.output_dir / "quality_keep.txt").write_text("\n".join(str(ep) for ep in keep_eps))
    (args.output_dir / "quality_top_50.txt").write_text(
        "\n".join(f"{ep}\t{score:.6g}" for ep, score in sorted_desc[:50])
    )
    (args.output_dir / "quality_bottom_50.txt").write_text(
        "\n".join(f"{ep}\t{score:.6g}" for ep, score in sorted_desc[::-1][:50])
    )

    with (args.output_dir / "quality_episodes.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "quality_score", "rank_desc", "drop"])
        for ep in all_eps:
            q = float(quality[ep]) if ep < len(quality) else float("nan")
            writer.writerow([ep, q, rank_map.get(ep, ""), int(ep in drop_set)])

    consistency = _compute_consistency_report(
        dataset_root=dataset_root,
        all_eps=all_eps,
        quality=quality,
        drop_set=drop_set,
        component_scores=component_scores,
        drop_quantile=float(effective_drop_quantile),
        drop_threshold=effective_drop_threshold,
        bootstrap_rounds=int(args.consistency_bootstrap_rounds),
        bootstrap_seed=int(args.consistency_bootstrap_seed),
        block_size=int(args.consistency_episode_block_size),
        terminal_radius_quantile=float(args.consistency_terminal_radius_quantile),
    )
    (args.output_dir / "quality_consistency_report.json").write_text(
        json.dumps(consistency, indent=2, ensure_ascii=False)
    )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"[quality] source={args.quality_source}")
    print(f"[quality] episodes={len(all_eps)}  dropped={len(drop_set)}  kept={len(keep_eps)}")
    if args.keep_ratio is not None:
        print(f"[quality] keep_ratio={float(args.keep_ratio):.6g} (top-K by quality only)")
    if threshold_quantile is not None:
        print(f"[quality] quantile_threshold={threshold_quantile:.6g} (q={args.drop_quantile})")
    print(f"[quality] outputs -> {args.output_dir}")


if __name__ == "__main__":
    main()
