#!/usr/bin/env python3
"""统一对比 ACT 与 OpenPI05 数据筛选偏好的可视化脚本。"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    import pyarrow.parquet as pq

    HAS_PYARROW = True
except Exception:
    HAS_PYARROW = False

try:
    import umap

    HAS_UMAP = True
except Exception:
    HAS_UMAP = False

try:
    from sklearn.manifold import TSNE

    HAS_SKLEARN = True
except Exception:
    HAS_SKLEARN = False


@dataclass
class EpisodeSeries:
    """统一的 episode 级分数字段。"""

    indices: np.ndarray
    scores: np.ndarray


@dataclass
class WindowSeries:
    """统一的 window 级分数字段。"""

    episode_index: np.ndarray
    start: np.ndarray
    end: np.ndarray
    scores: np.ndarray


def _ensure_numpy_core_aliases() -> None:
    """兼容旧 pickle 里 numpy 内部模块名变化。"""

    core = getattr(np, "_core", None)
    if core is None:
        core = getattr(np, "core", None)
    if core is None:
        return
    sys.modules.setdefault("numpy._core", core)
    if hasattr(core, "numeric"):
        sys.modules.setdefault("numpy._core.numeric", core.numeric)
    if hasattr(core, "multiarray"):
        sys.modules.setdefault("numpy._core.multiarray", core.multiarray)


def _load_pickle(path: Path) -> Dict[str, Any]:
    _ensure_numpy_core_aliases()
    with path.open("rb") as f:
        return pickle.load(f)


def _metadata(payload: Dict[str, Any]) -> Dict[str, Any]:
    meta = payload.get("metadata")
    return meta if isinstance(meta, dict) else {}


def _to_array(value: Any, dtype: Any = np.float64) -> np.ndarray:
    return np.asarray(value, dtype=dtype).reshape(-1)


def _maybe_series(payload: Dict[str, Any], *keys: str, dtype: Any = np.float64) -> Optional[np.ndarray]:
    meta = _metadata(payload)
    for key in keys:
        if key in payload:
            return _to_array(payload[key], dtype=dtype)
        if key in meta:
            return _to_array(meta[key], dtype=dtype)
    return None


def _resolve_episode_series(payload: Dict[str, Any], label: str) -> EpisodeSeries:
    """优先取显式 episode 分数，避免误把 window/sample score 当 episode score。"""

    scores = _maybe_series(payload, "episode_influence_scores")
    if scores is None:
        scores = _maybe_series(payload, "influence_scores")
        if scores is None:
            raise ValueError(f"{label}: 无法在 score pickle 中解析 episode 分数。")
        window_scores = _maybe_series(payload, "window_influence_scores")
        if window_scores is not None and scores.shape[0] == window_scores.shape[0]:
            raise ValueError(
                f"{label}: `influence_scores` 长度与 window 分数相同，疑似不是 episode 级分数；"
                "请提供包含 `episode_influence_scores` 的产物。"
            )

    episode_indices = _maybe_series(payload, "train_episode_indices", dtype=np.int64)
    if episode_indices is None or episode_indices.shape[0] != scores.shape[0]:
        episode_indices = np.arange(scores.shape[0], dtype=np.int64)
    return EpisodeSeries(indices=episode_indices.astype(np.int64), scores=scores.astype(np.float64))


def _resolve_window_series(payload: Dict[str, Any]) -> Optional[WindowSeries]:
    scores = _maybe_series(payload, "window_influence_scores")
    if scores is None or scores.size == 0:
        return None

    meta = _metadata(payload)
    episode_index = (
        _maybe_series(payload, "window_episode_index", "train_window_episode_indices", dtype=np.int64)
        if "window_episode_index" in payload or "window_episode_index" in meta or "train_window_episode_indices" in meta
        else None
    )
    if episode_index is None:
        episode_index = _maybe_series(payload, "train_window_episode_indices", dtype=np.int64)

    start = _maybe_series(payload, "window_start", "train_window_frame_starts", dtype=np.int64)
    end = _maybe_series(payload, "window_end", "train_window_frame_ends", dtype=np.int64)

    if episode_index is None or start is None:
        return None
    if end is None:
        end = start + 1

    size = min(scores.shape[0], episode_index.shape[0], start.shape[0], end.shape[0])
    return WindowSeries(
        episode_index=episode_index[:size].astype(np.int64),
        start=start[:size].astype(np.int64),
        end=end[:size].astype(np.int64),
        scores=scores[:size].astype(np.float64),
    )


def _rank_desc(scores: np.ndarray) -> np.ndarray:
    order = np.argsort(-scores, kind="mergesort")
    ranks = np.empty_like(order)
    ranks[order] = np.arange(scores.shape[0], dtype=np.int64)
    return ranks


def _score_percentile(scores: np.ndarray) -> np.ndarray:
    n = scores.shape[0]
    if n <= 1:
        return np.ones_like(scores, dtype=np.float64)
    ranks = _rank_desc(scores).astype(np.float64)
    return 1.0 - ranks / float(n - 1)


def _minmax(values: np.ndarray) -> np.ndarray:
    vals = np.asarray(values, dtype=np.float64)
    if vals.size == 0:
        return vals
    lo = float(np.min(vals))
    hi = float(np.max(vals))
    if np.isclose(lo, hi):
        return np.ones_like(vals)
    return (vals - lo) / (hi - lo)


def _keep_mask(scores: np.ndarray, keep_ratio: float) -> np.ndarray:
    keep_ratio = float(np.clip(keep_ratio, 0.0, 1.0))
    n = scores.shape[0]
    if n == 0:
        return np.zeros((0,), dtype=bool)
    keep_n = int(math.ceil(n * keep_ratio))
    keep_n = max(0, min(n, keep_n))
    order = np.argsort(-scores, kind="mergesort")
    mask = np.zeros((n,), dtype=bool)
    mask[order[:keep_n]] = True
    return mask


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    if x.size == 0 or y.size == 0:
        return float("nan")
    x_std = float(np.std(x))
    y_std = float(np.std(y))
    if x_std < 1e-12 or y_std < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _spearman_from_scores(x: np.ndarray, y: np.ndarray) -> float:
    return _pearson(_rank_desc(x).astype(np.float64), _rank_desc(y).astype(np.float64))


def _jaccard_curve(act_scores: np.ndarray, pi_scores: np.ndarray, ks: Sequence[int]) -> List[Dict[str, float]]:
    act_order = np.argsort(-act_scores, kind="mergesort")
    pi_order = np.argsort(-pi_scores, kind="mergesort")
    result: List[Dict[str, float]] = []
    for k in ks:
        kk = int(max(1, min(k, act_scores.shape[0], pi_scores.shape[0])))
        act_top = set(act_order[:kk].tolist())
        pi_top = set(pi_order[:kk].tolist())
        inter = len(act_top.intersection(pi_top))
        union = len(act_top.union(pi_top))
        result.append({"k": kk, "jaccard": float(inter / union if union else 1.0), "overlap": inter})
    return result


def _load_quality_csv(path: Path) -> Dict[int, Dict[str, float]]:
    quality: Dict[int, Dict[str, float]] = {}
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"质量文件缺少表头: {path}")
        if "episode" not in reader.fieldnames:
            raise ValueError(f"{path} 需要包含 `episode` 列。")
        if "quality_score" not in reader.fieldnames and "quality_raw" not in reader.fieldnames:
            raise ValueError(f"{path} 至少需要包含 `quality_score` 或 `quality_raw` 列。")
        for row in reader:
            try:
                episode = int(row["episode"])
            except Exception:
                continue
            record: Dict[str, float] = {}
            for key in ("quality_score", "quality_raw", "rollout_count", "rank_desc", "drop"):
                value = row.get(key)
                if value is None or value == "":
                    continue
                try:
                    record[key] = float(value)
                except Exception:
                    continue
            if "quality_score" in record:
                record["quality_score_csv"] = record["quality_score"]
            quality[episode] = record
    return quality


def _normalize_quality_for_visualization(values: np.ndarray) -> np.ndarray:
    vals = np.asarray(values, dtype=np.float64)
    out = np.full_like(vals, np.nan, dtype=np.float64)
    finite = np.isfinite(vals)
    if not np.any(finite):
        return out
    v = vals[finite]
    lo = float(np.min(v))
    hi = float(np.max(v))
    if hi <= lo:
        out[finite] = 1.0
        return out
    out[finite] = (v - lo) / (hi - lo)
    return np.clip(out, 0.0, 1.0)


def _resolve_visual_quality_signal(
    quality_records: Dict[int, Dict[str, float]],
) -> Tuple[Dict[int, float], str]:
    if not quality_records:
        return {}, "unavailable"

    episodes = sorted(quality_records)

    score_values = np.asarray(
        [quality_records[ep].get("quality_score", float("nan")) for ep in episodes],
        dtype=np.float64,
    )
    finite_score = score_values[np.isfinite(score_values)]
    if finite_score.size > 0 and float(np.max(finite_score) - np.min(finite_score)) > 1e-12:
        return {
            ep: float(score_values[idx])
            for idx, ep in enumerate(episodes)
            if np.isfinite(score_values[idx])
        }, "quality_score_csv"

    raw_values = np.asarray(
        [quality_records[ep].get("quality_raw", float("nan")) for ep in episodes],
        dtype=np.float64,
    )
    finite_raw = raw_values[np.isfinite(raw_values)]
    if finite_raw.size > 0 and float(np.max(finite_raw) - np.min(finite_raw)) > 0.0:
        normalized = _normalize_quality_for_visualization(raw_values)
        return {
            ep: float(normalized[idx])
            for idx, ep in enumerate(episodes)
            if np.isfinite(normalized[idx])
        }, "quality_raw_minmax_fallback"

    if finite_score.size > 0:
        return {
            ep: float(score_values[idx])
            for idx, ep in enumerate(episodes)
            if np.isfinite(score_values[idx])
        }, "quality_score_constant"

    return {}, "unavailable"


def _episode_files(dataset_root: Path) -> Dict[int, Path]:
    files: Dict[int, Path] = {}
    for p in sorted((dataset_root / "data").glob("chunk-*/episode_*.parquet")):
        stem = p.stem
        if not stem.startswith("episode_"):
            continue
        suffix = stem.split("_")[-1]
        if suffix.isdigit():
            files[int(suffix)] = p
    return files


def _episode_stats(dataset_root: Optional[Path]) -> Dict[int, Dict[str, float]]:
    if dataset_root is None:
        return {}
    if not HAS_PYARROW:
        print("警告: 未安装 pyarrow，跳过 episode 轨迹统计。", file=sys.stderr)
        return {}

    stats: Dict[int, Dict[str, float]] = {}
    files = _episode_files(dataset_root)
    for ep, path in files.items():
        schema = pq.read_schema(path)
        available = set(schema.names)
        cols = [c for c in ("action", "observation.state") if c in available]
        if not cols:
            continue
        table = pq.read_table(path, columns=cols)
        length = int(table.num_rows)
        record: Dict[str, float] = {"length": float(length)}

        if "action" in table.column_names:
            action = np.asarray(table["action"].to_pylist(), dtype=np.float64)
            if action.ndim == 2 and action.shape[0] > 0:
                record["action_energy"] = float(np.linalg.norm(action, axis=1).mean())
                if action.shape[0] >= 2:
                    record["action_jitter"] = float(np.linalg.norm(np.diff(action, axis=0), axis=1).mean())
                else:
                    record["action_jitter"] = 0.0

        if "observation.state" in table.column_names:
            state = np.asarray(table["observation.state"].to_pylist(), dtype=np.float64)
            if state.ndim == 2 and state.shape[0] > 0:
                record["state_energy"] = float(np.linalg.norm(state, axis=1).mean())
                record["terminal_state_norm"] = float(np.linalg.norm(state[-1]))
                record["state_displacement"] = float(np.linalg.norm(state[-1] - state[0]))
                if state.shape[0] >= 2:
                    record["state_jitter"] = float(np.linalg.norm(np.diff(state, axis=0), axis=1).mean())
                else:
                    record["state_jitter"] = 0.0
        stats[int(ep)] = record
    return stats


def _fit_2d(features: np.ndarray, seed: int) -> np.ndarray:
    if features.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float64)
    if features.shape[0] == 1:
        return np.zeros((1, 2), dtype=np.float64)
    if HAS_UMAP and features.shape[0] >= 4:
        reducer = umap.UMAP(
            n_components=2,
            random_state=seed,
            min_dist=0.15,
            n_neighbors=min(30, max(5, features.shape[0] // 10)),
        )
        return reducer.fit_transform(features)
    if HAS_SKLEARN and features.shape[0] >= 4:
        perplexity = min(30, max(3, (features.shape[0] - 1) // 3))
        return TSNE(
            n_components=2,
            random_state=seed,
            init="random",
            perplexity=perplexity,
            learning_rate="auto",
        ).fit_transform(features)

    # 用 PCA 作为最后兜底，避免额外依赖导致脚本不可用。
    centered = features - np.mean(features, axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    basis = vt[:2].T
    if basis.shape[1] < 2:
        basis = np.pad(basis, ((0, 0), (0, 2 - basis.shape[1])))
    return centered @ basis[:, :2]


def _cohort_label(act_keep: bool, pi_keep: bool) -> str:
    if act_keep and pi_keep:
        return "both_keep"
    if act_keep and not pi_keep:
        return "act_only"
    if (not act_keep) and pi_keep:
        return "pi05_only"
    return "both_drop"


def _write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _plot_episode_score_curves(
    out_path: Path,
    act_scores: np.ndarray,
    pi_scores: np.ndarray,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(_minmax(np.sort(act_scores)[::-1]), label="ACT", linewidth=2.0, color="#d55e00")
    ax.plot(_minmax(np.sort(pi_scores)[::-1]), label="OpenPI05", linewidth=2.0, color="#0072b2")
    ax.set_xlabel("Rank")
    ax.set_ylabel("MinMax-Normalized Score")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_rank_scatter(out_path: Path, episode_rows: List[Dict[str, Any]]) -> None:
    cohorts = {
        "both_keep": "#1b9e77",
        "act_only": "#d95f02",
        "pi05_only": "#7570b3",
        "both_drop": "#bdbdbd",
    }
    fig, ax = plt.subplots(figsize=(6.5, 6))
    for cohort, color in cohorts.items():
        xs = [float(r["act_rank_desc"]) for r in episode_rows if r["cohort"] == cohort]
        ys = [float(r["pi05_rank_desc"]) for r in episode_rows if r["cohort"] == cohort]
        if not xs:
            continue
        ax.scatter(xs, ys, s=26, alpha=0.8, label=cohort, c=color)
    lim = max(
        max((int(r["act_rank_desc"]) for r in episode_rows), default=0),
        max((int(r["pi05_rank_desc"]) for r in episode_rows), default=0),
    )
    ax.plot([0, lim], [0, lim], linestyle="--", linewidth=1.2, color="black", alpha=0.6)
    ax.set_xlabel("ACT Rank (0=best)")
    ax.set_ylabel("OpenPI05 Rank (0=best)")
    ax.set_title("Episode Rank Agreement")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_jaccard_curve(out_path: Path, curve: List[Dict[str, float]]) -> None:
    ks = [int(item["k"]) for item in curve]
    vals = [float(item["jaccard"]) for item in curve]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(ks, vals, marker="o", linewidth=2.0, color="#009e73")
    ax.set_xlabel("Top-k")
    ax.set_ylabel("Jaccard")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Top-k Overlap Between ACT and OpenPI05")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_keep_overlap(out_path: Path, episode_rows: List[Dict[str, Any]]) -> None:
    order = ["both_keep", "act_only", "pi05_only", "both_drop"]
    counts = [sum(1 for row in episode_rows if row["cohort"] == key) for key in order]
    colors = ["#1b9e77", "#d95f02", "#7570b3", "#bdbdbd"]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(order, counts, color=colors, alpha=0.9)
    ax.set_ylabel("Episodes")
    ax.set_title("Keep/Drop Cohort Counts")
    for idx, value in enumerate(counts):
        ax.text(idx, value + max(1, int(0.01 * max(counts or [1]))), str(value), ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_window_delta_heatmap(out_path: Path, window_rows: List[Dict[str, Any]]) -> None:
    if not window_rows:
        return
    xs = np.asarray([int(r["episode"]) for r in window_rows], dtype=np.int64)
    ys = np.asarray([int(r["start"]) for r in window_rows], dtype=np.int64)
    cs = np.asarray([float(r["delta_percentile"]) for r in window_rows], dtype=np.float64)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    scatter = ax.scatter(xs, ys, c=cs, cmap="coolwarm", s=20, alpha=0.9, vmin=-1.0, vmax=1.0)
    fig.colorbar(scatter, ax=ax, label="ACT Percentile - OpenPI05 Percentile")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Window Start")
    ax.set_title("Window-Level Preference Delta")
    ax.grid(alpha=0.15)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _build_window_rows(
    act_window: Optional[WindowSeries],
    pi_window: Optional[WindowSeries],
) -> Tuple[List[Dict[str, Any]], str]:
    """优先严格匹配，失败后回退到 episode+start 匹配。"""

    if act_window is None or pi_window is None:
        return [], "unavailable"

    act_window_map_strict = {
        (int(ep), int(start), int(end)): float(score)
        for ep, start, end, score in zip(
            act_window.episode_index, act_window.start, act_window.end, act_window.scores, strict=True
        )
    }
    pi_window_map_strict = {
        (int(ep), int(start), int(end)): float(score)
        for ep, start, end, score in zip(
            pi_window.episode_index, pi_window.start, pi_window.end, pi_window.scores, strict=True
        )
    }
    common_windows = sorted(set(act_window_map_strict).intersection(pi_window_map_strict))
    match_mode = "episode_start_end"

    if not common_windows:
        act_window_map_relaxed = {
            (int(ep), int(start)): float(score)
            for ep, start, score in zip(
                act_window.episode_index, act_window.start, act_window.scores, strict=True
            )
        }
        pi_window_map_relaxed = {
            (int(ep), int(start)): float(score)
            for ep, start, score in zip(
                pi_window.episode_index, pi_window.start, pi_window.scores, strict=True
            )
        }
        relaxed_keys = sorted(set(act_window_map_relaxed).intersection(pi_window_map_relaxed))
        if not relaxed_keys:
            return [], "none"

        act_scores = np.asarray([act_window_map_relaxed[key] for key in relaxed_keys], dtype=np.float64)
        pi_scores = np.asarray([pi_window_map_relaxed[key] for key in relaxed_keys], dtype=np.float64)
        act_pct = _score_percentile(act_scores)
        pi_pct = _score_percentile(pi_scores)
        rows = []
        for idx, (ep, start) in enumerate(relaxed_keys):
            rows.append(
                {
                    "episode": ep,
                    "start": start,
                    "end": "",
                    "act_score_raw": float(act_scores[idx]),
                    "pi05_score_raw": float(pi_scores[idx]),
                    "act_score_percentile": float(act_pct[idx]),
                    "pi05_score_percentile": float(pi_pct[idx]),
                    "delta_percentile": float(act_pct[idx] - pi_pct[idx]),
                }
            )
        return rows, "episode_start"

    act_scores = np.asarray([act_window_map_strict[key] for key in common_windows], dtype=np.float64)
    pi_scores = np.asarray([pi_window_map_strict[key] for key in common_windows], dtype=np.float64)
    act_pct = _score_percentile(act_scores)
    pi_pct = _score_percentile(pi_scores)
    rows = []
    for idx, (ep, start, end) in enumerate(common_windows):
        rows.append(
            {
                "episode": ep,
                "start": start,
                "end": end,
                "act_score_raw": float(act_scores[idx]),
                "pi05_score_raw": float(pi_scores[idx]),
                "act_score_percentile": float(act_pct[idx]),
                "pi05_score_percentile": float(pi_pct[idx]),
                "delta_percentile": float(act_pct[idx] - pi_pct[idx]),
            }
        )
    return rows, match_mode


def _plot_embedding(
    out_path: Path,
    coords: np.ndarray,
    episode_rows: List[Dict[str, Any]],
) -> None:
    if coords.shape[0] == 0:
        return

    cohorts = np.asarray([row["cohort"] for row in episode_rows], dtype=object)
    act_pct = np.asarray([float(row["act_score_percentile"]) for row in episode_rows], dtype=np.float64)
    pi_pct = np.asarray([float(row["pi05_score_percentile"]) for row in episode_rows], dtype=np.float64)
    delta = np.asarray([float(row["delta_percentile"]) for row in episode_rows], dtype=np.float64)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    panels = [
        ("ACT Percentile", act_pct, "viridis"),
        ("OpenPI05 Percentile", pi_pct, "viridis"),
        ("Percentile Delta", delta, "coolwarm"),
    ]
    for ax, (title, values, cmap) in zip(axes.ravel()[:3], panels, strict=True):
        sc = ax.scatter(coords[:, 0], coords[:, 1], c=values, cmap=cmap, s=28, alpha=0.9)
        fig.colorbar(sc, ax=ax)
        ax.set_title(title)
        ax.set_xlabel("Dim-1")
        ax.set_ylabel("Dim-2")

    cohort_colors = {
        "both_keep": "#1b9e77",
        "act_only": "#d95f02",
        "pi05_only": "#7570b3",
        "both_drop": "#bdbdbd",
    }
    ax = axes.ravel()[3]
    for cohort, color in cohort_colors.items():
        mask = cohorts == cohort
        if not np.any(mask):
            continue
        ax.scatter(coords[mask, 0], coords[mask, 1], s=28, alpha=0.9, label=cohort, c=color)
    ax.set_title("Cohorts")
    ax.set_xlabel("Dim-1")
    ax.set_ylabel("Dim-2")
    ax.legend()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="统一可视化 ACT 与 OpenPI05 的数据筛选偏好。")
    parser.add_argument("--act-score-pkl", type=Path, required=True, help="ACT score.pkl 路径")
    parser.add_argument("--pi05-score-pkl", type=Path, required=True, help="OpenPI05 score.pkl 路径")
    parser.add_argument("--dataset-root", type=Path, default=None, help="可选，LeRobot 数据集根目录，用于补充轨迹统计")
    parser.add_argument("--quality-csv", type=Path, default=None, help="可选，episode 级 quality_episodes.csv")
    parser.add_argument("--output-dir", type=Path, required=True, help="输出目录")
    parser.add_argument("--act-keep-ratio", type=float, default=0.8, help="ACT 视为保留的 top 比例")
    parser.add_argument("--pi05-keep-ratio", type=float, default=0.8, help="OpenPI05 视为保留的 top 比例")
    parser.add_argument("--seed", type=int, default=0, help="随机种子，仅用于 2D embedding")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    act_payload = _load_pickle(args.act_score_pkl)
    pi_payload = _load_pickle(args.pi05_score_pkl)

    act_episode = _resolve_episode_series(act_payload, "ACT")
    pi_episode = _resolve_episode_series(pi_payload, "OpenPI05")
    act_window = _resolve_window_series(act_payload)
    pi_window = _resolve_window_series(pi_payload)

    act_map = {int(ep): float(score) for ep, score in zip(act_episode.indices, act_episode.scores, strict=True)}
    pi_map = {int(ep): float(score) for ep, score in zip(pi_episode.indices, pi_episode.scores, strict=True)}
    common_eps = np.asarray(sorted(set(act_map).intersection(pi_map)), dtype=np.int64)
    if common_eps.size == 0:
        raise ValueError("ACT 与 OpenPI05 没有重叠的 episode index，无法比较。")

    act_scores = np.asarray([act_map[int(ep)] for ep in common_eps], dtype=np.float64)
    pi_scores = np.asarray([pi_map[int(ep)] for ep in common_eps], dtype=np.float64)
    act_pct = _score_percentile(act_scores)
    pi_pct = _score_percentile(pi_scores)
    act_rank = _rank_desc(act_scores)
    pi_rank = _rank_desc(pi_scores)
    act_keep = _keep_mask(act_scores, args.act_keep_ratio)
    pi_keep = _keep_mask(pi_scores, args.pi05_keep_ratio)

    quality_records = _load_quality_csv(args.quality_csv) if args.quality_csv is not None else {}
    quality_map, quality_signal_source = _resolve_visual_quality_signal(quality_records)
    stat_map = _episode_stats(args.dataset_root)

    episode_rows: List[Dict[str, Any]] = []
    for idx, ep in enumerate(common_eps.tolist()):
        row: Dict[str, Any] = {
            "episode": int(ep),
            "act_score_raw": float(act_scores[idx]),
            "pi05_score_raw": float(pi_scores[idx]),
            "act_score_percentile": float(act_pct[idx]),
            "pi05_score_percentile": float(pi_pct[idx]),
            "delta_percentile": float(act_pct[idx] - pi_pct[idx]),
            "act_rank_desc": int(act_rank[idx]),
            "pi05_rank_desc": int(pi_rank[idx]),
            "act_keep": int(bool(act_keep[idx])),
            "pi05_keep": int(bool(pi_keep[idx])),
        }
        row["cohort"] = _cohort_label(bool(act_keep[idx]), bool(pi_keep[idx]))
        if ep in quality_records:
            row.update(quality_records[ep])
        if ep in quality_map:
            row["quality_score"] = float(quality_map[ep])
        if ep in stat_map:
            row.update(stat_map[ep])
        episode_rows.append(row)

    episode_fieldnames = sorted({key for row in episode_rows for key in row.keys()})
    _write_csv(args.output_dir / "episode_comparison.csv", episode_rows, episode_fieldnames)

    window_rows, window_match_mode = _build_window_rows(act_window, pi_window)
    if window_rows:
        window_fieldnames = sorted({key for row in window_rows for key in row.keys()})
        _write_csv(args.output_dir / "window_comparison.csv", window_rows, window_fieldnames)

    # 计算 embedding 特征。只用连续统计量，避免类别字段扰动。
    feature_keys = [
        "quality_score",
        "length",
        "action_jitter",
        "state_jitter",
        "action_energy",
        "state_energy",
        "state_displacement",
        "terminal_state_norm",
    ]
    valid_rows = [row for row in episode_rows if any(key in row for key in feature_keys)]
    coords = np.zeros((0, 2), dtype=np.float64)
    if valid_rows:
        features: List[List[float]] = []
        valid_episode_rows: List[Dict[str, Any]] = []
        for row in episode_rows:
            vals = [float(row.get(key, float("nan"))) for key in feature_keys]
            if np.all(~np.isfinite(np.asarray(vals, dtype=np.float64))):
                continue
            features.append(vals)
            valid_episode_rows.append(row)
        if features:
            feature_arr = np.asarray(features, dtype=np.float64)
            col_means = np.nanmean(feature_arr, axis=0)
            # 在缺少 pyarrow 等依赖时，部分统计列可能整列缺失，这里退化为 0 防止嵌入阶段出现 NaN。
            col_means = np.where(np.isfinite(col_means), col_means, 0.0)
            inds = np.where(~np.isfinite(feature_arr))
            feature_arr[inds] = np.take(col_means, inds[1])
            col_std = np.nanstd(feature_arr, axis=0)
            col_std = np.where(col_std > 1e-8, col_std, 1.0)
            feature_arr = (feature_arr - np.mean(feature_arr, axis=0, keepdims=True)) / col_std
            coords = _fit_2d(feature_arr, seed=args.seed)
            _plot_embedding(args.output_dir / "episode_feature_embedding.png", coords, valid_episode_rows)

    ks = sorted(set([5, 10, 20, 30, 50, 80, 100, 150, 200, common_eps.shape[0]]))
    ks = [k for k in ks if 1 <= k <= common_eps.shape[0]]
    jaccard_curve = _jaccard_curve(act_scores, pi_scores, ks)

    summary = {
        "num_common_episodes": int(common_eps.shape[0]),
        "num_common_windows": int(len(window_rows)),
        "window_match_mode": window_match_mode,
        "act_keep_ratio": float(args.act_keep_ratio),
        "pi05_keep_ratio": float(args.pi05_keep_ratio),
        "quality_signal_source": quality_signal_source,
        "pearson_raw_score": _pearson(act_scores, pi_scores),
        "spearman_score": _spearman_from_scores(act_scores, pi_scores),
        "cohort_counts": {
            cohort: int(sum(1 for row in episode_rows if row["cohort"] == cohort))
            for cohort in ("both_keep", "act_only", "pi05_only", "both_drop")
        },
        "topk_jaccard": jaccard_curve,
    }

    # 生成 cohort 统计摘要，帮助解释两个模型筛选偏好差异。
    cohort_stats: Dict[str, Dict[str, float]] = {}
    stat_keys = [
        "act_score_raw",
        "pi05_score_raw",
        "act_score_percentile",
        "pi05_score_percentile",
        "delta_percentile",
        "quality_score",
        "length",
        "action_jitter",
        "state_jitter",
        "state_displacement",
        "terminal_state_norm",
    ]
    for cohort in ("both_keep", "act_only", "pi05_only", "both_drop"):
        rows = [row for row in episode_rows if row["cohort"] == cohort]
        stats: Dict[str, float] = {"count": float(len(rows))}
        for key in stat_keys:
            vals = np.asarray([float(row[key]) for row in rows if key in row], dtype=np.float64)
            if vals.size == 0:
                continue
            stats[f"{key}_mean"] = float(np.mean(vals))
            stats[f"{key}_median"] = float(np.median(vals))
        cohort_stats[cohort] = stats
    summary["cohort_stats"] = cohort_stats

    (args.output_dir / "comparison_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    _plot_episode_score_curves(
        args.output_dir / "episode_score_curves.png",
        act_scores,
        pi_scores,
        title="ACT vs OpenPI05 Episode Score Curves",
    )
    _plot_rank_scatter(args.output_dir / "episode_rank_scatter.png", episode_rows)
    _plot_jaccard_curve(args.output_dir / "topk_jaccard.png", jaccard_curve)
    _plot_keep_overlap(args.output_dir / "keep_overlap_bar.png", episode_rows)
    _plot_window_delta_heatmap(args.output_dir / "window_delta_heatmap.png", window_rows)

    print(f"Saved outputs to: {args.output_dir}")
    print(f"Common episodes: {common_eps.shape[0]}")
    print(f"Common windows: {len(window_rows)}")


if __name__ == "__main__":
    main()
