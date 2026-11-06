#!/usr/bin/env python3
"""可视化 joint episode-window influence 分数。"""

from __future__ import annotations

import argparse
import pathlib
import pickle
import sys
from typing import Dict, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _ensure_numpy_core_aliases() -> None:
    """兼容旧 pickle 中的 numpy 内部模块路径。"""
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


def _load_payload(path: pathlib.Path) -> Dict:
    _ensure_numpy_core_aliases()
    with path.open("rb") as f:
        return pickle.load(f)


def _resolve_episode_scores(payload: Dict) -> np.ndarray:
    metadata = payload.get("metadata") or {}
    if "episode_influence_scores" in metadata:
        return np.asarray(metadata["episode_influence_scores"], dtype=np.float64)
    if "influence_scores" in payload:
        return np.asarray(payload["influence_scores"], dtype=np.float64)
    raise ValueError("score.pkl 中缺少 episode 级 influence score。")


def _resolve_window_payload(payload: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    metadata = payload.get("metadata") or {}
    if "window_influence_scores" not in metadata:
        return (
            np.asarray([], dtype=np.float64),
            np.asarray([], dtype=np.int64),
            np.asarray([], dtype=np.int64),
        )
    return (
        np.asarray(metadata["window_influence_scores"], dtype=np.float64),
        np.asarray(metadata.get("window_episode_index"), dtype=np.int64),
        np.asarray(metadata.get("window_start"), dtype=np.int64),
    )


def _build_window_stats(
    episode_scores: np.ndarray,
    window_scores: np.ndarray,
    window_episode_index: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """按轨迹聚合 window 分数，返回均值/最小值/最大值/计数。"""
    num_episodes = episode_scores.shape[0]
    mean_scores = np.full(num_episodes, np.nan, dtype=np.float64)
    min_scores = np.full(num_episodes, np.nan, dtype=np.float64)
    max_scores = np.full(num_episodes, np.nan, dtype=np.float64)
    counts = np.zeros(num_episodes, dtype=np.int64)

    if window_scores.size == 0 or window_episode_index.size == 0:
        return mean_scores, min_scores, max_scores, counts

    for episode_idx in range(num_episodes):
        mask = window_episode_index == episode_idx
        if not np.any(mask):
            continue
        values = window_scores[mask]
        mean_scores[episode_idx] = float(np.mean(values))
        min_scores[episode_idx] = float(np.min(values))
        max_scores[episode_idx] = float(np.max(values))
        counts[episode_idx] = int(values.shape[0])
    return mean_scores, min_scores, max_scores, counts


def _add_zero_line(ax: plt.Axes) -> None:
    ax.axhline(0.0, color="black", linewidth=1.0, linestyle="--", alpha=0.6)


def plot_scores(score_pkl: pathlib.Path, output_path: pathlib.Path, title: str | None) -> pathlib.Path:
    payload = _load_payload(score_pkl)
    episode_scores = _resolve_episode_scores(payload)
    window_scores, window_episode_index, window_start = _resolve_window_payload(payload)
    mean_scores, min_scores, max_scores, counts = _build_window_stats(
        episode_scores, window_scores, window_episode_index
    )

    num_episodes = episode_scores.shape[0]
    episode_index = np.arange(num_episodes, dtype=np.int64)
    sorted_index = np.argsort(episode_scores)[::-1]

    fig, axes = plt.subplots(2, 2, figsize=(18, 10), constrained_layout=True)
    ax_episode, ax_sorted, ax_window_scatter, ax_window_stats = axes.ravel()

    colors = np.where(episode_scores >= 0.0, "#d55e00", "#0072b2")
    ax_episode.bar(episode_index, episode_scores, color=colors, width=0.9, alpha=0.85)
    _add_zero_line(ax_episode)
    ax_episode.set_title("Episode Influence Scores")
    ax_episode.set_xlabel("Episode Index")
    ax_episode.set_ylabel("Influence Score")

    ax_sorted.plot(np.arange(num_episodes), episode_scores[sorted_index], color="#1b9e77", linewidth=1.6)
    ax_sorted.scatter(
        np.arange(num_episodes),
        episode_scores[sorted_index],
        s=16,
        c=np.where(episode_scores[sorted_index] >= 0.0, "#d55e00", "#0072b2"),
    )
    _add_zero_line(ax_sorted)
    ax_sorted.set_title("Sorted Episode Scores")
    ax_sorted.set_xlabel("Rank")
    ax_sorted.set_ylabel("Influence Score")

    if window_scores.size > 0 and window_episode_index.size == window_scores.size:
        scatter = ax_window_scatter.scatter(
            window_episode_index,
            window_start,
            c=window_scores,
            s=30,
            cmap="coolwarm",
            alpha=0.9,
        )
        fig.colorbar(scatter, ax=ax_window_scatter, label="Window Score")
        ax_window_scatter.set_title("Window Scores by Episode")
        ax_window_scatter.set_xlabel("Episode Index")
        ax_window_scatter.set_ylabel("Window Start")
    else:
        ax_window_scatter.text(0.5, 0.5, "No window-level scores", ha="center", va="center")
        ax_window_scatter.set_axis_off()

    valid_mask = counts > 0
    if np.any(valid_mask):
        valid_index = episode_index[valid_mask]
        ax_window_stats.fill_between(
            valid_index,
            min_scores[valid_mask],
            max_scores[valid_mask],
            color="#56b4e9",
            alpha=0.25,
            label="Min/Max window score",
        )
        ax_window_stats.plot(valid_index, mean_scores[valid_mask], color="#009e73", linewidth=1.8, label="Mean window score")
        ax_window_stats.scatter(valid_index, episode_scores[valid_mask], s=18, color="#cc79a7", label="Episode score")
        _add_zero_line(ax_window_stats)
        ax_window_stats.set_title("Episode vs Window Aggregation")
        ax_window_stats.set_xlabel("Episode Index")
        ax_window_stats.set_ylabel("Score")
        ax_window_stats.legend()
    else:
        ax_window_stats.text(0.5, 0.5, "No window statistics", ha="center", va="center")
        ax_window_stats.set_axis_off()

    summary_lines = [
        f"episodes={num_episodes}",
        f"episode_score_min={episode_scores.min():.4f}",
        f"episode_score_max={episode_scores.max():.4f}",
        f"episode_score_mean={episode_scores.mean():.4f}",
        f"windows={window_scores.shape[0]}",
    ]
    fig.suptitle(title or score_pkl.stem, fontsize=16, fontweight="bold")
    fig.text(0.01, 0.01, " | ".join(summary_lines), fontsize=10)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="可视化轨迹 influence 分数图。")
    parser.add_argument("--score-pkl", type=pathlib.Path, required=True, help="score.pkl 路径")
    parser.add_argument("--output", type=pathlib.Path, default=None, help="输出图片路径，默认保存在 score.pkl 同目录")
    parser.add_argument("--title", type=str, default=None, help="图标题")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = args.output
    if output_path is None:
        output_path = args.score_pkl.with_suffix(".png")
    saved_path = plot_scores(args.score_pkl, output_path, args.title)
    print(saved_path)


if __name__ == "__main__":
    main()
