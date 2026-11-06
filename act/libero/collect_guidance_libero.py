#!/usr/bin/env python3
"""Plan additional Libero data collection using bottle start positions and influence scores.

This script mirrors the plotting/influence utilities used for the bottle task, but adapts
the parsing to Libero's lerobot-format dataset:
- Bottle start poses are read from ``init_positions.txt`` (the ``bbq_sauce`` lines).
- Influence scores come from a pickle/npz/npy file (per-episode or per-sample).
- The script bins influence over a grid, highlights existing points, and proposes new
  collection locations.
  指导数据采集
"""

from __future__ import annotations

import argparse
import ast
import json
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter


@dataclass
class Position:
    episode: int
    x: float
    y: float
    norm_x: float | None = None
    norm_y: float | None = None


@dataclass
class GridCellRecommendation:
    row: int
    col: int
    score: float
    center_x: float
    center_y: float
    sampled_x: float
    sampled_y: float


def parse_libero_init_positions(path: Path, target_object: str = "bbq_sauce") -> List[Position]:
    """Extract (x, y) positions for a single object from init_positions.txt."""

    pattern = re.compile(
        rf"^\d+:\s*{re.escape(target_object)}(?:_\d+)?\s*->\s*(\[[^\]]+\])", flags=re.IGNORECASE
    )
    positions: List[Position] = []

    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            match = pattern.match(line)
            if not match:
                continue
            coords = ast.literal_eval(match.group(1))
            if len(coords) < 2:
                continue
            positions.append(Position(episode=len(positions), x=float(coords[0]), y=float(coords[1])))

    if not positions:
        raise RuntimeError(f"No positions for '{target_object}' found in {path}")
    return positions


def normalize(positions: Iterable[Position]) -> tuple[float, float, float, float]:
    xs = [p.x for p in positions]
    ys = [p.y for p in positions]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    span_x = max(max_x - min_x, 1e-9)
    span_y = max(max_y - min_y, 1e-9)

    for p in positions:
        p.norm_x = (p.x - min_x) / span_x
        p.norm_y = (p.y - min_y) / span_y

    return min_x, max_x, min_y, max_y


def _read_episode_lengths(train_root: Path) -> List[int]:
    """Parse HF-style episodes.jsonl to recover per-episode lengths."""

    meta_path = train_root / "meta" / "episodes.jsonl"
    lengths: List[int] = []
    with meta_path.open("r") as f:
        for line in f:
            record = json.loads(line)
            lengths.append(int(record["length"]))
    return lengths


def _episode_indices_from_lengths(lengths: Sequence[int]) -> np.ndarray:
    """Map sample index -> episode id given per-episode lengths."""

    ids: List[int] = []
    for ep_id, length in enumerate(lengths):
        ids.extend([ep_id] * int(length))
    return np.asarray(ids, dtype=np.int64)


def _aggregate_samples_to_episodes(sample_scores: np.ndarray, lengths: Sequence[int]) -> np.ndarray:
    """Average per-sample scores into per-episode scores using lengths."""

    episode_ids = _episode_indices_from_lengths(lengths)
    num_eps = int(episode_ids.max()) + 1
    accum = np.zeros(num_eps, dtype=np.float64)
    counts = np.zeros(num_eps, dtype=np.int64)
    np.add.at(accum, episode_ids, sample_scores)
    np.add.at(counts, episode_ids, 1)
    counts = np.maximum(counts, 1)
    return accum / counts


def load_influence_array(
    path: Path,
    *,
    expected_episodes: int | None = None,
    train_dataset_path: Path | None = None,
) -> np.ndarray:
    """
    Load influence scores (per-episode or per-sample) from npy/npz/pkl.

    If scores are per-sample and a train_dataset_path is provided, they will be
    aggregated to per-episode using meta/episodes.jsonl lengths.
    """

    if not path.exists():
        raise FileNotFoundError(f"Influence file not found: {path}")

    array: np.ndarray | None = None
    if path.suffix in {".npy", ".npz"}:
        loaded = np.load(path, allow_pickle=False)
        array = loaded["arr_0"] if isinstance(loaded, np.lib.npyio.NpzFile) else loaded
    elif path.suffix == ".pkl":
        with path.open("rb") as f:
            payload = pickle.load(f)
        for key in ("influence_scores", "scores", "scores_per_episode"):
            if isinstance(payload, dict) and key in payload:
                array = payload[key]
                break
        if array is None and isinstance(payload, np.ndarray):
            array = payload
        if array is None:
            raise RuntimeError(f"Could not find influence array in {path}")
    else:
        raise ValueError(f"Unsupported influence file type: {path.suffix}")

    scores = np.asarray(array, dtype=np.float64).reshape(-1)

    if train_dataset_path and scores.ndim == 1:
        lengths = _read_episode_lengths(train_dataset_path)
        num_eps = len(lengths)
        total_samples = sum(lengths)
        if scores.shape[0] == total_samples:
            scores = _aggregate_samples_to_episodes(scores, lengths)
        elif scores.shape[0] != num_eps:
            print(
                f"Warning: influence array length {scores.shape[0]} does not match "
                f"episodes ({num_eps}) or samples ({total_samples}); keeping raw shape."
            )

    if expected_episodes is not None and scores.shape[0] != expected_episodes:
        print(
            f"Note: influence array length {scores.shape[0]} does not equal number of positions {expected_episodes}."
        )
    return scores


def bin_influence_by_grid(
    positions: List[Position],
    influences: np.ndarray,
    cols: int,
    rows: int,
    x_range: tuple[float, float] | None = None,
    y_range: tuple[float, float] | None = None,
    x_edges: np.ndarray | None = None,
    y_edges: np.ndarray | None = None,
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    """Compute average influence per grid cell over (x, y) bins."""

    if x_edges is not None and y_edges is not None:
        x_edges = np.asarray(x_edges, dtype=np.float64)
        y_edges = np.asarray(y_edges, dtype=np.float64)
        cols = len(x_edges) - 1
        rows = len(y_edges) - 1
    elif x_range is not None and y_range is not None:
        x_min, x_max = x_range
        y_min, y_max = y_range
        x_edges = np.linspace(x_min, x_max, cols + 1)
        y_edges = np.linspace(y_min, y_max, rows + 1)
    else:
        raise ValueError("Either (x_range, y_range) or (x_edges, y_edges) must be provided.")

    grid_sum = np.zeros((rows, cols), dtype=np.float64)
    grid_count = np.zeros((rows, cols), dtype=np.int64)

    for p in positions:
        if p.episode >= influences.shape[0]:
            continue
        if not (x_edges[0] <= p.x <= x_edges[-1] and y_edges[0] <= p.y <= y_edges[-1]):
            continue
        cx = np.searchsorted(x_edges, p.x, side="right") - 1
        cy = np.searchsorted(y_edges, p.y, side="right") - 1
        cx = int(np.clip(cx, 0, cols - 1))
        cy = int(np.clip(cy, 0, rows - 1))
        grid_sum[cy, cx] += float(influences[p.episode])
        grid_count[cy, cx] += 1

    with np.errstate(invalid="ignore", divide="ignore"):
        grid_mean = grid_sum / np.maximum(grid_count, 1e-9)
        grid_mean[grid_count == 0] = np.nan

    return grid_mean, (x_edges, y_edges)


def recommend_cells_from_influence_grid(
    grid_mean: np.ndarray,
    grid_edges: tuple[np.ndarray, np.ndarray],
    *,
    threshold: float | None = None,
    budget: int | None = None,
    noise_std: float = 0.002,
    seed: int = 42,
    allowed_x: tuple[float, float] | None = None,
    allowed_y: tuple[float, float] | None = None,
    max_only: bool = False,
    max_tol: float = 1e-12,
    max_rel: float | None = None,
    fill_budget_with_noise: bool = True,
) -> list[GridCellRecommendation]:
    """Select grid cells for additional collection based on influence scores."""

    if grid_mean.size == 0:
        return []

    x_edges, y_edges = grid_edges
    rng = np.random.default_rng(seed)
    candidates: list[tuple[float, int, int, float, float, float, float, float, float]] = []
    finite_max = np.nanmax(grid_mean) if np.isfinite(np.nanmax(grid_mean)) else None

    for row in range(grid_mean.shape[0]):
        for col in range(grid_mean.shape[1]):
            score = float(grid_mean[row, col])
            if not np.isfinite(score):
                continue
            if threshold is not None and score < threshold:
                continue
            if max_only and finite_max is not None:
                tol = max_tol
                if max_rel is not None and max_rel > 0:
                    tol = max(tol, abs(finite_max) * max_rel)
                if score < finite_max - tol:
                    continue
            x_left, x_right = x_edges[col], x_edges[col + 1]
            y_bottom, y_top = y_edges[row], y_edges[row + 1]
            center_x = 0.5 * (x_left + x_right)
            center_y = 0.5 * (y_bottom + y_top)
            center_allowed = True
            if allowed_x is not None:
                center_allowed = center_allowed and (allowed_x[0] <= center_x <= allowed_x[1])
            if allowed_y is not None:
                center_allowed = center_allowed and (allowed_y[0] <= center_y <= allowed_y[1])
            if not center_allowed:
                continue

            candidates.append(
                (score, row, col, center_x, center_y, x_left, x_right, y_bottom, y_top)
            )

    candidates.sort(key=lambda item: item[0], reverse=True)
    if budget is not None and budget > 0:
        candidates = candidates[:budget]

    recommendations: list[GridCellRecommendation] = []

    def _sample_point(center_x: float, center_y: float, x_left: float, x_right: float, y_bottom: float, y_top: float) -> tuple[float, float]:
        if noise_std > 0:
            sx = rng.normal(center_x, noise_std)
            sy = rng.normal(center_y, noise_std)
            sx = float(np.clip(sx, x_left, x_right))
            sy = float(np.clip(sy, y_bottom, y_top))
        else:
            sx, sy = center_x, center_y
        return sx, sy

    for score, row, col, center_x, center_y, x_left, x_right, y_bottom, y_top in candidates:
        if noise_std > 0:
            sampled_x, sampled_y = _sample_point(center_x, center_y, x_left, x_right, y_bottom, y_top)
        else:
            sampled_x, sampled_y = center_x, center_y

        recommendations.append(
            GridCellRecommendation(
                row=row,
                col=col,
                score=score,
                center_x=center_x,
                center_y=center_y,
                sampled_x=sampled_x,
                sampled_y=sampled_y,
            )
        )

    if budget is not None and fill_budget_with_noise and candidates and len(recommendations) < budget:
        idx = 0
        while len(recommendations) < budget:
            score, row, col, center_x, center_y, x_left, x_right, y_bottom, y_top = candidates[idx % len(candidates)]
            sampled_x, sampled_y = _sample_point(center_x, center_y, x_left, x_right, y_bottom, y_top)
            recommendations.append(
                GridCellRecommendation(
                    row=row,
                    col=col,
                    score=score,
                    center_x=center_x,
                    center_y=center_y,
                    sampled_x=sampled_x,
                    sampled_y=sampled_y,
                )
            )
            idx += 1

    return recommendations


def plot_positions(
    positions: List[Position],
    output_path: Path,
    grid_mean: np.ndarray | None = None,
    grid_edges: tuple[np.ndarray, np.ndarray] | None = None,
    recommendations: Sequence[GridCellRecommendation] | None = None,
    influence_cmap: str = "coolwarm",
    figsize: tuple[float, float] = (10, 8),
    axis_range: tuple[tuple[float, float], tuple[float, float]] | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=figsize)

    if grid_mean is not None and grid_edges is not None:
        x_edges, y_edges = grid_edges
        vmin = np.nanmin(grid_mean)
        vmax = np.nanmax(grid_mean)
        if not (np.isfinite(vmin) and np.isfinite(vmax)):
            vmin, vmax = -1.0, 1.0
        elif vmin == vmax:
            eps = abs(vmin) * 0.05 if abs(vmin) > 0 else 1e-12
            vmin -= eps
            vmax += eps
        mesh = ax.pcolormesh(
            x_edges,
            y_edges,
            grid_mean,
            cmap=influence_cmap,
            shading="auto",
            alpha=0.55,
            vmin=vmin,
            vmax=vmax,
        )
        cbar = fig.colorbar(mesh, ax=ax, pad=0.02)
        cbar.set_label("Mean influence")

    ax.scatter([p.x for p in positions], [p.y for p in positions], s=70, c="#1f77b4", label="Episodes")

    if recommendations:
        ax.scatter(
            [rec.sampled_x for rec in recommendations],
            [rec.sampled_y for rec in recommendations],
            s=140,
            c="#D500F9",
            marker="*",
            edgecolors="black",
            linewidths=1.0,
            label=f"Collect ({len(recommendations)})",
        )

    if axis_range is not None:
        min_x, max_x = axis_range[0]
        min_y, max_y = axis_range[1]
    else:
        min_x, max_x, min_y, max_y = normalize(positions)

    def _make_ticks(lo: float, hi: float, step: float = 0.01) -> list[float]:
        # Strictly stay inside [lo, hi] with fixed step.
        ticks = list(np.arange(lo, hi + 1e-9, step))
        return [round(t, 4) for t in ticks]

    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)
    ax.set_xticks(_make_ticks(min_x, max_x, step=0.01))
    ax.set_yticks(_make_ticks(min_y, max_y, step=0.01))
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal")
    ax.grid(True, linestyle="--", alpha=0.4)

    ax.xaxis.set_major_formatter(FuncFormatter(lambda val, _: f"{val:.3f}"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda val, _: f"{val:.3f}"))

    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Libero collection guidance from init_positions.txt and influence scores."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/workspace/liushuo/cupid/datasets/local/libero-act"),
        help="Path to the Libero lerobot dataset root.",
    )
    parser.add_argument(
        "--init-positions",
        type=Path,
        default=None,
        help="Path to init_positions.txt (defaults to <dataset-root>/init_positions.txt).",
    )
    parser.add_argument(
        "--object",
        type=str,
        default="bbq_sauce",
        help="Object name whose start pose should be parsed.",
    )
    parser.add_argument(
        "--influence-file",
        type=Path,
        default=Path("/workspace/liushuo/cupid/new_test20/score_libero20.pkl"),
        help="Influence scores (pkl/npy/npz).",
    )
    parser.add_argument(
        "--grid-cols",
        type=int,
        default=4,
        help="Grid columns for influence averaging.",
    )
    parser.add_argument(
        "--grid-rows",
        type=int,
        default=4,
        help="Grid rows for influence averaging.",
    )
    parser.add_argument(
        "--grid-step",
        type=float,
        default=0.01,
        help="Grid step (m); overrides grid-cols/rows when grid ranges are set.",
    )
    parser.add_argument(
        "--grid-padding",
        type=float,
        default=0.01,
        help="Extra margin (m) added around data extent when building the grid.",
    )
    parser.add_argument(
        "--x-range",
        type=float,
        nargs=2,
        metavar=("MIN_X", "MAX_X"),
        default=(0.025, 0.075),
        help="Override x range for binning/plotting (default: 0.025 0.075).",
    )
    parser.add_argument(
        "--y-range",
        type=float,
        nargs=2,
        metavar=("MIN_Y", "MAX_Y"),
        default=(-0.125, -0.075),
        help="Override y range for binning/plotting (default: -0.125 -0.075).",
    )
    parser.add_argument(
        "--grid-x-range",
        type=float,
        nargs=2,
        metavar=("GRID_MIN_X", "GRID_MAX_X"),
        default=(0.035, 0.065),
        help="Range used to bin/color influence (default: 0.035 0.065).",
    )
    parser.add_argument(
        "--grid-y-range",
        type=float,
        nargs=2,
        metavar=("GRID_MIN_Y", "GRID_MAX_Y"),
        default=(-0.115, -0.085),
        help="Range used to bin/color influence (default: -0.115 -0.085).",
    )
    parser.add_argument(
        "--collection-threshold",
        type=float,
        default=None,
        help="Select grid cells with mean influence >= this value.",
    )
    parser.add_argument(
        "--collection-budget",
        type=int,
        default=10,
        help="Max number of grid cells to collect.",
    )
    parser.add_argument(
        "--collection-range",
        type=float,
        nargs=4,
        metavar=("MIN_X", "MAX_X", "MIN_Y", "MAX_Y"),
        default=(0.035, 0.065, -0.115, -0.085),
        help="Restrict recommended samples to this box (default focuses on center region).",
    )
    parser.add_argument(
        "--collection-max-only",
        action="store_true",
        default=True,
        help="Only select cells with the maximum influence (red cells).",
    )
    parser.add_argument(
        "--collection-max-tol",
        type=float,
        default=1e-12,
        help="Tolerance when comparing against max influence for red-cell selection.",
    )
    parser.add_argument(
        "--collection-max-rel",
        type=float,
        default=0.05,
        help="Relative tolerance to max influence (e.g., 0.05 keeps >=95% of max; helps include shallow red).",
    )
    parser.add_argument(
        "--collection-fill-budget",
        action="store_true",
        default=True,
        help="If enabled, keep sampling (with noise) from allowed cells until budget is met.",
    )
    parser.add_argument(
        "--collection-noise-std",
        type=float,
        default=0.002,
        help="Std dev (meters) of Gaussian noise around grid cell center.",
    )
    parser.add_argument(
        "--collection-seed",
        type=int,
        default=42,
        help="Random seed for noisy collection point generation.",
    )
    parser.add_argument(
        "--collection-output",
        type=Path,
        help="Optional CSV path to write recommended collection points.",
    )
    parser.add_argument(
        "--collection-txt-output",
        type=Path,
        default=Path.cwd() / "collection_points.txt",
        help="Optional TXT path to write recommended collection points (x,y).",
    )
    parser.add_argument(
        "--figure-output",
        type=Path,
        default=Path.cwd() / "libero_collection_guidance.png",
        help="Output path for the influence/position plot.",
    )
    parser.add_argument(
        "--influence-cmap",
        type=str,
        default="coolwarm",
        help="Colormap for influence heatmap.",
    )
    args = parser.parse_args()

    init_path = args.init_positions or (args.dataset_root / "init_positions.txt")
    positions = parse_libero_init_positions(init_path, target_object=args.object)
    data_min_x, data_max_x, data_min_y, data_max_y = normalize(positions)

    padding = args.grid_padding
    x_range = tuple(args.x_range) if args.x_range else (data_min_x - padding, data_max_x + padding)
    y_range = tuple(args.y_range) if args.y_range else (data_min_y - padding, data_max_y + padding)

    grid_x_range = tuple(args.grid_x_range) if args.grid_x_range else x_range
    grid_y_range = tuple(args.grid_y_range) if args.grid_y_range else y_range

    def _build_edges(rng: tuple[float, float], step: float) -> np.ndarray:
        edges = np.arange(rng[0], rng[1] + step * 0.5, step, dtype=np.float64)
        if edges[-1] < rng[1] - 1e-9:
            edges = np.concatenate([edges, [rng[1]]])
        return edges

    x_edges = _build_edges(grid_x_range, args.grid_step)
    y_edges = _build_edges(grid_y_range, args.grid_step)
    grid_cols = len(x_edges) - 1
    grid_rows = len(y_edges) - 1

    influences = load_influence_array(
        args.influence_file,
        expected_episodes=len(positions),
        train_dataset_path=args.dataset_root,
    )
    grid_mean, grid_edges = bin_influence_by_grid(
        positions,
        influences,
        cols=grid_cols,
        rows=grid_rows,
        x_range=None,
        y_range=None,
        x_edges=x_edges,
        y_edges=y_edges,
    )
    print(
        f"Loaded influence array of shape {influences.shape} and binned into grid "
        f"{grid_mean.shape} (cols={grid_cols}, rows={grid_rows})."
    )

    threshold = args.collection_threshold
    budget = args.collection_budget if args.collection_budget > 0 else None
    recommendations = recommend_cells_from_influence_grid(
        grid_mean,
        grid_edges,
        threshold=threshold,
        budget=budget,
        noise_std=args.collection_noise_std,
        seed=args.collection_seed,
        allowed_x=(args.collection_range[0], args.collection_range[1]) if args.collection_range else None,
        allowed_y=(args.collection_range[2], args.collection_range[3]) if args.collection_range else None,
        max_only=args.collection_max_only,
        max_tol=args.collection_max_tol,
        max_rel=args.collection_max_rel,
        fill_budget_with_noise=args.collection_fill_budget,
    )
    if recommendations:
        print("Recommended collection cells (row, col, score -> sampled x,y):")
        for rec in recommendations:
            print(
                f"  ({rec.row:02d}, {rec.col:02d}), score={rec.score:.6e}, "
                f"center=({rec.center_x:.4f}, {rec.center_y:.4f}) -> "
                f"({rec.sampled_x:.4f}, {rec.sampled_y:.4f})"
            )
    else:
        print("No grid cells met the threshold/budget for collection.")

    if recommendations and args.collection_output:
        args.collection_output.write_text(
            "x,y,row,col,score,center_x,center_y\n"
            + "\n".join(
                f"{rec.sampled_x:.6f},{rec.sampled_y:.6f},{rec.row},{rec.col},"
                f"{rec.score:.10e},{rec.center_x:.6f},{rec.center_y:.6f}"
                for rec in recommendations
            )
        )
        print(f"Wrote {len(recommendations)} recommended collection points to {args.collection_output}")
    if recommendations and args.collection_txt_output:
        args.collection_txt_output.write_text(
            "\n".join(f"{rec.sampled_x:.6f},{rec.sampled_y:.6f}" for rec in recommendations)
        )
        print(f"Wrote {len(recommendations)} points (x,y) to {args.collection_txt_output}")

    plot_positions(
        positions,
        args.figure_output,
        grid_mean=grid_mean,
        grid_edges=grid_edges,
        recommendations=recommendations,
        influence_cmap=args.influence_cmap,
        axis_range=(grid_x_range, grid_y_range),
    )

    print(f"Parsed {len(positions)} episodes from {init_path}")
    print(f"x range (plot/grid): {grid_x_range[0]:.4f} – {grid_x_range[1]:.4f}")
    print(f"y range (plot/grid): {grid_y_range[0]:.4f} – {grid_y_range[1]:.4f}")
    print(f"Figure written to: {args.figure_output}")


if __name__ == "__main__":
    main()
