"""
Utility for visualizing influence scores stored in score_libero20.pkl-style files.
If提供 eval_rollout_dir，会按 rollout 文件名里的 x/y 坐标把分数放回网格，可选仅展示中间 3×3 区域，并可用 SVG 刻度保持与 random_20.svg 一致。

Example:
  python plot_libero.py \
    --pkl /workspace/liushuo/cupid/new_test20/score_libero20.pkl \
    --eval-dir /workspace/liushuo/cupid/outputs/eval_grid_20 \
    --output /workspace/liushuo/cupid/new_test20/score_libero20_heatmap.png
"""

import argparse
import math
import os
import pickle
from typing import Iterable, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from pathlib import Path
import re
import random


def _pick_scores(payload: dict) -> np.ndarray:
    """Pick influence scores from common locations in the payload."""
    if isinstance(payload, dict):
        if isinstance(payload.get("influence_scores"), Iterable):
            return np.asarray(payload["influence_scores"], dtype=float)
        train = payload.get("train")
        if isinstance(train, dict) and isinstance(train.get("influence_scores"), Iterable):
            return np.asarray(train["influence_scores"], dtype=float)
    raise ValueError("No influence_scores found in the pickle file.")


def _build_grid(scores: np.ndarray, cols: Optional[int] = None) -> np.ndarray:
    """Reshape scores into a grid, padding with NaN if needed."""
    n = scores.size
    cols = cols or math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    grid = np.full((rows, cols), np.nan, dtype=float)
    grid.flat[:n] = scores
    return grid


def _coords_from_rollouts(eval_dir: Optional[str]) -> list[tuple[float, float, str]]:
    """Parse rollout filenames to recover (x, y) coordinates."""
    if not eval_dir:
        return []
    base = Path(eval_dir)
    if not base.exists():
        return []
    pat = re.compile(r"x([\d\.\-]+)_y([\d\.\-]+)\.pkl")
    coords: list[tuple[float, float, str]] = []
    for path in sorted(base.glob("rollout_ep*.pkl")):
        m = pat.search(path.name)
        if not m:
            continue
        try:
            x = float(m.group(1))
            y = float(m.group(2))
        except Exception:
            continue
        coords.append((x, y, path.name))
    return coords


def _build_grid_from_coords(scores: np.ndarray, coords: list[tuple[float, float, str]]) -> tuple[np.ndarray, list[float], list[float]]:
    """Place scores onto a coordinate grid using rollout x/y values (direct mapping)."""
    if not coords:
        return _build_grid(scores), [], []

    xs = sorted({x for x, _, _ in coords})
    ys = sorted({y for _, y, _ in coords}, reverse=True)  # y 从上到下
    grid = np.full((len(ys), len(xs)), np.nan, dtype=float)

    for idx, (x, y, _) in enumerate(coords[: scores.size]):
        try:
            r = ys.index(y)
            c = xs.index(x)
        except ValueError:
            continue
        grid[r, c] = scores[idx]
    return grid, xs, ys


def _ticks_from_svg(svg_path: Optional[str]) -> tuple[list[float], list[float]]:
    """Extract x/y tick labels from an SVG (e.g., random_20.svg)."""
    if not svg_path:
        return [], []
    path = Path(svg_path)
    if not path.exists():
        return [], []
    try:
        import xml.etree.ElementTree as ET

        root = ET.parse(path).getroot()
    except Exception:
        return [], []

    xs: list[float] = []
    ys: list[float] = []
    for elem in root.iter():
        if "text" not in elem.tag:
            continue
        text = "".join(elem.itertext()).strip()
        try:
            val = float(text)
        except Exception:
            continue
        x_attr = elem.attrib.get("x")
        y_attr = elem.attrib.get("y")
        if x_attr is not None and float(y_attr or 0) > 500:  # x 轴刻度在底部
            xs.append(val)
        elif x_attr is not None and float(x_attr) < 0:  # y 轴刻度在左侧
            ys.append(val)
    xs = sorted(set(xs))
    ys = sorted(set(ys), reverse=True)  # y 从上到下
    return xs, ys


def _edges_from_ticks(ticks: list[float]) -> list[float]:
    """Compute cell edges from ordered tick centers."""
    if len(ticks) < 2:
        return []
    diffs = [ticks[i + 1] - ticks[i] for i in range(len(ticks) - 1)]
    first = diffs[0]
    last = diffs[-1]
    edges = [ticks[0] - first / 2]
    edges.extend((ticks[i] + ticks[i + 1]) / 2 for i in range(len(ticks) - 1))
    edges.append(ticks[-1] + last / 2)
    return edges


def _bin_scores_to_grid(
    scores: np.ndarray,
    coords: list[tuple[float, float, str]],
    xs_tick: list[float],
    ys_tick: list[float],
) -> tuple[np.ndarray, list[float], list[float], list[float], list[float]]:
    """Average scores into grid cells defined by tick centers."""
    if not xs_tick or not ys_tick or not coords:
        grid, xs, ys = _build_grid_from_coords(scores, coords)
        return grid, xs, ys, [], []

    xs_tick = sorted(xs_tick)
    ys_tick_desc = sorted(ys_tick, reverse=True)
    xs_edges = _edges_from_ticks(xs_tick)
    ys_edges_asc = _edges_from_ticks(sorted(ys_tick))
    if not xs_edges or not ys_edges_asc:
        grid, xs, ys = _build_grid_from_coords(scores, coords)
        return grid, xs, ys, [], []

    grid_sum = np.zeros((len(ys_tick), len(xs_tick)), dtype=float)
    grid_cnt = np.zeros_like(grid_sum, dtype=int)

    for idx, (x, y, _) in enumerate(coords[: scores.size]):
        c = np.searchsorted(xs_edges, x, side="right") - 1
        r_asc = np.searchsorted(ys_edges_asc, y, side="right") - 1
        if c < 0 or c >= len(xs_tick) or r_asc < 0 or r_asc >= len(ys_tick):
            continue
        r = len(ys_tick) - 1 - r_asc  # convert asc index to plot row (top=large y)
        grid_sum[r, c] += float(scores[idx])
        grid_cnt[r, c] += 1

    with np.errstate(invalid="ignore", divide="ignore"):
        grid_mean = grid_sum / np.maximum(grid_cnt, 1)
        grid_mean[grid_cnt == 0] = np.nan

    return grid_mean, xs_tick, ys_tick_desc, xs_edges, ys_edges_asc


def _build_grid_from_ticks(
    scores: np.ndarray,
    coords: list[tuple[float, float, str]],
    xs_tick: list[float],
    ys_tick: list[float],
) -> tuple[np.ndarray, list[float], list[float], dict[int, tuple[int, int]]]:
    """Build grid using preset ticks; fill scores where coords match ticks."""
    if not xs_tick or not ys_tick:
        return _build_grid_from_coords(scores, coords)

    def _find_index(val: float, ticks: list[float], eps: float = 1e-5) -> Optional[int]:
        for i, t in enumerate(ticks):
            if abs(val - t) <= eps:
                return i
        return None

    grid = np.full((len(ys_tick), len(xs_tick)), np.nan, dtype=float)
    idx_to_cell: dict[int, tuple[int, int]] = {}
    for idx, (x, y, _) in enumerate(coords[: scores.size]):
        r = _find_index(y, ys_tick)
        c = _find_index(x, xs_tick)
        if r is None or c is None:
            continue
        grid[r, c] = scores[idx]
        idx_to_cell[idx] = (r, c)
    return grid, xs_tick, ys_tick, idx_to_cell


def _center_crop_grid(
    grid: np.ndarray,
    xs: list[float],
    ys: list[float],
    size: int = 3,
) -> Tuple[np.ndarray, list[float], list[float]]:
    """Pick the size×size window with the most valid cells, fallback to center."""
    if grid.size == 0 or len(xs) < size or len(ys) < size:
        return grid, xs, ys

    best = None  # (count, sum_scores, r_start, c_start)
    for r_start in range(len(ys) - size + 1):
        for c_start in range(len(xs) - size + 1):
            window = grid[r_start : r_start + size, c_start : c_start + size]
            valid = np.isfinite(window)
            count = int(valid.sum())
            score_sum = float(np.nansum(window))
            key = (count, score_sum, -r_start, -c_start)  # prefer upper-left if tie
            if best is None or key > best[:4]:
                best = (*key, r_start, c_start)

    if best is None:
        return grid, xs, ys

    _, _, _, _, r_start, c_start = best
    xs_crop = xs[c_start : c_start + size]
    ys_crop = ys[r_start : r_start + size]
    grid_crop = grid[r_start : r_start + size, c_start : c_start + size]
    return grid_crop, xs_crop, ys_crop


def plot_heatmap(
    pkl_path: str,
    output_path: Optional[str] = None,
    cols: Optional[int] = None,
    eval_dir: Optional[str] = None,
    top_k_highlight: int = 3,
    center_3x3: bool = True,
    svg_ticks: Optional[str] = None,
    budget: int = 0,
    jitter: float = 0.0,
    seed: int = 0,
    samples_out: Optional[str] = None,
) -> str:
    with open(pkl_path, "rb") as f:
        payload = pickle.load(f)

    scores = _pick_scores(payload)
    inferred_eval_dir = eval_dir or (payload.get("config", {}) or {}).get("eval_rollout_dir")
    coords = _coords_from_rollouts(inferred_eval_dir)
    xs_tick, ys_tick = _ticks_from_svg(svg_ticks)
    grid, xs, ys, xs_edges, ys_edges_asc = _bin_scores_to_grid(scores, coords, xs_tick, ys_tick)
    if not coords and not xs_tick:
        grid = _build_grid(scores, cols)
        if center_3x3:
            grid, xs, ys = _center_crop_grid(grid, xs, ys, size=3)
            xs_edges = _edges_from_ticks(xs) if len(xs) > 1 else []
            ys_edges_asc = _edges_from_ticks(sorted(ys)) if len(ys) > 1 else []
    elif center_3x3 and xs_tick and ys_tick:
        # 保留完整刻度网格，但只展示中心 3x3 的影响力，其余置 NaN
        size = 3
        r_start = max((len(ys) - size) // 2, 0)
        c_start = max((len(xs) - size) // 2, 0)
        mask = np.zeros_like(grid, dtype=bool)
        mask[r_start : r_start + size, c_start : c_start + size] = True
        grid = np.where(mask, grid, np.nan)

    fig, ax = plt.subplots(figsize=(6, 5))
    cmap = plt.colormaps["coolwarm"].copy()
    cmap.set_bad(color="#f0f0f0")  # 缺失值置灰
    finite_mask = np.isfinite(grid)
    vmin = float(np.nanmin(grid)) if finite_mask.any() else None
    vmax = float(np.nanmax(grid)) if finite_mask.any() else None

    use_extent = xs_edges and ys_edges_asc and len(xs_edges) == len(xs) + 1 and len(ys_edges_asc) == len(ys) + 1
    if use_extent:
        grid_vis = np.flipud(grid)  # 转为 y 轴升序
        x_centers = [(xs_edges[i] + xs_edges[i + 1]) / 2 for i in range(len(xs_edges) - 1)]
        y_centers = [(ys_edges_asc[i] + ys_edges_asc[i + 1]) / 2 for i in range(len(ys_edges_asc) - 1)]
        im = ax.pcolormesh(xs_edges, ys_edges_asc, grid_vis, cmap=cmap, shading="flat", vmin=vmin, vmax=vmax)
        # 画蓝色网格线（刻度边界）
        for x_line in xs_edges:
            ax.axvline(x_line, color="blue", linewidth=0.8)
        for y_line in ys_edges_asc:
            ax.axhline(y_line, color="blue", linewidth=0.8)
    else:
        im = ax.imshow(grid, cmap=cmap, aspect="auto", origin="upper", vmin=vmin, vmax=vmax)
        x_centers = list(range(grid.shape[1]))
        y_centers = list(range(grid.shape[0]))
        ax.set_xticks(np.arange(-0.5, grid.shape[1], 1), minor=True)
        ax.set_yticks(np.arange(-0.5, grid.shape[0], 1), minor=True)
        ax.grid(which="minor", color="blue", linewidth=0.8, linestyle="-")
        ax.tick_params(which="minor", bottom=False, left=False)

    # Annotate each valid cell with its mean value.
    if use_extent:
        for r in range(grid_vis.shape[0]):  # grid_vis is flipped
            for c in range(grid_vis.shape[1]):
                value = grid_vis[r, c]
                if not np.isfinite(value):
                    continue
                ax.text(x_centers[c], y_centers[r], f"{value:.2e}", ha="center", va="center", color="white", fontsize=8)
    else:
        for r in range(grid.shape[0]):
            for c in range(grid.shape[1]):
                value = grid[r, c]
                if not np.isfinite(value):
                    continue
                ax.text(c, r, f"{value:.2e}", ha="center", va="center", color="white", fontsize=8)

    # 高分格子加红框标记（按单元均值）。
    flat_vals = (grid_vis if use_extent else grid).flatten()
    valid_mask = np.isfinite(flat_vals)
    if valid_mask.any():
        valid_indices = np.nonzero(valid_mask)[0]
        top_indices = valid_indices[np.argsort(flat_vals[valid_indices])[::-1][:top_k_highlight]]
        for idx in top_indices:
            r, c = divmod(int(idx), grid_vis.shape[1] if use_extent else grid.shape[1])
            if use_extent:
                width = x_centers[1] - x_centers[0] if len(x_centers) > 1 else 0.01
                height = y_centers[1] - y_centers[0] if len(y_centers) > 1 else 0.01
                ax.add_patch(Rectangle((x_centers[c] - width / 2, y_centers[r] - height / 2), width, height, fill=False, edgecolor="red", linewidth=2))
            else:
                ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False, edgecolor="red", linewidth=2))

    ax.set_title("Influence Score Heatmap")
    if xs and ys:
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_xticks(range(len(xs)))
        ax.set_xticklabels([f"{x:.3f}" for x in xs], rotation=45, ha="right")
        ax.set_yticks(range(len(ys)))
        ax.set_yticklabels([f"{y:.3f}" for y in ys])
    #     if xs_tick and ys_tick:
    #         # 若需要保持与 SVG 相同的刻度线，在轴顶部/右侧添加对齐标尺
    #         ax_secondary_x = ax.secondary_xaxis("top")
    #         pos_x = np.linspace(0, len(xs) - 1, num=len(xs_tick)) if len(xs) > 1 else [0] * len(xs_tick)
    #         ax_secondary_x.set_xticks(pos_x)
    #         ax_secondary_x.set_xticklabels([f"{x:.3f}" for x in xs_tick], rotation=45, ha="left")
    #         ax_secondary_x.set_xlabel("SVG x ticks")
    #         ax_secondary_y = ax.secondary_yaxis("right")
    #         pos_y = np.linspace(0, len(ys) - 1, num=len(ys_tick)) if len(ys) > 1 else [0] * len(ys_tick)
    #         ax_secondary_y.set_yticks(pos_y)
    #         ax_secondary_y.set_yticklabels([f"{y:.3f}" for y in ys_tick])
    #         ax_secondary_y.set_ylabel("SVG y ticks")
    else:
        ax.set_xlabel("Column")
        ax.set_ylabel("Row")

    fig.colorbar(im, ax=ax, label="Influence score")

    output_path = output_path or os.path.splitext(pkl_path)[0] + "_heatmap.png"
    fig.tight_layout()
    save_path = Path(output_path)
    try:
        fig.savefig(save_path, dpi=300)
        final_path = save_path
    except PermissionError:
        alt = Path.cwd() / save_path.name
        fig.savefig(alt, dpi=300)
        final_path = alt
        print(f"Permission denied writing {save_path}, saved to {alt}")
    plt.close(fig)

    # 生成采集坐标并输出到 txt（按分数降序，添加扰动）
    if budget > 0 and xs and ys:
        rng = random.Random(seed)
        samples = []
        flat = []
        for r in range(grid.shape[0]):
            for c in range(grid.shape[1]):
                val = grid[r, c]
                if np.isfinite(val):
                    flat.append((val, r, c))
        flat.sort(reverse=True, key=lambda t: t[0])
        for val, r, c in flat[: budget]:
            if xs_edges and len(xs_edges) == len(xs) + 1:
                x_left, x_right = xs_edges[c], xs_edges[c + 1]
                center_x = 0.5 * (x_left + x_right)
            else:
                center_x = xs[c]
                x_left, x_right = center_x, center_x

            if ys_edges_asc and len(ys_edges_asc) == len(ys) + 1:
                asc_idx = len(ys) - 1 - r
                y_low, y_high = ys_edges_asc[asc_idx], ys_edges_asc[asc_idx + 1]
                center_y = 0.5 * (y_low + y_high)
            else:
                center_y = ys[r]
                y_low, y_high = center_y, center_y

            jitter_x = rng.uniform(-jitter, jitter) if jitter > 0 else 0.0
            jitter_y = rng.uniform(-jitter, jitter) if jitter > 0 else 0.0
            sampled_x = np.clip(center_x + jitter_x, min(x_left, x_right), max(x_left, x_right))
            sampled_y = np.clip(center_y + jitter_y, min(y_low, y_high), max(y_low, y_high))
            samples.append((sampled_x, sampled_y, val))

        if samples_out is None:
            samples_out = str(Path(pkl_path).with_suffix(".coords.txt"))
        out_path = Path(samples_out)
        try:
            with out_path.open("w", encoding="utf-8") as f:
                for x, y, val in samples:
                    f.write(f"{x:.6f} {y:.6f} {val:.6e}\n")
            samples_path = out_path
        except PermissionError:
            alt = Path.cwd() / out_path.name
            with alt.open("w", encoding="utf-8") as f:
                for x, y, val in samples:
                    f.write(f"{x:.6f} {y:.6f} {val:.6e}\n")
            samples_path = alt
            print(f"Permission denied writing {out_path}, saved coords to {alt}")
        print(f"Saved {len(samples)} sample coords to {samples_path}")

    return str(final_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot heatmap from influence scores.")
    parser.add_argument("--pkl", required=True, help="Path to score_libero20.pkl file.")
    parser.add_argument("--output", help="Output PNG path (default: alongside the pickle).")
    parser.add_argument(
        "--cols",
        type=int,
        help="Number of columns for the grid (default: ceil(sqrt(n_samples))).",
    )
    parser.add_argument(
        "--eval-dir",
        help="Eval rollout directory (rollout_ep*_x*_y*.pkl). If omitted, tries to read from config.eval_rollout_dir in the pickle.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Highlight the top-K highest scores with red borders to标记极不确定位置。",
    )
    parser.add_argument(
        "--no-center-3x3",
        action="store_false",
        dest="center_3x3",
        help="关闭 3x3 中心裁剪（默认开启）。",
    )
    parser.add_argument(
        "--svg-ticks",
        default=str(Path(__file__).resolve().parent.parent / "LIBERO/figures/random_20.svg"),
        help="从 SVG 读取刻度线，确保网格划分与 random_20.svg 一致，默认指向 LIBERO/figures/random_20.svg。",
    )
    parser.add_argument("--budget", type=int, default=0, help="根据最高分数输出采集点数量（0 表示不输出）。")
    parser.add_argument("--jitter", type=float, default=0.002, help="采集点随机扰动幅度（米）。")
    parser.add_argument("--seed", type=int, default=0, help="随机扰动的随机种子。")
    parser.add_argument("--samples-out", help="采集坐标输出 txt 路径（默认与 pkl 同名 .coords.txt）。")
    args = parser.parse_args()

    out = plot_heatmap(
        args.pkl,
        args.output,
        args.cols,
        args.eval_dir,
        args.top_k,
        args.center_3x3,
        args.svg_ticks,
        args.budget,
        args.jitter,
        args.seed,
        args.samples_out,
    )
    print(f"Saved heatmap to {out}")


if __name__ == "__main__":
    main()
