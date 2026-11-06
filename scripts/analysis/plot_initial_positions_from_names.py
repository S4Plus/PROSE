#!/usr/bin/env python3
"""Parse bottle start positions from episode folder names and plot the spread.

You can optionally point the script at a training log; any ``dataset.episodes``
list found in that log will be highlighted on the plot.

Folders are expected to follow the pattern: ``epo_<idx>_(<x>,<y>)`` where
``<x>`` and ``<y>`` are floating-point coordinates. The script extracts every
coordinate, reports summary statistics, and writes a figure that shows both the
raw coordinates and the same points mapped proportionally onto an image-sized
canvas (default 640x480).
"""

from __future__ import annotations

# Allow execution from any working directory without installing this checkout.
import sys as _prose_sys
from pathlib import Path as _ProsePath
_prose_sys.path.insert(0, str(_ProsePath(__file__).resolve().parents[2]))

import numpy as np
import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple
import pickle
import json
import csv

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


COORD_PATTERN = re.compile(
    r"^epo_(\d+)_\(\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*\)$"
)

PICK_X_RANGE = (1.062, 1.262)
PICK_Y_RANGE = (2.620, 3.120)
TEST_GRID_COLS = 4
TEST_GRID_ROWS = 10
DEFAULT_GRID_COLS = 4
DEFAULT_GRID_ROWS = 10


@dataclass
class Position:
    episode: int
    x: float
    y: float
    norm_x: float | None = None
    norm_y: float | None = None
    img_x: float | None = None
    img_y: float | None = None


@dataclass
class ImageTransform:
    min_x: float
    min_y: float
    scale: float
    offset_x: float
    offset_y: float


@dataclass
class GridCellRecommendation:
    row: int
    col: int
    score: float
    center_x: float
    center_y: float
    sampled_x: float
    sampled_y: float


def parse_points_csv(path: Path) -> List[Position]:
    points: List[Position] = []
    with path.open("r", newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)
    if not rows:
        return points
    header = [c.strip().lower() for c in rows[0]]
    start_idx = 1 if "x" in header and "y" in header else 0
    x_idx = header.index("x") if "x" in header else 0
    y_idx = header.index("y") if "y" in header else 1
    for row in rows[start_idx:]:
        if len(row) <= max(x_idx, y_idx):
            continue
        try:
            x = float(row[x_idx])
            y = float(row[y_idx])
        except ValueError:
            continue
        points.append(Position(episode=-1, x=x, y=y))
    return points


def parse_positions(root: Path) -> List[Position]:
    positions: List[Position] = []
    for entry in root.iterdir():
        if not entry.is_dir():
            continue
        match = COORD_PATTERN.match(entry.name)
        if not match:
            continue
        positions.append(
            Position(
                episode=int(match.group(1)),
                x=float(match.group(2)),
                y=float(match.group(3)),
            )
        )

    positions.sort(key=lambda pos: pos.episode)
    if not positions:
        raise RuntimeError(f"No episode folders matched pattern under {root}")
    return positions


def parse_positions_from_hdf5(path: Path, object_key: str) -> List[Position]:
    try:
        import h5py  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("h5py is required to read .hdf5 files. Install via `pip install h5py`.") from exc

    positions: List[Position] = []
    with h5py.File(path, "r") as f:
        data = f.get("data")
        if data is None:
            raise RuntimeError(f"No 'data' group found in {path}")
        demo_keys = sorted(data.keys(), key=lambda k: int(k.split("_")[1]))
        if not demo_keys:
            raise RuntimeError(f"No demos found under {path}/data")
        # Ensure object key exists
        sample_demo = data[demo_keys[0]]
        obj_group = sample_demo.get("datagen_info/object_poses")
        if obj_group is None or object_key not in obj_group:
            available = list(obj_group.keys()) if obj_group is not None else []
            raise RuntimeError(
                f"Object key '{object_key}' not found in datagen_info/object_poses. "
                f"Available: {available}"
            )
        for demo_key in demo_keys:
            demo_idx = int(demo_key.split("_")[1])
            pose_seq = data[demo_key]["datagen_info"]["object_poses"][object_key]
            pos = pose_seq[0, :3, 3]
            positions.append(Position(episode=demo_idx, x=float(pos[0]), y=float(pos[1])))

    positions.sort(key=lambda pos: pos.episode)
    return positions


def parse_episode_ids_from_log(log_path: Path) -> list[int]:
    """Extract the episode ids list from a training log."""

    content = log_path.read_text(encoding="utf-8", errors="ignore")
    matches = re.findall(r"episodes[^\[]*\[([^\]]+)\]", content, flags=re.IGNORECASE)
    if not matches:
        raise RuntimeError(f"No episode list found in log {log_path}")

    # Pick the longest match; this captures the full list even if it spans lines.
    candidate = max(matches, key=len)
    episode_ids = sorted({int(num) for num in re.findall(r"\d+", candidate)})
    if not episode_ids:
        raise RuntimeError(f"Failed to parse episode ids from {log_path}")

    return episode_ids




def parse_test_results_from_pkl(
    test_dir: Path, coords_map: dict[int, tuple[float, float]]
) -> tuple[list[Position], list[Position], list[Path]]:
    """Parse test results using epXXXX or episode_XXXX filenames mapped to grid coords."""

    success_positions: list[Position] = []
    fail_positions: list[Position] = []
    skipped: list[Path] = []
    name_re = re.compile(r"^(?:ep|episode)_(\d+)_")

    for path in sorted(test_dir.glob("*.pkl")):
        m = name_re.match(path.name)
        if not m:
            skipped.append(path)
            continue
        idx = int(m.group(1))
        if idx not in coords_map:
            skipped.append(path)
            continue
        ok: bool | None = None
        try:
            with path.open("rb") as f:
                df = pickle.load(f)
            if "success" in df:
                ok = bool(df["success"].iloc[-1])
        except Exception:
            ok = None

        if ok is None:
            lower = path.name.lower()
            if "success" in lower:
                ok = True
            elif "failure" in lower or "fail" in lower:
                ok = False
            else:
                skipped.append(path)
                continue

        x, y = coords_map[idx]
        pos = Position(episode=idx, x=x, y=y)
        (success_positions if ok else fail_positions).append(pos)

    return success_positions, fail_positions, skipped


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

    lengths: List[int] = []
    meta_path = train_root / "meta" / "episodes.jsonl"
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


def load_loss_array(
    path: Path,
    *,
    expected_episodes: int | None = None,
    train_dataset_path: Path | None = None,
) -> np.ndarray:
    """
    Load loss values (per-episode or per-sample) from mmap/npy/npz/pkl.

    If losses are per-sample and train_dataset_path is provided, they will be
    aggregated to per-episode using meta/episodes.jsonl lengths.
    """

    if not path.exists():
        raise FileNotFoundError(f"Loss file not found: {path}")

    array: np.ndarray | None = None
    if path.suffix == ".mmap":
        array = np.load(path, mmap_mode="r")
    elif path.suffix in {".npy", ".npz"}:
        loaded = np.load(path, allow_pickle=False)
        array = loaded["arr_0"] if isinstance(loaded, np.lib.npyio.NpzFile) else loaded
    elif path.suffix == ".pkl":
        with path.open("rb") as f:
            payload = pickle.load(f)
        for key in ("loss", "losses", "train_set_loss", "train_loss", "policy_loss"):
            if isinstance(payload, dict) and key in payload:
                array = payload[key]
                break
        if array is None and isinstance(payload, np.ndarray):
            array = payload
        if array is None:
            raise RuntimeError(f"Could not find loss array in {path}")
    else:
        raise ValueError(f"Unsupported loss file type: {path.suffix}")

    scores = np.asarray(array, dtype=np.float64).reshape(-1)

    if train_dataset_path and scores.ndim == 1:
        lengths = _read_episode_lengths(train_dataset_path)
        num_eps = len(lengths)
        total_samples = sum(lengths)
        if scores.shape[0] == total_samples:
            scores = _aggregate_samples_to_episodes(scores, lengths)
        elif scores.shape[0] != num_eps:
            print(
                f"Warning: loss array length {scores.shape[0]} does not match "
                f"episodes ({num_eps}) or samples ({total_samples}); keeping raw shape."
            )

    if expected_episodes is not None and scores.shape[0] != expected_episodes:
        print(
            f"Note: loss array length {scores.shape[0]} does not equal number of positions {expected_episodes}."
        )
    return scores


def bin_influence_by_grid(
    positions: List[Position],
    influences: np.ndarray,
    cols: int,
    rows: int,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    """Compute average influence per grid cell over (x, y) bins."""

    x_min, x_max = x_range
    y_min, y_max = y_range
    dx = (x_max - x_min) / cols
    dy = (y_max - y_min) / rows
    grid_sum = np.zeros((rows, cols), dtype=np.float64)
    grid_count = np.zeros((rows, cols), dtype=np.int64)

    for p in positions:
        if p.episode >= influences.shape[0]:
            continue
        if not (x_min <= p.x <= x_max and y_min <= p.y <= y_max):
            continue
        cx = min(cols - 1, max(0, int((p.x - x_min) / dx)))
        cy = min(rows - 1, max(0, int((p.y - y_min) / dy)))
        grid_sum[cy, cx] += float(influences[p.episode])
        grid_count[cy, cx] += 1

    with np.errstate(invalid="ignore", divide="ignore"):
        grid_mean = grid_sum / np.maximum(grid_count, 1e-9)
        grid_mean[grid_count == 0] = np.nan

    x_edges = np.linspace(x_min, x_max, cols + 1)
    y_edges = np.linspace(y_min, y_max, rows + 1)
    return grid_mean, (x_edges, y_edges)


def bin_counts_by_grid(
    positions: Sequence[Position],
    cols: int,
    rows: int,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    """Count training samples per grid cell."""

    x_min, x_max = x_range
    y_min, y_max = y_range
    dx = (x_max - x_min) / cols
    dy = (y_max - y_min) / rows
    counts = np.zeros((rows, cols), dtype=np.int64)

    for p in positions:
        if not (x_min <= p.x <= x_max and y_min <= p.y <= y_max):
            continue
        cx = min(cols - 1, max(0, int((p.x - x_min) / dx)))
        cy = min(rows - 1, max(0, int((p.y - y_min) / dy)))
        counts[cy, cx] += 1

    x_edges = np.linspace(x_min, x_max, cols + 1)
    y_edges = np.linspace(y_min, y_max, rows + 1)
    return counts, (x_edges, y_edges)


def bin_success_by_grid(
    success_positions: Sequence[Position],
    fail_positions: Sequence[Position],
    cols: int,
    rows: int,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[np.ndarray, np.ndarray]]:
    """Compute success rate per grid cell from success/fail positions."""

    x_min, x_max = x_range
    y_min, y_max = y_range
    dx = (x_max - x_min) / cols
    dy = (y_max - y_min) / rows
    success = np.zeros((rows, cols), dtype=np.int64)
    total = np.zeros((rows, cols), dtype=np.int64)

    def _accumulate(pos_list: Sequence[Position], is_success: bool) -> None:
        for p in pos_list:
            if not (x_min <= p.x <= x_max and y_min <= p.y <= y_max):
                continue
            cx = min(cols - 1, max(0, int((p.x - x_min) / dx)))
            cy = min(rows - 1, max(0, int((p.y - y_min) / dy)))
            total[cy, cx] += 1
            if is_success:
                success[cy, cx] += 1

    _accumulate(success_positions, True)
    _accumulate(fail_positions, False)

    with np.errstate(invalid="ignore", divide="ignore"):
        rate = success / np.maximum(total, 1)
        rate[total == 0] = np.nan

    x_edges = np.linspace(x_min, x_max, cols + 1)
    y_edges = np.linspace(y_min, y_max, rows + 1)
    return rate, success, total, (x_edges, y_edges)


def _rankdata(values: np.ndarray) -> np.ndarray:
    """Simple rankdata with average ranks for ties (1..N)."""

    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    n = len(values)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and values[order[j + 1]] == values[order[i]]:
            j += 1
        rank = 0.5 * (i + j) + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = rank
        i = j + 1
    return ranks


def _spearmanr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2:
        return float("nan")
    rx = _rankdata(x)
    ry = _rankdata(y)
    return float(np.corrcoef(rx, ry)[0, 1])


def _weighted_corr(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    if x.size < 2:
        return float("nan")
    w_sum = float(np.sum(w))
    if w_sum <= 0:
        return float("nan")
    wx = float(np.sum(w * x)) / w_sum
    wy = float(np.sum(w * y)) / w_sum
    cov = float(np.sum(w * (x - wx) * (y - wy))) / w_sum
    varx = float(np.sum(w * (x - wx) ** 2)) / w_sum
    vary = float(np.sum(w * (y - wy) ** 2)) / w_sum
    if varx <= 0 or vary <= 0:
        return float("nan")
    return cov / float(np.sqrt(varx * vary))


def _minmax_norm(values: np.ndarray) -> np.ndarray:
    finite = np.isfinite(values)
    if not np.any(finite):
        return np.zeros_like(values, dtype=np.float64)
    vmin = float(np.nanmin(values))
    vmax = float(np.nanmax(values))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or (vmax - vmin) <= 1e-12:
        return np.zeros_like(values, dtype=np.float64)
    return (values - vmin) / (vmax - vmin)


def recommend_cells_from_influence_grid(
    grid_mean: np.ndarray,
    grid_edges: tuple[np.ndarray, np.ndarray],
    *,
    threshold: float | None = None,
    budget: int | None = None,
    noise_std: float = 0.01,
    seed: int = 42,
) -> list[GridCellRecommendation]:
    """
    Select grid cells for additional collection based on influence scores.

    Higher influence (red on the heatmap) is treated as more uncertain. Cells
    with NaNs are skipped. If a threshold is provided, only cells with scores
    >= threshold are considered. If budget is provided, the top-K cells by
    score are kept. The final sampling position is the cell center plus
    Gaussian noise (clamped to the cell bounds).
    """

    if grid_mean.size == 0:
        return []

    x_edges, y_edges = grid_edges
    rng = np.random.default_rng(seed)
    candidates: list[tuple[float, int, int, float, float, float, float, float, float]] = []

    for row in range(grid_mean.shape[0]):
        for col in range(grid_mean.shape[1]):
            score = float(grid_mean[row, col])
            if not np.isfinite(score):
                continue
            if threshold is not None and score < threshold:
                continue
            x_left, x_right = x_edges[col], x_edges[col + 1]
            y_bottom, y_top = y_edges[row], y_edges[row + 1]
            center_x = 0.5 * (x_left + x_right)
            center_y = 0.5 * (y_bottom + y_top)
            candidates.append(
                (score, row, col, center_x, center_y, x_left, x_right, y_bottom, y_top)
            )

    candidates.sort(key=lambda item: item[0], reverse=True)
    if budget is not None and budget > 0:
        candidates = candidates[: budget]

    recommendations: list[GridCellRecommendation] = []
    for score, row, col, center_x, center_y, x_left, x_right, y_bottom, y_top in candidates:
        if noise_std > 0:
            sampled_x = rng.normal(center_x, noise_std)
            sampled_y = rng.normal(center_y, noise_std)
            sampled_x = float(np.clip(sampled_x, x_left, x_right))
            sampled_y = float(np.clip(sampled_y, y_bottom, y_top))
        else:
            sampled_x = center_x
            sampled_y = center_y

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

    return recommendations


def map_to_image(
    positions: Iterable[Position],
    width: int,
    height: int,
) -> ImageTransform:
    # 使用单一缩放因子保持比例放大，将点云完整放入给定画布
    xs = [p.x for p in positions]
    ys = [p.y for p in positions]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    span_x = max(max_x - min_x, 1e-9)
    span_y = max(max_y - min_y, 1e-9)

    scale = min(width / span_x, height / span_y)
    offset_x = (width - span_x * scale) / 2
    offset_y = (height - span_y * scale) / 2

    for p in positions:
        dx = p.x - min_x
        dy = p.y - min_y
        p.img_x = offset_x + dx * scale
        # 图像坐标原点在左上角，y 轴向下
        p.img_y = height - (offset_y + dy * scale)

    return ImageTransform(
        min_x=min_x, min_y=min_y, scale=scale, offset_x=offset_x, offset_y=offset_y
    )

YAW_LIST = [
    -np.pi / 4,
    -np.pi / 8,
    0.0,
    np.pi / 8,
    np.pi / 4,
]

def get_pose_by_index(idx: int):
    pick_x_range = (1.062, 1.262)
    pick_y_range = (2.62, 3.12)
    place_x_range = (-0.6, -0.58)
    place_y_range = (-0.3, 0)

    cols = 4
    rows = 10

    dx = (pick_x_range[1] - pick_x_range[0]) / cols
    dy = (pick_y_range[1] - pick_y_range[0]) / rows

    # 生成 40 个单元格中心 + 28 个边界点，共 68 个测试点
    centers = [
        (pick_x_range[0] + (i + 0.5) * dx, pick_y_range[0] + (j + 0.5) * dy)
        for j in range(rows)
        for i in range(cols)
    ]
    x_grid = [pick_x_range[0] + i * dx for i in range(cols + 1)]
    y_grid = [pick_y_range[0] + j * dy for j in range(rows + 1)]
    boundary = []
    for x in x_grid:
        boundary.append((x, pick_y_range[0]))
        boundary.append((x, pick_y_range[1]))
    for y in y_grid[1:-1]:
        boundary.append((pick_x_range[0], y))
        boundary.append((pick_x_range[1], y))

    test_points = centers + boundary  # len = 68
    assert 0 <= idx < len(test_points), f"idx must be in [0, {len(test_points)})"

    px, py = test_points[idx]

    place_x = 0.5 * (place_x_range[0] + place_x_range[1])
    place_y = 0.5 * (place_y_range[0] + place_y_range[1])

    yaw = YAW_LIST[idx % len(YAW_LIST)]

    pick_quat = euler_angles_to_quat([0, 0, yaw])
    place_quat = euler_angles_to_quat([0, 0, yaw])

    pos_pick = np.array([px, py, 0.9])
    pos_place = np.array([place_x, place_y, 0.10493])

    pick_pose = {"position": pos_pick, "orientation": pick_quat}
    place_pose = {"position": pos_place, "orientation": place_quat}
    return pick_pose, place_pose

def generate_test_positions(
    min_x: float,
    max_x: float,
    min_y: float,
    max_y: float,
    cols: int = 10,
    rows: int = 4,
) -> list[tuple[float, float]]:
    dx = (max_x - min_x) / cols
    dy = (max_y - min_y) / rows

    centers: list[tuple[float, float]] = []
    for j in range(rows):
        for i in range(cols):
            centers.append((min_x + (i + 0.5) * dx, min_y + (j + 0.5) * dy))

    x_grid = [min_x + i * dx for i in range(cols + 1)]
    y_grid = [min_y + j * dy for j in range(rows + 1)]
    boundary: list[tuple[float, float]] = []
    for x in x_grid:
        boundary.append((x, min_y))
        boundary.append((x, max_y))
    for y in y_grid[1:-1]:
        boundary.append((min_x, y))
        boundary.append((max_x, y))

    return centers + boundary


def plot_positions(
    positions: List[Position],
    output_path: Path,
    width: int,
    height: int,
    transform: ImageTransform,
    selected_episode_ids: set[int] | None = None,
    test_all: Sequence[Position] | None = None,
    test_fail: Sequence[Position] | None = None,
    grid_mean: np.ndarray | None = None,
    grid_edges: tuple[np.ndarray, np.ndarray] | None = None,
    recommended_positions: Sequence[Position] | None = None,
    influence_cmap: str = "coolwarm",
    influence_vmin: float | None = None,
    influence_vmax: float | None = None,
    heatmap_label: str = "Mean influence",
    *,
    # 图大小（英寸）
    fig_size: tuple[float, float] = (50,20),

    # 点样式
    point_size: float = 70,
    point_color: str = "#1f77b4",
    point_alpha: float = 0.85,
    point_edge_color: str = "white",
    point_edge_width: float = 0.6,
    marker: str = "o",

    # 字体大小
    title_fontsize: int = 28,
    label_fontsize: int = 22,
    tick_fontsize: int = 18,

    # 网格样式
    grid_color: str = "#FF1744",
    grid_linewidth: float = 2,
    grid_alpha: float = 0.35,
    # 选中点样式
    selected_point_size: float = 110,
    selected_point_color: str = "#FF6F00",
    selected_point_alpha: float = 0.9,
    selected_marker: str = "o",
    selected_edge_color: str = "black",
    selected_edge_width: float = 0.8,
    # 测试结果样式
    test_all_size: float = 80,
    test_all_color: str = "#90A4AE",
    test_all_edge: str = "black",
    test_all_edge_width: float = 0.7,
    test_all_marker: str = "s",
    test_fail_size: float = 140,
    test_fail_edge: str = "#2979FF",
    test_fail_edge_width: float = 10,
    test_fail_marker: str = "o",
    # 坐标轴范围
    x_range: tuple[float, float] | None = None,
    y_range: tuple[float, float] | None = None,
    # 标题
    title_override: str | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=fig_size)
    # 给左侧留出空间放图例，避免遮挡或被裁剪
    fig.subplots_adjust(left=0.2, right=0.95)

    # Optional background heatmap by grid cell.
    if grid_mean is not None and grid_edges is not None:
        x_edges, y_edges = grid_edges
        vmin = influence_vmin if influence_vmin is not None else np.nanmin(grid_mean)
        vmax = influence_vmax if influence_vmax is not None else np.nanmax(grid_mean)
        if not np.isfinite(vmin) or not np.isfinite(vmax) or np.isclose(vmin, vmax):
            vmin, vmax = vmin if np.isfinite(vmin) else -1.0, (vmax if np.isfinite(vmax) else 1.0)
            if vmin == vmax:
                vmax = vmin + 1e-3
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
        cbar = fig.colorbar(mesh, ax=ax, pad=0.015)
        cbar.set_label(heatmap_label, fontsize=label_fontsize)

    selected_positions = (
        [p for p in positions if p.episode in selected_episode_ids]
        if selected_episode_ids
        else []
    )
    background_positions = (
        [p for p in positions if p.episode not in selected_episode_ids]
        if selected_episode_ids
        else positions
    )

    if background_positions:
        ax.scatter(
            [p.x for p in background_positions],
            [p.y for p in background_positions],
            s=point_size,
            c=point_color,
            alpha=point_alpha,
            marker=marker,
            edgecolors=point_edge_color,
            linewidths=point_edge_width,
            label="All episodes" if selected_positions else None,
        )

    if selected_positions:
        ax.scatter(
            [p.x for p in selected_positions],
            [p.y for p in selected_positions],
            s=selected_point_size,
            c=selected_point_color,
            alpha=selected_point_alpha,
            marker=selected_marker,
            edgecolors=selected_edge_color,
            linewidths=selected_edge_width,
            label=f"Selected ({len(selected_positions)})",
        )

    if recommended_positions:
        ax.scatter(
            [p.x for p in recommended_positions],
            [p.y for p in recommended_positions],
            s=150,
            c="#D500F9",
            alpha=0.9,
            marker="*",
            edgecolors="black",
            linewidths=1.0,
            label=f"Recommend collect ({len(recommended_positions)})",
        )

    if test_all:
        ax.scatter(
            [p.x for p in test_all],
            [p.y for p in test_all],
            s=test_all_size,
            c=test_all_color,
            alpha=0.85,
            marker=test_all_marker,
            edgecolors=test_all_edge,
            linewidths=test_all_edge_width,
            label=f"Test points ({len(test_all)})",
        )
    if test_fail:
        ax.scatter(
            [p.x for p in test_fail],
            [p.y for p in test_fail],
            s=test_fail_size,
            facecolors="#FF1744",
            edgecolors=test_fail_edge,
            linewidths=test_fail_edge_width,
            marker=test_fail_marker,
            label=f"Test fail ({len(test_fail)})",
        )

    def _make_ticks(min_val: float, max_val: float, step: float = 0.05) -> list[float]:
        ticks: list[float] = []
        current = min_val
        while current <= max_val + 1e-9:
            ticks.append(round(current, 3))
            current += step
        return ticks

    if x_range is None:
        min_x = min(p.x for p in positions)
        max_x = max(p.x for p in positions)
    else:
        min_x, max_x = x_range
    if y_range is None:
        min_y = min(p.y for p in positions)
        max_y = max(p.y for p in positions)
    else:
        min_y, max_y = y_range

    ax.set_xticks(_make_ticks(min_x, max_x))
    ax.set_yticks(_make_ticks(min_y, max_y))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda val, _: f"{val:.3f}"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda val, _: f"{val:.3f}"))

    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)

    ax.set_xlabel("x", fontsize=label_fontsize)
    ax.set_ylabel("y", fontsize=label_fontsize)
    ax.tick_params(axis="both", labelsize=tick_fontsize)

    ax.set_aspect("equal")

    ax.grid(True, linestyle="--", linewidth=grid_linewidth, alpha=grid_alpha, color=grid_color)

    if title_override is not None:
        title = title_override
    else:
        title = "Static_Pick_Bottle_400 initial bottle positions"
        if selected_episode_ids:
            title += f" — highlighting {len(selected_positions)} episodes"
        if grid_mean is not None:
            title += " — mean influence per cell"

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(
            loc="center right",
            bbox_to_anchor=(-0.22, 0.5),
            borderaxespad=1.0,
        )

    fig.suptitle(title, fontsize=title_fontsize)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)



def generate_test_points(
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    cols: int = TEST_GRID_COLS,
    rows: int = TEST_GRID_ROWS,
) -> list[tuple[float, float]]:
    dx = (x_range[1] - x_range[0]) / cols
    dy = (y_range[1] - y_range[0]) / rows

    centers: list[tuple[float, float]] = []
    for j in range(rows):
        for i in range(cols):
            centers.append((x_range[0] + (i + 0.5) * dx, y_range[0] + (j + 0.5) * dy))

    x_grid = [x_range[0] + i * dx for i in range(cols + 1)]
    y_grid = [y_range[0] + j * dy for j in range(rows + 1)]
    boundary: list[tuple[float, float]] = []
    for x in x_grid:
        boundary.append((x, y_range[0]))
        boundary.append((x, y_range[1]))
    for y in y_grid[1:-1]:
        boundary.append((x_range[0], y))
        boundary.append((x_range[1], y))

    return centers + boundary


def get_pose_by_index(idx: int, x_range: tuple[float, float], y_range: tuple[float, float]) -> tuple[float, float]:
    """Return the (x, y) pick position for a deterministic 68-point grid."""

    test_points = generate_test_points(x_range=x_range, y_range=y_range)
    if not 0 <= idx < len(test_points):
        raise ValueError(f"idx must be in [0, {len(test_points)})")
    return test_points[idx]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Parse initial bottle positions encoded in episode folder names and "
            "plot both the raw and proportionally mapped coordinates."
        )
    )
    # 参数说明（中文）：
    # --root: 训练/采集数据所在目录，文件夹名中包含坐标信息，用来解析瓶子初始位置
    # --width/--height: 将坐标按比例映射到图像时使用的画布宽高
    # --output: 绘制散点图的输出路径
    # --log: 训练日志路径，解析其中的 episodes 列表以高亮展示
    # --grid-output: 若提供，则把 68 个测试点坐标以 CSV 写入该路径
    # --test-dir: 评测结果（epXXXX_*.pkl）所在目录，用于叠加成功/失败标记
    # --test-coords: 评测坐标文件（已弃用，保留兼容）；当前由脚本自动生成 68 点网格
    parser.add_argument(
        "--root",
        type=Path,
        default="/ext_workspace/liushuo/position/Static_Pick_Bottle_400/",
        help="Path containing episode folders (default: script directory).",
    )
    parser.add_argument(
        "--hdf5",
        type=Path,
        default=None,
        help="Optional: demo.hdf5 path (overrides --root).",
    )
    parser.add_argument(
        "--object-key",
        type=str,
        default="square_nut",
        help="Object key under datagen_info/object_poses when reading from hdf5.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=640,
        help="Image width used for proportional mapping (default: 640)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Image height used for proportional mapping (default: 480)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path.cwd() / "initial_positions_from_names.png",
        help="Output path for the generated figure (default: ./initial_positions_from_names.png)",
    )
    parser.add_argument(
        "--log",
        type=Path,
        help="Training log file containing the sampled episodes to highlight",
    )
    parser.add_argument(
        "--grid-output",
        type=Path,
        help="Optional path to write generated 68-point test grid (CSV)",
    )
    parser.add_argument(
        "--test-dir",
        type=Path,
        help="Path to test eval pickles (epXXXX_*.pkl) containing idx/success",
    )
    parser.add_argument(
        "--test-coords",
        type=Path,
        default=Path.cwd() / "test.txt",
        help="Path to test.txt that lists pick positions for each test idx",
    )
    parser.add_argument(
        "--influence-file",
        type=Path,
        help="Path to influence scores (npy/npz/pkl). If provided, averages per grid cell and colors the background.",
    )
    parser.add_argument(
        "--loss-file",
        type=Path,
        help="Optional: path to loss values (mmap/npy/npz/pkl). Used for grid analysis vs success rate.",
    )
    parser.add_argument(
        "--train-dataset-path",
        type=Path,
        default=Path("/ext_workspace/liushuo/Static_Pick_Bottle_400"),
        help="Train dataset root (used to aggregate per-sample influence into per-episode via meta/episodes.jsonl).",
    )
    parser.add_argument(
        "--grid-cols",
        type=int,
        default=DEFAULT_GRID_COLS,
        help="Grid columns for influence averaging (default matches test grid).",
    )
    parser.add_argument(
        "--grid-rows",
        type=int,
        default=DEFAULT_GRID_ROWS,
        help="Grid rows for influence averaging (default matches test grid).",
    )
    parser.add_argument(
        "--x-range",
        type=float,
        nargs=2,
        default=None,
        metavar=("X_MIN", "X_MAX"),
        help="Override x-axis range (min max).",
    )
    parser.add_argument(
        "--y-range",
        type=float,
        nargs=2,
        default=None,
        metavar=("Y_MIN", "Y_MAX"),
        help="Override y-axis range (min max).",
    )
    parser.add_argument(
        "--influence-cmap",
        type=str,
        default="coolwarm",
        help="Colormap for influence background.",
    )
    parser.add_argument(
        "--collection-threshold",
        type=float,
        default=None,
        help="Select grid cells with mean influence >= this value for new collection.",
    )
    parser.add_argument(
        "--collection-budget",
        type=int,
        default=0,
        help="Max number of grid cells to collect (0 disables budget).",
    )
    parser.add_argument(
        "--collection-noise-std",
        type=float,
        default=0.01,
        help="Std dev (meters) of Gaussian noise added around the grid cell center when proposing collection points.",
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
        "--overlay-points-csv",
        type=Path,
        default=None,
        help="Optional CSV with x,y columns to overlay as collection points.",
    )
    parser.add_argument(
        "--uncertainty-cmap",
        type=str,
        default="Reds",
        help="Colormap for uncertainty heatmap (single-hue recommended).",
    )
    parser.add_argument(
        "--min-cell-samples",
        type=int,
        default=1,
        help="Minimum test samples per grid cell for filtered correlations (set to 1 to disable filtering).",
    )
    args = parser.parse_args()

    if args.hdf5 is None and args.root and args.root.is_file() and args.root.suffix in {".hdf5", ".h5"}:
        args.hdf5 = args.root

    if args.hdf5 is not None:
        positions = parse_positions_from_hdf5(args.hdf5, args.object_key)
        dataset_label = args.hdf5.stem
    else:
        positions = parse_positions(args.root)
        dataset_label = args.root.name
    min_x, max_x, min_y, max_y = normalize(positions)
    transform = map_to_image(positions, width=args.width, height=args.height)
    overlay_positions: list[Position] = []
    if args.overlay_points_csv:
        overlay_positions = parse_points_csv(args.overlay_points_csv)
        if overlay_positions:
            print(f"Loaded {len(overlay_positions)} overlay points from {args.overlay_points_csv}")
        else:
            print(f"Warning: no points found in {args.overlay_points_csv}")
    selected_ids: set[int] | None = None
    if args.log:
        selected_ids = set(parse_episode_ids_from_log(args.log))
        available_ids = {p.episode for p in positions}
        missing_ids = sorted(selected_ids - available_ids)
        if missing_ids:
            print(f"Warning: {len(missing_ids)} ids from log not found in {args.root}: {missing_ids}")
        print(f"Highlighting {len(selected_ids & available_ids)} episodes parsed from {args.log}")

    # Resolve plotting ranges
    if args.x_range is not None:
        x_range = (float(args.x_range[0]), float(args.x_range[1]))
    elif args.hdf5 is not None:
        x_range = (min_x, max_x)
    else:
        x_range = PICK_X_RANGE

    if args.y_range is not None:
        y_range = (float(args.y_range[0]), float(args.y_range[1]))
    elif args.hdf5 is not None:
        y_range = (min_y, max_y)
    else:
        y_range = PICK_Y_RANGE

    test_points = generate_test_points(x_range=x_range, y_range=y_range, cols=args.grid_cols, rows=args.grid_rows)
    test_all: list[Position] | None = [
        Position(episode=idx, x=xy[0], y=xy[1]) for idx, xy in enumerate(test_points)
    ]
    test_fail: list[Position] | None = None
    train_counts, _ = bin_counts_by_grid(
        positions,
        cols=args.grid_cols,
        rows=args.grid_rows,
        x_range=x_range,
        y_range=y_range,
    )
    success_rate: np.ndarray | None = None
    success_counts: np.ndarray | None = None
    total_counts: np.ndarray | None = None
    success_edges: tuple[np.ndarray, np.ndarray] | None = None
    if args.test_dir:
        coords_map = {idx: xy for idx, xy in enumerate(test_points)}
        success_eps, fail_eps, skipped = parse_test_results_from_pkl(args.test_dir, coords_map)
        # Mark fail positions; success are implicit via absence of red circle.
        test_fail = fail_eps
        if skipped:
            print(f"Warning: skipped {len(skipped)} pkl files (no coords or read failure): {[p.name for p in skipped]}")
        print(
            f"Overlay test results from {args.test_dir}: "
            f"{len(success_eps)} success, {len(fail_eps)} fail; total grid points {len(test_all)}"
        )
        success_rate, success_counts, total_counts, success_edges = bin_success_by_grid(
            success_eps,
            fail_eps,
            cols=args.grid_cols,
            rows=args.grid_rows,
            x_range=x_range,
            y_range=y_range,
        )
    else:
        print(f"Overlaying {len(test_points)} test grid points generated by get_pose_by_index()")

    grid_mean: np.ndarray | None = None
    grid_edges: tuple[np.ndarray, np.ndarray] | None = None
    loss_grid: np.ndarray | None = None
    loss_edges: tuple[np.ndarray, np.ndarray] | None = None
    recommendations: list[GridCellRecommendation] = []
    if args.influence_file:
        influences = load_influence_array(
            args.influence_file,
            expected_episodes=len(positions),
            train_dataset_path=args.train_dataset_path,
        )
        grid_mean, grid_edges = bin_influence_by_grid(
            positions,
            influences,
            cols=args.grid_cols,
            rows=args.grid_rows,
            x_range=x_range,
            y_range=y_range,
        )
        print(
            f"Loaded influence array of shape {influences.shape} and binned into grid "
            f"{grid_mean.shape} (cols={args.grid_cols}, rows={args.grid_rows})."
        )
        threshold = args.collection_threshold
        budget = args.collection_budget if args.collection_budget > 0 else None
        if threshold is not None or budget is not None:
            recommendations = recommend_cells_from_influence_grid(
                grid_mean,
                grid_edges,
                threshold=threshold,
                budget=budget,
                noise_std=args.collection_noise_std,
                seed=args.collection_seed,
            )
            if recommendations:
                print("Recommended collection cells (row, col, score -> sampled x,y):")
                for rec in recommendations:
                    print(
                        f"  ({rec.row:02d}, {rec.col:02d}), score={rec.score:.4f}, "
                        f"center=({rec.center_x:.3f}, {rec.center_y:.3f}) -> "
                        f"({rec.sampled_x:.3f}, {rec.sampled_y:.3f})"
                    )
            else:
                print("No grid cells met the threshold/budget for collection.")
    else:
        if args.collection_threshold is not None or args.collection_budget > 0:
            print("Collection planning requested but --influence-file is missing; skipping.")

    if args.loss_file:
        losses = load_loss_array(
            args.loss_file,
            expected_episodes=len(positions),
            train_dataset_path=args.train_dataset_path,
        )
        loss_grid, loss_edges = bin_influence_by_grid(
            positions,
            losses,
            cols=args.grid_cols,
            rows=args.grid_rows,
            x_range=x_range,
            y_range=y_range,
        )
        print(
            f"Loaded loss array of shape {losses.shape} and binned into grid "
            f"{loss_grid.shape} (cols={args.grid_cols}, rows={args.grid_rows})."
        )

    recommended_positions: list[Position] | None = None
    if recommendations:
        recommended_positions = [
            Position(episode=-1, x=rec.sampled_x, y=rec.sampled_y) for rec in recommendations
        ]
    if overlay_positions:
        if recommended_positions is None:
            recommended_positions = []
        recommended_positions.extend(overlay_positions)

    heatmap_grid = grid_mean
    heatmap_edges = grid_edges
    heatmap_cmap = args.influence_cmap
    heatmap_label = "Mean influence"
    if heatmap_grid is None and loss_grid is not None:
        heatmap_grid = loss_grid
        heatmap_edges = loss_edges
        heatmap_cmap = "magma"
        heatmap_label = "Mean loss"

    title_suffix = None
    if heatmap_grid is not None:
        title_suffix = "mean loss per cell" if heatmap_label == "Mean loss" else "mean influence per cell"

    plot_positions(
        positions,
        args.output,
        args.width,
        args.height,
        transform,
        selected_episode_ids=selected_ids,
        test_all=test_all,
        test_fail=test_fail,
        grid_mean=heatmap_grid,
        grid_edges=heatmap_edges,
        recommended_positions=recommended_positions,
        influence_cmap=heatmap_cmap,
        heatmap_label=heatmap_label,
        x_range=x_range,
        y_range=y_range,
        title_override=f"{dataset_label} initial positions — {title_suffix}"
        if title_suffix is not None
        else f"{dataset_label} initial positions",
    )

    # If we have both influence grid and test outcomes, analyze correlation.
    if (
        args.test_dir
        and grid_mean is not None
        and grid_edges is not None
        and success_rate is not None
        and success_counts is not None
        and total_counts is not None
        and success_edges is not None
    ):
        success_positions = success_eps if "success_eps" in locals() else []
        fail_positions = fail_eps if "fail_eps" in locals() else []
        mask = np.isfinite(grid_mean) & np.isfinite(success_rate) & (total_counts > 0)
        flat_infl = grid_mean[mask]
        flat_succ = success_rate[mask]
        pearson = float(np.corrcoef(flat_infl, flat_succ)[0, 1]) if flat_infl.size > 1 else float("nan")
        spearman = _spearmanr(flat_infl, flat_succ)
        weights = total_counts[mask].astype(np.float64)
        w_pearson = _weighted_corr(flat_infl, flat_succ, weights)
        w_spearman = _weighted_corr(_rankdata(flat_infl), _rankdata(flat_succ), weights)

        min_pearson = float("nan")
        min_spearman = float("nan")
        if args.min_cell_samples > 1:
            min_mask = mask & (total_counts >= args.min_cell_samples)
            min_infl = grid_mean[min_mask]
            min_succ = success_rate[min_mask]
            min_pearson = float(np.corrcoef(min_infl, min_succ)[0, 1]) if min_infl.size > 1 else float("nan")
            min_spearman = _spearmanr(min_infl, min_succ)

        count_mask = np.isfinite(success_rate) & (total_counts > 0) & (train_counts > 0)
        flat_train_counts = train_counts[count_mask].astype(np.float64)
        flat_succ_counts = success_rate[count_mask]
        count_weights = total_counts[count_mask].astype(np.float64)
        pearson_counts = (
            float(np.corrcoef(flat_train_counts, flat_succ_counts)[0, 1])
            if flat_train_counts.size > 1
            else float("nan")
        )
        spearman_counts = _spearmanr(flat_train_counts, flat_succ_counts)
        w_pearson_counts = _weighted_corr(flat_train_counts, flat_succ_counts, count_weights)
        w_spearman_counts = _weighted_corr(_rankdata(flat_train_counts), _rankdata(flat_succ_counts), count_weights)

        min_pearson_counts = float("nan")
        min_spearman_counts = float("nan")
        if args.min_cell_samples > 1:
            count_min_mask = count_mask & (total_counts >= args.min_cell_samples)
            min_train_counts = train_counts[count_min_mask].astype(np.float64)
            min_succ_counts = success_rate[count_min_mask]
            min_pearson_counts = (
                float(np.corrcoef(min_train_counts, min_succ_counts)[0, 1])
                if min_train_counts.size > 1
                else float("nan")
            )
            min_spearman_counts = _spearmanr(min_train_counts, min_succ_counts)

        print("Grid success analysis:")
        print(f"  Cells with data: {flat_infl.size}")
        if flat_infl.size > 0:
            print(f"  Influence mean: min={float(np.nanmin(flat_infl)):.6g}, max={float(np.nanmax(flat_infl)):.6g}")
            print(f"  Success rate:   min={float(np.nanmin(flat_succ)):.3f}, max={float(np.nanmax(flat_succ)):.3f}")
        print(f"  Pearson(influence, success_rate) = {pearson:.4f}")
        print(f"  Spearman(influence, success_rate) = {spearman:.4f}")
        print(f"  Weighted Pearson(influence, success_rate) = {w_pearson:.4f}")
        print(f"  Weighted Spearman(influence, success_rate) = {w_spearman:.4f}")
        if args.min_cell_samples > 1:
            print(
                f"  Filtered (n>={args.min_cell_samples}) Pearson(influence, success_rate) = {min_pearson:.4f}"
            )
            print(
                f"  Filtered (n>={args.min_cell_samples}) Spearman(influence, success_rate) = {min_spearman:.4f}"
            )
        if flat_train_counts.size > 0:
            print(f"  Pearson(train_count, success_rate) = {pearson_counts:.4f}")
            print(f"  Spearman(train_count, success_rate) = {spearman_counts:.4f}")
            print(f"  Weighted Pearson(train_count, success_rate) = {w_pearson_counts:.4f}")
            print(f"  Weighted Spearman(train_count, success_rate) = {w_spearman_counts:.4f}")
            if args.min_cell_samples > 1:
                print(
                    f"  Filtered (n>={args.min_cell_samples}) Pearson(train_count, success_rate) = {min_pearson_counts:.4f}"
                )
                print(
                    f"  Filtered (n>={args.min_cell_samples}) Spearman(train_count, success_rate) = {min_spearman_counts:.4f}"
                )

        cells: list[dict[str, float | int]] = []
        for row in range(grid_mean.shape[0]):
            for col in range(grid_mean.shape[1]):
                if total_counts[row, col] <= 0:
                    continue
                mean_val = grid_mean[row, col]
                succ_val = success_rate[row, col]
                if not np.isfinite(mean_val) or not np.isfinite(succ_val):
                    continue
                cells.append(
                    {
                        "row": row,
                        "col": col,
                        "mean_influence": float(mean_val),
                        "success_rate": float(succ_val),
                        "total": int(total_counts[row, col]),
                    }
                )

        def _print_cells(title: str, items: list[dict[str, float | int]]) -> None:
            print(title)
            if not items:
                print("  (no cells)")
                return
            for item in items:
                print(
                    f"  ({int(item['row']):02d},{int(item['col']):02d}) "
                    f"infl={item['mean_influence']:.6g} "
                    f"succ={item['success_rate']:.3f} "
                    f"n={int(item['total'])}"
                )

        if cells:
            k = min(5, len(cells))
            by_infl_desc = sorted(cells, key=lambda d: d["mean_influence"], reverse=True)
            by_infl_asc = list(reversed(by_infl_desc))
            by_succ_desc = sorted(cells, key=lambda d: d["success_rate"], reverse=True)
            by_succ_asc = list(reversed(by_succ_desc))
            _print_cells(f"  Top-{k} influence cells:", by_infl_desc[:k])
            _print_cells(f"  Bottom-{k} influence cells:", by_infl_asc[:k])
            _print_cells(f"  Top-{k} success cells:", by_succ_desc[:k])
            _print_cells(f"  Bottom-{k} success cells:", by_succ_asc[:k])

        metrics_path = args.output.with_name(f"{args.output.stem}_grid_metrics.csv")
        with metrics_path.open("w") as f:
            f.write("row,col,mean_influence,success_rate,success,total\n")
            for row in range(grid_mean.shape[0]):
                for col in range(grid_mean.shape[1]):
                    mean_val = grid_mean[row, col]
                    succ_val = success_rate[row, col]
                    succ = success_counts[row, col]
                    total = total_counts[row, col]
                    if not np.isfinite(mean_val) and total == 0:
                        continue
                    f.write(
                        f"{row},{col},{mean_val:.8g},{succ_val:.6g},{int(succ)},{int(total)}\n"
                    )
        print(f"  Wrote grid metrics to: {metrics_path}")

        # Plot success-rate heatmap
        fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
        x_edges, y_edges = success_edges
        mesh = ax.pcolormesh(
            x_edges,
            y_edges,
            success_rate,
            cmap="viridis",
            shading="auto",
            vmin=0.0,
            vmax=1.0,
        )
        if success_positions:
            ax.scatter(
                [p.x for p in success_positions],
                [p.y for p in success_positions],
                s=35,
                c="white",
                edgecolors="black",
                linewidths=0.6,
                label=f"Success ({len(success_positions)})",
            )
        if fail_positions:
            ax.scatter(
                [p.x for p in fail_positions],
                [p.y for p in fail_positions],
                s=45,
                facecolors="none",
                edgecolors="#FF1744",
                linewidths=1.2,
                label=f"Fail ({len(fail_positions)})",
            )
        ax.set_title("Success Rate per Grid Cell")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_xlim(x_range[0], x_range[1])
        ax.set_ylim(y_range[0], y_range[1])
        if success_positions or fail_positions:
            ax.legend(loc="upper right")
        fig.colorbar(mesh, ax=ax, label="Success rate")
        success_path = args.output.with_name(f"{args.output.stem}_success_rate.png")
        fig.savefig(success_path, dpi=200)
        plt.close(fig)
        print(f"  Wrote success-rate heatmap to: {success_path}")

        # Scatter influence vs success rate
        if flat_infl.size > 0:
            fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
            ax.scatter(flat_infl, flat_succ, s=40, alpha=0.8)
            ax.set_xlabel("Mean influence (cell)")
            ax.set_ylabel("Success rate (cell)")
            ax.set_title("Influence vs Success Rate")
            ax.grid(True, linestyle="--", alpha=0.3)
            ax.text(
                0.02,
                0.98,
                f"Pearson: {pearson:.3f}\nSpearman: {spearman:.3f}\nCells: {flat_infl.size}",
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=9,
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
            )
            scatter_path = args.output.with_name(f"{args.output.stem}_influence_vs_success.png")
            fig.savefig(scatter_path, dpi=200)
            plt.close(fig)
            print(f"  Wrote influence vs success scatter to: {scatter_path}")

        # Scatter count vs success rate
        if flat_train_counts.size > 0:
            fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
            ax.scatter(flat_train_counts, flat_succ_counts, s=40, alpha=0.8)
            ax.set_xlabel("Train samples per cell")
            ax.set_ylabel("Success rate (cell)")
            ax.set_title("Train Sample Count vs Success Rate")
            ax.grid(True, linestyle="--", alpha=0.3)
            ax.text(
                0.02,
                0.98,
                f"Pearson: {pearson_counts:.3f}\nSpearman: {spearman_counts:.3f}\nCells: {flat_train_counts.size}",
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=9,
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
            )
            count_scatter_path = args.output.with_name(f"{args.output.stem}_count_vs_success.png")
            fig.savefig(count_scatter_path, dpi=200)
            plt.close(fig)
            print(f"  Wrote count vs success scatter to: {count_scatter_path}")

    # Optional: analyze loss vs success (lower loss -> better mastery).
    if args.loss_file and loss_grid is not None and loss_edges is not None:
        # Loss heatmap (standalone)
        fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
        x_edges, y_edges = loss_edges
        mesh = ax.pcolormesh(
            x_edges,
            y_edges,
            loss_grid,
            cmap="magma",
            shading="auto",
        )
        ax.set_title("Mean Loss per Grid Cell")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_xlim(x_range[0], x_range[1])
        ax.set_ylim(y_range[0], y_range[1])
        fig.colorbar(mesh, ax=ax, label="Mean loss")
        loss_heatmap_path = args.output.with_name(f"{args.output.stem}_loss_heatmap.png")
        fig.savefig(loss_heatmap_path, dpi=200)
        plt.close(fig)
        print(f"  Wrote loss heatmap to: {loss_heatmap_path}")

        if success_rate is not None and total_counts is not None:
            mask = np.isfinite(loss_grid) & np.isfinite(success_rate) & (total_counts > 0)
            flat_loss = loss_grid[mask]
            flat_succ = success_rate[mask]
            pearson_loss = float(np.corrcoef(flat_loss, flat_succ)[0, 1]) if flat_loss.size > 1 else float("nan")
            spearman_loss = _spearmanr(flat_loss, flat_succ)
            pearson_mastery = float(np.corrcoef(-flat_loss, flat_succ)[0, 1]) if flat_loss.size > 1 else float("nan")
            spearman_mastery = _spearmanr(-flat_loss, flat_succ)
            weights = total_counts[mask].astype(np.float64)
            w_pearson_loss = _weighted_corr(flat_loss, flat_succ, weights)
            w_spearman_loss = _weighted_corr(_rankdata(flat_loss), _rankdata(flat_succ), weights)
            w_pearson_mastery = _weighted_corr(-flat_loss, flat_succ, weights)
            w_spearman_mastery = _weighted_corr(_rankdata(-flat_loss), _rankdata(flat_succ), weights)

            min_pearson_loss = float("nan")
            min_spearman_loss = float("nan")
            min_pearson_mastery = float("nan")
            min_spearman_mastery = float("nan")
            if args.min_cell_samples > 1:
                min_mask = mask & (total_counts >= args.min_cell_samples)
                min_loss = loss_grid[min_mask]
                min_succ = success_rate[min_mask]
                min_pearson_loss = float(np.corrcoef(min_loss, min_succ)[0, 1]) if min_loss.size > 1 else float("nan")
                min_spearman_loss = _spearmanr(min_loss, min_succ)
                min_pearson_mastery = (
                    float(np.corrcoef(-min_loss, min_succ)[0, 1]) if min_loss.size > 1 else float("nan")
                )
                min_spearman_mastery = _spearmanr(-min_loss, min_succ)

            print("Grid loss analysis:")
            print(f"  Cells with data: {flat_loss.size}")
            if flat_loss.size > 0:
                print(f"  Loss mean:       min={float(np.nanmin(flat_loss)):.6g}, max={float(np.nanmax(flat_loss)):.6g}")
                print(f"  Success rate:    min={float(np.nanmin(flat_succ)):.3f}, max={float(np.nanmax(flat_succ)):.3f}")
            print(f"  Pearson(loss, success_rate) = {pearson_loss:.4f}")
            print(f"  Spearman(loss, success_rate) = {spearman_loss:.4f}")
            print(f"  Pearson(-loss, success_rate) = {pearson_mastery:.4f}")
            print(f"  Spearman(-loss, success_rate) = {spearman_mastery:.4f}")
            print(f"  Weighted Pearson(loss, success_rate) = {w_pearson_loss:.4f}")
            print(f"  Weighted Spearman(loss, success_rate) = {w_spearman_loss:.4f}")
            print(f"  Weighted Pearson(-loss, success_rate) = {w_pearson_mastery:.4f}")
            print(f"  Weighted Spearman(-loss, success_rate) = {w_spearman_mastery:.4f}")
            if args.min_cell_samples > 1:
                print(
                    f"  Filtered (n>={args.min_cell_samples}) Pearson(loss, success_rate) = {min_pearson_loss:.4f}"
                )
                print(
                    f"  Filtered (n>={args.min_cell_samples}) Spearman(loss, success_rate) = {min_spearman_loss:.4f}"
                )
                print(
                    f"  Filtered (n>={args.min_cell_samples}) Pearson(-loss, success_rate) = {min_pearson_mastery:.4f}"
                )
                print(
                    f"  Filtered (n>={args.min_cell_samples}) Spearman(-loss, success_rate) = {min_spearman_mastery:.4f}"
                )

            loss_metrics_path = args.output.with_name(f"{args.output.stem}_loss_grid_metrics.csv")
            with loss_metrics_path.open("w") as f:
                f.write("row,col,mean_loss,success_rate,total\n")
                for row in range(loss_grid.shape[0]):
                    for col in range(loss_grid.shape[1]):
                        mean_val = loss_grid[row, col]
                        succ_val = success_rate[row, col]
                        total = total_counts[row, col]
                        if not np.isfinite(mean_val) and total == 0:
                            continue
                        f.write(f"{row},{col},{mean_val:.8g},{succ_val:.6g},{int(total)}\n")
            print(f"  Wrote loss grid metrics to: {loss_metrics_path}")

            # Uncertainty: combine high loss and high failure (1 - success rate).
            fail_rate = 1.0 - success_rate
            fail_mask = mask & np.isfinite(fail_rate)
            if np.any(fail_mask):
                loss_vals = loss_grid[fail_mask]
                fail_vals = fail_rate[fail_mask]
                loss_norm = _minmax_norm(loss_vals)
                fail_norm = _minmax_norm(fail_vals)
                uncertainty_vals = 0.5 * (loss_norm + fail_norm)

                uncertainty_grid = np.full_like(loss_grid, np.nan, dtype=np.float64)
                uncertainty_grid[fail_mask] = uncertainty_vals

                fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
                x_edges, y_edges = loss_edges
                mesh = ax.pcolormesh(
                    x_edges,
                    y_edges,
                    uncertainty_grid,
                    cmap=args.uncertainty_cmap,
                    shading="auto",
                )
                if overlay_positions:
                    ax.scatter(
                        [p.x for p in overlay_positions],
                        [p.y for p in overlay_positions],
                        s=150,
                        c="#D500F9",
                        alpha=0.9,
                        marker="*",
                        edgecolors="black",
                        linewidths=1.0,
                        label=f"Collection points ({len(overlay_positions)})",
                    )
                ax.set_title("Uncertainty (loss + fail rate)")
                ax.set_xlabel("x")
                ax.set_ylabel("y")
                ax.set_xlim(x_range[0], x_range[1])
                ax.set_ylim(y_range[0], y_range[1])
                if overlay_positions:
                    ax.legend(loc="upper right")
                fig.colorbar(mesh, ax=ax, label="Uncertainty (0-1)")
                uncertainty_heatmap_path = args.output.with_name(
                    f"{args.output.stem}_uncertainty_heatmap.png"
                )
                fig.savefig(uncertainty_heatmap_path, dpi=200)
                plt.close(fig)
                print(f"  Wrote uncertainty heatmap to: {uncertainty_heatmap_path}")

                rows, cols = np.where(fail_mask)
                order = np.argsort(uncertainty_vals)[::-1]
                top_k = min(10, order.size)
                print("  Top uncertainty cells (loss + fail rate):")
                for idx in order[:top_k]:
                    row = int(rows[idx])
                    col = int(cols[idx])
                    x0, x1 = x_edges[col], x_edges[col + 1]
                    y0, y1 = y_edges[row], y_edges[row + 1]
                    print(
                        f"  ({row:02d},{col:02d}) unc={uncertainty_vals[idx]:.3f} "
                        f"loss={loss_vals[idx]:.6g} succ={success_rate[row, col]:.3f} "
                        f"total={int(total_counts[row, col])} x[{x0:.3f},{x1:.3f}] y[{y0:.3f},{y1:.3f}]"
                    )

                uncertainty_path = args.output.with_name(
                    f"{args.output.stem}_uncertainty_grid_metrics.csv"
                )
                with uncertainty_path.open("w") as f:
                    f.write("row,col,uncertainty,mean_loss,success_rate,fail_rate,total\n")
                    for row in range(loss_grid.shape[0]):
                        for col in range(loss_grid.shape[1]):
                            if total_counts[row, col] <= 0:
                                continue
                            unc_val = uncertainty_grid[row, col]
                            if not np.isfinite(unc_val):
                                continue
                            f.write(
                                f"{row},{col},{unc_val:.8g},"
                                f"{loss_grid[row, col]:.8g},"
                                f"{success_rate[row, col]:.6g},"
                                f"{fail_rate[row, col]:.6g},"
                                f"{int(total_counts[row, col])}\n"
                            )
                print(f"  Wrote uncertainty grid metrics to: {uncertainty_path}")

            if flat_loss.size > 0:
                fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
                ax.scatter(-flat_loss, flat_succ, s=40, alpha=0.8)
                ax.set_xlabel("-Mean loss (cell)")
                ax.set_ylabel("Success rate (cell)")
                ax.set_title("Mastery Proxy (-loss) vs Success Rate")
                ax.grid(True, linestyle="--", alpha=0.3)
                ax.text(
                    0.02,
                    0.98,
                    f"Pearson: {pearson_mastery:.3f}\nSpearman: {spearman_mastery:.3f}\nCells: {flat_loss.size}",
                    transform=ax.transAxes,
                    va="top",
                    ha="left",
                    fontsize=9,
                    bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
                )
                scatter_path = args.output.with_name(f"{args.output.stem}_loss_vs_success.png")
                fig.savefig(scatter_path, dpi=200)
                plt.close(fig)
                print(f"  Wrote loss vs success scatter to: {scatter_path}")

    grid_points = test_points
    if args.grid_output:
        args.grid_output.write_text(
            "x,y\n" + "\n".join(f"{x:.3f},{y:.3f}" for x, y in grid_points)
        )
        print(f"Wrote {len(grid_points)} grid test points to {args.grid_output}")
    else:
        print("Grid test points (x,y):")
        for x, y in grid_points:
            print(f"{x:.3f},{y:.3f}")

    if recommendations and args.collection_output:
        args.collection_output.write_text(
            "x,y,row,col,score,center_x,center_y\n"
            + "\n".join(
                f"{rec.sampled_x:.5f},{rec.sampled_y:.5f},{rec.row},{rec.col},"
                f"{rec.score:.6f},{rec.center_x:.5f},{rec.center_y:.5f}"
                for rec in recommendations
            )
        )
        print(f"Wrote {len(recommendations)} recommended collection points to {args.collection_output}")

    print(f"Parsed {len(positions)} episode folders from {args.root}")
    print(f"x range: {min_x:.3f} – {max_x:.3f}")
    print(f"y range: {min_y:.3f} – {max_y:.3f}")
    print(f"Figure written to: {args.output}")


if __name__ == "__main__":
    main()
