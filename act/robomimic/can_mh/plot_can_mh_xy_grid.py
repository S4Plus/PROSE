#!/usr/bin/env python3
"""
Visualize can placement XY coordinates for PickPlaceCan / can_mh with grid partition.

Supports three data sources:
1) case json (default): mimicgen eval config with "cases" entries that contain can.x / can.y
2) robomimic hdf5 datagen poses: data/*/datagen_info/object_poses/<object_name>
3) robomimic hdf5 states: read x / y from selected indices in data/*/states[frame_index]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np


def _safe_edges(vmin: float, vmax: float, bins: int) -> np.ndarray:
    if bins < 1:
        raise ValueError("bins must be >= 1")
    if np.isclose(vmin, vmax):
        eps = 1e-6
        vmin -= eps
        vmax += eps
    return np.linspace(vmin, vmax, bins + 1, dtype=float)


def _load_from_case_json(case_json: Path, object_name: str) -> tuple[np.ndarray, np.ndarray, list[str], tuple[float, float] | None, tuple[float, float] | None]:
    payload = json.loads(case_json.read_text(encoding="utf-8"))
    cases = payload.get("cases", [])

    xs: list[float] = []
    ys: list[float] = []
    tags: list[str] = []
    for item in cases:
        obj = item.get(object_name)
        if not isinstance(obj, dict):
            continue
        x = obj.get("x")
        y = obj.get("y")
        if x is None or y is None:
            continue
        xs.append(float(x))
        ys.append(float(y))
        tags.append(str(item.get("tag", "unknown")))

    if not xs:
        raise RuntimeError(f"No '{object_name}.x/y' found in {case_json}")

    bx = None
    by = None
    bounds = payload.get("bounds", {})
    obj_bounds = bounds.get(object_name, {}) if isinstance(bounds, dict) else {}
    if isinstance(obj_bounds, dict):
        x_arr = obj_bounds.get("x")
        y_arr = obj_bounds.get("y")
        if isinstance(x_arr, list) and len(x_arr) == 2:
            bx = (float(x_arr[0]), float(x_arr[1]))
        if isinstance(y_arr, list) and len(y_arr) == 2:
            by = (float(y_arr[0]), float(y_arr[1]))

    return np.asarray(xs), np.asarray(ys), tags, bx, by


def _extract_pose_xy(pose_arr: np.ndarray, frame_index: int) -> tuple[float, float]:
    # Common case: (T, 4, 4) pose matrices.
    if pose_arr.ndim == 3 and pose_arr.shape[-2:] == (4, 4):
        fi = int(np.clip(frame_index, 0, pose_arr.shape[0] - 1))
        pose = pose_arr[fi]
        return float(pose[0, 3]), float(pose[1, 3])

    # Fallback: flattened pose, e.g. (T, 16) or (16,).
    if pose_arr.ndim == 2 and pose_arr.shape[1] == 16:
        fi = int(np.clip(frame_index, 0, pose_arr.shape[0] - 1))
        flat = pose_arr[fi].reshape(4, 4)
        return float(flat[0, 3]), float(flat[1, 3])
    if pose_arr.ndim == 1 and pose_arr.size == 16:
        flat = pose_arr.reshape(4, 4)
        return float(flat[0, 3]), float(flat[1, 3])

    raise RuntimeError(f"Unsupported pose shape: {pose_arr.shape}")


def _iter_hdf5_demos(h5_file) -> Iterable[str]:
    data_group = h5_file.get("data")
    if data_group is None:
        return []
    return sorted(list(data_group.keys()))


def _load_from_hdf5_datagen_pose(hdf5_path: Path, object_name: str, frame_index: int) -> tuple[np.ndarray, np.ndarray]:
    try:
        import h5py  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("h5py is required for --hdf5 mode") from exc

    xs: list[float] = []
    ys: list[float] = []
    with h5py.File(hdf5_path, "r") as f:
        for demo_name in _iter_hdf5_demos(f):
            pose_ds = f.get(f"data/{demo_name}/datagen_info/object_poses/{object_name}")
            if pose_ds is None:
                continue
            arr = np.asarray(pose_ds)
            x, y = _extract_pose_xy(arr, frame_index=frame_index)
            xs.append(x)
            ys.append(y)

    if not xs:
        raise RuntimeError(f"No object poses found at data/*/datagen_info/object_poses/{object_name} in {hdf5_path}")
    return np.asarray(xs), np.asarray(ys)


def _load_from_hdf5_states(
    hdf5_path: Path,
    frame_index: int,
    state_x_index: int,
    state_y_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    try:
        import h5py  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("h5py is required for --hdf5 mode") from exc

    xs: list[float] = []
    ys: list[float] = []
    with h5py.File(hdf5_path, "r") as f:
        for demo_name in _iter_hdf5_demos(f):
            states_ds = f.get(f"data/{demo_name}/states")
            if states_ds is None:
                continue
            states = np.asarray(states_ds)
            if states.ndim != 2:
                continue
            fi = int(np.clip(frame_index, 0, states.shape[0] - 1))
            if state_x_index < 0 or state_y_index < 0:
                raise ValueError("state_x_index/state_y_index must be >= 0")
            if state_x_index >= states.shape[1] or state_y_index >= states.shape[1]:
                raise RuntimeError(
                    f"states dim={states.shape[1]} is smaller than requested indices "
                    f"({state_x_index}, {state_y_index}) in demo {demo_name}"
                )
            xs.append(float(states[fi, state_x_index]))
            ys.append(float(states[fi, state_y_index]))

    if not xs:
        raise RuntimeError(f"No states found at data/*/states in {hdf5_path}")
    return np.asarray(xs), np.asarray(ys)


def _plot_with_grid(
    x: np.ndarray,
    y: np.ndarray,
    out_png: Path,
    grid_x: int,
    grid_y: int,
    x_bounds: tuple[float, float] | None,
    y_bounds: tuple[float, float] | None,
    title: str,
    tags: list[str] | None = None,
    out_grid_csv: Path | None = None,
) -> None:
    if x.shape[0] != y.shape[0]:
        raise ValueError("x and y must have same length")

    xmin = float(np.min(x)) if x_bounds is None else float(x_bounds[0])
    xmax = float(np.max(x)) if x_bounds is None else float(x_bounds[1])
    ymin = float(np.min(y)) if y_bounds is None else float(y_bounds[0])
    ymax = float(np.max(y)) if y_bounds is None else float(y_bounds[1])

    x_edges = _safe_edges(xmin, xmax, grid_x)
    y_edges = _safe_edges(ymin, ymax, grid_y)
    hist, _, _ = np.histogram2d(x, y, bins=[x_edges, y_edges])

    fig, ax = plt.subplots(figsize=(8, 8), dpi=160)
    if tags and len(tags) == len(x):
        tag_arr = np.asarray(tags)
        for t in sorted(np.unique(tag_arr)):
            mask = tag_arr == t
            ax.scatter(x[mask], y[mask], s=36, alpha=0.85, label=str(t))
        ax.legend(loc="upper right", frameon=True)
    else:
        ax.scatter(x, y, s=36, alpha=0.85, label="can")

    # Grid lines.
    for xv in x_edges:
        ax.axvline(float(xv), color="gray", linewidth=0.8, alpha=0.45)
    for yv in y_edges:
        ax.axhline(float(yv), color="gray", linewidth=0.8, alpha=0.45)

    # Per-cell count annotation.
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    for i, xc in enumerate(x_centers):
        for j, yc in enumerate(y_centers):
            c = int(hist[i, j])
            if c > 0:
                ax.text(
                    float(xc),
                    float(yc),
                    str(c),
                    color="black",
                    fontsize=8,
                    ha="center",
                    va="center",
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.55, pad=1.2),
                )

    ax.set_title(title)
    ax.set_xlabel("can x (m)")
    ax.set_ylabel("can y (m)")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(float(x_edges[0]), float(x_edges[-1]))
    ax.set_ylim(float(y_edges[0]), float(y_edges[-1]))
    ax.grid(False)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)

    if out_grid_csv is not None:
        out_grid_csv.parent.mkdir(parents=True, exist_ok=True)
        with out_grid_csv.open("w", encoding="utf-8") as f:
            f.write("ix,iy,x_left,x_right,y_bottom,y_top,count\n")
            for i in range(grid_x):
                for j in range(grid_y):
                    f.write(
                        f"{i},{j},{x_edges[i]:.8f},{x_edges[i+1]:.8f},"
                        f"{y_edges[j]:.8f},{y_edges[j+1]:.8f},{int(hist[i, j])}\n"
                    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize can_mh can XY placement with grid partition.")
    parser.add_argument(
        "--case-json",
        type=Path,
        default=Path("/workspace/liushuo/cupid/mimicgen/eval/test/PickPlaceCan.json"),
        help="Path to PickPlaceCan case json with bounds/cases.",
    )
    parser.add_argument(
        "--hdf5",
        type=Path,
        default=None,
        help="Optional robomimic hdf5 dataset.",
    )
    parser.add_argument(
        "--hdf5-mode",
        type=str,
        default="auto",
        choices=("auto", "datagen_pose", "states"),
        help="How to read XY from hdf5. auto: datagen_pose first, then states.",
    )
    parser.add_argument("--object-name", type=str, default="can", help="Object key name in json / hdf5.")
    parser.add_argument("--frame-index", type=int, default=0, help="Frame index in object pose trajectory for hdf5 mode.")
    parser.add_argument(
        "--state-x-index",
        type=int,
        default=31,
        help="x index in states vector when --hdf5-mode states (PickPlaceCan can default=31).",
    )
    parser.add_argument(
        "--state-y-index",
        type=int,
        default=32,
        help="y index in states vector when --hdf5-mode states (PickPlaceCan can default=32).",
    )
    parser.add_argument("--grid-x", type=int, default=7, help="Number of grid cells along x.")
    parser.add_argument("--grid-y", type=int, default=7, help="Number of grid cells along y.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/workspace/liushuo/cupid/act/robomimic/can_mh/can_mh_can_xy_grid.png"),
        help="Output image path.",
    )
    parser.add_argument(
        "--grid-csv",
        type=Path,
        default=Path("/workspace/liushuo/cupid/act/robomimic/can_mh/can_mh_can_xy_grid_counts.csv"),
        help="Optional per-grid count csv output path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.hdf5 is not None:
        used_mode = args.hdf5_mode
        if args.hdf5_mode == "datagen_pose":
            x, y = _load_from_hdf5_datagen_pose(args.hdf5, object_name=args.object_name, frame_index=args.frame_index)
        elif args.hdf5_mode == "states":
            x, y = _load_from_hdf5_states(
                args.hdf5,
                frame_index=args.frame_index,
                state_x_index=args.state_x_index,
                state_y_index=args.state_y_index,
            )
        else:
            try:
                x, y = _load_from_hdf5_datagen_pose(
                    args.hdf5, object_name=args.object_name, frame_index=args.frame_index
                )
                used_mode = "datagen_pose"
            except Exception:
                x, y = _load_from_hdf5_states(
                    args.hdf5,
                    frame_index=args.frame_index,
                    state_x_index=args.state_x_index,
                    state_y_index=args.state_y_index,
                )
                used_mode = "states"
        tags = None
        x_bounds = None
        y_bounds = None
        title = f"can_mh {args.object_name} XY from hdf5/{used_mode} ({len(x)} demos)"
    else:
        x, y, tags, x_bounds, y_bounds = _load_from_case_json(args.case_json, object_name=args.object_name)
        title = f"PickPlaceCan {args.object_name} XY ({len(x)} cases)"

    _plot_with_grid(
        x=x,
        y=y,
        out_png=args.output,
        grid_x=args.grid_x,
        grid_y=args.grid_y,
        x_bounds=x_bounds,
        y_bounds=y_bounds,
        title=title,
        tags=tags,
        out_grid_csv=args.grid_csv,
    )
    print(f"[done] saved figure: {args.output}")
    print(f"[done] saved grid counts: {args.grid_csv}")


if __name__ == "__main__":
    main()
