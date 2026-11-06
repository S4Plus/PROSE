#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

try:
    import pyarrow.parquet as pq
except Exception as exc:  # pragma: no cover - import guard
    raise SystemExit(f"pyarrow is required to run this script: {exc}")


def load_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def list_parquet_files(data_root: Path) -> Dict[int, Path]:
    parquet_map: Dict[int, Path] = {}
    for path in data_root.rglob("episode_*.parquet"):
        name = path.name
        if not name.startswith("episode_"):
            continue
        idx_str = name.replace("episode_", "").replace(".parquet", "")
        if not idx_str.isdigit():
            continue
        parquet_map[int(idx_str)] = path
    return dict(sorted(parquet_map.items()))


def summarize_lengths(episodes: List[dict]) -> Dict[str, float]:
    lengths = np.array([e.get("length", 0) for e in episodes], dtype=np.int64)
    return {
        "count": float(len(lengths)),
        "min": float(lengths.min()) if len(lengths) else 0.0,
        "max": float(lengths.max()) if len(lengths) else 0.0,
        "mean": float(lengths.mean()) if len(lengths) else 0.0,
        "std": float(lengths.std()) if len(lengths) else 0.0,
    }


def compute_global_stats(
    parquet_map: Dict[int, Path],
    fields: Iterable[str],
) -> Tuple[dict, dict, dict, dict]:
    dim = 14
    sums = {field: np.zeros(dim, dtype=np.float64) for field in fields}
    sums2 = {field: np.zeros(dim, dtype=np.float64) for field in fields}
    mins = {field: np.full(dim, np.inf, dtype=np.float64) for field in fields}
    maxs = {field: np.full(dim, -np.inf, dtype=np.float64) for field in fields}
    counts = {field: 0 for field in fields}

    action_state_abs_sum = np.zeros(dim, dtype=np.float64)
    action_state_count = 0

    # For correlation between action and state per joint
    corr_sum_x = np.zeros(dim, dtype=np.float64)
    corr_sum_y = np.zeros(dim, dtype=np.float64)
    corr_sum_x2 = np.zeros(dim, dtype=np.float64)
    corr_sum_y2 = np.zeros(dim, dtype=np.float64)
    corr_sum_xy = np.zeros(dim, dtype=np.float64)

    nan_counts = {field: 0 for field in fields}

    for episode_index, parquet_path in parquet_map.items():
        table = pq.read_table(parquet_path, columns=list(fields))
        for field in fields:
            col = table.column(field).to_pylist()
            arr = np.asarray(col, dtype=np.float64)
            if arr.ndim != 2 or arr.shape[1] != dim:
                raise ValueError(
                    f"Unexpected shape for {field} in {parquet_path}: {arr.shape}"
                )
            if np.isnan(arr).any() or np.isinf(arr).any():
                nan_counts[field] += int(np.isnan(arr).sum() + np.isinf(arr).sum())
            sums[field] += arr.sum(axis=0)
            sums2[field] += (arr ** 2).sum(axis=0)
            mins[field] = np.minimum(mins[field], arr.min(axis=0))
            maxs[field] = np.maximum(maxs[field], arr.max(axis=0))
            counts[field] += arr.shape[0]

        if "action" in fields and "observation.state" in fields:
            action = np.asarray(table.column("action").to_pylist(), dtype=np.float64)
            state = np.asarray(
                table.column("observation.state").to_pylist(), dtype=np.float64
            )
            action_state_abs_sum += np.abs(action - state).sum(axis=0)
            action_state_count += action.shape[0]

            corr_sum_x += action.sum(axis=0)
            corr_sum_y += state.sum(axis=0)
            corr_sum_x2 += (action ** 2).sum(axis=0)
            corr_sum_y2 += (state ** 2).sum(axis=0)
            corr_sum_xy += (action * state).sum(axis=0)

    stats = {}
    for field in fields:
        count = counts[field]
        mean = sums[field] / max(count, 1)
        var = sums2[field] / max(count, 1) - mean ** 2
        var = np.maximum(var, 0.0)
        std = np.sqrt(var)
        stats[field] = {
            "count": int(count),
            "min": mins[field].tolist(),
            "max": maxs[field].tolist(),
            "mean": mean.tolist(),
            "std": std.tolist(),
            "nan_or_inf": int(nan_counts[field]),
        }

    mae = action_state_abs_sum / max(action_state_count, 1)
    corr_num = corr_sum_xy - (corr_sum_x * corr_sum_y / max(action_state_count, 1))
    corr_den = np.sqrt(
        (corr_sum_x2 - (corr_sum_x ** 2) / max(action_state_count, 1))
        * (corr_sum_y2 - (corr_sum_y ** 2) / max(action_state_count, 1))
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = np.where(corr_den == 0, np.nan, corr_num / corr_den)

    return stats, mae, corr, nan_counts


def analyze_episode_stats(stats_rows: List[dict]) -> Dict[str, object]:
    # Compute per-episode max velocity norm and basic outlier detection.
    max_vel_norm = []
    for row in stats_rows:
        vel = row["stats"].get("observation.velocity", {})
        max_vals = np.asarray(vel.get("max", []), dtype=np.float64)
        if max_vals.size == 0:
            max_vel_norm.append(float("nan"))
            continue
        max_vel_norm.append(float(np.linalg.norm(max_vals)))

    arr = np.asarray(max_vel_norm, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size:
        med = float(np.median(finite))
        mad = float(np.median(np.abs(finite - med)))
        # Robust z-score using MAD
        if mad == 0:
            z = np.zeros_like(arr)
        else:
            z = 0.6745 * (arr - med) / mad
        outliers = [
            int(stats_rows[i]["episode_index"]) for i, zi in enumerate(z) if zi > 3.5
        ]
    else:
        med = mad = 0.0
        outliers = []

    return {
        "max_velocity_norm": max_vel_norm,
        "max_velocity_norm_median": med,
        "max_velocity_norm_mad": mad,
        "velocity_outlier_episodes": outliers,
    }


def _compute_image_metrics_from_frames(frame_paths: List[Path]) -> Dict[str, List[float]]:
    metrics: Dict[str, List[float]] = {"brightness": [], "sharpness": []}
    try:
        import cv2
    except Exception:
        return metrics

    for frame_path in frame_paths:
        frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if frame is None:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        metrics["brightness"].append(float(gray.mean()) / 255.0)
        metrics["sharpness"].append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))
    return metrics


def _extract_frames_ffmpeg(
    video_path: Path,
    frame_indices: List[int],
    out_dir: Path,
) -> List[Path]:
    if not frame_indices:
        return []
    safe_mkdir(out_dir)
    select_expr = "+".join([f"eq(n\\,{idx})" for idx in frame_indices])
    out_pattern = str(out_dir / "frame_%04d.png")
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-hwaccel",
        "none",
        "-i",
        str(video_path),
        "-vf",
        f"select={select_expr}",
        "-vsync",
        "0",
        "-vframes",
        str(len(frame_indices)),
        out_pattern,
    ]
    try:
        subprocess.run(cmd, check=False)
    except FileNotFoundError:
        return []
    return sorted(out_dir.glob("frame_*.png"))


def sample_video_metrics(
    videos_root: Path,
    episode_indices: List[int],
    frame_indices: List[int],
    method: str = "opencv",
) -> Dict[str, Dict[str, List[float]]]:
    metrics: Dict[str, Dict[str, List[float]]] = {}
    has_cv2 = False
    if method == "opencv":
        try:
            import cv2  # noqa: F401
            has_cv2 = True
        except Exception:
            has_cv2 = False

    video_keys = [
        "observation.images.cam_high",
        "observation.images.cam_left_wrist",
        "observation.images.cam_right_wrist",
    ]

    for key in video_keys:
        metrics[key] = {"brightness": [], "sharpness": []}
        for ep in episode_indices:
            video_path = videos_root / key / f"episode_{ep:06d}.mp4"
            if not video_path.exists():
                continue
            if method == "opencv" and has_cv2:
                import cv2

                cap = cv2.VideoCapture(str(video_path))
                if cap.isOpened():
                    for fi in frame_indices:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
                        ok, frame = cap.read()
                        if not ok or frame is None:
                            continue
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        brightness = float(gray.mean()) / 255.0
                        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
                        metrics[key]["brightness"].append(brightness)
                        metrics[key]["sharpness"].append(sharpness)
                    cap.release()

            if method == "ffmpeg" or not metrics[key]["brightness"]:
                frame_dir = videos_root / "_frames_tmp" / key / f"episode_{ep:06d}"
                frame_paths = _extract_frames_ffmpeg(video_path, frame_indices, frame_dir)
                frame_metrics = _compute_image_metrics_from_frames(frame_paths)
                metrics[key]["brightness"].extend(frame_metrics["brightness"])
                metrics[key]["sharpness"].extend(frame_metrics["sharpness"])
    return metrics


def save_plots(
    out_dir: Path,
    joint_names: List[str],
    lengths: List[int],
    mae: np.ndarray,
    corr: np.ndarray,
    stats: dict,
    vel_info: dict,
    image_metrics: dict,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    safe_mkdir(out_dir)

    # Episode length distribution
    plt.figure(figsize=(6, 4))
    plt.hist(lengths, bins=20, color="#4C78A8")
    plt.title("Episode length distribution")
    plt.xlabel("length")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(out_dir / "episode_lengths.png", dpi=160)
    plt.close()

    # Action vs state MAE per joint
    plt.figure(figsize=(10, 4))
    plt.bar(joint_names, mae, color="#F58518")
    plt.xticks(rotation=45, ha="right")
    plt.title("Action vs state MAE per joint")
    plt.ylabel("MAE")
    plt.tight_layout()
    plt.savefig(out_dir / "action_state_mae.png", dpi=160)
    plt.close()

    # Action vs state correlation per joint
    plt.figure(figsize=(10, 4))
    plt.bar(joint_names, corr, color="#54A24B")
    plt.xticks(rotation=45, ha="right")
    plt.title("Action vs state correlation per joint")
    plt.ylabel("corr")
    plt.ylim(-1.0, 1.0)
    plt.tight_layout()
    plt.savefig(out_dir / "action_state_corr.png", dpi=160)
    plt.close()

    # Action range per joint
    action_stats = stats.get("action", {})
    if action_stats:
        min_vals = np.asarray(action_stats.get("min", []), dtype=np.float64)
        max_vals = np.asarray(action_stats.get("max", []), dtype=np.float64)
        ranges = max_vals - min_vals
        plt.figure(figsize=(10, 4))
        plt.bar(joint_names, ranges, color="#B279A2")
        plt.xticks(rotation=45, ha="right")
        plt.title("Action range per joint")
        plt.ylabel("range")
        plt.tight_layout()
        plt.savefig(out_dir / "action_ranges.png", dpi=160)
        plt.close()

    # Velocity max norm per episode
    vmax = np.asarray(vel_info.get("max_velocity_norm", []), dtype=np.float64)
    if vmax.size:
        plt.figure(figsize=(8, 4))
        plt.plot(vmax, color="#4C78A8")
        plt.title("Max velocity norm per episode")
        plt.xlabel("episode index")
        plt.ylabel("||v_max||")
        plt.tight_layout()
        plt.savefig(out_dir / "velocity_max_norm.png", dpi=160)
        plt.close()

    # Image metrics histograms
    if image_metrics:
        for key, values in image_metrics.items():
            for metric, arr in values.items():
                if not arr:
                    continue
                plt.figure(figsize=(6, 4))
                plt.hist(arr, bins=30, color="#72B7B2")
                plt.title(f"{key} {metric} distribution")
                plt.xlabel(metric)
                plt.ylabel("count")
                plt.tight_layout()
                safe_key = key.replace("/", "_").replace(".", "_")
                plt.savefig(out_dir / f"{safe_key}_{metric}.png", dpi=160)
                plt.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Dataset quality analysis")
    parser.add_argument(
        "--dataset",
        default="/workspace/liushuo/cupid/act/swk/six_object_pick_conveyor_first2",
        help="Dataset root directory",
    )
    parser.add_argument(
        "--out",
        default="/workspace/liushuo/cupid/act/swk/outputs/dataset_quality/six_object_pick_conveyor_first2",
        help="Output directory for reports and plots",
    )
    parser.add_argument(
        "--video-samples",
        type=int,
        default=8,
        help="Number of episodes to sample for image metrics",
    )
    parser.add_argument(
        "--image-metrics-method",
        choices=["opencv", "ffmpeg"],
        default="opencv",
        help="Method to sample image metrics from videos.",
    )
    args = parser.parse_args()

    dataset_root = Path(args.dataset)
    out_dir = Path(args.out)
    safe_mkdir(out_dir)

    info = json.loads((dataset_root / "meta" / "info.json").read_text())
    episodes = load_jsonl(dataset_root / "meta" / "episodes.jsonl")
    stats_rows = load_jsonl(dataset_root / "meta" / "episodes_stats.jsonl")

    parquet_map = list_parquet_files(dataset_root / "data")
    fields = [
        "observation.state",
        "action",
        "observation.velocity",
        "observation.effort",
    ]

    stats, mae, corr, nan_counts = compute_global_stats(parquet_map, fields)
    vel_info = analyze_episode_stats(stats_rows)

    lengths = [int(e.get("length", 0)) for e in episodes]

    joint_names = info["features"]["action"]["names"][0]

    # Sample image metrics for quick quality checks
    sample_eps = [e.get("episode_index", 0) for e in episodes[: args.video_samples]]
    frame_indices = [0, 50, 150, 250]
    image_metrics = sample_video_metrics(
        dataset_root / "videos" / "chunk-000",
        sample_eps,
        frame_indices,
        method=args.image_metrics_method,
    )

    # Save summary
    summary = {
        "dataset": str(dataset_root),
        "episodes": int(info.get("total_episodes", len(episodes))),
        "frames": int(info.get("total_frames", 0)),
        "fps": float(info.get("fps", 0)),
        "episode_length": summarize_lengths(episodes),
        "nan_counts": nan_counts,
        "stats": stats,
        "action_state_mae": mae.tolist(),
        "action_state_corr": corr.tolist(),
        "velocity": vel_info,
        "image_metrics": image_metrics,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    save_plots(out_dir, joint_names, lengths, mae, corr, stats, vel_info, image_metrics)
    print(f"Saved report to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
