#!/usr/bin/env python3
"""可视化 stack_cube_205 在 mysys 筛选时被去掉的轨迹和片段。"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess

import cv2
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="可视化 stack_cube_205 joint episode-window 筛选时被移除的轨迹和局部片段。")
    parser.add_argument(
        "--selection-dir",
        type=Path,
        default=Path("openpi-script/output/joint_episode_window_serial/stack_cube_205_joint_episode_window/selection"),
        help="selection_summary.json、episode_scores.csv、window_scores.csv、joint_episode_window_selection.npz 所在目录。",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/workspace/liushuo/dataset/stack_cube_205"),
        help="stack_cube_205 LeRobot 数据集根目录。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("tmp/stack_cube_205_mysys_filter_viz"),
        help="可视化输出目录。",
    )
    parser.add_argument(
        "--camera",
        type=str,
        default="fixed_camera",
        choices=("fixed_camera", "wrist_camera"),
        help="抽帧使用的视频视角。",
    )
    parser.add_argument("--heatmap-bins", type=int, default=200, help="全局热力图每条 episode 的归一化时间桶数。")
    parser.add_argument("--top-k-dropped", type=int, default=8, help="整条被删除轨迹的预览条数。")
    parser.add_argument("--top-k-partial", type=int, default=10, help="局部片段被删除 episode 的预览条数。")
    parser.add_argument("--frames-per-episode", type=int, default=4, help="每条 episode 预览抽取的关键帧数量。")
    parser.add_argument("--max-preview-rows", type=int, default=10, help="单张 contact sheet 最多绘制的 episode 行数。")
    return parser.parse_args()


def _load_selection_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def _contiguous_false_segments(keep_mask: np.ndarray) -> list[tuple[int, int]]:
    removed = ~keep_mask
    if removed.size == 0:
        return []
    padded = np.pad(removed.astype(np.int8), (1, 1), constant_values=0)
    changes = np.diff(padded)
    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0]
    return [(int(start), int(end)) for start, end in zip(starts, ends, strict=True) if end > start]


def _find_video_path(dataset_root: Path, camera: str, episode: int) -> Path | None:
    pattern = f"videos/chunk-*/observation.images.{camera}/episode_{episode:06d}.mp4"
    matches = sorted(dataset_root.glob(pattern))
    return matches[0] if matches else None


def _find_episode_parquet(dataset_root: Path, episode: int) -> Path | None:
    pattern = f"data/chunk-*/episode_{episode:06d}.parquet"
    matches = sorted(dataset_root.glob(pattern))
    return matches[0] if matches else None


def _load_episode_length(dataset_root: Path, episode: int) -> int:
    parquet_path = _find_episode_parquet(dataset_root, episode)
    if parquet_path is None:
        return 0
    frame_df = pd.read_parquet(parquet_path, columns=["frame_index"])
    if frame_df.empty:
        return 0
    return int(frame_df["frame_index"].max()) + 1


def _read_video_frame(video_path: Path | None, frame_index: int) -> np.ndarray:
    if video_path is None or not video_path.exists():
        return np.full((240, 320, 3), 235, dtype=np.uint8)

    cmd = [
        "ffmpeg",
        "-loglevel",
        "error",
        "-threads",
        "1",
        "-i",
        str(video_path),
        "-vf",
        f"select=eq(n\\,{max(0, int(frame_index))})",
        "-vframes",
        "1",
        "-f",
        "image2pipe",
        "-vcodec",
        "png",
        "-",
    ]
    result = subprocess.run(cmd, check=False, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    if result.returncode != 0 or not result.stdout:
        return np.full((240, 320, 3), 235, dtype=np.uint8)
    decoded = cv2.imdecode(np.frombuffer(result.stdout, dtype=np.uint8), cv2.IMREAD_COLOR)
    if decoded is None:
        return np.full((240, 320, 3), 235, dtype=np.uint8)
    return cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)


def _sample_removed_frames(keep_mask: np.ndarray, frames_per_episode: int) -> list[int]:
    removed_indices = np.flatnonzero(~keep_mask)
    if removed_indices.size == 0:
        return []
    if removed_indices.size <= frames_per_episode:
        return [int(x) for x in removed_indices.tolist()]
    positions = np.linspace(0, removed_indices.size - 1, frames_per_episode, dtype=int)
    return [int(removed_indices[idx]) for idx in positions.tolist()]


def _normalize_removed_mask(keep_mask: np.ndarray, bins: int) -> np.ndarray:
    if keep_mask.size == 0:
        return np.ones((bins,), dtype=np.float32)
    xs = np.linspace(0, keep_mask.size, bins + 1)
    out = np.zeros((bins,), dtype=np.float32)
    removed = (~keep_mask).astype(np.float32)
    for i in range(bins):
        start = int(math.floor(xs[i]))
        end = int(math.ceil(xs[i + 1]))
        end = min(end, keep_mask.size)
        if end <= start:
            start = min(start, keep_mask.size - 1)
            end = min(start + 1, keep_mask.size)
        out[i] = float(np.mean(removed[start:end]))
    return out


def _build_episode_records(
    episode_scores: pd.DataFrame,
    window_scores: pd.DataFrame,
    dataset_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[int, np.ndarray]]:
    keep_masks: dict[int, np.ndarray] = {}
    segment_rows: list[dict[str, float | int | bool]] = []
    episode_rows: list[dict[str, float | int | bool]] = []

    window_group = {
        int(episode): group.sort_values(["start", "end"], kind="stable").reset_index(drop=True)
        for episode, group in window_scores.groupby("episode", sort=False)
    }

    for row in episode_scores.itertuples(index=False):
        episode = int(row.episode)
        episode_length = int(row.episode_length)
        if episode_length <= 0:
            episode_length = _load_episode_length(dataset_root, episode)
        keep_episode = bool(row.keep_episode)
        keep_mask = np.zeros((episode_length,), dtype=bool) if not keep_episode else np.zeros((episode_length,), dtype=bool)

        for _, window in window_group.get(episode, pd.DataFrame()).iterrows():
            if int(window["keep_window"]) != 1:
                continue
            start = max(0, int(window["start"]))
            end = min(episode_length, int(window["end"]))
            if end > start:
                keep_mask[start:end] = True

        if not keep_episode:
            keep_mask[:] = False

        keep_masks[episode] = keep_mask
        removed_segments = _contiguous_false_segments(keep_mask)
        removed_frames = int(np.count_nonzero(~keep_mask))
        removed_ratio = removed_frames / max(episode_length, 1)

        episode_rows.append(
            {
                "episode": episode,
                "keep_episode": int(keep_episode),
                "episode_joint_score": float(row.episode_joint_score),
                "episode_influence": float(row.episode_influence),
                "episode_quality": float(row.episode_quality),
                "episode_length": episode_length,
                "selected_windows": int(row.selected_windows),
                "num_windows": int(row.num_windows),
                "window_coverage_ratio": float(row.window_coverage_ratio),
                "removed_frames": removed_frames,
                "removed_ratio": removed_ratio,
                "num_removed_segments": len(removed_segments),
                "whole_episode_dropped": int(not keep_episode),
            }
        )

        for segment_idx, (start, end) in enumerate(removed_segments):
            segment_rows.append(
                {
                    "episode": episode,
                    "keep_episode": int(keep_episode),
                    "segment_index": int(segment_idx),
                    "start": int(start),
                    "end": int(end),
                    "length": int(end - start),
                    "center": int((start + end - 1) // 2),
                    "removed_ratio_in_episode": removed_ratio,
                    "episode_joint_score": float(row.episode_joint_score),
                    "whole_episode_dropped": int(not keep_episode),
                }
            )

    episode_df = pd.DataFrame(episode_rows).sort_values(
        ["whole_episode_dropped", "removed_ratio", "episode_joint_score", "episode"],
        ascending=[False, False, True, True],
        kind="stable",
    )
    segments_df = pd.DataFrame(segment_rows).sort_values(
        ["whole_episode_dropped", "length", "episode_joint_score", "episode", "start"],
        ascending=[False, False, True, True, True],
        kind="stable",
    )
    return episode_df, segments_df, keep_masks


def _plot_overview(
    episode_df: pd.DataFrame,
    keep_masks: dict[int, np.ndarray],
    output_path: Path,
    bins: int,
) -> None:
    ordered_episodes = episode_df["episode"].astype(int).tolist()
    heatmap = np.stack([_normalize_removed_mask(keep_masks[episode], bins=bins) for episode in ordered_episodes], axis=0)
    score_values = episode_df["episode_joint_score"].to_numpy(dtype=np.float32)
    removed_ratios = episode_df["removed_ratio"].to_numpy(dtype=np.float32)
    keep_episode = episode_df["keep_episode"].to_numpy(dtype=np.int64)

    cmap = LinearSegmentedColormap.from_list("keep_remove", ["#1b9e77", "#fee08b", "#d73027"])

    fig = plt.figure(figsize=(18, max(10, 0.18 * len(ordered_episodes))))
    grid = fig.add_gridspec(1, 2, width_ratios=[2.8, 1.2], wspace=0.18)

    ax0 = fig.add_subplot(grid[0, 0])
    im = ax0.imshow(heatmap, aspect="auto", interpolation="nearest", cmap=cmap, vmin=0.0, vmax=1.0)
    ax0.set_title("stack_cube_205 mysys removal heatmap (sorted by removed ratio)")
    ax0.set_xlabel("normalized time")
    ax0.set_ylabel("Episode")
    tick_positions = np.arange(len(ordered_episodes))
    tick_step = max(1, len(ordered_episodes) // 25)
    ax0.set_yticks(tick_positions[::tick_step])
    ax0.set_yticklabels([f"{ordered_episodes[i]:03d}" for i in tick_positions[::tick_step]])
    cbar = fig.colorbar(im, ax=ax0, fraction=0.018, pad=0.01)
    cbar.set_label("removed fraction (0=kept, 1=removed)")

    ax1 = fig.add_subplot(grid[0, 1])
    colors = np.where(keep_episode == 1, "#f46d43", "#d73027")
    markers = np.where(keep_episode == 1, "o", "X")
    for x, y, color, marker in zip(score_values, removed_ratios, colors, markers, strict=True):
        ax1.scatter(x, y, c=color, marker=marker, s=36, alpha=0.85)
    ax1.set_title("Episode score vs removed ratio")
    ax1.set_xlabel("episode_joint_score")
    ax1.set_ylabel("removed_ratio")
    ax1.grid(alpha=0.25, linestyle="--")

    dropped = int(np.count_nonzero(keep_episode == 0))
    partial = int(np.count_nonzero((keep_episode == 1) & (removed_ratios > 0)))
    full_keep = int(np.count_nonzero((keep_episode == 1) & (removed_ratios == 0)))
    fig.suptitle(
        f"whole-drop: {dropped} | partial-drop: {partial} | fully-kept: {full_keep} | "
        f"mean removed ratio: {float(np.mean(removed_ratios)):.3f}",
        fontsize=14,
        y=0.995,
    )
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_contact_sheet(
    *,
    episode_df: pd.DataFrame,
    keep_masks: dict[int, np.ndarray],
    dataset_root: Path,
    camera: str,
    output_path: Path,
    frames_per_episode: int,
    title: str,
    max_rows: int,
) -> None:
    if episode_df.empty:
        return

    view_df = episode_df.head(max_rows).reset_index(drop=True)
    fig = plt.figure(figsize=(4.8 * (frames_per_episode + 1), 2.2 * len(view_df)))
    grid = fig.add_gridspec(len(view_df), frames_per_episode + 1, wspace=0.1, hspace=0.45)

    for row_idx, row in enumerate(view_df.itertuples(index=False)):
        episode = int(row.episode)
        keep_mask = keep_masks[episode]
        removed_segments = _contiguous_false_segments(keep_mask)
        removed_frames = _sample_removed_frames(keep_mask, frames_per_episode)
        video_path = _find_video_path(dataset_root, camera, episode)

        ax_timeline = fig.add_subplot(grid[row_idx, 0])
        ax_timeline.set_xlim(0, max(1, keep_mask.size))
        ax_timeline.set_ylim(0, 1)
        ax_timeline.set_yticks([])
        ax_timeline.set_xlabel("frame")
        ax_timeline.set_title(
            f"ep {episode:03d} | score={float(row.episode_joint_score):.3f} | "
            f"removed={int(row.removed_frames)}/{int(row.episode_length)} ({float(row.removed_ratio):.1%})",
            fontsize=10,
        )
        ax_timeline.broken_barh([(0, max(1, keep_mask.size))], (0.2, 0.6), facecolors="#1b9e77")
        if removed_segments:
            ax_timeline.broken_barh(
                [(start, end - start) for start, end in removed_segments],
                (0.2, 0.6),
                facecolors="#d73027",
            )
        ax_timeline.grid(alpha=0.2, linestyle="--", axis="x")

        if not removed_frames:
            removed_frames = [max(0, keep_mask.size // 2)]

        for col_idx in range(frames_per_episode):
            ax_img = fig.add_subplot(grid[row_idx, col_idx + 1])
            frame_id = removed_frames[min(col_idx, len(removed_frames) - 1)]
            image = _read_video_frame(video_path, frame_id)
            ax_img.imshow(image)
            ax_img.set_xticks([])
            ax_img.set_yticks([])
            ax_img.set_title(f"frame {frame_id}", fontsize=9)
            if video_path is None:
                ax_img.text(
                    0.5,
                    0.5,
                    "missing video",
                    ha="center",
                    va="center",
                    transform=ax_img.transAxes,
                    fontsize=10,
                    color="black",
                    bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.75},
                )

    fig.suptitle(title, fontsize=16, y=0.995)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    selection_dir = args.selection_dir.resolve()
    dataset_root = args.dataset_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    episode_scores_path = selection_dir / "episode_scores.csv"
    window_scores_path = selection_dir / "window_scores.csv"
    selection_npz_path = selection_dir / "joint_episode_window_selection.npz"

    episode_scores = pd.read_csv(episode_scores_path)
    window_scores = pd.read_csv(window_scores_path)
    _ = _load_selection_npz(selection_npz_path)

    episode_df, segments_df, keep_masks = _build_episode_records(episode_scores, window_scores, dataset_root)

    episode_df.to_csv(output_dir / "episode_removal_summary.csv", index=False)
    segments_df.to_csv(output_dir / "removed_segments.csv", index=False)

    _plot_overview(
        episode_df=episode_df,
        keep_masks=keep_masks,
        output_path=output_dir / "overview.png",
        bins=args.heatmap_bins,
    )

    dropped_df = episode_df[episode_df["whole_episode_dropped"] == 1].copy()
    partial_df = episode_df[
        (episode_df["whole_episode_dropped"] == 0) & (episode_df["removed_frames"] > 0)
    ].copy()

    _plot_contact_sheet(
        episode_df=dropped_df.head(args.top_k_dropped),
        keep_masks=keep_masks,
        dataset_root=dataset_root,
        camera=args.camera,
        output_path=output_dir / "top_dropped_episodes.png",
        frames_per_episode=args.frames_per_episode,
        title=f"Whole dropped episodes ({args.camera})",
        max_rows=args.max_preview_rows,
    )
    _plot_contact_sheet(
        episode_df=partial_df.head(args.top_k_partial),
        keep_masks=keep_masks,
        dataset_root=dataset_root,
        camera=args.camera,
        output_path=output_dir / "top_partial_removed_episodes.png",
        frames_per_episode=args.frames_per_episode,
        title=f"Partial removed episodes ({args.camera})",
        max_rows=args.max_preview_rows,
    )

    stats = {
        "num_episodes": int(len(episode_df)),
        "num_whole_episode_dropped": int(len(dropped_df)),
        "num_partial_removed_episodes": int(len(partial_df)),
        "mean_removed_ratio": float(episode_df["removed_ratio"].mean()),
        "median_removed_ratio": float(episode_df["removed_ratio"].median()),
        "total_removed_segments": int(len(segments_df)),
        "camera": args.camera,
        "selection_dir": str(selection_dir),
        "dataset_root": str(dataset_root),
    }
    (output_dir / "summary.json").write_text(json.dumps(stats, indent=2, sort_keys=True) + "\n")

    print(f"可视化已写出到: {output_dir}")
    print(f"- {output_dir / 'overview.png'}")
    print(f"- {output_dir / 'episode_removal_summary.csv'}")
    print(f"- {output_dir / 'removed_segments.csv'}")
    print(f"- {output_dir / 'top_dropped_episodes.png'}")
    print(f"- {output_dir / 'top_partial_removed_episodes.png'}")


if __name__ == "__main__":
    main()
