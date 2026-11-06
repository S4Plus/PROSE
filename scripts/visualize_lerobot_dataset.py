#!/usr/bin/env python3

"""Visualize trajectories stored in a LeRobot-format dataset.

The script reads episode-level parquet files and optionally extracts video frames
to give a quick sense of the data quality. Example usage:

    python scripts/visualize_lerobot_dataset.py \
        --dataset-path /root/.cache/huggingface/lerobot/ls/Static_Pick_Bottle_403 \
        --episodes 0 42 128 \
        --output-dir outputs/visualizations

Required third-party packages:
    - numpy
    - pyarrow
    - matplotlib
    - imageio (only when --include-video is enabled)
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Sequence

import numpy as np

LOGGER = logging.getLogger("visualize_lerobot_dataset")


def _flatten_feature_names(raw_names: Sequence | None) -> list[str]:
    if not raw_names:
        return []
    if isinstance(raw_names, Sequence) and raw_names and isinstance(raw_names[0], Sequence):
        # Names in meta/info.json are stored as a list of lists.
        return list(raw_names[0])
    return list(raw_names)


def _lazy_import_pyarrow():
    try:
        import pyarrow.parquet as pq  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise SystemExit(
            "pyarrow is required to read episode parquet files. Install it via `pip install pyarrow`."
        ) from exc
    return pq


def _lazy_import_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise SystemExit(
            "matplotlib is required for plotting. Install it via `pip install matplotlib`."
        ) from exc
    return plt


def _lazy_import_imageio():
    try:
        import imageio.v3 as iio  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise SystemExit(
            "imageio is required for sampling video frames. Install it via `pip install imageio imageio-ffmpeg`."
        ) from exc
    return iio


def load_dataset_metadata(dataset_root: Path) -> tuple[dict, dict[int, dict]]:
    meta_dir = dataset_root / "meta"
    info_path = meta_dir / "info.json"
    episodes_path = meta_dir / "episodes.jsonl"

    if not info_path.exists():
        raise FileNotFoundError(f"Could not find info.json under {meta_dir}")
    if not episodes_path.exists():
        raise FileNotFoundError(f"Could not find episodes.jsonl under {meta_dir}")

    with info_path.open("r") as f:
        info = json.load(f)

    episodes: dict[int, dict] = {}
    with episodes_path.open("r") as f:
        for line in f:
            if not line.strip():
                continue
            entry = json.loads(line)
            episodes[entry["episode_index"]] = entry

    return info, episodes


def _chunked_array_to_numpy(chunked_array) -> np.ndarray:
    values = chunked_array.to_pylist()
    if len(values) == 0:
        return np.empty((0,), dtype=np.float32)

    first = values[0]
    if isinstance(first, dict):
        raise TypeError("Nested dict columns are not supported in this visualizer.")

    if isinstance(first, list):
        arr = np.asarray(values, dtype=np.float32)
        if arr.ndim == 2 and arr.shape[1] == 1:
            arr = arr[:, 0]
    else:
        arr = np.asarray(values, dtype=np.float32)
    return arr


def load_episode_arrays(
    dataset_root: Path,
    info: dict,
    episode_index: int,
    requested_keys: Sequence[str],
):
    pq = _lazy_import_pyarrow()
    chunk_size = info.get("chunks_size", 1000)
    chunk_idx = episode_index // chunk_size
    data_rel = info["data_path"].format(episode_chunk=chunk_idx, episode_index=episode_index)
    data_path = dataset_root / data_rel
    if not data_path.exists():
        raise FileNotFoundError(f"Missing parquet file for episode {episode_index}: {data_path}")

    table = pq.read_table(data_path)
    arrays = {}
    for key in requested_keys:
        if key not in table.column_names:
            LOGGER.warning("Column '%s' missing in %s", key, data_path)
            continue
        arrays[key] = _chunked_array_to_numpy(table[key])
    return arrays


def sample_episode_indices(
    episodes: dict[int, dict],
    explicit: Sequence[int] | None,
    num_random: int,
    seed: int,
) -> list[int]:
    all_indices = sorted(episodes.keys())
    if not all_indices:
        raise ValueError("Dataset does not contain any episodes.")

    if explicit:
        missing = [idx for idx in explicit if idx not in episodes]
        if missing:
            raise ValueError(f"Episode indices not found: {missing}")
        return sorted(set(explicit))

    rng = random.Random(seed)
    num = min(num_random, len(all_indices))
    return sorted(rng.sample(all_indices, num))


def _normalize_time_axis(timestamps: np.ndarray | None, length: int, fps: float) -> np.ndarray:
    if timestamps is not None and len(timestamps) == length:
        return np.asarray(timestamps, dtype=np.float32).squeeze()
    return np.arange(length, dtype=np.float32) / float(fps)


def _plot_series(ax, time_axis: np.ndarray, values: np.ndarray, labels: list[str], title: str) -> None:
    if values.size == 0:
        ax.set_title(f"{title} (missing)")
        return

    dim = values.shape[1] if values.ndim == 2 else 1
    if dim == 1:
        ax.plot(time_axis, values, label=labels[0] if labels else "value", linewidth=1.5)
    else:
        for i in range(dim):
            name = labels[i] if i < len(labels) else f"dim_{i}"
            ax.plot(time_axis, values[:, i], label=name, linewidth=1.2)
        ax.legend(loc="upper right", fontsize=8, ncol=2)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Value")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.3)


def _stack_contact_sheet(frames: list[np.ndarray]) -> np.ndarray:
    if not frames:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    # Ensure all frames share the same height by center-cropping/padding.
    heights = [frame.shape[0] for frame in frames]
    min_h = min(heights)
    processed = []
    for frame in frames:
        if frame.shape[0] != min_h:
            top = (frame.shape[0] - min_h) // 2
            frame = frame[top : top + min_h, ...]
        processed.append(frame)
    return np.concatenate(processed, axis=1)


def _sample_video_frames(
    dataset_root: Path,
    info: dict,
    episode_index: int,
    video_key: str,
    num_samples: int,
    fallback_length: int,
) -> list[np.ndarray]:
    video_path_template = info.get("video_path")
    if not video_path_template:
        return []

    chunk_size = info.get("chunks_size", 1000)
    chunk_idx = episode_index // chunk_size
    video_rel = video_path_template.format(
        episode_chunk=chunk_idx,
        video_key=video_key,
        episode_index=episode_index,
    )
    video_path = dataset_root / video_rel
    if not video_path.exists():
        LOGGER.warning("Video %s missing for episode %d", video_rel, episode_index)
        return []

    iio = _lazy_import_imageio()
    try:
        metadata = iio.immeta(video_path)
        total_frames = int(metadata.get("nframes") or fallback_length)
    except Exception:
        total_frames = fallback_length

    if total_frames <= 0:
        total_frames = fallback_length
    if total_frames <= 0:
        return []

    frame_indices = np.linspace(0, total_frames - 1, num_samples, dtype=int)
    frames: list[np.ndarray] = []
    for idx in frame_indices:
        try:
            frame = iio.imread(video_path, index=int(idx))
        except Exception as exc:  # pragma: no cover - video backend issues
            LOGGER.warning("Failed to read frame %d from %s: %s", idx, video_path, exc)
            break
        if frame.ndim == 2:
            frame = np.stack([frame] * 3, axis=-1)
        frames.append(frame)
    return frames


def plot_episode(
    dataset_root: Path,
    info: dict,
    episode_meta: dict,
    episode_index: int,
    output_dir: Path,
    include_video: bool,
    video_keys: Sequence[str],
    num_video_frames: int,
):
    plt = _lazy_import_matplotlib()

    features = info.get("features", {})
    state_names = _flatten_feature_names(features.get("observation.state", {}).get("names"))
    action_names = _flatten_feature_names(features.get("action", {}).get("names"))

    arrays = load_episode_arrays(
        dataset_root,
        info,
        episode_index,
        requested_keys=["timestamp", "observation.state", "action"],
    )

    timestamps = arrays.get("timestamp")
    states = arrays.get("observation.state")
    actions = arrays.get("action")

    length = 0
    for arr in (states, actions, timestamps):
        if arr is not None:
            length = max(length, len(arr))
    if length == 0:
        raise RuntimeError(f"Episode {episode_index} did not contain any numeric data.")

    time_axis = _normalize_time_axis(timestamps, length, info["fps"])

    n_video_rows = len(video_keys) if include_video else 0
    total_rows = 2 + n_video_rows
    height = 4 + n_video_rows * 2
    fig, axes = plt.subplots(total_rows, 1, figsize=(12, height), squeeze=False)
    ax_iter = iter(axes.flatten())

    state_ax = next(ax_iter)
    if states is not None:
        _plot_series(state_ax, time_axis[: len(states)], states, state_names, "Observation States")
    else:
        state_ax.set_title("Observation States (not available)")
        state_ax.axis("off")

    action_ax = next(ax_iter)
    if actions is not None:
        _plot_series(action_ax, time_axis[: len(actions)], actions, action_names, "Actions")
    else:
        action_ax.set_title("Actions (not available)")
        action_ax.axis("off")

    if include_video:
        for video_key in video_keys:
            video_ax = next(ax_iter)
            frames = _sample_video_frames(
                dataset_root,
                info,
                episode_index,
                video_key=video_key,
                num_samples=num_video_frames,
                fallback_length=episode_meta.get("length", len(time_axis)),
            )
            if frames:
                sheet = _stack_contact_sheet(frames)
                video_ax.imshow(sheet)
                video_ax.set_title(f"{video_key} (sampled {len(frames)} frames)")
                video_ax.axis("off")
            else:
                video_ax.set_title(f"{video_key} (no frames decoded)")
                video_ax.axis("off")

    fig.suptitle(
        f"Episode {episode_index} — task: {episode_meta.get('tasks', ['?'])[0]} "
        f"(length={episode_meta.get('length', 'NA')} steps)",
        fontsize=14,
    )
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    output_dir.mkdir(parents=True, exist_ok=True)
    outfile = output_dir / f"episode_{episode_index:06d}.png"
    fig.savefig(outfile)
    plt.close(fig)
    LOGGER.info("Saved visualization for episode %d to %s", episode_index, outfile)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize LeRobot dataset episodes from parquet + video files.",
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        required=True,
        help="Path to the dataset root (the directory containing meta/data/videos).",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        nargs="+",
        default=None,
        help="Specific episode indices to visualize. Defaults to sampling randomly.",
    )
    parser.add_argument(
        "--num-random",
        type=int,
        default=3,
        help="How many episodes to sample when --episodes is not provided.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed used when sampling episodes.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/lerobot_visualizations"),
        help="Directory where the generated PNG files should be saved.",
    )
    parser.add_argument(
        "--include-video",
        action="store_true",
        help="Decode accompanying MP4 files and show sampled frames.",
    )
    parser.add_argument(
        "--video-keys",
        type=str,
        nargs="+",
        default=[
            "observation.images.head_camera",
            "observation.images.right_wrist_camera",
        ],
        help="Which camera streams to visualize (matched to meta/info.json entries).",
    )
    parser.add_argument(
        "--video-frames",
        type=int,
        default=6,
        help="How many frames to sample per camera when --include-video is enabled.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()

    dataset_root = args.dataset_path.expanduser().resolve()
    info, episodes = load_dataset_metadata(dataset_root)

    selected = sample_episode_indices(episodes, args.episodes, args.num_random, args.seed)
    LOGGER.info("Visualizing episodes: %s", selected)

    for idx in selected:
        plot_episode(
            dataset_root=dataset_root,
            info=info,
            episode_meta=episodes[idx],
            episode_index=idx,
            output_dir=args.output_dir,
            include_video=args.include_video,
            video_keys=args.video_keys,
            num_video_frames=args.video_frames,
        )


if __name__ == "__main__":
    main()
