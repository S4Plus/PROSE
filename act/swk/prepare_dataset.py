#!/usr/bin/env python3
"""
Trim a LeRobot dataset by keeping only the first part of each episode.

Example:
  python swk/prepare_dataset.py \
    --input /ext_workspace/liushuo/swk/six_object_pick_conveyor \
    --output ./swk/six_object_pick_conveyor_first2 \
    --keep-objects 2 --total-objects 6 --overwrite
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Iterable, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Trim a LeRobot dataset to only the first part of each episode."
    )
    parser.add_argument("--input", type=Path, required=True, help="Input dataset root.")
    parser.add_argument("--output", type=Path, required=True, help="Output dataset root.")
    parser.add_argument(
        "--keep-frames",
        type=int,
        default=None,
        help="Keep only this many frames per episode (overrides keep-objects).",
    )
    parser.add_argument("--keep-objects", type=int, default=2, help="Objects to keep.")
    parser.add_argument(
        "--total-objects", type=int, default=6, help="Total objects per episode."
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Override chunk size (defaults to input info.json chunks_size).",
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="Override task text in episodes.jsonl and tasks.jsonl.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output.")
    parser.add_argument("--skip-videos", action="store_true", help="Do not copy videos.")
    parser.add_argument("--skip-images", action="store_true", help="Do not copy images.")
    parser.add_argument(
        "--skip-video-stats",
        action="store_true",
        help="Skip recomputing video stats (keeps existing stats if available).",
    )
    parser.add_argument("--ffmpeg", type=str, default=None, help="ffmpeg binary path.")
    parser.add_argument("--ffprobe", type=str, default=None, help="ffprobe binary path.")
    return parser.parse_args()


def load_jsonl(path: Path) -> List[dict]:
    with path.open("r") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")


def find_parquet_paths(data_root: Path) -> Dict[int, Path]:
    mapping: Dict[int, Path] = {}
    for path in data_root.rglob("episode_*[0-9].parquet"):
        name = path.name
        if not name.startswith("episode_") or not name.endswith(".parquet"):
            continue
        idx = int(name.replace("episode_", "").replace(".parquet", ""))
        mapping[idx] = path
    if not mapping:
        raise FileNotFoundError(f"No parquet episodes found under {data_root}")
    return mapping


def find_vision_paths(root: Path, ext: str) -> Dict[str, Dict[int, Path]]:
    mapping: Dict[str, Dict[int, Path]] = {}
    if not root.is_dir():
        return mapping
    for path in root.rglob(f"episode_*{ext}"):
        name = path.name
        if not name.startswith("episode_") or not name.endswith(ext):
            continue
        m = re.search(r"episode_(\d+)" + re.escape(ext) + r"$", name)
        if not m:
            continue
        idx = int(m.group(1))
        key = path.parent.name
        if key.startswith("chunk-"):
            key = path.parent.parent.name
        mapping.setdefault(key, {})[idx] = path
    return mapping


def get_chunk_dir(base: Path, episode_index: int, chunk_size: int, key: str | None = None) -> Path:
    chunk_id = episode_index // chunk_size
    path = base / f"chunk-{chunk_id:03d}"
    if key:
        path = path / key
    path.mkdir(parents=True, exist_ok=True)
    return path


def compute_keep_len(length: int, keep_frames: int | None, keep_objects: int, total_objects: int) -> int:
    if keep_frames is not None:
        if keep_frames <= 0:
            raise ValueError("--keep-frames must be positive.")
        return min(length, keep_frames)
    if total_objects <= 0 or keep_objects <= 0:
        raise ValueError("--total-objects and --keep-objects must be positive.")
    if keep_objects >= total_objects:
        return length
    keep_len = int(math.floor(length * (keep_objects / total_objects)))
    return max(1, min(length, keep_len))


def update_table_indices(table, episode_index: int, start_idx: int, length: int):
    import pyarrow as pa

    def replace_column(name: str, values: Iterable) -> None:
        nonlocal table
        if name not in table.schema.names:
            return
        field = table.schema.field(name)
        arr = pa.array(list(values), type=field.type)
        table = table.set_column(table.schema.get_field_index(name), name, arr)

    replace_column("episode_index", [episode_index] * length)
    replace_column("index", range(start_idx, start_idx + length))
    replace_column("frame_index", range(length))
    return table


def _parse_rate(value: str) -> float | None:
    if not value:
        return None
    if "/" in value:
        num, denom = value.split("/", 1)
        try:
            return float(num) / float(denom)
        except ValueError:
            return None
    try:
        return float(value)
    except ValueError:
        return None


def _run_ffprobe(ffprobe: str, path: Path) -> dict:
    cmd = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=nb_frames,avg_frame_rate,r_frame_rate,duration",
        "-of",
        "json",
        str(path),
    ]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return json.loads(result.stdout)


def _run_ffprobe_count(ffprobe: str, path: Path) -> dict:
    cmd = [
        ffprobe,
        "-v",
        "error",
        "-count_frames",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=nb_read_frames",
        "-of",
        "json",
        str(path),
    ]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return json.loads(result.stdout)


def get_video_frame_count(ffprobe: str, path: Path) -> int | None:
    try:
        data = _run_ffprobe(ffprobe, path)
    except subprocess.CalledProcessError:
        return None
    streams = data.get("streams") or []
    if not streams:
        return None
    stream = streams[0]
    nb_frames = stream.get("nb_frames")
    if nb_frames and str(nb_frames).isdigit():
        return int(nb_frames)

    try:
        data = _run_ffprobe_count(ffprobe, path)
        streams = data.get("streams") or []
        if streams:
            nb_read = streams[0].get("nb_read_frames")
            if nb_read and str(nb_read).isdigit():
                return int(nb_read)
    except subprocess.CalledProcessError:
        pass

    duration = stream.get("duration")
    fps = _parse_rate(stream.get("avg_frame_rate") or stream.get("r_frame_rate") or "")
    if duration and fps:
        try:
            return max(1, int(round(float(duration) * fps)))
        except ValueError:
            return None
    return None


def trim_video(ffmpeg: str, src: Path, dst: Path, frames: int) -> None:
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(src),
        "-frames:v",
        str(frames),
        "-c:v",
        "copy",
        "-an",
        str(dst),
    ]
    subprocess.run(cmd, check=True)


def estimate_num_samples(
    dataset_len: int, min_num_samples: int = 100, max_num_samples: int = 10_000, power: float = 0.75
) -> int:
    if dataset_len < min_num_samples:
        min_num_samples = dataset_len
    return max(min_num_samples, min(int(dataset_len**power), max_num_samples))


def sample_indices(data_len: int) -> list[int]:
    import numpy as np

    num_samples = estimate_num_samples(data_len)
    return [int(round(x)) for x in list(np.linspace(0, data_len - 1, num_samples))]


def auto_downsample_height_width(
    img: "np.ndarray", target_size: int = 150, max_size_threshold: int = 300
) -> "np.ndarray":
    _, height, width = img.shape
    if max(width, height) < max_size_threshold:
        return img
    downsample_factor = int(width / target_size) if width > height else int(height / target_size)
    return img[:, ::downsample_factor, ::downsample_factor]


def compute_video_stats(path: Path, ffprobe: str | None) -> dict:
    import imageio.v2 as imageio
    import numpy as np

    total_frames = get_video_frame_count(ffprobe, path) if ffprobe else None
    reader = imageio.get_reader(str(path), "ffmpeg")

    if total_frames is None or total_frames <= 0:
        frames = list(reader)
        reader.close()
        total_frames = len(frames)
        sampled = sample_indices(total_frames)
        images = []
        for idx in sampled:
            frame = frames[idx]
            img = np.transpose(frame, (2, 0, 1))
            img = auto_downsample_height_width(img)
            images.append(img.astype(np.uint8))
    else:
        sampled = sample_indices(total_frames)
        sample_set = set(sampled)
        images = []
        for idx, frame in enumerate(reader):
            if idx in sample_set:
                img = np.transpose(frame, (2, 0, 1))
                img = auto_downsample_height_width(img)
                images.append(img.astype(np.uint8))
        reader.close()

    if not images:
        raise RuntimeError(f"No frames sampled from {path}")

    arr = np.stack(images, axis=0)
    axes_to_reduce = (0, 2, 3)
    minv = np.min(arr, axis=axes_to_reduce, keepdims=True)
    maxv = np.max(arr, axis=axes_to_reduce, keepdims=True)
    meanv = np.mean(arr, axis=axes_to_reduce, keepdims=True)
    stdv = np.std(arr, axis=axes_to_reduce, keepdims=True)
    count = np.array([len(images)])

    def fmt(x):
        return np.squeeze(x / 255.0, axis=0).tolist()

    return {
        "min": fmt(minv),
        "max": fmt(maxv),
        "mean": fmt(meanv),
        "std": fmt(stdv),
        "count": count.tolist(),
    }


def compute_column_stats(column) -> dict:
    import numpy as np
    import pyarrow as pa

    col = column.combine_chunks()
    length = len(col)
    if length <= 0:
        raise ValueError("Cannot compute stats for empty column.")

    if pa.types.is_fixed_size_list(col.type):
        list_size = col.type.list_size
        values = col.values.to_numpy(zero_copy_only=False).reshape(length, list_size)
        min_vals = np.min(values, axis=0).astype(float).tolist()
        max_vals = np.max(values, axis=0).astype(float).tolist()
        mean_vals = np.mean(values, axis=0).astype(float).tolist()
        std_vals = np.std(values, axis=0).astype(float).tolist()
        return {
            "min": min_vals,
            "max": max_vals,
            "mean": mean_vals,
            "std": std_vals,
            "count": [length],
        }

    values = col.to_numpy(zero_copy_only=False)
    is_int = np.issubdtype(values.dtype, np.integer)
    min_val = int(values.min()) if is_int else float(values.min())
    max_val = int(values.max()) if is_int else float(values.max())
    mean_val = float(values.mean())
    std_val = float(values.std())
    return {
        "min": [min_val],
        "max": [max_val],
        "mean": [mean_val],
        "std": [std_val],
        "count": [length],
    }


def main() -> None:
    args = parse_args()
    try:
        import numpy as np  # noqa: F401
        import pyarrow as pa  # noqa: F401
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise SystemExit("Missing dependencies: install numpy and pyarrow.") from exc

    input_root = args.input.expanduser().resolve()
    output_root = args.output.expanduser().resolve()
    if output_root.exists():
        if not args.overwrite:
            raise SystemExit(f"Output already exists: {output_root} (use --overwrite to replace)")
        shutil.rmtree(output_root)

    info_path = input_root / "meta" / "info.json"
    if not info_path.is_file():
        raise SystemExit(f"Missing info.json: {info_path}")
    info = json.loads(info_path.read_text())
    chunk_size = args.chunk_size or int(info.get("chunks_size", 1000))
    video_keys = [
        key for key, meta in info.get("features", {}).items() if meta.get("dtype") == "video"
    ]
    ffmpeg = args.ffmpeg or shutil.which("ffmpeg")
    ffprobe = args.ffprobe or shutil.which("ffprobe")
    recompute_video_stats = not args.skip_video_stats and not args.skip_videos
    if video_keys and not args.skip_videos:
        if not ffmpeg:
            raise SystemExit("ffmpeg not found (use --ffmpeg or add to PATH).")
        if not ffprobe:
            raise SystemExit("ffprobe not found (use --ffprobe or add to PATH).")
    if recompute_video_stats:
        try:
            import imageio.v2  # noqa: F401
        except ImportError as exc:
            raise SystemExit("Missing dependency: install imageio to recompute video stats.") from exc

    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "data").mkdir(parents=True, exist_ok=True)
    (output_root / "meta").mkdir(parents=True, exist_ok=True)

    episodes = load_jsonl(input_root / "meta" / "episodes.jsonl")
    episodes = sorted(episodes, key=lambda x: int(x["episode_index"]))
    stats_by_episode = {}
    stats_path = input_root / "meta" / "episodes_stats.jsonl"
    if stats_path.is_file():
        stats_by_episode = {int(x["episode_index"]): x["stats"] for x in load_jsonl(stats_path)}

    parquet_map = find_parquet_paths(input_root / "data")
    video_map = find_vision_paths(input_root / "videos", ".mp4")
    image_map = {}
    image_map.update(find_vision_paths(input_root / "images", ".png"))
    image_map.update(find_vision_paths(input_root / "images", ".jpg"))

    new_episodes: List[dict] = []
    new_stats: List[dict] = []
    total_frames = 0
    total_videos = 0

    for rec in episodes:
        episode_index = int(rec["episode_index"])
        parquet_path = parquet_map.get(episode_index)
        if parquet_path is None:
            print(f"Warning: missing parquet for episode {episode_index}, skipping.")
            continue

        table = pq.read_table(parquet_path)
        length = table.num_rows
        keep_len = compute_keep_len(
            length, args.keep_frames, args.keep_objects, args.total_objects
        )
        table = table.slice(0, keep_len)
        table = update_table_indices(table, episode_index, total_frames, keep_len)

        chunk_dir = get_chunk_dir(output_root / "data", episode_index, chunk_size)
        pq.write_table(table, chunk_dir / f"episode_{episode_index:06d}.parquet")

        video_stats = {}
        if not args.skip_videos:
            for key in video_keys:
                src = video_map.get(key, {}).get(episode_index)
                if src is None:
                    continue
                dst_dir = get_chunk_dir(output_root / "videos", episode_index, chunk_size, key)
                dst = dst_dir / f"episode_{episode_index:06d}.mp4"
                video_frames = get_video_frame_count(ffprobe, src)
                if video_frames is None:
                    video_keep = keep_len
                else:
                    ratio = keep_len / length if length else 1.0
                    video_keep = max(1, int(round(video_frames * ratio)))
                if video_frames is not None and video_keep >= video_frames:
                    shutil.copy2(src, dst)
                else:
                    trim_video(ffmpeg, src, dst, video_keep)
                if recompute_video_stats:
                    video_stats[key] = compute_video_stats(dst, ffprobe)
                total_videos += 1

        if not args.skip_images:
            for key, mapping in image_map.items():
                src = mapping.get(episode_index)
                if src is None:
                    continue
                dst_dir = get_chunk_dir(output_root / "images", episode_index, chunk_size, key)
                shutil.copy2(src, dst_dir / src.name)

        rec_new = dict(rec)
        rec_new["length"] = keep_len
        if args.task:
            rec_new["tasks"] = [args.task]
        new_episodes.append(rec_new)

        stats = {}
        for name in table.schema.names:
            stats[name] = compute_column_stats(table.column(name))
        if recompute_video_stats and video_stats:
            for key, vstats in video_stats.items():
                stats[key] = vstats
        else:
            base_stats = stats_by_episode.get(episode_index)
            if base_stats:
                for key in video_keys:
                    if key in base_stats:
                        stats[key] = base_stats[key]
        new_stats.append({"episode_index": episode_index, "stats": stats})

        total_frames += keep_len

    write_jsonl(output_root / "meta" / "episodes.jsonl", new_episodes)
    if new_stats:
        write_jsonl(output_root / "meta" / "episodes_stats.jsonl", new_stats)

    info["total_episodes"] = len(new_episodes)
    info["total_frames"] = total_frames
    info["chunks_size"] = chunk_size
    info["total_chunks"] = math.ceil(len(new_episodes) / chunk_size) if chunk_size else 0
    if not args.skip_videos:
        info["total_videos"] = total_videos
    if args.task:
        info["total_tasks"] = 1

    splits = info.get("splits")
    if isinstance(splits, dict) and "train" in splits:
        splits["train"] = f"0:{len(new_episodes)}"
        info["splits"] = splits

    with (output_root / "meta" / "info.json").open("w") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

    tasks_path = input_root / "meta" / "tasks.jsonl"
    if args.task:
        write_jsonl(output_root / "meta" / "tasks.jsonl", [{"task_index": 0, "task": args.task}])
    elif tasks_path.is_file():
        shutil.copy2(tasks_path, output_root / "meta" / "tasks.jsonl")

    norm_stats = input_root / "norm_stats.json"
    if norm_stats.is_file():
        shutil.copy2(norm_stats, output_root / "norm_stats.json")

    print(f"Trimmed dataset written to: {output_root}")
    print(f"  episodes: {len(new_episodes)}")
    print(f"  frames:   {total_frames}")
    print(f"  videos:   {total_videos}")


if __name__ == "__main__":
    main()
