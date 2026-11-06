#!/usr/bin/env python3
"""
Merge two LeRobot datasets and downsample the higher-FPS dataset to match the lower FPS.
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
        description="Merge two LeRobot datasets with FPS normalization."
    )
    parser.add_argument("--input-a", type=Path, required=True, help="First dataset root.")
    parser.add_argument("--input-b", type=Path, required=True, help="Second dataset root.")
    parser.add_argument("--output", type=Path, required=True, help="Output dataset root.")
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Override chunk size (defaults to input A info.json chunks_size).",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output.")
    parser.add_argument("--skip-videos", action="store_true", help="Do not copy videos.")
    parser.add_argument("--skip-images", action="store_true", help="Do not copy images.")
    parser.add_argument(
        "--skip-video-stats",
        action="store_true",
        help="Skip recomputing video stats (keeps existing stats if available).",
    )
    parser.add_argument(
        "--reencode-videos",
        action="store_true",
        help="Re-encode all videos to target FPS/codec (ensures consistent metadata).",
    )
    parser.add_argument("--ffmpeg", type=str, default=None, help="ffmpeg binary path.")
    parser.add_argument("--ffprobe", type=str, default=None, help="ffprobe binary path.")
    parser.add_argument(
        "--video-encoder",
        type=str,
        default=None,
        help="ffmpeg encoder name for re-encoding videos (e.g., libx264).",
    )
    parser.add_argument(
        "--video-codec",
        type=str,
        default=None,
        help="Metadata codec name to write into info.json (e.g., h264).",
    )
    parser.add_argument(
        "--video-pix-fmt",
        type=str,
        default=None,
        help="Pixel format for re-encoded videos (e.g., yuv420p).",
    )
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


def get_dataset_fps(info: dict) -> float:
    if "fps" in info:
        return float(info["fps"])
    for meta in info.get("features", {}).values():
        if meta.get("dtype") == "video":
            fps = meta.get("info", {}).get("video.fps")
            if fps is not None:
                return float(fps)
    raise ValueError("Cannot determine FPS from info.json.")


def compare_features(info_a: dict, info_b: dict) -> None:
    features_a = info_a.get("features", {})
    features_b = info_b.get("features", {})
    if set(features_a.keys()) != set(features_b.keys()):
        raise ValueError("Feature keys differ between datasets.")
    for key in features_a.keys():
        fa = features_a[key]
        fb = features_b[key]
        if fa.get("dtype") != fb.get("dtype"):
            raise ValueError(f"Feature dtype mismatch for {key}.")
        if fa.get("shape") != fb.get("shape"):
            raise ValueError(f"Feature shape mismatch for {key}.")
        if fa.get("names") != fb.get("names"):
            raise ValueError(f"Feature names mismatch for {key}.")
        if fa.get("dtype") == "video":
            info_a_vid = dict(fa.get("info", {}))
            info_b_vid = dict(fb.get("info", {}))
            info_a_vid.pop("video.fps", None)
            info_b_vid.pop("video.fps", None)
            if info_a_vid != info_b_vid:
                raise ValueError(f"Video feature metadata mismatch for {key}.")


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


def update_table_indices(
    table,
    episode_index: int,
    start_idx: int,
    length: int,
    task_index: int | None,
    timestamps: List[float] | None,
):
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
    if task_index is not None:
        replace_column("task_index", [task_index] * length)
    if timestamps is not None and "timestamp" in table.schema.names:
        field = table.schema.field("timestamp")
        if pa.types.is_fixed_size_list(field.type):
            list_size = field.type.list_size
            if list_size != 1:
                raise ValueError("timestamp column has unexpected list size.")
            values = [[float(ts)] for ts in timestamps]
        else:
            values = [float(ts) for ts in timestamps]
        replace_column("timestamp", values)
    return table


def build_downsample_indices(length: int, ratio: float) -> List[int]:
    import numpy as np

    if ratio <= 1.0:
        return list(range(length))
    target_len = int(round(length / ratio))
    target_len = max(1, min(length, target_len))
    if target_len == 1:
        return [0]
    indices = [int(round(x)) for x in np.linspace(0, length - 1, target_len)]
    filtered: List[int] = []
    last = -1
    for idx in indices:
        if idx <= last:
            continue
        filtered.append(idx)
        last = idx
    return filtered


def extract_timestamps(table) -> List[float] | None:
    import pyarrow as pa

    if "timestamp" not in table.schema.names:
        return None
    col = table.column("timestamp").combine_chunks()
    if pa.types.is_fixed_size_list(col.type):
        list_size = col.type.list_size
        values = col.values.to_numpy(zero_copy_only=False)
        if list_size != 1:
            raise ValueError("timestamp column has unexpected list size.")
        arr = values.reshape(len(col))
    else:
        arr = col.to_numpy(zero_copy_only=False)
    return [float(x) for x in arr]


def build_uniform_timestamps(length: int, target_fps: float) -> List[float]:
    if length <= 0:
        return []
    dt = 1.0 / target_fps
    return [i * dt for i in range(length)]


def build_downsample_indices_from_timestamps(
    timestamps: List[float], target_fps: float
) -> List[int]:
    import numpy as np

    if not timestamps:
        return []
    if len(timestamps) == 1:
        return [0]

    t0 = timestamps[0]
    t_end = timestamps[-1]
    if t_end <= t0:
        return [0]
    duration = t_end - t0
    target_len = int(round(duration * target_fps)) + 1
    target_len = max(1, min(len(timestamps), target_len))

    ts = np.array(timestamps, dtype=np.float64)
    target_times = t0 + (np.arange(target_len) / target_fps)
    candidates = np.searchsorted(ts, target_times, side="left")

    selected: List[int] = []
    last = -1
    for target_time, idx in zip(target_times, candidates):
        if idx <= 0:
            cand = 0
        elif idx >= len(ts):
            cand = len(ts) - 1
        else:
            prev = idx - 1
            if abs(ts[idx] - target_time) < abs(target_time - ts[prev]):
                cand = idx
            else:
                cand = prev
        if cand <= last:
            cand = last + 1
        if cand >= len(ts):
            break
        selected.append(int(cand))
        last = cand
    if not selected:
        selected = [0]
    return selected


def default_encoder_for_codec(codec: str | None) -> str:
    if not codec:
        return "libx264"
    codec = codec.lower()
    if codec == "av1":
        return "libsvtav1"
    if codec == "h264":
        return "libx264"
    if codec == "hevc":
        return "libx265"
    if codec == "vp9":
        return "libvpx-vp9"
    return "libx264"


def codec_for_encoder(encoder: str | None) -> str | None:
    if not encoder:
        return None
    encoder = encoder.lower()
    if "svtav1" in encoder or "aom" in encoder:
        return "av1"
    if "x265" in encoder:
        return "hevc"
    if "x264" in encoder:
        return "h264"
    if "vpx-vp9" in encoder:
        return "vp9"
    return None


def downsample_video(
    ffmpeg: str,
    src: Path,
    dst: Path,
    target_fps: float,
    encoder: str,
    pix_fmt: str | None,
) -> None:
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(src),
        "-vf",
        f"fps={target_fps}",
        "-c:v",
        encoder,
    ]
    if pix_fmt:
        cmd.extend(["-pix_fmt", pix_fmt])
    cmd.extend(["-an", str(dst)])
    subprocess.run(cmd, check=True)


def collect_tasks(episodes: Iterable[dict]) -> List[str]:
    tasks: List[str] = []
    seen = set()
    for rec in episodes:
        for task in rec.get("tasks", []) or []:
            if task not in seen:
                tasks.append(task)
                seen.add(task)
    return tasks


def main() -> None:
    args = parse_args()
    try:
        import numpy as np  # noqa: F401
        import pyarrow as pa  # noqa: F401
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise SystemExit("Missing dependencies: install numpy and pyarrow.") from exc

    input_a = args.input_a.expanduser().resolve()
    input_b = args.input_b.expanduser().resolve()
    output_root = args.output.expanduser().resolve()

    info_a_path = input_a / "meta" / "info.json"
    info_b_path = input_b / "meta" / "info.json"
    if not info_a_path.is_file() or not info_b_path.is_file():
        raise SystemExit("Missing info.json in one of the datasets.")

    info_a = json.loads(info_a_path.read_text())
    info_b = json.loads(info_b_path.read_text())

    if info_a.get("robot_type") != info_b.get("robot_type"):
        raise SystemExit("robot_type mismatch between datasets.")
    if info_a.get("codebase_version") != info_b.get("codebase_version"):
        raise SystemExit("codebase_version mismatch between datasets.")
    compare_features(info_a, info_b)

    fps_a = get_dataset_fps(info_a)
    fps_b = get_dataset_fps(info_b)
    target_fps = min(fps_a, fps_b)

    chunk_size = args.chunk_size or int(info_a.get("chunks_size", 1000))

    if output_root.exists():
        if not args.overwrite:
            raise SystemExit(f"Output already exists: {output_root} (use --overwrite to replace)")
        shutil.rmtree(output_root)

    video_keys = [
        key for key, meta in info_a.get("features", {}).items() if meta.get("dtype") == "video"
    ]
    input_video_codec = None
    input_video_pix_fmt = None
    if video_keys:
        sample_key = video_keys[0]
        input_video_codec = info_a["features"][sample_key].get("info", {}).get("video.codec")
        input_video_pix_fmt = info_a["features"][sample_key].get("info", {}).get("video.pix_fmt")

    ffmpeg = args.ffmpeg or shutil.which("ffmpeg")
    ffprobe = args.ffprobe or shutil.which("ffprobe")
    if video_keys and not args.skip_videos:
        if not ffmpeg:
            raise SystemExit("ffmpeg not found (use --ffmpeg or add to PATH).")
        if not args.skip_video_stats and not ffprobe:
            raise SystemExit("ffprobe not found (use --ffprobe or add to PATH).")

    needs_downsample = fps_a > target_fps or fps_b > target_fps
    encoder = args.video_encoder
    if video_keys and not args.skip_videos and (needs_downsample or args.reencode_videos):
        if not encoder:
            encoder = default_encoder_for_codec(input_video_codec)
        output_codec = args.video_codec or codec_for_encoder(encoder) or input_video_codec
        output_pix_fmt = args.video_pix_fmt or input_video_pix_fmt
        if output_codec != input_video_codec and not args.reencode_videos and needs_downsample:
            raise SystemExit(
                "Output codec differs from input. Use --reencode-videos to re-encode all videos "
                "or set --video-encoder/--video-codec to match input codec."
            )
    else:
        output_codec = input_video_codec
        output_pix_fmt = input_video_pix_fmt

    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "data").mkdir(parents=True, exist_ok=True)
    (output_root / "meta").mkdir(parents=True, exist_ok=True)

    episodes_a = load_jsonl(input_a / "meta" / "episodes.jsonl")
    episodes_b = load_jsonl(input_b / "meta" / "episodes.jsonl")
    episodes_a = sorted(episodes_a, key=lambda x: int(x["episode_index"]))
    episodes_b = sorted(episodes_b, key=lambda x: int(x["episode_index"]))

    tasks_a = collect_tasks(episodes_a)
    tasks_a_set = set(tasks_a)
    tasks_b = [t for t in collect_tasks(episodes_b) if t not in tasks_a_set]
    tasks = tasks_a + tasks_b
    task_to_idx = {task: idx for idx, task in enumerate(tasks)}

    stats_a = {}
    stats_b = {}
    stats_path = input_a / "meta" / "episodes_stats.jsonl"
    if stats_path.is_file():
        stats_a = {int(x["episode_index"]): x["stats"] for x in load_jsonl(stats_path)}
    stats_path = input_b / "meta" / "episodes_stats.jsonl"
    if stats_path.is_file():
        stats_b = {int(x["episode_index"]): x["stats"] for x in load_jsonl(stats_path)}

    parquet_a = find_parquet_paths(input_a / "data")
    parquet_b = find_parquet_paths(input_b / "data")
    video_a = find_vision_paths(input_a / "videos", ".mp4")
    video_b = find_vision_paths(input_b / "videos", ".mp4")
    image_a: Dict[str, Dict[int, Path]] = {}
    image_b: Dict[str, Dict[int, Path]] = {}
    image_a.update(find_vision_paths(input_a / "images", ".png"))
    image_a.update(find_vision_paths(input_a / "images", ".jpg"))
    image_b.update(find_vision_paths(input_b / "images", ".png"))
    image_b.update(find_vision_paths(input_b / "images", ".jpg"))

    new_episodes: List[dict] = []
    new_stats: List[dict] = []
    total_frames = 0
    total_videos = 0
    next_episode_index = 0

    def process_dataset(
        root: Path,
        episodes: List[dict],
        parquet_map: Dict[int, Path],
        video_map: Dict[str, Dict[int, Path]],
        image_map: Dict[str, Dict[int, Path]],
        stats_by_episode: Dict[int, dict],
        fps: float,
    ) -> None:
        nonlocal total_frames, total_videos, next_episode_index
        import pyarrow as pa
        import pyarrow.parquet as pq

        downsample_ratio = fps / target_fps if fps > target_fps else 1.0

        for rec in episodes:
            episode_index = int(rec["episode_index"])
            parquet_path = parquet_map.get(episode_index)
            if parquet_path is None:
                print(f"Warning: missing parquet for episode {episode_index}, skipping.")
                continue

            table = pq.read_table(parquet_path)
            length = table.num_rows
            timestamps = extract_timestamps(table)
            if downsample_ratio > 1.0:
                if timestamps is not None:
                    indices = build_downsample_indices_from_timestamps(timestamps, target_fps)
                else:
                    indices = build_downsample_indices(length, downsample_ratio)
                table = table.take(pa.array(indices))
                length = table.num_rows

            tasks_list = rec.get("tasks", []) or []
            task_index = task_to_idx.get(tasks_list[0]) if tasks_list else None
            if timestamps is not None:
                timestamps = build_uniform_timestamps(length, target_fps)
            table = update_table_indices(
                table, next_episode_index, total_frames, length, task_index, timestamps
            )

            chunk_dir = get_chunk_dir(output_root / "data", next_episode_index, chunk_size)
            pq.write_table(table, chunk_dir / f"episode_{next_episode_index:06d}.parquet")

            video_stats = {}
            if not args.skip_videos:
                reencode = args.reencode_videos or downsample_ratio > 1.0
                for key in video_keys:
                    src = video_map.get(key, {}).get(episode_index)
                    if src is None:
                        continue
                    dst_dir = get_chunk_dir(output_root / "videos", next_episode_index, chunk_size, key)
                    dst = dst_dir / f"episode_{next_episode_index:06d}.mp4"
                    if reencode:
                        if not encoder:
                            raise SystemExit("Missing video encoder for re-encoding.")
                        downsample_video(ffmpeg, src, dst, target_fps, encoder, output_pix_fmt)
                    else:
                        shutil.copy2(src, dst)
                    if not args.skip_video_stats and not args.skip_videos:
                        video_stats[key] = compute_video_stats(dst, ffprobe)
                    total_videos += 1

            if not args.skip_images:
                for key, mapping in image_map.items():
                    src = mapping.get(episode_index)
                    if src is None:
                        continue
                    dst_dir = get_chunk_dir(output_root / "images", next_episode_index, chunk_size, key)
                    shutil.copy2(src, dst_dir / src.name)

            rec_new = dict(rec)
            rec_new["episode_index"] = next_episode_index
            rec_new["length"] = length
            new_episodes.append(rec_new)

            stats = {}
            for name in table.schema.names:
                stats[name] = compute_column_stats(table.column(name))
            if not args.skip_video_stats and video_stats:
                for key, vstats in video_stats.items():
                    stats[key] = vstats
            elif stats_by_episode:
                base_stats = stats_by_episode.get(episode_index)
                if base_stats:
                    for key in video_keys:
                        if key in base_stats:
                            stats[key] = base_stats[key]
            new_stats.append({"episode_index": next_episode_index, "stats": stats})

            total_frames += length
            next_episode_index += 1

    process_dataset(input_a, episodes_a, parquet_a, video_a, image_a, stats_a, fps_a)
    process_dataset(input_b, episodes_b, parquet_b, video_b, image_b, stats_b, fps_b)

    write_jsonl(output_root / "meta" / "episodes.jsonl", new_episodes)
    if new_stats:
        write_jsonl(output_root / "meta" / "episodes_stats.jsonl", new_stats)
    if tasks:
        write_jsonl(
            output_root / "meta" / "tasks.jsonl",
            [{"task_index": idx, "task": task} for idx, task in enumerate(tasks)],
        )

    info = dict(info_a)
    info["total_episodes"] = len(new_episodes)
    info["total_frames"] = total_frames
    info["total_tasks"] = len(tasks)
    info["chunks_size"] = chunk_size
    info["total_chunks"] = math.ceil(len(new_episodes) / chunk_size) if chunk_size else 0
    info["fps"] = target_fps
    if not args.skip_videos:
        info["total_videos"] = total_videos

    splits = info.get("splits")
    if not isinstance(splits, dict):
        splits = {}
    splits["train"] = f"0:{len(new_episodes)}"
    info["splits"] = splits

    for key in video_keys:
        meta = info["features"][key]
        vid_info = dict(meta.get("info", {}))
        vid_info["video.fps"] = target_fps
        if output_codec:
            vid_info["video.codec"] = output_codec
        if output_pix_fmt:
            vid_info["video.pix_fmt"] = output_pix_fmt
        meta["info"] = vid_info
        info["features"][key] = meta

    with (output_root / "meta" / "info.json").open("w") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

    print(f"Merged dataset written to: {output_root}")
    print(f"  episodes: {len(new_episodes)}")
    print(f"  frames:   {total_frames}")
    if not args.skip_videos:
        print(f"  videos:   {total_videos}")


if __name__ == "__main__":
    main()
