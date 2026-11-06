#!/usr/bin/env python3
"""导出 stack_cube_205 的筛选后 LeRobot 数据集，并把夹爪宽度映射成夹爪命令。

设计目标：
1. 基于 mysys / joint episode-window 的筛选结果，生成可迁移的新 LeRobot 数据集目录。
2. 不直接裁掉单帧，而是保留“命中样本所在的完整 episode”，避免破坏 OpenPI 的 action chunk 语义。
3. 在新数据集中重映射 `kept_sample_indices.npy`，保证后续训练仍然精确复现筛选样本集合。
4. 将 `observation.state[-1]` 与 `action[-1]` 从 `gripper_width` 映射为 `gripper_command`。
5. 为新数据集重算 subset `norm_stats.json`，便于后续服务器直接训练。

示例：
uv run python myscripts/export_filtered_stack_cube_205_lerobot.py \
  --dataset-root /workspace/liushuo/dataset/stack_cube_205 \
  --score-pkl /workspace/liushuo/cupid/openpi/tmp/stack_cube_205_grass_window/influence_grass_window.pkl \
  --selection-npz /workspace/liushuo/cupid/openpi/openpi-script/output/joint_episode_window_serial/stack_cube_205_joint_episode_window/selection/joint_episode_window_selection.npz \
  --output-dir /workspace/liushuo/dataset/stack_cube_205_mysys_export \
  --train-config pi05_stack_cube_205_lora \
  --overwrite
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
import shutil
import sys
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import tqdm

OPENPI_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = OPENPI_ROOT / "src"
if str(OPENPI_ROOT) not in sys.path:
    sys.path.insert(0, str(OPENPI_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import openpi.shared.normalize as normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import retrain as _retrain
from scripts import compute_norm_stats as norm_stats_script


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="导出 stack_cube_205 筛选后的 LeRobot 数据集，并将夹爪宽度映射为夹爪命令。"
    )
    parser.add_argument("--dataset-root", type=Path, required=True, help="原始 stack_cube_205 LeRobot 数据集根目录。")
    parser.add_argument("--output-dir", type=Path, required=True, help="导出的新数据集根目录。")
    parser.add_argument(
        "--kept-sample-indices",
        type=Path,
        default=None,
        help="已有的 kept_sample_indices 文件（.npy/.json/.txt）。若提供，则不再从 selection_npz 推导。",
    )
    parser.add_argument(
        "--score-pkl",
        type=Path,
        default=None,
        help="OpenPI score pickle。与 --selection-npz 搭配使用，用于把筛选窗口映射回 sample indices。",
    )
    parser.add_argument(
        "--selection-npz",
        type=Path,
        default=None,
        help="joint episode-window 筛选输出的 joint_episode_window_selection.npz。",
    )
    parser.add_argument(
        "--train-config",
        type=str,
        default="pi05_stack_cube_205_lora",
        help="用于重算 subset norm_stats 的 OpenPI 训练配置名。",
    )
    parser.add_argument(
        "--gripper-threshold",
        type=float,
        default=None,
        help="夹爪宽度转命令的阈值。默认自动用保留 episode 的 action 最小/最大值中点估计。",
    )
    parser.add_argument(
        "--gripper-open-value",
        type=float,
        default=1.0,
        help="映射后的 open 命令值，默认 1.0。",
    )
    parser.add_argument(
        "--gripper-close-value",
        type=float,
        default=0.0,
        help="映射后的 close 命令值，默认 0.0。",
    )
    parser.add_argument(
        "--skip-norm-stats",
        action="store_true",
        help="只导出数据集和 remapped kept_sample_indices，不重算 norm_stats。",
    )
    parser.add_argument(
        "--max-frames-for-norm-stats",
        type=int,
        default=None,
        help="可选：重算 norm_stats 时最多使用多少帧，默认使用全部保留样本。",
    )
    parser.add_argument("--overwrite", action="store_true", help="若输出目录已存在，则覆盖。")
    return parser.parse_args()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def _load_sample_indices(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        values = np.load(path)
    elif suffix == ".json":
        with path.open() as f:
            values = np.asarray(json.load(f))
    elif suffix in {".txt", ".list"}:
        with path.open() as f:
            values = np.asarray([int(line.strip()) for line in f if line.strip()])
    else:
        raise ValueError(f"不支持的 sample indices 格式: {path}")

    values = np.asarray(values, dtype=np.int64).reshape(-1)
    if values.size == 0:
        raise ValueError(f"sample indices 为空: {path}")
    if np.any(values < 0):
        raise ValueError(f"sample indices 含负数: {path}")
    return np.unique(values)


def _load_selection(selection_npz: Path) -> dict[str, np.ndarray]:
    with np.load(selection_npz) as data:
        required = ("episode_keep_mask", "window_episode", "window_start", "window_end", "window_keep")
        missing = [key for key in required if key not in data]
        if missing:
            raise ValueError(f"{selection_npz} 缺少必要字段: {missing}")
        return {key: np.asarray(data[key]) for key in data.files}


def _build_episode_sample_lookup(metadata: dict[str, Any]) -> dict[int, dict[str, np.ndarray]]:
    if "train_sample_episode_indices" not in metadata or "train_dataset_sample_indices" not in metadata:
        raise ValueError("score_pkl metadata 缺少 train_sample_episode_indices / train_dataset_sample_indices")

    sample_episode_indices = np.asarray(metadata["train_sample_episode_indices"], dtype=np.int64).reshape(-1)
    dataset_sample_indices = np.asarray(metadata["train_dataset_sample_indices"], dtype=np.int64).reshape(-1)
    if sample_episode_indices.shape[0] != dataset_sample_indices.shape[0]:
        raise ValueError("score_pkl metadata 的 sample shapes 不一致")

    local_frame_indices = np.empty_like(sample_episode_indices, dtype=np.int64)
    counters: dict[int, int] = {}
    for idx, episode in enumerate(sample_episode_indices.tolist()):
        episode = int(episode)
        frame_idx = counters.get(episode, 0)
        local_frame_indices[idx] = frame_idx
        counters[episode] = frame_idx + 1

    episode_lookup: dict[int, dict[str, np.ndarray]] = {}
    for episode in sorted(np.unique(sample_episode_indices).tolist()):
        mask = sample_episode_indices == int(episode)
        episode_lookup[int(episode)] = {
            "dataset_sample_indices": dataset_sample_indices[mask].astype(np.int64, copy=False),
            "local_frame_indices": local_frame_indices[mask].astype(np.int64, copy=False),
        }
    return episode_lookup


def _materialize_kept_sample_indices(
    selection: dict[str, np.ndarray],
    episode_sample_lookup: dict[int, dict[str, np.ndarray]],
) -> np.ndarray:
    window_episode = np.asarray(selection["window_episode"], dtype=np.int64).reshape(-1)
    window_start = np.asarray(selection["window_start"], dtype=np.int64).reshape(-1)
    window_end = np.asarray(selection["window_end"], dtype=np.int64).reshape(-1)
    window_keep = np.asarray(selection["window_keep"], dtype=np.int64).reshape(-1).astype(bool)

    if not (
        window_episode.shape[0] == window_start.shape[0] == window_end.shape[0] == window_keep.shape[0]
    ):
        raise ValueError("selection_npz 中 window 数组长度不一致")

    kept_chunks: list[np.ndarray] = []
    for episode, start, end, keep in zip(
        window_episode.tolist(),
        window_start.tolist(),
        window_end.tolist(),
        window_keep.tolist(),
        strict=True,
    ):
        if not keep:
            continue
        if int(end) <= int(start):
            continue
        lookup = episode_sample_lookup.get(int(episode))
        if lookup is None:
            continue
        frame_indices = lookup["local_frame_indices"]
        sample_indices = lookup["dataset_sample_indices"]
        mask = (frame_indices >= int(start)) & (frame_indices < int(end))
        if np.any(mask):
            kept_chunks.append(sample_indices[mask])

    if not kept_chunks:
        raise ValueError("筛选结果映射后没有任何保留 sample")
    return np.unique(np.concatenate(kept_chunks).astype(np.int64, copy=False))


def _resolve_kept_sample_indices(args: argparse.Namespace) -> tuple[np.ndarray, dict[str, Any]]:
    if args.kept_sample_indices is not None:
        indices = _load_sample_indices(args.kept_sample_indices)
        return indices, {
            "source_kind": "kept_sample_indices",
            "kept_sample_indices_path": str(args.kept_sample_indices),
        }

    if args.selection_npz is None or args.score_pkl is None:
        raise ValueError("请提供 --kept-sample-indices，或同时提供 --selection-npz 与 --score-pkl")

    payload = _retrain._load_score_payload(args.score_pkl)  # noqa: SLF001
    if not isinstance(payload, dict) or "metadata" not in payload or not isinstance(payload["metadata"], dict):
        raise ValueError(f"{args.score_pkl} 不包含预期 metadata")
    selection = _load_selection(args.selection_npz)
    episode_lookup = _build_episode_sample_lookup(payload["metadata"])
    indices = _materialize_kept_sample_indices(selection, episode_lookup)
    return indices, {
        "source_kind": "selection_npz",
        "selection_npz": str(args.selection_npz),
        "score_pkl": str(args.score_pkl),
    }


def _load_episode_rows(dataset_root: Path) -> list[dict[str, Any]]:
    rows = _load_jsonl(dataset_root / "meta" / "episodes.jsonl")
    rows = sorted(rows, key=lambda row: int(row["episode_index"]))
    return rows


def _build_episode_offsets(episode_rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    episode_ids = np.asarray([int(row["episode_index"]) for row in episode_rows], dtype=np.int64)
    lengths = np.asarray([int(row["length"]) for row in episode_rows], dtype=np.int64)
    starts = np.zeros_like(lengths)
    if lengths.size > 1:
        starts[1:] = np.cumsum(lengths[:-1], dtype=np.int64)
    ends = starts + lengths
    return episode_ids, starts, ends


def _map_sample_indices_to_episode_local(
    sample_indices: np.ndarray,
    episode_ids: np.ndarray,
    episode_starts: np.ndarray,
    episode_ends: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    positions = np.searchsorted(episode_ends, sample_indices, side="right")
    if np.any(positions < 0) or np.any(positions >= len(episode_ids)):
        raise ValueError("存在 sample index 超出 episode 边界")

    mapped_episode_ids = episode_ids[positions]
    local_frames = sample_indices - episode_starts[positions]
    invalid = (sample_indices < episode_starts[positions]) | (sample_indices >= episode_ends[positions])
    if np.any(invalid):
        bad = sample_indices[np.flatnonzero(invalid)[0]]
        raise ValueError(f"sample index 映射失败: {bad}")
    return mapped_episode_ids.astype(np.int64), local_frames.astype(np.int64)


def _width_to_command(values: np.ndarray, *, threshold: float, close_value: float, open_value: float) -> np.ndarray:
    out = np.full(values.shape, open_value, dtype=np.float32)
    out[np.asarray(values, dtype=np.float32) <= float(threshold)] = float(close_value)
    return out


def _replace_last_dim_as_command(
    column: pa.ChunkedArray,
    *,
    threshold: float,
    close_value: float,
    open_value: float,
    field_name: str,
) -> pa.Array:
    data = np.asarray(column.to_pylist(), dtype=np.float32)
    if data.ndim != 2 or data.shape[1] < 1:
        raise ValueError(f"{field_name} 维度异常，期望二维且最后一维存在，实际为 {data.shape}")
    data[:, -1] = _width_to_command(
        data[:, -1],
        threshold=threshold,
        close_value=close_value,
        open_value=open_value,
    )
    return pa.array(data.tolist(), type=column.type)


def _infer_gripper_threshold(dataset_root: Path, selected_episodes: list[int]) -> float:
    mins: list[float] = []
    maxs: list[float] = []
    for episode_id in selected_episodes:
        path = dataset_root / "data" / "chunk-000" / f"episode_{episode_id:06d}.parquet"
        table = pq.read_table(path, columns=["action"])
        actions = np.asarray(table["action"].to_pylist(), dtype=np.float32)
        if actions.ndim != 2 or actions.shape[1] < 1:
            raise ValueError(f"{path} action 维度异常: {actions.shape}")
        mins.append(float(np.min(actions[:, -1])))
        maxs.append(float(np.max(actions[:, -1])))
    if not mins or not maxs:
        raise ValueError("无法估计夹爪阈值，未读到任何保留 episode")
    return (min(mins) + max(maxs)) / 2.0


def _compute_vector_stats(array: np.ndarray) -> dict[str, Any]:
    data = np.asarray(array)
    return {
        "min": np.min(data, axis=0).tolist(),
        "max": np.max(data, axis=0).tolist(),
        "mean": np.mean(data, axis=0).tolist(),
        "std": np.std(data, axis=0).tolist(),
        "count": [int(data.shape[0])],
    }


def _compute_scalar_stats(array: np.ndarray) -> dict[str, Any]:
    data = np.asarray(array).reshape(-1)
    return {
        "min": [data.min().item()],
        "max": [data.max().item()],
        "mean": [float(data.mean())],
        "std": [float(data.std())],
        "count": [int(data.shape[0])],
    }


def _load_source_episode_stats(dataset_root: Path) -> dict[int, dict[str, Any]]:
    path = dataset_root / "meta" / "episodes_stats.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"原始数据集缺少 {path}")
    rows = _load_jsonl(path)
    return {int(row["episode_index"]): row for row in rows}


def _build_export_episode_stats(
    *,
    source_stats_row: dict[str, Any],
    new_episode_id: int,
    new_index_start: int,
    new_table: pa.Table,
) -> dict[str, Any]:
    num_rows = int(new_table.num_rows)
    state = np.asarray(new_table["observation.state"].to_pylist(), dtype=np.float32)
    action = np.asarray(new_table["action"].to_pylist(), dtype=np.float32)
    timestamp = np.asarray(new_table["timestamp"].to_pylist())
    frame_index = np.arange(num_rows, dtype=np.int64)
    global_index = np.arange(new_index_start, new_index_start + num_rows, dtype=np.int64)
    episode_index = np.full((num_rows,), new_episode_id, dtype=np.int64)
    task_index = np.asarray(new_table["task_index"].to_pylist(), dtype=np.int64)

    stats = json.loads(json.dumps(source_stats_row["stats"]))
    stats["observation.state"] = _compute_vector_stats(state)
    stats["action"] = _compute_vector_stats(action)
    stats["timestamp"] = _compute_scalar_stats(timestamp)
    stats["frame_index"] = _compute_scalar_stats(frame_index)
    stats["episode_index"] = _compute_scalar_stats(episode_index)
    stats["index"] = _compute_scalar_stats(global_index)
    stats["task_index"] = _compute_scalar_stats(task_index)
    return {
        "episode_index": int(new_episode_id),
        "stats": stats,
    }


def _copy_selected_episodes(
    *,
    dataset_root: Path,
    output_dir: Path,
    selected_old_episode_ids: list[int],
    old_to_new_episode: dict[int, int],
    gripper_threshold: float,
    gripper_close_value: float,
    gripper_open_value: float,
    source_episode_stats: dict[int, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[int, int], int, list[dict[str, Any]]]:
    output_data_dir = output_dir / "data" / "chunk-000"
    output_data_dir.mkdir(parents=True, exist_ok=True)

    output_video_root = output_dir / "videos" / "chunk-000"
    source_video_root = dataset_root / "videos"

    new_episode_rows: list[dict[str, Any]] = []
    new_episode_stats_rows: list[dict[str, Any]] = []
    new_episode_starts: dict[int, int] = {}
    global_index = 0
    copied_video_count = 0

    for old_episode_id in selected_old_episode_ids:
        new_episode_id = old_to_new_episode[old_episode_id]
        src_parquet = dataset_root / "data" / "chunk-000" / f"episode_{old_episode_id:06d}.parquet"
        dst_parquet = output_data_dir / f"episode_{new_episode_id:06d}.parquet"
        table = pq.read_table(src_parquet)
        num_rows = table.num_rows

        new_episode_starts[new_episode_id] = global_index
        new_arrays: list[pa.Array | pa.ChunkedArray] = []
        for field in table.schema:
            name = field.name
            if name == "episode_index":
                arr = pa.array(np.full((num_rows,), new_episode_id, dtype=np.int64), type=field.type)
            elif name == "frame_index":
                arr = pa.array(np.arange(num_rows, dtype=np.int64), type=field.type)
            elif name == "index":
                arr = pa.array(np.arange(global_index, global_index + num_rows, dtype=np.int64), type=field.type)
            elif name == "observation.state":
                arr = _replace_last_dim_as_command(
                    table[name],
                    threshold=gripper_threshold,
                    close_value=gripper_close_value,
                    open_value=gripper_open_value,
                    field_name=name,
                )
            elif name == "action":
                arr = _replace_last_dim_as_command(
                    table[name],
                    threshold=gripper_threshold,
                    close_value=gripper_close_value,
                    open_value=gripper_open_value,
                    field_name=name,
                )
            else:
                arr = table[name]
            new_arrays.append(arr)

        new_table = pa.Table.from_arrays(new_arrays, schema=table.schema)
        pq.write_table(new_table, dst_parquet)
        new_episode_stats_rows.append(
            _build_export_episode_stats(
                source_stats_row=source_episode_stats[old_episode_id],
                new_episode_id=new_episode_id,
                new_index_start=global_index,
                new_table=new_table,
            )
        )

        new_episode_rows.append(
            {
                "episode_index": int(new_episode_id),
                "tasks": ["Pick up the blue block and place it on top of the red block."],
                "length": int(num_rows),
            }
        )
        global_index += num_rows

        if source_video_root.exists():
            for src_video in sorted(source_video_root.glob(f"chunk-*/**/episode_{old_episode_id:06d}.mp4")):
                camera_dir = src_video.parent.name
                dst_video = output_video_root / camera_dir / f"episode_{new_episode_id:06d}.mp4"
                dst_video.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_video, dst_video)
                copied_video_count += 1

    return new_episode_rows, new_episode_starts, copied_video_count, new_episode_stats_rows


def _write_dataset_metadata(
    *,
    dataset_root: Path,
    output_dir: Path,
    new_episode_rows: list[dict[str, Any]],
    new_episode_stats_rows: list[dict[str, Any]],
    copied_video_count: int,
) -> None:
    meta_dir = output_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    with (dataset_root / "meta" / "info.json").open() as f:
        info = json.load(f)

    total_frames = int(sum(int(row["length"]) for row in new_episode_rows))
    total_episodes = int(len(new_episode_rows))

    info["total_episodes"] = total_episodes
    info["total_frames"] = total_frames
    info["total_videos"] = int(copied_video_count)
    info["total_chunks"] = 1
    info["splits"] = {"train": f"0:{total_episodes}"}

    for key in ("observation.state", "action"):
        names = info.get("features", {}).get(key, {}).get("names")
        if isinstance(names, list) and names:
            names = list(names)
            names[-1] = "gripper_command"
            info["features"][key]["names"] = names

    (meta_dir / "info.json").write_text(json.dumps(info, indent=4, ensure_ascii=True) + "\n")
    _write_jsonl(meta_dir / "episodes.jsonl", new_episode_rows)
    _write_jsonl(meta_dir / "episodes_stats.jsonl", new_episode_stats_rows)

    src_tasks = dataset_root / "meta" / "tasks.jsonl"
    if src_tasks.exists():
        tasks = _load_jsonl(src_tasks)
        for row in tasks:
            if int(row.get("task_index", -1)) == 0 and "task" in row:
                row["task"] = "Pick up the blue block and place it on top of the red block."
        _write_jsonl(meta_dir / "tasks.jsonl", tasks)


def _remap_kept_sample_indices(
    kept_sample_indices: np.ndarray,
    *,
    episode_ids: np.ndarray,
    episode_starts: np.ndarray,
    episode_ends: np.ndarray,
    old_to_new_episode: dict[int, int],
    new_episode_starts: dict[int, int],
) -> np.ndarray:
    mapped_episode_ids, local_frames = _map_sample_indices_to_episode_local(
        kept_sample_indices,
        episode_ids,
        episode_starts,
        episode_ends,
    )
    remapped = np.empty_like(kept_sample_indices, dtype=np.int64)
    for idx, (old_episode_id, local_frame) in enumerate(zip(mapped_episode_ids.tolist(), local_frames.tolist(), strict=True)):
        new_episode_id = old_to_new_episode[int(old_episode_id)]
        remapped[idx] = int(new_episode_starts[new_episode_id]) + int(local_frame)
    return remapped


def _build_export_data_factory(
    data_factory: _config.DataConfigFactory,
    *,
    dataset_root: Path,
    sample_indices_path: Path,
) -> _config.DataConfigFactory:
    return dataclasses.replace(
        data_factory,
        repo_id=str(dataset_root),
        sample_indices_path=str(sample_indices_path),
        assets=_config.AssetsConfig(
            assets_dir=str(dataset_root.parent),
            asset_id=dataset_root.name,
        ),
    )


def _compute_subset_norm_stats(
    *,
    train_config_name: str,
    dataset_root: Path,
    sample_indices_path: Path,
    output_dir: Path,
    max_frames: int | None,
) -> None:
    train_cfg = _config.get_config(train_config_name)
    data_factory = _build_export_data_factory(
        train_cfg.data,
        dataset_root=dataset_root,
        sample_indices_path=sample_indices_path,
    )
    data_cfg = data_factory.create(train_cfg.assets_dirs, train_cfg.model)
    dataset = _data_loader.create_torch_dataset(data_cfg, train_cfg.model.action_horizon, train_cfg.model)
    dataset = _data_loader.TransformedDataset(
        dataset,
        [
            *data_cfg.repack_transforms.inputs,
            *data_cfg.data_transforms.inputs,
            norm_stats_script.RemoveStrings(),
        ],
    )

    if max_frames is not None and max_frames < len(dataset):
        num_batches = max(1, max_frames // train_cfg.batch_size)
        shuffle = True
    else:
        num_batches = max(1, len(dataset) // train_cfg.batch_size)
        shuffle = False

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=train_cfg.batch_size,
        shuffle=shuffle,
        num_workers=train_cfg.num_workers,
        collate_fn=_data_loader._collate_fn,  # noqa: SLF001
        drop_last=True,
    )

    stats = {key: normalize.RunningStats() for key in ("state", "actions")}
    for batch in tqdm.tqdm(data_loader, total=num_batches, desc="Computing subset norm_stats"):
        for key in stats:
            stats[key].update(np.asarray(batch[key]))

    norm_stats = {key: running.get_statistics() for key, running in stats.items()}
    normalize.save(output_dir, norm_stats)


def _compute_subset_norm_stats_with_fallback(
    *,
    train_config_name: str,
    dataset_root: Path,
    sample_indices_path: Path,
    output_dir: Path,
    max_frames: int | None,
) -> bool:
    try:
        _compute_subset_norm_stats(
            train_config_name=train_config_name,
            dataset_root=dataset_root,
            sample_indices_path=sample_indices_path,
            output_dir=output_dir,
            max_frames=max_frames,
        )
        return False
    except PermissionError:
        train_cfg = _config.get_config(train_config_name)
        single_worker_factory = dataclasses.replace(
            _build_export_data_factory(
                train_cfg.data,
                dataset_root=dataset_root,
                sample_indices_path=sample_indices_path,
            ),
        )
        single_worker_cfg = dataclasses.replace(train_cfg, data=single_worker_factory, num_workers=0)
        data_cfg = single_worker_cfg.data.create(single_worker_cfg.assets_dirs, single_worker_cfg.model)
        dataset = _data_loader.create_torch_dataset(data_cfg, single_worker_cfg.model.action_horizon, single_worker_cfg.model)
        dataset = _data_loader.TransformedDataset(
            dataset,
            [
                *data_cfg.repack_transforms.inputs,
                *data_cfg.data_transforms.inputs,
                norm_stats_script.RemoveStrings(),
            ],
        )
        if max_frames is not None and max_frames < len(dataset):
            num_batches = max(1, max_frames // single_worker_cfg.batch_size)
            shuffle = True
        else:
            num_batches = max(1, len(dataset) // single_worker_cfg.batch_size)
            shuffle = False
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=single_worker_cfg.batch_size,
            shuffle=shuffle,
            num_workers=0,
            collate_fn=_data_loader._collate_fn,  # noqa: SLF001
            drop_last=True,
        )
        stats = {key: normalize.RunningStats() for key in ("state", "actions")}
        for batch in tqdm.tqdm(data_loader, total=num_batches, desc="Computing subset norm_stats"):
            for key in stats:
                stats[key].update(np.asarray(batch[key]))
        norm_stats = {key: running.get_statistics() for key, running in stats.items()}
        normalize.save(output_dir, norm_stats)
        return True


def _write_export_summary(
    *,
    output_dir: Path,
    args: argparse.Namespace,
    source_meta: dict[str, Any],
    kept_sample_indices: np.ndarray,
    remapped_kept_sample_indices: np.ndarray,
    selected_old_episode_ids: list[int],
    gripper_threshold: float,
    source_details: dict[str, Any],
) -> None:
    summary = {
        "method": "export_filtered_stack_cube_205_lerobot",
        "dataset_root": str(args.dataset_root),
        "output_dir": str(args.output_dir),
        "train_config": args.train_config,
        "selected_episode_count": int(len(selected_old_episode_ids)),
        "selected_old_episode_ids": [int(x) for x in selected_old_episode_ids],
        "kept_sample_count": int(kept_sample_indices.shape[0]),
        "remapped_sample_count": int(remapped_kept_sample_indices.shape[0]),
        "gripper_threshold": float(gripper_threshold),
        "gripper_open_value": float(args.gripper_open_value),
        "gripper_close_value": float(args.gripper_close_value),
        "source_total_episodes": int(source_meta["total_episodes"]),
        "source_total_frames": int(source_meta["total_frames"]),
        **source_details,
    }
    (output_dir / "export_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n")


def _write_train_hint(
    *,
    output_dir: Path,
    train_config_name: str,
) -> None:
    dataset_root = output_dir.resolve()
    sample_indices = (output_dir / "kept_sample_indices.npy").resolve()
    parent = output_dir.parent.resolve()
    asset_id = output_dir.name

    cmd = (
        "CUDA_VISIBLE_DEVICES=0 "
        "XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 "
        "WANDB_MODE=disabled "
        "uv run scripts/train.py "
        f"{train_config_name} "
        f"--exp-name {asset_id}_train "
        f"--data.repo-id {dataset_root} "
        f"--data.sample-indices-path {sample_indices} "
        f"--data.assets.assets-dir {parent} "
        f"--data.assets.asset-id {asset_id} "
        "--overwrite"
    )
    (output_dir / "train_command.sh").write_text(cmd + "\n")


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"{args.output_dir} 已存在，请显式传 --overwrite")
        shutil.rmtree(args.output_dir)

    kept_sample_indices, source_details = _resolve_kept_sample_indices(args)
    episode_rows = _load_episode_rows(args.dataset_root)
    episode_ids, episode_starts, episode_ends = _build_episode_offsets(episode_rows)
    mapped_episode_ids, _ = _map_sample_indices_to_episode_local(
        kept_sample_indices,
        episode_ids,
        episode_starts,
        episode_ends,
    )
    selected_old_episode_ids = sorted(np.unique(mapped_episode_ids).astype(np.int64).tolist())
    old_to_new_episode = {old_id: new_id for new_id, old_id in enumerate(selected_old_episode_ids)}
    source_episode_stats = _load_source_episode_stats(args.dataset_root)

    with (args.dataset_root / "meta" / "info.json").open() as f:
        source_info = json.load(f)

    gripper_threshold = (
        float(args.gripper_threshold)
        if args.gripper_threshold is not None
        else _infer_gripper_threshold(args.dataset_root, selected_old_episode_ids)
    )

    new_episode_rows, new_episode_starts, copied_video_count, new_episode_stats_rows = _copy_selected_episodes(
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        selected_old_episode_ids=selected_old_episode_ids,
        old_to_new_episode=old_to_new_episode,
        gripper_threshold=gripper_threshold,
        gripper_close_value=args.gripper_close_value,
        gripper_open_value=args.gripper_open_value,
        source_episode_stats=source_episode_stats,
    )
    _write_dataset_metadata(
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        new_episode_rows=new_episode_rows,
        new_episode_stats_rows=new_episode_stats_rows,
        copied_video_count=copied_video_count,
    )

    remapped_kept_sample_indices = _remap_kept_sample_indices(
        kept_sample_indices,
        episode_ids=episode_ids,
        episode_starts=episode_starts,
        episode_ends=episode_ends,
        old_to_new_episode=old_to_new_episode,
        new_episode_starts=new_episode_starts,
    )
    np.save(args.output_dir / "kept_sample_indices.npy", remapped_kept_sample_indices.astype(np.int64))
    np.save(args.output_dir / "original_kept_sample_indices.npy", kept_sample_indices.astype(np.int64))

    if not args.skip_norm_stats:
        used_single_worker = _compute_subset_norm_stats_with_fallback(
            train_config_name=args.train_config,
            dataset_root=args.output_dir,
            sample_indices_path=args.output_dir / "kept_sample_indices.npy",
            output_dir=args.output_dir,
            max_frames=args.max_frames_for_norm_stats,
        )
        if used_single_worker:
            print("norm_stats fallback: 使用 num_workers=0 重试成功。")

    _write_export_summary(
        output_dir=args.output_dir,
        args=args,
        source_meta=source_info,
        kept_sample_indices=kept_sample_indices,
        remapped_kept_sample_indices=remapped_kept_sample_indices,
        selected_old_episode_ids=selected_old_episode_ids,
        gripper_threshold=gripper_threshold,
        source_details=source_details,
    )
    _write_train_hint(output_dir=args.output_dir, train_config_name=args.train_config)

    print(f"导出完成: {args.output_dir}")
    print(f"保留 episode 数: {len(selected_old_episode_ids)}")
    print(f"原始 kept samples: {kept_sample_indices.shape[0]}")
    print(f"重映射 kept samples: {remapped_kept_sample_indices.shape[0]}")
    print(f"gripper threshold: {gripper_threshold:.6f}")
    print(f"训练命令已写入: {args.output_dir / 'train_command.sh'}")


if __name__ == "__main__":
    main()
