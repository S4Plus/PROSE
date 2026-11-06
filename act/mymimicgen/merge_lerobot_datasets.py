#!/usr/bin/env python3
"""
合并多个 lerobot 数据集目录（parquet 格式）到一个新目录，顺序拼接 episode。

假设输入目录结构：
  data/chunk-000/episode_000000.parquet
  meta/episodes.jsonl
  meta/episodes_stats.jsonl
  meta/info.json
  meta/tasks.jsonl
  norm_stats.json

行为：
  - 重新编号 episode_index 从 0 开始连续；
  - 修正每条 parquet 的 episode_index、index（全局样本下标）；
  - 复制 episodes.jsonl / episodes_stats.jsonl，并调整索引；
  - info.json 中 total_episodes/total_frames 更新为合计值；
  - 复制 tasks.jsonl、norm_stats.json（取第一份），可选复制 actions.txt；
  - 输出 data 目录按 chunk_size 分片（默认 1000）。

用法示例：
python mymimicgen/merge_lerobot_datasets.py \
  --inputs /path/to/ds1 /path/to/ds2 \
  --output /path/to/merged_ds \
  --chunk-size 1000 \
  --copy-actions
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import re
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pyarrow as pa
import pyarrow.parquet as pq


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="合并 lerobot parquet 数据集")
    p.add_argument("--inputs", nargs="+", required=True, type=Path, help="输入数据集目录列表")
    p.add_argument("--output", type=Path, required=True, help="输出目录")
    p.add_argument("--chunk-size", type=int, default=1000, help="每个 chunk 的最大 episode 数")
    p.add_argument("--overwrite", action="store_true", help="输出已存在时先删除")
    p.add_argument("--copy-actions", action="store_true", help="若存在 episode_xxx_actions.txt，一并复制并改名")
    p.add_argument("--repo-id", type=str, default=None, help="可选：在输出 meta/info.json 中写入 repo_id 字段")
    return p.parse_args()


def find_parquet_paths(data_root: Path) -> Dict[int, Path]:
    """扫描 data 下所有 episode_XXXXXX.parquet，返回 {old_idx: path}。"""
    mapping: Dict[int, Path] = {}
    for path in data_root.rglob("episode_*[0-9].parquet"):
        m = re.search(r"episode_(\d+)\.parquet$", path.name)
        if not m:
            continue
        idx = int(m.group(1))
        mapping[idx] = path
    if not mapping:
        raise FileNotFoundError(f"未找到 parquet：{data_root}")
    return mapping


def load_jsonl(path: Path) -> List[dict]:
    with path.open("r") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")


def update_table(table: pa.Table, new_ep: int, start_idx: int) -> pa.Table:
    """更新 episode_index 与 index 列，保留其他字段与 metadata。"""
    length = table.num_rows
    meta = table.schema.metadata

    def replace_column(name: str, values: Iterable) -> pa.Table:
        nonlocal table
        if name not in table.schema.names:
            return table
        field = table.schema.field(name)
        arr = pa.array(list(values), type=field.type)
        table = table.set_column(table.schema.get_field_index(name), name, arr)
        return table

    table = replace_column("episode_index", [new_ep] * length)
    table = replace_column("index", range(start_idx, start_idx + length))
    if meta is not None:
        table = table.replace_schema_metadata(meta)
    return table


def recompute_index_stats(length: int, offset: int) -> dict:
    """给定长度与偏移，重建 index 统计字段。"""
    start = offset
    end = offset + length - 1
    mean = (start + end) / 2
    # 离散均匀分布 0..L-1 的方差 = (L^2-1)/12
    var = (length * length - 1) / 12 if length > 1 else 0.0
    std = math.sqrt(var)
    return {"min": [start], "max": [end], "mean": [mean], "std": [std], "count": [length]}


def find_vision_paths(vision_root: Path, ext: str = ".mp4") -> Dict[str, Dict[int, Path]]:
    """扫描 videos 或 images 目录，返回 {video_key: {old_idx: path}}。"""
    mapping: Dict[str, Dict[int, Path]] = {}
    if not vision_root.is_dir():
        return mapping
    
    # 假设结构为 chunk-XXX/video_key/episode_YYYYYY.mp4
    # 或者 video_key/episode_YYYYYY.mp4
    for path in vision_root.rglob(f"episode_*{ext}"):
        m = re.search(r"episode_(\d+)" + re.escape(ext) + r"$", path.name)
        if not m:
            continue
        idx = int(m.group(1))
        # 尝试从路径中找 video_key
        # 通常是 path.parent.name
        key = path.parent.name
        if key.startswith("chunk-"):
            # 如果结构是 key/chunk-XXX/episode_YYY
            key = path.parent.parent.name
        
        if key not in mapping:
            mapping[key] = {}
        mapping[key][idx] = path
    return mapping


def merge_dataset(
    inputs: List[Path],
    output: Path,
    chunk_size: int,
    copy_actions: bool,
    overwrite: bool,
    repo_id: str | None,
) -> None:
    if output.exists():
        if not overwrite:
            raise SystemExit(f"输出目录已存在: {output}（若需覆盖请加 --overwrite）")
        print(f"清理输出目录: {output}")
        shutil.rmtree(output)
    
    (output / "data").mkdir(parents=True, exist_ok=True)
    (output / "meta").mkdir(parents=True, exist_ok=True)

    new_episodes: List[dict] = []
    new_stats: List[dict] = []
    total_frames = 0
    total_videos_count = 0

    base_info = None
    base_norm = None
    tasks_path = None
    
    video_keys = []

    chunk_dirs: Dict[int, Path] = {}

    def get_chunk_dir(base_name: str, ep_idx: int, key: str = None) -> Path:
        chunk_id = ep_idx // chunk_size
        path = output / base_name / f"chunk-{chunk_id:03d}"
        if key:
            path = path / key
        path.mkdir(parents=True, exist_ok=True)
        return path

    for ds_idx, ds_root in enumerate(inputs):
        ds_root = ds_root.expanduser().resolve()
        print(f"\n正在处理 [{ds_idx+1}/{len(inputs)}]: {ds_root}")
        data_root = ds_root / "data"
        meta_root = ds_root / "meta"
        video_root = ds_root / "videos"
        image_root = ds_root / "images"

        if not data_root.is_dir() or not meta_root.is_dir():
            print(f"警告: 跳过非法目录（缺少 data/meta）：{ds_root}")
            continue

        if base_info is None:
            info_path = meta_root / "info.json"
            base_info = json.load(info_path.open("r"))
            norm_path = ds_root / "norm_stats.json"
            if norm_path.is_file():
                with norm_path.open("r") as f:
                    base_norm = f.read()
            t_path = meta_root / "tasks.jsonl"
            if t_path.is_file():
                tasks_path = t_path
            
            # 记录视频/图像 Key
            video_keys = [k for k, v in base_info.get("features", {}).items() if v.get("dtype") == "video"]
            print(f"发现视频特征: {video_keys}")

        ep_meta = load_jsonl(meta_root / "episodes.jsonl")
        stats_meta = {}
        stats_path = meta_root / "episodes_stats.jsonl"
        if stats_path.is_file():
            stats_meta = {int(x["episode_index"]): x["stats"] for x in load_jsonl(stats_path)}

        parquet_map = find_parquet_paths(data_root)
        video_map = find_vision_paths(video_root, ".mp4")
        image_map = find_vision_paths(image_root, ".png") # 或者 .jpg，看实际情况

        for rec in ep_meta:
            old_ep = int(rec["episode_index"])
            if old_ep not in parquet_map:
                print(f"  警告: 找不到 episode_{old_ep:06d}.parquet，跳过")
                continue
            
            pq_path = parquet_map[old_ep]
            table = pq.read_table(pq_path)
            length = table.num_rows
            new_ep = len(new_episodes)
            start_idx = total_frames

            table = update_table(table, new_ep=new_ep, start_idx=start_idx)

            # 写入新的 chunk (Parquet)
            chunk_data_dir = get_chunk_dir("data", new_ep)
            new_name_base = f"episode_{new_ep:06d}"
            pq.write_table(table, chunk_data_dir / f"{new_name_base}.parquet")

            # 复制视频
            for v_key in video_keys:
                if v_key in video_map and old_ep in video_map[v_key]:
                    v_src = video_map[v_key][old_ep]
                    v_dst_dir = get_chunk_dir("videos", new_ep, v_key)
                    shutil.copy2(v_src, v_dst_dir / f"{new_name_base}.mp4")
                    total_videos_count += 1
            
            # 复制图像 (如果有)
            for i_key in image_map:
                if old_ep in image_map[i_key]:
                    i_src = image_map[i_key][old_ep]
                    i_dst_dir = get_chunk_dir("images", new_ep, i_key)
                    shutil.copy2(i_src, i_dst_dir / f"{new_name_base}.png")

            if copy_actions:
                act_src = pq_path.with_name(pq_path.stem + "_actions.txt")
                if act_src.is_file():
                    act_dst = chunk_data_dir / f"{new_name_base}_actions.txt"
                    shutil.copy2(act_src, act_dst)

            # 更新 episode 元信息
            rec_new = copy.deepcopy(rec)
            rec_new["episode_index"] = new_ep
            new_episodes.append(rec_new)

            # 更新 stats
            if old_ep in stats_meta:
                s = copy.deepcopy(stats_meta[old_ep])
                s["episode_index"] = {"min": [new_ep], "max": [new_ep], "mean": [float(new_ep)], "std": [0.0], "count": [length]}
                s["index"] = recompute_index_stats(length, start_idx)
                new_stats.append({"episode_index": new_ep, "stats": s})

            total_frames += length
            if (new_ep + 1) % 100 == 0:
                print(f"  已处理 {new_ep + 1} 个 episode...")

    # 写 meta
    write_jsonl(output / "meta" / "episodes.jsonl", new_episodes)
    if new_stats:
        write_jsonl(output / "meta" / "episodes_stats.jsonl", new_stats)

    # info.json 更新总量
    if base_info:
        base_info["total_episodes"] = len(new_episodes)
        base_info["total_frames"] = total_frames
        base_info["total_videos"] = total_videos_count
        base_info["chunks_size"] = chunk_size
        base_info["total_chunks"] = math.ceil(len(new_episodes) / chunk_size)
        if repo_id is not None:
            base_info["repo_id"] = repo_id
        
        with (output / "meta" / "info.json").open("w") as f:
            json.dump(base_info, f, ensure_ascii=False, indent=2)

    if tasks_path:
        shutil.copy2(tasks_path, output / "meta" / "tasks.jsonl")
    if base_norm:
        (output / "norm_stats.json").write_text(base_norm)

    print(f"\n合并完成: {output}")
    print(f"  总 episodes: {len(new_episodes)}")
    print(f"  总 frames:   {total_frames}")
    print(f"  总 videos:   {total_videos_count}")
    print(f"  总 chunks:   {base_info.get('total_chunks', 0) if base_info else 0}")


def main() -> None:
    args = parse_args()
    merge_dataset(
        inputs=args.inputs,
        output=args.output,
        chunk_size=args.chunk_size,
        copy_actions=args.copy_actions,
        overwrite=args.overwrite,
        repo_id=args.repo_id,
    )


if __name__ == "__main__":
    main()
