#!/usr/bin/env python3
"""构建 mh + ph 偏好补充 / 随机补充两组对比数据集。

目标：
1. 物化 `can_mh` 两阶段过滤后的基础数据集。
2. 从 `can_ph` 中按偏好原型筛出补充子集。
3. 从 `can_ph` 中按相同数量随机筛出对照子集。
4. 分别合并为两份可直接训练策略模型的 LeRobot 数据集。

说明：
- 这里故意不使用 quality / influence 分数，只使用偏好原型。
- 默认偏好原型仍来自 `episode_comparison.csv` 中的 `pi05_only` cohort。
- 若用于 ACT 侧实验，可显式传入 `--prototype-source act_keep` 或 `act_topk`。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


DEFAULT_CAN_MH_DATASET = Path("/workspace/liushuo/cupid/datasets/lerobot/can_mh")
DEFAULT_CAN_MH_KEEP = Path(
    "/workspace/liushuo/cupid/openpi/openpi-script/output/joint_episode_window_serial/"
    "can_mh_joint_episode_window/selection/kept_episodes.txt"
)
DEFAULT_CAN_PH_DATASET = Path("/workspace/liushuo/robomimic/robomimic/datasets/lerobot/can_ph")
DEFAULT_COMPARE_CSV = Path("/workspace/liushuo/cupid/output/compare_act_openpi_can_mh_retest/episode_comparison.csv")
DEFAULT_OUTPUT_ROOT = Path("/workspace/liushuo/cupid/output/openpi05_canph_ablation")

PREFERENCE_FEATURES = (
    "length",
    "action_jitter",
    "state_jitter",
    "state_displacement",
)


@dataclass
class PrototypeStats:
    """偏好原型的均值和标准差。"""

    mean: dict[str, float]
    std: dict[str, float]
    count: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="构建 OpenPI05 的 can_mh/can_ph 对比实验数据集。")
    parser.add_argument("--can-mh-dataset", type=Path, default=DEFAULT_CAN_MH_DATASET)
    parser.add_argument("--can-mh-keep-list", type=Path, default=DEFAULT_CAN_MH_KEEP)
    parser.add_argument("--can-ph-dataset", type=Path, default=DEFAULT_CAN_PH_DATASET)
    parser.add_argument("--compare-csv", type=Path, default=DEFAULT_COMPARE_CSV)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--prototype-source",
        type=str,
        default="cohort",
        choices=("act_keep", "act_topk", "pi05_keep", "pi05_topk", "cohort"),
        help=(
            "偏好原型的来源。"
            "act_keep=优先使用 episode_comparison.csv 中 act_keep==1 的样本；"
            "act_topk=按 ACT 排名分数直接取 top 样本；"
            "pi05_keep=优先使用 episode_comparison.csv 中 pi05_keep==1 的样本；"
            "pi05_topk=按 OpenPI05 排名分数直接取 top 样本；"
            "cohort=兼容旧逻辑，按 cohort 字段筛选。"
        ),
    )
    parser.add_argument(
        "--cohort",
        type=str,
        default="pi05_only",
        help="仅在 --prototype-source=cohort 时生效，用于提取偏好原型的 cohort 名称。",
    )
    parser.add_argument(
        "--prototype-top-k",
        type=int,
        default=None,
        help=(
            "仅在 --prototype-source=act_topk / act_keep 回退 / pi05_topk / pi05_keep 回退时生效；"
            "指定原型样本数。"
        ),
    )
    parser.add_argument(
        "--prototype-top-ratio",
        type=float,
        default=0.2,
        help="仅在 --prototype-top-k 为空时生效；按 ACT 排名选择前多少比例样本构建原型。",
    )
    parser.add_argument(
        "--supplement-count",
        type=int,
        default=None,
        help="补充的 can_ph episode 数。若不填，则按 supplement-ratio 从两阶段 can_mh 基础集推导。",
    )
    parser.add_argument(
        "--supplement-ratio",
        type=float,
        default=0.2,
        help="当 supplement-count 为空时，按该比例乘以 can_mh 两阶段基础集规模。",
    )
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _safe_std(values: list[float]) -> float:
    std = float(np.std(np.asarray(values, dtype=np.float64)))
    return max(std, 1e-6)


def _parse_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except Exception:
        return None


def _select_model_top_rows(
    rows: list[dict[str, Any]],
    *,
    prefix: str,
    top_k: int | None,
    top_ratio: float,
) -> list[dict[str, Any]]:
    ranked_by_desc = []
    for row in rows:
        rank_desc = _parse_float(row.get(f"{prefix}_rank_desc"))
        if rank_desc is None:
            continue
        ranked_by_desc.append((rank_desc, row))
    if ranked_by_desc:
        ranked_by_desc.sort(key=lambda x: x[0])
        ordered_rows = [row for _, row in ranked_by_desc]
    else:
        ranked_by_score = []
        for row in rows:
            score = _parse_float(row.get(f"{prefix}_score_raw"))
            if score is None:
                continue
            ranked_by_score.append((score, row))
        if not ranked_by_score:
            raise ValueError(
                f"对比表中既没有 {prefix}_rank_desc，也没有 {prefix}_score_raw，无法提取 {prefix} 原型。"
            )
        ranked_by_score.sort(key=lambda x: x[0], reverse=True)
        ordered_rows = [row for _, row in ranked_by_score]

    if top_k is None:
        keep_n = int(math.ceil(len(ordered_rows) * float(top_ratio)))
    else:
        keep_n = int(top_k)
    keep_n = max(1, min(len(ordered_rows), keep_n))
    return ordered_rows[:keep_n]


def load_preference_prototype(
    compare_csv: Path,
    *,
    prototype_source: str,
    cohort: str,
    prototype_top_k: int | None,
    prototype_top_ratio: float,
) -> PrototypeStats:
    rows = list(csv.DictReader(compare_csv.open("r", encoding="utf-8")))
    if not rows:
        raise ValueError(f"空的对比文件: {compare_csv}")

    if prototype_source == "cohort":
        selected_rows = [row for row in rows if row.get("cohort") == cohort]
        if not selected_rows:
            raise ValueError(f"在 {compare_csv} 中没有找到 cohort={cohort} 的记录。")
    elif prototype_source == "act_keep":
        selected_rows = []
        for row in rows:
            flag = _parse_float(row.get("act_keep"))
            if flag is not None and flag > 0:
                selected_rows.append(row)
        if not selected_rows:
            selected_rows = _select_model_top_rows(
                rows,
                prefix="act",
                top_k=prototype_top_k,
                top_ratio=prototype_top_ratio,
            )
    elif prototype_source == "act_topk":
        selected_rows = _select_model_top_rows(
            rows,
            prefix="act",
            top_k=prototype_top_k,
            top_ratio=prototype_top_ratio,
        )
    elif prototype_source == "pi05_keep":
        selected_rows = []
        for row in rows:
            flag = _parse_float(row.get("pi05_keep"))
            if flag is not None and flag > 0:
                selected_rows.append(row)
        if not selected_rows:
            selected_rows = _select_model_top_rows(
                rows,
                prefix="pi05",
                top_k=prototype_top_k,
                top_ratio=prototype_top_ratio,
            )
    elif prototype_source == "pi05_topk":
        selected_rows = _select_model_top_rows(
            rows,
            prefix="pi05",
            top_k=prototype_top_k,
            top_ratio=prototype_top_ratio,
        )
    else:
        raise ValueError(f"不支持的 prototype_source={prototype_source}")

    mean: dict[str, float] = {}
    std: dict[str, float] = {}
    for key in PREFERENCE_FEATURES:
        values = [float(row[key]) for row in selected_rows]
        mean[key] = float(np.mean(np.asarray(values, dtype=np.float64)))
        std[key] = _safe_std(values)
    return PrototypeStats(mean=mean, std=std, count=len(selected_rows))


def read_episode_list(path: Path) -> list[int]:
    values: list[int] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        values.append(int(line))
    if not values:
        raise ValueError(f"空的 episode 列表: {path}")
    return values


def update_table(table: pa.Table, new_ep: int, start_idx: int) -> pa.Table:
    """重写 episode_index 和全局 index 列。"""
    length = table.num_rows
    metadata = table.schema.metadata

    def replace_column(name: str, values: list[int]) -> None:
        nonlocal table
        if name not in table.schema.names:
            return
        field = table.schema.field(name)
        arr = pa.array(values, type=field.type)
        table = table.set_column(table.schema.get_field_index(name), name, arr)

    replace_column("episode_index", [new_ep] * length)
    replace_column("index", list(range(start_idx, start_idx + length)))
    if metadata is not None:
        table = table.replace_schema_metadata(metadata)
    return table


def recompute_index_stats(length: int, offset: int) -> dict[str, list[float]]:
    start = offset
    end = offset + length - 1
    mean = (start + end) / 2
    var = (length * length - 1) / 12 if length > 1 else 0.0
    return {
        "min": [start],
        "max": [end],
        "mean": [mean],
        "std": [math.sqrt(var)],
        "count": [length],
    }


def update_episode_stats(stats: dict[str, Any], new_ep: int, start_idx: int, length: int) -> dict[str, Any]:
    if "episode_index" in stats:
        stats["episode_index"] = {
            "min": [new_ep],
            "max": [new_ep],
            "mean": [float(new_ep)],
            "std": [0.0],
            "count": [length],
        }
    if "index" in stats:
        stats["index"] = recompute_index_stats(length, start_idx)
    return stats


def subset_lerobot_dataset(
    src: Path,
    keep_episode_ids: list[int],
    out: Path,
    *,
    overwrite: bool,
) -> None:
    """按 episode 列表物化新的 LeRobot 子数据集。"""
    if out.exists():
        if not overwrite:
            raise FileExistsError(f"输出目录已存在: {out}")
        shutil.rmtree(out)

    episodes_path = src / "meta" / "episodes.jsonl"
    info_path = src / "meta" / "info.json"
    if not episodes_path.exists() or not info_path.exists():
        raise FileNotFoundError(f"缺少必要元数据: {src}")

    episodes = read_jsonl(episodes_path)
    episodes_by_ep = {int(row["episode_index"]): row for row in episodes}
    info = json.loads(info_path.read_text(encoding="utf-8"))
    chunks_size = int(info.get("chunks_size", 1000))
    data_pattern = info.get("data_path", "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet")
    video_pattern = info.get("video_path", "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4")
    video_keys = [k for k, v in info.get("features", {}).items() if v.get("dtype") == "video"]

    stats_rows_by_ep: dict[int, dict[str, Any]] = {}
    stats_path = src / "meta" / "episodes_stats.jsonl"
    if stats_path.exists():
        for row in read_jsonl(stats_path):
            ep = int(row.get("episode_index"))
            stats_rows_by_ep[ep] = row

    (out / "meta").mkdir(parents=True, exist_ok=True)
    (out / "data").mkdir(parents=True, exist_ok=True)
    (out / "videos").mkdir(parents=True, exist_ok=True)

    tasks_src = src / "meta" / "tasks.jsonl"
    if tasks_src.exists():
        shutil.copy2(tasks_src, out / "meta" / "tasks.jsonl")
    norm_src = src / "norm_stats.json"
    if norm_src.exists():
        shutil.copy2(norm_src, out / "norm_stats.json")

    kept_rows: list[dict[str, Any]] = []
    kept_stats_rows: list[dict[str, Any]] = []
    global_index = 0

    for new_ep, old_ep in enumerate(keep_episode_ids):
        if old_ep not in episodes_by_ep:
            raise KeyError(f"{src} 中不存在 episode {old_ep}")

        old_chunk = old_ep // chunks_size
        new_chunk = new_ep // chunks_size
        data_rel = data_pattern.format(episode_chunk=old_chunk, episode_index=old_ep)
        data_src = src / data_rel
        if not data_src.exists():
            raise FileNotFoundError(f"缺少 parquet 文件: {data_src}")

        table = pq.read_table(data_src)
        length = int(table.num_rows)
        table = update_table(table, new_ep=new_ep, start_idx=global_index)

        new_data_rel = data_pattern.format(episode_chunk=new_chunk, episode_index=new_ep)
        data_dst = out / new_data_rel
        data_dst.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, data_dst)

        row = dict(episodes_by_ep[old_ep])
        row["episode_index"] = new_ep
        row["length"] = length
        kept_rows.append(row)

        if old_ep in stats_rows_by_ep:
            stats_row = dict(stats_rows_by_ep[old_ep])
            stats_row["episode_index"] = new_ep
            stats_row["stats"] = update_episode_stats(
                dict(stats_row.get("stats", {})),
                new_ep,
                global_index,
                length,
            )
            kept_stats_rows.append(stats_row)

        for key in video_keys:
            video_rel = video_pattern.format(episode_chunk=old_chunk, episode_index=old_ep, video_key=key)
            video_src = src / video_rel
            if not video_src.exists():
                raise FileNotFoundError(f"缺少视频文件: {video_src}")
            new_video_rel = video_pattern.format(episode_chunk=new_chunk, episode_index=new_ep, video_key=key)
            video_dst = out / new_video_rel
            video_dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(video_src, video_dst)

        global_index += length

    write_jsonl(out / "meta" / "episodes.jsonl", kept_rows)
    if kept_stats_rows:
        write_jsonl(out / "meta" / "episodes_stats.jsonl", kept_stats_rows)

    info["total_episodes"] = len(kept_rows)
    info["total_frames"] = int(sum(int(row["length"]) for row in kept_rows))
    if "total_videos" in info:
        info["total_videos"] = len(kept_rows) * len(video_keys)
    info["total_chunks"] = ((len(kept_rows) - 1) // chunks_size) + 1 if kept_rows else 0
    info["splits"] = {"train": f"0:{len(kept_rows)}"}
    (out / "meta" / "info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def episode_parquet_files(dataset_root: Path) -> dict[int, Path]:
    mapping: dict[int, Path] = {}
    for path in sorted((dataset_root / "data").glob("chunk-*/episode_*.parquet")):
        stem = path.stem
        if not stem.startswith("episode_"):
            continue
        suffix = stem.split("_")[-1]
        if suffix.isdigit():
            mapping[int(suffix)] = path
    if not mapping:
        raise FileNotFoundError(f"在 {dataset_root} 下未找到 parquet episode 文件。")
    return mapping


def compute_episode_stats(dataset_root: Path) -> dict[int, dict[str, float]]:
    """按与 compare_act_openpi_selection.py 一致的口径计算轨迹统计。"""
    stats: dict[int, dict[str, float]] = {}
    for ep, path in episode_parquet_files(dataset_root).items():
        table = pq.read_table(path, columns=["action", "observation.state"])
        action = np.asarray(table["action"].to_pylist(), dtype=np.float64)
        state = np.asarray(table["observation.state"].to_pylist(), dtype=np.float64)

        record: dict[str, float] = {"episode": float(ep), "length": float(table.num_rows)}
        record["action_jitter"] = (
            float(np.linalg.norm(np.diff(action, axis=0), axis=1).mean()) if action.shape[0] >= 2 else 0.0
        )
        record["state_jitter"] = (
            float(np.linalg.norm(np.diff(state, axis=0), axis=1).mean()) if state.shape[0] >= 2 else 0.0
        )
        record["state_displacement"] = float(np.linalg.norm(state[-1] - state[0])) if state.shape[0] > 0 else 0.0
        stats[ep] = record
    return stats


def preference_score(record: dict[str, float], prototype: PrototypeStats) -> float:
    distance = 0.0
    for key in PREFERENCE_FEATURES:
        value = float(record[key])
        mu = prototype.mean[key]
        sigma = prototype.std[key]
        z = (value - mu) / sigma
        distance += z * z
    return -distance


def write_rank_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def merge_datasets(inputs: list[Path], output: Path, overwrite: bool) -> None:
    """合并多个 LeRobot 数据集，沿 episode 维顺序拼接。"""
    if output.exists():
        if not overwrite:
            raise FileExistsError(f"输出目录已存在: {output}")
        shutil.rmtree(output)

    (output / "data").mkdir(parents=True, exist_ok=True)
    (output / "meta").mkdir(parents=True, exist_ok=True)

    all_episode_rows: list[dict[str, Any]] = []
    all_stats_rows: list[dict[str, Any]] = []
    total_frames = 0
    base_info: dict[str, Any] | None = None
    base_norm_src: Path | None = None
    base_tasks_src: Path | None = None
    chunks_size = 1000
    video_keys: list[str] = []

    for ds_idx, ds_root in enumerate(inputs):
        info = json.loads((ds_root / "meta" / "info.json").read_text(encoding="utf-8"))
        if base_info is None:
            base_info = info
            chunks_size = int(info.get("chunks_size", 1000))
            video_keys = [k for k, v in info.get("features", {}).items() if v.get("dtype") == "video"]
            norm_src = ds_root / "norm_stats.json"
            if norm_src.exists():
                base_norm_src = norm_src
            tasks_src = ds_root / "meta" / "tasks.jsonl"
            if tasks_src.exists():
                base_tasks_src = tasks_src

        episodes = read_jsonl(ds_root / "meta" / "episodes.jsonl")
        stats_rows = {
            int(row["episode_index"]): row for row in read_jsonl(ds_root / "meta" / "episodes_stats.jsonl")
        } if (ds_root / "meta" / "episodes_stats.jsonl").exists() else {}

        for row in episodes:
            old_ep = int(row["episode_index"])
            new_ep = len(all_episode_rows)
            old_chunk = old_ep // chunks_size
            new_chunk = new_ep // chunks_size
            data_src = ds_root / f"data/chunk-{old_chunk:03d}/episode_{old_ep:06d}.parquet"
            if not data_src.exists():
                raise FileNotFoundError(f"缺少 parquet 文件: {data_src}")

            table = pq.read_table(data_src)
            length = int(table.num_rows)
            table = update_table(table, new_ep=new_ep, start_idx=total_frames)

            data_dst = output / f"data/chunk-{new_chunk:03d}/episode_{new_ep:06d}.parquet"
            data_dst.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(table, data_dst)

            for key in video_keys:
                video_src = ds_root / f"videos/chunk-{old_chunk:03d}/{key}/episode_{old_ep:06d}.mp4"
                if not video_src.exists():
                    raise FileNotFoundError(f"缺少视频文件: {video_src}")
                video_dst = output / f"videos/chunk-{new_chunk:03d}/{key}/episode_{new_ep:06d}.mp4"
                video_dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(video_src, video_dst)

            new_row = dict(row)
            new_row["episode_index"] = new_ep
            new_row["length"] = length
            all_episode_rows.append(new_row)

            if old_ep in stats_rows:
                stats_row = dict(stats_rows[old_ep])
                stats_row["episode_index"] = new_ep
                stats_row["stats"] = update_episode_stats(
                    dict(stats_row.get("stats", {})),
                    new_ep,
                    total_frames,
                    length,
                )
                all_stats_rows.append(stats_row)

            total_frames += length

    if base_info is None:
        raise ValueError("没有可合并的数据集输入。")

    if base_norm_src is not None:
        shutil.copy2(base_norm_src, output / "norm_stats.json")
    if base_tasks_src is not None:
        shutil.copy2(base_tasks_src, output / "meta" / "tasks.jsonl")

    write_jsonl(output / "meta" / "episodes.jsonl", all_episode_rows)
    if all_stats_rows:
        write_jsonl(output / "meta" / "episodes_stats.jsonl", all_stats_rows)

    base_info["total_episodes"] = len(all_episode_rows)
    base_info["total_frames"] = total_frames
    if "total_videos" in base_info:
        base_info["total_videos"] = len(all_episode_rows) * len(video_keys)
    base_info["total_chunks"] = ((len(all_episode_rows) - 1) // chunks_size) + 1 if all_episode_rows else 0
    base_info["splits"] = {"train": f"0:{len(all_episode_rows)}"}
    (output / "meta" / "info.json").write_text(
        json.dumps(base_info, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    output_root = args.output_root.resolve()
    manifests_dir = output_root / "manifests"
    datasets_dir = output_root / "datasets"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    datasets_dir.mkdir(parents=True, exist_ok=True)

    keep_can_mh = read_episode_list(args.can_mh_keep_list)
    base_count = len(keep_can_mh)
    supplement_count = args.supplement_count
    if supplement_count is None:
        supplement_count = int(round(base_count * float(args.supplement_ratio)))
    supplement_count = max(1, supplement_count)

    mh_name = args.can_mh_dataset.name
    ph_name = args.can_ph_dataset.name

    prototype = load_preference_prototype(
        args.compare_csv,
        prototype_source=args.prototype_source,
        cohort=args.cohort,
        prototype_top_k=args.prototype_top_k,
        prototype_top_ratio=args.prototype_top_ratio,
    )
    can_ph_stats = compute_episode_stats(args.can_ph_dataset)
    if supplement_count > len(can_ph_stats):
        raise ValueError(f"请求的 supplement_count={supplement_count} 超过 can_ph episode 总数 {len(can_ph_stats)}")

    ranking_rows: list[dict[str, Any]] = []
    for ep, record in sorted(can_ph_stats.items()):
        pref = preference_score(record, prototype)
        row = {
            "episode": ep,
            "preference_score": pref,
            **{key: record[key] for key in PREFERENCE_FEATURES},
        }
        ranking_rows.append(row)

    ranking_rows.sort(key=lambda x: float(x["preference_score"]), reverse=True)
    preference_keep = [int(row["episode"]) for row in ranking_rows[:supplement_count]]

    rng = random.Random(args.random_seed)
    can_ph_all = sorted(can_ph_stats.keys())
    random_keep = sorted(rng.sample(can_ph_all, k=supplement_count))

    preference_set = set(preference_keep)
    random_set = set(random_keep)
    for rank, row in enumerate(ranking_rows, start=1):
        row["preference_rank"] = rank
        row["selected_preference"] = int(int(row["episode"]) in preference_set)
        row["selected_random"] = int(int(row["episode"]) in random_set)

    rank_csv = manifests_dir / f"{ph_name}_preference_ranking.csv"
    write_rank_csv(
        rank_csv,
        ranking_rows,
        fieldnames=[
            "episode",
            "preference_rank",
            "preference_score",
            "length",
            "action_jitter",
            "state_jitter",
            "state_displacement",
            "selected_preference",
            "selected_random",
        ],
    )
    (manifests_dir / f"{ph_name}_preference_keep.txt").write_text(
        "\n".join(str(ep) for ep in preference_keep) + "\n",
        encoding="utf-8",
    )
    (manifests_dir / f"{ph_name}_random_keep.txt").write_text(
        "\n".join(str(ep) for ep in random_keep) + "\n",
        encoding="utf-8",
    )

    can_mh_subset = datasets_dir / f"{mh_name}_twostage_keep"
    can_ph_pref_subset = datasets_dir / f"{ph_name}_act_pref_k{supplement_count:03d}"
    can_ph_rand_subset = datasets_dir / f"{ph_name}_random_k{supplement_count:03d}_seed{args.random_seed}"
    merged_pref = datasets_dir / f"{mh_name}_twostage_plus_{ph_name}_pref_k{supplement_count:03d}"
    merged_rand = datasets_dir / f"{mh_name}_twostage_plus_{ph_name}_random_k{supplement_count:03d}_seed{args.random_seed}"

    subset_lerobot_dataset(args.can_mh_dataset, keep_can_mh, can_mh_subset, overwrite=args.overwrite)
    subset_lerobot_dataset(args.can_ph_dataset, preference_keep, can_ph_pref_subset, overwrite=args.overwrite)
    subset_lerobot_dataset(args.can_ph_dataset, random_keep, can_ph_rand_subset, overwrite=args.overwrite)

    merge_datasets([can_mh_subset, can_ph_pref_subset], merged_pref, overwrite=args.overwrite)
    merge_datasets([can_mh_subset, can_ph_rand_subset], merged_rand, overwrite=args.overwrite)

    summary = {
        "can_mh_dataset": str(args.can_mh_dataset),
        "can_mh_keep_list": str(args.can_mh_keep_list),
        "can_mh_kept_episodes": base_count,
        "can_ph_dataset": str(args.can_ph_dataset),
        "can_ph_total_episodes": len(can_ph_stats),
        "preference_compare_csv": str(args.compare_csv),
        "preference_prototype_source": args.prototype_source,
        "preference_cohort": args.cohort,
        "preference_prototype_top_k": args.prototype_top_k,
        "preference_prototype_top_ratio": args.prototype_top_ratio,
        "preference_prototype_count": prototype.count,
        "preference_prototype_mean": prototype.mean,
        "preference_prototype_std": prototype.std,
        "supplement_count": supplement_count,
        "random_seed": args.random_seed,
        "outputs": {
            "can_mh_twostage_dataset": str(can_mh_subset),
            "can_ph_preference_dataset": str(can_ph_pref_subset),
            "can_ph_random_dataset": str(can_ph_rand_subset),
            "merged_preference_dataset": str(merged_pref),
            "merged_random_dataset": str(merged_rand),
            "preference_rank_csv": str(rank_csv),
        },
    }
    (manifests_dir / "build_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
