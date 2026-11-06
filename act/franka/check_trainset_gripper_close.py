#!/usr/bin/env python3
"""将训练集样本送入 ACT 模型，检查首步输出是否会闭合夹爪。

默认行为：
1. 扫描训练集 parquet，估计夹爪开合双峰宽度，并用中点作为 close/open 阈值。
2. 找出真正的 `state=open -> action=close` transition，这类样本最能回答“模型会不会发出闭合指令”。
3. 将这些样本送入 checkpoint，读取 `predict_action_chunk(...)[0, 0, 7]`，
   判断模型首步是否进入 close 模态。

说明：
- 当前 Franka joint ACT 数据语义为：
  `state = qpos(7) + gripper_width(1)`
  `action = next_qpos(7) + next_gripper_width(1)`
- 因此这里检查的是“给定当前训练集观测，模型预测的下一步 gripper_width 是否会落到 close 区间”。
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pyarrow.parquet as pq
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
LEROBOT_SRC = SCRIPT_DIR.parents[1] / "lerobot_2.1" / "src"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="检查训练集输入到 ACT 后是否会输出闭合夹爪指令")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "/workspace/liushuo/cupid/outputs/stack_cube_franka_joint_act/checkpoints/100000/pretrained_model"
        ),
        help="ACT checkpoint 的 pretrained_model 目录",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/workspace/liushuo/dataset/stack_cube_franka_joint_lerobot"),
        help="LeRobot 数据集根目录",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default="local/stack_cube_franka_joint",
        help="LeRobot repo_id",
    )
    parser.add_argument(
        "--video-backend",
        type=str,
        default="pyav",
        help="视频解码后端。CPU 环境下用 pyav 更稳妥。",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="推理设备，默认自动选择；若请求 cuda 但不可用，会自动回退到 cpu。",
    )
    parser.add_argument(
        "--max-open-to-close",
        type=int,
        default=None,
        help="最多检查多少条 open->close transition，默认检查全部。",
    )
    parser.add_argument(
        "--max-open-to-open",
        type=int,
        default=64,
        help="额外抽样多少条 open->open 作为对照组，设为 0 则不检查。",
    )
    parser.add_argument(
        "--max-close-to-close",
        type=int,
        default=64,
        help="额外抽样多少条 close->close 作为对照组，设为 0 则不检查。",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子",
    )
    parser.add_argument(
        "--save-json",
        type=Path,
        default=None,
        help="可选，将详细结果保存为 JSON 文件",
    )
    return parser.parse_args()


def ensure_runtime_env() -> None:
    """将 HuggingFace datasets cache 指到可写目录，避免只读文件系统报错。"""
    os.environ.setdefault("HF_HOME", "/tmp/hf_home")
    os.environ.setdefault("HF_DATASETS_CACHE", "/tmp/hf_datasets_cache")
    if str(LEROBOT_SRC) not in sys.path:
        sys.path.insert(0, str(LEROBOT_SRC))


def resolve_checkpoint(path: Path) -> Path:
    path = path.expanduser().resolve()
    if path.is_dir() and (path / "config.json").is_file() and (path / "model.safetensors").is_file():
        return path
    if path.is_dir() and (path / "pretrained_model").is_dir():
        inner = path / "pretrained_model"
        if (inner / "config.json").is_file() and (inner / "model.safetensors").is_file():
            return inner
    raise FileNotFoundError(f"无法解析 pretrained_model 目录: {path}")


@dataclass
class TransitionIndex:
    global_index: int
    episode_index: int
    frame_index: int
    state_gripper: float
    gt_gripper: float
    split: str


@dataclass
class PredictionRecord:
    global_index: int
    episode_index: int
    frame_index: int
    split: str
    state_gripper: float
    gt_gripper: float
    pred_gripper: float
    gt_is_close: bool
    pred_is_close: bool
    pred_first_close_step: int | None
    gt_first_close_step: int | None


def iter_episode_parquets(dataset_root: Path) -> list[Path]:
    data_dir = dataset_root / "data" / "chunk-000"
    files = sorted(data_dir.glob("episode_*.parquet"))
    if not files:
        raise FileNotFoundError(f"未找到 parquet episode 文件: {data_dir}")
    return files


def build_transition_indices(dataset_root: Path) -> tuple[float, float, float, dict[str, list[TransitionIndex]]]:
    """扫描 parquet，估计 close/open 阈值并构建 transition 索引。"""
    files = iter_episode_parquets(dataset_root)

    all_action_gripper = []
    raw_records: list[tuple[int, np.ndarray, np.ndarray, np.ndarray]] = []
    for path in files:
        episode_index = int(path.stem.split("_")[1])
        table = pq.read_table(path, columns=["observation.state", "action", "frame_index"])
        states = np.stack(table["observation.state"].to_pylist()).astype(np.float32)
        actions = np.stack(table["action"].to_pylist()).astype(np.float32)
        frame_indices = np.asarray(table["frame_index"].to_pylist(), dtype=np.int64)
        raw_records.append((episode_index, frame_indices, states, actions))
        all_action_gripper.append(actions[:, 7])

        if states.ndim != 2 or states.shape[1] != 8:
            raise ValueError(f"{path} state 维度异常: {states.shape}")
        if actions.ndim != 2 or actions.shape[1] != 8:
            raise ValueError(f"{path} action 维度异常: {actions.shape}")
        if frame_indices.ndim != 1 or frame_indices.shape[0] != states.shape[0]:
            raise ValueError(f"{path} frame_index 维度异常: {frame_indices.shape}")

    action_gripper = np.concatenate(all_action_gripper)
    low = float(action_gripper.min())
    high = float(action_gripper.max())
    threshold = (low + high) / 2.0

    splits = {
        "open_to_close": [],
        "open_to_open": [],
        "close_to_close": [],
        "close_to_open": [],
    }
    global_offset = 0
    for episode_index, frame_indices, states, actions in raw_records:
        length = states.shape[0]
        state_close = states[:, 7] <= threshold
        action_close = actions[:, 7] <= threshold

        for local_idx in range(length):
            if not state_close[local_idx] and action_close[local_idx]:
                split = "open_to_close"
            elif not state_close[local_idx] and not action_close[local_idx]:
                split = "open_to_open"
            elif state_close[local_idx] and action_close[local_idx]:
                split = "close_to_close"
            else:
                split = "close_to_open"

            splits[split].append(
                TransitionIndex(
                    global_index=global_offset + local_idx,
                    episode_index=episode_index,
                    frame_index=int(frame_indices[local_idx]),
                    state_gripper=float(states[local_idx, 7]),
                    gt_gripper=float(actions[local_idx, 7]),
                    split=split,
                )
            )
        global_offset += length

    return low, high, threshold, splits


def sample_indices(items: list[TransitionIndex], limit: int | None, rng: random.Random) -> list[TransitionIndex]:
    if limit is None or limit >= len(items):
        return list(items)
    return sorted(rng.sample(items, limit), key=lambda x: x.global_index)


def build_eval_list(
    splits: dict[str, list[TransitionIndex]],
    max_open_to_close: int | None,
    max_open_to_open: int,
    max_close_to_close: int,
    seed: int,
) -> list[TransitionIndex]:
    rng = random.Random(seed)
    selected: list[TransitionIndex] = []
    selected.extend(sample_indices(splits["open_to_close"], max_open_to_close, rng))
    if max_open_to_open > 0:
        selected.extend(sample_indices(splits["open_to_open"], max_open_to_open, rng))
    if max_close_to_close > 0:
        selected.extend(sample_indices(splits["close_to_close"], max_close_to_close, rng))
    selected.sort(key=lambda x: x.global_index)
    return selected


def get_device(requested: str | None) -> torch.device:
    if requested is None:
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        print("[警告] 请求使用 cuda，但当前不可用，自动回退到 cpu。")
        requested = "cpu"
    return torch.device(requested)


def summarize(records: Iterable[PredictionRecord], threshold: float) -> dict[str, object]:
    rows = list(records)
    if not rows:
        return {"count": 0}

    gt = np.asarray([r.gt_gripper for r in rows], dtype=np.float32)
    pred = np.asarray([r.pred_gripper for r in rows], dtype=np.float32)
    gt_close = gt <= threshold
    pred_close = pred <= threshold
    pred_first_close_steps = [r.pred_first_close_step for r in rows]
    gt_first_close_steps = [r.gt_first_close_step for r in rows]

    tp = int(np.logical_and(gt_close, pred_close).sum())
    tn = int(np.logical_and(~gt_close, ~pred_close).sum())
    fp = int(np.logical_and(~gt_close, pred_close).sum())
    fn = int(np.logical_and(gt_close, ~pred_close).sum())

    result = {
        "count": int(len(rows)),
        "gt_close_ratio": float(gt_close.mean()),
        "pred_close_ratio": float(pred_close.mean()),
        "gt_any_close_ratio": float(np.mean([step is not None for step in gt_first_close_steps])),
        "pred_any_close_ratio": float(np.mean([step is not None for step in pred_first_close_steps])),
        "mean_gt_gripper": float(gt.mean()),
        "mean_pred_gripper": float(pred.mean()),
        "mean_abs_error": float(np.abs(pred - gt).mean()),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": float((tp + tn) / len(rows)),
    }
    if tp + fp > 0:
        result["precision_close"] = float(tp / (tp + fp))
    if tp + fn > 0:
        result["recall_close"] = float(tp / (tp + fn))
    valid_pred_steps = [step for step in pred_first_close_steps if step is not None]
    valid_gt_steps = [step for step in gt_first_close_steps if step is not None]
    if valid_gt_steps:
        result["mean_gt_first_close_step"] = float(np.mean(valid_gt_steps))
    if valid_pred_steps:
        result["mean_pred_first_close_step"] = float(np.mean(valid_pred_steps))
        result["max_pred_first_close_step"] = int(np.max(valid_pred_steps))
        step_hist: dict[str, int] = {}
        for step in sorted(set(valid_pred_steps)):
            step_hist[str(step)] = int(sum(1 for x in valid_pred_steps if x == step))
        result["pred_first_close_hist"] = step_hist
    return result


def main() -> None:
    args = parse_args()
    ensure_runtime_env()

    from lerobot.datasets.factory import resolve_delta_timestamps
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
    from lerobot.policies.act.modeling_act import ACTPolicy

    checkpoint = resolve_checkpoint(args.checkpoint)
    dataset_root = args.dataset_root.expanduser().resolve()
    device = get_device(args.device)

    low, high, threshold, splits = build_transition_indices(dataset_root)
    selected = build_eval_list(
        splits=splits,
        max_open_to_close=args.max_open_to_close,
        max_open_to_open=args.max_open_to_open,
        max_close_to_close=args.max_close_to_close,
        seed=args.seed,
    )

    print("=== 数据集夹爪统计 ===")
    print(f"checkpoint: {checkpoint}")
    print(f"dataset_root: {dataset_root}")
    print(f"repo_id: {args.repo_id}")
    print(f"device: {device}")
    print(f"video_backend: {args.video_backend}")
    print(f"close_width: {low:.6f}")
    print(f"open_width:  {high:.6f}")
    print(f"threshold:   {threshold:.6f}")
    print()
    print("=== transition 计数 ===")
    for key in ["open_to_close", "open_to_open", "close_to_close", "close_to_open"]:
        print(f"{key}: {len(splits[key])}")
    print()
    print("=== 本次评估样本数 ===")
    for key in ["open_to_close", "open_to_open", "close_to_close"]:
        current = sum(1 for item in selected if item.split == key)
        print(f"{key}: {current}")
    print()

    policy = ACTPolicy.from_pretrained(str(checkpoint))
    policy.to(device)
    policy.eval()
    policy.reset()
    ds_meta = LeRobotDatasetMetadata(args.repo_id, root=dataset_root)
    delta_timestamps = resolve_delta_timestamps(policy.config, ds_meta)
    dataset = LeRobotDataset(
        repo_id=args.repo_id,
        root=dataset_root,
        delta_timestamps=delta_timestamps,
        video_backend=args.video_backend,
    )

    results: list[PredictionRecord] = []
    for idx, meta in enumerate(selected, start=1):
        item = dataset[meta.global_index]
        batch = {
            "observation.images.fixed_camera": item["observation.images.fixed_camera"].unsqueeze(0).float().to(device),
            "observation.images.wrist_camera": item["observation.images.wrist_camera"].unsqueeze(0).float().to(device),
            "observation.state": item["observation.state"].unsqueeze(0).float().to(device),
        }
        with torch.no_grad():
            pred_chunk = policy.predict_action_chunk(batch)
        pred_chunk_np = pred_chunk[0].detach().cpu().numpy()
        pred_gripper = float(pred_chunk_np[0, 7])
        gt_chunk_np = item["action"].detach().cpu().numpy()
        pred_close_steps = np.where(pred_chunk_np[:, 7] <= threshold)[0]
        gt_close_steps = np.where(gt_chunk_np[:, 7] <= threshold)[0]
        record = PredictionRecord(
            global_index=meta.global_index,
            episode_index=meta.episode_index,
            frame_index=meta.frame_index,
            split=meta.split,
            state_gripper=meta.state_gripper,
            gt_gripper=meta.gt_gripper,
            pred_gripper=pred_gripper,
            gt_is_close=meta.gt_gripper <= threshold,
            pred_is_close=pred_gripper <= threshold,
            pred_first_close_step=None if len(pred_close_steps) == 0 else int(pred_close_steps[0]),
            gt_first_close_step=None if len(gt_close_steps) == 0 else int(gt_close_steps[0]),
        )
        results.append(record)

        if idx <= 5 or idx == len(selected):
            print(
                f"[{idx:03d}/{len(selected):03d}] "
                f"{meta.split} ep={meta.episode_index:03d} frame={meta.frame_index:03d} "
                f"state={meta.state_gripper:.6f} gt={meta.gt_gripper:.6f} pred={pred_gripper:.6f} "
                f"gt_first_close_step={record.gt_first_close_step} "
                f"pred_first_close_step={record.pred_first_close_step}"
            )

    print()
    print("=== 汇总 ===")
    all_summary = summarize(results, threshold)
    print(json.dumps(all_summary, ensure_ascii=False, indent=2))
    print()

    for split in ["open_to_close", "open_to_open", "close_to_close"]:
        split_rows = [r for r in results if r.split == split]
        print(f"=== {split} ===")
        print(json.dumps(summarize(split_rows, threshold), ensure_ascii=False, indent=2))
        if split == "open_to_close":
            failures = [r for r in split_rows if not r.pred_is_close]
            if failures:
                print("首步未预测出闭合的 open_to_close 样本示例：")
                for row in failures[:10]:
                    print(
                        f"  ep={row.episode_index:03d} frame={row.frame_index:03d} "
                        f"state={row.state_gripper:.6f} gt={row.gt_gripper:.6f} pred={row.pred_gripper:.6f} "
                        f"pred_first_close_step={row.pred_first_close_step}"
                    )
            else:
                print("所有 open_to_close 样本首步都预测到了闭合模态。")
        print()

    if args.save_json is not None:
        payload = {
            "checkpoint": str(checkpoint),
            "dataset_root": str(dataset_root),
            "repo_id": args.repo_id,
            "device": str(device),
            "video_backend": args.video_backend,
            "close_width": low,
            "open_width": high,
            "threshold": threshold,
            "split_counts": {key: len(val) for key, val in splits.items()},
            "selected_counts": {
                key: sum(1 for item in selected if item.split == key)
                for key in ["open_to_close", "open_to_open", "close_to_close"]
            },
            "summary": all_summary,
            "records": [asdict(row) for row in results],
        }
        save_path = args.save_json.expanduser().resolve()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"详细结果已保存到: {save_path}")


if __name__ == "__main__":
    main()
