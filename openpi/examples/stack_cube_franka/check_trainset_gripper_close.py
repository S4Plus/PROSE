#!/usr/bin/env python3
"""将训练集关键样本送入 OpenPI checkpoint，检查是否会输出闭合夹爪。

默认行为：
1. 扫描训练集 parquet，估计夹爪开合双峰宽度，并用中点作为 close/open 阈值。
2. 找出真正的 `state=open -> gt=close` transition，这类样本最能回答“模型会不会发出闭合指令”。
3. 用 OpenPI `policy.infer(...)` 对这些样本做开环推理，读取 `actions[0, 7]`，
   判断模型首步是否进入 close 模态。

说明：
- 当前 stack_cube_franka 配置使用 delta joint actions，但最后 1 维 gripper width 保持绝对值。
- `policy.infer(...)` 经过输出变换后返回的是绝对 action，因此这里可直接检查 `actions[:, 7]`。
"""

from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import Iterable

import numpy as np
import pyarrow.parquet as pq


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="检查训练集输入到 OpenPI 后是否会输出闭合夹爪指令")
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path(
            "/workspace/liushuo/cupid/openpi/checkpoints/pi05_stack_cube_franka_lora/stack_cube_franka_joint_delta_30k/29999"
        ),
        help="OpenPI checkpoint 目录（包含 params/train_state/assets）",
    )
    parser.add_argument(
        "--config-name",
        type=str,
        default="pi05_stack_cube_franka_lora",
        help="OpenPI 训练配置名",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/workspace/liushuo/dataset/stack_cube_franka_joint_lerobot"),
        help="LeRobot 数据集根目录",
    )
    parser.add_argument(
        "--default-prompt",
        type=str,
        default="stack cube",
        help="若样本里没有 prompt，则注入该默认 prompt",
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
        "--jax-platform",
        type=str,
        default=None,
        help="可选：显式指定 JAX 平台，例如 cpu / gpu。",
    )
    parser.add_argument(
        "--save-json",
        type=Path,
        default=None,
        help="可选，将详细结果保存为 JSON 文件",
    )
    return parser.parse_args()


def ensure_runtime_env(jax_platform: str | None) -> None:
    """配置运行时环境，避免 cache 和 JAX 预分配导致的问题。"""
    os.environ.setdefault("HF_HOME", "/tmp/hf_home")
    os.environ.setdefault("HF_DATASETS_CACHE", "/tmp/hf_datasets_cache")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    if jax_platform:
        os.environ["JAX_PLATFORMS"] = jax_platform


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
    files = sorted((dataset_root / "data").glob("chunk-*/episode_*.parquet"))
    if not files:
        raise FileNotFoundError(f"未找到 parquet episode 文件: {dataset_root / 'data'}")
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

        if states.ndim != 2 or states.shape[1] != 8:
            raise ValueError(f"{path} state 维度异常: {states.shape}")
        if actions.ndim != 2 or actions.shape[1] != 8:
            raise ValueError(f"{path} action 维度异常: {actions.shape}")
        if frame_indices.ndim != 1 or frame_indices.shape[0] != states.shape[0]:
            raise ValueError(f"{path} frame_index 维度异常: {frame_indices.shape}")

        raw_records.append((episode_index, frame_indices, states, actions))
        all_action_gripper.append(actions[:, 7])

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
        state_close = states[:, 7] <= threshold
        action_close = actions[:, 7] <= threshold
        for local_idx in range(states.shape[0]):
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
        global_offset += states.shape[0]

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


def to_numpy_tree(data: Any) -> Any:
    """把 LeRobot 样本里的 torch.Tensor 递归转成 numpy，便于送入 JAX policy。"""
    try:
        import torch
    except ImportError:
        torch = None

    if torch is not None and isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    if isinstance(data, dict):
        return {k: to_numpy_tree(v) for k, v in data.items()}
    if isinstance(data, list):
        return [to_numpy_tree(v) for v in data]
    if isinstance(data, tuple):
        return tuple(to_numpy_tree(v) for v in data)
    return data


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
    ensure_runtime_env(args.jax_platform)

    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
    from openpi.policies import policy_config as _policy_config
    from openpi.training import config as _config

    checkpoint_dir = args.checkpoint_dir.expanduser().resolve()
    dataset_root = args.dataset_root.expanduser().resolve()
    train_config = _config.get_config(args.config_name)

    low, high, threshold, splits = build_transition_indices(dataset_root)
    selected = build_eval_list(
        splits=splits,
        max_open_to_close=args.max_open_to_close,
        max_open_to_open=args.max_open_to_open,
        max_close_to_close=args.max_close_to_close,
        seed=args.seed,
    )

    print("=== 数据集夹爪统计 ===")
    print(f"checkpoint_dir: {checkpoint_dir}")
    print(f"config_name:     {args.config_name}")
    print(f"dataset_root:    {dataset_root}")
    print(f"close_width:     {low:.6f}")
    print(f"open_width:      {high:.6f}")
    print(f"threshold:       {threshold:.6f}")
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

    print("加载 OpenPI policy...")
    policy = _policy_config.create_trained_policy(
        train_config,
        checkpoint_dir,
        default_prompt=args.default_prompt,
    )
    print("OpenPI policy 加载完成。")

    dataset_meta = LeRobotDatasetMetadata(str(dataset_root), root=dataset_root)
    delta_timestamps = {"action": [t / dataset_meta.fps for t in range(train_config.model.action_horizon)]}
    dataset = LeRobotDataset(str(dataset_root), root=dataset_root, delta_timestamps=delta_timestamps)

    results: list[PredictionRecord] = []
    for idx, meta in enumerate(selected, start=1):
        item = dataset[meta.global_index]
        obs = to_numpy_tree(item)
        out = policy.infer(obs)

        pred_chunk_np = np.asarray(out["actions"])
        gt_chunk_np = np.asarray(obs["action"])
        pred_gripper = float(pred_chunk_np[0, 7])

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

        if idx <= 5 or idx == len(selected) or idx % 10 == 0:
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
            "checkpoint_dir": str(checkpoint_dir),
            "config_name": args.config_name,
            "dataset_root": str(dataset_root),
            "default_prompt": args.default_prompt,
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
