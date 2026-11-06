#!/usr/bin/env python3
"""
审计 OpenVLA/LeRobot 数据集中 joint 与 gripper 维度是否按预期对齐。

重点回答两个问题：
1. 训练集 `observation.state` 和 `action` 的每一维分别代表什么。
2. `state[t][:joint_dim]` 与 `action[t][:joint_dim]` 是否只是时间步不同，而不是语义错位。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _resolve_checkpoint_dir(path_str: str | Path) -> Path:
    root = Path(path_str).expanduser().resolve()
    if root.is_file():
        root = root.parent
    if (root / "lerobot_finetune_config.json").is_file():
        return root

    candidates: list[tuple[int, Path]] = []
    if root.parent.exists():
        pattern = f"{root.name}--*_chkpt"
        for child in root.parent.glob(pattern):
            if not (child / "lerobot_finetune_config.json").is_file():
                continue
            try:
                step = int(child.name[len(root.name) + 2 : -len("_chkpt")])
            except Exception:
                step = -1
            candidates.append((step, child))
    if candidates:
        candidates.sort(key=lambda item: (item[0], item[1].name))
        return candidates[-1][1]
    raise FileNotFoundError(f"无法从 {path_str} 解析 checkpoint 目录")


def _load_sample_arrays(dataset_root: Path, max_episodes: int) -> tuple[np.ndarray, np.ndarray]:
    episodes_path = dataset_root / "meta" / "episodes.jsonl"
    episode_rows = []
    with episodes_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                episode_rows.append(json.loads(line))

    states = []
    actions = []
    for row in episode_rows[: max(max_episodes, 1)]:
        episode_index = int(row["episode_index"])
        parquet_path = dataset_root / "data" / f"chunk-{episode_index // 1000:03d}" / f"episode_{episode_index:06d}.parquet"
        table = pq.read_table(parquet_path, columns=["observation.state", "action"])
        data = table.to_pydict()
        states.append(np.asarray(data["observation.state"], dtype=np.float32))
        actions.append(np.asarray(data["action"], dtype=np.float32))

    if not states or not actions:
        raise ValueError(f"在 {dataset_root} 下没有读取到 state/action 数据")

    return np.concatenate(states, axis=0), np.concatenate(actions, axis=0)


def _fmt_list(values: list[Any]) -> str:
    return "[" + ", ".join(str(v) for v in values) + "]"


def main() -> None:
    parser = argparse.ArgumentParser(description="审计 OpenVLA joint/state/action 维度对齐关系。")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--max-episodes", type=int, default=20)
    args = parser.parse_args()

    dataset_root = args.dataset_root.expanduser().resolve()
    info = _read_json(dataset_root / "meta" / "info.json")
    norm_stats = _read_json(dataset_root / "norm_stats.json")["norm_stats"]

    state_feature = info["features"]["observation.state"]
    action_feature = info["features"]["action"]
    state_names = [str(x) for x in state_feature.get("names", [])]
    action_names = [str(x) for x in action_feature.get("names", [])]
    state_dim = int(state_feature["shape"][0])
    action_dim = int(action_feature["shape"][0])

    print("== 数据集维度定义 ==")
    print(f"dataset_root: {dataset_root}")
    print(f"state_dim: {state_dim}")
    print(f"action_dim: {action_dim}")
    print(f"state_names: {_fmt_list(state_names)}")
    print(f"action_names: {_fmt_list(action_names)}")
    print()

    joint_overlap = min(
        len([n for n in state_names if "joint" in n]),
        len([n for n in action_names if "joint" in n]),
        state_dim,
        action_dim,
    )
    shared_name_match = [
        (idx, state_names[idx], action_names[idx], state_names[idx] == action_names[idx])
        for idx in range(min(len(state_names), len(action_names)))
    ]

    print("== 逐维语义对齐 ==")
    for idx, state_name, action_name, matched in shared_name_match:
        tag = "OK" if matched else "MISMATCH"
        print(f"dim {idx}: state={state_name} | action={action_name} | {tag}")
    if state_dim > action_dim:
        for idx in range(action_dim, state_dim):
            print(f"dim {idx}: state-only={state_names[idx] if idx < len(state_names) else f'state_{idx}'}")
    print()

    state_q01 = np.asarray(norm_stats["state"]["q01"], dtype=np.float32)
    state_q99 = np.asarray(norm_stats["state"]["q99"], dtype=np.float32)
    action_q01 = np.asarray(norm_stats["actions"]["q01"], dtype=np.float32)
    action_q99 = np.asarray(norm_stats["actions"]["q99"], dtype=np.float32)

    print("== 归一化统计范围 ==")
    for idx in range(min(joint_overlap, action_dim)):
        print(
            "joint dim {}: state[q01,q99]=[{:.6f}, {:.6f}] | action[q01,q99]=[{:.6f}, {:.6f}]".format(
                idx,
                float(state_q01[idx]),
                float(state_q99[idx]),
                float(action_q01[idx]),
                float(action_q99[idx]),
            )
        )
    if state_dim > joint_overlap:
        for idx in range(joint_overlap, state_dim):
            print(
                "state extra dim {}: {} range=[{:.6f}, {:.6f}]".format(
                    idx,
                    state_names[idx] if idx < len(state_names) else f"state_{idx}",
                    float(state_q01[idx]),
                    float(state_q99[idx]),
                )
            )
    if action_dim > joint_overlap:
        for idx in range(joint_overlap, action_dim):
            print(
                "action extra dim {}: {} range=[{:.6f}, {:.6f}]".format(
                    idx,
                    action_names[idx] if idx < len(action_names) else f"action_{idx}",
                    float(action_q01[idx]),
                    float(action_q99[idx]),
                )
            )
    print()

    states, actions = _load_sample_arrays(dataset_root, max_episodes=args.max_episodes)
    compare_joint_dim = min(joint_overlap, states.shape[1], actions.shape[1])
    diffs = np.abs(states[:, :compare_joint_dim] - actions[:, :compare_joint_dim])
    state_abs = np.abs(states[:, :compare_joint_dim])
    action_abs = np.abs(actions[:, :compare_joint_dim])

    print("== 时间步差异审计 ==")
    print("说明: 这里比较的是 state[t] 与 action[t]。若 action 定义为下一步 joint target，则数值应接近但不必完全相等。")
    for idx in range(compare_joint_dim):
        print(
            "joint dim {}: state_mean_abs={:.6f} | action_mean_abs={:.6f} | diff_mean_abs={:.6f} | diff_p99={:.6f}".format(
                idx,
                float(state_abs[:, idx].mean()),
                float(action_abs[:, idx].mean()),
                float(diffs[:, idx].mean()),
                float(np.quantile(diffs[:, idx], 0.99)),
            )
        )
    print()

    if state_dim >= 9 and action_dim >= 8:
        left = states[:, 7]
        right = states[:, 8]
        open_width = left - right
        gcmd = actions[:, 7]
        print("== Gripper 映射分析 ==")
        print(
            "state gripper: left={} right={} | action gripper: {}".format(
                state_names[7] if len(state_names) > 7 else "state_7",
                state_names[8] if len(state_names) > 8 else "state_8",
                action_names[7] if len(action_names) > 7 else "action_7",
            )
        )
        print(
            "corr(action_gripper_cmd, left)={:.6f} | corr(action_gripper_cmd, right)={:.6f} | "
            "corr(action_gripper_cmd, open_width=left-right)={:.6f}".format(
                float(np.corrcoef(gcmd, left)[0, 1]),
                float(np.corrcoef(gcmd, right)[0, 1]),
                float(np.corrcoef(gcmd, open_width)[0, 1]),
            )
        )
        print(
            "结论: gripper 不是 1:1 同维度对齐；当前 schema 是 `双指 qpos state -> 单标量 gripper_cmd action`。"
        )
        print()

    if args.checkpoint_dir is not None:
        checkpoint_dir = _resolve_checkpoint_dir(args.checkpoint_dir)
        finetune_cfg = _read_json(checkpoint_dir / "lerobot_finetune_config.json")
        print("== Checkpoint 配置 ==")
        print(f"checkpoint_dir: {checkpoint_dir}")
        print(f"dataset_name: {finetune_cfg.get('dataset_name')}")
        print(f"action_chunk_size: {finetune_cfg.get('action_chunk_size')}")
        print(f"use_proprio: {finetune_cfg.get('use_proprio')}")
        print(f"num_images_in_input: {finetune_cfg.get('num_images_in_input')}")
        print(f"task_description: {finetune_cfg.get('task_description')}")
        print(f"norm_stats_path: {finetune_cfg.get('norm_stats_path')}")
        print()

    strict_joint_match = all(matched for _, _, _, matched in shared_name_match[:compare_joint_dim])
    if strict_joint_match:
        print("结论: joint 维度语义顺序对齐，当前更像是协议层用错了 HTTP 客户端，而不是 joint 维度错位。")
    else:
        print("结论: 检测到 joint 维度命名不一致，存在真实错位风险，需要进一步修正 state/action 映射。")


if __name__ == "__main__":
    main()
