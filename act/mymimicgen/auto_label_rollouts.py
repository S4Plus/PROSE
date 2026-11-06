#!/usr/bin/env python3
"""
根据 rollout_ep*.pkl 自动打失败标签。
支持两种数据格式：
1) 完整 mimicgen obs（包含 robot0_eef_pos/object 等）
2) 精简版 eval 存储（只有 observation.state=关节+夹爪）

标签集合：
  success / grasp_fail / drop_after_grasp / insertion_fail / ood

用法示例：
python auto_label_rollouts.py \
  --rollout-dir outputs/eval_cli/2026-01-05/15-56-10 \
  --goal-pos 0.0 0.0 0.0 \
  --outfile outputs/eval_cli/2026-01-05/15-56-10/failure_labels_auto.csv

若未提供 goal_pos，将用末帧物体位置作为“成功”参考，只能区分部分错误，建议提供目标位置。
"""

from __future__ import annotations

import argparse
import math
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


FAILURE_TAGS = ["success", "grasp_fail", "drop_after_grasp", "insertion_fail", "ood"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--rollout-dir", type=Path, required=True, help="包含 rollout_ep*.pkl 的目录")
    p.add_argument("--outfile", type=Path, default=None, help="输出 CSV，默认 <rollout-dir>/failure_labels_auto.csv")
    p.add_argument("--goal-pos", type=float, nargs=3, default=None, help="目标位置 (x y z)，用于放置判定")
    p.add_argument("--close-thresh", type=float, default=-0.01, help="夹爪闭合判定阈值（qpos 差分）")
    p.add_argument("--reopen-early-steps", type=int, default=10, help="闭合后多少步内重新张开视为抓取失败")
    p.add_argument("--drop-reopen-steps", type=int, default=15, help="闭合超过该步数后再张开视为抓取后掉落")
    return p.parse_args()


def load_episode(pkl_path: Path) -> Dict:
    with pkl_path.open("rb") as f:
        return pickle.load(f)


def l2(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.linalg.norm(a - b, axis=-1)


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    """q: (...,4) xyzw"""
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    m00 = 1 - 2 * (yy + zz)
    m01 = 2 * (xy - wz)
    m02 = 2 * (xz + wy)
    m10 = 2 * (xy + wz)
    m11 = 1 - 2 * (xx + zz)
    m12 = 2 * (yz - wx)
    m20 = 2 * (xz - wy)
    m21 = 2 * (yz + wx)
    m22 = 1 - 2 * (xx + yy)
    return np.stack(
        [
            np.stack([m00, m01, m02], axis=-1),
            np.stack([m10, m11, m12], axis=-1),
            np.stack([m20, m21, m22], axis=-1),
        ],
        axis=-2,
    )


def quat_angle(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """返回两个四元数的角度差（弧度），q 格式 xyzw"""
    dot = np.clip(np.sum(q1 * q2, axis=-1), -1.0, 1.0)
    return 2 * np.arccos(np.abs(dot))


def detect_grasp_event(gripper_qpos: np.ndarray, thresh: float = -0.01) -> bool:
    """简单抓取事件：开合值下降/变小"""
    if len(gripper_qpos) < 3:
        return False
    dq = np.diff(gripper_qpos, axis=0)
    return np.min(dq) < thresh


def label_with_minimal_state(
    state: np.ndarray,
    success_flag: bool,
    d_close: float = -0.01,
    reopen_early_steps: int = 10,
    drop_reopen_steps: int = 15,
) -> Tuple[str, str]:
    """
    当只有 observation.state(9) 可用时的退化打标：
    - 假设 state = 7 关节 + 2 夹爪 qpos
    - 只能粗略区分：success / grasp_fail / drop_after_grasp / insertion_fail
    """
    gripper = state[:, -2:]
    dq = np.diff(gripper, axis=0)
    close_idx = np.where(dq < d_close)[0]
    reopen_idx = np.where(dq > abs(d_close))[0]

    if success_flag:
        return "success", "final_success_flag"
    if len(close_idx) == 0:
        return "grasp_fail", "no_gripper_close"

    first_close = int(close_idx[0])
    first_reopen = int(reopen_idx[0]) if len(reopen_idx) > 0 else None

    if first_reopen is not None:
        gap = first_reopen - first_close
        if gap <= reopen_early_steps:
            return "grasp_fail", "close_then_reopen_early"
        if gap >= drop_reopen_steps:
            return "drop_after_grasp", f"reopen_after_{gap}_steps"
    # 有闭合但无明显重开，判为插入失败
    return "insertion_fail", "closed_no_success_flag"


def label_episode(
    data: Dict,
    goal_pos: Optional[np.ndarray],
    close_thresh: float,
    reopen_early_steps: int,
    drop_reopen_steps: int,
) -> Tuple[str, str]:
    # 支持两种存储格式：
    # A) eval episodes: flat dict with observation.state
    # B) mimicgen full obs: observation -> robot0_eef_pos, robot0_gripper_qpos, object, etc.
    # 优先使用完整 obs；若缺失则退化为 state-only 规则。
    success_flag = False
    if "next.success" in data:
        try:
            success_flag = bool(data["next.success"][-1].item())
        except Exception:
            success_flag = False

    if "observation" not in data:
        if "observation.state" in data:
            state = np.asarray(data["observation.state"])
            return label_with_minimal_state(
                state,
                success_flag,
                d_close=close_thresh,
                reopen_early_steps=reopen_early_steps,
                drop_reopen_steps=drop_reopen_steps,
            )
        if "obs" in data:
            obs = data["obs"]
            if "observation" in obs:
                data = obs  # try nested

    if "observation" not in data:
        # 仍然缺 obs，退化为 success / ood
        return ("success" if success_flag else "ood"), "no_obs_keys"

    obs = data["observation"]
    if not isinstance(obs, dict):
        # unexpected structure
        if "state" in data:
            return label_with_minimal_state(
                np.asarray(data["state"]),
                success_flag,
                d_close=close_thresh,
                reopen_early_steps=reopen_early_steps,
                drop_reopen_steps=drop_reopen_steps,
            )
        return ("success" if success_flag else "ood"), "obs_not_dict"

    # 完整 obs 模式
    if ("robot0_eef_pos" not in obs) or ("robot0_gripper_qpos" not in obs) or ("object" not in obs):
        # 缺关键字段，退化
        if "state" in obs:
            return label_with_minimal_state(
                np.asarray(obs["state"]),
                success_flag,
                d_close=close_thresh,
                reopen_early_steps=reopen_early_steps,
                drop_reopen_steps=drop_reopen_steps,
            )
        return ("success" if success_flag else "ood"), "incomplete_obs"

    eef_pos = np.asarray(obs["robot0_eef_pos"])
    gripper = np.asarray(obs["robot0_gripper_qpos"])
    obj = np.asarray(obs["object"])

    obj_pos = obj[:, :3]
    obj_vel = obj[:, 7:10] if obj.shape[1] >= 10 else None

    # 估计目标位置：优先 cli 提供，其次用末帧物体位置
    goal = goal_pos if goal_pos is not None else obj_pos[-1]

    dist_obj_goal = l2(obj_pos, goal)
    final_dist = float(dist_obj_goal[-1])

    success = final_dist < 0.035
    if success and obj_vel is not None:
        success = success and float(np.linalg.norm(obj_vel[-1])) < 0.02
    if success_flag:
        success = True

    if success:
        return "success", f"final_dist={final_dist:.3f}"

    # 仍用简化规则（基于夹爪开合）来分类
    return label_with_minimal_state(
        np.concatenate([np.zeros((gripper.shape[0], 7)), gripper], axis=1),
        success_flag=False,
        d_close=close_thresh,
        reopen_early_steps=reopen_early_steps,
        drop_reopen_steps=drop_reopen_steps,
    )


def main() -> None:
    args = parse_args()
    rollout_dir = args.rollout_dir.expanduser().resolve()
    if not rollout_dir.is_dir():
        sys.exit(f"rollout 目录不存在: {rollout_dir}")

    outfile = (
        args.outfile
        if args.outfile is not None
        else rollout_dir / "failure_labels_auto.csv"
    )
    outfile = outfile.expanduser().resolve()

    goal = np.array(args.goal_pos, dtype=float) if args.goal_pos is not None else None

    pkl_paths = sorted(rollout_dir.glob("rollout_*.pkl"))
    if not pkl_paths:
        sys.exit(f"未找到 rollout_ep*.pkl: {rollout_dir}")

    rows = []
    for pkl_path in pkl_paths:
        ep_idx_str = pkl_path.stem.replace("rollout_ep", "")
        try:
            ep_idx = int(ep_idx_str)
        except ValueError:
            ep_idx = -1
        data = load_episode(pkl_path)
        label, note = label_episode(
            data,
            goal,
            close_thresh=args.close_thresh,
            reopen_early_steps=args.reopen_early_steps,
            drop_reopen_steps=args.drop_reopen_steps,
        )
        rows.append((ep_idx, label, note, str(pkl_path)))
        print(f"ep {ep_idx_str}: {label} | {note}")

    outfile.parent.mkdir(parents=True, exist_ok=True)
    with outfile.open("w") as f:
        f.write("episode_index,label,note,pkl_path\n")
        for r in rows:
            f.write(f"{r[0]},{r[1]},{r[2]},{r[3]}\n")

    print(f"\n已写入: {outfile}，共 {len(rows)} 条")


if __name__ == "__main__":
    main()
