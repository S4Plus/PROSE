#!/usr/bin/env python3
"""
针对 eval_with_raw_logging.py 生成的 rollout_raw_ep*.pkl 做失败分类。

分类标签：
- success
- grasp_fail          （夹爪未闭合或闭合后物体几乎未抬升）
- drop_before_place   （抬起后未接近柱子就掉落/松开）
- place_fail          （接近柱子时未对准掉落或卡在柱子上）

依赖字段：raw_obs 中的 SquareNut_pos, peg1_pos, robot0_gripper_qpos。
"""

from __future__ import annotations

import argparse
import glob
import pickle
from pathlib import Path
from typing import List, Tuple

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--rollout-dir", type=Path, required=True, help="包含 rollout_raw_ep*.pkl 的目录")
    p.add_argument("--outfile", type=Path, default=None, help="输出 CSV，默认 <rollout-dir>/failure_labels_raw.csv")
    p.add_argument("--close-thresh", type=float, default=-0.005, help="夹爪闭合判定阈值（qpos 差分）")
    p.add_argument("--reopen-thresh", type=float, default=0.005, help="夹爪重新张开阈值（qpos 差分）")
    p.add_argument("--lift-thresh", type=float, default=0.02, help="抬升高度阈值 (m)")
    p.add_argument("--place-thresh", type=float, default=0.15, help="接近柱子判定距离阈值 (m)")
    p.add_argument("--drop-height-thresh", type=float, default=0.02, help="掉落判定：抬起后高度回落到初始+该值")
    p.add_argument("--reopen-early-steps", type=int, default=10, help="闭合后多少步内张开视为抓取失败")
    p.add_argument("--drop-reopen-steps", type=int, default=20, help="闭合超过该步数后张开视为掉落")
    p.add_argument("--approach-dist-thresh", type=float, default=0.06, help="未接近柱子的距离阈值（大于该值掉落视为早期掉落）")
    return p.parse_args()


def load_raw(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def classify_episode(
    raw_list: List[dict],
    close_thresh: float,
    reopen_thresh: float,
    lift_thresh: float,
    place_thresh: float,
    drop_height_thresh: float,
    reopen_early_steps: int,
    drop_reopen_steps: int,
    approach_dist_thresh: float,
) -> Tuple[str, str]:
    # 提取轨迹
    nut_pos = np.stack([r["SquareNut_pos"] for r in raw_list])  # (T+1,3)
    peg_pos = np.stack([r["peg1_pos"] for r in raw_list])
    gripper = np.stack([r["robot0_gripper_qpos"] for r in raw_list])  # (T+1,2)
    grip_scalar = gripper.mean(axis=1)
    dq = np.diff(grip_scalar)
    close_idx = np.where(dq < close_thresh)[0]
    reopen_idx = np.where(dq > reopen_thresh)[0]

    h0 = float(nut_pos[0, 2])
    max_h = float(nut_pos[:, 2].max())
    lift_amt = max_h - h0
    lifted = lift_amt > lift_thresh

    first_close = int(close_idx[0]) if len(close_idx) else None
    first_reopen = None
    if first_close is not None:
        after = reopen_idx[reopen_idx > first_close]
        if len(after):
            first_reopen = int(after[0])

    dist_to_peg = np.linalg.norm(nut_pos - peg_pos, axis=1)
    final_dist = float(dist_to_peg[-1])

    # 成功判定（位置接近 peg）
    if final_dist < place_thresh:
        return "success", f"final_dist={final_dist:.3f}"

    # 未闭合
    if first_close is None:
        return "grasp_fail", "no_gripper_close"

    # 抬升后掉落：高度回落到初始+阈值 或 闭合后较晚重开
    drop_step = None
    post_heights = nut_pos[first_close + 1 :, 2]
    if post_heights.size > 0:
        below = np.where(post_heights < h0 + drop_height_thresh)[0]
        if len(below) > 0:
            drop_step = first_close + 1 + int(below[0])
    if drop_step is None and first_reopen is not None:
        gap = first_reopen - first_close
        if gap <= reopen_early_steps:
            return "grasp_fail", f"close_then_reopen_{gap}"
        if gap >= drop_reopen_steps:
            drop_step = first_reopen

    if not lifted:
        return "grasp_fail", f"no_lift({lift_amt:.3f})"

    if drop_step is not None:
        drop_dist = float(dist_to_peg[drop_step])
        if drop_dist > approach_dist_thresh:
            return "drop_before_place", f"drop_dist={drop_dist:.3f}"
        else:
            return "place_fail", f"drop_near_peg_dist={drop_dist:.3f}"

    # 抬起但未插入
    return "place_fail", f"final_dist={final_dist:.3f}"


def main() -> None:
    args = parse_args()
    rollout_dir = args.rollout_dir.expanduser().resolve()
    if not rollout_dir.is_dir():
        raise SystemExit(f"目录不存在: {rollout_dir}")

    outfile = args.outfile or (rollout_dir / "failure_labels_raw.csv")
    outfile = outfile.expanduser().resolve()

    paths = sorted(rollout_dir.glob("rollout_raw_ep*.pkl"))
    if not paths:
        raise SystemExit(f"未找到 rollout_raw_ep*.pkl: {rollout_dir}")

    rows = []
    for p in paths:
        ep_str = p.stem.replace("rollout_raw_ep", "")
        try:
            ep_idx = int(ep_str)
        except ValueError:
            ep_idx = -1
        data = load_raw(p)
        raw_list = data["raw_obs"]
        label, note = classify_episode(
            raw_list,
            close_thresh=args.close_thresh,
            reopen_thresh=args.reopen_thresh,
            lift_thresh=args.lift_thresh,
            place_thresh=args.place_thresh,
            drop_height_thresh=args.drop_height_thresh,
            reopen_early_steps=args.reopen_early_steps,
            drop_reopen_steps=args.drop_reopen_steps,
            approach_dist_thresh=args.approach_dist_thresh,
        )
        rows.append((ep_idx, label, note, str(p)))
        print(f"ep {ep_str}: {label} | {note}")

    outfile.parent.mkdir(parents=True, exist_ok=True)
    with outfile.open("w") as f:
        f.write("episode_index,label,note,pkl_path\n")
        for r in rows:
            f.write(f"{r[0]},{r[1]},{r[2]},{r[3]}\n")
    print(f"\n已写入: {outfile}，共 {len(rows)} 条")


if __name__ == "__main__":
    main()
