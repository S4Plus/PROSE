#!/usr/bin/env python3
"""
可视化一批 rollout_raw_ep*.pkl 中 peg / nut 的 XY 轨迹。

用法示例：
python mymimicgen/plot_nut_peg_positions.py \
  --rollout-dir /workspace/liushuo/cupid/mymimicgen/outputs/eval_raw/2026-01-06/20-29-38 \
  --outfile positions.png

默认取前 100 条（按文件名排序），在 10x10 网格里画 nut/peg 的 XY 轨迹：
  - peg 轨迹（通常静止）蓝色虚线
  - nut 轨迹橙色实线，起点圆点，终点叉号
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot nut/peg XY trajectories from rollout_raw_ep*.pkl.")
    p.add_argument("--rollout-dir", type=Path, required=True, help="包含 rollout_raw_ep*.pkl 的目录")
    p.add_argument("--outfile", type=Path, default=None, help="输出图片路径，默认 <rollout-dir>/positions.png")
    p.add_argument("--limit", type=int, default=100, help="最多可视化多少条（默认 100）")
    p.add_argument("--nut-key", type=str, default="SquareNut_pos", help="nut 位置键名")
    p.add_argument("--peg-key", type=str, default="peg1_pos", help="peg 位置键名")
    return p.parse_args()


def load_positions(pkl_path: Path, nut_key: str, peg_key: str) -> Tuple[np.ndarray, np.ndarray]:
    with pkl_path.open("rb") as f:
        data = pickle.load(f)
    raw = data.get("raw_obs") or data.get("raw") or []
    if not raw:
        raise ValueError(f"{pkl_path.name} 缺少 raw_obs/raw")
    nut = np.stack([r[nut_key] for r in raw], axis=0)  # (T,3)
    peg = np.stack([r[peg_key] for r in raw], axis=0)  # (T,3)
    return nut, peg


def main() -> None:
    args = parse_args()
    rollout_dir = args.rollout_dir.expanduser().resolve()
    paths: List[Path] = sorted(rollout_dir.glob("rollout_raw_ep*.pkl"))[: args.limit]
    if not paths:
        raise SystemExit(f"未找到 rollout_raw_ep*.pkl: {rollout_dir}")

    n = len(paths)
    grid = int(np.ceil(np.sqrt(n)))
    fig, axes = plt.subplots(grid, grid, figsize=(grid * 2.4, grid * 2.4), squeeze=False)

    for idx, (ax, p) in enumerate(zip(axes.flatten(), paths)):
        try:
            nut, peg = load_positions(p, args.nut_key, args.peg_key)
        except Exception as e:
            ax.text(0.5, 0.5, f"err: {e}", ha="center", va="center", fontsize=6)
            ax.axis("off")
            continue

        ax.plot(peg[:, 0], peg[:, 1], "b--", linewidth=0.8, alpha=0.7, label="peg")
        ax.plot(nut[:, 0], nut[:, 1], color="orange", linewidth=1.0, alpha=0.9, label="nut")
        ax.scatter(nut[0, 0], nut[0, 1], color="orange", s=8)
        ax.scatter(nut[-1, 0], nut[-1, 1], color="orange", marker="x", s=12)
        ax.set_title(p.stem, fontsize=6)
        ax.set_xticks([])
        ax.set_yticks([])

    # 隐藏多余子图
    for ax in axes.flatten()[n:]:
        ax.axis("off")

    plt.tight_layout()
    outfile = args.outfile or (rollout_dir / "positions.png")
    outfile = outfile.expanduser().resolve()
    fig.savefig(outfile, dpi=200)
    print(f"已保存: {outfile}")


if __name__ == "__main__":
    main()
