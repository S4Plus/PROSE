#!/usr/bin/env python3
"""左臂蓝色圆环套轴承任务的 MP4 + NPY -> LeRobot 转换入口。

该脚本是 `convert_mp4_joint_state_to_lerobot.py` 的任务专用封装：

```
raw_dir/
  episode_0/
    episode_0.mp4
    franka_joint_state.npy
  episode_1/
    episode_1.mp4
    franka_joint_state.npy
```

默认写入的 LeRobot task prompt:
`pick up the blue ring and fit it onto the bearing.`
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


TASK_NAME = "left_arm_pick_blue_ring_to_bearing"
TASK_PROMPT = "pick up the blue ring and fit it onto the bearing."
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = Path("/workspace/liushuo/data/mp4")
LEROBOT_ROOT = Path("/workspace/liushuo/data/dataset")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="将左臂抓起蓝色圈并套进轴承的 MP4 + franka_joint_state.npy 数据转成 LeRobot 数据集"
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=RAW_ROOT / TASK_NAME,
        help="包含 episode_* 子目录的原始目录",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default=f"local/{TASK_NAME}",
        help="LeRobot repo_id，默认使用本地 repo_id",
    )
    parser.add_argument(
        "--lerobot-home",
        type=Path,
        default=LEROBOT_ROOT / TASK_NAME,
        help="LeRobot 输出目录",
    )
    parser.add_argument(
        "--task",
        type=str,
        default=TASK_PROMPT,
        help="写入 LeRobot 的任务语言指令",
    )
    parser.add_argument("--fps", type=int, default=30, help="数据帧率")
    parser.add_argument("--robot-type", type=str, default="panda", help="LeRobot 元数据中的机器人类型")
    parser.add_argument("--overwrite", action="store_true", help="若输出目录已存在则覆盖")
    parser.add_argument("--max-episodes", type=int, default=None, help="仅转换前 N 条有效 episode")
    parser.add_argument(
        "--frame-mismatch-policy",
        choices=("trim", "skip", "error"),
        default="trim",
        help="视频帧数和 joint_state 长度不一致时的处理方式",
    )
    parser.add_argument("--image-writer-processes", type=int, default=4, help="LeRobot 图像编码进程数")
    parser.add_argument("--image-writer-threads", type=int, default=8, help="LeRobot 图像编码线程数")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    converter = PROJECT_ROOT / "franka" / "convert_mp4_joint_state_to_lerobot.py"
    command = [
        sys.executable,
        str(converter),
        "--raw-dir",
        str(args.raw_dir),
        "--repo-id",
        args.repo_id,
        "--lerobot-home",
        str(args.lerobot_home),
        "--task",
        args.task,
        "--fps",
        str(args.fps),
        "--robot-type",
        args.robot_type,
        "--frame-mismatch-policy",
        args.frame_mismatch_policy,
        "--image-writer-processes",
        str(args.image_writer_processes),
        "--image-writer-threads",
        str(args.image_writer_threads),
    ]
    if args.overwrite:
        command.append("--overwrite")
    if args.max_episodes is not None:
        command.extend(["--max-episodes", str(args.max_episodes)])

    print(f"[task-name] {TASK_NAME}", flush=True)
    print(f"[task-prompt] {args.task}", flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


if __name__ == "__main__":
    main()
