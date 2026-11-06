#!/usr/bin/env python3
"""将 MP4 + franka_joint_state.npy 格式转成 LeRobot 数据集。

输入目录约定：

```
raw_dir/
  episode_0/
    episode_0.mp4
    franka_joint_state.npy
  episode_1/
    episode_1.mp4
    franka_joint_state.npy
```

`franka_joint_state.npy` 期望为 `(T, 8)`，前 7 维为关节角，最后 1 维为 gripper width。
输出动作语义为下一时刻关节角 + 下一时刻二值 gripper command。
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[1]
LEROBOT_SRC_CANDIDATES = (
    PROJECT_ROOT / "lerobot_2.1" / "src",
    PROJECT_ROOT.parent / "lerobot_2.1" / "src",
)
GRIPPER_OPEN_CMD = np.float32(0.0)
GRIPPER_CLOSE_CMD = np.float32(1.0)


class SkipEpisodeError(RuntimeError):
    """用于跳过字段不完整或帧数异常的 episode。"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="将 MP4 + franka_joint_state.npy 转成 LeRobot 数据集")
    parser.add_argument("--raw-dir", type=Path, required=True, help="包含 episode_* 子目录的原始目录")
    parser.add_argument("--repo-id", type=str, required=True, help="LeRobot repo_id")
    parser.add_argument("--lerobot-home", type=Path, required=True, help="LeRobot 输出根目录")
    parser.add_argument("--task", type=str, required=True, help="写入 LeRobot 的任务语言指令")
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


def ensure_lerobot_importable() -> None:
    """优先使用当前 uv 环境里的 LeRobot；源码路径只作为回退。"""
    try:
        import lerobot  # noqa: F401

        return
    except ModuleNotFoundError:
        pass

    for src_dir in LEROBOT_SRC_CANDIDATES:
        if src_dir.is_dir():
            sys.path.insert(0, str(src_dir))
            return
    searched = ", ".join(str(path) for path in LEROBOT_SRC_CANDIDATES)
    raise FileNotFoundError(f"未找到 lerobot 源码目录，已检查: {searched}")


def episode_index_from_path(path: Path) -> int:
    if not path.name.startswith("episode_"):
        raise ValueError(f"非法 episode 目录名: {path.name}")
    return int(path.name.split("_", maxsplit=1)[1])


def iter_episode_dirs(raw_dir: Path) -> list[Path]:
    dirs = [path for path in raw_dir.glob("episode_*") if path.is_dir()]
    return sorted(dirs, key=episode_index_from_path)


def find_mp4_path(episode_dir: Path) -> Path:
    preferred = episode_dir / f"{episode_dir.name}.mp4"
    if preferred.exists():
        return preferred
    candidates = sorted(path for path in episode_dir.glob("*.mp4") if not path.name.startswith("."))
    if not candidates:
        raise SkipEpisodeError(f"{episode_dir.name} 缺少 mp4 文件")
    if len(candidates) > 1:
        raise SkipEpisodeError(f"{episode_dir.name} 存在多个 mp4 文件，无法自动选择")
    return candidates[0]


def discretize_gripper_width(gripper_width: np.ndarray) -> np.ndarray:
    """将连续 gripper width 离散成二值开闭命令。"""
    min_width = float(np.min(gripper_width))
    max_width = float(np.max(gripper_width))
    if not np.isfinite(min_width) or not np.isfinite(max_width):
        raise ValueError(f"gripper_width 含非有限值: min={min_width}, max={max_width}")
    if max_width - min_width <= 1e-6:
        return np.full(gripper_width.shape, GRIPPER_OPEN_CMD, dtype=np.float32)
    threshold = np.float32((min_width + max_width) * 0.5)
    return np.where(gripper_width <= threshold, GRIPPER_CLOSE_CMD, GRIPPER_OPEN_CMD).astype(np.float32)


def build_state_action(joint_state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """从 joint_state 构造当前 state 和下一时刻 absolute action。"""
    if joint_state.ndim != 2 or joint_state.shape[1] < 8:
        raise ValueError(f"joint_state 维度异常，期望 (T, >=8)，实际 {joint_state.shape}")
    qpos = joint_state[:, :7].astype(np.float32, copy=False)
    gripper_cmd = discretize_gripper_width(joint_state[:, 7].astype(np.float32, copy=False))[:, None]

    state = np.concatenate([qpos, gripper_cmd], axis=1).astype(np.float32, copy=False)
    next_qpos = np.concatenate([qpos[1:], qpos[-1:]], axis=0)
    next_gripper = np.concatenate([gripper_cmd[1:], gripper_cmd[-1:]], axis=0)
    action = np.concatenate([next_qpos, next_gripper], axis=1).astype(np.float32, copy=False)
    return state, action


def read_video_frames(mp4_path: Path) -> np.ndarray:
    """读取 MP4 为 uint8 HWC RGB 帧序列。"""
    cap = cv2.VideoCapture(str(mp4_path))
    if not cap.isOpened():
        raise SkipEpisodeError(f"{mp4_path} 无法打开")

    frames: list[np.ndarray] = []
    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    finally:
        cap.release()

    if not frames:
        raise SkipEpisodeError(f"{mp4_path} 没有可读取视频帧")
    return np.stack(frames, axis=0).astype(np.uint8, copy=False)


def load_episode_payload(episode_dir: Path, frame_mismatch_policy: str) -> dict[str, object]:
    mp4_path = find_mp4_path(episode_dir)
    joint_state_path = episode_dir / "franka_joint_state.npy"
    if not joint_state_path.exists():
        raise SkipEpisodeError(f"{episode_dir.name} 缺少 franka_joint_state.npy")

    try:
        joint_state = np.load(joint_state_path)
    except Exception as exc:
        raise SkipEpisodeError(f"{episode_dir.name} 无法读取 franka_joint_state.npy: {exc}") from exc

    state, action = build_state_action(joint_state)
    images = read_video_frames(mp4_path)

    if images.shape[0] != state.shape[0]:
        message = f"{episode_dir.name} 帧数不一致: video={images.shape[0]}, joint_state={state.shape[0]}"
        if frame_mismatch_policy == "error":
            raise ValueError(message)
        if frame_mismatch_policy == "skip":
            raise SkipEpisodeError(message)
        length = min(images.shape[0], state.shape[0])
        images = images[:length]
        state = state[:length]
        action = action[:length]

    return {
        "episode_index": episode_index_from_path(episode_dir),
        "mp4": mp4_path.name,
        "num_frames": int(state.shape[0]),
        "state": state,
        "action": action,
        "images": images,
    }


def main() -> None:
    args = parse_args()
    if not args.raw_dir.exists():
        raise FileNotFoundError(f"原始数据目录不存在: {args.raw_dir}")

    episode_dirs = iter_episode_dirs(args.raw_dir)
    if not episode_dirs:
        raise FileNotFoundError(f"未在 {args.raw_dir} 下找到 episode_* 子目录")

    payloads: list[dict[str, object]] = []
    skipped_episodes: list[dict[str, int | str]] = []
    for episode_dir in episode_dirs:
        if args.max_episodes is not None and len(payloads) >= args.max_episodes:
            break
        try:
            payloads.append(load_episode_payload(episode_dir, args.frame_mismatch_policy))
        except SkipEpisodeError as exc:
            print(f"[skip] {exc}")
            skipped_episodes.append(
                {
                    "episode_index": episode_index_from_path(episode_dir),
                    "episode": episode_dir.name,
                    "reason": str(exc),
                }
            )

    if not payloads:
        raise RuntimeError(f"{args.raw_dir} 没有成功读取任何有效 episode")

    sample = payloads[0]
    image_shape = tuple(int(dim) for dim in sample["images"].shape[1:])
    state_dim = int(sample["state"].shape[1])
    action_dim = int(sample["action"].shape[1])

    os.environ["HF_LEROBOT_HOME"] = str(args.lerobot_home.resolve())
    ensure_lerobot_importable()
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset  # type: ignore  # noqa: E402
    except ModuleNotFoundError:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset  # type: ignore  # noqa: E402

    output_path = args.lerobot_home.resolve()
    if output_path.exists():
        if not args.overwrite:
            raise FileExistsError(f"输出目录已存在: {output_path}，如需覆盖请添加 --overwrite")
        shutil.rmtree(output_path)

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        root=output_path,
        robot_type=args.robot_type,
        fps=args.fps,
        features={
            "observation.images.wrist_camera": {
                "dtype": "video",
                "shape": image_shape,
                "names": ["height", "width", "channel"],
            },
            "observation.state": {
                "dtype": "float32",
                "shape": (state_dim,),
                "names": [f"joint_{idx}" for idx in range(7)] + ["gripper_cmd"],
            },
            "action": {
                "dtype": "float32",
                "shape": (action_dim,),
                "names": [f"joint_{idx}" for idx in range(7)] + ["gripper_cmd"],
            },
        },
        use_videos=True,
        image_writer_processes=args.image_writer_processes,
        image_writer_threads=args.image_writer_threads,
    )

    total_frames = 0
    converted_episodes: list[dict[str, int | str]] = []
    for payload in payloads:
        num_frames = int(payload["num_frames"])
        for frame_idx in range(num_frames):
            dataset.add_frame(
                {
                    "observation.images.wrist_camera": payload["images"][frame_idx],
                    "observation.state": payload["state"][frame_idx],
                    "action": payload["action"][frame_idx],
                    "task": args.task,
                },
            )
        dataset.save_episode()
        total_frames += num_frames
        converted_episodes.append(
            {
                "episode_index": int(payload["episode_index"]),
                "mp4": str(payload["mp4"]),
                "num_frames": num_frames,
            }
        )

    summary = {
        "raw_dir": str(args.raw_dir.resolve()),
        "repo_id": args.repo_id,
        "lerobot_home": str(output_path),
        "total_episodes": len(converted_episodes),
        "skipped_episodes_count": len(skipped_episodes),
        "total_frames": total_frames,
        "fps": args.fps,
        "task": args.task,
        "state_semantics": "joint_state[:7]+gripper_cmd(1)",
        "action_semantics": "next_joint_state[:7]+next_gripper_cmd(1)",
        "gripper_command_source": "joint_state[:,7] -> binary command (0=open, 1=close)",
        "image_source": "episode mp4 -> wrist_camera",
        "converted_episodes": converted_episodes,
        "skipped_episodes": skipped_episodes,
    }
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "conversion_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
