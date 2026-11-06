#!/usr/bin/env python3
"""将双臂三相机 MP4 + joint_state.npy 数据转成 LeRobot 数据集。

输入目录约定：

```
raw_dir/
  episode_0/
    agent_view.mp4
    left_wrist.mp4
    right_wrist.mp4
    left_arm_joint_state.npy
    right_arm_joint_state.npy
```

左右臂 joint_state 支持 `(T, 9)` 和 `(T, 8)`：
- `(T, 9)`: timestamp + 7 维关节角 + gripper width。
- `(T, 8)`: 7 维关节角 + gripper width。

输出 state/action 均为 `left(7+gripper_cmd) + right(7+gripper_cmd)`。
动作使用下一时刻 absolute target，夹爪命令二值化为 `0=open, 1=close`。
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
from tqdm.auto import tqdm


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[1]
LEROBOT_SRC_CANDIDATES = (
    PROJECT_ROOT / "lerobot_2.1" / "src",
    PROJECT_ROOT.parent / "lerobot_2.1" / "src",
)
CAMERA_FILES = {
    "agent_view": "agent_view.mp4",
    "left_wrist": "left_wrist.mp4",
    "right_wrist": "right_wrist.mp4",
}
GRIPPER_OPEN_CMD = np.float32(0.0)
GRIPPER_CLOSE_CMD = np.float32(1.0)


class SkipEpisodeError(RuntimeError):
    """用于跳过字段不完整或帧数异常的 episode。"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="将双臂三相机数据转成 LeRobot 数据集")
    parser.add_argument("--raw-dir", type=Path, required=True, help="包含 episode_* 子目录的原始目录")
    parser.add_argument("--repo-id", type=str, required=True, help="LeRobot repo_id")
    parser.add_argument("--lerobot-home", type=Path, required=True, help="LeRobot 输出目录")
    parser.add_argument("--task", type=str, required=True, help="写入 LeRobot 的任务语言指令")
    parser.add_argument("--fps", type=int, default=30, help="数据帧率")
    parser.add_argument("--robot-type", type=str, default="dual_franka", help="LeRobot 元数据中的机器人类型")
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


def discretize_gripper_width(gripper_width: np.ndarray) -> np.ndarray:
    """将连续 gripper width 按 episode 内中点阈值离散成开闭命令。"""
    min_width = float(np.min(gripper_width))
    max_width = float(np.max(gripper_width))
    if not np.isfinite(min_width) or not np.isfinite(max_width):
        raise ValueError(f"gripper_width 含非有限值: min={min_width}, max={max_width}")
    if max_width - min_width <= 1e-6:
        return np.full(gripper_width.shape, GRIPPER_OPEN_CMD, dtype=np.float32)
    threshold = np.float32((min_width + max_width) * 0.5)
    return np.where(gripper_width <= threshold, GRIPPER_CLOSE_CMD, GRIPPER_OPEN_CMD).astype(np.float32)


def split_joint_state(joint_state: np.ndarray, arm_name: str) -> tuple[np.ndarray, np.ndarray]:
    """从单臂 joint_state 中取出 7 维关节角和二值 gripper command。"""
    if joint_state.ndim != 2 or joint_state.shape[1] not in (8, 9):
        raise ValueError(f"{arm_name} joint_state 维度异常，期望 (T, 8/9)，实际 {joint_state.shape}")
    if joint_state.shape[1] == 9:
        qpos = joint_state[:, 1:8]
        gripper_width = joint_state[:, 8]
    else:
        qpos = joint_state[:, :7]
        gripper_width = joint_state[:, 7]
    qpos = qpos.astype(np.float32, copy=False)
    gripper_cmd = discretize_gripper_width(gripper_width.astype(np.float32, copy=False))[:, None]
    return qpos, gripper_cmd


def build_state_action(left_joint_state: np.ndarray, right_joint_state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """构造 left+right 的当前 state 和下一时刻 absolute action。"""
    left_qpos, left_gripper = split_joint_state(left_joint_state, "left")
    right_qpos, right_gripper = split_joint_state(right_joint_state, "right")
    length = min(left_qpos.shape[0], right_qpos.shape[0])
    left = np.concatenate([left_qpos[:length], left_gripper[:length]], axis=1)
    right = np.concatenate([right_qpos[:length], right_gripper[:length]], axis=1)
    state = np.concatenate([left, right], axis=1).astype(np.float32, copy=False)
    action = np.concatenate([state[1:], state[-1:]], axis=0).astype(np.float32, copy=False)
    return state, action


def read_video_frames(mp4_path: Path, *, desc: str) -> np.ndarray:
    """读取 MP4 为 uint8 HWC RGB 帧序列。"""
    cap = cv2.VideoCapture(str(mp4_path))
    if not cap.isOpened():
        raise SkipEpisodeError(f"{mp4_path} 无法打开")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames: list[np.ndarray] = []
    try:
        progress = tqdm(
            total=frame_count if frame_count > 0 else None,
            desc=desc,
            unit="frame",
            leave=False,
            dynamic_ncols=True,
            file=sys.stdout,
        )
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
            progress.update(1)
    finally:
        progress.close()
        cap.release()

    if not frames:
        raise SkipEpisodeError(f"{mp4_path} 没有可读取视频帧")
    return np.stack(frames, axis=0).astype(np.uint8, copy=False)


def load_required_file(episode_dir: Path, filename: str) -> Path:
    path = episode_dir / filename
    if not path.exists():
        raise SkipEpisodeError(f"{episode_dir.name} 缺少 {filename}")
    return path


def load_episode_payload(episode_dir: Path, frame_mismatch_policy: str) -> dict[str, object]:
    video_paths = {name: load_required_file(episode_dir, filename) for name, filename in CAMERA_FILES.items()}
    left_state_path = load_required_file(episode_dir, "left_arm_joint_state.npy")
    right_state_path = load_required_file(episode_dir, "right_arm_joint_state.npy")

    try:
        left_joint_state = np.load(left_state_path)
        right_joint_state = np.load(right_state_path)
    except Exception as exc:
        raise SkipEpisodeError(f"{episode_dir.name} 无法读取 joint_state.npy: {exc}") from exc

    state, action = build_state_action(left_joint_state, right_joint_state)
    images = {
        name: read_video_frames(path, desc=f"{episode_dir.name}:{name}") for name, path in video_paths.items()
    }

    lengths = {"state": state.shape[0], **{name: frames.shape[0] for name, frames in images.items()}}
    if len(set(lengths.values())) != 1:
        message = f"{episode_dir.name} 帧数不一致: {lengths}"
        if frame_mismatch_policy == "error":
            raise ValueError(message)
        if frame_mismatch_policy == "skip":
            raise SkipEpisodeError(message)
        length = min(lengths.values())
        state = state[:length]
        action = action[:length]
        images = {name: frames[:length] for name, frames in images.items()}

    return {
        "episode_index": episode_index_from_path(episode_dir),
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
    print(f"[scan] found {len(episode_dirs)} episode dirs under {args.raw_dir}", flush=True)
    for episode_dir in tqdm(
        episode_dirs,
        desc="Scanning episodes",
        unit="episode",
        dynamic_ncols=True,
        file=sys.stdout,
    ):
        if args.max_episodes is not None and len(payloads) >= args.max_episodes:
            break
        try:
            payloads.append(load_episode_payload(episode_dir, args.frame_mismatch_policy))
        except SkipEpisodeError as exc:
            print(f"[skip] {exc}", flush=True)
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
    sample_images = sample["images"]
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

    image_features = {
        f"observation.images.{name}": {
            "dtype": "video",
            "shape": tuple(int(dim) for dim in sample_images[name].shape[1:]),
            "names": ["height", "width", "channel"],
        }
        for name in CAMERA_FILES
    }
    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        root=output_path,
        robot_type=args.robot_type,
        fps=args.fps,
        features={
            **image_features,
            "observation.state": {
                "dtype": "float32",
                "shape": (state_dim,),
                "names": [
                    *[f"left_joint_{idx}" for idx in range(7)],
                    "left_gripper_cmd",
                    *[f"right_joint_{idx}" for idx in range(7)],
                    "right_gripper_cmd",
                ],
            },
            "action": {
                "dtype": "float32",
                "shape": (action_dim,),
                "names": [
                    *[f"left_joint_{idx}" for idx in range(7)],
                    "left_gripper_cmd",
                    *[f"right_joint_{idx}" for idx in range(7)],
                    "right_gripper_cmd",
                ],
            },
        },
        use_videos=True,
        image_writer_processes=args.image_writer_processes,
        image_writer_threads=args.image_writer_threads,
    )

    total_frames = 0
    converted_episodes: list[dict[str, int | str]] = []
    for payload in tqdm(
        payloads,
        desc="Writing episodes",
        unit="episode",
        dynamic_ncols=True,
        file=sys.stdout,
    ):
        num_frames = int(payload["num_frames"])
        images = payload["images"]
        episode_name = f"episode_{int(payload['episode_index'])}"
        for frame_idx in tqdm(
            range(num_frames),
            desc=f"Writing {episode_name}",
            unit="frame",
            leave=False,
            dynamic_ncols=True,
            file=sys.stdout,
        ):
            dataset.add_frame(
                {
                    "observation.images.agent_view": images["agent_view"][frame_idx],
                    "observation.images.left_wrist": images["left_wrist"][frame_idx],
                    "observation.images.right_wrist": images["right_wrist"][frame_idx],
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
                "episode": episode_name,
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
        "state_semantics": "left(qpos7+gripper_cmd1)+right(qpos7+gripper_cmd1)",
        "action_semantics": "next_left(qpos7+gripper_cmd1)+next_right(qpos7+gripper_cmd1)",
        "gripper_command_source": "per-arm gripper_width -> binary command (0=open, 1=close)",
        "image_source": "agent_view/right_wrist/left_wrist mp4",
        "converted_episodes": converted_episodes,
        "skipped_episodes": skipped_episodes,
    }
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "conversion_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
