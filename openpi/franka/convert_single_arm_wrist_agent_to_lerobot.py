#!/usr/bin/env python3
"""将单臂 wrist+agent 相机 MP4 + joint_state.npy 数据转成 LeRobot 数据集。

输入目录约定：

```
raw_dir/
  episode_0/
    agent_view.mp4
    left_wrist.mp4
    left_arm_joint_state.npy
```

`left_arm_joint_state.npy` 支持两种形状：
- `(T, 9)`: timestamp + 7 维关节角 + gripper width。
- `(T, 8)`: 7 维关节角 + gripper width。

输出动作语义为下一时刻 absolute joint target + 下一时刻二值 gripper command。
训练侧再通过 OpenPI config 将前 7 维 joint target 转成 delta action，最后 1 维
gripper command 保持 absolute，约定 `0=open, 1=close`。
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
}
JOINT_STATE_FILE = "left_arm_joint_state.npy"
GRIPPER_OPEN_CMD = np.float32(0.0)
GRIPPER_CLOSE_CMD = np.float32(1.0)


class SkipEpisodeError(RuntimeError):
    """用于跳过字段不完整或帧数异常的 episode。"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="将单臂 wrist+agent MP4 数据转成 LeRobot 数据集")
    parser.add_argument("--raw-dir", type=Path, required=True, help="包含 episode_* 子目录的原始目录")
    parser.add_argument("--repo-id", type=str, required=True, help="LeRobot repo_id；本地训练建议传数据集绝对路径")
    parser.add_argument("--lerobot-home", type=Path, required=True, help="LeRobot 输出目录")
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


def load_required_file(episode_dir: Path, filename: str) -> Path:
    path = episode_dir / filename
    if not path.exists():
        raise SkipEpisodeError(f"{episode_dir.name} 缺少 {filename}")
    return path


def video_frame_count(mp4_path: Path) -> int:
    cap = cv2.VideoCapture(str(mp4_path))
    if not cap.isOpened():
        raise SkipEpisodeError(f"{mp4_path} 无法打开")
    try:
        return int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()


def read_first_rgb_frame(mp4_path: Path) -> np.ndarray:
    cap = cv2.VideoCapture(str(mp4_path))
    if not cap.isOpened():
        raise SkipEpisodeError(f"{mp4_path} 无法打开")
    try:
        ok, frame_bgr = cap.read()
        if not ok:
            raise SkipEpisodeError(f"{mp4_path} 没有可读取视频帧")
        return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).astype(np.uint8, copy=False)
    finally:
        cap.release()


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


def split_joint_state(joint_state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """从 joint_state 中取出 7 维关节角和二值 gripper command。"""
    if joint_state.ndim != 2 or joint_state.shape[1] not in (8, 9):
        raise ValueError(f"joint_state 维度异常，期望 (T, 8/9)，实际 {joint_state.shape}")
    if joint_state.shape[1] == 9:
        qpos = joint_state[:, 1:8]
        gripper_width = joint_state[:, 8]
    else:
        qpos = joint_state[:, :7]
        gripper_width = joint_state[:, 7]
    qpos = qpos.astype(np.float32, copy=False)
    gripper_cmd = discretize_gripper_width(gripper_width.astype(np.float32, copy=False))[:, None]
    return qpos, gripper_cmd


def build_state_action(joint_state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """构造当前 state 和下一时刻 absolute action。"""
    qpos, gripper_cmd = split_joint_state(joint_state)
    state = np.concatenate([qpos, gripper_cmd], axis=1).astype(np.float32, copy=False)
    next_qpos = np.concatenate([qpos[1:], qpos[-1:]], axis=0)
    next_gripper = np.concatenate([gripper_cmd[1:], gripper_cmd[-1:]], axis=0)
    action = np.concatenate([next_qpos, next_gripper], axis=1).astype(np.float32, copy=False)
    return state, action


def load_state_action(episode_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    joint_state_path = load_required_file(episode_dir, JOINT_STATE_FILE)
    try:
        joint_state = np.load(joint_state_path)
    except Exception as exc:
        raise SkipEpisodeError(f"{episode_dir.name} 无法读取 {JOINT_STATE_FILE}: {exc}") from exc
    return build_state_action(joint_state)


def infer_sample(episode_dirs: list[Path]) -> tuple[dict[str, tuple[int, ...]], int, int]:
    """从第一条有效 episode 推断图像形状和 state/action 维度。"""
    last_error: Exception | None = None
    for episode_dir in episode_dirs:
        try:
            state, action = load_state_action(episode_dir)
            image_shapes = {
                name: tuple(int(dim) for dim in read_first_rgb_frame(load_required_file(episode_dir, filename)).shape)
                for name, filename in CAMERA_FILES.items()
            }
            return image_shapes, int(state.shape[1]), int(action.shape[1])
        except (SkipEpisodeError, ValueError) as exc:
            last_error = exc
            continue
    raise RuntimeError(f"未找到可用于推断 schema 的有效 episode，最后错误: {last_error}")


def trim_or_validate_lengths(
    episode_dir: Path,
    state: np.ndarray,
    action: np.ndarray,
    video_paths: dict[str, Path],
    frame_mismatch_policy: str,
) -> tuple[np.ndarray, np.ndarray, int, dict[str, int]]:
    lengths = {"state": int(state.shape[0])}
    lengths.update({name: video_frame_count(path) for name, path in video_paths.items()})
    if any(length <= 0 for length in lengths.values()):
        raise SkipEpisodeError(f"{episode_dir.name} 存在空序列: {lengths}")
    if len(set(lengths.values())) == 1:
        return state, action, int(state.shape[0]), lengths

    message = f"{episode_dir.name} 帧数不一致: {lengths}"
    if frame_mismatch_policy == "error":
        raise ValueError(message)
    if frame_mismatch_policy == "skip":
        raise SkipEpisodeError(message)

    length = min(lengths.values())
    print(f"[trim] {message} -> length={length}", flush=True)
    return state[:length], action[:length], length, lengths


def open_video_captures(video_paths: dict[str, Path]) -> dict[str, cv2.VideoCapture]:
    captures = {name: cv2.VideoCapture(str(path)) for name, path in video_paths.items()}
    failed = [name for name, cap in captures.items() if not cap.isOpened()]
    if failed:
        for cap in captures.values():
            cap.release()
        raise SkipEpisodeError(f"无法打开视频: {failed}")
    return captures


def read_capture_frame(capture: cv2.VideoCapture, camera_name: str, episode_name: str, frame_idx: int) -> np.ndarray:
    ok, frame_bgr = capture.read()
    if not ok:
        raise SkipEpisodeError(f"{episode_name}:{camera_name} 在 frame={frame_idx} 读取失败")
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).astype(np.uint8, copy=False)


def write_episode(
    dataset,
    episode_dir: Path,
    task: str,
    frame_mismatch_policy: str,
) -> dict[str, int | str | dict[str, int]]:
    video_paths = {name: load_required_file(episode_dir, filename) for name, filename in CAMERA_FILES.items()}
    state, action = load_state_action(episode_dir)
    state, action, num_frames, raw_lengths = trim_or_validate_lengths(
        episode_dir, state, action, video_paths, frame_mismatch_policy
    )
    captures = open_video_captures(video_paths)
    try:
        for frame_idx in tqdm(
            range(num_frames),
            desc=f"Writing {episode_dir.name}",
            unit="frame",
            leave=False,
            dynamic_ncols=True,
            file=sys.stdout,
        ):
            dataset.add_frame(
                {
                    "observation.images.agent_view": read_capture_frame(
                        captures["agent_view"], "agent_view", episode_dir.name, frame_idx
                    ),
                    "observation.images.left_wrist": read_capture_frame(
                        captures["left_wrist"], "left_wrist", episode_dir.name, frame_idx
                    ),
                    "observation.state": state[frame_idx],
                    "action": action[frame_idx],
                    "task": task,
                },
            )
    finally:
        for capture in captures.values():
            capture.release()

    dataset.save_episode()
    gripper_values, gripper_counts = np.unique(state[:, 7].astype(np.int32), return_counts=True)
    gripper_hist = {str(int(value)): int(count) for value, count in zip(gripper_values, gripper_counts, strict=True)}
    return {
        "episode_index": episode_index_from_path(episode_dir),
        "episode": episode_dir.name,
        "num_frames": num_frames,
        "raw_lengths": raw_lengths,
        "gripper_cmd_counts": gripper_hist,
    }


def main() -> None:
    args = parse_args()
    if not args.raw_dir.exists():
        raise FileNotFoundError(f"原始数据目录不存在: {args.raw_dir}")

    episode_dirs = iter_episode_dirs(args.raw_dir)
    if not episode_dirs:
        raise FileNotFoundError(f"未在 {args.raw_dir} 下找到 episode_* 子目录")
    if args.max_episodes is not None:
        episode_dirs = episode_dirs[: args.max_episodes]

    image_shapes, state_dim, action_dim = infer_sample(episode_dirs)

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
            "observation.images.agent_view": {
                "dtype": "video",
                "shape": image_shapes["agent_view"],
                "names": ["height", "width", "channel"],
            },
            "observation.images.left_wrist": {
                "dtype": "video",
                "shape": image_shapes["left_wrist"],
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
    converted_episodes: list[dict[str, int | str | dict[str, int]]] = []
    skipped_episodes: list[dict[str, int | str]] = []
    print(f"[scan] found {len(episode_dirs)} episode dirs under {args.raw_dir}", flush=True)
    for episode_dir in tqdm(
        episode_dirs,
        desc="Converting episodes",
        unit="episode",
        dynamic_ncols=True,
        file=sys.stdout,
    ):
        try:
            converted = write_episode(dataset, episode_dir, args.task, args.frame_mismatch_policy)
        except SkipEpisodeError as exc:
            print(f"[skip] {exc}", flush=True)
            skipped_episodes.append(
                {
                    "episode_index": episode_index_from_path(episode_dir),
                    "episode": episode_dir.name,
                    "reason": str(exc),
                }
            )
            continue
        total_frames += int(converted["num_frames"])
        converted_episodes.append(converted)

    if not converted_episodes:
        raise RuntimeError(f"{args.raw_dir} 没有成功转换任何有效 episode")

    gripper_total = {"0": 0, "1": 0}
    for episode in converted_episodes:
        counts = episode["gripper_cmd_counts"]
        assert isinstance(counts, dict)
        for key, value in counts.items():
            gripper_total[key] = gripper_total.get(key, 0) + int(value)

    summary = {
        "raw_dir": str(args.raw_dir.resolve()),
        "repo_id": args.repo_id,
        "lerobot_home": str(output_path),
        "total_episodes": len(converted_episodes),
        "skipped_episodes_count": len(skipped_episodes),
        "total_frames": total_frames,
        "fps": args.fps,
        "task": args.task,
        "state_semantics": "joint_state[:,1:8]+gripper_cmd(1) for 9D input, or joint_state[:,:7]+gripper_cmd(1)",
        "action_semantics": "next_joint_state_qpos(7)+next_gripper_cmd(1)",
        "gripper_command_source": "gripper_width -> binary command (0=open, 1=close), per-episode midpoint threshold",
        "image_source": "agent_view.mp4 + left_wrist.mp4",
        "image_shapes": image_shapes,
        "gripper_cmd_total_counts": gripper_total,
        "converted_episodes": converted_episodes,
        "skipped_episodes": skipped_episodes,
    }
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "conversion_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
