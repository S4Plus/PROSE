#!/usr/bin/env python3
"""将单臂 Franka 的 stack_cube HDF5 episode 转成 LeRobot 数据集。

数据语义按当前 Franka 关节控制训练需求固定为：

- state: 当前关节角 qpos(7) + 当前夹爪开闭命令 gripper_cmd(1)
- action: 下一时刻关节角 next_qpos(7) + 下一时刻夹爪开闭命令 next_gripper_cmd(1)
- 图像:
  - observations/images/overhead -> observation.images.fixed_camera
  - observations/images/hand -> observation.images.wrist_camera
  - 保持原始尺寸，不做压缩或下采样

说明：
- 原始 HDF5 顶层 `action` 是末端位姿控制信号，这里不再作为训练 action 使用。
- `observations/eff_angular` 已验证与原始 `action[:, -1]` 完全一致，这里会离散成二值夹爪命令。
- 夹爪命令约定：`0=张开`，`1=闭合`。

典型用法：
   python /workspace/liushuo/cupid/act/franka/convert_stack_cube_hdf5_to_lerobot.py \
        --raw-dir /ext_workspace/liushuo/stack_cube/ \
        --repo-id local/stack_cube_joint_102\
        --lerobot-home /workspace/liushuo/dataset/stack_cube_102 \
        --overwrite
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import h5py
import numpy as np


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]
LEROBOT_SRC = PROJECT_ROOT / "lerobot_2.1" / "src"
GRIPPER_OPEN_CMD = np.float32(0.0)
GRIPPER_CLOSE_CMD = np.float32(1.0)


class SkipEpisodeError(RuntimeError):
    """用于跳过不满足转换条件的 episode。"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="将 stack_cube HDF5 episode 转成 LeRobot joint-control 数据集")
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("/workspace/liushuo/dataset/stack_cube"),
        help="原始 HDF5 所在目录",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="输出 LeRobot repo_id，例如 local/stack_cube_joint",
    )
    parser.add_argument(
        "--lerobot-home",
        type=Path,
        required=True,
        help="LeRobot 根目录，会作为 HF_LEROBOT_HOME",
    )
    parser.add_argument("--fps", type=int, default=30, help="数据帧率")
    parser.add_argument(
        "--task",
        type=str,
        default="stack cube",
        help="当 HDF5 中缺少根属性 task_prompt 时使用的回退 task 文本",
    )
    parser.add_argument("--robot-type", type=str, default="panda", help="LeRobot 元数据中的机器人类型")
    parser.add_argument("--overwrite", action="store_true", help="若输出目录已存在，是否覆盖")
    parser.add_argument("--max-episodes", type=int, default=None, help="仅转换前 N 条 episode，便于快速验证")
    parser.add_argument(
        "--image-writer-processes",
        type=int,
        default=4,
        help="LeRobot 图像编码进程数",
    )
    parser.add_argument(
        "--image-writer-threads",
        type=int,
        default=8,
        help="LeRobot 图像编码线程数",
    )
    return parser.parse_args()


def ensure_lerobot_importable() -> None:
    if not LEROBOT_SRC.is_dir():
        raise FileNotFoundError(f"未找到 lerobot 源码目录: {LEROBOT_SRC}")
    if str(LEROBOT_SRC) not in sys.path:
        sys.path.insert(0, str(LEROBOT_SRC))


def episode_index_from_path(path: Path) -> int:
    stem = path.stem
    if not stem.startswith("episode_"):
        raise ValueError(f"非法文件名，期望 episode_<idx>.hdf5: {path.name}")
    return int(stem.split("_", maxsplit=1)[1])


def iter_episode_files(raw_dir: Path) -> list[Path]:
    files = [path for path in raw_dir.glob("episode_*.hdf5") if path.is_file()]
    return sorted(files, key=episode_index_from_path)


def build_state_action(
    qpos: np.ndarray,
    gripper_cmd: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """构造当前 state 与下一时刻 absolute joint action。"""
    if qpos.ndim != 2 or qpos.shape[1] != 7:
        raise ValueError(f"qpos 维度异常，期望 (T, 7)，实际 {qpos.shape}")
    if gripper_cmd.ndim != 1:
        raise ValueError(f"gripper_cmd 维度异常，期望 (T,)，实际 {gripper_cmd.shape}")
    if qpos.shape[0] != gripper_cmd.shape[0]:
        raise ValueError(
            f"qpos 与 gripper_cmd 长度不一致: qpos={qpos.shape[0]}, gripper={gripper_cmd.shape[0]}"
        )

    grip = gripper_cmd[:, None]
    state = np.concatenate([qpos, grip], axis=1).astype(np.float32, copy=False)

    next_qpos = np.concatenate([qpos[1:], qpos[-1:]], axis=0)
    next_grip = np.concatenate([grip[1:], grip[-1:]], axis=0)
    action = np.concatenate([next_qpos, next_grip], axis=1).astype(np.float32, copy=False)
    return state, action


def discretize_gripper_width(gripper_width: np.ndarray) -> np.ndarray:
    """将连续夹爪宽度离散成二值开闭命令。"""
    if gripper_width.ndim != 1:
        raise ValueError(f"gripper_width 维度异常，期望 (T,)，实际 {gripper_width.shape}")

    min_width = float(np.min(gripper_width))
    max_width = float(np.max(gripper_width))
    if not np.isfinite(min_width) or not np.isfinite(max_width):
        raise ValueError(f"gripper_width 含非有限值: min={min_width}, max={max_width}")

    # 小宽度视为闭合，大宽度视为张开；若整条轨迹几乎无变化，默认保留“张开”以避免伪造闭合标签。
    if max_width - min_width <= 1e-6:
        return np.full(gripper_width.shape, GRIPPER_OPEN_CMD, dtype=np.float32)

    threshold = np.float32((min_width + max_width) * 0.5)
    return np.where(
        gripper_width <= threshold,
        GRIPPER_CLOSE_CMD,
        GRIPPER_OPEN_CMD,
    ).astype(np.float32, copy=False)


def _decode_hdf5_attr(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def load_episode_payload(hdf5_path: Path, default_task: str) -> dict[str, object]:
    with h5py.File(hdf5_path, "r") as h5_file:
        if "observations" not in h5_file:
            raise KeyError(f"{hdf5_path} 缺少 observations 组")

        raw_task = h5_file.attrs.get("task_prompt")
        task = _decode_hdf5_attr(raw_task).strip() if raw_task is not None else str(default_task).strip()
        if not task:
            task = str(default_task).strip()

        obs = h5_file["observations"]
        qpos = np.asarray(obs["qpos"], dtype=np.float32)
        gripper_width = np.asarray(obs["eff_angular"], dtype=np.float32)
        gripper_cmd = discretize_gripper_width(gripper_width)

        if "images" not in obs:
            raise KeyError(f"{hdf5_path} 缺少 observations/images 组")
        images = obs["images"]
        if "hand" not in images:
            raise SkipEpisodeError(f"{hdf5_path.name} 缺少 observations/images/hand，已跳过")
        fixed_images = np.asarray(images["overhead"], dtype=np.uint8)
        wrist_images = np.asarray(images["hand"], dtype=np.uint8)

        if fixed_images.ndim != 4 or fixed_images.shape[-1] != 3:
            raise ValueError(f"fixed camera 图像维度非法: {fixed_images.shape}")
        if wrist_images.ndim != 4 or wrist_images.shape[-1] != 3:
            raise ValueError(f"wrist camera 图像维度非法: {wrist_images.shape}")

        state, action = build_state_action(qpos=qpos, gripper_cmd=gripper_cmd)

        lengths = {
            state.shape[0],
            action.shape[0],
            fixed_images.shape[0],
            wrist_images.shape[0],
        }
        if len(lengths) != 1:
            raise ValueError(
                f"{hdf5_path} 序列长度不一致: "
                f"state={state.shape[0]}, action={action.shape[0]}, "
                f"fixed={fixed_images.shape[0]}, wrist={wrist_images.shape[0]}"
            )

        return {
            "task": task,
            "state": state,
            "action": action,
            "fixed_images": fixed_images,
            "wrist_images": wrist_images,
        }


def main() -> None:
    args = parse_args()

    if not args.raw_dir.exists():
        raise FileNotFoundError(f"原始数据目录不存在: {args.raw_dir}")

    episode_files = iter_episode_files(args.raw_dir)
    if not episode_files:
        raise FileNotFoundError(f"未在 {args.raw_dir} 下找到任何 episode_*.hdf5")

    if args.max_episodes is not None:
        if args.max_episodes <= 0:
            raise ValueError("--max-episodes 必须为正整数")
        episode_files = episode_files[: args.max_episodes]

    sample_payload = None
    sample_episode_path = None
    skipped_episodes: list[dict[str, int | str]] = []
    for episode_path in episode_files:
        try:
            sample_payload = load_episode_payload(episode_path, default_task=args.task)
            sample_episode_path = episode_path
            break
        except SkipEpisodeError as exc:
            print(f"[skip] {exc}")
            skipped_episodes.append(
                {
                    "episode_index": episode_index_from_path(episode_path),
                    "file": episode_path.name,
                    "reason": str(exc),
                }
            )
    if sample_payload is None or sample_episode_path is None:
        raise RuntimeError("所有 episode 都缺少 wrist 相机，无法完成转换")

    fixed_shape = tuple(int(dim) for dim in sample_payload["fixed_images"].shape[1:])
    wrist_shape = tuple(int(dim) for dim in sample_payload["wrist_images"].shape[1:])
    state_dim = int(sample_payload["state"].shape[1])
    action_dim = int(sample_payload["action"].shape[1])

    os.environ["HF_LEROBOT_HOME"] = str(args.lerobot_home.resolve())
    ensure_lerobot_importable()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset  # type: ignore  # noqa: E402

    output_path = args.lerobot_home.resolve()
    if output_path.exists():
        if not args.overwrite:
            raise FileExistsError(f"输出目录已存在: {output_path}，如需覆盖请添加 --overwrite")
        shutil.rmtree(output_path)

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        root=args.lerobot_home.resolve(),
        robot_type=args.robot_type,
        fps=args.fps,
        features={
            "observation.images.fixed_camera": {
                "dtype": "video",
                "shape": fixed_shape,
                "names": ["height", "width", "channel"],
            },
            "observation.images.wrist_camera": {
                "dtype": "video",
                "shape": wrist_shape,
                "names": ["height", "width", "channel"],
            },
            "observation.state": {
                "dtype": "float32",
                "shape": (state_dim,),
                "names": [
                    "joint_0",
                    "joint_1",
                    "joint_2",
                    "joint_3",
                    "joint_4",
                    "joint_5",
                    "joint_6",
                    "gripper_cmd",
                ],
            },
            "action": {
                "dtype": "float32",
                "shape": (action_dim,),
                "names": [
                    "joint_0",
                    "joint_1",
                    "joint_2",
                    "joint_3",
                    "joint_4",
                    "joint_5",
                    "joint_6",
                    "gripper_cmd",
                ],
            },
        },
        use_videos=True,
        image_writer_processes=args.image_writer_processes,
        image_writer_threads=args.image_writer_threads,
    )

    total_frames = 0
    converted_episodes: list[dict[str, int | str]] = []
    episode_tasks: set[str] = set()

    for episode_path in episode_files:
        try:
            payload = load_episode_payload(episode_path, default_task=args.task)
        except SkipEpisodeError as exc:
            print(f"[skip] {exc}")
            skipped_episodes.append(
                {
                    "episode_index": episode_index_from_path(episode_path),
                    "file": episode_path.name,
                    "reason": str(exc),
                }
            )
            continue
        num_frames = int(payload["state"].shape[0])
        episode_task = str(payload["task"])
        episode_tasks.add(episode_task)
        for frame_idx in range(num_frames):
            dataset.add_frame(
                {
                    "observation.images.fixed_camera": payload["fixed_images"][frame_idx],
                    "observation.images.wrist_camera": payload["wrist_images"][frame_idx],
                    "observation.state": payload["state"][frame_idx],
                    "action": payload["action"][frame_idx],
                },
                task=episode_task,
            )

        dataset.save_episode()
        total_frames += num_frames
        converted_episodes.append(
            {
                "episode_index": episode_index_from_path(episode_path),
                "file": episode_path.name,
                "num_frames": num_frames,
                "task": episode_task,
            }
        )

    summary = {
        "raw_dir": str(args.raw_dir.resolve()),
        "repo_id": args.repo_id,
        "lerobot_home": str(args.lerobot_home.resolve()),
        "output_path": str(output_path),
        "sample_episode": sample_episode_path.name,
        "total_episodes": len(converted_episodes),
        "skipped_episodes_count": len(skipped_episodes),
        "total_frames": total_frames,
        "fps": args.fps,
        "task_fallback": args.task,
        "episode_tasks": sorted(episode_tasks),
        "robot_type": args.robot_type,
        "state_semantics": "qpos(7)+gripper_cmd(1)",
        "action_semantics": "next_qpos(7)+next_gripper_cmd(1)",
        "gripper_command_source": "observations/eff_angular -> binary command (0=open, 1=close)",
        "fixed_camera_source": "observations/images/overhead (keep original resolution)",
        "wrist_camera_source": "observations/images/hand",
        "converted_episodes": converted_episodes,
        "skipped_episodes": skipped_episodes,
    }

    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "conversion_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
