#!/usr/bin/env python3
"""将原始 LIBERO HDF5 demos 转成可供默认 OpenPI LIBERO 配置训练的 LeRobot 数据。

目标语义严格对齐默认 `pi05_libero`：
- 图像: `image` / `wrist_image`
- 状态: `ee_states(6) + gripper_states(2)` -> 8 维
- 动作: 原始 HDF5 `actions` -> 7 维
- 文本: 从文件名恢复 task 文本，并作为 `task` 字段写入

典型用法：
    cd /workspace/liushuo/cupid/openpi
    uv run python /workspace/liushuo/cupid/act/libero/convert_libero_hdf5_to_openpi_lerobot.py \
        --raw-dir /workspace/liushuo/mimiclabs/data/datasets/libero_object \
        --repo-id local/libero_object_openpi_default \
        --lerobot-home /workspace/liushuo/cupid/datasets/openpi_lerobot \
        --overwrite
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="将原始 LIBERO HDF5 转成默认 OpenPI LIBERO 训练所需的 LeRobot 数据")
    parser.add_argument("--raw-dir", type=Path, required=True, help="原始 LIBERO HDF5 所在目录")
    parser.add_argument("--repo-id", type=str, required=True, help="输出 LeRobot repo_id，例如 local/libero_object_openpi_default")
    parser.add_argument("--lerobot-home", type=Path, required=True, help="输出 LeRobot 根目录，会作为 HF_LEROBOT_HOME")
    parser.add_argument("--fps", type=int, default=20, help="数据集帧率，LIBERO 通常为 20")
    parser.add_argument("--overwrite", action="store_true", help="若输出目录已存在，是否覆盖")
    parser.add_argument("--push-to-hub", action="store_true", help="转换完成后是否推送到 Hugging Face Hub")
    return parser.parse_args()


def normalize_task_from_filename(path: Path) -> str:
    """从 `pick_up_xxx_demo.hdf5` 恢复自然语言 task。"""
    stem = path.stem
    if stem.endswith("_demo"):
        stem = stem[: -len("_demo")]
    return stem.replace("_", " ")


def iter_hdf5_files(raw_dir: Path) -> Iterable[Path]:
    for path in sorted(raw_dir.glob("*.hdf5")):
        if path.is_file():
            yield path


def build_state(obs_group: h5py.Group) -> np.ndarray:
    """默认 OpenPI LIBERO 使用 ee_states(6) + gripper_states(2) -> 8 维 state。"""
    ee_states = np.asarray(obs_group["ee_states"], dtype=np.float32)
    gripper_states = np.asarray(obs_group["gripper_states"], dtype=np.float32)
    if ee_states.ndim != 2 or ee_states.shape[1] != 6:
        raise ValueError(f"ee_states 维度异常: {ee_states.shape}")
    if gripper_states.ndim != 2 or gripper_states.shape[1] != 2:
        raise ValueError(f"gripper_states 维度异常: {gripper_states.shape}")
    return np.concatenate([ee_states, gripper_states], axis=-1)


def main() -> None:
    args = parse_args()

    if not args.raw_dir.exists():
        raise FileNotFoundError(f"原始数据目录不存在: {args.raw_dir}")

    hdf5_files = list(iter_hdf5_files(args.raw_dir))
    if not hdf5_files:
        raise FileNotFoundError(f"未在 {args.raw_dir} 下找到任何 .hdf5 文件")

    os.environ["HF_LEROBOT_HOME"] = str(args.lerobot_home.resolve())

    from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME  # type: ignore
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset  # type: ignore

    output_path = HF_LEROBOT_HOME / args.repo_id
    if output_path.exists():
        if not args.overwrite:
            raise FileExistsError(f"输出目录已存在: {output_path}，如需覆盖请加 --overwrite")
        shutil.rmtree(output_path)

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        robot_type="panda",
        fps=args.fps,
        features={
            "image": {
                "dtype": "image",
                "shape": (128, 128, 3),
                "names": ["height", "width", "channel"],
            },
            "wrist_image": {
                "dtype": "image",
                "shape": (128, 128, 3),
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (8,),
                "names": ["state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["actions"],
            },
        },
        image_writer_threads=8,
        image_writer_processes=4,
    )

    tasks = []
    total_episodes = 0
    total_frames = 0

    for task_index, hdf5_path in enumerate(hdf5_files):
        task_text = normalize_task_from_filename(hdf5_path)
        tasks.append({"task_index": task_index, "task": task_text})

        with h5py.File(hdf5_path, "r") as f:
            if "data" not in f:
                raise KeyError(f"{hdf5_path} 缺少顶层 data 组")

            demo_keys = sorted(f["data"].keys())
            for demo_key in demo_keys:
                demo = f["data"][demo_key]
                obs = demo["obs"]
                base_images = np.asarray(obs["agentview_rgb"], dtype=np.uint8)
                wrist_images = np.asarray(obs["eye_in_hand_rgb"], dtype=np.uint8)
                states = build_state(obs)
                actions = np.asarray(demo["actions"], dtype=np.float32)

                lengths = {base_images.shape[0], wrist_images.shape[0], states.shape[0], actions.shape[0]}
                if len(lengths) != 1:
                    raise ValueError(
                        f"{hdf5_path}:{demo_key} 长度不一致，"
                        f"image={base_images.shape[0]}, wrist={wrist_images.shape[0]}, "
                        f"state={states.shape[0]}, action={actions.shape[0]}"
                    )

                for t in range(actions.shape[0]):
                    dataset.add_frame(
                        {
                            "image": base_images[t],
                            "wrist_image": wrist_images[t],
                            "state": states[t],
                            "actions": actions[t],
                            "task": task_text,
                        }
                    )
                    total_frames += 1

                dataset.save_episode()
                total_episodes += 1

    meta_dir = output_path / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    (meta_dir / "tasks.jsonl").write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in tasks) + "\n",
        encoding="utf-8",
    )

    summary = {
        "raw_dir": str(args.raw_dir),
        "repo_id": args.repo_id,
        "output_path": str(output_path),
        "total_tasks": len(tasks),
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "fps": args.fps,
        "state_semantics": "ee_states(6)+gripper_states(2)",
        "action_semantics": "raw_hdf5_actions(7)",
    }
    (output_path / "conversion_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    if args.push_to_hub:
        dataset.push_to_hub(
            tags=["libero", "panda", "lerobot", "openpi"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
