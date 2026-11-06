#!/usr/bin/env python3
"""
将 MimicGen / robomimic HDF5 数据集转换为 LeRobot v2.1 格式，
使用“下一帧绝对关节角(7) + 夹爪指令(1)”作为动作。

状态：当前关节角(7) + 当前夹爪 qpos(2)
动作：下一帧关节角(7) + 下一帧夹爪 cmd(1)（来自原始 actions 最后一维）
图片：agentview / wrist
python mymimicgen/3_convert.py  \
   --source /workspace/liushuo/cupid/mimicgen/tmp/core_datasets/square/demo_src_square_task_D1/demo.hdf5  \
   --output /workspace/liushuo/cupid/datasets/square_d1_50 \
   --repo-id ls/square_d1_50
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from tqdm import tqdm


FEATURES = {
    "observation.images.agentview": {
        "dtype": "video",
        "shape": (3, 84, 84),
        "names": ["c", "h", "w"],
    },
    "observation.images.wrist": {
        "dtype": "video",
        "shape": (3, 84, 84),
        "names": ["c", "h", "w"],
    },
    "observation.state": {
        "dtype": "float32",
        "shape": (9,),  # 7 joints + gripper qpos
        "names": [
            "joint_0",
            "joint_1",
            "joint_2",
            "joint_3",
            "joint_4",
            "joint_5",
            "joint_6",
            "gripper_qpos_l",
            "gripper_qpos_r",
        ],
    },
    "action": {
        "dtype": "float32",
        "shape": (8,),  # next joint pos(7) + next gripper cmd(1)
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
}


def convert_dataset(source: Path, output_dir: Path, repo_id: str, fps: int = 20) -> None:
    source = Path(source)
    output_dir = Path(output_dir)

    with h5py.File(source, "r") as f:
        demos = sorted(f["data"].keys())
        print(f"Found {len(demos)} demos in {source}")

        # Update image shapes from first demo
        demo0 = f["data"][demos[0]]
        h, w, c = demo0["obs"]["agentview_image"].shape[1:]
        FEATURES["observation.images.agentview"]["shape"] = (c, h, w)
        FEATURES["observation.images.wrist"]["shape"] = (c, h, w)

        dataset = LeRobotDataset.create(
            repo_id=repo_id,
            fps=fps,
            features=FEATURES,
            root=output_dir,
            use_videos=True,
        )

        for key in tqdm(demos, desc="Converting demos"):
            grp = f["data"][key]
            obs = grp["obs"]

            agentview = obs["agentview_image"][:]  # (T, H, W, C)
            wrist = obs["robot0_eye_in_hand_image"][:]
            joint_pos = obs["robot0_joint_pos"][:]  # (T, 7)
            gripper_qpos = obs["robot0_gripper_qpos"][:]  # (T, 2) -> use mean as scalar

            actions_raw = grp["actions"][:]  # (T, 7) last dim is gripper cmd
            gripper_cmd = actions_raw[:, -1:]  # (T,1)

            T_len = joint_pos.shape[0]
            # Next-frame targets; repeat last for padding
            target_joint = np.concatenate([joint_pos[1:], joint_pos[-1:]], axis=0)
            target_gripper_cmd = np.concatenate([gripper_cmd[1:], gripper_cmd[-1:]], axis=0)

            abs_actions = np.concatenate([target_joint, target_gripper_cmd], axis=1).astype(np.float32)
            state_vec = np.concatenate(
                [joint_pos, gripper_qpos], axis=1
            ).astype(np.float32)

            for t in range(T_len):
                frame = {
                    "observation.images.agentview": torch.from_numpy(agentview[t]).permute(2, 0, 1),
                    "observation.images.wrist": torch.from_numpy(wrist[t]).permute(2, 0, 1),
                    "observation.state": torch.from_numpy(state_vec[t]),
                    "action": torch.from_numpy(abs_actions[t]),
                }
                dataset.add_frame(frame, task=str(Path(source).stem))
            dataset.save_episode()

        print(f"Dataset saved to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path, help="Path to MimicGen HDF5.")
    parser.add_argument("--output", required=True, type=Path, help="Output directory for LeRobot dataset.")
    parser.add_argument("--repo-id", required=True, help="repo_id for LeRobotDataset.")
    parser.add_argument("--fps", type=int, default=20)
    args = parser.parse_args()
    convert_dataset(args.source, args.output, args.repo_id, fps=args.fps)


if __name__ == "__main__":
    main()
