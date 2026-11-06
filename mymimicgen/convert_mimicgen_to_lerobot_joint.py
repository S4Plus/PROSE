
"""
Convert MimicGen (Robosuite) HDF5 dataset to LeRobot format.
Extracts absolute EEF pose actions from observations.
"""
import h5py
import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from pathlib import Path
import os
import argparse
from tqdm import tqdm

# Avoid numba cache errors when importing robosuite utilities
os.environ.setdefault("NUMBA_CACHE_DIR", str(Path.home() / ".cache" / "numba"))
os.environ.setdefault("NUMBA_DISABLE_CACHING", "1")
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
import robosuite.utils.transform_utils as T

def convert_dataset(source_path, repo_id, output_dir, fps=20):
    source_path = Path(source_path)
    output_dir = Path(output_dir)
    
    # 1. Inspect source
    with h5py.File(source_path, 'r') as f:
        demos = list(f['data'].keys())
        num_demos = len(demos)
        print(f"Found {num_demos} demos in {source_path}")
        
        # 2. Prepare LeRobot Dataset Structure
        # We need to define features.
        # Images: agentview_image, robot0_eye_in_hand_image
        # State: robot0_joint_pos + gripper_qpos
        # Action: next-frame robot0_joint_pos + gripper_qpos (absolute joint angles)
        
        # Note: MimicGen images are usually (H, W, C), LeRobot/HF expects channels first usually? 
        # Actually LeRobotDataset handles conversion if we pass tensors or formatted arrays.
        # But we need to check if they are encoded. Robosuite usually stores raw uncompressed or sometimes compressed.
        
        features = {
            "observation.images.agentview": {
                "dtype": "video",
                "shape": (3, 84, 84), # Assumes 84x84, will check
                "names": ["c", "h", "w"]
            },
            "observation.images.wrist": {
                "dtype": "video",
                "shape": (3, 84, 84),
                "names": ["c", "h", "w"]
            },
            "observation.state": {
                "dtype": "float64",
                "shape": (9,), # 7 joints + gripper qpos (2)
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
                "shape": (9,), # 7 joint angles + gripper qpos (2)
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
                ]
            }
        }
        
        # Check image shape
        demo0 = f['data'][demos[0]]
        img_shape = demo0['obs']['agentview_image'].shape[1:] # e.g. (84, 84, 3)
        H, W, C = img_shape
        features["observation.images.agentview"]["shape"] = (C, H, W)
        features["observation.images.wrist"]["shape"] = (C, H, W) # Assuming same for eye_in_hand

        # Create dataset
        dataset = LeRobotDataset.create(
            repo_id=repo_id,
            fps=fps,
            root=output_dir,
            features=features,
            use_videos=True # Encode videos
        )

        for demo_key in tqdm(demos):
            demo_grp = f['data'][demo_key]
            obs_grp = demo_grp['obs']
            
            # Read data
            # Images need to be permuted to (T, C, H, W) for LeRobot? 
            # LeRobot add_frame expects dict of data.
            
            agentview = obs_grp['agentview_image'][:] # (T, H, W, C)
            wristview = obs_grp['robot0_eye_in_hand_image'][:]
            gripper_qpos = obs_grp['robot0_gripper_qpos'][:]  # (T, 2)
            joint_pos = obs_grp["robot0_joint_pos"][:]  # (T, 7) for Panda
            
            # Use next-frame absolute joint positions + gripper qpos as the action target.
            # For the last frame, repeat the final pose/qpos to keep lengths aligned.
            target_joint = np.concatenate([joint_pos[1:], joint_pos[-1:]], axis=0)
            target_gripper = np.concatenate([gripper_qpos[1:], gripper_qpos[-1:]], axis=0)  # (T, 2)

            abs_actions = np.concatenate([target_joint, target_gripper], axis=1).astype(np.float32)
            
            # State vector
            state_vec = np.concatenate([joint_pos, gripper_qpos], axis=1)
            
            # Add frames
            for t in range(len(agentview)):
                frame = {
                    "observation.images.agentview": torch.from_numpy(agentview[t]).permute(2, 0, 1), # HWC -> CHW
                    "observation.images.wrist": torch.from_numpy(wristview[t]).permute(2, 0, 1),
                    "observation.state": torch.from_numpy(state_vec[t]),
                    "action": torch.from_numpy(abs_actions[t])
                }
                dataset.add_frame(frame, task="square_d1_abs")
            
            dataset.save_episode()

        print(f"Dataset saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="/workspace/liushuo/cupid/mimicgen/datasets/core/square_d1.hdf5")
    parser.add_argument("--output", type=str, default="/workspace/liushuo/cupid/datasets/square_d1_abs_1")
    parser.add_argument("--repo-id", type=str, default="ls/square_d1_abs_1")
    args = parser.parse_args()
    
    convert_dataset(args.source, args.repo_id, args.output)



#  python mymimicgen/convert_mimicgen_to_lerobot_abs.py \
#   --source /workspace/liushuo/cupid/mimicgen/tmp/core_datasets/square/demo_src_square_task_D1/demo.hdf5 \
#   --output /workspace/liushuo/cupid/datasets/square_d1_abs_1_0.3_0.2 \
#   --repo-id ls/square_d1_abs_0.3_0.2



#绝对关节角度训练act
# python mymimicgen/convert_mimicgen_to_lerobot_abs.py \
#     --source /workspace/liushuo/cupid/mimicgen/tmp/core_datasets/square/demo_src_square_task_D1/demo.hdf5 \
#     --output /workspace/liushuo/cupid/datasets/square_d1_joint \
#     --repo-id ls/square_d1_joint