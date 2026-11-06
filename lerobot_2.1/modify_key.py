from datasets import Dataset, concatenate_datasets
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from tqdm import tqdm
from pathlib import Path
import torch
import numpy as np

def debug():
    import debugpy
    debugpy.listen(("0.0.0.0", 5678))
    print("✅ Waiting for debugger to attach on port 5678...")
    debugpy.wait_for_client()
    print("Start to debugging")

def modify_and_save_dataset(repo_id: str, output_dir: str):
    """
    读取一个 LeRobot dataset，逐个 episode 修改 state 和 action，并保存为 .parquet 文件。
    """
    print(f"Loading metadata from {repo_id}")
    meta_info = LeRobotDatasetMetadata(repo_id)
    num_episode = meta_info.total_episodes
    print(f"Total episodes found: {num_episode}")

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    for episode_idx in tqdm(range(num_episode), desc="Processing Episodes"):
        lerobot_data = LeRobotDataset(repo_id, episodes=[episode_idx])
        hf_dataset = lerobot_data.hf_dataset
        episode_len = len(hf_dataset)

        state_data = np.array(hf_dataset["observation.state"])
        action_data = np.array(hf_dataset["action"])

        # ==================================下面是对数据集修改的部分==================================
        joints = state_data[:, :6]
        gripper =  state_data[:, 6:7]
        latent = state_data[:, 7:]
        
        # 修改 State
        modified_state = np.concatenate([joints,gripper,latent], axis=1) # TODO: joint + gripper
        # 修改 Action
        modified_action = np.roll(modified_state,shift=-1, axis=0) # TODO: 用下一时刻的 joint 当 action
        modified_action[-1] = modified_state[-1]
        # ========================================修改结束======================================================

        data_dict = {}
        for key in hf_dataset.features.keys():
            if key == "observation.state":
                data_dict[key] = [s for s in modified_state]
            elif key == "action":
                data_dict[key] = [a for a in modified_action]
            else:
                # 其他字段保持不变
                data_dict[key] = hf_dataset[key]

        new_hf_data = Dataset.from_dict(data_dict)

        ep_data_path = output_root / f"episode_{episode_idx:06d}.parquet"
        ep_data_path.parent.mkdir(parents=True, exist_ok=True)
        new_hf_data.to_parquet(ep_data_path)

        print(f"✅ Episode {episode_idx} saved to {ep_data_path}")

    print("\n======== Processing Complete ========")
    print(f"All modified parquet files are saved in '{output_dir}'.")


if __name__ == "__main__":
    # debug()
    repo_id = "llly/usbinsert_v2_10_25"
    output_dir = "modified_dataset"
    modify_and_save_dataset(repo_id, output_dir)
