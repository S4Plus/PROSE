from datasets import Dataset, concatenate_datasets
from lerobot.datasets.lerobot_dataset import LeRobotDataset,LeRobotDatasetMetadata
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

    episodes_stats = meta_info.episodes_stats
    from lerobot.datasets.utils import write_episode_stats
    fields_to_compute = ["action", "observation.state"]  # 可以按需扩展

    for episode_idx in tqdm(range(num_episode), desc="Processing Episodes"):
        lerobot_data = LeRobotDataset(repo_id, episodes=[episode_idx])
        hf_dataset = lerobot_data.hf_dataset

        for field in fields_to_compute:
            data = np.array(hf_dataset[field])  # 转为 NumPy 数组
            stats = {
                "min": data.min(axis=0),
                "max": data.max(axis=0),
                "mean": data.mean(axis=0),
                "std": data.std(axis=0)
            }
            for _key in stats.keys():
                episodes_stats[episode_idx][field][_key] = stats[_key]

        ep_data_path = "episodes_stats.jsonl"
        write_episode_stats(
            episode_index=episode_idx,
            episode_stats=episodes_stats[episode_idx],
            local_dir=output_root
        )
        print(f"✅ Episode {episode_idx} saved to {ep_data_path}")

    print("\n======== Processing Complete ========")
    print(f"All modified parquet files are saved in '{output_dir}'.")


if __name__ == "__main__":
    debug()
    repo_id = "llly/usbinsert_v2_10_25_joint"
    output_dir = "modified_dataset"
    modify_and_save_dataset(repo_id, output_dir)
