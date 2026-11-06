#!/usr/bin/env python3
"""
LeRobot Dataset Converter
Converts HDF5 episode files to LeRobot dataset format for Hugging Face Hub.

Usage:
    python convert_lerobot.py --config config.yaml
"""

import h5py
import numpy as np
import cv2
import yaml
from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from tqdm import tqdm
import shutil
import os
import argparse
import logging
from typing import Dict, Any, List

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LeRobotDatasetConverter:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.repo_id = config["repo_id"]
        self.fps = config.get("fps", 30)
        self.raw_data_dir = Path(config["raw_data_dir"])
        self.output_dir = Path(config.get("output_dir", os.path.expanduser("~/.cache/huggingface/lerobot"))) / self.repo_id
        self.batch_size = config.get("batch_size", 1)
        
        self.features = {
            "observation.images.camera_high": {
                "dtype": "video",
                "shape": (360, 640, 3),  # Updated to 360x640 (height x width x channels)
                "names": ["height", "width", "rgb"],
            },
            "observation.images.camera_wrist_right": {
                "dtype": "video",
                "shape": (360, 640, 3),  # Updated to 360x640 (height x width x channels)
                "names": ["height", "width", "rgb"],
            },
            "observation.state": {
                "dtype": "float32",
                "shape": (7,),
                "names": {"motors": ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "gripper"]},
            },
            "action": {
                "dtype": "float32",
                "shape": (7,),
                "names": {"motors": ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "gripper"]},
            },
        }

    def _validate_batch_size(self, num_episodes: int):
        """Check if episode count is divisible by batch size"""
        if num_episodes % self.batch_size != 0:
            raise ValueError(
                f"Cannot divide {num_episodes} episodes into batches of {self.batch_size}. "
                f"Please adjust batch_size in config or number of episodes."
            )
        logger.info(f"Using batch size: {self.batch_size}")

    def create_dataset(self) -> LeRobotDataset:
        """Create dataset with configured batch size"""
        if self.output_dir.exists():
            logger.warning(f"Clearing existing output directory...")
            shutil.rmtree(self.output_dir)
            
        return LeRobotDataset.create(
            repo_id=self.repo_id,
            fps=self.fps,
            features=self.features,
            use_videos=True,
            image_writer_processes=4,
            image_writer_threads=16,
            batch_encoding_size=self.batch_size,
        )

    def process_episode(self, hdf5_path: Path, dataset: LeRobotDataset) -> bool:
        """Process single episode with corrected feature names and data types"""
        try:
            with h5py.File(hdf5_path, "r") as hdf5_file:
                task = "pick bottle"
                qpos = hdf5_file["observations/qpos"][:].astype(np.float32)

                actions = np.empty_like(qpos)
                actions[:-1] = qpos[1:]
                actions[-1] = qpos[-1]

                image_data = {
                    "camera_high": hdf5_file["observations/images/head_camera"][:],
                    "camera_wrist_right": hdf5_file["observations/images/right_wrist_camera"][:]
                }

                # Debugging: Check shape and dtype
                for cam_name, img_data in image_data.items():
                    logger.info(f"{cam_name} shape: {img_data.shape}, dtype: {img_data.dtype}")
                    if img_data.ndim != 4 or img_data.shape[-1] != 3 or img_data.dtype != np.uint8:
                        logger.warning(f"Unexpected format for {cam_name}: Expected (frames, h, w, 3) uint8")

                decoded_images = {}
                for cam_name, img_data in image_data.items():
                    decoded_images[cam_name] = [
                        cv2.resize(cv2.cvtColor(img_bytes, cv2.COLOR_BGR2RGB), (640, 360))  # Resize to 640x360 (width x height)
                        for img_bytes in img_data
                    ]

                min_length = min([len(qpos), len(actions)] + [len(imgs) for imgs in decoded_images.values()])

                for i in range(min_length):
                    dataset.add_frame(
                        {
                            **{f"observation.images.{cam}": decoded_images[cam][i] for cam in decoded_images},
                            "observation.state": qpos[i],
                            "action": actions[i].astype(np.float32),
                        },
                        task=task
                    )

                dataset.save_episode()
                return True

        except Exception as e:
            logger.error(f"Error processing {hdf5_path.name}: {str(e)}")
            return False

    def run_conversion(self):
        """Run conversion with batch size validation"""
        episode_files = sorted(
            self.raw_data_dir.glob("episode_*.hdf5"),
            key=lambda x: int(x.stem.split("_")[1])
        )
        
        if not episode_files:
            raise FileNotFoundError(f"No episode files found in {self.raw_data_dir}")

        self._validate_batch_size(len(episode_files))

        dataset = self.create_dataset()
        success_count = 0

        for hdf5_file in tqdm(episode_files, desc="Processing episodes"):
            if self.process_episode(hdf5_file, dataset):
                success_count += 1

        logger.info(f"Successfully processed {success_count}/{len(episode_files)} episodes")

def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    try:
        config = load_config(args.config)
        converter = LeRobotDatasetConverter(config)
        converter.run_conversion()
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        raise

if __name__ == "__main__":
    main()