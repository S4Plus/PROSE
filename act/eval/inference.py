#!/usr/bin/env python3
from __future__ import annotations

"""
Lightweight ROS2 inference node for ACT checkpoints.

Subscribes to one or multiple camera feeds plus (optionally) joint states,
runs the policy, and publishes an action chunk as a `Float32MultiArray`.
python inference_act.py --checkpoint <path>
"""

import argparse
import time
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np
import rclpy
import torch
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, JointState
from std_msgs.msg import Float32MultiArray

from lerobot.policies.act.modeling_act import ACTPolicy


def preprocess_image(
    img_msg: CompressedImage, width: int, height: int, device: torch.device
) -> torch.Tensor:
    """Decode a compressed ROS image message into a normalized CHW tensor."""
    np_arr = np.frombuffer(img_msg.data, np.uint8)
    bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Failed to decode compressed image.")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (width, height))
    tensor = torch.from_numpy(resized).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    return tensor.to(device)


def preprocess_state(msg: JointState, device: torch.device) -> torch.Tensor:
    """Convert JointState positions to a batched tensor."""
    if not msg.position:
        raise ValueError("JointState message has no positions.")
    positions = np.asarray(msg.position, dtype=np.float32)
    return torch.from_numpy(positions).unsqueeze(0).to(device)


def parse_camera_topic_args(entries: list[str] | None) -> Dict[str, str]:
    """Parse `--camera-topic key:=topic` CLI arguments into a dictionary."""
    topics: Dict[str, str] = {}
    if not entries:
        return topics

    for raw in entries:
        if "=" not in raw:
            raise ValueError(
                f"Camera topic '{raw}' is invalid. Expected format 'observation.images.front:=/topic'."
            )
        key, topic = raw.split("=", 1)
        key = key.strip()
        topic = topic.strip()
        if not key or not topic:
            raise ValueError(f"Camera topic '{raw}' is invalid. Empty key/topic found.")
        topics[key] = topic
    return topics


class ACTPolicyNode(Node):
    def __init__(self, args: argparse.Namespace):
        super().__init__("ros_act_inference")

        self.device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.target_width = int(args.image_width)
        self.target_height = int(args.image_height)
        self.dt = float(args.dt)

        self.policy = self._load_policy(Path(args.checkpoint).expanduser())
        self.n_obs_steps = self.policy.config.n_obs_steps
        self.chunk_size = args.chunk_size or self.policy.config.n_action_steps
        self.image_keys = list(self.policy.config.image_features.keys())
        self.expect_state = self.policy.config.robot_state_feature is not None

        self.latest_state: Optional[JointState] = None
        self.latest_images: Dict[str, Optional[CompressedImage]] = {key: None for key in self.image_keys}

        self.camera_topics = self._resolve_camera_topics(args)
        for key, topic in self.camera_topics.items():
            self.create_subscription(
                CompressedImage,
                topic,
                self._make_image_callback(key),
                10,
            )

        if self.expect_state:
            if not args.joint_topic:
                raise ValueError("Policy expects 'observation.state' but no --joint-topic was provided.")
            self.create_subscription(JointState, args.joint_topic, self._joint_callback, 10)

        self.action_pub = self.create_publisher(Float32MultiArray, args.action_topic, 10)

        self.get_logger().info(
            f"Policy loaded on {self.device} "
            f"(obs_steps={self.n_obs_steps}, action_chunk={self.chunk_size}, "
            f"cameras={','.join(self.image_keys) or 'none'})."
        )

    def _load_policy(self, checkpoint: Path) -> ACTPolicy:
        if checkpoint.is_file():
            checkpoint = checkpoint.parent
        policy = ACTPolicy.from_pretrained(str(checkpoint))
        policy.to(self.device)
        policy.eval()
        policy.reset()
        return policy

    def _resolve_camera_topics(self, args: argparse.Namespace) -> Dict[str, str]:
        if not self.image_keys:
            return {}
        topics = parse_camera_topic_args(args.camera_topic)

        heuristic_topics = {
            "observation.images.head_camera": args.head_topic,
            "observation.images.front": args.head_topic,
            "observation.images.right_wrist_camera": args.wrist_topic,
            "observation.images.left_wrist_camera": args.wrist_topic,
            "observation.images.wrist": args.wrist_topic,
        }
        for key in self.image_keys:
            if key in topics:
                continue
            fallback = heuristic_topics.get(key)
            if fallback:
                topics[key] = fallback

        missing = [key for key in self.image_keys if key not in topics or topics[key] is None]
        if missing:
            raise ValueError(
                f"No ROS topic configured for image inputs {missing}. "
                "Either pass --head-topic/--wrist-topic when applicable or provide "
                "--camera-topic KEY:=TOPIC for each policy image input."
            )
        return topics

    def _make_image_callback(self, key: str):
        def _callback(msg: CompressedImage):
            self.latest_images[key] = msg

        return _callback

    def _joint_callback(self, msg: JointState):
        self.latest_state = msg

    def _clear_latest(self):
        for key in self.latest_images:
            self.latest_images[key] = None
        self.latest_state = None

    def try_publish_action(self):
        if self.expect_state and self.latest_state is None:
            return

        for key in self.image_keys:
            if self.latest_images[key] is None:
                return

        try:
            img_tensors = {
                key: preprocess_image(msg, self.target_width, self.target_height, self.device)
                for key, msg in self.latest_images.items()
                if msg is not None
            }
            state_vec = (
                preprocess_state(self.latest_state, self.device) if self.expect_state and self.latest_state else None
            )
        except ValueError as exc:
            self.get_logger().warning(f"Preprocess failed: {exc}")
            self._clear_latest()
            return

        batch = {}
        if state_vec is not None:
            batch["observation.state"] = state_vec
        batch.update(img_tensors)

        self._clear_latest()

        with torch.no_grad():
            action_chunk = self.policy.predict_action_chunk(batch).squeeze(0)
            action_chunk = action_chunk[: self.chunk_size]

        msg = Float32MultiArray()
        msg.data = action_chunk.detach().cpu().float().reshape(-1).tolist()
        self.action_pub.publish(msg)
        self.get_logger().info(f"Published {action_chunk.shape} actions.")

    def spin_once(self):
        rclpy.spin_once(self, timeout_sec=0.01)
        time.sleep(self.dt)
        self.try_publish_action()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ROS2 ACTPolicy inference node.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="outputs/train/act_policy/checkpoints/last/pretrained_model",
        help="Path to the pretrained ACT directory or safetensors file.",
    )
    parser.add_argument("--device", default=None, help="Torch device string (default: auto).")
    parser.add_argument("--chunk-size", type=int, default=None, help="Number of actions to publish per message.")
    parser.add_argument(
        "--camera-topic",
        action="append",
        default=[],
        help=(
            "Mapping from policy image key to ROS topic, e.g. "
            "--camera-topic observation.images.front=/camera/front/image_raw. "
            "Repeat for each camera input."
        ),
    )
    parser.add_argument("--head-topic", default="/camera/head/image_rgb", help="Fallback topic for head/front images.")
    parser.add_argument(
        "--wrist-topic",
        default="/camera/wrist/image_rgb",
        help="Fallback topic for wrist images when --camera-topic is not provided.",
    )
    parser.add_argument("--joint-topic", default="/pi0_eval/joint_states", help="Joint state topic name.")
    parser.add_argument("--action-topic", default="/joint_action", help="Action chunk output topic.")
    parser.add_argument("--image-width", type=int, default=640, help="Resized image width.")
    parser.add_argument("--image-height", type=int, default=360, help="Resized image height.")
    parser.add_argument("--dt", type=float, default=0.02, help="Sleep interval between inference loops.")
    return parser.parse_args()


def main():
    args = parse_args()
    rclpy.init()
    node = ACTPolicyNode(args)
    try:
        while rclpy.ok():
            node.spin_once()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()


if __name__ == "__main__":
    main()
