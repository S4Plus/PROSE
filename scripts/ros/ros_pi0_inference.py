
# Allow execution from any working directory without installing this checkout.
import sys as _prose_sys
from pathlib import Path as _ProsePath
_prose_sys.path.insert(0, str(_ProsePath(__file__).resolve().parents[2]))

# ros_pi0_inference.py inference是模型推理，接收图像和关节信息，发送推理结果

import time
import torch
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, CompressedImage
from std_msgs.msg import Float32MultiArray, String
import cv2
import numpy as np
from cv_bridge import CvBridge
import os, sys

# --- OpenPI imports ---
from openpi.training import config as _config
from openpi.policies import policy_config

device = "cuda" if torch.cuda.is_available() else "cpu"

def preprocess_image(img_msg, IMG_W=640, IMG_H=360):
    """Decode CompressedImage -> numpy -> (C,H,W) float32 normalized"""
    np_arr = np.frombuffer(img_msg.data, np.uint8)
    cv_img = cv2.resize(cv2.cvtColor(cv2.imdecode(np_arr, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB), (IMG_W, IMG_H))
    img_array = np.transpose(cv_img, (2, 0, 1)).astype(np.float32) / 255.0
    return img_array

class Pi0Listener(Node):
    def __init__(self, topics: dict, control_topic: str, model_path: str):
        super().__init__('pi0_inference_listener')
        self.latest_msgs = {name: None for name in topics.keys()}
        self.subscribers = []
        self.model_path = model_path
        self.policy = None
        self.load_policy()

        for key, info in topics.items():
            sub = self.create_subscription(
                info["msg_type"], info["topic"], self._make_callback(key), 10
            )
            self.subscribers.append(sub)

        self.control_sub = self.create_subscription(
            String, control_topic, self._control_callback, 10
        )

    def load_policy(self):
        self.get_logger().info(f"Loading Pi0 policy from: {self.model_path}...")

        # ✅ 释放旧模型
        if self.policy is not None:
            del self.policy
            self.policy = None
            if device == "cuda":
                torch.cuda.empty_cache()

        try:
            config = _config.get_config("pi0_belt_pick_bottle_fin")
            self.policy = policy_config.create_trained_policy(config, self.model_path)
            self.get_logger().info("Policy loaded successfully.")
        except Exception as e:
            self.get_logger().error(f"Failed to load policy: {e}")
            self.policy = None

    def _make_callback(self, topic_key):
        def callback(msg):
            self.latest_msgs[topic_key] = msg
        return callback

    def clear_all_topics(self):
        """Sets all latest message caches to None."""
        self.get_logger().debug('Clearing all cached observations.')
        for key in self.latest_msgs:
            self.latest_msgs[key] = None

    def _control_callback(self, msg):
        if msg.data == 'RESET':
            self.get_logger().info('RESET signal received.')
            self.clear_all_topics()
            self.load_policy()

    def get_latest(self):
        return dict(self.latest_msgs)

def main():
    rclpy.init()

    topics = {
        "head_images": {"topic": "/camera/head/image_rgb", "msg_type": CompressedImage},
        "wrist_images": {"topic": "/camera/wrist/image_rgb", "msg_type": CompressedImage},
        "robot_joints": {"topic": "/pi0_eval/joint_states", "msg_type": JointState},
    }
    control_topic_name = "/inference_control"
    checkpoint_dir = "/ext_workspace/liushuo/checkpoints/pi05_static_pick_bottle_finetune/static_pick_bottle/10000/"

    listener = Pi0Listener(topics, control_topic=control_topic_name, model_path=checkpoint_dir)
    bridge = CvBridge()
    action_pub = listener.create_publisher(Float32MultiArray, '/joint_action', 10)

    # 控制频率
    dt = 0.02  # 20Hz，每个动作间隔 50ms
    IMG_H = 360
    IMG_W = 640

    try:
        while rclpy.ok():
            rclpy.spin_once(listener, timeout_sec=0.01)

            if listener.policy is None:
                listener.get_logger().warn("Policy is not available, waiting...", throttle_duration_sec=2)
                time.sleep(1)
                continue

            latest = listener.get_latest()
            joint_msg = latest["robot_joints"]
            head_img_msg = latest["head_images"]
            wrist_img_msg = latest["wrist_images"]

            if joint_msg is None or head_img_msg is None :
                print("Waiting for new synchronized data...", end='\r')
                time.sleep(dt)
                continue

            # --- Build observation ---
            qpos = np.array(joint_msg.position, dtype=np.float32)
            print("recived qpos", qpos)
            img_head = preprocess_image(head_img_msg, IMG_W, IMG_H)
            img_wrist = preprocess_image(wrist_img_msg, IMG_W, IMG_H)

            observation = {
                "observation/image": img_head,
                "observation/wrist_image": img_wrist,
                "observation/state": qpos,
            }

            # --- Pi0 inference ---
            result = listener.policy.infer(observation)
            action_chunk = result["actions"]  # shape: (50, action_dim)

            listener.get_logger().info(f"Publishing {len(action_chunk)} actions from chunk.")

            # 发布 50 个动作
          
            print(action_chunk)
            msg = Float32MultiArray()
            msg.data = action_chunk.flatten().tolist()
            action_pub.publish(msg)
            time.sleep(dt)

            # 用完清空缓存，等新数据
            listener.clear_all_topics()

    except KeyboardInterrupt:
        pass
    finally:
        listener.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
