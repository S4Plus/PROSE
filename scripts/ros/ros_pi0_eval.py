
# Allow execution from any working directory without installing this checkout.
import sys as _prose_sys
from pathlib import Path as _ProsePath
_prose_sys.path.insert(0, str(_ProsePath(__file__).resolve().parents[2]))

# ros_eval.py (MODIFIED)
#eval指的是仿真环境接受推理结果，发送图像和关节信息，并记录此次任务是否成功

import sys
import numpy as np
from scipy.spatial.transform import Rotation as R
import argparse
import os
import pickle
import time
import torch
import cv2
from einops import rearrange
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState, CompressedImage
from std_msgs.msg import Float32MultiArray, String  # MODIFIED: Added String

# ==============================================================================
# Isaac Sim 相关依赖库
# ==============================================================================
from scipy.spatial.transform import Rotation, Slerp
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})

from isaacsim.core.utils.rotations import euler_angles_to_quat
from isaacsim.core.api import World
from omni.isaac.core.robots import Robot
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.core.utils.stage import add_reference_to_stage
from omni.isaac.core.prims import RigidPrim, XFormPrim
import omni.usd
from isaacsim.sensors.camera import Camera
from pxr import UsdGeom

class IsaacROS2Bridge(Node):
    def __init__(self):
        super().__init__('isaac_ros2_bridge')

        # Observation Publishers
        self.head_pub = self.create_publisher(CompressedImage, '/camera/head/image_rgb', 1)
        self.wrist_pub = self.create_publisher(CompressedImage, '/camera/wrist/image_rgb', 1)
        self.joint_pub = self.create_publisher(JointState, '/pi0_eval/joint_states', 1)
        
        # MODIFIED: Added control publisher
        self.control_pub = self.create_publisher(String, '/inference_control', 1)

        # Action Subscriber
        self.action_sub = self.create_subscription(
            Float32MultiArray, '/joint_action', self.action_callback, 1
        )

        self.latest_action = None

    def action_callback(self, msg):
        self.latest_action = np.array(msg.data)

    # MODIFIED: Added method to send reset signal
    def send_reset_signal(self):
        msg = String()
        msg.data = 'RESET'
        self.control_pub.publish(msg)
        self.get_logger().info('Published RESET signal to inference node.')

    def publish_images(self, head_img, wrist_img):
        # Encode images to JPEG
        _, head_buf = cv2.imencode('.jpg', head_img)
        _, wrist_buf = cv2.imencode('.jpg', wrist_img)
# 
        # Create CompressedImage messages
        head_msg = CompressedImage()
        wrist_msg = CompressedImage()

        current_time = self.get_clock().now().to_msg()
        head_msg.header.stamp = current_time
        wrist_msg.header.stamp = current_time
        head_msg.format = "jpeg"
        wrist_msg.format = "jpeg"
        head_msg.data = head_buf.tobytes()
        wrist_msg.data = wrist_buf.tobytes()

        # Publish
        self.head_pub.publish(head_msg)
        self.wrist_pub.publish(wrist_msg)

    def publish_joint_states(self, qpos):
        # print("send qpos", qpos)
        joint_state = JointState()
        joint_state.header.stamp = self.get_clock().now().to_msg()
        joint_state.name = [f"joint_{i}" for i in range(len(qpos))]
        joint_state.position = qpos.tolist()
        self.joint_pub.publish(joint_state)

    def get_latest_action(self):
        return self.latest_action

def generate_random_poses():
    pick_x_range = (1.06219, 1.26219)
    pick_y_range = (2.62, 3.12) 
    place_x_range = (-0.6, -0.58)
    place_y_range = (-0.3, 0)
    yaw = np.random.uniform(-np.pi / 4, np.pi / 4)
    pick_quat = euler_angles_to_quat([0, 0, yaw])
    place_quat = euler_angles_to_quat([0, 0, yaw])
    pos_pick = np.array([np.random.uniform(*pick_x_range), np.random.uniform(*pick_y_range), 0.9])
    pos_place = np.array([np.random.uniform(*place_x_range), np.random.uniform(*place_y_range), 0.10493])
    # # while abs(pos_pick[1] - pos_place[1]) < 0.3:
    #     pos_place = np.array([np.random.uniform(*place_x_range), np.random.uniform(*place_y_range), 0.10493])
    return {"position": pos_pick, "orientation": pick_quat}, {"position": pos_place, "orientation": place_quat}

def get_camera(prim_path, frequency, resolution):
    stage = omni.usd.get_context().get_stage()
    camera_prim = stage.GetPrimAtPath(prim_path)
    return Camera(prim_path=prim_path, frequency=frequency, resolution=resolution)

def main():
    # ... (omitting unchanged initialization code for brevity) ...
    usd_file_path = "/home/mirrow/FlexUMI/assets/sim_env/Belt_R1_Demo_pj.usd"
    world = World(physics_dt=1.0/20.0, rendering_dt=1.0/20.0, set_defaults=True)
    add_reference_to_stage(usd_file_path, "/World")

    r1_prim_path = "/World/r1"
    r1 = Robot(r1_prim_path)

    bottle_prim_path = "/World/bottle"
    bottle_prim = RigidPrim(prim_path=bottle_prim_path)

    basket_prim_path = "/World/black_box"
    basket_prim = XFormPrim(prim_path=basket_prim_path)

    CAMERA_FREQ = 20
    RESOLUTION_HEAD = (640, 360)
    RESOLUTION_WRIST = (640, 480)
    HEAD_CAMERA_PRIM_PATH = "/World/r1/torso_link4/visuals/zed_link/ZED_X_01/base_link/ZED_X/CameraLeft"
    RIGHT_WRIST_CAMERA_PRIM_PATH = "/World/r1/right_arm_link6/right_realsense_link/right_wrist_camera"
    head_camera = get_camera(HEAD_CAMERA_PRIM_PATH, CAMERA_FREQ, RESOLUTION_HEAD)
    right_wrist_camera = get_camera(RIGHT_WRIST_CAMERA_PRIM_PATH, CAMERA_FREQ, RESOLUTION_WRIST)

    init_torso_pos = np.array([-0.37779998779296875 , -0.014299999922513962 , -1.0469000339508057])
    init_leftarm_pos_6dof = np.array([1.448511, 2.941702, -2.557447, -0.359362, -0.049362, 0.359362])
    init_rightarm_pos_6dof = np.array([-0.845532, 2.346170, -1.114468, 0.026809, 1.369149, -1.172340])
    gripper_val = 0.05
    init_joint_positions = np.zeros(26)
    init_joint_positions[3] = init_torso_pos[0]
    init_joint_positions[7:9] = init_torso_pos[1:3]
    init_joint_positions[10:22][::2] = init_leftarm_pos_6dof
    init_joint_positions[10:22][1::2] = init_rightarm_pos_6dof
    init_joint_positions[22:26] = np.array([gripper_val] * 4)
    

    bottle_number = 0
    success_count = 0
    Test_num = 60
    max_timesteps = 250
    video_save_dir = os.path.join("./eval", "eval_belt_pick_bottle")
    os.makedirs(video_save_dir, exist_ok=True)
    print(f"Videos will be saved to: {video_save_dir}")

    for episode in range(Test_num):
        print(f"\n===== Starting Evaluation Episode {episode + 1}/{Test_num} =====")

        # FIXED: Clear latest_action at the start of each episode to prevent carryover from previous episode
        ros2_node.latest_action = None

        # --- Scene load and reset ---
        world.reset()
       
        # MODIFIED: Send reset signal to inference node at the start of the episode
       
        ros2_node.send_reset_signal()
        # Give a moment for the signal to be processed
        time.sleep(0.1)

        r1.initialize()
        bottle_prim.initialize()
        basket_prim.initialize()
        head_camera.initialize()
        right_wrist_camera.initialize()

        # --- Randomize bottle and basket positions ---
        pick_cfg, place_cfg = generate_random_poses()
        r1.set_joint_positions(init_joint_positions)
        bottle_prim.set_world_pose(position=np.array(pick_cfg["position"]), orientation=np.array(pick_cfg["orientation"]))
        # basket_prim.set_world_pose(position=np.array(place_cfg["position"]), orientation=np.array(place_cfg["orientation"]))

        # stage = omni.usd.get_context().get_stage()
        # parent_prim = stage.GetPrimAtPath('/World/base_basic_pbr/base_basic_pbr')
        # bottle_list = []
        # for child_prim in parent_prim.GetChildren():
        #     # 4. 判断Prim的类型是否为 'Xform'
        #     if child_prim.GetTypeName() == 'Xform':
        #         bottle_list.append(child_prim)
        # for child_prim in parent_prim.GetChildren():
        # # 4. 判断Prim的类型是否为 'Xform'
        #     if child_prim.GetTypeName() == 'Xform':
        #         # 5. 设置其为不可见
        #         # UsdGeom.Imageable 是处理可见性的API
        #         imageable = UsdGeom.Imageable(child_prim)
        #         # 'inherited' 表示让子物体继承父物体的可见性。
        #         # 'invisible' 表示强制其不可见。
        #         imageable.GetVisibilityAttr().Set(UsdGeom.Tokens.invisible)
        # imageable = UsdGeom.Imageable(bottle_list[3])
        # imageable.GetVisibilityAttr().Set(UsdGeom.Tokens.visible)
        # bottle_number += 1

        for _ in range(30):
            world.step(render=True)

        t = 0

        temp_video_path = os.path.join(video_save_dir, f"episode_{episode+1}_temp.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(temp_video_path, fourcc, CAMERA_FREQ, RESOLUTION_HEAD)

        # --- Single episode inference loop ---
        while t < max_timesteps and simulation_app.is_running():
            # Get current state and images for observation
            current_joint_state = r1.get_joint_positions()
            obs_qpos_raw = np.concatenate([current_joint_state[10:22][1::2], [current_joint_state[25]]])
            img_head_camera = head_camera.get_rgb()
            img_right_wrist = right_wrist_camera.get_rgb()
            img_right_wrist = cv2.resize(np.array(img_right_wrist), (640, 360))
            if img_head_camera is None or img_right_wrist is None:
                print(f"Frame {t}: Waiting for camera images...")
                world.step(render=True)
                continue
            
            # Publish observations once
            ros2_node.publish_images(img_head_camera, img_right_wrist)
            ros2_node.publish_joint_states(obs_qpos_raw)
            rclpy.spin_once(ros2_node, timeout_sec=0.001)
            frame_bgr = cv2.cvtColor(img_head_camera[:, :, :3], cv2.COLOR_RGB2BGR)
            video_writer.write(frame_bgr)
            
            # FIXED: Inner loop to wait for action chunk without republishing observations
            action_chunk = None
            wait_count = 0
            while action_chunk is None or len(action_chunk) < 50 * 7:
                rclpy.spin_once(ros2_node, timeout_sec=0.001)
                action_chunk = ros2_node.get_latest_action()
                if action_chunk is None or len(action_chunk) < 50 * 7:
                    time.sleep(0.02)
                    wait_count += 1
                    if wait_count % 50 == 0:  # Print every second
                        print(f"Frame {t}: Waiting for valid action chunk... (waited {wait_count * 0.02:.1f}s)")
                    continue
            action_chunk = np.array(action_chunk).reshape(50, -1)
            
            # FIXED: Copy from current joint positions and update after each step to prevent teleporting
            # MODIFIED: Execute the full chunk of 20 actions without publishing new observations
            chunk_actions_executed = 0
            for i in range(50):
                if t >= max_timesteps:
                    break
                
                # Extract the i-th action from the chunk
                single_action = action_chunk[i]
                target_qpos = single_action[:7]  # Assuming action_dim starts with some prefix, adjust if needed
                new_joint_positions = np.copy(init_joint_positions)  # Copy current state
                new_joint_positions[11:22:2] = target_qpos[:6]
                gripper_val_current = target_qpos[6]
                if gripper_val_current < 0.03:
                    gripper_val_current = 0.01
                new_joint_positions[24:26] = np.array([gripper_val_current] * 2)
                r1.apply_action(ArticulationAction(joint_positions=new_joint_positions))
                
                # Step the simulation after applying action
                world.step(render=True)
                
                # Update current_joint_state after step for next action
                current_joint_state = r1.get_joint_positions()
                
                # Save frame for this sub-step
                current_head_img = head_camera.get_rgb()
                if current_head_img is not None:
                    frame_bgr_sub = cv2.cvtColor(current_head_img[:, :, :3], cv2.COLOR_RGB2BGR)
                    video_writer.write(frame_bgr_sub)
                else:
                    # Fallback to previous frame if new image not ready
                    video_writer.write(frame_bgr)
                
                # time.sleep(0.05)
                print(f"Episode {episode + 1}, Frame: {t}, Chunk Action {i+1}/50, Target Joint: {np.round(target_qpos, 3)}", end="\r")
                t += 1
                chunk_actions_executed += 1

            # Clear the latest action after executing the chunk to prepare for next observation
            ros2_node.latest_action = None
            print(f"\nExecuted chunk of {chunk_actions_executed} actions at t={t}")
            
        video_writer.release()
        
        # FIXED: Clear latest_action at end of episode as well
        ros2_node.latest_action = None
        
        # ... (omitting unchanged success/failure logic for brevity) ...
        final_bottle_pos, _ = bottle_prim.get_world_pose()
        target_pos, target_orientation_wxyz = basket_prim.get_world_pose()
        vec_bottle_in_world = final_bottle_pos - target_pos
        target_orientation_xyzw = np.roll(target_orientation_wxyz, -1)
        basket_rotation = Rotation.from_quat(target_orientation_xyzw)
        vec_bottle_in_local = basket_rotation.inv().apply(vec_bottle_in_world)
        basket_length = 0.43
        basket_width = 0.25
        bottle_local_x = vec_bottle_in_local[0]
        bottle_local_y = vec_bottle_in_local[1]
        is_within_bounds = (abs(bottle_local_x) <= basket_width / 2.0) and \
                        (abs(bottle_local_y) <= basket_length / 2.0)
        if is_within_bounds:
            final_video_path = os.path.join(video_save_dir, f"episode_{episode+1}_success.mp4")
            os.rename(temp_video_path, final_video_path)
            print(f"\nEpisode {episode + 1}: SUCCESS! Bottle is inside the rotated basket's boundary.")
            success_count += 1
        else:
            final_video_path = os.path.join(video_save_dir, f"episode_{episode+1}_failure.mp4")
            os.rename(temp_video_path, final_video_path)
            print(f"\nEpisode {episode + 1}: FAILURE. Bottle is outside the rotated basket's boundary. Position in basket frame: x={bottle_local_x:.3f}, y={bottle_local_y:.3f}")

    success_rate = (success_count / Test_num) * 100
    print(f"Success Rate: {success_rate:.2f}%")
    simulation_app.close()

if __name__ == '__main__':
    rclpy.init()
    ros2_node = IsaacROS2Bridge()
    main()