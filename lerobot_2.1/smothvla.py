"""
    ACT 模型在源络机器人上的推理代码。
    用 end pose 训练，输入观测包含 head camera， wrist camera，以及 gelsight image
    输出为 绝对 end pose

    格式：
    observation.image 是 front camera + wrist camera 
    states 用的是绝对tcp位姿 + 夹爪宽度 + 6维度力传感器
    action 是 tcp 位姿 + 夹爪宽度
"""
import time
import torch
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
import rclpy
from sensor_msgs.msg import Image, JointState,CompressedImage
from scipy.spatial.transform import Rotation as R
import os
import sys
import time
sys.path.append(os.path.join(os.getcwd(), "lib"))
# from RobotLib import Robot
from geometry_msgs.msg import PoseStamped
import threading
from rclpy.executors import MultiThreadedExecutor
from real_world.data_receiver import DataReceiver
import cv2
import numpy as np
import os
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import sys
sys.path.append(os.path.join(os.getcwd(), "lib"))
from RobotLib import Robot

def euler_to_wxyz_quaternion(euler_angles):
    """
    Convert Euler angles (XYZ, radians) to a quaternion in wxyz format.
    
    Args:
        euler_angles (list): [roll, pitch, yaw] in radians (XYZ convention).
    
    Returns:
        list: Quaternion [qw, qx, qy, qz] in wxyz format.
    """
    # Create rotation object from Euler angles
    rotation = R.from_euler('xyz', euler_angles, degrees=False)
    # Get quaternion in xyzw format [qx, qy, qz, qw]
    xyzw_quat = rotation.as_quat()
    # Convert to wxyz format [qw, qx, qy, qz]
    wxyz_quat = [xyzw_quat[3], xyzw_quat[0], xyzw_quat[1], xyzw_quat[2]]
    return wxyz_quat

def debug(host="0.0.0.0", port=5678, wait_client=True):
    import debugpy
    debugpy.listen((host, port))
    print(f"✅ debugpy is listening on {host}:{port}")
    if wait_client:
        print("⏳ Waiting for debugger to attach...")
        debugpy.wait_for_client()
        print(" Debugger attached, execution will continue.")


def matrix_to_pose(T):
    """Converts a 4x4 transformation matrix to a pose [t_x, t_y, t_z, q_w, q_x, q_y, q_z]."""
    r = R.from_matrix(T[:3, :3])
    q_xyzw = r.as_quat()
    q_wxyz = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])
    t = T[:3, 3]
    return np.concatenate((t, q_wxyz))

def pose_to_matrix(pose):
    """Converts a pose [t_x, t_y, t_z, q_w, q_x, q_y, q_z] to a 4x4 transformation matrix."""
    t = np.array(pose[:3])
    q_wxyz = np.array(pose[3:])
    q_xyzw = np.array([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]])
    r = R.from_quat(q_xyzw)
    T = np.eye(4)
    T[:3, :3] = r.as_matrix()
    T[:3, 3] = t
    return T


TOPIC_CONFIG = {
    "front_camera": {   # 自定义 key，方便取数据
        "topic": "/camera_head_front/color/image_raw/compressed",
        "msg_type": CompressedImage,
        "type": "image",
    },
    "left_wrist_camera":{
        "topic": '/left/color/image_raw/compressed',
        "msg_type": CompressedImage,
        "type": "image"
    },
}

device = "cuda"

trajectory_points = []
            
# ====== 获取下一步动作 ======
def get_next_action_chunk(observation, policy, device="cuda"):
    batch = {
        "observation.state": observation["observation.state"].unsqueeze(0).to(device),  # (1, dof)
        # "observation.images.head_camera": observation["observation.images.front"].unsqueeze(0).to(device),  # (1, 3, H, W)
        # "observation.images.wrist_left_camera": observation["observation.images.left_wrist"].to(device) 
        "observation.images.wrist_left_camera": observation["observation.images.left_wrist"].unsqueeze(0).to(device), #  如果只有一个摄像头
        "task": "Insert the USB flash into the usb strip"
    }
    with torch.no_grad():
        pred_action = policy.predict_action_chunk(batch)
    return pred_action.cpu()  # (dof,)

            

def data_receiver(logger_node):
    # 多线程执行器是必须的，因为回调函数和主循环（通过线程实现）会并行
    executor = MultiThreadedExecutor()
    executor.add_node(logger_node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        logger_node.get_logger().info('Keyboard interrupt, shutting down.')
    finally:
        logger_node.on_shutdown()
        logger_node.destroy_node()
        rclpy.shutdown()
def visualize_trajectory(current_points, ax, fig):
    """可视化轨迹的函数"""
    global trajectory_points
    
    # 添加新的轨迹点
    trajectory_points.extend(current_points)
    # 限制轨迹点数量，防止内存占用过大
    if len(trajectory_points) > 1000:
        trajectory_points = trajectory_points[-1000:]
    
    # 清空当前图像
    ax.clear()
    
    # 提取x, y, z坐标
    if trajectory_points:
        x = [p[0] for p in trajectory_points]
        y = [p[1] for p in trajectory_points]
        z = [p[2] for p in trajectory_points]
        
        # 绘制轨迹
        ax.plot(x, y, z, 'b-', linewidth=1.5, label='Trajectory')
        # 标记最新的点
        ax.scatter(x[-1], y[-1], z[-1], c='r', s=50, label='Current Position')
    
    # 设置坐标轴标签和范围（根据你的实际工作空间调整）
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_xlim([-0.5, 0.5])  # 示例范围，需根据实际情况调整
    ax.set_ylim([-0.5, 0.5])
    ax.set_zlim([0, 1.0])
    ax.set_title('End Effector Trajectory')
    ax.legend()
    
    # 更新图像
    fig.canvas.draw()
    fig.canvas.flush_events()

# ====== main 循环 ======
def main():
    rclpy.init()
    logger_node = DataReceiver(TOPIC_CONFIG)
    data_receiver_thread = threading.Thread(target=data_receiver, args=(logger_node,))
    data_receiver_thread.start()

    component_type = 1  # 1: 左臂
    robot =  Robot("192.168.11.3:50051", "", "")
    robot.set_arm_enable(component_type, 1)
    robot.set_arm_mode(component_type, 0) # 0 位置控制，1伺服控制
    robot.set_arm_state(component_type, 0)
    robot.set_gripper_enable(component_type, 1)
    robot.set_gripper_mode(component_type, 1)
 
     # 加载策略

    policy = SmolVLAPolicy.from_pretrained("outputs/train/smothvla/checkpoints/100000/pretrained_model").to(device)

    # 控制频率
    rate_hz = 30
    dt = 1.0 / rate_hz
    IMG_H = 360
    IMG_W = 640

    T_C_EE_initial = pose_to_matrix(robot.get_arm_end_pose(component_type)[1])
    T_C_TCP_initial = pose_to_matrix(robot.get_arm_position(component_type)[1])
    T_EE_TCP = np.linalg.inv(T_C_EE_initial) @ T_C_TCP_initial
    T_TCP_EE = np.linalg.inv(T_EE_TCP)
    
    trajectory_points = []
    # 创建可视化窗口
    plt.ion()  # 开启交互模式
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    while rclpy.ok():
        start = time.time()
        latest = logger_node.get_latest_message_all()

        img_msg = latest["front_camera"]
        wrist_camera_msg = latest["left_wrist_camera"]
        gripper_width = robot.get_gripper_position(component_type)[1]
        tcp_pose = pose_to_matrix(robot.get_arm_position(component_type)[1])
        current_tcp_pos = tcp_pose[:3, 3].tolist()
        current_tcp_euler = R.from_matrix(tcp_pose[:3, :3]).as_euler("xyz", degrees=False).tolist()
        force_data = robot.get_force_sensor_data(component_type)[1]

        if img_msg is None or wrist_camera_msg is None or force_data is None:
            if img_msg is None:
                print("No front camera image received.")
            if wrist_camera_msg is None:
                print("No wrist camera image received.")
            if force_data is None:
                print("No force sensor data received.")
            continue  # 等待数据

        # === 构造 state ===
        state = np.concatenate([current_tcp_pos, current_tcp_euler, [gripper_width], force_data])
        state = torch.tensor(state, dtype=torch.float32).to(device)

        # === 转换 image ===
        np_arr = np.frombuffer(img_msg.data, np.uint8)           # 将压缩数据转为 uint8 数组
        cv_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)      # 解码为 BGR 图像
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)       # BGR -> RGB
        cv_img = cv2.resize(cv_img, (IMG_W, IMG_H))         # 调整大小
        img_tensor = torch.tensor(cv_img, dtype=torch.float32).permute(2,0,1) / 255.0

        # === 构造 wrist camera image ===
        np_arr = np.frombuffer(wrist_camera_msg.data, np.uint8)           # 将压缩数据转为 uint8 数组
        cv_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)      # 解码为 BGR 图像
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)       # BGR -> RGB
        cv_img = cv2.resize(cv_img, (IMG_W, IMG_H))         # 调整大小
        wrist_camera_img_tensor = torch.tensor(cv_img, dtype=torch.float32).permute(2,0,1) / 255.0

        from torchvision.utils import save_image
        save_image(wrist_camera_img_tensor, "wrist.png")
        # === 组合 observation ===
        observation = {
            "observation.state": state,
            # "observation.images.front": img_tensor.to(device),
            "observation.images.left_wrist": wrist_camera_img_tensor.to(device)
        }

        # === 运行 policy ===
        use_action_num = 50
        actions = get_next_action_chunk(observation, policy, device=device)[0]
        actions = actions[:use_action_num] # 只使用前 use_action_num 个动作
        actions = [actions[-1]] # 只使用前 use_action_num 个动作
        
        drop_action = 0 # 舍弃前10 步
        step = 50
        selected_actions = actions[drop_action::step]  # 每隔5步取一个
        
        # print(f"Predicted action: {action.numpy()}")
        trajectory_points_to_plot = []
        for action in selected_actions:
            # 提取位置信息 (x, y, z)
            position = action[:3].cpu().numpy()  # 假设action在GPU上，需要转到CPU
            trajectory_points_to_plot.append(position)
            
            gripper = action[-1]       # 夹持器控制信号 (1D)
            ee_pose_euler = action[:-1] # 欧拉角姿态 (6D: x, y, z, roll, pitch, yaw)

            # 3欧拉角到四元数的转换
            euler_angles = ee_pose_euler[3:]
            quaternion_wxyz = euler_to_wxyz_quaternion(euler_angles)
            ee_pose_quaternion_7d = np.concatenate([position, quaternion_wxyz])
        
            robot.set_arm_position(component_type, ee_pose_quaternion_7d.tolist(),0.3,0.1,True) 
            robot.set_gripper_position(component_type, 0.005)
        
        # 可视化轨迹
        # visualize_trajectory(trajectory_points_to_plot, ax, fig)

        end = time.time()
        print(f"fps: {1.0/(end-start)}")


if __name__ == "__main__":
    # debug()
    main()

