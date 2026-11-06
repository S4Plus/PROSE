import os
import numpy as np
import h5py
from PIL import Image

def resize_images_pil(images, target_size=(128, 128)):
    resized_images = []
    for i in range(images.shape[0]):
        # 转换为PIL图像，调整大小，再转回numpy数组
        img = Image.fromarray(images[i])
        resized_img = img.resize(target_size, Image.Resampling.LANCZOS)
        resized_images.append(np.array(resized_img))
    return np.array(resized_images)

def convert_to_hdf5(episode_path, episode_num, hdf5_output_dir):
    # 定义 HDF5 文件路径
    hdf5_file_path = os.path.join(hdf5_output_dir, f"episode_{episode_num}.hdf5")
    
    # 打开 HDF5 文件
    with h5py.File(hdf5_file_path, 'w') as hdf5_file:
        # 创建 action 数据集
        joint_trajectory_path = os.path.join(episode_path, "data", "rightarm_joint.npy")
        joint_trajectory = np.load(joint_trajectory_path)

        episode_length = joint_trajectory.shape[0]

        action = joint_trajectory[:episode_length - 1, :7]  # 假设前 6 列是关节角度，第 7 列是夹爪宽度


        la_trajectory_path = os.path.join(episode_path, "LA", "la_villa.npy")
        la_trajectory = np.load(la_trajectory_path)
        la = la_trajectory[:, :64]


        hdf5_file.create_dataset('action', data=action)
        
        # 创建 observations 数据集
        observations = hdf5_file.create_group('observations')
        
        # 创建 qpos 数据集
        qpos = np.concatenate([action, la], axis=1)  # 将 action 和 la 拼接在一起作为 qpos
        observations.create_dataset('qpos', data=qpos)
        
        # 创建 images 数据集
        images = observations.create_group('images')
        
        # 处理 head_camera 数据
        head_camera_path = os.path.join(episode_path, "CameraLeft", "all_rgb.npy")
        head_camera_data = np.load(head_camera_path)
        resized_data = resize_images_pil(head_camera_data, (128, 128))
        # print(f"Resized head_camera data shape: {resized_data.shape}")
        images.create_dataset('head_camera', data=resized_data[:episode_length - 1])
        
        # # 处理 left_wrist_camera 数据
        # left_wrist_camera_path = os.path.join(episode_path, "left_wrist", "all_rgb.npy")
        # left_wrist_camera_data = np.load(left_wrist_camera_path)
        # images.create_dataset('left_wrist_camera', data=left_wrist_camera_data)


        # 处理 right_wrist_camera 数据
        # right_wrist_camera_path = os.path.join(episode_path, "right_wrist_camera", "all_rgb.npy")
        # right_wrist_camera_data = np.load(right_wrist_camera_path)
        # images.create_dataset('right_wrist_camera', data=right_wrist_camera_data)

    print(f"Data for episode {episode_num} has been successfully converted to {hdf5_file_path}")

def batch_convert_to_hdf5(base_dir, hdf5_output_dir):
    # 确保输出目录存在
    os.makedirs(hdf5_output_dir, exist_ok=True)
    
    # 遍历 base_dir 下的所有 epo_x 文件夹
    for episode_dir in sorted(os.listdir(base_dir)):
        if episode_dir.startswith("epo_") and episode_dir[4:].isdigit():
            episode_num = int(episode_dir[4:])
            episode_path = os.path.join(base_dir, episode_dir)
            convert_to_hdf5(episode_path, episode_num, hdf5_output_dir)

# 示例调用
base_dir = "/home/Static_Pick_Bottle_Randomized"  # 替换为你的 data_generate 目录路径
hdf5_output_dir = "/home/Static_Pick_Bottle_Randomized/hdf5_action"  # 替换为你希望保存 HDF5 文件的目录路径
batch_convert_to_hdf5(base_dir, hdf5_output_dir)