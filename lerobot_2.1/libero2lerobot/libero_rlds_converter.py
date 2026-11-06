# -*- coding: utf-8 -*-
"""
Libero统一数据转换器

支持自动识别RLDS和HDF5格式，自动解析成LeRobotDataSet格式，支持多线程操作
"""

import argparse
import ast
import json
import logging
import shutil
import os
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Callable, Tuple
import concurrent.futures
from functools import partial

import cv2
import h5py
import numpy as np
from tqdm import tqdm

# 检查依赖
try:
    import tensorflow_datasets as tfds
    import tensorflow as tf
    HAS_TF = True
except ImportError:
    HAS_TF = False
    logging.warning("tensorflow_datasets未安装，RLDS支持将被禁用")

try:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.common.constants import HF_LEROBOT_HOME
except ImportError:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.constants import HF_LEROBOT_HOME


# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatasetFormatDetector:
    """数据集格式检测器"""
    
    @staticmethod
    def detect_format(data_path: Union[str, Path]) -> str:
        """
        自动检测数据集格式
        
        Args:
            data_path: 数据集路径
            
        Returns:
            str: 'rlds' 或 'hdf5'
        """
        data_path = Path(data_path)

        # 如果是单文件，直接根据后缀判断
        if data_path.is_file():
            suffix = data_path.suffix.lower()
            if suffix in {".hdf5", ".h5"}:
                logger.info("检测到HDF5格式（单文件）")
                return "hdf5"
            if suffix.startswith(".tfrecord"):
                logger.info("检测到RLDS格式（tfrecord单文件）")
                return "rlds"
            raise ValueError(f"无法检测数据格式：{data_path}")
        
        # 检查HDF5格式：查找.hdf5或.h5文件
        hdf5_files = list(data_path.rglob("*.hdf5")) + list(data_path.rglob("*.h5"))
        
        # 检查RLDS格式：查找tfrecord文件或dataset_info.json
        rlds_indicators = (
            list(data_path.rglob("*.tfrecord*")) + 
            list(data_path.rglob("dataset_info.json")) +
            list(data_path.rglob("features.json"))
        )
        
        if hdf5_files and not rlds_indicators:
            logger.info(f"检测到HDF5格式，找到{len(hdf5_files)}个HDF5文件")
            return "hdf5"
        elif rlds_indicators and not hdf5_files:
            logger.info(f"检测到RLDS格式，找到相关文件："
                       f"{[f.name for f in rlds_indicators[:3]]}")
            return "rlds"
        elif hdf5_files and rlds_indicators:
            logger.warning("同时发现HDF5和RLDS文件，优先使用HDF5格式")
            return "hdf5"
        else:
            raise ValueError(f"无法检测数据格式：{data_path}")


class HDF5Processor:
    """HDF5数据处理器"""
    
    def __init__(
        self,
        image_size: Tuple[int, int] = (128,128),
        state_dim: int = 9,
        use_videos: bool = False,
    ):
        """
        初始化HDF5处理器
        
        Args:
            image_size: 图像尺寸 (height, width) - 匹配numpy数组格式
            use_videos: 是否使用视频格式
        """
        self.image_size = image_size  # (height, width)
        self.state_dim = state_dim
        self.use_videos = use_videos

    def _build_state(self, joint_states: np.ndarray, gripper_states: Optional[np.ndarray] = None) -> np.ndarray:
        """构造 observation.state，默认使用 joint + gripper。"""
        joint_states = np.asarray(joint_states)
        if gripper_states is None:
            return joint_states
        gripper_states = np.asarray(gripper_states)
        return np.concatenate([joint_states, gripper_states], axis=-1)

    def _build_action(
        self,
        joint_states: np.ndarray,
        raw_actions: Optional[np.ndarray] = None,
        gripper_states: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """构造 8 维 action: 下一时刻 joint target(7) + 当前时刻 gripper_cmd(1)。"""
        joint_states = np.asarray(joint_states)
        next_joint_targets = np.concatenate([joint_states[1:], joint_states[-1:]], axis=0)

        gripper_cmd = None
        if raw_actions is not None:
            raw_actions = np.asarray(raw_actions)
            if raw_actions.ndim >= 2 and raw_actions.shape[-1] >= 1:
                gripper_cmd = raw_actions[..., -1:]

        if gripper_cmd is None and gripper_states is not None:
            gripper_states = np.asarray(gripper_states)
            if gripper_states.ndim >= 2 and gripper_states.shape[-1] >= 2:
                # 双指开合量压成单标量，作为兜底 gripper_cmd。
                gripper_cmd = gripper_states[..., :1] - gripper_states[..., 1:2]
            elif gripper_states.ndim >= 2 and gripper_states.shape[-1] >= 1:
                gripper_cmd = gripper_states[..., :1]

        if gripper_cmd is None:
            gripper_cmd = np.zeros((len(joint_states), 1), dtype=joint_states.dtype)

        return np.concatenate([next_joint_targets, gripper_cmd], axis=-1)

    def _prepare_image(self, image: np.ndarray) -> np.ndarray:
        """修正MuJoCo默认的垂直翻转，只在尺寸不一致时才缩放。"""
        # MuJoCo/OpenGL 渲染原点在左下，raw 帧通常是上下颠倒存储的
        img = np.flip(image, 0)
        if img.shape[:2] != self.image_size:
            img = cv2.resize(img, (self.image_size[1], self.image_size[0]))
        return img
    
    def get_default_features(self, use_videos: bool = True) -> Dict[str, Dict[str, Any]]:
        """获取Libero数据集的默认特征配置"""
        image_dtype = "video" if use_videos else "image"
        
        return {
            "observation.images.front": {
                "dtype": image_dtype,
                "shape": (*self.image_size, 3),
                "names": ["height", "width", "channel"],
            },
            "observation.images.wrist": {
                "dtype": image_dtype,
                "shape": (*self.image_size, 3),
                "names": ["height", "width", "channel"],
            },
            "observation.state": {
                "dtype": "float32",
                "shape": (self.state_dim,),
                "names": [f"state_{i}" for i in range(self.state_dim)],
            },
            "action": {
                "dtype": "float32",
                "shape": (8,),
                "names": [f"action_{i}" for i in range(8)],
            }
        }
    
    def process_episode(self, episode_path: Path, dataset: LeRobotDataset, task_name: str) -> bool:
        """
        处理单个episode数据
        
        Args:
            episode_path: episode文件路径
            dataset: LeRobot数据集
            task_name: 任务名称
            
        Returns:
            bool: 处理是否成功
        """
        try:
            with h5py.File(episode_path, "r") as file:
                logger.debug(f"HDF5文件键: {list(file.keys())}")
                
                # 检测HDF5文件格式，支持多种Libero格式
                if "data" in file:
                    # 新的Libero格式：data/demo_N/...
                    return self._process_libero_demo_format(file, dataset, task_name, episode_path)
                else:
                    logger.warning(f"未识别的HDF5格式: {episode_path}")
                    return False

        except (FileNotFoundError, OSError, KeyError) as e:
            logger.error(f"跳过 {episode_path}: {str(e)}")
            return False
            

    def _process_libero_demo_format(self, file: h5py.File, dataset: LeRobotDataset, task_name: str, file_path: Optional[Path] = None) -> bool:
        """处理新的Libero demo格式：data/demo_N/..."""
        data_group = file["data"]
        
        # 尝试从文件根级别提取任务信息
        task_str = self._extract_task_info(file, task_name, file_path)
        
        # 获取所有demo
        demo_keys = [k for k in data_group.keys() if k.startswith("demo_")]
        demo_keys.sort(key=lambda x: int(x.split("_")[1]))  # 按数字排序
        
        for demo_key in demo_keys:
            demo_group = data_group[demo_key]
            logger.info(f"处理 {demo_key}")
            
            # 尝试从demo级别提取任务信息
            demo_task_str = self._extract_task_info(demo_group, task_str, file_path)
            
            # 读取原始动作数据
            raw_actions = np.array(demo_group["actions"])

            # 读取观察数据
            obs_group = demo_group["obs"]
            
            # 读取关节状态和夹爪状态，组合作为 observation.state
            joint_states = np.array(obs_group["joint_states"])
            gripper_states = np.array(obs_group["gripper_states"]) if "gripper_states" in obs_group else None
            state = self._build_state(joint_states, gripper_states)
            action = self._build_action(joint_states, raw_actions, gripper_states)
            
            # 读取图像数据
            agentview_rgb = np.array(obs_group["agentview_rgb"])  # 前视图像
            eye_in_hand_rgb = np.array(obs_group["eye_in_hand_rgb"])  # 腕部图像
            
            # 确保所有数组长度一致
            num_frames = min(len(action), len(state), len(agentview_rgb), len(eye_in_hand_rgb))
            
            # 处理每一帧
            for i in tqdm(range(num_frames), desc=f"处理 {demo_key}", leave=False):
                # 处理图像：矫正垂直翻转，必要时再缩放
                front_img = self._prepare_image(agentview_rgb[i])
                wrist_img = self._prepare_image(eye_in_hand_rgb[i])
                
                
                # 准备帧数据 - 参考RLDS格式
                frame_data = {
                    "action": action[i].astype(np.float32),
                    "observation.state": state[i].astype(np.float32),
                    "observation.images.front": front_img,
                    "observation.images.wrist": wrist_img,
                }
                
                # 添加帧到数据集 - 修复API调用
                dataset.add_frame(frame_data, task=demo_task_str)
            
            # 每个demo保存为一个episode
            dataset.save_episode()
        
        return True

    def _extract_task_info(self, group: h5py.Group, default_task: str, file_path: Optional[Path] = None) -> str:
        """从HDF5组中提取任务信息"""
        # 尝试多种可能的任务信息字段
        task_fields = [
            "language_instruction", "task_description", "task", "description",
            "instruction", "goal", "task_name", "task_info"
        ]
        
        for field in task_fields:
            if field in group:
                try:
                    task_data = group[field]
                    if hasattr(task_data, 'asstr'):
                        # 处理字符串数组
                        task_str = task_data.asstr()[()]
                    elif isinstance(task_data, (bytes, str)):
                        # 处理字节或字符串
                        task_str = task_data.decode() if isinstance(task_data, bytes) else task_data
                    else:
                        # 尝试转换为字符串
                        task_str = str(task_data[()])
                    
                    if task_str and task_str.strip():
                        logger.info(f"从字段 '{field}' 提取到任务: {task_str}")
                        return task_str.strip()
                except Exception as e:
                    logger.debug(f"提取字段 '{field}' 失败: {e}")
                    continue
        
        # 如果没有找到任务信息，尝试从文件名提取
        if file_path is not None:
            task_str = self._extract_task_from_filename(file_path)
            if task_str:
                logger.info(f"从文件名提取到任务: {task_str}")
                return task_str
        
        # 如果都没有找到任务信息，返回默认值
        return default_task
    
    def _extract_task_from_filename(self, file_path: Path) -> Optional[str]:
        """从文件名中提取任务描述"""
        try:
            # 获取文件名（不含扩展名）
            filename = file_path.stem
            
            # 常见的后缀需要移除
            suffixes_to_remove = [
                '_demo', '_trajectory', '_episode', '_data', '_hdf5',
                'demo', 'trajectory', 'episode', 'data', 'hdf5'
            ]
            
            # 移除后缀
            task_name = filename
            for suffix in suffixes_to_remove:
                if task_name.endswith(suffix):
                    task_name = task_name[:-len(suffix)]
                    break
            
            # 将下划线替换为空格，使其更易读
            task_name = task_name.replace('_', ' ')
            
            # 如果任务名称太短，可能不是有效的任务描述
            if len(task_name.strip()) < 5:
                return None
            
            return task_name.strip()
            
        except Exception as e:
            logger.debug(f"从文件名提取任务失败: {e}")
            return None

    def _extract_libero_demo_frames(self, file: h5py.File, task_name: str, file_path: Optional[Path] = None) -> List[List[Dict]]:
        """提取Libero demo格式的帧数据（用于多线程处理）- 返回按demo分组的数据"""
        demos_frames = []
        data_group = file["data"]
        
        # 尝试从文件根级别提取任务信息
        task_str = self._extract_task_info(file, task_name, file_path)
        
        # 获取所有demo
        demo_keys = [k for k in data_group.keys() if k.startswith("demo_")]
        demo_keys.sort(key=lambda x: int(x.split("_")[1]))  # 按数字排序
        
        for demo_key in demo_keys:
            demo_frames = []
            demo_group = data_group[demo_key]
            
            # 尝试从demo级别提取任务信息
            demo_task_str = self._extract_task_info(demo_group, task_str, file_path)
            
            # 读取原始动作数据
            raw_actions = np.array(demo_group["actions"])

            # 读取观察数据
            obs_group = demo_group["obs"]
            
            # 读取关节状态和夹爪状态，组合作为 observation.state
            joint_states = np.array(obs_group["joint_states"])
            gripper_states = np.array(obs_group["gripper_states"]) if "gripper_states" in obs_group else None
            state = self._build_state(joint_states, gripper_states)
            action = self._build_action(joint_states, raw_actions, gripper_states)
            
            # 读取图像数据
            agentview_rgb = np.array(obs_group["agentview_rgb"])  # 前视图像
            eye_in_hand_rgb = np.array(obs_group["eye_in_hand_rgb"])  # 腕部图像
            
            # 确保所有数组长度一致
            num_frames = min(len(action), len(state), len(agentview_rgb), len(eye_in_hand_rgb))
            
            # 处理每一帧
            for i in range(num_frames):
                # 处理图像：矫正垂直翻转，必要时再缩放
                front_img = self._prepare_image(agentview_rgb[i])
                wrist_img = self._prepare_image(eye_in_hand_rgb[i])
                
                # 准备帧数据 - 参考RLDS格式
                frame_data = {
                    "action": action[i].astype(np.float32),
                    "observation.state": state[i].astype(np.float32),
                    "observation.images.front": front_img,
                    "observation.images.wrist": wrist_img,
                    "task": demo_task_str,  # 暂时保留在字典中
                }
                demo_frames.append(frame_data)
            
            demos_frames.append(demo_frames)
        
        return demos_frames

    def _extract_direct_episode_frames(self, file: h5py.File, task_name: str, file_path: Optional[Path] = None) -> List[List[Dict]]:
        """提取直接episode格式的帧数据（兜底方案，用于多线程处理）- 返回按episode分组的数据"""
        # 兜底方案：将整个文件作为一个episode
        episode_frames = []
        
        # 尝试提取任务信息
        task_str = self._extract_task_info(file, task_name, file_path)
        
        # 优先使用 joint + gripper 作为状态；如果缺失则退化到其他状态字段
        state_data = None
        if "joint_states" in file:
            joint_states = np.array(file["joint_states"])
            gripper_states = np.array(file["gripper_states"]) if "gripper_states" in file else None
            state_data = self._build_state(joint_states, gripper_states)
            logger.info(f"找到状态数据: joint_states(+gripper_states), shape: {state_data.shape}")
        else:
            state_keys = ["states", "robot_states", "state"]
            for key in state_keys:
                if key in file:
                    state_data = np.array(file[key])
                    logger.info(f"找到状态数据: {key}, shape: {state_data.shape}")
                    break

        if state_data is None:
            raise KeyError("未找到任何状态数据")
        
        # 尝试找到动作数据
        action_keys = ["actions", "action"]
        action_data = None
        
        for key in action_keys:
            if key in file:
                action_data = np.array(file[key])
                break

        if "joint_states" in file:
            joint_states = np.array(file["joint_states"])
            gripper_states = np.array(file["gripper_states"]) if "gripper_states" in file else None
            action_data = self._build_action(joint_states, action_data, gripper_states)
        elif action_data is None:
            logger.warning("未找到动作数据，使用状态数据作为动作")
            action_data = state_data
        
        # 尝试找到图像数据
        image_keys = ["agentview_rgb", "images", "rgb"]
        image_data = None
        
        for key in image_keys:
            if key in file:
                image_data = np.array(file[key])
                break
        
        if image_data is None:
            logger.warning("未找到图像数据，将创建空图像")
            image_data = np.zeros((len(state_data), *self.image_size, 3), dtype=np.uint8)
        
        # 确保所有数组长度一致
        num_frames = min(len(state_data), len(action_data), len(image_data))
        
        # 处理每一帧
        for i in range(num_frames):
            # 处理图像
            if image_data.ndim == 4:  # 有时间维度
                img = self._prepare_image(image_data[i])
            else:  # 没有时间维度，使用第一张图
                if len(image_data) > 0:
                    img = self._prepare_image(image_data[0])
                else:
                    img = np.zeros((*self.image_size, 3), dtype=np.uint8)
            

            
            frame_data = {
                "action": action_data[i].astype(np.float32),
                "observation.state": state_data[i].astype(np.float32),
                "observation.images.front": img,
                "observation.images.wrist": img,  # 使用相同图像作为腕部视图
                "task": task_str,  # 暂时保留在字典中
            }
            episode_frames.append(frame_data)
        
        return [episode_frames]  # 返回单个episode的列表


class RLDSProcessor:
    """RLDS数据处理器"""
    
    def __init__(self):
        if not HAS_TF:
            raise ImportError("tensorflow_datasets是RLDS处理所必需的，请运行: pip install tensorflow tensorflow_datasets")
        # 添加与HDF5Processor兼容的属性
        self.image_size = (256, 256)
        self.use_videos = False
    
    def get_default_features(self, use_videos: bool = True) -> Dict[str, Dict[str, Any]]:
        """获取Libero数据集的默认特征配置"""
        image_dtype = "video" if use_videos else "image"
        
        return {
            "observation.images.front": {
                "dtype": image_dtype,
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "observation.images.wrist": {
                "dtype": image_dtype,
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "observation.state": {
                "dtype": "float32",
                "shape": (8,),
                "names": [f"state_{i}" for i in range(8)],
            },
            "action": {
                "dtype": "float32",
                "shape": (8,),
                "names": [f"action_{i}" for i in range(8)],
            }
        }
    
    def process_dataset(self, dataset: LeRobotDataset, data_source: Union[str, Path]):
        """处理RLDS数据集"""
        # Libero数据集名称列表。根据自己需求修改
        raw_dataset_names = [
            "libero_10_no_noops",
            "libero_goal_no_noops", 
            "libero_object_no_noops",
            "libero_spatial_no_noops",
        ]
        
        episode_idx = 0
        
        for raw_dataset_name in raw_dataset_names:
            logger.info(f"处理RLDS数据集: {raw_dataset_name}")
            
            try:
                # 加载RLDS数据集
                raw_dataset = tfds.load(
                    raw_dataset_name, 
                    data_dir=data_source, 
                    split="train",
                    try_gcs=False
                )
                
                for episode in raw_dataset:
                    logger.info(f"处理episode {episode_idx + 1}")
                    
                    # 获取任务描述
                    steps_list = list(episode["steps"].as_numpy_iterator())
                    task_str = f"episode_{episode_idx}"
                    
                    if steps_list and "language_instruction" in steps_list[0]:
                        task_str = steps_list[0]["language_instruction"].decode()
                    
                    # 处理episode中的每个step
                    for step_idx, step in enumerate(steps_list):
                        frame_data = {
                            "observation.images.front": step["observation"]["image"],
                            "observation.images.wrist": step["observation"]["wrist_image"], 
                            "observation.state": step["observation"]["state"].astype(np.float32),
                            "action": step["action"].astype(np.float32),
                        }
                        # 修复API调用
                        dataset.add_frame(frame_data, task=task_str)
                    
                    dataset.save_episode()
                    episode_idx += 1
                    
            except Exception as e:
                logger.warning(f"处理数据集 {raw_dataset_name} 时出错: {e}")
                continue


class UnifiedConverter:
    """统一转换器类"""
    
    def __init__(self, num_workers: int = 4):
        """
        初始化统一转换器
        
        Args:
            num_workers: 并行处理的工作线程数
        """
        self.num_workers = num_workers
        self.detector = DatasetFormatDetector()
    
    def convert_dataset(
        self,
        data_dir: Union[str, Path],
        repo_id: str,
        output_dir: Optional[Union[str, Path]] = None,
        push_to_hub: bool = False,
        use_videos: bool = True,
        robot_type: str = "panda",
        fps: int = 20,
        task_name: str = "default_task",
        hub_config: Optional[Dict[str, Any]] = None,
        clean_existing: bool = True,
        image_writer_threads: int = 10,
        image_writer_processes: int = 5,
        run_compute_stats: bool = False,
        **kwargs
    ) -> LeRobotDataset:
        """
        统一转换接口
        
        Args:
            data_dir: 数据目录路径
            repo_id: 数据集仓库ID
            output_dir: 输出目录
            push_to_hub: 是否推送到Hub
            use_videos: 是否使用视频格式
            robot_type: 机器人类型
            fps: 帧率
            task_name: 任务名称（HDF5格式使用）
            hub_config: Hub配置
            clean_existing: 是否清理现有数据集
            image_writer_threads: 图像写入线程数
            image_writer_processes: 图像写入进程数
            run_compute_stats: 是否计算统计信息
            
        Returns:
            LeRobotDataset: 转换后的数据集
        """
        data_path = Path(data_dir)
        
        # 自动检测格式
        format_type = self.detector.detect_format(data_path)
        logger.info(f"检测到数据格式: {format_type}")
        
        # 根据格式选择处理器和特征
        if format_type == "hdf5":
            image_size, state_dim = self._infer_hdf5_specs(data_path)
            if image_size is None:
                image_size = (256, 256)
                logger.info("未能推断到图像尺寸，使用默认 256x256")
            else:
                logger.info(f"推断到图像尺寸: {image_size[0]}x{image_size[1]}")

            if state_dim is None:
                state_dim = 9
                logger.info("未能推断到状态维度，使用默认 9")
            else:
                logger.info(f"推断到状态维度: {state_dim}")

            processor = HDF5Processor(image_size=image_size, state_dim=state_dim)
            features = processor.get_default_features(use_videos)

        else:  # rlds
            processor = RLDSProcessor()
            features = processor.get_default_features(use_videos)
        
        # 设置输出路径
        if output_dir is None:
            lerobot_root = HF_LEROBOT_HOME
        else:
            lerobot_root = Path(output_dir)
        
        os.environ["LEROBOT_HOME"] = str(lerobot_root)
        lerobot_dataset_dir = lerobot_root / repo_id
        
        # 清理现有数据集
        if clean_existing and lerobot_dataset_dir.exists():
            logger.info(f"清理现有数据集: {lerobot_dataset_dir}")
            shutil.rmtree(lerobot_dataset_dir)
        
        lerobot_root.mkdir(parents=True, exist_ok=True)
        
        # 创建LeRobot数据集
        logger.info(f"创建LeRobot数据集: {repo_id}")
        dataset = LeRobotDataset.create(
            repo_id=repo_id,
            robot_type=robot_type,
            fps=fps,
            features=features,
            use_videos=use_videos,
            image_writer_processes=image_writer_processes,
            image_writer_threads=image_writer_threads,
        )
        
        # 处理数据
        if format_type == "hdf5":
            self._process_hdf5_data(processor, dataset, data_path, task_name)
        else:  # rlds
            # 确保是RLDSProcessor实例
            if isinstance(processor, RLDSProcessor):
                processor.process_dataset(dataset, data_path)
            else:
                raise TypeError("RLDS格式需要RLDSProcessor实例")
        
        # 移除consolidate调用，因为API可能已更改
        logger.info("数据集处理完成")
        
        # 推送到Hub
        if push_to_hub:
            if hub_config is None:
                hub_config = self._get_default_hub_config()
            logger.info("推送到Hugging Face Hub...")
            dataset.push_to_hub(**hub_config)
        
        logger.info("✅ 数据集转换完成!")
        return dataset

    def _infer_hdf5_specs(self, data_path: Path) -> Tuple[Optional[Tuple[int, int]], Optional[int]]:
        """尝试从样本episode推断图像尺寸和状态维度。"""
        image_size: Optional[Tuple[int, int]] = None
        state_dim: Optional[int] = None

        episodes: List[Path] = []
        if data_path.is_file():
            if data_path.suffix.lower() in {".hdf5", ".h5"}:
                episodes = [data_path]
            else:
                return image_size, state_dim
        else:
            for ep_dir in data_path.iterdir():
                if ep_dir.is_dir():
                    ep_path = ep_dir / "data" / "trajectory.hdf5"
                    if ep_path.exists():
                        episodes.append(ep_path)
        
        if not episodes:
            episodes = list(data_path.rglob("*.hdf5")) + list(data_path.rglob("*.h5"))
        
        for ep_path in episodes:
            try:
                with h5py.File(ep_path, "r") as file:
                    target_groups: List[h5py.Group] = []
                    if "data" in file:
                        data_group = file["data"]
                        demo_keys = [k for k in data_group.keys() if k.startswith("demo_")]
                        if not demo_keys:
                            continue
                        obs_group = data_group[demo_keys[0]].get("obs")
                        if obs_group is not None:
                            target_groups.append(obs_group)
                    else:
                        target_groups.append(file)
                    
                    for group in target_groups:
                        if state_dim is None and "joint_states" in group:
                            js = group["joint_states"]
                            state_dim = int(js.shape[-1])
                            if "gripper_states" in group:
                                gs = group["gripper_states"]
                                state_dim += int(gs.shape[-1])

                        if image_size is None:
                            for key in ("agentview_rgb", "eye_in_hand_rgb", "images", "rgb"):
                                if key in group:
                                    image_data = group[key]
                                    if image_data.ndim == 4:
                                        image_size = (int(image_data.shape[1]), int(image_data.shape[2]))
                                    elif image_data.ndim == 3:
                                        image_size = (int(image_data.shape[0]), int(image_data.shape[1]))
                                    break

                    if image_size is not None and state_dim is not None:
                        break
            except Exception as e:
                logger.debug(f"推断图像/状态信息失败 ({ep_path}): {e}")
        
        return image_size, state_dim
    
    def _process_hdf5_data(self, processor: Union[HDF5Processor, RLDSProcessor], dataset: LeRobotDataset, data_path: Path, task_name: str):
        """使用多线程处理HDF5数据"""
        # 确保是HDF5Processor
        if not isinstance(processor, HDF5Processor):
            raise TypeError("processor必须是HDF5Processor实例")
            
        # 查找所有episode
        episodes = []
        if data_path.is_file():
            if data_path.suffix.lower() in {".hdf5", ".h5"}:
                episodes = [data_path]
        else:
            for ep_dir in data_path.iterdir():
                if ep_dir.is_dir():
                    ep_path = ep_dir / "data" / "trajectory.hdf5"
                    if ep_path.exists():
                        episodes.append(ep_path)
        
        if not episodes:
            # 直接查找HDF5文件
            episodes = list(data_path.rglob("*.hdf5")) + list(data_path.rglob("*.h5"))
        
        logger.info(f"找到 {len(episodes)} 个episode文件")
        
        if self.num_workers == 1:
            # 单线程处理
            for ep_path in tqdm(episodes, desc="处理Episodes"):
                processor.process_episode(ep_path, dataset, task_name)
                logger.info(f"处理完成: {ep_path.name}")
        else:
            # 多线程处理
            self._process_episodes_parallel(processor, dataset, episodes, task_name)
    
    def _process_episodes_parallel(self, processor: HDF5Processor, dataset: LeRobotDataset, episodes: List[Path], task_name: str):
        """并行处理episodes，使用HDF5Processor的统一方法"""
        # 创建处理函数
        def process_single_episode(ep_path: Path) -> Tuple[Path, bool, List[List[Dict]]]:
            """处理单个episode并返回帧数据"""
            demos_frames = []
            try:
                with h5py.File(ep_path, "r") as file:
                    logger.debug(f"多线程处理HDF5文件键: {list(file.keys())}")
                    
                    # 使用与HDF5Processor相同的检测和处理逻辑
                    if "data" in file:
                        # 新的Libero格式：data/demo_N/...
                        demos_frames = processor._extract_libero_demo_frames(file, task_name, ep_path)
                    else:
                        # 尝试其他格式的兜底处理
                        demos_frames = processor._extract_direct_episode_frames(file, task_name, ep_path)
                    
                return ep_path, True, demos_frames
            except Exception as e:
                logger.error(f"处理 {ep_path} 失败: {e}")
                return ep_path, False, []
        
        # 并行处理
        logger.info(f"使用 {self.num_workers} 个工作线程并行处理episodes")
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            # 提交所有任务
            future_to_episode = {executor.submit(process_single_episode, ep): ep for ep in episodes}
            
            # 收集结果
            for future in tqdm(concurrent.futures.as_completed(future_to_episode), total=len(episodes), desc="处理Episodes"):
                ep_path, success, demos_frames_list = future.result()
                
                if success and demos_frames_list:
                    # 每个demo作为独立的episode保存
                    for demo_idx, demo_frames in enumerate(demos_frames_list):
                        for frame_data in demo_frames:
                            # 提取task信息
                            task_str = frame_data.pop("task", "default_task")
                            dataset.add_frame(frame_data, task=task_str)
                        dataset.save_episode()
                        logger.info(f"保存episode: {ep_path.name}_demo_{demo_idx}")
    
    def _get_default_hub_config(self) -> Dict[str, Any]:
        """获取默认Hub配置"""
        return {
            "tags": ["libero", "robotics", "lerobot", "unified"],
            "private": False,
            "push_videos": True,
            "license": "apache-2.0",
        }


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="Libero统一数据转换器 - 支持RLDS和HDF5格式自动识别",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 自动检测格式并转换
  python unified_converter.py \\
    --data-dir /path/to/data \\
    --repo-id username/dataset_name \\
    --push-to-hub \\
    --use-videos \\
    --num-workers 4

  # HDF5格式，指定配置文件
  python unified_converter.py \\
    --data-dir /path/to/hdf5/data \\
    --repo-id username/hdf5_dataset \\
    --config config.json \\
    --task-name "pick_and_place" \\
    --num-workers 8
        """
    )
    
    # 必需参数
    parser.add_argument("--data-dir", type=str, required=True, help="数据目录路径")
    parser.add_argument("--repo-id", type=str, required=True, help="数据集仓库ID")
    
    # 输出配置
    parser.add_argument("--output-dir", type=str, default=None, help="输出目录")
    parser.add_argument("--push-to-hub", action="store_true", help="推送到Hub")
    parser.add_argument("--private", action="store_true", help="创建私有数据集")
    
    # 数据格式
    parser.add_argument("--use-videos", action="store_true", default=True, help="使用视频格式")
    parser.add_argument("--robot-type", type=str, default="panda", help="机器人类型")
    parser.add_argument("--fps", type=int, default=20, help="帧率")
    
    # HDF5特定参数
    parser.add_argument("--task-name", type=str, default="default_task", help="任务名称")
    
    # 性能参数
    parser.add_argument("--num-workers", type=int, default=2, help="并行工作线程数")
    parser.add_argument("--image-writer-processes", type=int, default=5, help="图像写入进程数")
    parser.add_argument("--image-writer-threads", type=int, default=1, help="图像写入线程数")
    
    # Hub配置
    parser.add_argument("--license", type=str, default="apache-2.0", help="数据集许可证")
    parser.add_argument("--tags", nargs="+", default=["libero", "robotics", "lerobot"], help="数据集标签")
    
    # 调试选项
    parser.add_argument("--verbose", action="store_true", help="详细日志")
    parser.add_argument("--dry-run", action="store_true", help="试运行模式")
    
    args = parser.parse_args()
    
    # 设置日志级别
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # 验证参数
    if not Path(args.data_dir).exists():
        logger.error(f"数据目录不存在: {args.data_dir}")
        return 1
    
    if "/" not in args.repo_id:
        logger.error(f"repo_id格式错误: {args.repo_id}")
        return 1
    
    logger.info("📋 转换配置:")
    logger.info(f"  数据源: {args.data_dir}")
    logger.info(f"  仓库ID: {args.repo_id}")
    logger.info(f"  并行线程数: {args.num_workers}")
    logger.info(f"  使用视频: {args.use_videos}")
    logger.info(f"  推送到Hub: {args.push_to_hub}")
    
    if args.dry_run:
        logger.info("✅ 试运行完成，参数验证通过")
        return 0
    
    # 执行转换
    try:
        converter = UnifiedConverter(num_workers=args.num_workers)
        
        hub_config = {
            "tags": args.tags,
            "private": args.private,
            "license": args.license,
        }

        
        dataset = converter.convert_dataset(
            data_dir=args.data_dir,
            repo_id=args.repo_id,
            output_dir=args.output_dir,
            push_to_hub=args.push_to_hub,
            use_videos=args.use_videos,
            robot_type=args.robot_type,
            fps=args.fps,
            task_name=args.task_name,
            hub_config=hub_config,
            image_writer_processes=args.image_writer_processes,
            image_writer_threads=args.image_writer_threads,
        )
        
        logger.info("✅ 转换完成!")
        return 0
        
    except Exception as e:
        logger.error(f"转换失败: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main()) 
