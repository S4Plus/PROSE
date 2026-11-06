#!/usr/bin/env python3
"""保存 OpenPI eval rollouts 为标准格式.

这个脚本帮助将 OpenPI 的 rollout 数据保存为 influence 计算所需的格式。

使用示例:
  # 从 OpenPI policy server rollout
  python save_eval_rollouts.py \
      --input-dir ./raw_rollouts \
      --output-dir ./eval_rollouts \
      --format dict

  # 或者直接在代码中使用
  from save_eval_rollouts import EpisodeSaver
  
  saver = EpisodeSaver("./eval_rollouts")
  saver.add_step(observations, action)
  saver.add_step(observations, action)
  ...
  saver.save_episode(reward=1.0, episode_id=0)
"""

from __future__ import annotations

import argparse
import pathlib
import pickle
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class EpisodeBuffer:
    """临时缓存一个 episode 的数据."""
    
    observations: List[Dict[str, np.ndarray]] = field(default_factory=list)
    actions: List[np.ndarray] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    
    def add_step(
        self,
        observation: Dict[str, np.ndarray],
        action: np.ndarray,
        reward: float = 0.0,
    ):
        """添加一个 timestep."""
        self.observations.append(observation)
        self.actions.append(action)
        self.rewards.append(reward)
    
    def clear(self):
        """清空缓存."""
        self.observations.clear()
        self.actions.clear()
        self.rewards.clear()
    
    def __len__(self) -> int:
        return len(self.actions)


class EpisodeSaver:
    """保存 eval episodes 的工具类."""
    
    def __init__(self, output_dir: str | pathlib.Path, format: str = "dict"):
        """
        Args:
            output_dir: 输出目录
            format: 保存格式,支持 "dict" 或 "pandas"
        """
        self.output_dir = pathlib.Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        if format not in ["dict", "pandas"]:
            raise ValueError(f"不支持的格式: {format}, 应为 'dict' 或 'pandas'")
        self.format = format
        
        self.buffer = EpisodeBuffer()
        self.episode_count = 0
    
    def add_step(
        self,
        observation: Dict[str, Any],
        action: np.ndarray | List[float],
        reward: float = 0.0,
    ):
        """添加一个 timestep 到当前 episode.
        
        Args:
            observation: 观测字典,例如 {"images": {...}, "state": np.ndarray}
            action: 动作数组
            reward: 当前步的 reward (可选)
        """
        # 转换为 numpy
        obs_np = {}
        for k, v in observation.items():
            if isinstance(v, dict):
                obs_np[k] = {kk: np.asarray(vv) for kk, vv in v.items()}
            else:
                obs_np[k] = np.asarray(v)
        
        action_np = np.asarray(action)
        
        self.buffer.add_step(obs_np, action_np, reward)
    
    def save_episode(
        self,
        reward: Optional[float] = None,
        success: Optional[bool] = None,
        episode_id: Optional[int] = None,
    ):
        """保存当前 episode 并清空 buffer.
        
        Args:
            reward: Episode 总 reward (如果为 None,则求和 buffer 中的 rewards)
            success: 成功标志 (可选)
            episode_id: Episode ID (如果为 None,自动递增)
        """
        if len(self.buffer) == 0:
            print("警告: buffer 为空,跳过保存")
            return
        
        if episode_id is None:
            episode_id = self.episode_count
        
        if reward is None:
            reward = sum(self.buffer.rewards)
        
        # 保存
        if self.format == "dict":
            self._save_as_dict(episode_id, reward, success)
        else:
            self._save_as_pandas(episode_id, reward, success)
        
        # 清空并递增计数
        self.buffer.clear()
        self.episode_count += 1
        
        print(f"已保存 episode {episode_id}, reward={reward:.4f}, 长度={len(self.buffer)}")
    
    def _save_as_dict(self, episode_id: int, reward: float, success: Optional[bool]):
        """保存为 dict 格式."""
        # 聚合 observations
        obs_stacked = {}
        keys = self.buffer.observations[0].keys()
        
        for key in keys:
            # 检查是否为嵌套 dict
            if isinstance(self.buffer.observations[0][key], dict):
                obs_stacked[key] = {}
                sub_keys = self.buffer.observations[0][key].keys()
                for sub_key in sub_keys:
                    obs_stacked[key][sub_key] = np.stack(
                        [obs[key][sub_key] for obs in self.buffer.observations]
                    )
            else:
                obs_stacked[key] = np.stack([obs[key] for obs in self.buffer.observations])
        
        # 聚合 actions
        actions_stacked = np.stack(self.buffer.actions)
        
        # 构建数据
        data = {
            "observations": obs_stacked,
            "actions": actions_stacked,
            "reward": reward,
        }
        
        if success is not None:
            data["success"] = success
        
        # 保存
        filename = self.output_dir / f"episode_{episode_id:06d}.pkl"
        with open(filename, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    def _save_as_pandas(self, episode_id: int, reward: float, success: Optional[bool]):
        """保存为 Pandas DataFrame 格式."""
        # 构建 DataFrame
        rows = []
        
        for t in range(len(self.buffer)):
            row = {}
            
            # 添加观测
            for key, val in self.buffer.observations[t].items():
                if isinstance(val, dict):
                    for sub_key, sub_val in val.items():
                        row[f"observation.{key}.{sub_key}"] = sub_val
                else:
                    row[f"observation.{key}"] = val
            
            # 添加动作
            row["action"] = self.buffer.actions[t]
            
            # 添加 reward
            row["reward"] = self.buffer.rewards[t]
            
            rows.append(row)
        
        df = pd.DataFrame(rows)
        
        # 添加成功标志
        if success is not None:
            df["success"] = False
            df.loc[df.index[-1], "success"] = success
        
        # 保存
        filename = self.output_dir / f"episode_{episode_id:06d}.pkl"
        with open(filename, "wb") as f:
            pickle.dump(df, f, protocol=pickle.HIGHEST_PROTOCOL)


def convert_rollouts(
    input_dir: str | pathlib.Path,
    output_dir: str | pathlib.Path,
    format: str = "dict",
):
    """批量转换 rollout 文件.
    
    假设输入目录包含原始 rollout PKL 文件,尝试自动检测格式并转换。
    
    Args:
        input_dir: 输入目录
        output_dir: 输出目录
        format: 目标格式 ("dict" 或 "pandas")
    """
    input_path = pathlib.Path(input_dir)
    output_path = pathlib.Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    rollout_files = sorted(input_path.glob("*.pkl"))
    print(f"找到 {len(rollout_files)} 个 rollout 文件")
    
    saver = EpisodeSaver(output_path, format=format)
    
    for idx, rollout_file in enumerate(rollout_files):
        try:
            with open(rollout_file, "rb") as f:
                data = pickle.load(f)
            
            # 尝试解析
            if isinstance(data, pd.DataFrame):
                # 已经是 DataFrame 格式
                _convert_from_dataframe(data, saver)
            elif isinstance(data, dict):
                # Dict 格式
                _convert_from_dict(data, saver)
            else:
                print(f"警告: 未知格式 {rollout_file}, 跳过")
                continue
            
            saver.save_episode(episode_id=idx)
        
        except Exception as e:
            print(f"处理 {rollout_file} 时出错: {e}")
            continue
    
    print(f"转换完成,共 {saver.episode_count} 个 episodes")


def _convert_from_dataframe(df: pd.DataFrame, saver: EpisodeSaver):
    """从 DataFrame 转换."""
    for _, row in df.iterrows():
        obs = {}
        action = None
        reward = 0.0
        
        for col, val in row.items():
            if col.startswith("observation."):
                key = col.replace("observation.", "")
                obs[key] = val
            elif col in ["action", "actions"]:
                action = val
            elif col in ["reward", "next.reward"]:
                reward = float(val)
        
        if action is not None:
            saver.add_step(obs, action, reward)


def _convert_from_dict(data: Dict[str, Any], saver: EpisodeSaver):
    """从 dict 转换."""
    observations = data.get("observations", {})
    actions = data.get("actions", [])
    rewards = data.get("rewards", [0.0] * len(actions))
    
    # 逐步添加
    T = len(actions)
    for t in range(T):
        obs_t = {k: v[t] if isinstance(v, (list, np.ndarray)) else v for k, v in observations.items()}
        saver.add_step(obs_t, actions[t], rewards[t] if t < len(rewards) else 0.0)


def main():
    parser = argparse.ArgumentParser(
        description="转换 OpenPI rollouts 为标准格式",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument("--input-dir", required=True, help="输入目录 (原始 rollouts)")
    parser.add_argument("--output-dir", required=True, help="输出目录")
    parser.add_argument(
        "--format",
        choices=["dict", "pandas"],
        default="dict",
        help="输出格式 (dict 或 pandas)",
    )
    
    args = parser.parse_args()
    
    convert_rollouts(args.input_dir, args.output_dir, args.format)


if __name__ == "__main__":
    main()
