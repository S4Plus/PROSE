#!/usr/bin/env python3
"""OpenPI LoRA Influence Function Calculator.

针对 OpenPI 的 LoRA 模型计算 influence 分数,专门用于数据选择和影响力分析。

这个脚本实现了四阶段流程:
1. Shadow training + 梯度缓存 (仅 LoRA 参数)
2. Rollout 处理 + return gradient 计算
3. LiSSA 逆 Hessian 向量积求解
4. Influence 评分

使用示例:
  python openpi/scripts/influence_pi05_lora.py \\
      --config-name pi05_square_d1_lora \\
      --checkpoint-dir ./checkpoints/pi05_square_d1_lora/exp1/20000 \\
      --eval-rollouts-dir ./eval_rollouts \\
      --output-path ./influence_scores.pkl
"""

from __future__ import annotations

import argparse
import base64
import dataclasses
import io
import logging
import math
import pathlib
import pickle
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import flax.nnx as nnx
import pandas as pd
from PIL import Image

# OpenPI imports
import openpi.models.model as _model
import openpi.transforms as _transforms
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.sharding as sharding
import openpi.training.weight_loaders as _weight_loaders


LOGGER = logging.getLogger("influence_pi05_lora")
MAX_PROJECTION_BYTES = int(2 * 1024 ** 3)  # 2GB


def _leaf_value(leaf: Any) -> Any:
    """Support both nnx Variable leaves and raw JAX arrays."""
    return getattr(leaf, "value", leaf)


def _replace_leaf_value(template: Any, value: Any) -> Any:
    """Support both nnx Variable leaves and raw JAX arrays."""
    replace_fn = getattr(template, "replace", None)
    if callable(replace_fn):
        return replace_fn(value=value)
    return value


# ========================================================================
# Configuration
# ========================================================================


@dataclass
class InfluenceConfig:
    """影响函数计算配置."""
    
    config_name: str
    checkpoint_dir: str
    eval_rollouts_dir: str  # eval rollout 保存目录
    output_path: str = "./influence_scores.pkl"
    
    # Data loading
    batch_size: int = 8
    num_workers: int = 0
    seed: int = 0
    
    # Projection
    proj_dim: int = 4096
    
    # LiSSA
    lissa_depth: int = 8
    lissa_samples: int = 4
    lissa_scale: float = 10.0
    lissa_damping: float = 0.01
    
    # HVP batch size (for stochastic Hessian)
    hvp_batch_size: int = 4

    # Stage 2 eval rollout chunk size. One gradient is computed per chunk instead of per step.
    stage2_batch_size: int = 32

    # Optional Stage 2 approximations on eval rollout coverage.
    max_eval_episodes: Optional[int] = None
    max_eval_steps_per_episode: Optional[int] = None

    # Stage 1 training batches to process. If None, auto-infer one epoch.
    max_train_batches: Optional[int] = None

    # Logging interval for stage progress.
    progress_every: int = 1
    
    # Optional: 保存中间结果
    save_projection: bool = False
    stage_cache_prefix: Optional[str] = None


# ========================================================================
# Eval Rollout 数据格式 & 加载
# ========================================================================


@dataclass
class EvalEpisode:
    """单个 eval episode 的数据结构."""
    
    observations: Dict[str, jax.Array]  # e.g., {"images": ..., "state": ...}
    actions: jax.Array  # [T, action_horizon, action_dim]
    reward: float  # episode 总 reward
    length: int  # episode 长度


class EvalRolloutsDataset:
    """从目录加载 eval rollout episodes.
    
    预期格式:每个 episode 保存为 episode_XXXXX.pkl,包含:
        {
            "observations": {...},  # 逐步观测
            "actions": np.ndarray,  # [T, action_dim]
            "reward": float,        # 可选,默认用 success
            "success": bool,        # 可选
        }
    """
    
    def __init__(self, rollouts_dir: pathlib.Path, action_horizon: int = 1):
        self.rollouts_dir = pathlib.Path(rollouts_dir)
        if not self.rollouts_dir.exists():
            raise ValueError(f"Rollouts 目录不存在: {self.rollouts_dir}")
        self.action_horizon = max(int(action_horizon), 1)
        
        # 加载所有 episodes
        self.episodes: List[EvalEpisode] = []
        self.episode_files = sorted(self.rollouts_dir.glob("episode_*.pkl"))
        
        if len(self.episode_files) == 0:
            # 尝试读取 Pandas pickle format (类似 ACT)
            self.episode_files = sorted(self.rollouts_dir.glob("*.pkl"))
        
        LOGGER.info(f"找到 {len(self.episode_files)} 个 eval episode 文件")
        
        for ep_file in self.episode_files:
            try:
                with open(ep_file, "rb") as f:
                    ep_data = pickle.load(f)
                
                # 尝试多种格式
                if isinstance(ep_data, pd.DataFrame):
                    # Pandas format (类似 ACT)
                    episode = self._load_from_pandas(ep_data)
                elif isinstance(ep_data, dict):
                    # Dict format
                    episode = self._load_from_dict(ep_data)
                else:
                    LOGGER.warning(f"未知 episode 格式: {ep_file}, 跳过")
                    continue
                
                self.episodes.append(episode)
            except Exception as e:
                LOGGER.warning(f"加载 {ep_file} 失败: {e}")
                continue
        
        LOGGER.info(f"成功加载 {len(self.episodes)} 个 eval episodes")

    @staticmethod
    def _expand_future_action_chunks(actions: np.ndarray, action_horizon: int) -> np.ndarray:
        """将单步执行动作重组为未来 action chunk."""
        actions = np.asarray(actions, dtype=np.float32)
        if actions.ndim != 2:
            raise ValueError(f"非法 2D action 形状: {actions.shape}")
        num_steps = int(actions.shape[0])
        if num_steps == 0:
            return np.zeros((0, action_horizon, actions.shape[-1]), dtype=np.float32)

        expanded = np.empty((num_steps, action_horizon, actions.shape[-1]), dtype=np.float32)
        for start in range(num_steps):
            end = min(start + action_horizon, num_steps)
            chunk = actions[start:end]
            chunk_len = int(chunk.shape[0])
            expanded[start, :chunk_len] = chunk
            if chunk_len < action_horizon:
                expanded[start, chunk_len:] = chunk[-1]
        return expanded

    def _normalize_episode_actions(self, actions: Any) -> np.ndarray:
        """确保 eval actions 与模型 action_horizon 对齐."""
        actions_np = np.asarray(actions, dtype=np.float32)
        if actions_np.ndim == 1:
            actions_np = actions_np[None, :]
        if actions_np.ndim == 2:
            return self._expand_future_action_chunks(actions_np, self.action_horizon)
        if actions_np.ndim == 3:
            if actions_np.shape[1] == self.action_horizon:
                return actions_np
            if actions_np.shape[1] == 1:
                return self._expand_future_action_chunks(actions_np[:, 0, :], self.action_horizon)
            if actions_np.shape[1] < self.action_horizon:
                pad = np.repeat(actions_np[:, -1:, :], self.action_horizon - actions_np.shape[1], axis=1)
                return np.concatenate([actions_np, pad], axis=1)
            return actions_np[:, : self.action_horizon, :]
        if actions_np.ndim < 1:
            raise ValueError(f"非法 action 形状: {actions_np.shape}")
        return actions_np
    
    def _load_from_pandas(self, df: pd.DataFrame) -> EvalEpisode:
        """从 Pandas DataFrame 加载 episode."""
        # 提取 observations
        obs_dict = {}
        for col in df.columns:
            if col.startswith("observation."):
                key = col.replace("observation.", "")
                values = df[col].values
                # 尝试 stack
                try:
                    if isinstance(values[0], np.ndarray):
                        obs_dict[key] = np.stack([np.asarray(v) for v in values], axis=0)
                    else:
                        obs_dict[key] = np.asarray(values)
                except Exception:
                    obs_dict[key] = np.asarray(values)
        
        # 提取 actions
        if "action" in df.columns:
            actions = np.stack([np.asarray(a) for a in df["action"].values], axis=0)
        elif "actions" in df.columns:
            actions = np.stack([np.asarray(a) for a in df["actions"].values], axis=0)
        else:
            raise ValueError("未找到 action 列")
        actions = self._normalize_episode_actions(actions)
        
        # 提取 reward
        if "reward" in df.columns:
            reward = float(df["reward"].sum())
        elif "next.reward" in df.columns:
            reward = float(df["next.reward"].sum())
        elif "success" in df.columns:
            reward = float(df["success"].iloc[-1])
        elif "next.success" in df.columns:
            reward = float(df["next.success"].iloc[-1])
        else:
            reward = 0.0
        
        return EvalEpisode(
            observations=obs_dict,
            actions=actions,
            reward=reward,
            length=len(df),
        )

    @staticmethod
    def _maybe_decode_image(image: Any) -> np.ndarray | None:
        """Decode an image field to uint8 HWC if present."""
        if image is None:
            return None
        if isinstance(image, str):
            payload = image.split(",", 1)[-1]
            try:
                image_bytes = base64.b64decode(payload)
                image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            except Exception as exc:
                raise ValueError(f"无法解码 base64 图像: {exc}") from exc

        arr = np.asarray(image)
        if arr.ndim != 3:
            return None
        if arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
            arr = np.transpose(arr, (1, 2, 0))
        if arr.dtype != np.uint8:
            if np.issubdtype(arr.dtype, np.floating):
                if float(np.nanmax(arr)) <= 1.0 and float(np.nanmin(arr)) >= 0.0:
                    arr = arr * 255.0
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        return arr

    @staticmethod
    def _first_present(mapping: Dict[str, Any], keys: Sequence[str]) -> Any:
        for key in keys:
            if key in mapping:
                return mapping[key]
        return None

    def _load_from_rows_dict(self, data: Dict[str, Any]) -> EvalEpisode:
        """Load rollout files with schema containing rows/policy_calls."""
        rows = data.get("rows")
        if not isinstance(rows, list) or len(rows) == 0:
            raise ValueError("rows 为空")

        states: List[np.ndarray] = []
        actions: List[np.ndarray] = []
        base_images: List[np.ndarray | None] = []
        wrist_images: List[np.ndarray | None] = []
        base_masks: List[bool] = []
        wrist_masks: List[bool] = []

        for row in rows:
            if not isinstance(row, dict):
                raise ValueError("rows 中存在非 dict 元素")

            action = None
            policy_output = row.get("policy_output")
            if isinstance(policy_output, dict):
                action = policy_output.get("action")
            if action is None:
                action = row.get("policy_action_chunk_raw")
            if action is None:
                action = row.get("action", row.get("action_raw"))
            if action is None:
                raise KeyError("action")
            actions.append(np.asarray(action, dtype=np.float32))

            row_obs = row.get("obs")
            state = row_obs.get("state") if isinstance(row_obs, dict) else None
            if state is None:
                policy_input = row.get("policy_input")
                if isinstance(policy_input, dict):
                    state = policy_input.get("state")
            if state is None:
                raise KeyError("state")
            states.append(np.asarray(state, dtype=np.float32))

            images_map = None
            if isinstance(row_obs, dict):
                if isinstance(row_obs.get("image"), dict):
                    images_map = row_obs["image"]
                elif isinstance(row_obs.get("images"), dict):
                    images_map = row_obs["images"]
            if images_map is None:
                policy_input = row.get("policy_input")
                if isinstance(policy_input, dict) and isinstance(policy_input.get("images"), dict):
                    images_map = policy_input["images"]

            if not isinstance(images_map, dict):
                base_images.append(None)
                wrist_images.append(None)
                base_masks.append(False)
                wrist_masks.append(False)
                continue

            base_raw = self._first_present(
                images_map,
                (
                    "observation.images.agentview",
                    "agentview",
                    "base_0_rgb",
                ),
            )
            wrist_raw = self._first_present(
                images_map,
                (
                    "observation.images.wrist",
                    "observation.images.robot0_eye_in_hand",
                    "wrist",
                    "left_wrist_0_rgb",
                ),
            )
            base_img = self._maybe_decode_image(base_raw)
            wrist_img = self._maybe_decode_image(wrist_raw)

            base_images.append(base_img)
            wrist_images.append(wrist_img)
            base_masks.append(base_img is not None)
            wrist_masks.append(wrist_img is not None)

        states_np = np.stack(states, axis=0).astype(np.float32)
        actions_np = self._normalize_episode_actions(np.stack(actions, axis=0))

        base_template = next((img for img in base_images if img is not None), None)
        wrist_template = next((img for img in wrist_images if img is not None), None)
        if base_template is None and wrist_template is None:
            raise ValueError("rows 中未找到可用图像")
        if base_template is None:
            base_template = np.zeros_like(wrist_template)
        if wrist_template is None:
            wrist_template = np.zeros_like(base_template)

        base_stack = np.stack(
            [img if img is not None else np.zeros_like(base_template) for img in base_images],
            axis=0,
        )
        wrist_stack = np.stack(
            [img if img is not None else np.zeros_like(wrist_template) for img in wrist_images],
            axis=0,
        )
        right_wrist_stack = np.zeros_like(base_stack)

        obs = {
            "state": states_np,
            "image": {
                "base_0_rgb": base_stack,
                "left_wrist_0_rgb": wrist_stack,
                "right_wrist_0_rgb": right_wrist_stack,
            },
            "image_mask": {
                "base_0_rgb": np.asarray(base_masks, dtype=np.bool_),
                "left_wrist_0_rgb": np.asarray(wrist_masks, dtype=np.bool_),
                "right_wrist_0_rgb": np.zeros(len(rows), dtype=np.bool_),
            },
        }

        if "reward" in data:
            reward = data["reward"]
            if isinstance(reward, (list, np.ndarray)):
                reward = float(np.sum(reward))
            else:
                reward = float(reward)
        elif any("reward" in r for r in rows):
            reward = float(sum(float(r.get("reward", 0.0)) for r in rows))
        else:
            reward = float(data.get("success", rows[-1].get("success", 0.0)))

        return EvalEpisode(
            observations=obs,
            actions=actions_np,
            reward=reward,
            length=len(actions_np),
        )

    def _load_from_dict(self, data: Dict[str, Any]) -> EvalEpisode:
        """从 dict 加载 episode."""
        if "rows" in data:
            return self._load_from_rows_dict(data)

        obs = {}
        for k, v in data.get("observations", {}).items():
            obs[k] = np.asarray(v)

        actions = self._normalize_episode_actions(data["actions"])
        
        reward = data.get("reward", data.get("success", 0.0))
        if isinstance(reward, (list, np.ndarray)):
            reward = float(np.sum(reward))
        else:
            reward = float(reward)
        
        length = len(actions)
        
        return EvalEpisode(
            observations=obs,
            actions=actions,
            reward=reward,
            length=length,
        )
    
    def __len__(self) -> int:
        return len(self.episodes)
    
    def __getitem__(self, idx: int) -> EvalEpisode:
        return self.episodes[idx]
    
    @property
    def episode_rewards(self) -> np.ndarray:
        """所有 episode 的 reward."""
        return np.array([ep.reward for ep in self.episodes], dtype=np.float64)
    
    @property
    def episode_lengths(self) -> np.ndarray:
        """所有 episode 的长度."""
        return np.array([ep.length for ep in self.episodes], dtype=np.int64)


# ========================================================================
# 参数管理 & 投影
# ========================================================================


class ParameterVectorizer:
    """JAX 参数展平/还原工具."""
    
    def __init__(self, params_pytree):
        """params_pytree: 从 nnx.State 中提取的可训练参数."""
        self.params_pytree = params_pytree
        # 提取所有叶子节点
        flat_params = jax.tree.leaves(params_pytree)
        self.shapes = [p.shape for p in flat_params]
        self.numels = [p.size for p in flat_params]
        self.total_dim = sum(self.numels)
        LOGGER.info(f"参数向量化: 总维度 = {self.total_dim}")
    
    def flatten(self, pytree) -> jnp.ndarray:
        """将 pytree 展平为一维向量."""
        flat_params = jax.tree.leaves(pytree)
        return jnp.concatenate([p.reshape(-1) for p in flat_params])
    
    def unflatten(self, vector: jnp.ndarray):
        """将一维向量还原为 pytree."""
        if vector.size != self.total_dim:
            raise ValueError(f"向量大小 {vector.size} 不匹配参数总维度 {self.total_dim}")
        
        slices = []
        offset = 0
        for numel, shape in zip(self.numels, self.shapes):
            slices.append(vector[offset : offset + numel].reshape(shape))
            offset += numel
        
        # 重建 pytree
        return jax.tree.unflatten(jax.tree.structure(self.params_pytree), slices)


class CountSketchProjector:
    """Count Sketch 随机投影 (内存高效)."""
    
    def __init__(self, input_dim: int, proj_dim: int, seed: int):
        self.input_dim = input_dim
        self.proj_dim = proj_dim
        self.scale = 1.0 / jnp.sqrt(float(proj_dim))
        
        rng = jax.random.key(seed)
        rng1, rng2 = jax.random.split(rng)
        
        self.indices = jax.random.randint(rng1, (input_dim,), 0, proj_dim)
        self.signs = jax.random.randint(rng2, (input_dim,), 0, 2).astype(jnp.float32) * 2.0 - 1.0
    
    def project(self, vector: jnp.ndarray) -> jnp.ndarray:
        """投影一个向量."""
        if vector.size != self.input_dim:
            raise ValueError(f"向量大小 {vector.size} 不匹配输入维度 {self.input_dim}")
        
        projected = jnp.zeros(self.proj_dim, dtype=vector.dtype)
        projected = projected.at[self.indices].add(vector * self.signs)
        return projected * self.scale
    
    def state_dict(self) -> Dict[str, np.ndarray]:
        return {
            "indices": np.asarray(self.indices),
            "signs": np.asarray(self.signs),
            "scale": np.asarray(self.scale),
            "proj_dim": np.asarray(self.proj_dim),
        }


# ========================================================================
# Influence Calculator
# ========================================================================


class InfluenceCalculator:
    """OpenPI LoRA Influence 计算器."""
    
    def __init__(self, cfg: InfluenceConfig):
        self.cfg = cfg
        
        # 设置随机种子
        np.random.seed(cfg.seed)
        
        # 加载配置和模型
        self.train_cfg = _config.get_config(cfg.config_name)
        LOGGER.info(f"加载训练配置: {cfg.config_name}")
        
        # 创建模型
        rng = jax.random.key(cfg.seed)
        self.model = self.train_cfg.model.create(rng)
        self.graphdef, self.state = nnx.split(self.model)
        
        # 加载 checkpoint
        ckpt_path = pathlib.Path(cfg.checkpoint_dir) / "params"
        if not ckpt_path.exists():
            raise ValueError(f"Checkpoint 不存在: {ckpt_path}")
        
        LOGGER.info(f"加载 checkpoint: {ckpt_path}")
        ckpt_params = _model.restore_params(str(ckpt_path))
        # replace_by_pure_dict mutates in place and returns None
        self.state.replace_by_pure_dict(ckpt_params)
        self.model = nnx.merge(self.graphdef, self.state)
        
        # 提取可训练参数 (仅 LoRA)
        all_params = nnx.state(self.model)
        trainable_state = all_params.filter(self.train_cfg.trainable_filter)
        LOGGER.info(f"可训练参数过滤完成")
        
        # 提取 value
        self.trainable_template = trainable_state
        trainable_values = jax.tree.map(_leaf_value, trainable_state)
        
        # 参数向量化
        self.vectorizer = ParameterVectorizer(trainable_values)
        self.flat_params = self.vectorizer.flatten(trainable_values)
        LOGGER.info(f"参数维度: {self.vectorizer.total_dim}")
        self._refresh_compute_kernels()
        
        # 设置投影
        self.projection_matrix = None
        self.projector = None
        
        if cfg.proj_dim > 0:
            bytes_required = self.vectorizer.total_dim * cfg.proj_dim * 4
            if bytes_required <= MAX_PROJECTION_BYTES:
                # 尝试密集投影
                try:
                    proj_rng = jax.random.key(cfg.seed + 1)
                    self.projection_matrix = jax.random.normal(
                        proj_rng, (self.vectorizer.total_dim, cfg.proj_dim)
                    ) / jnp.sqrt(cfg.proj_dim)
                    LOGGER.info(f"使用密集投影矩阵: {self.projection_matrix.shape}")
                except Exception as e:
                    LOGGER.warning(f"密集投影分配失败: {e}, 使用 CountSketch")
            
            if self.projection_matrix is None:
                self.projector = CountSketchProjector(
                    self.vectorizer.total_dim, cfg.proj_dim, cfg.seed
                )
                LOGGER.info(f"使用 CountSketch 投影: {self.vectorizer.total_dim} -> {cfg.proj_dim}")
        
        # 加载训练数据
        LOGGER.info("加载训练数据...")
        self.train_loader = self._build_train_loader()
        
        # 加载 eval rollouts
        LOGGER.info("加载 eval rollouts...")
        self.eval_dataset = EvalRolloutsDataset(
            pathlib.Path(cfg.eval_rollouts_dir),
            action_horizon=int(cfg.train_cfg.model.action_horizon),
        )
        self.eval_rewards = jnp.asarray(self.eval_dataset.episode_rewards, dtype=jnp.float32)
        self.eval_input_transforms = self._build_eval_input_transforms()
        
        # 输出路径
        self.output_path = pathlib.Path(cfg.output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Stage cache
        if cfg.stage_cache_prefix:
            self.cache_dir = pathlib.Path(cfg.stage_cache_prefix)
        else:
            self.cache_dir = self.output_path.parent
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _drop_none_entries(tree: Any) -> Any:
        if isinstance(tree, dict):
            return {k: InfluenceCalculator._drop_none_entries(v) for k, v in tree.items() if v is not None}
        return tree

    def _refresh_compute_kernels(self) -> None:
        """Build reusable gradient/HVP kernels for current trainable parameter subset."""

        def single_sample_loss(values_pytree, obs_i, act_i, rng):
            model = self._apply_trainable_values(values_pytree)
            model.train()
            obs_batched = self._drop_none_entries(jax.tree.map(lambda x: jnp.expand_dims(x, axis=0), obs_i))
            act_batched = jnp.expand_dims(act_i, axis=0)
            obs = _model.Observation.from_dict(obs_batched)
            loss = model.compute_loss(rng, obs, act_batched, train=True)
            return jnp.mean(loss)

        def chunk_loss(values_pytree, batch, rng):
            model = self._apply_trainable_values(values_pytree)
            model.train()
            batch = self._drop_none_entries(batch)
            obs = _model.Observation.from_dict(batch)
            loss = model.compute_loss(rng, obs, batch["actions"], train=True)
            return jnp.sum(jnp.mean(loss, axis=-1))

        def batch_mean_loss(values_pytree, batch, rng):
            model = self._apply_trainable_values(values_pytree)
            model.train()
            batch = self._drop_none_entries(batch)
            obs = _model.Observation.from_dict(batch)
            losses = model.compute_loss(rng, obs, batch["actions"], train=True)
            return jnp.mean(losses)

        self._stage1_batch_grad_impl = jax.vmap(
            jax.grad(single_sample_loss),
            in_axes=(None, 0, 0, None),
        )
        self._stage1_batch_grad_jit = jax.jit(self._stage1_batch_grad_impl)
        self._stage1_use_jit = True

        self._stage2_grad_impl = jax.grad(chunk_loss)
        self._stage2_grad_jit = jax.jit(self._stage2_grad_impl)
        self._stage2_use_jit = True

        def hvp_impl(values_pytree, vector_pytree, batch, rng):
            batch_mean_grad = lambda values: jax.grad(batch_mean_loss)(values, batch, rng)
            _, hvp_pytree = jax.jvp(batch_mean_grad, (values_pytree,), (vector_pytree,))
            return hvp_pytree

        self._hvp_impl = hvp_impl
        self._hvp_jit = jax.jit(hvp_impl)
        self._hvp_use_jit = True
    
    def _build_train_loader(self):
        """构建训练数据加载器."""
        # 使用 OpenPI 的 data_loader
        data_loader = _data_loader.create_data_loader(
            self.train_cfg,
            sharding=None,
            shuffle=False,
            num_batches=None,
            skip_norm_stats=False,
            framework="jax",
        )
        return data_loader

    def _infer_train_num_batches(self) -> Optional[int]:
        """Infer one-epoch batch count from OpenPI loader internals."""
        # Manual override has highest priority.
        if self.cfg.max_train_batches is not None:
            value = int(self.cfg.max_train_batches)
            if value <= 0:
                raise ValueError("--max-train-batches must be > 0")
            return value

        loader_impl = getattr(self.train_loader, "_data_loader", None)
        if loader_impl is None:
            return None

        num_batches = getattr(loader_impl, "_num_batches", None)
        if num_batches is not None:
            value = int(num_batches)
            return value if value > 0 else None

        torch_loader = getattr(loader_impl, "torch_loader", None)
        if torch_loader is not None:
            try:
                value = int(len(torch_loader))
                return value if value > 0 else None
            except Exception:
                return None

        return None

    def _normalize_train_batch(self, batch: Any) -> Dict[str, Any]:
        """Normalize loader output to dict format expected by influence code."""
        if isinstance(batch, dict):
            if "actions" not in batch:
                raise KeyError("训练 batch 缺少 actions 字段")
            return batch

        if isinstance(batch, tuple) and len(batch) == 2:
            obs, actions = batch
            if isinstance(obs, _model.Observation):
                batch_dict = obs.to_dict()
            elif isinstance(obs, dict):
                batch_dict = dict(obs)
            else:
                raise TypeError(f"不支持的 observation 类型: {type(obs)}")
            batch_dict["actions"] = actions
            return batch_dict

        raise TypeError(f"不支持的训练 batch 类型: {type(batch)}")

    def _build_eval_input_transforms(self) -> List[Any]:
        """Build the subset of training transforms that should also apply to eval rollouts."""
        data_cfg = self.train_loader.data_config()
        transforms: List[Any] = []

        # Eval rollouts in mimicgen rows format are already converted into the post-SquareInputs
        # structure, so we only keep transforms that operate on the canonical dict schema.
        for transform in data_cfg.data_transforms.inputs:
            if isinstance(transform, (_transforms.DeltaActions, _transforms.SubsampleActions)):
                transforms.append(transform)

        norm_stats = {}
        if data_cfg.repo_id != "fake":
            if data_cfg.norm_stats is None:
                raise ValueError("Eval preprocessing requires norm_stats, but none were found in train config.")
            norm_stats = data_cfg.norm_stats

        transforms.append(_transforms.Normalize(norm_stats, use_quantiles=data_cfg.use_quantile_norm))
        transforms.extend(data_cfg.model_transforms.inputs)
        return transforms

    def _clone_eval_tree(self, tree: Any) -> Any:
        """Create a cheap copy of nested eval rollout data before in-place transforms."""
        if isinstance(tree, dict):
            return {k: self._clone_eval_tree(v) for k, v in tree.items()}
        return np.asarray(tree)

    def _slice_eval_tree(self, tree: Any, index: int) -> Any:
        """Extract one sample from a batched eval tree."""
        if isinstance(tree, dict):
            return {k: self._slice_eval_tree(v, index) for k, v in tree.items()}
        return np.asarray(tree)[index]

    def _stack_eval_trees(self, trees: List[Any]) -> Any:
        """Stack a list of per-sample trees back into a batch."""
        if not trees:
            raise ValueError("Cannot stack empty eval trees.")
        first = trees[0]
        if isinstance(first, dict):
            return {k: self._stack_eval_trees([tree[k] for tree in trees]) for k in first}
        if first is None:
            return None
        return np.stack([np.asarray(tree) for tree in trees], axis=0)

    def _prepare_eval_batch(self, observations: Dict[str, Any], actions: Any) -> Dict[str, Any]:
        """Apply training-time preprocessing to an eval rollout batch."""
        observations = self._clone_eval_tree(observations)
        actions = np.asarray(actions, dtype=np.float32)
        batch_size = int(actions.shape[0])
        samples = []
        for i in range(batch_size):
            sample = self._slice_eval_tree(observations, i)
            sample["actions"] = np.asarray(actions[i], dtype=np.float32)
            for transform in self.eval_input_transforms:
                sample = transform(sample)
            samples.append(sample)

        batch = self._stack_eval_trees(samples)
        batch_size = int(np.asarray(batch["state"]).shape[0])
        batch["state"] = np.asarray(batch["state"], dtype=np.float32)
        batch["actions"] = np.asarray(batch["actions"], dtype=np.float32)
        batch["image"] = {k: np.asarray(v, dtype=np.float32) for k, v in batch["image"].items()}
        batch["image_mask"] = {k: np.asarray(v, dtype=np.bool_) for k, v in batch["image_mask"].items()}

        for key in ("tokenized_prompt", "tokenized_prompt_mask", "token_ar_mask", "token_loss_mask"):
            value = batch.get(key)
            if value is None:
                continue
            arr = np.asarray(value)
            if arr.ndim == 1:
                arr = np.broadcast_to(arr[None, :], (batch_size, arr.shape[0]))
            if key == "tokenized_prompt":
                arr = arr.astype(np.int32, copy=False)
            elif key == "token_ar_mask":
                arr = arr.astype(np.int32, copy=False)
            else:
                arr = arr.astype(np.bool_, copy=False)
            batch[key] = arr
        return jax.tree.map(jnp.asarray, batch)

    @staticmethod
    def _iter_slices(length: int, chunk_size: int) -> Iterator[tuple[int, int]]:
        """Yield contiguous [start, end) slices."""
        if chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")
        for start in range(0, length, chunk_size):
            yield start, min(start + chunk_size, length)

    def _stage2_episode_view(self) -> tuple[List[EvalEpisode], jnp.ndarray, np.ndarray]:
        """Return the eval subset actually used by Stage 2."""
        episodes = self.eval_dataset.episodes
        max_eval_episodes = self.cfg.max_eval_episodes
        if max_eval_episodes is not None:
            value = int(max_eval_episodes)
            if value <= 0:
                raise ValueError("--max-eval-episodes must be > 0")
            episodes = episodes[:value]

        max_eval_steps = self.cfg.max_eval_steps_per_episode
        if max_eval_steps is not None and int(max_eval_steps) <= 0:
            raise ValueError("--max-eval-steps-per-episode must be > 0")

        lengths = np.asarray(
            [
                min(ep.length, int(max_eval_steps)) if max_eval_steps is not None else ep.length
                for ep in episodes
            ],
            dtype=np.int64,
        )
        rewards = jnp.asarray([ep.reward for ep in episodes], dtype=jnp.float32)
        return episodes, rewards, lengths
    
    def _apply_trainable_values(self, values_pytree):
        """将可训练参数值应用到模型."""
        model = nnx.merge(self.graphdef, self.state)
        # 更新可训练参数
        updated_state = jax.tree.map(
            _replace_leaf_value,
            self.trainable_template,
            values_pytree
        )
        nnx.update(model, updated_state)
        return model

    def _flatten_batched_pytree(self, pytree_batch: Any) -> jnp.ndarray:
        """Flatten a batched pytree to [batch, dim]."""
        flat_params = jax.tree.leaves(pytree_batch)
        if not flat_params:
            return jnp.zeros((0, self.vectorizer.total_dim), dtype=jnp.float32)
        return jnp.concatenate([leaf.reshape((leaf.shape[0], -1)) for leaf in flat_params], axis=-1)

    def _project_gradients(self, grads_flat: jnp.ndarray) -> np.ndarray:
        """Project a batch of flattened gradients."""
        if self.projection_matrix is not None:
            projected = grads_flat @ self.projection_matrix
        elif self.projector is not None:
            projected = jax.vmap(self.projector.project)(grads_flat)
        else:
            projected = grads_flat
        return np.asarray(projected)
    
    def _compute_loss_for_batch(self, rng, batch, values_pytree) -> jnp.ndarray:
        """计算一个 batch 的 loss (per-sample)."""
        batch = self._normalize_train_batch(batch)
        model = self._apply_trainable_values(values_pytree)
        model.train()
        
        # 准备输入
        obs = _model.Observation.from_dict(batch)
        acts = batch["actions"]
        
        # 计算 loss
        losses = model.compute_loss(rng, obs, acts, train=True)
        return losses
    
    def _per_sample_gradients(self, batch, rng, *, batch_idx: int | None = None, total_batches: int | None = None) -> jnp.ndarray:
        """计算 batch 中每个样本的梯度 (展平)."""
        batch = self._normalize_train_batch(batch)
        batch_size = batch["actions"].shape[0]
        trainable_values = jax.tree.map(_leaf_value, self.trainable_template)
        obs_batch = self._drop_none_entries({k: v for k, v in batch.items() if k != "actions"})
        start = time.perf_counter()
        grad_fn = self._stage1_batch_grad_jit if self._stage1_use_jit else self._stage1_batch_grad_impl
        try:
            grad_pytree = grad_fn(trainable_values, obs_batch, batch["actions"], rng)
        except Exception as exc:
            if self._stage1_use_jit:
                LOGGER.warning("Stage1 batched grad JIT 失败，回退到非 JIT 路径: %s", exc)
                self._stage1_use_jit = False
                grad_pytree = self._stage1_batch_grad_impl(trainable_values, obs_batch, batch["actions"], rng)
            else:
                raise
        grads_flat = self._flatten_batched_pytree(grad_pytree)
        elapsed = time.perf_counter() - start
        if batch_idx is not None and total_batches is not None:
            LOGGER.info(
                "    [Stage1] batch %d/%d batched per-sample grad 完成 (%d samples, %.2fs)",
                batch_idx + 1,
                total_batches,
                batch_size,
                elapsed,
            )
        else:
            LOGGER.info("    [Stage1] batched per-sample grad 完成 (%d samples, %.2fs)", batch_size, elapsed)
        return grads_flat
    
    def _project_gradient(self, grad_flat: jnp.ndarray) -> np.ndarray:
        """投影梯度."""
        if self.projection_matrix is not None:
            projected = jnp.dot(grad_flat, self.projection_matrix)
        elif self.projector is not None:
            projected = self.projector.project(grad_flat)
        else:
            projected = grad_flat
        
        return np.asarray(projected)
    
    def _cache_train_gradients(self) -> np.ndarray:
        """缓存训练数据梯度 (投影后)."""
        LOGGER.info("阶段 1: 缓存训练梯度...")
        
        cached = []
        rng = jax.random.key(self.cfg.seed + 100)

        total_batches = self._infer_train_num_batches()
        if total_batches is None:
            raise RuntimeError("无法推断 Stage1 训练 batch 总数，请显式传入 --max-train-batches。")
        LOGGER.info("  Stage1 计划处理 %d 个训练 batches", total_batches)

        stage_start = time.perf_counter()
        progress_every = max(int(self.cfg.progress_every), 1)

        train_iter = iter(self.train_loader)
        for batch_idx in range(total_batches):
            fetch_start = time.perf_counter()
            LOGGER.info("  [Stage1] 正在抓取 batch %d/%d ...", batch_idx + 1, total_batches)
            try:
                batch = next(train_iter)
            except StopIteration:
                LOGGER.warning("  [Stage1] 训练数据提前结束于 batch %d/%d", batch_idx + 1, total_batches)
                break
            fetch_elapsed = time.perf_counter() - fetch_start
            LOGGER.info("  [Stage1] batch %d/%d 抓取完成, 耗时 %.2fs", batch_idx + 1, total_batches, fetch_elapsed)

            batch = self._normalize_train_batch(batch)
            batch_size = int(batch["actions"].shape[0])
            batch_start = time.perf_counter()

            if batch_idx == 0 or (batch_idx + 1) % progress_every == 0:
                LOGGER.info("  [Stage1] batch %d/%d 开始 (batch_size=%d)", batch_idx + 1, total_batches, batch_size)

            batch_grads = self._per_sample_gradients(
                batch,
                rng,
                batch_idx=batch_idx,
                total_batches=total_batches,
            )

            projected = self._project_gradients(batch_grads)
            cached.extend(projected)

            batch_elapsed = time.perf_counter() - batch_start
            elapsed = time.perf_counter() - stage_start
            done_batches = batch_idx + 1
            avg_batch_sec = elapsed / float(done_batches)
            eta_sec = max(total_batches - done_batches, 0) * avg_batch_sec
            if batch_idx == 0 or done_batches % progress_every == 0 or done_batches == total_batches:
                LOGGER.info(
                    "  [Stage1] batch %d/%d 完成, 本批 %.2fs, 累计样本 %d, ETA %.1f min",
                    done_batches,
                    total_batches,
                    batch_elapsed,
                    len(cached),
                    eta_sec / 60.0,
                )

            rng = jax.random.fold_in(rng, batch_idx)
        
        result = np.stack(cached, axis=0)
        LOGGER.info(f"阶段 1 完成: 缓存 {result.shape[0]} 个训练样本梯度")
        return result
    
    def _compute_return_gradient(self) -> jnp.ndarray:
        """计算 return gradient (基于 eval episodes)."""
        LOGGER.info("阶段 2: 计算 return gradient...")

        episodes, eval_rewards, episode_lengths = self._stage2_episode_view()
        num_episodes = len(episodes)
        if num_episodes == 0:
            LOGGER.warning("没有 eval episodes,返回零梯度")
            return jnp.zeros(self.vectorizer.total_dim, dtype=jnp.float32)

        # Centered rewards
        centered_rewards = eval_rewards - jnp.mean(eval_rewards)

        # 累积梯度
        return_grad = jnp.zeros(self.vectorizer.total_dim, dtype=jnp.float32)

        stage2_batch_size = int(self.cfg.stage2_batch_size)
        if stage2_batch_size <= 0:
            raise ValueError("--stage2-batch-size must be > 0")
        total_chunks = int(sum(math.ceil(length / stage2_batch_size) for length in episode_lengths))
        LOGGER.info(
            "  [Stage2] 使用 chunked gradient: batch_size=%d, episodes=%d, total_chunks=%d, total_steps=%d",
            stage2_batch_size,
            num_episodes,
            total_chunks,
            int(episode_lengths.sum()),
        )

        base_rng = jax.random.key(self.cfg.seed + 200)
        stage_start = time.perf_counter()
        progress_every = max(int(self.cfg.progress_every), 1)
        trainable_values = jax.tree.map(_leaf_value, self.trainable_template)
        done_chunks = 0
        
        for ep_idx, episode in enumerate(episodes):
            episode_grad = jnp.zeros(self.vectorizer.total_dim, dtype=jnp.float32)
            ep_start = time.perf_counter()
            episode_length = int(episode_lengths[ep_idx])

            for chunk_idx, (start, end) in enumerate(self._iter_slices(episode_length, stage2_batch_size)):
                obs_t = jax.tree.map(lambda x, s=start, e=end: x[s:e], episode.observations)
                act_t = episode.actions[start:end]
                batch_t = self._prepare_eval_batch(obs_t, act_t)
                chunk_rng = jax.random.fold_in(jax.random.fold_in(base_rng, ep_idx), chunk_idx)

                grad_fn = self._stage2_grad_jit if self._stage2_use_jit else self._stage2_grad_impl
                try:
                    grad_pytree = grad_fn(trainable_values, batch_t, chunk_rng)
                except Exception as exc:
                    if self._stage2_use_jit:
                        LOGGER.warning("Stage2 chunk grad JIT 失败，回退到非 JIT 路径: %s", exc)
                        self._stage2_use_jit = False
                        grad_pytree = self._stage2_grad_impl(trainable_values, batch_t, chunk_rng)
                    else:
                        raise
                grad_flat = self.vectorizer.flatten(grad_pytree)
                episode_grad += grad_flat
                done_chunks += 1
            
            # 乘上 centered reward
            return_grad += centered_rewards[ep_idx] * episode_grad
            
            done_eps = ep_idx + 1
            if ep_idx == 0 or done_eps % progress_every == 0 or done_eps == num_episodes:
                elapsed = time.perf_counter() - stage_start
                avg_ep_sec = elapsed / float(done_eps)
                eta_sec = max(num_episodes - done_eps, 0) * avg_ep_sec
                LOGGER.info(
                    "  [Stage2] episode %d/%d 完成, chunks %d/%d, 本集 %.2fs, ETA %.1f min",
                    done_eps,
                    num_episodes,
                    done_chunks,
                    total_chunks,
                    time.perf_counter() - ep_start,
                    eta_sec / 60.0,
                )
        
        # 归一化
        return_grad = return_grad / float(num_episodes)
        
        LOGGER.info(f"阶段 2 完成: return gradient norm = {jnp.linalg.norm(return_grad):.4f}")
        return return_grad
    
    def _stochastic_hvp(self, vector: jnp.ndarray, rng) -> jnp.ndarray:
        """计算随机 Hessian-vector product."""
        # 从训练数据中随机采样一个 batch
        batch = self._normalize_train_batch(next(iter(self.train_loader)))
        
        trainable_values = jax.tree.map(_leaf_value, self.trainable_template)
        
        vector_pytree = self.vectorizer.unflatten(vector)
        hvp_fn = self._hvp_jit if self._hvp_use_jit else self._hvp_impl
        try:
            hvp_pytree = hvp_fn(trainable_values, vector_pytree, batch, rng)
        except Exception as exc:
            if self._hvp_use_jit:
                LOGGER.warning("Stage3 HVP JIT 失败，回退到非 JIT 路径: %s", exc)
                self._hvp_use_jit = False
                hvp_pytree = self._hvp_impl(trainable_values, vector_pytree, batch, rng)
            else:
                raise

        hvp_flat = self.vectorizer.flatten(hvp_pytree)
        return hvp_flat
    
    def _solve_inverse_hvp(self, vector: jnp.ndarray) -> jnp.ndarray:
        """使用 LiSSA 求解逆 HVP."""
        LOGGER.info("阶段 3: 求解逆 HVP (LiSSA)...")
        
        estimate = jnp.zeros_like(vector)
        rng = jax.random.key(self.cfg.seed + 300)
        stage_start = time.perf_counter()
        progress_every = max(int(self.cfg.progress_every), 1)
        
        for sample_idx in range(self.cfg.lissa_samples):
            cur = vector
            sample_start = time.perf_counter()
            
            for depth_idx in range(self.cfg.lissa_depth):
                hvp = self._stochastic_hvp(cur, rng)
                cur = vector + (1.0 - self.cfg.lissa_damping) * cur - hvp / self.cfg.lissa_scale
                
                rng = jax.random.fold_in(rng, depth_idx)
                done_depth = depth_idx + 1
                if sample_idx == 0 and (done_depth == 1 or done_depth % progress_every == 0 or done_depth == self.cfg.lissa_depth):
                    LOGGER.info(
                        "  [Stage3] sample %d/%d depth %d/%d 完成",
                        sample_idx + 1,
                        self.cfg.lissa_samples,
                        done_depth,
                        self.cfg.lissa_depth,
                    )
            
            estimate += cur
            done_samples = sample_idx + 1
            elapsed = time.perf_counter() - stage_start
            avg_sample_sec = elapsed / float(done_samples)
            eta_sec = max(self.cfg.lissa_samples - done_samples, 0) * avg_sample_sec
            LOGGER.info(
                "  [Stage3] sample %d/%d 完成, 本sample %.2fs, ETA %.1f min",
                done_samples,
                self.cfg.lissa_samples,
                time.perf_counter() - sample_start,
                eta_sec / 60.0,
            )
            
            rng = jax.random.fold_in(rng, sample_idx + 1000)
        
        estimate = estimate / float(self.cfg.lissa_samples)
        
        LOGGER.info(f"阶段 3 完成: inverse HVP norm = {jnp.linalg.norm(estimate):.4f}")
        return estimate
    
    def _score_influences(
        self, train_projections: np.ndarray, inv_hvp: jnp.ndarray
    ) -> np.ndarray:
        """计算 influence scores."""
        LOGGER.info("阶段 4: 计算 influence scores...")
        
        # 将 inv_hvp 投影
        if self.projection_matrix is not None:
            inv_hvp_proj = jnp.dot(inv_hvp, self.projection_matrix)
        elif self.projector is not None:
            inv_hvp_proj = self.projector.project(inv_hvp)
        else:
            inv_hvp_proj = inv_hvp
        
        # 转换为 numpy
        inv_hvp_proj_np = np.asarray(inv_hvp_proj)
        
        # 计算 dot product
        scores = -np.dot(train_projections, inv_hvp_proj_np)
        
        LOGGER.info(f"阶段 4 完成: scores shape = {scores.shape}")
        return scores
    
    def run(self) -> Dict[str, Any]:
        """运行完整流程."""
        LOGGER.info("=" * 60)
        LOGGER.info("开始 OpenPI LoRA Influence 计算")
        LOGGER.info("=" * 60)
        
        # Stage 1: 训练梯度
        cache_train_path = self.cache_dir / f"{self.output_path.stem}_stage1_train_projections.npy"
        if cache_train_path.exists():
            LOGGER.info(f"从缓存加载训练梯度: {cache_train_path}")
            train_projections = np.load(cache_train_path)
        else:
            train_projections = self._cache_train_gradients()
            np.save(cache_train_path, train_projections)
            LOGGER.info(f"保存训练梯度缓存: {cache_train_path}")
        
        # Stage 2: Return gradient
        cache_return_path = self.cache_dir / f"{self.output_path.stem}_stage2_return_gradient.npy"
        if cache_return_path.exists():
            LOGGER.info(f"从缓存加载 return gradient: {cache_return_path}")
            return_grad = jnp.asarray(np.load(cache_return_path))
        else:
            return_grad = self._compute_return_gradient()
            np.save(cache_return_path, np.asarray(return_grad))
            LOGGER.info(f"保存 return gradient 缓存: {cache_return_path}")
        
        # Stage 3: Inverse HVP
        cache_inv_hvp_path = self.cache_dir / f"{self.output_path.stem}_stage3_inverse_hvp.npy"
        if cache_inv_hvp_path.exists():
            LOGGER.info(f"从缓存加载 inverse HVP: {cache_inv_hvp_path}")
            inv_hvp = jnp.asarray(np.load(cache_inv_hvp_path))
        else:
            inv_hvp = self._solve_inverse_hvp(return_grad)
            np.save(cache_inv_hvp_path, np.asarray(inv_hvp))
            LOGGER.info(f"保存 inverse HVP 缓存: {cache_inv_hvp_path}")
        
        # Stage 4: Influence scores
        influence_scores = self._score_influences(train_projections, inv_hvp)
        stage2_episodes, stage2_rewards, stage2_lengths = self._stage2_episode_view()
        
        # 构建结果
        result = {
            "config": dataclasses.asdict(self.cfg),
            "train_projections": train_projections,
            "return_gradient": np.asarray(return_grad),
            "inverse_hvp": np.asarray(inv_hvp),
            "influence_scores": influence_scores,
            "metadata": {
                "episode_rewards": np.asarray(self.eval_rewards),
                "episode_lengths": self.eval_dataset.episode_lengths,
                "stage2_num_episodes": len(stage2_episodes),
                "stage2_episode_rewards": np.asarray(stage2_rewards),
                "stage2_episode_lengths": stage2_lengths,
            },
        }
        
        if self.projection_matrix is not None and self.cfg.save_projection:
            result["projection_matrix"] = np.asarray(self.projection_matrix)
        
        if self.projector is not None and self.cfg.save_projection:
            result["projection_state"] = self.projector.state_dict()
        
        # 保存结果
        with open(self.output_path, "wb") as f:
            pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        LOGGER.info(f"结果已保存: {self.output_path}")
        LOGGER.info("=" * 60)
        LOGGER.info("计算完成!")
        LOGGER.info("=" * 60)
        
        return result


# ========================================================================
# CLI
# ========================================================================


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        force=True,
    )
    
    parser = argparse.ArgumentParser(
        description="OpenPI LoRA Influence 计算器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    parser.add_argument("--config-name", required=True, help="训练配置名称 (例如 pi05_square_d1_lora)")
    parser.add_argument("--checkpoint-dir", required=True, help="Checkpoint 目录")
    parser.add_argument("--eval-rollouts-dir", required=True, help="Eval rollouts 目录")
    parser.add_argument("--output-path", default="./influence_scores.pkl", help="输出路径")
    
    parser.add_argument("--batch-size", type=int, default=8, help="训练 batch size")
    parser.add_argument("--hvp-batch-size", type=int, default=4, help="HVP batch size")
    parser.add_argument("--stage2-batch-size", type=int, default=32, help="Stage2 eval gradient chunk size")
    parser.add_argument("--max-eval-episodes", type=int, default=None, help="Stage2 最多使用多少个 eval episodes")
    parser.add_argument("--max-eval-steps-per-episode", type=int, default=None, help="Stage2 每个 eval episode 最多使用多少步")
    parser.add_argument("--max-train-batches", type=int, default=None, help="Stage1 训练 batch 数 (默认自动推断 1 个 epoch)")
    parser.add_argument("--progress-every", type=int, default=1, help="每隔多少个 step 打印一次进度日志")
    parser.add_argument("--num-workers", type=int, default=0, help="数据加载 workers")
    parser.add_argument("--seed", type=int, default=0, help="随机种子")
    
    parser.add_argument("--proj-dim", type=int, default=4096, help="投影维度")
    parser.add_argument("--lissa-depth", type=int, default=8, help="LiSSA 深度")
    parser.add_argument("--lissa-samples", type=int, default=4, help="LiSSA 样本数")
    parser.add_argument("--lissa-scale", type=float, default=10.0, help="LiSSA scale")
    parser.add_argument("--lissa-damping", type=float, default=0.01, help="LiSSA damping")
    
    parser.add_argument("--save-projection", action="store_true", help="保存投影矩阵")
    parser.add_argument("--stage-cache-prefix", help="Stage 缓存路径前缀")
    
    args = parser.parse_args()
    
    cfg = InfluenceConfig(
        config_name=args.config_name,
        checkpoint_dir=args.checkpoint_dir,
        eval_rollouts_dir=args.eval_rollouts_dir,
        output_path=args.output_path,
        batch_size=args.batch_size,
        hvp_batch_size=args.hvp_batch_size,
        stage2_batch_size=args.stage2_batch_size,
        max_eval_episodes=args.max_eval_episodes,
        max_eval_steps_per_episode=args.max_eval_steps_per_episode,
        max_train_batches=args.max_train_batches,
        progress_every=args.progress_every,
        num_workers=args.num_workers,
        seed=args.seed,
        proj_dim=args.proj_dim,
        lissa_depth=args.lissa_depth,
        lissa_samples=args.lissa_samples,
        lissa_scale=args.lissa_scale,
        lissa_damping=args.lissa_damping,
        save_projection=args.save_projection,
        stage_cache_prefix=args.stage_cache_prefix,
    )
    
    calculator = InfluenceCalculator(cfg)
    calculator.run()


if __name__ == "__main__":
    main()
