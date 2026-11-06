from __future__ import annotations

"""TRAK pipeline for ACT/LeRobot policies.

This script shares the same TRAK workflow as ``train_trak_diffusion.py`` but
loads ACT checkpoints (``model.safetensors`` + ``config.json``) using the
helpers from the upstream LeRobot repository.  It creates per-sample ACT
losses for the training dataset, optionally for a holdout split, and scores
evaluation rollouts stored under ``<eval_dir>/episodes``.
"""

from typing import Dict, Iterable, List, Optional, TextIO, Tuple

import sys

from pathlib import Path as _RootPath

sys.path.insert(0, str(_RootPath(__file__).resolve().parents[1]))
def _enable_line_buffering(stream: TextIO) -> None:
    """Best-effort line buffering without touching raw file descriptors."""

    try:
        stream.reconfigure(line_buffering=True)
    except (AttributeError, ValueError):  # pragma: no cover - depends on host Python
        pass


_enable_line_buffering(sys.stdout)
_enable_line_buffering(sys.stderr)

import os
import base64
import io
import json

# Reduce cuda memory fragmentation.
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import glob
import logging
import pickle
import random
import pathlib
import shutil

import click
import numpy as np
import pyarrow.parquet as pq
import torch
import trak
import tqdm
from torch.utils.data import DataLoader, Dataset, IterableDataset

from prose.compat.hf_datasets import ensure_hf_list_feature
from prose.paths import lerobot_root, repository_root

REPO_ROOT = repository_root()
LEROBOT_REPO = lerobot_root()
LEROBOT_SRC = LEROBOT_REPO / "src"
if LEROBOT_SRC.exists() and str(LEROBOT_SRC) not in sys.path:
    sys.path.insert(0, str(LEROBOT_SRC))

from lerobot.constants import ACTION
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from third_party.trak.trak.gradient_computers import AbstractGradientComputer
from third_party.trak.trak.modelout_functions import AbstractModelOutput


from lerobot.policies.act.modeling_act import ACTPolicy  # noqa: E402


IMAGE_ALIASES = {
    "head_camera": "observation.images.head_camera",
    "fixed_camera": "observation.images.fixed_camera",
    "left_camera": "observation.images.left_camera",
    "right_camera": "observation.images.right_camera",
    "cam_high": "observation.images.cam_high",
    "cam_left_wrist": "observation.images.cam_left_wrist",
    "cam_right_wrist": "observation.images.cam_right_wrist",
    "agentview": "observation.images.agentview",
    "wrist": "observation.images.wrist",
    "wrist_camera": "observation.images.wrist_camera",
    "right_wrist_camera": "observation.images.right_wrist_camera",
    "left_wrist_camera": "observation.images.left_wrist_camera",
    "front": "observation.images.front",
    "observation.images.shouldercamera0": "observation.images.cam_high",
    "observation.images.shouldercamera1": "observation.images.cam_high",
    "observation.images.robot0_eye_in_hand": "observation.images.cam_left_wrist",
    "observation.images.robot1_eye_in_hand": "observation.images.cam_right_wrist",
    "shouldercamera0": "observation.images.cam_high",
    "shouldercamera1": "observation.images.cam_high",
    "robot0_eye_in_hand": "observation.images.cam_left_wrist",
    "robot1_eye_in_hand": "observation.images.cam_right_wrist",
}
STATE_ALIASES = {
    "state": "observation.state",
    "joint_state": "observation.state",
}
ARM_ENV_STATE_MAP = {
    "left": 0.0,
    "right": 1.0,
}
IMAGE_ALIAS_GROUPS = (
    (
        "fixed_camera",
        "observation.images.fixed_camera",
    ),
    (
        "head_camera",
        "observation.images.head_camera",
        "cam_high",
        "observation.images.cam_high",
        "shouldercamera0",
        "observation.images.shouldercamera0",
        "shouldercamera1",
        "observation.images.shouldercamera1",
        "front",
        "observation.images.front",
        "agentview",
        "observation.images.agentview",
    ),
    (
        "left_camera",
        "observation.images.left_camera",
        "cam_left_wrist",
        "observation.images.cam_left_wrist",
        "robot0_eye_in_hand",
        "observation.images.robot0_eye_in_hand",
        "left_wrist_camera",
        "observation.images.left_wrist_camera",
    ),
    (
        "right_camera",
        "observation.images.right_camera",
        "cam_right_wrist",
        "observation.images.cam_right_wrist",
        "robot1_eye_in_hand",
        "observation.images.robot1_eye_in_hand",
        "right_wrist_camera",
        "observation.images.right_wrist_camera",
    ),
    (
        "wrist_camera",
        "observation.images.wrist_camera",
    ),
    (
        "wrist",
        "observation.images.wrist",
    ),
)
IMAGE_ALIAS_CANDIDATES: Dict[str, Tuple[str, ...]] = {}
for group in IMAGE_ALIAS_GROUPS:
    canonical_group = tuple(
        key if key.startswith("observation.images.") else f"observation.images.{key}"
        for key in group
    )
    for key in group:
        IMAGE_ALIAS_CANDIDATES[key] = canonical_group
        if key.startswith("observation.images."):
            IMAGE_ALIAS_CANDIDATES.setdefault(key.split(".", maxsplit=2)[-1], canonical_group)


def _cfg_feature_keys(cfg, prefix: str) -> Tuple[str, ...]:
    """提取配置中指定前缀的特征键，兼容 input_features/image_features 两种来源。"""

    keys: List[str] = []
    for attr in ("image_features", "input_features"):
        features = getattr(cfg, attr, None)
        if features is None or not hasattr(features, "keys"):
            continue
        for key in features.keys():
            if isinstance(key, str) and key.startswith(prefix):
                keys.append(key)
    return tuple(dict.fromkeys(keys))


def _resolve_batch_key(key: str, cfg) -> str:
    """按配置解析 batch 键，优先选择 checkpoint 真正声明的特征名。"""

    canonical = IMAGE_ALIASES.get(key, STATE_ALIASES.get(key, key))
    if not isinstance(canonical, str) or not canonical.startswith("observation.images."):
        return canonical

    available = set(_cfg_feature_keys(cfg, "observation.images."))
    if not available:
        return canonical

    candidates = [canonical]
    candidates.extend(IMAGE_ALIAS_CANDIDATES.get(key, ()))
    candidates.extend(IMAGE_ALIAS_CANDIDATES.get(canonical, ()))
    for candidate in dict.fromkeys(candidates):
        if candidate in available:
            return candidate
    return canonical


def _extract_singleton_value(value):
    """提取 DataLoader batch 中常见的单元素包装值。"""

    if isinstance(value, (list, tuple)) and len(value) == 1:
        return _extract_singleton_value(value[0])
    if torch.is_tensor(value) and value.numel() == 1:
        return value.item()
    if isinstance(value, np.ndarray) and value.size == 1:
        return value.reshape(-1)[0].item()
    return value


def _infer_env_state_from_meta(meta: object, cfg) -> Optional[np.ndarray]:
    """从 rollout meta 中恢复训练所需的环境条件向量。"""

    if not isinstance(meta, dict):
        return None
    episode_info = meta.get("episode_info")
    if not isinstance(episode_info, dict):
        return None

    arm_name = _extract_singleton_value(episode_info.get("{a}"))
    if arm_name is None:
        return None
    arm_name = str(arm_name).strip().lower()
    if arm_name not in ARM_ENV_STATE_MAP:
        return None

    env_dim = 1
    try:
        env_feature = cfg.input_features.get("observation.environment_state")
        if env_feature is not None:
            env_dim = int(env_feature.shape[0])
    except Exception:
        env_dim = 1

    if env_dim <= 0:
        return None
    if env_dim == 1:
        return np.asarray([ARM_ENV_STATE_MAP[arm_name]], dtype=np.float32)

    env_state = np.zeros(env_dim, dtype=np.float32)
    env_state[0 if arm_name == "left" else min(1, env_dim - 1)] = 1.0
    return env_state


def _infer_env_state_from_batch(batch: Dict[str, object], cfg) -> Optional[np.ndarray]:
    """从 batch 的多种元信息来源恢复环境条件向量。"""

    inferred = _infer_env_state_from_meta(batch.get("meta"), cfg)
    if inferred is not None:
        return inferred

    episode_info = batch.get("episode_info")
    if isinstance(episode_info, dict):
        inferred = _infer_env_state_from_meta({"episode_info": episode_info}, cfg)
        if inferred is not None:
            return inferred

    arm_name = _extract_singleton_value(batch.get("arm_id"))
    if arm_name is None:
        policy_input = batch.get("policy_input")
        if isinstance(policy_input, dict):
            arm_name = _extract_singleton_value(policy_input.get("arm_id"))

    if arm_name is None:
        return None

    arm_name = str(arm_name).strip().lower()
    if arm_name not in ARM_ENV_STATE_MAP:
        return None

    env_dim = 1
    try:
        env_feature = cfg.input_features.get("observation.environment_state")
        if env_feature is not None:
            env_dim = int(env_feature.shape[0])
    except Exception:
        env_dim = 1

    if env_dim <= 0:
        return None
    if env_dim == 1:
        return np.asarray([ARM_ENV_STATE_MAP[arm_name]], dtype=np.float32)

    env_state = np.zeros(env_dim, dtype=np.float32)
    env_state[0 if arm_name == "left" else min(1, env_dim - 1)] = 1.0
    return env_state


def _merge_observation_dict(batch: Dict[str, object], observation: object, cfg) -> None:
    """将嵌套 observation 结构展开为 prepare_act_batch 可消费的扁平键。"""

    if not isinstance(observation, dict):
        return

    images = observation.get("images")
    if isinstance(images, dict):
        for key, value in images.items():
            canonical = _resolve_batch_key(key, cfg)
            if canonical not in batch:
                batch[canonical] = value

    if "state" in observation and "state" not in batch and "observation.state" not in batch:
        batch["state"] = observation["state"]
    if "env_state" in observation and "observation.env_state" not in batch:
        batch["observation.env_state"] = observation["env_state"]

    for key, value in observation.items():
        if key in {"images", "state", "env_state"}:
            continue
        canonical = _resolve_batch_key(key, cfg)
        if not isinstance(canonical, str) or not canonical.startswith("observation.images."):
            continue
        payload = value.get("rgb") if isinstance(value, dict) and "rgb" in value else value
        if canonical not in batch:
            batch[canonical] = payload


def _lookup_required_image(mapping: Dict[str, torch.Tensor], feature_key: str) -> torch.Tensor:
    """读取模型要求的图像特征，并在等价别名之间自动回退。"""

    if feature_key in mapping:
        return mapping[feature_key]
    candidates = [feature_key]
    candidates.extend(IMAGE_ALIAS_CANDIDATES.get(feature_key, ()))
    for candidate in dict.fromkeys(candidates):
        if candidate in mapping:
            return mapping[candidate]
    available = sorted(key for key in mapping if isinstance(key, str) and key.startswith("observation.images."))
    raise KeyError(
        f"Missing required image feature '{feature_key}'. Available image keys: {available}"
    )


def _ensure_hf_cache_env(root: pathlib.Path) -> None:
    """Keep Hugging Face caches under the dataset root to avoid permission issues."""

    os.environ.setdefault("HF_HOME", str(root))
    os.environ.setdefault("HF_DATASETS_CACHE", str(root / "hf_datasets_cache"))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(root / "hub"))


def _infer_repo_id(dataset_root: pathlib.Path, cfg_repo_id: Optional[str]) -> str:
    """Infer repo_id from config or dataset path (fallback to folder name)."""

    if cfg_repo_id:
        return cfg_repo_id
    if dataset_root.is_file():
        dataset_root = dataset_root.parent
    parts = dataset_root.parts
    if len(parts) >= 2:
        return "/".join(parts[-2:])
    return dataset_root.name


def get_parameter_names(model: torch.nn.Module, keys: Iterable[str]) -> List[str]:
    """Return parameter names containing any of the provided substrings."""

    key_list = list(keys)
    param_names = list(dict(model.named_parameters()).keys())
    return sorted([name for name in param_names if any(k in name for k in key_list) and "dummy" not in name])


def load_act_policy(pretrained_dir: pathlib.Path, device: torch.device) -> Tuple[ACTPolicy, object]:
    """Load an ACT policy directory (config + safetensors)."""

    checkpoint_dir = pretrained_dir
    if checkpoint_dir.is_file():
        checkpoint_dir = checkpoint_dir.parent
    policy = ACTPolicy.from_pretrained(str(checkpoint_dir))
    policy.to(device)
    policy.eval()
    policy.reset()
    # 返回 config 供下游步骤推断图像键、delta_t 等设置。
    return policy, policy.config


def build_lerobot_dataset(dataset_path: pathlib.Path, cfg) -> LeRobotDataset:
    """Instantiate a LeRobot dataset with delta timestamps derived from the ACT config."""

    dataset_root = dataset_path.expanduser()
    if dataset_root.is_file():
        dataset_root = dataset_root.parent

    _ensure_hf_cache_env(dataset_root)
    ensure_hf_list_feature()
    repo_id = _infer_repo_id(dataset_root, getattr(cfg, "repo_id", None))
    metadata = LeRobotDatasetMetadata(repo_id=repo_id, root=dataset_root, force_cache_sync=False)

    delta_ts: Dict[str, List[float]] = {}
    if cfg.action_delta_indices is not None:
        delta_ts["action"] = [i / metadata.fps for i in cfg.action_delta_indices]
    if cfg.observation_delta_indices is not None:
        for key in metadata.image_keys:
            if key.startswith("observation."):
                delta_ts[key] = [i / metadata.fps for i in cfg.observation_delta_indices]
        if cfg.robot_state_feature is not None:
            delta_ts["observation.state"] = [i / metadata.fps for i in cfg.observation_delta_indices]

    dataset = LeRobotDataset(
        repo_id=repo_id,
        root=dataset_root,
        delta_timestamps=delta_ts or None,
        download_videos=False,
    )
    _patch_dataset_task_maps_if_needed(dataset, dataset_root)
    return dataset


def _build_runtime_task_maps(dataset_root: pathlib.Path) -> Tuple[Dict[int, str], Dict[str, int]]:
    """从 episodes.jsonl 与 parquet 中重建 task_index 映射。"""

    meta_dir = dataset_root / "meta"
    episodes_path = meta_dir / "episodes.jsonl"
    if not episodes_path.exists():
        raise FileNotFoundError(f"Missing episodes.jsonl: {episodes_path}")

    tasks_by_index: Dict[int, str] = {}
    with episodes_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            episode = json.loads(line)
            episode_index = int(episode["episode_index"])
            tasks = episode.get("tasks", [])
            if len(tasks) != 1:
                raise ValueError(
                    f"Expected exactly one task in episode {episode_index}, got {len(tasks)}."
                )

            parquet_path = dataset_root / "data" / f"chunk-{episode_index // 1000:03d}" / f"episode_{episode_index:06d}.parquet"
            if not parquet_path.exists():
                raise FileNotFoundError(f"Missing episode parquet: {parquet_path}")

            table = pq.read_table(parquet_path, columns=["task_index"])
            task_index = int(table.column("task_index")[0].as_py())
            task_text = str(tasks[0])

            previous = tasks_by_index.get(task_index)
            if previous is not None and previous != task_text:
                raise ValueError(
                    f"task_index={task_index} maps to multiple tasks: {previous!r} vs {task_text!r}"
                )
            tasks_by_index[task_index] = task_text

    task_to_task_index = {task: task_index for task_index, task in tasks_by_index.items()}
    return tasks_by_index, task_to_task_index


def _patch_dataset_task_maps_if_needed(dataset: LeRobotDataset, dataset_root: pathlib.Path) -> None:
    """仅在元信息不完整时修复 task 映射，避免 RobotWin 数据集取样报 KeyError。"""

    try:
        runtime_tasks, runtime_task_to_index = _build_runtime_task_maps(dataset_root)
    except Exception:
        return

    current_tasks = getattr(dataset.meta, "tasks", {})
    needs_patch = False
    for task_index, task_text in runtime_tasks.items():
        if current_tasks.get(task_index) != task_text:
            needs_patch = True
            break

    if not needs_patch:
        return

    dataset.meta.tasks = runtime_tasks
    dataset.meta.task_to_task_index = runtime_task_to_index


class ActTraKDataset(Dataset):
    """Lightweight wrapper that extracts ACT-relevant keys from a LeRobot dataset sample."""

    def __init__(self, dataset: LeRobotDataset, image_keys: Iterable[str]):
        self.dataset = dataset
        self.image_keys = tuple(image_keys)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.dataset[idx]
        filtered: Dict[str, torch.Tensor] = {
            "action": sample["action"],
        }
        if "action_is_pad" in sample:
            filtered["action_is_pad"] = sample["action_is_pad"]
        for key in ("observation.state", "observation.env_state", "observation.environment_state"):
            if key in sample:
                filtered[key] = sample[key]
        for key in ("task_index", "episode_index", "frame_index"):
            if key in sample:
                filtered[key] = sample[key]
        for key in self.image_keys:
            if key in sample:
                filtered[key] = sample[key]
        return filtered


class EvalPklEpisodeDataset(IterableDataset):
    """Lightweight episode reader for pkl rollouts without metadata.yaml."""

    def __init__(self, dataset_path: pathlib.Path, exec_horizon: int, sample_history: int):
        super().__init__()
        self.dataset_path = dataset_path
        self.exec_horizon = exec_horizon
        self.sample_history = sample_history

        self.episode_files = sorted(
            p for p in glob.glob(str(dataset_path / "*.pkl"))
        )
        if len(self.episode_files) == 0:
            raise FileNotFoundError(f"No pkl episodes found under {dataset_path}")

        self._episodes = [self._load_episode(p) for p in self.episode_files]
        self._episode_lengths = [len(ep) for ep in self._episodes]
        self._episode_successes = [
            self._infer_episode_success(ep, path)
            for ep, path in zip(self._episodes, self.episode_files)
        ]
        self._length = sum(self._episode_lengths)

    @staticmethod
    def _load_episode(path: str) -> "pd.DataFrame":
        with open(path, "rb") as f:
            obj = pickle.load(f)

        raw_payload = obj if isinstance(obj, dict) else None

        # Flatten {"frames": [...]} or {"rows": [...]} rollouts into a list of
        # per-step dicts while keeping scalar episode-level metadata.
        if isinstance(obj, dict):
            step_key = None
            if isinstance(obj.get("frames"), list):
                step_key = "frames"
            elif isinstance(obj.get("rows"), list):
                step_key = "rows"
            if step_key is not None:
                step_rows = obj[step_key]
                extra = {
                    k: v
                    for k, v in obj.items()
                    if k != step_key and not isinstance(v, (list, tuple, dict, np.ndarray))
                }
                if extra:
                    obj = [dict(frame, **extra) for frame in step_rows]
                else:
                    obj = step_rows

        # Helper wrappers to mimic pandas DataFrame for minimal access (len, iloc[j].to_dict()).
        class _Row:
            def __init__(self, data):
                self._data = data

            def to_dict(self):
                return dict(self._data)

        class _ILoc:
            def __init__(self, rows):
                self._rows = rows

            def __getitem__(self, idx):
                return _Row(self._rows[idx])

        class _SimpleEpisode:
            def __init__(self, rows):
                self._rows = rows
                self.iloc = _ILoc(rows)

            def __len__(self):
                return len(self._rows)

        try:
            import pandas as pd  # type: ignore
        except Exception:
            pd = None  # fallback to simple wrapper

        # If already a DataFrame, return directly
        if pd is not None and hasattr(obj, "iloc"):
            return obj

        def _seq_len(value) -> Optional[int]:
            if isinstance(value, np.ndarray):
                if value.ndim == 0:
                    return None
                return int(value.shape[0])
            if isinstance(value, (list, tuple)):
                return len(value)
            return None

        def _seq_item(value, idx: int):
            if isinstance(value, (list, tuple, np.ndarray)):
                return value[idx]
            return value

        def _rows_from_obs_actions(obs_dict: dict, actions, payload: dict) -> List[dict]:
            rows: List[dict] = []
            reward_key = "rewards" if "rewards" in payload else ("reward" if "reward" in payload else None)
            success_key = (
                "success"
                if "success" in payload
                else ("successes" if "successes" in payload else None)
            )

            lengths: List[int] = []
            action_len = _seq_len(actions)
            if action_len is not None:
                lengths.append(action_len)
            obs_lengths = [_seq_len(v) for v in obs_dict.values() if _seq_len(v) is not None]
            lengths.extend(obs_lengths)
            if reward_key is not None:
                reward_len = _seq_len(payload.get(reward_key))
                if reward_len is not None:
                    lengths.append(reward_len)
            if success_key is not None:
                success_len = _seq_len(payload.get(success_key))
                if success_len is not None:
                    lengths.append(success_len)

            length = min(lengths) if lengths else 0

            for i in range(int(length)):
                row: Dict[str, object] = {"action": _seq_item(actions, i)}

                if reward_key is not None:
                    reward_val = payload.get(reward_key)
                    if _seq_len(reward_val) is not None:
                        row["reward"] = _seq_item(reward_val, i)
                    else:
                        row["reward"] = reward_val

                if success_key is not None:
                    success_val = payload.get(success_key)
                    if _seq_len(success_val) is not None:
                        row["success"] = _seq_item(success_val, i)
                    else:
                        row["success"] = success_val

                if "observation.state" in obs_dict:
                    row["observation.state"] = _seq_item(obs_dict["observation.state"], i)
                elif "state" in obs_dict and not isinstance(obs_dict.get("state"), dict):
                    row["observation.state"] = _seq_item(obs_dict["state"], i)
                elif (
                    "robot0_joint_pos" in obs_dict
                    and "robot0_gripper_qpos" in obs_dict
                ):
                    joint = np.asarray(_seq_item(obs_dict["robot0_joint_pos"], i))
                    gripper = np.asarray(_seq_item(obs_dict["robot0_gripper_qpos"], i))
                    row["observation.state"] = np.concatenate([joint, gripper], axis=-1)
                elif (
                    "robot0_joint_pos_cos" in obs_dict
                    and "robot0_joint_pos_sin" in obs_dict
                    and "robot0_gripper_qpos" in obs_dict
                ):
                    cos = np.asarray(_seq_item(obs_dict["robot0_joint_pos_cos"], i))
                    sin = np.asarray(_seq_item(obs_dict["robot0_joint_pos_sin"], i))
                    joint = np.arctan2(sin, cos)
                    gripper = np.asarray(_seq_item(obs_dict["robot0_gripper_qpos"], i))
                    row["observation.state"] = np.concatenate([joint, gripper], axis=-1)
                elif "robot0_proprio-state" in obs_dict:
                    proprio = np.asarray(_seq_item(obs_dict["robot0_proprio-state"], i))
                    if proprio.shape[-1] >= 9:
                        row["observation.state"] = proprio[..., :9]

                # 对于 {"observations": {"images": {...}}} 这种嵌套结构，
                # 这里需要把每一帧图像展开到 row 级，后续 prepare_act_batch
                # 才能按 checkpoint 声明的图像特征名解析。
                nested_images = obs_dict.get("images")
                if isinstance(nested_images, dict):
                    for image_key, image_value in nested_images.items():
                        frame_value = _seq_item(image_value, i)
                        if isinstance(frame_value, dict) and "rgb" in frame_value:
                            frame_value = frame_value["rgb"]
                        row[image_key] = frame_value

                if "observation.images.agentview" in obs_dict:
                    row["observation.images.agentview"] = _seq_item(
                        obs_dict["observation.images.agentview"], i
                    )
                if "observation.images.wrist" in obs_dict:
                    row["observation.images.wrist"] = _seq_item(
                        obs_dict["observation.images.wrist"], i
                    )
                if "agentview_image" in obs_dict:
                    row["observation.images.agentview"] = _seq_item(
                        obs_dict["agentview_image"], i
                    )
                if "robot0_eye_in_hand_image" in obs_dict:
                    row["observation.images.wrist"] = _seq_item(
                        obs_dict["robot0_eye_in_hand_image"], i
                    )

                rows.append(row)
            return rows

        def _rows_from_mapping(mapping: dict) -> List[dict]:
            lengths = {
                key: _seq_len(value)
                for key, value in mapping.items()
                if not isinstance(value, dict)
            }
            seq_lengths = [val for val in lengths.values() if val is not None]
            if not seq_lengths:
                length = 0
            elif any(val == 0 for val in seq_lengths):
                length = 0
            else:
                # Use the shortest non-singleton sequence to avoid indexing past
                # available rollout steps when fields have mismatched lengths.
                variable_lengths = [val for val in seq_lengths if val > 1]
                length = min(variable_lengths) if variable_lengths else 1
            rows: List[dict] = []
            for i in range(int(length)):
                row = {}
                for key, value in mapping.items():
                    if isinstance(value, dict):
                        row[key] = value
                        continue
                    seq_len = lengths.get(key)
                    if seq_len is None:
                        row[key] = value
                    elif seq_len == 1 and length > 1:
                        row[key] = _seq_item(value, 0)
                    else:
                        row[key] = _seq_item(value, i)
                rows.append(row)
            return rows

        def _merge_policy_calls_into_rows(rows: List[dict], payload: dict) -> List[dict]:
            policy_calls = payload.get("policy_calls")
            if not isinstance(policy_calls, list) or not rows:
                return rows

            request_by_call_id = {}
            response_by_call_id = {}
            chunk_by_call_id = {}
            for call in policy_calls:
                if not isinstance(call, dict):
                    continue
                call_id = call.get("call_id")
                try:
                    call_id = int(call_id)
                except (TypeError, ValueError):
                    continue
                request = call.get("request")
                response = call.get("response")
                action_chunk = call.get("action_chunk_raw")
                if isinstance(request, dict):
                    request_by_call_id[call_id] = request
                if isinstance(response, dict):
                    response_by_call_id[call_id] = response
                if action_chunk is not None:
                    chunk_by_call_id[call_id] = action_chunk

            if not request_by_call_id and not response_by_call_id and not chunk_by_call_id:
                return rows

            for row in rows:
                if not isinstance(row, dict):
                    continue
                call_id = row.get("policy_call_id")
                try:
                    call_id = int(call_id)
                except (TypeError, ValueError):
                    continue
                if "policy_input" not in row and call_id in request_by_call_id:
                    row["policy_input"] = request_by_call_id[call_id]
                if "policy_output" not in row and call_id in response_by_call_id:
                    row["policy_output"] = response_by_call_id[call_id]
                if "policy_action_chunk_raw" not in row and call_id in chunk_by_call_id:
                    row["policy_action_chunk_raw"] = chunk_by_call_id[call_id]
            return rows

        rows = []
        if isinstance(obj, list):
            rows = [dict(r) for r in obj]
            if isinstance(raw_payload, dict):
                rows = _merge_policy_calls_into_rows(rows, raw_payload)
        elif isinstance(obj, dict):
            obs_key = None
            for key in ("observations", "observation", "obs"):
                if key in obj and isinstance(obj[key], dict):
                    obs_key = key
                    break
            action_key = "actions" if "actions" in obj else ("action" if "action" in obj else None)
            if obs_key is not None and action_key is not None:
                rows = _rows_from_obs_actions(obj[obs_key], obj[action_key], obj)
            else:
                rows = _rows_from_mapping(obj)
            rows = _merge_policy_calls_into_rows(rows, obj)

        if pd is not None:
            try:
                return pd.DataFrame(rows)
            except Exception:
                pass

        return _SimpleEpisode(rows)

    @staticmethod
    def _infer_episode_success(episode: "pd.DataFrame", path: str) -> bool:
        """Infer success from the dataframe or filename (epXXXX_succ/fail.pkl)."""

        def _as_bool(val):
            try:
                return bool(val.item())  # torch / numpy scalars
            except Exception:
                return bool(val)

        first_row = episode.iloc[0].to_dict() if len(episode) > 0 else {}
        last_row = episode.iloc[-1].to_dict() if len(episode) > 0 else {}

        if hasattr(episode, "columns"):
            cols = set(episode.columns)
            if "success" in cols:
                return _as_bool(episode["success"].iloc[-1])
            if "next.success" in cols:
                return _as_bool(episode["next.success"].iloc[-1])

        for key in ("success", "next.success"):
            if key in last_row:
                return _as_bool(last_row[key])
            if key in first_row:
                return _as_bool(first_row[key])

        stem = pathlib.Path(path).name.lower()
        if "fail" in stem:
            return False
        if "succ" in stem or "success" in stem:
            return True
        return bool(first_row.get("success", True))

    def __len__(self) -> int:
        return self._length

    @property
    def episode_lengths(self) -> List[int]:
        return self._episode_lengths

    @property
    def episode_successes(self) -> List[int]:
        return self._episode_successes

    def __iter__(self):
        for ep in self._episodes:
            for idx in range(
                self.exec_horizon * self.sample_history,
                len(ep),
                self.exec_horizon,
            ):
                sample = [
                    ep.iloc[j].to_dict()
                    for j in range(
                        idx - self.exec_horizon * self.sample_history,
                        idx + 1,
                        self.exec_horizon,
                    )
                ]
                if len(sample) == 1:
                    yield self._normalize_sample(sample[0])
                else:
                    yield [self._normalize_sample(item) for item in sample]

    @staticmethod
    def _prune_none_dict_entries(value):
        """Drop dict entries with ``None`` values so DataLoader collation can proceed."""

        if isinstance(value, dict):
            return {
                key: EvalPklEpisodeDataset._prune_none_dict_entries(item)
                for key, item in value.items()
                if item is not None
            }
        if isinstance(value, list):
            return [EvalPklEpisodeDataset._prune_none_dict_entries(item) for item in value]
        if isinstance(value, tuple):
            return tuple(EvalPklEpisodeDataset._prune_none_dict_entries(item) for item in value)
        return value

    @staticmethod
    def _normalize_sample(sample: Dict[str, object]) -> Dict[str, object]:
        if not isinstance(sample, dict):
            return sample
        if "frames" in sample and isinstance(sample["frames"], dict):
            sample = dict(sample["frames"])
        obs = sample.get("obs")
        if isinstance(obs, dict):
            for key, value in obs.items():
                if key not in sample:
                    sample[key] = value
            sample.pop("obs", None)
        images = sample.get("images")
        if isinstance(images, dict):
            for key, value in images.items():
                sample[key] = value
            sample.pop("images", None)
        return EvalPklEpisodeDataset._prune_none_dict_entries(sample)


def _tensor_from_array(array: np.ndarray | torch.Tensor) -> torch.Tensor:
    tensor = torch.as_tensor(array)
    if tensor.dtype == torch.uint8:
        tensor = tensor.float() / 255.0
    else:
        tensor = tensor.float()
    return tensor


def _decode_image_value(value):
    """Decode common serialized image payloads (base64 JPEG strings)."""

    def _decode_one(item):
        if not isinstance(item, str):
            return item
        text = item.strip()
        if not text:
            return item
        if text.startswith("data:image") and "," in text:
            text = text.split(",", 1)[1]
        try:
            raw = base64.b64decode(text, validate=False)
        except Exception:
            return item
        try:
            from PIL import Image  # type: ignore

            with Image.open(io.BytesIO(raw)) as img:
                return np.asarray(img.convert("RGB"))
        except Exception:
            return item

    if isinstance(value, str):
        return _decode_one(value)
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return value
        decoded = [_decode_one(item) for item in value]
        if all(isinstance(item, np.ndarray) for item in decoded):
            try:
                return np.stack(decoded, axis=0)
            except Exception:
                return decoded
        return decoded
    return value


def _format_image_tensor(tensor: torch.Tensor, expected_channels: Optional[int] = None) -> torch.Tensor:
    if tensor.dim() == 5:
        # (B, T, C, H, W) -> use most recent frame.
        tensor = tensor[:, -1, ...]
    elif tensor.dim() == 4:
        pass
    elif tensor.dim() == 3:
        tensor = tensor.unsqueeze(0)
    elif tensor.dim() == 2:
        tensor = tensor.unsqueeze(0).unsqueeze(-1)

    if tensor.dim() != 4:
        raise ValueError(f"Unexpected image tensor rank {tensor.dim()} for shape {tuple(tensor.shape)}")

    if expected_channels is not None:
        if tensor.shape[1] == expected_channels:
            return tensor
        if tensor.shape[-1] == expected_channels:
            return tensor.permute(0, 3, 1, 2)

    if tensor.shape[1] > tensor.shape[-1]:
        return tensor.permute(0, 3, 1, 2)
    return tensor


def _format_state_tensor(tensor: torch.Tensor, expected_dim: Optional[int] = None) -> torch.Tensor:
    """将状态张量整理为 `(B, D)`，兼容单样本向量与批量标量条件量。"""

    if tensor.dim() == 3:
        tensor = tensor[:, -1, :]
    elif tensor.dim() == 2:
        return tensor
    elif tensor.dim() == 1:
        # 这里最容易混淆两种语义：
        # 1. 单样本向量 `(D,)`，需要变成 `(1, D)`；
        # 2. 批量标量 `(B,)`，例如 env_state 的 shape=[1]，需要变成 `(B, 1)`。
        if expected_dim == 1:
            tensor = tensor.unsqueeze(-1)
        elif expected_dim is not None and tensor.shape[0] == expected_dim:
            tensor = tensor.unsqueeze(0)
        elif expected_dim is not None and expected_dim > 0 and tensor.numel() % expected_dim == 0:
            tensor = tensor.view(-1, expected_dim)
        else:
            tensor = tensor.unsqueeze(0)
    elif tensor.dim() == 0:
        tensor = tensor.reshape(1, 1)
    return tensor


def _format_action_and_mask(
    action: torch.Tensor,
    mask: Optional[torch.Tensor],
    chunk_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if action.dim() == 2:
        # If action is (T, A) for a single sample, keep T as horizon.
        # If action is (B, A), treat as batch with a single-step horizon.
        if action.shape[0] == chunk_size:
            action = action.unsqueeze(0)
        else:
            action = action.unsqueeze(1)
    if action.dim() != 3:
        raise ValueError(f"Unexpected action tensor shape {tuple(action.shape)}")
    batch_size, horizon, action_dim = action.shape

    if mask is None:
        mask = torch.zeros(batch_size, horizon, dtype=torch.bool, device=action.device)
    else:
        if mask.dim() == 1:
            mask = mask.unsqueeze(0)
        mask = mask.bool()

    if horizon < chunk_size:
        pad = action[:, -1:, :].expand(batch_size, chunk_size - horizon, action_dim)
        action = torch.cat([action, pad], dim=1)
        pad_mask = torch.ones(batch_size, chunk_size - horizon, dtype=torch.bool, device=action.device)
        mask = torch.cat([mask, pad_mask], dim=1)
    elif horizon > chunk_size:
        action = action[:, :chunk_size, :]
        mask = mask[:, :chunk_size]
    return action, mask


def prepare_act_batch(
    batch: Dict[str, torch.Tensor],
    cfg,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """Format heterogeneous batch dictionaries into ACT-ready tensors."""

    if "policy_input" in batch and isinstance(batch["policy_input"], dict):
        policy_input = batch["policy_input"]
        images = policy_input.get("images")
        if isinstance(images, dict):
            for key, value in images.items():
                canonical = _resolve_batch_key(key, cfg)
                if canonical not in batch:
                    batch[canonical] = value
        if "state" in policy_input and "state" not in batch and "observation.state" not in batch:
            batch["state"] = policy_input["state"]
        if "env_state" in policy_input and "observation.env_state" not in batch:
            batch["observation.env_state"] = policy_input["env_state"]

    if "obs" in batch:
        _merge_observation_dict(batch, batch.pop("obs"), cfg)

    if "observation" in batch:
        _merge_observation_dict(batch, batch.pop("observation"), cfg)

    if "observations" in batch and isinstance(batch["observations"], dict):
        rollout_obs = batch.pop("observations")
        joint_action = rollout_obs.get("joint_action")
        if isinstance(joint_action, dict) and "vector" in joint_action:
            if "state" not in batch and "observation.state" not in batch:
                batch["state"] = joint_action["vector"]
        _merge_observation_dict(batch, rollout_obs.get("observation"), cfg)

    # RobotWin 数据会把条件量存成 observation.env_state。
    # 这里优先复用真实 env_state；只有缺失时才回退到 task_index one-hot。
    if "observation.environment_state" not in batch:
        if "observation.env_state" in batch:
            batch["observation.environment_state"] = batch["observation.env_state"]
        elif "env_state" in batch:
            batch["observation.environment_state"] = batch["env_state"]
        else:
            inferred_env_state = _infer_env_state_from_batch(batch, cfg)
            if inferred_env_state is not None:
                batch["observation.environment_state"] = inferred_env_state

        env_feature = None
        try:
            env_feature = cfg.input_features.get("observation.environment_state")
        except Exception:
            env_feature = None
        if "observation.environment_state" not in batch and env_feature is not None and "task_index" in batch:
            num_tasks = int(env_feature.shape[0])
            task_index = torch.as_tensor(batch["task_index"], dtype=torch.long)
            if task_index.dim() == 0:
                task_index = task_index.unsqueeze(0)
            task_index = task_index.clamp(min=0, max=max(num_tasks - 1, 0))
            env_state = torch.zeros(task_index.shape[0], num_tasks, dtype=torch.float32)
            env_state.scatter_(1, task_index.view(-1, 1), 1.0)
            batch["observation.environment_state"] = env_state

    processed: Dict[str, torch.Tensor] = {}
    if "actions" in batch:
        action_array = batch["actions"]
    elif "action" in batch:
        action_array = batch["action"]
    else:
        raise KeyError("Batch is missing required action key ('actions' or 'action').")
    action = _tensor_from_array(action_array).to(device)
    action_is_pad = batch.get("action_is_pad")
    if action_is_pad is not None:
        action_is_pad = torch.as_tensor(action_is_pad, device=device)
    action, mask = _format_action_and_mask(action, action_is_pad, cfg.chunk_size)
    processed["action"] = action
    processed["action_is_pad"] = mask  # True 表示该时间步是 padding，loss 需屏蔽

    for key, tensor in batch.items():
        canonical = _resolve_batch_key(key, cfg)
        if canonical.startswith("observation.images"):
            image_feature = cfg.image_features.get(canonical)
            expected_channels = image_feature.shape[0] if image_feature is not None else None
            decoded = _decode_image_value(tensor)
            formatted = _format_image_tensor(_tensor_from_array(decoded), expected_channels)
            processed[canonical] = formatted.to(device)
        elif canonical in ("observation.state", "observation.environment_state"):
            feature = None
            expected_dim = None
            try:
                feature = cfg.input_features.get(canonical)
            except Exception:
                feature = None
            if feature is not None and getattr(feature, "shape", None):
                expected_dim = int(feature.shape[0])
            processed[canonical] = _format_state_tensor(
                _tensor_from_array(tensor),
                expected_dim=expected_dim,
            ).to(device)

    return processed


class ACTModelOutput(AbstractModelOutput):
    """Per-sample ACT reconstruction loss (+ optional KL term)."""

    def __init__(self):
        super().__init__()

    def get_output(
        self,
        model: ACTPolicy,
        weights,
        buffers,
        batch: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        normalized = model.normalize_inputs(batch)
        if model.config.image_features:
            normalized = dict(normalized)
            normalized["observation.images"] = [
                _lookup_required_image(normalized, key) for key in model.config.image_features
            ]

        targets = model.normalize_targets(normalized)
        actions_hat, (mu_hat, log_sigma_x2_hat) = model.model(targets)

        target_actions = targets[ACTION]
        mask = (~targets["action_is_pad"]).unsqueeze(-1).float()
        diff = target_actions - actions_hat
        #更新loss平滑函数为：SmoothHinge(s, t) = t log(1+exp( x，  t )),
        delta = 0.001#测试不同的取值对实验结果的影响，l1是差值不可微分
        scaled = diff / delta
        pseudo_huber = (delta * delta) * (torch.sqrt(1 + scaled.square()) - 1)
        l1_loss = (pseudo_huber * mask).reshape(diff.shape[0], -1).mean(dim=1)

        if model.config.use_vae and mu_hat is not None and log_sigma_x2_hat is not None:
            kld = (-0.5 * (1 + log_sigma_x2_hat - mu_hat.pow(2) - log_sigma_x2_hat.exp())).sum(dim=1)
            return l1_loss + kld * model.config.kl_weight
        return l1_loss

    def get_out_to_loss_grad(self, model, weights, buffers, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.ones(batch["action"].shape[0], device=batch["action"].device).unsqueeze(-1)


class ACTGradientComputer(AbstractGradientComputer):
    """Iterative per-sample gradients tailored for ACT policies."""

    def __init__(
        self,
        model: torch.nn.Module,
        task: AbstractModelOutput,
        grad_dim: int,
        dtype: torch.dtype,
        device: torch.device,
        grad_wrt: Optional[Iterable[str]] = None,
    ) -> None:
        super().__init__(model, task, grad_dim, dtype, device)
        self.grad_wrt = set(grad_wrt) if grad_wrt is not None else None
        self.logger = logging.getLogger("ACTGradientComputer")
        self.load_model_params(model)

    def load_model_params(self, model) -> None:
        entries = []
        autograd_tensors: List[torch.nn.Parameter] = []
        for name, param in model.named_parameters():
            if self.grad_wrt is not None and name not in self.grad_wrt:
                continue
            idx = None
            if param.requires_grad:
                idx = len(autograd_tensors)
                autograd_tensors.append(param)
            entries.append({"index": idx, "size": param.numel()})

        total_size = sum(entry["size"] for entry in entries)
        if total_size != self.grad_dim:
            raise ValueError(
                f"Gradient dimension mismatch: expected {self.grad_dim}, but found {total_size} elements."
            )
        self.param_entries = entries
        self.autograd_tensors = autograd_tensors
        self.model = model

    def compute_per_sample_grad(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        losses = self.modelout_fn.get_output(self.model, None, None, batch)
        batch_size = losses.shape[0]
        grads = torch.zeros(batch_size, self.grad_dim, device=self.device, dtype=self.dtype)

        for i in range(batch_size):
            # 逐样本构建 autograd 图，TRAK 才能学习单条示范的影响力。
            if self.autograd_tensors:
                grad_tensors = torch.autograd.grad(
                    losses[i],
                    self.autograd_tensors,
                    retain_graph=i < batch_size - 1,
                    allow_unused=True,
                )
            else:
                grad_tensors = []

            flat_chunks = []
            for entry in self.param_entries:
                idx = entry["index"]
                size = entry["size"]
                if idx is None:
                    flat_chunks.append(torch.zeros(size, device=self.device, dtype=self.dtype))
                    continue
                grad_tensor = grad_tensors[idx]
                if grad_tensor is None:
                    flat_chunks.append(torch.zeros(size, device=self.device, dtype=self.dtype))
                else:
                    flat_chunks.append(grad_tensor.reshape(-1).to(self.dtype))
            grads[i] = torch.cat(flat_chunks)
        return grads

    def compute_loss_grad(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.modelout_fn.get_out_to_loss_grad(self.model, None, None, batch)


@click.command()
# Experiment params.
@click.option("--exp_name", type=str, required=True)
@click.option("--eval_dir", type=str, required=True)
@click.option("--pretrained_dir", type=str, required=True)
@click.option("--train_dataset_path", type=str, required=True)
@click.option("--holdout_dataset_path", type=str, default=None)
@click.option("--model_keys", type=str, default=None)
# TRAK params.
@click.option("--model_id", type=int, required=True)
@click.option("--proj_dim", type=int, default=2048)
@click.option("--proj_max_batch_size", type=int, default=32)
@click.option("--lambda_reg", type=float, default=0.0)
@click.option("--use_half_precision", type=bool, default=False)
# Other params.
@click.option("--batch_size", type=int, default=4)
@click.option("--num_workers", type=int, default=4)
@click.option("--device", type=str, default="cuda:0")
@click.option("--overwrite", type=bool, default=False)
@click.option("--seed", type=int, default=0)
# Optionals.
@click.option("--featurize_holdout", type=bool, default=False)
@click.option("--finalize_scores", type=bool, default=False)
def main(
    # Experiment params.
    exp_name: str,
    eval_dir: str,
    pretrained_dir: str,
    train_dataset_path: str,
    holdout_dataset_path: Optional[str],
    model_keys: Optional[str],
    # TRAK params.
    model_id: int,
    proj_dim: int,
    proj_max_batch_size: int,
    lambda_reg: float,
    use_half_precision: bool,
    # Other params.
    batch_size: int,
    num_workers: int,
    device: str,
    overwrite: bool,
    seed: int,
    # Optionals.
    featurize_holdout: bool,
    finalize_scores: bool,
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    device_t = torch.device(device)

    pretrained_path = pathlib.Path(pretrained_dir).expanduser()
    policy, cfg = load_act_policy(pretrained_path, device_t)
    grad_wrt = None
    if model_keys:
        grad_wrt = get_parameter_names(policy, model_keys.split(","))

    base_train_set = build_lerobot_dataset(pathlib.Path(train_dataset_path), cfg)
    train_set = ActTraKDataset(base_train_set, cfg.image_features.keys())
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        num_workers=num_workers,
        persistent_workers=False,
        pin_memory=True,
        shuffle=False,
    )
    train_set_size = len(train_set)

    holdout_loader = None
    holdout_set_size = 0
    if featurize_holdout:
        if holdout_dataset_path is None:
            raise ValueError("Provide --holdout_dataset_path when --featurize_holdout=True")
        base_holdout = build_lerobot_dataset(pathlib.Path(holdout_dataset_path), cfg)
        holdout_set = ActTraKDataset(base_holdout, cfg.image_features.keys())
        holdout_loader = DataLoader(
            holdout_set,
            batch_size=batch_size,
            num_workers=num_workers,
            persistent_workers=False,
            pin_memory=True,
            shuffle=False,
        )
        holdout_set_size = len(holdout_set)

    task = ACTModelOutput()
    save_dir = pathlib.Path(eval_dir).expanduser() / exp_name
    if save_dir.exists():
        if not overwrite:
            raise ValueError(f"Output path {save_dir} already exists. Use --overwrite=1 to replace it.")
        shutil.rmtree(save_dir)

    # 初始化 TRAKer：注册模型、梯度计算器和随机投影，用于后续影响分数估计
    #随机投影：降维
    traker = trak.TRAKer(
        model=policy,
        task=task,
        train_set_size=train_set_size + holdout_set_size,
        gradient_computer=ACTGradientComputer,
        device=device_t,
        grad_wrt=grad_wrt,
        proj_dim=proj_dim,
        proj_max_batch_size=proj_max_batch_size,
        lambda_reg=lambda_reg,
        save_dir=str(save_dir),
        use_half_precision=use_half_precision,
    )
    traker.load_checkpoint(policy.state_dict(), model_id=model_id)

    def _featurize_loader(loader: Optional[DataLoader], desc: str) -> None:
        if loader is None:
            return
        for batch in tqdm.tqdm(loader, desc=desc):
            formatted = prepare_act_batch(batch, cfg, device_t)
            num_samples = formatted["action"].shape[0]
            # 将示范样本编码为低秩特征，作为 g_train。
            traker.featurize(formatted, num_samples=num_samples)

    _featurize_loader(train_loader, "Featurizing train set")
    _featurize_loader(holdout_loader, "Featurizing holdout set")
    hessian_lim = train_set_size if holdout_loader is not None else None
    # finalize_features 会求解/存储近似的 H^{-1} 投影基，用于影响函数的闭式计算。
    traker.finalize_features(model_ids=[model_id], hessian_lim=hessian_lim)

    eval_path = pathlib.Path(eval_dir).expanduser() / "episodes"
    test_set = EvalPklEpisodeDataset(
        dataset_path=eval_path,
        exec_horizon=1,
        sample_history=0,
    )
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)
    test_set_size = len(test_set)

    traker.start_scoring_checkpoint(
        exp_name="all_episodes",
        checkpoint=policy.state_dict(),
        model_id=model_id,
        num_targets=test_set_size,
    )
    for batch in tqdm.tqdm(test_loader, desc="Scoring test set"):
        formatted = prepare_act_batch(batch, cfg, device_t)
        num_samples = formatted["action"].shape[0]
        # 用测试 rollout 的梯度投影到训练特征空间，得到 g_eval ⋅ Φ_train。
        traker.score(batch=formatted, num_samples=num_samples)

    print(f"Completed featurizing traker for {pretrained_path}")
    print(f"Results saved to {save_dir}")

    if finalize_scores:
        # 组合训练/评估特征，输出最终的影响矩阵 (train_samples × test_eps)。
        scores = np.array(
            traker.finalize_scores(
                exp_name="all_episodes",
                model_ids=[model_id],
                allow_skip=False,
            )
        )
        assert scores.shape == (train_set_size + holdout_set_size, test_set_size)
        print(f"Completed scoring for traker at {save_dir}")


if __name__ == "__main__":
    main()
