"""
Helpers for fine-tuning OpenVLA on LeRobot-format datasets.
"""

import dataclasses
import json
import logging
import math
import os
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn as nn
import tqdm
from huggingface_hub import snapshot_download
from peft import LoraConfig, PeftModel, get_peft_model
from PIL import Image
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor
from transformers.modeling_outputs import CausalLMOutputWithPast

from experiments.robot.openvla_utils import check_model_logic_mismatch, model_is_on_hf_hub, update_auto_map
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from prismatic.models.action_heads import L1RegressionActionHead
from prismatic.models.backbones.llm.prompting import PurePromptBuilder
from prismatic.models.projectors import ProprioProjector
from prismatic.training.train_utils import get_current_action_mask, get_next_actions_mask
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics


LOGGER = logging.getLogger(__name__)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _stringify_paths(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {k: _stringify_paths(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_stringify_paths(v) for v in value]
    return value


def _find_single_file(directory: Path, prefix: str) -> Optional[Path]:
    matches = sorted(directory.glob(f"{prefix}--*_checkpoint.pt"))
    if not matches:
        return None
    if len(matches) > 1:
        raise ValueError(f"Expected exactly one {prefix} checkpoint in {directory}, found {len(matches)}.")
    return matches[0]


def _bounds_q99_normalize(values: np.ndarray, stats: dict[str, Any]) -> np.ndarray:
    q01 = np.asarray(stats["q01"], dtype=np.float32)
    q99 = np.asarray(stats["q99"], dtype=np.float32)
    denom = np.where(np.abs(q99 - q01) > 1e-6, q99 - q01, 1.0)
    out = 2.0 * ((values - q01) / denom) - 1.0
    return np.clip(out, -1.0, 1.0).astype(np.float32)


def _patch_runtime_vla_constants(*, action_dim: int, proprio_dim: int, action_chunk_size: int) -> None:
    module_names: set[str] = set()

    for module_name in list(sys.modules):
        if "modeling_prismatic" in module_name:
            module_names.add(module_name)
        elif module_name.endswith(".action_heads") or module_name.endswith(".constants") or module_name.endswith(".train_utils"):
            module_names.add(module_name)

    patched_modules = []
    for module_name in sorted(module_names):
        module = sys.modules.get(module_name)
        if module is None:
            continue
        changed = False
        if hasattr(module, "ACTION_DIM"):
            setattr(module, "ACTION_DIM", int(action_dim))
            changed = True
        if hasattr(module, "PROPRIO_DIM"):
            setattr(module, "PROPRIO_DIM", int(proprio_dim))
            changed = True
        if hasattr(module, "NUM_ACTIONS_CHUNK"):
            setattr(module, "NUM_ACTIONS_CHUNK", int(action_chunk_size))
            changed = True
        if changed:
            patched_modules.append(module_name)

    LOGGER.info(
        "Patched LeRobot/OpenVLA runtime constants: ACTION_DIM=%d PROPRIO_DIM=%d NUM_ACTIONS_CHUNK=%d modules=%s",
        action_dim,
        proprio_dim,
        action_chunk_size,
        patched_modules,
    )


def _normal_normalize(values: np.ndarray, stats: dict[str, Any]) -> np.ndarray:
    mean = np.asarray(stats["mean"], dtype=np.float32)
    std = np.asarray(stats["std"], dtype=np.float32)
    denom = np.where(std > 1e-6, std, 1.0)
    return ((values - mean) / denom).astype(np.float32)


def _compute_stats(values: np.ndarray) -> dict[str, list[float]]:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"Expected a 2D array, got shape {values.shape}")
    return {
        "mean": values.mean(axis=0).astype(np.float32).tolist(),
        "std": values.std(axis=0).astype(np.float32).tolist(),
        "q01": np.quantile(values, 0.01, axis=0).astype(np.float32).tolist(),
        "q99": np.quantile(values, 0.99, axis=0).astype(np.float32).tolist(),
    }


@dataclass(frozen=True)
class LeRobotSampleRecord:
    episode_index: int
    frame_index: int
    global_index: int


@dataclass
class LeRobotEpisodePayload:
    actions: np.ndarray
    states: np.ndarray
    frame_indices: np.ndarray
    global_indices: np.ndarray
    task_indices: np.ndarray


@dataclass
class OpenVLALeRobotConfig:
    vla_path: str = "openvla/openvla-7b"
    dataset_root: Path = Path("/workspace/liushuo/cupid/datasets/lerobot/can_mh")
    run_root_dir: Path = Path("runs")
    output_dir: Optional[Path] = None
    resume_checkpoint_dir: Optional[Path] = None
    resume_step: Optional[int] = None
    dataset_name: str = "can_mh"
    task_description: Optional[str] = None
    sample_indices_path: Optional[Path] = None
    norm_stats_path: Optional[Path] = None
    action_chunk_size: int = 8
    batch_size: int = 4
    grad_accumulation_steps: int = 1
    max_steps: int = 10_000
    learning_rate: float = 5e-4
    lr_warmup_steps: int = 0
    num_steps_before_decay: int = 8_000
    save_freq: int = 1_000
    shuffle: bool = True
    num_workers: int = 0
    seed: int = 7
    use_lora: bool = True
    lora_rank: int = 32
    lora_dropout: float = 0.0
    merge_lora_during_training: bool = False
    num_images_in_input: int = 2
    use_proprio: bool = True
    use_wandb: bool = False
    wandb_entity: Optional[str] = None
    wandb_project: Optional[str] = None
    run_id_note: Optional[str] = None
    save_latest_checkpoint_only: bool = False


class LeRobotDatasetMetadata:
    def __init__(
        self,
        dataset_root: Path,
        *,
        dataset_name: Optional[str] = None,
        task_description_override: Optional[str] = None,
        norm_stats_path: Optional[Path] = None,
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.dataset_name = dataset_name or self.dataset_root.name
        self.info = _read_json(self.dataset_root / "meta" / "info.json")
        self.episodes = _read_jsonl(self.dataset_root / "meta" / "episodes.jsonl")
        self.tasks = _read_jsonl(self.dataset_root / "meta" / "tasks.jsonl")
        self.norm_stats_source = Path(norm_stats_path) if norm_stats_path is not None else self.dataset_root / "norm_stats.json"
        self.norm_stats = _read_json(self.norm_stats_source)["norm_stats"]
        self.action_stats = self.norm_stats["actions"]
        self.state_stats = self.norm_stats["state"]
        self.action_dim = int(self.info["features"]["action"]["shape"][0])
        self.state_dim = int(self.info["features"]["observation.state"]["shape"][0])
        self.total_episodes = int(self.info["total_episodes"])
        self.total_frames = int(self.info["total_frames"])
        self.primary_image_key = self._pick_primary_image_key()
        self.wrist_image_key = self._pick_wrist_image_key()
        self.task_map = {int(row["task_index"]): row["task"] for row in self.tasks}
        self.default_task_description = task_description_override or self.task_map.get(0, self.dataset_name)
        self.sample_records = self._build_sample_records()
        self._maybe_fix_action_stats()
        self.dataset_statistics = self._build_dataset_statistics()

    def _pick_primary_image_key(self) -> str:
        features = self.info["features"]
        preferred = [
            "observation.images.agentview",
            "observation.images.image_primary",
            "observation.images.primary",
        ]
        for key in preferred:
            if key in features:
                return key
        candidates = sorted(k for k in features if k.startswith("observation.images."))
        if not candidates:
            raise ValueError(f"No image feature found in {self.dataset_root}")
        return candidates[0]

    def _pick_wrist_image_key(self) -> Optional[str]:
        features = self.info["features"]
        preferred = [
            "observation.images.robot0_eye_in_hand",
            "observation.images.wrist",
            "observation.images.left_wrist",
        ]
        for key in preferred:
            if key in features:
                return key
        candidates = sorted(
            k for k in features if k.startswith("observation.images.") and k != self.primary_image_key
        )
        return candidates[0] if candidates else None

    def _build_sample_records(self) -> list[LeRobotSampleRecord]:
        records: list[LeRobotSampleRecord] = []
        running_index = 0
        for row in self.episodes:
            episode_index = int(row["episode_index"])
            length = int(row["length"])
            for frame_index in range(length):
                records.append(
                    LeRobotSampleRecord(
                        episode_index=episode_index,
                        frame_index=frame_index,
                        global_index=running_index + frame_index,
                    )
                )
            running_index += length
        return records

    def _build_dataset_statistics(self) -> dict[str, Any]:
        return {
            self.dataset_name: {
                "action": {k: np.asarray(v, dtype=np.float32) for k, v in self.action_stats.items()},
                "proprio": {k: np.asarray(v, dtype=np.float32) for k, v in self.state_stats.items()},
                "num_trajectories": self.total_episodes,
                "num_transitions": self.total_frames,
            }
        }

    def _action_stats_look_mismatched(self) -> bool:
        action_q01 = np.asarray(self.action_stats.get("q01"), dtype=np.float32)
        action_q99 = np.asarray(self.action_stats.get("q99"), dtype=np.float32)
        state_q01 = np.asarray(self.state_stats.get("q01"), dtype=np.float32)
        state_q99 = np.asarray(self.state_stats.get("q99"), dtype=np.float32)
        if (
            action_q01.ndim != 1
            or action_q99.ndim != 1
            or state_q01.ndim != 1
            or state_q99.ndim != 1
            or action_q01.size != self.action_dim
            or action_q99.size != self.action_dim
            or state_q01.size != self.state_dim
            or state_q99.size != self.state_dim
            or self.action_dim < 2
            or self.state_dim < self.action_dim
        ):
            return False

        joint_action_span = float(max(np.max(np.abs(action_q01[:-1])), np.max(np.abs(action_q99[:-1]))))
        joint_state_span = float(max(np.max(np.abs(state_q01[:-2])), np.max(np.abs(state_q99[:-2]))))
        # For absolute-joint datasets, the first action_dim-1 coordinates should be on the same scale as
        # the joint-position portion of the proprio state. If action stats are dramatically smaller, the
        # saved action normalization is likely wrong.
        return joint_state_span > 1.5 and joint_action_span < 1.5 and joint_action_span < 0.5 * joint_state_span

    def _load_all_actions(self) -> np.ndarray:
        import pyarrow.parquet as pq

        actions: list[np.ndarray] = []
        for row in self.episodes:
            episode_index = int(row["episode_index"])
            table = pq.read_table(self.parquet_path(episode_index), columns=["action"])
            actions.append(np.asarray(table.column("action").to_pylist(), dtype=np.float32))
        if not actions:
            raise ValueError(f"No actions found under {self.dataset_root}")
        return np.concatenate(actions, axis=0)

    def _maybe_fix_action_stats(self) -> None:
        if not self._action_stats_look_mismatched():
            return
        actions = self._load_all_actions()
        fixed_stats = _compute_stats(actions)
        LOGGER.warning(
            "Detected mismatched action normalization stats in %s; recomputing action stats from parquet data.",
            self.norm_stats_source,
        )
        self.action_stats = {k: np.asarray(v, dtype=np.float32) for k, v in fixed_stats.items()}

    def parquet_path(self, episode_index: int) -> Path:
        return self.dataset_root / "data" / f"chunk-{episode_index // 1000:03d}" / f"episode_{episode_index:06d}.parquet"

    def video_path(self, episode_index: int, video_key: str) -> Path:
        return (
            self.dataset_root
            / "videos"
            / f"chunk-{episode_index // 1000:03d}"
            / video_key
            / f"episode_{episode_index:06d}.mp4"
        )

    def normalize_actions(self, actions: np.ndarray) -> np.ndarray:
        return _bounds_q99_normalize(np.asarray(actions, dtype=np.float32), self.action_stats)

    def normalize_state(self, state: np.ndarray) -> np.ndarray:
        return _bounds_q99_normalize(np.asarray(state, dtype=np.float32), self.state_stats)

    @lru_cache(maxsize=32)
    def load_episode_payload(self, episode_index: int) -> LeRobotEpisodePayload:
        import pyarrow.parquet as pq

        table = pq.read_table(
            self.parquet_path(episode_index),
            columns=["action", "observation.state", "frame_index", "index", "task_index"],
        )
        data = table.to_pydict()
        return LeRobotEpisodePayload(
            actions=np.asarray(data["action"], dtype=np.float32),
            states=np.asarray(data["observation.state"], dtype=np.float32),
            frame_indices=np.asarray(data["frame_index"], dtype=np.int64).reshape(-1),
            global_indices=np.asarray(data["index"], dtype=np.int64).reshape(-1),
            task_indices=np.asarray(data["task_index"], dtype=np.int64).reshape(-1),
        )


@lru_cache(maxsize=8)
def _load_video_frames(video_path_str: str) -> np.ndarray:
    frames = imageio.mimread(video_path_str, memtest=False)
    if not frames:
        raise ValueError(f"No frames decoded from {video_path_str}")
    return np.stack([np.asarray(frame, dtype=np.uint8) for frame in frames], axis=0)


class LeRobotOpenVLADataset(Dataset):
    def __init__(
        self,
        metadata: LeRobotDatasetMetadata,
        *,
        action_tokenizer: ActionTokenizer,
        base_tokenizer: Any,
        image_transform: Any,
        action_chunk_size: int,
        num_images_in_input: int,
        use_proprio: bool,
        sample_indices_path: Optional[Path] = None,
        task_description: Optional[str] = None,
    ) -> None:
        self.metadata = metadata
        self.action_tokenizer = action_tokenizer
        self.base_tokenizer = base_tokenizer
        self.image_transform = image_transform
        self.action_chunk_size = int(action_chunk_size)
        self.num_images_in_input = int(num_images_in_input)
        self.use_proprio = bool(use_proprio)
        self.task_description_override = task_description
        self.dataset_statistics = metadata.dataset_statistics
        self.sample_records = self._filter_sample_records(sample_indices_path)
        self.neutral_action = np.zeros((self.metadata.action_dim,), dtype=np.float32)

    def _filter_sample_records(self, sample_indices_path: Optional[Path]) -> list[LeRobotSampleRecord]:
        if sample_indices_path is None:
            return list(self.metadata.sample_records)
        keep = set(np.load(sample_indices_path).astype(np.int64).reshape(-1).tolist())
        return [record for record in self.metadata.sample_records if record.global_index in keep]

    def __len__(self) -> int:
        return len(self.sample_records)

    def _task_description(self, task_index: int) -> str:
        if self.task_description_override is not None:
            return self.task_description_override
        return self.metadata.task_map.get(task_index, self.metadata.default_task_description)

    def _load_image(self, episode_index: int, frame_index: int, video_key: str) -> np.ndarray:
        frames = _load_video_frames(str(self.metadata.video_path(episode_index, video_key)))
        if frame_index >= frames.shape[0]:
            raise IndexError(f"Frame {frame_index} is out of range for {video_key} in episode {episode_index}")
        return frames[frame_index]

    def _build_action_chunk(self, actions: np.ndarray, frame_index: int) -> np.ndarray:
        end = min(frame_index + self.action_chunk_size, actions.shape[0])
        chunk = np.asarray(actions[frame_index:end], dtype=np.float32)
        if chunk.shape[0] < self.action_chunk_size:
            pad = np.repeat(self.neutral_action[None, :], self.action_chunk_size - chunk.shape[0], axis=0)
            chunk = np.concatenate([chunk, pad], axis=0)
        return chunk

    def __getitem__(self, idx: int) -> dict[str, Any]:
        record = self.sample_records[idx]
        payload = self.metadata.load_episode_payload(record.episode_index)
        task_index = int(payload.task_indices[record.frame_index]) if payload.task_indices.size else 0
        state = payload.states[record.frame_index]
        action_chunk = self._build_action_chunk(payload.actions, record.frame_index)

        normalized_state = self.metadata.normalize_state(state)
        normalized_actions = self.metadata.normalize_actions(action_chunk)

        prompt_builder = PurePromptBuilder("openvla")
        current_action_string = self.action_tokenizer(normalized_actions[0])
        future_actions_string = "".join(self.action_tokenizer(normalized_actions[1:]))
        action_chunk_string = current_action_string + future_actions_string
        prompt_builder.add_turn(
            "human",
            f"What action should the robot take to {self._task_description(task_index).lower()}?",
        )
        prompt_builder.add_turn("gpt", action_chunk_string)

        input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
        labels = list(input_ids)
        action_chunk_len = len(action_chunk_string)

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)
        labels[: -(action_chunk_len + 1)] = -100

        image_primary = Image.fromarray(self._load_image(record.episode_index, record.frame_index, self.metadata.primary_image_key))
        pixel_values = self.image_transform(image_primary)

        sample = {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "labels": labels,
            "dataset_name": self.metadata.dataset_name,
            "actions": normalized_actions.astype(np.float32),
            "episode_index": record.episode_index,
            "frame_index": record.frame_index,
            "index": record.global_index,
            "task_index": task_index,
        }

        if self.num_images_in_input > 1 and self.metadata.wrist_image_key is not None:
            wrist_img = Image.fromarray(
                self._load_image(record.episode_index, record.frame_index, self.metadata.wrist_image_key)
            )
            sample["pixel_values_wrist"] = self.image_transform(wrist_img)
        if self.use_proprio:
            sample["proprio"] = normalized_state.astype(np.float32)
        return sample


def register_openvla_auto_classes() -> None:
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)


def resolve_vla_path(vla_path: str) -> str:
    if model_is_on_hf_hub(vla_path):
        return snapshot_download(repo_id=vla_path)
    register_openvla_auto_classes()
    return vla_path


def _prepare_vla_path(vla_path: str) -> str:
    resolved = resolve_vla_path(vla_path.rstrip("/"))
    update_auto_map(resolved)
    check_model_logic_mismatch(resolved)
    return resolved


def _get_run_dir(cfg: OpenVLALeRobotConfig) -> Path:
    if cfg.output_dir is not None:
        return Path(cfg.output_dir)
    note = f"--{cfg.run_id_note}" if cfg.run_id_note else ""
    return (
        cfg.run_root_dir
        / (
            f"{Path(cfg.vla_path).name}+{cfg.dataset_name}"
            f"+b{cfg.batch_size * cfg.grad_accumulation_steps}"
            f"+lr-{cfg.learning_rate}"
            f"+lora-r{cfg.lora_rank}{note}"
        )
    )


def save_finetune_config(run_dir: Path, cfg: OpenVLALeRobotConfig) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = _stringify_paths(dataclasses.asdict(cfg))
    (run_dir / "lerobot_finetune_config.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def load_finetune_config(run_dir: Path) -> OpenVLALeRobotConfig:
    data = _read_json(Path(run_dir) / "lerobot_finetune_config.json")
    for key in (
        "dataset_root",
        "run_root_dir",
        "output_dir",
        "resume_checkpoint_dir",
        "sample_indices_path",
        "norm_stats_path",
    ):
        if data.get(key) is not None:
            data[key] = Path(data[key])
    return OpenVLALeRobotConfig(**data)


def _infer_checkpoint_step(checkpoint_dir: Path) -> int:
    action_head_checkpoint = _find_single_file(checkpoint_dir, "action_head")
    if action_head_checkpoint is None:
        raise ValueError(f"无法从 {checkpoint_dir} 推断恢复步数，缺少 action_head checkpoint。")

    stem = action_head_checkpoint.stem
    if not stem.endswith("_checkpoint"):
        raise ValueError(f"无法从 {action_head_checkpoint.name} 推断恢复步数。")

    step_str = stem.removesuffix("_checkpoint").rsplit("--", 1)[-1]
    try:
        return int(step_str)
    except ValueError as exc:
        raise ValueError(f"无法从 {action_head_checkpoint.name} 推断恢复步数。") from exc


def _compute_learning_rate(cfg: OpenVLALeRobotConfig, step_idx: int) -> float:
    lr = float(cfg.learning_rate)
    if cfg.lr_warmup_steps > 0 and step_idx <= int(cfg.lr_warmup_steps):
        warmup_progress = min(step_idx / float(cfg.lr_warmup_steps), 1.0)
        lr = float(cfg.learning_rate) * (0.1 + 0.9 * warmup_progress)
    if step_idx > int(cfg.num_steps_before_decay):
        lr *= 0.1
    return lr


def _set_optimizer_lr(optimizer: AdamW, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


def load_checkpoint_bundle(
    checkpoint_dir: Path,
    *,
    device: torch.device,
    base_vla_path: Optional[str] = None,
) -> dict[str, Any]:
    checkpoint_dir = Path(checkpoint_dir)
    cfg = load_finetune_config(checkpoint_dir)
    metadata = LeRobotDatasetMetadata(
        cfg.dataset_root,
        dataset_name=cfg.dataset_name,
        task_description_override=cfg.task_description,
        norm_stats_path=cfg.norm_stats_path,
    )
    _patch_runtime_vla_constants(
        action_dim=metadata.action_dim,
        proprio_dim=metadata.state_dim,
        action_chunk_size=cfg.action_chunk_size,
    )
    resolved_vla_path = _prepare_vla_path(base_vla_path or cfg.vla_path)
    processor = AutoProcessor.from_pretrained(resolved_vla_path, trust_remote_code=True)
    base_vla = AutoModelForVision2Seq.from_pretrained(
        resolved_vla_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    base_vla.vision_backbone.set_num_images_in_input(cfg.num_images_in_input)

    adapter_dir = checkpoint_dir / "lora_adapter"
    if not adapter_dir.exists():
        raise ValueError(f"Missing LoRA adapter directory: {adapter_dir}")
    vla = PeftModel.from_pretrained(base_vla, adapter_dir, is_trainable=True).to(device)
    vla.train()

    proprio_projector = None
    if cfg.use_proprio:
        proprio_checkpoint = _find_single_file(checkpoint_dir, "proprio_projector")
        if proprio_checkpoint is None:
            raise ValueError(f"Missing proprio projector checkpoint in {checkpoint_dir}")
        proprio_projector = ProprioProjector(llm_dim=vla.llm_dim, proprio_dim=metadata.state_dim).to(device)
        proprio_projector.load_state_dict(torch.load(proprio_checkpoint, map_location="cpu", weights_only=True))
        proprio_projector = proprio_projector.to(device)
        proprio_projector.train()

    action_head_checkpoint = _find_single_file(checkpoint_dir, "action_head")
    if action_head_checkpoint is None:
        raise ValueError(f"Missing action head checkpoint in {checkpoint_dir}")
    action_head = L1RegressionActionHead(
        input_dim=vla.llm_dim,
        hidden_dim=vla.llm_dim,
        action_dim=metadata.action_dim,
    ).to(torch.bfloat16).to(device)
    action_head.load_state_dict(torch.load(action_head_checkpoint, map_location="cpu", weights_only=True))
    action_head = action_head.to(device)
    action_head.train()

    return {
        "cfg": cfg,
        "processor": processor,
        "vla": vla,
        "action_head": action_head,
        "proprio_projector": proprio_projector,
        "metadata": metadata,
        "resolved_vla_path": resolved_vla_path,
    }


def build_batch_loss(
    *,
    vla: Any,
    action_head: L1RegressionActionHead,
    proprio_projector: Optional[ProprioProjector],
    batch: dict[str, torch.Tensor],
    device: torch.device,
    action_chunk_size: int,
    action_dim: int,
    num_patches: int,
    use_proprio: bool,
) -> tuple[torch.Tensor, dict[str, float]]:
    proprio = batch.get("proprio")
    if use_proprio and proprio is not None and proprio.ndim == 1:
        proprio = proprio.unsqueeze(0)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output: CausalLMOutputWithPast = vla(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            pixel_values=batch["pixel_values"].to(torch.bfloat16).to(device),
            labels=batch["labels"].to(device),
            output_hidden_states=True,
            proprio=proprio.to(device) if use_proprio and proprio is not None else None,
            proprio_projector=proprio_projector,
            use_film=False,
        )

    labels = batch["labels"].to(device)
    actions = batch["actions"].to(device).to(torch.bfloat16)
    current_action_mask = get_current_action_mask(labels[:, 1:])
    next_actions_mask = get_next_actions_mask(labels[:, 1:])
    text_hidden_states = output.hidden_states[-1][:, num_patches:-1]
    action_hidden_states = text_hidden_states[current_action_mask | next_actions_mask].reshape(
        actions.shape[0], action_chunk_size * action_dim, -1
    )
    predicted_actions = action_head.predict_action(action_hidden_states.to(torch.bfloat16))
    loss = torch.nn.L1Loss()(actions, predicted_actions)

    current_l1 = torch.nn.L1Loss()(actions[:, 0], predicted_actions[:, 0]).item()
    next_l1 = torch.nn.L1Loss()(actions[:, 1:], predicted_actions[:, 1:]).item()
    return loss, {"loss_value": float(loss.item()), "curr_action_l1_loss": current_l1, "next_actions_l1_loss": next_l1}


def save_lerobot_norm_stats(path: Path, state_values: np.ndarray, action_values: np.ndarray) -> None:
    payload = {"norm_stats": {"state": _compute_stats(state_values), "actions": _compute_stats(action_values)}}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def recompute_norm_stats_for_indices(
    metadata: LeRobotDatasetMetadata,
    sample_indices: Sequence[int],
    output_path: Path,
) -> None:
    keep = set(int(idx) for idx in sample_indices)
    states: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    for record in metadata.sample_records:
        if record.global_index not in keep:
            continue
        payload = metadata.load_episode_payload(record.episode_index)
        states.append(np.asarray(payload.states[record.frame_index], dtype=np.float32))
        actions.append(np.asarray(payload.actions[record.frame_index], dtype=np.float32))
    if not states or not actions:
        raise ValueError("No samples selected for norm_stats recomputation.")
    save_lerobot_norm_stats(output_path, np.stack(states, axis=0), np.stack(actions, axis=0))


def _save_training_checkpoint(
    *,
    cfg: OpenVLALeRobotConfig,
    run_dir: Path,
    log_step: int,
    vla: Any,
    processor: Any,
    metadata: LeRobotDatasetMetadata,
    proprio_projector: Optional[ProprioProjector],
    action_head: L1RegressionActionHead,
) -> None:
    if cfg.save_latest_checkpoint_only:
        checkpoint_dir = run_dir
        suffix = "latest_checkpoint.pt"
    else:
        checkpoint_dir = Path(str(run_dir) + f"--{log_step}_chkpt")
        suffix = f"{log_step}_checkpoint.pt"

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    adapter_dir = checkpoint_dir / "lora_adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    processor.save_pretrained(checkpoint_dir)
    vla.save_pretrained(adapter_dir)
    save_finetune_config(checkpoint_dir, cfg)
    save_dataset_statistics(metadata.dataset_statistics, checkpoint_dir)
    torch.save(action_head.state_dict(), checkpoint_dir / f"action_head--{suffix}")
    if proprio_projector is not None:
        torch.save(proprio_projector.state_dict(), checkpoint_dir / f"proprio_projector--{suffix}")

    if cfg.merge_lora_during_training:
        resolved_vla_path = _prepare_vla_path(cfg.vla_path)
        base_vla = AutoModelForVision2Seq.from_pretrained(
            resolved_vla_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        base_vla.vision_backbone.set_num_images_in_input(cfg.num_images_in_input)
        merged_vla = PeftModel.from_pretrained(base_vla, adapter_dir).merge_and_unload()
        merged_vla.save_pretrained(checkpoint_dir)


def finetune_openvla_lerobot(cfg: OpenVLALeRobotConfig) -> Path:
    if not cfg.use_lora:
        raise ValueError("This script only supports LoRA fine-tuning.")
    if torch.cuda.device_count() <= 0:
        raise RuntimeError("A CUDA device is required for OpenVLA fine-tuning.")

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device("cuda:0")
    run_dir = _get_run_dir(cfg)
    run_dir.mkdir(parents=True, exist_ok=True)
    save_finetune_config(run_dir, cfg)

    metadata = LeRobotDatasetMetadata(
        cfg.dataset_root,
        dataset_name=cfg.dataset_name,
        task_description_override=cfg.task_description,
        norm_stats_path=cfg.norm_stats_path,
    )
    _patch_runtime_vla_constants(
        action_dim=metadata.action_dim,
        proprio_dim=metadata.state_dim,
        action_chunk_size=cfg.action_chunk_size,
    )
    resume_checkpoint_dir = Path(cfg.resume_checkpoint_dir) if cfg.resume_checkpoint_dir is not None else None
    start_step = 0

    if resume_checkpoint_dir is not None:
        if not resume_checkpoint_dir.exists():
            raise FileNotFoundError(f"恢复目录不存在: {resume_checkpoint_dir}")
        bundle = load_checkpoint_bundle(
            resume_checkpoint_dir,
            device=device,
            base_vla_path=cfg.vla_path,
        )
        processor = bundle["processor"]
        vla = bundle["vla"]
        action_head = bundle["action_head"]
        proprio_projector = bundle["proprio_projector"]
        start_step = int(cfg.resume_step) if cfg.resume_step is not None else _infer_checkpoint_step(resume_checkpoint_dir)
        LOGGER.warning(
            "Resuming from weights only at step %d using checkpoint %s. Optimizer state is not restored.",
            start_step,
            resume_checkpoint_dir,
        )
    else:
        resolved_vla_path = _prepare_vla_path(cfg.vla_path)
        processor = AutoProcessor.from_pretrained(resolved_vla_path, trust_remote_code=True)
        vla = AutoModelForVision2Seq.from_pretrained(
            resolved_vla_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).to(device)
        vla.vision_backbone.set_num_images_in_input(cfg.num_images_in_input)

        lora_config = LoraConfig(
            r=cfg.lora_rank,
            lora_alpha=min(cfg.lora_rank, 16),
            lora_dropout=cfg.lora_dropout,
            target_modules="all-linear",
            init_lora_weights="gaussian",
        )
        vla = get_peft_model(vla, lora_config).to(device)
        vla.train()

        proprio_projector = None
        if cfg.use_proprio:
            proprio_projector = ProprioProjector(llm_dim=vla.llm_dim, proprio_dim=metadata.state_dim).to(device)
            proprio_projector.train()

        action_head = L1RegressionActionHead(
            input_dim=vla.llm_dim,
            hidden_dim=vla.llm_dim,
            action_dim=metadata.action_dim,
        ).to(torch.bfloat16).to(device)
        action_head.train()

    trainable_params = [param for param in vla.parameters() if param.requires_grad]
    trainable_params.extend(param for param in action_head.parameters() if param.requires_grad)
    if proprio_projector is not None:
        trainable_params.extend(param for param in proprio_projector.parameters() if param.requires_grad)

    optimizer = AdamW(trainable_params, lr=cfg.learning_rate)
    if start_step > 0:
        _set_optimizer_lr(optimizer, _compute_learning_rate(cfg, start_step))
    action_tokenizer = ActionTokenizer(processor.tokenizer)
    dataset = LeRobotOpenVLADataset(
        metadata,
        action_tokenizer=action_tokenizer,
        base_tokenizer=processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        action_chunk_size=cfg.action_chunk_size,
        num_images_in_input=cfg.num_images_in_input,
        use_proprio=cfg.use_proprio,
        sample_indices_path=cfg.sample_indices_path,
        task_description=cfg.task_description,
    )
    samples_per_step = int(cfg.batch_size * cfg.grad_accumulation_steps)
    steps_per_epoch = max(1, math.ceil(len(dataset) / max(samples_per_step, 1)))
    planned_samples = int(cfg.max_steps * samples_per_step)
    if planned_samples < len(dataset):
        LOGGER.warning(
            "Training budget covers only %.2f%% of one epoch: dataset_frames=%d samples_per_step=%d max_steps=%d "
            "(planned_samples=%d). This commonly underfits OpenVLA and can produce policies that never close the gripper.",
            100.0 * planned_samples / max(len(dataset), 1),
            len(dataset),
            samples_per_step,
            cfg.max_steps,
            planned_samples,
        )
    if int(cfg.num_steps_before_decay) < steps_per_epoch:
        LOGGER.warning(
            "LR decay begins before one full epoch: num_steps_before_decay=%d < steps_per_epoch=%d. "
            "For small robot datasets this often decays too early.",
            int(cfg.num_steps_before_decay),
            steps_per_epoch,
        )
    save_dataset_statistics(dataset.dataset_statistics, run_dir)
    collator = PaddedCollatorForActionPrediction(
        processor.tokenizer.model_max_length,
        processor.tokenizer.pad_token_id,
        padding_side="right",
    )
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=cfg.shuffle,
        num_workers=cfg.num_workers,
        collate_fn=collator,
        drop_last=False,
    )

    num_patches = vla.vision_backbone.get_num_patches() * cfg.num_images_in_input
    if cfg.use_proprio:
        num_patches += 1

    if start_step > cfg.max_steps:
        raise ValueError(f"resume_step={start_step} 大于 max_steps={cfg.max_steps}，无法继续训练。")

    step = start_step
    optimizer.zero_grad(set_to_none=True)
    progress = tqdm.tqdm(total=cfg.max_steps, initial=start_step, desc="finetune_openvla_lerobot", leave=False)
    while step < cfg.max_steps:
        for batch in dataloader:
            loss, _ = build_batch_loss(
                vla=vla,
                action_head=action_head,
                proprio_projector=proprio_projector,
                batch=batch,
                device=device,
                action_chunk_size=cfg.action_chunk_size,
                action_dim=metadata.action_dim,
                num_patches=num_patches,
                use_proprio=cfg.use_proprio,
            )
            (loss / cfg.grad_accumulation_steps).backward()

            if (step + 1) % cfg.grad_accumulation_steps == 0:
                next_step = step + 1
                _set_optimizer_lr(optimizer, _compute_learning_rate(cfg, next_step))
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                step = next_step
                progress.update(1)

                if step % cfg.save_freq == 0:
                    _save_training_checkpoint(
                        cfg=cfg,
                        run_dir=run_dir,
                        log_step=step,
                        vla=vla,
                        processor=processor,
                        metadata=metadata,
                        proprio_projector=proprio_projector,
                        action_head=action_head,
                    )
                if step >= cfg.max_steps:
                    break
        else:
            continue
        break

    final_step = max(step, cfg.max_steps)
    _save_training_checkpoint(
        cfg=cfg,
        run_dir=run_dir,
        log_step=final_step,
        vla=vla,
        processor=processor,
        metadata=metadata,
        proprio_projector=proprio_projector,
        action_head=action_head,
    )
    progress.close()
    return run_dir
