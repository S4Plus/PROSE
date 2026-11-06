#!/usr/bin/env python3
"""
GraSS-style window-level influence attribution for OpenVLA LoRA checkpoints on LeRobot data.
"""

from __future__ import annotations

import argparse
import base64
import dataclasses
import io
import json
import logging
import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import torch
from PIL import Image

from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.lerobot_openvla import (
    LeRobotDatasetMetadata,
    LeRobotOpenVLADataset,
    build_batch_loss,
    load_checkpoint_bundle,
)
from prismatic.models.backbones.llm.prompting import PurePromptBuilder

LOGGER = logging.getLogger("influence_openvla_lora_grass_window")


@dataclass
class ProjectionSpec:
    proj_type: str
    dims: tuple[int, ...]


@dataclass
class WindowInfluenceConfig:
    checkpoint_dir: Path
    eval_rollouts_dir: Path
    output_path: Path
    dataset_root: Optional[Path] = None
    base_vla_path: Optional[str] = None
    task_description: Optional[str] = None
    score_level: str = "window"
    seed: int = 7
    max_train_samples: Optional[int] = None
    max_eval_episodes: Optional[int] = None
    max_eval_steps_per_episode: Optional[int] = None
    stage1_microbatch_size: int = 1
    stage2_batch_size: int = 16
    progress_every: int = 200
    sparsification: str = "random_mask-32*32"
    projection: str = "sjlt-1024"
    hessian: str = "eFIM"
    damping: float = 1e-4
    save_preconditioners: bool = False
    stage_cache_prefix: Optional[Path] = None
    window_size: int = 50
    window_stride: Optional[int] = None
    window_aggregation: str = "sum"
    quality_score: str = "zscore_mean"


@dataclass
class EvalEpisode:
    observations: dict[str, Any]
    actions: np.ndarray
    reward: float
    length: int


def _parse_sparsification(spec: str) -> ProjectionSpec:
    spec = spec.strip().lower()
    if spec == "identity":
        return ProjectionSpec("identity", (-1, -1))
    proj_type, dims = spec.split("-", 1)
    d1, d2 = dims.split("*", 1)
    return ProjectionSpec(proj_type, (int(d1), int(d2)))


def _parse_projection(spec: str) -> ProjectionSpec:
    spec = spec.strip().lower()
    if spec == "identity":
        return ProjectionSpec("identity", (-1,))
    proj_type, dim = spec.split("-", 1)
    return ProjectionSpec(proj_type, (int(dim),))


def _zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return values
    std = float(np.std(values))
    if std < 1e-12:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - float(np.mean(values))) / std).astype(np.float32)


def _reduce(values: np.ndarray, mode: str) -> float:
    values = np.asarray(values, dtype=np.float32)
    if mode == "sum":
        return float(np.sum(values))
    if mode == "mean":
        return float(np.mean(values))
    raise ValueError(f"Unsupported aggregation mode: {mode}")


def _decode_image(image: Any) -> np.ndarray | None:
    if image is None:
        return None
    if isinstance(image, str):
        payload = image.split(",", 1)[-1]
        image = Image.open(io.BytesIO(base64.b64decode(payload))).convert("RGB")
    arr = np.asarray(image)
    if arr.ndim != 3:
        return None
    if arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.dtype != np.uint8:
        if np.issubdtype(arr.dtype, np.floating) and arr.size > 0:
            arr = np.clip(arr * 255.0 if float(np.nanmax(arr)) <= 1.0 else arr, 0, 255)
        arr = arr.astype(np.uint8)
    return arr


def _first_present(mapping: Mapping[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


class EvalRolloutsDataset:
    def __init__(self, rollouts_dir: Path):
        files = sorted(Path(rollouts_dir).glob("episode_*.pkl"))
        if not files:
            files = sorted(Path(rollouts_dir).glob("*.pkl"))
        if not files:
            raise ValueError(f"No rollout pickle files found in {rollouts_dir}")
        self.episodes = [self._load_file(path) for path in files]

    def _load_file(self, path: Path) -> EvalEpisode:
        with path.open("rb") as f:
            payload = pickle.load(f)
        if isinstance(payload, dict) and payload.get("storage") == "chunked_pickle_v1":
            payload = self._materialize_chunked_payload(path, payload)
        if isinstance(payload, dict) and "rows" in payload:
            return self._load_from_rows(payload)
        if isinstance(payload, dict):
            return self._load_from_dict(payload)
        raise TypeError(f"Unsupported rollout payload type: {type(payload)}")

    def _materialize_chunked_payload(self, path: Path, payload: dict[str, Any]) -> dict[str, Any]:
        chunk_dir = path.parent / f"episode_{int(payload['episode']):06d}.chunks"
        if not chunk_dir.is_dir():
            raise FileNotFoundError(f"Missing chunk directory for rollout manifest: {chunk_dir}")

        rows: list[dict[str, Any]] = []
        for chunk_name in payload.get("rows_chunk_files", []):
            with (chunk_dir / str(chunk_name)).open("rb") as f:
                chunk_rows = pickle.load(f)
            if not isinstance(chunk_rows, list):
                raise TypeError(f"Unexpected rows chunk payload type in {chunk_dir / str(chunk_name)}")
            rows.extend(chunk_rows)

        policy_calls: list[dict[str, Any]] = []
        for chunk_name in payload.get("policy_calls_chunk_files", []):
            with (chunk_dir / str(chunk_name)).open("rb") as f:
                chunk_calls = pickle.load(f)
            if not isinstance(chunk_calls, list):
                raise TypeError(f"Unexpected policy chunk payload type in {chunk_dir / str(chunk_name)}")
            policy_calls.extend(chunk_calls)

        episode_success = bool(payload.get("success", False))
        for row in rows:
            if isinstance(row, dict):
                row["success"] = episode_success

        merged = dict(payload)
        merged["rows"] = rows
        merged["policy_calls"] = policy_calls
        return merged

    def _load_from_rows(self, payload: dict[str, Any]) -> EvalEpisode:
        rows = payload["rows"]
        policy_call_map = {
            int(call["call_id"]): call
            for call in payload.get("policy_calls", [])
            if isinstance(call, dict) and call.get("call_id") is not None
        }
        states: list[np.ndarray] = []
        actions: list[np.ndarray] = []
        base_images: list[np.ndarray | None] = []
        wrist_images: list[np.ndarray | None] = []
        for row in rows:
            policy_call = None
            if isinstance(row, dict) and row.get("policy_call_id") is not None:
                try:
                    policy_call = policy_call_map.get(int(row["policy_call_id"]))
                except Exception:
                    policy_call = None

            action = None
            if isinstance(row.get("policy_output"), dict):
                action = row["policy_output"].get("action")
            if action is None and isinstance(policy_call, dict) and isinstance(policy_call.get("response"), dict):
                action = policy_call["response"].get("action")
            if action is None:
                action = row.get("policy_action_chunk_raw")
            if action is None and isinstance(policy_call, dict):
                action = policy_call.get("action_chunk_raw")
            if action is None:
                action = row.get("action", row.get("action_raw"))
            if action is None:
                raise KeyError("action")
            actions.append(np.asarray(action, dtype=np.float32))

            row_obs = row.get("obs", {})
            state = row_obs.get("state")
            if state is None and isinstance(row.get("policy_input"), dict):
                state = row["policy_input"].get("state")
            if state is None and isinstance(policy_call, dict) and isinstance(policy_call.get("request"), dict):
                state = policy_call["request"].get("state")
            if state is None:
                raise KeyError("state")
            states.append(np.asarray(state, dtype=np.float32))

            images_map = None
            if isinstance(row_obs.get("image"), dict):
                images_map = row_obs["image"]
            elif isinstance(row_obs.get("images"), dict):
                images_map = row_obs["images"]
            elif isinstance(row.get("policy_input"), dict) and isinstance(row["policy_input"].get("images"), dict):
                images_map = row["policy_input"]["images"]
            elif isinstance(policy_call, dict) and isinstance(policy_call.get("request"), dict):
                maybe_images = policy_call["request"].get("images")
                if isinstance(maybe_images, dict):
                    images_map = maybe_images

            if not isinstance(images_map, dict):
                base_images.append(None)
                wrist_images.append(None)
                continue

            base_images.append(
                _decode_image(
                    _first_present(images_map, ("observation.images.agentview", "agentview", "base_0_rgb"))
                )
            )
            wrist_images.append(
                _decode_image(
                    _first_present(
                        images_map,
                        ("observation.images.robot0_eye_in_hand", "observation.images.wrist", "wrist"),
                    )
                )
            )

        reward = payload.get("reward", payload.get("success", 0.0))
        if isinstance(reward, (list, np.ndarray)):
            reward = float(np.sum(reward))
        else:
            reward = float(reward)

        return EvalEpisode(
            observations={
                "state": np.stack(states, axis=0).astype(np.float32),
                "base_images": base_images,
                "wrist_images": wrist_images,
            },
            actions=np.stack(actions, axis=0).astype(np.float32),
            reward=reward,
            length=len(rows),
        )

    def _load_from_dict(self, payload: dict[str, Any]) -> EvalEpisode:
        observations = payload.get("observations", {})
        reward = payload.get("reward", payload.get("success", 0.0))
        if isinstance(reward, (list, np.ndarray)):
            reward = float(np.sum(reward))
        return EvalEpisode(
            observations={
                "state": np.asarray(observations["state"], dtype=np.float32),
                "base_images": list(np.asarray(observations["base_images"], dtype=object)),
                "wrist_images": list(np.asarray(observations.get("wrist_images", []), dtype=object)),
            },
            actions=np.asarray(payload["actions"], dtype=np.float32),
            reward=float(reward),
            length=int(len(payload["actions"])),
        )

    @property
    def episode_rewards(self) -> np.ndarray:
        return np.asarray([episode.reward for episode in self.episodes], dtype=np.float32)

    @property
    def episode_lengths(self) -> np.ndarray:
        return np.asarray([episode.length for episode in self.episodes], dtype=np.int64)


@dataclass
class DenseProjector:
    matrix: np.ndarray

    def project(self, values: np.ndarray) -> np.ndarray:
        return values @ self.matrix


@dataclass
class CountSketchProjector:
    indices: np.ndarray
    signs: np.ndarray
    scale: float
    output_dim: int

    def project(self, values: np.ndarray) -> np.ndarray:
        if values.ndim == 1:
            values = values[None, :]
            squeeze = True
        else:
            squeeze = False
        out = np.zeros((values.shape[0], self.output_dim), dtype=np.float32)
        for i in range(values.shape[0]):
            np.add.at(out[i], self.indices, values[i] * self.signs)
        out *= self.scale
        return out[0] if squeeze else out


def _build_1d_projector(
    *,
    input_dim: int,
    output_dim: int,
    proj_type: str,
    seed: int,
) -> DenseProjector | CountSketchProjector | None:
    if proj_type == "identity" or output_dim < 0 or output_dim >= input_dim:
        return None

    rng = np.random.default_rng(seed)
    if proj_type == "normal":
        matrix = rng.normal(size=(input_dim, output_dim)).astype(np.float32)
        matrix /= math.sqrt(float(output_dim))
        return DenseProjector(matrix=matrix)
    if proj_type in {"sjlt", "countsketch"}:
        indices = rng.integers(0, output_dim, size=(input_dim,), dtype=np.int64)
        signs = rng.choice(np.asarray([-1.0, 1.0], dtype=np.float32), size=(input_dim,))
        return CountSketchProjector(
            indices=indices,
            signs=signs.astype(np.float32),
            scale=1.0 / math.sqrt(float(output_dim)),
            output_dim=output_dim,
        )
    if proj_type == "random_mask":
        choice = np.sort(rng.choice(input_dim, size=min(output_dim, input_dim), replace=False))
        matrix = np.zeros((input_dim, len(choice)), dtype=np.float32)
        matrix[choice, np.arange(len(choice))] = math.sqrt(input_dim / max(len(choice), 1))
        return DenseProjector(matrix=matrix)
    raise ValueError(f"Unsupported projector type: {proj_type}")


@dataclass
class LeafCompressionState:
    name: str
    original_shape: tuple[int, ...]
    rows: int
    cols: int
    row_projector: DenseProjector | CountSketchProjector | None
    col_projector: DenseProjector | CountSketchProjector | None
    stage2_projector: DenseProjector | CountSketchProjector | None
    output_dim: int


class LeafCompressor:
    def __init__(
        self,
        *,
        name: str,
        shape: Sequence[int],
        sparsification: ProjectionSpec,
        projection: ProjectionSpec,
        seed: int,
    ) -> None:
        original_shape = tuple(int(dim) for dim in shape)
        if len(original_shape) == 0:
            rows, cols = 1, 1
        elif len(original_shape) == 1:
            rows, cols = int(original_shape[0]), 1
        else:
            rows = int(np.prod(original_shape[:-1]))
            cols = int(original_shape[-1])

        row_target, col_target = sparsification.dims
        row_dim = rows if row_target < 0 else min(row_target, rows)
        col_dim = cols if col_target < 0 else min(col_target, cols)
        row_projector = _build_1d_projector(input_dim=rows, output_dim=row_dim, proj_type=sparsification.proj_type, seed=seed)
        col_projector = _build_1d_projector(
            input_dim=cols,
            output_dim=col_dim,
            proj_type=sparsification.proj_type,
            seed=seed + 1,
        )

        inter_dim = row_dim * col_dim
        final_target = projection.dims[0]
        final_dim = inter_dim if final_target < 0 else min(final_target, inter_dim)
        stage2_projector = _build_1d_projector(
            input_dim=inter_dim,
            output_dim=final_dim,
            proj_type=projection.proj_type,
            seed=seed + 2,
        )
        if stage2_projector is None:
            final_dim = inter_dim

        self.state = LeafCompressionState(
            name=name,
            original_shape=original_shape,
            rows=rows,
            cols=cols,
            row_projector=row_projector,
            col_projector=col_projector,
            stage2_projector=stage2_projector,
            output_dim=final_dim,
        )

    def compress(self, grad: np.ndarray) -> np.ndarray:
        grad = np.asarray(grad, dtype=np.float32)
        if grad.shape != self.state.original_shape:
            grad = grad.reshape(self.state.original_shape)
        grad = grad.reshape(self.state.rows, self.state.cols)

        if self.state.row_projector is not None:
            if isinstance(self.state.row_projector, DenseProjector):
                grad = self.state.row_projector.matrix.T @ grad
            else:
                out = np.zeros((self.state.row_projector.output_dim, grad.shape[1]), dtype=np.float32)
                np.add.at(
                    out,
                    self.state.row_projector.indices,
                    grad * self.state.row_projector.signs[:, None],
                )
                grad = out * self.state.row_projector.scale

        if self.state.col_projector is not None:
            if isinstance(self.state.col_projector, DenseProjector):
                grad = grad @ self.state.col_projector.matrix
            else:
                out = np.zeros((grad.shape[0], self.state.col_projector.output_dim), dtype=np.float32)
                for row_idx in range(grad.shape[0]):
                    np.add.at(
                        out[row_idx],
                        self.state.col_projector.indices,
                        grad[row_idx] * self.state.col_projector.signs,
                    )
                grad = out * self.state.col_projector.scale

        flat = grad.reshape(-1)
        if self.state.stage2_projector is not None:
            flat = self.state.stage2_projector.project(flat)
        return np.asarray(flat, dtype=np.float32).reshape(-1)


class WindowInfluenceRunner:
    def __init__(self, cfg: WindowInfluenceConfig):
        self.cfg = cfg
        self.device = torch.device("cuda:0")
        self.bundle = load_checkpoint_bundle(
            cfg.checkpoint_dir,
            device=self.device,
            base_vla_path=cfg.base_vla_path,
        )
        self.finetune_cfg = self.bundle["cfg"]
        self.metadata: LeRobotDatasetMetadata = self.bundle["metadata"]
        if cfg.dataset_root is not None and cfg.dataset_root != self.metadata.dataset_root:
            self.metadata = LeRobotDatasetMetadata(
                cfg.dataset_root,
                dataset_name=self.finetune_cfg.dataset_name,
                task_description_override=cfg.task_description or self.finetune_cfg.task_description,
                norm_stats_path=self.finetune_cfg.norm_stats_path,
            )
        self.processor = self.bundle["processor"]
        self.vla = self.bundle["vla"]
        self.action_head = self.bundle["action_head"]
        self.proprio_projector = self.bundle["proprio_projector"]
        self.action_tokenizer = ActionTokenizer(self.processor.tokenizer)
        self.collator = PaddedCollatorForActionPrediction(
            self.processor.tokenizer.model_max_length,
            self.processor.tokenizer.pad_token_id,
            padding_side="right",
        )
        self.train_dataset = LeRobotOpenVLADataset(
            self.metadata,
            action_tokenizer=self.action_tokenizer,
            base_tokenizer=self.processor.tokenizer,
            image_transform=self.processor.image_processor.apply_transform,
            action_chunk_size=self.finetune_cfg.action_chunk_size,
            num_images_in_input=self.finetune_cfg.num_images_in_input,
            use_proprio=self.finetune_cfg.use_proprio,
            sample_indices_path=self.finetune_cfg.sample_indices_path,
            task_description=cfg.task_description or self.finetune_cfg.task_description,
        )
        self.eval_dataset = EvalRolloutsDataset(cfg.eval_rollouts_dir)
        self.lora_params = [
            (name, param)
            for name, param in self.vla.named_parameters()
            if param.requires_grad and "lora_" in name
        ]
        if not self.lora_params:
            raise ValueError("No trainable LoRA parameters found in the checkpoint.")

        self.sparsification = _parse_sparsification(cfg.sparsification)
        self.projection = _parse_projection(cfg.projection)
        self.compressors = [
            LeafCompressor(
                name=name,
                shape=tuple(param.shape),
                sparsification=self.sparsification,
                projection=self.projection,
                seed=cfg.seed + idx * 97 + 1,
            )
            for idx, (name, param) in enumerate(self.lora_params)
        ]
        self.num_patches = self.vla.vision_backbone.get_num_patches() * self.finetune_cfg.num_images_in_input
        if self.finetune_cfg.use_proprio:
            self.num_patches += 1

        self.output_path = cfg.output_path
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_dir = cfg.stage_cache_prefix or (self.output_path.parent / f"{self.output_path.stem}.grass_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.window_stride = cfg.window_stride or cfg.window_size

    def _zero_all_grads(self) -> None:
        self.vla.zero_grad(set_to_none=True)
        self.action_head.zero_grad(set_to_none=True)
        if self.proprio_projector is not None:
            self.proprio_projector.zero_grad(set_to_none=True)

    def _compute_lora_grads(self, batch: dict[str, torch.Tensor], *, scale: float = 1.0) -> dict[str, np.ndarray]:
        self._zero_all_grads()
        loss, _ = build_batch_loss(
            vla=self.vla,
            action_head=self.action_head,
            proprio_projector=self.proprio_projector,
            batch=batch,
            device=self.device,
            action_chunk_size=self.finetune_cfg.action_chunk_size,
            action_dim=self.metadata.action_dim,
            num_patches=self.num_patches,
            use_proprio=self.finetune_cfg.use_proprio,
        )
        (loss * float(scale)).backward()
        return {
            name: param.grad.detach().float().cpu().numpy().copy()
            for name, param in self.lora_params
        }

    def _compress_grads(self, grad_dict: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        return {
            compressor.state.name: compressor.compress(grad_dict[compressor.state.name])
            for compressor in self.compressors
        }

    def _iter_train_indices(self) -> list[int]:
        total = len(self.train_dataset)
        if self.cfg.max_train_samples is not None:
            total = min(total, int(self.cfg.max_train_samples))
        return list(range(total))

    def _cache_train_grads(self) -> dict[str, np.ndarray]:
        LOGGER.info("Stage 1: caching compressed train gradients")
        buffers = {name: [] for name, _ in self.lora_params}
        train_indices = self._iter_train_indices()
        progress_every = max(int(self.cfg.progress_every), 1)
        for offset, dataset_idx in enumerate(train_indices, start=1):
            instance = self.train_dataset[dataset_idx]
            batch = self.collator([instance])
            compressed = self._compress_grads(self._compute_lora_grads(batch))
            for name, vector in compressed.items():
                buffers[name].append(vector.astype(np.float32))
            if offset == 1 or offset % progress_every == 0 or offset == len(train_indices):
                LOGGER.info("  [Stage1] sample %d/%d", offset, len(train_indices))
        return {name: np.stack(vectors, axis=0) for name, vectors in buffers.items()}

    def _compute_preconditioners(self, train_grads: dict[str, np.ndarray]) -> dict[str, np.ndarray | None]:
        LOGGER.info("Stage 2: computing compressed-space preconditioners")
        if self.cfg.hessian.lower() == "identity":
            return {name: None for name, _ in self.lora_params}
        preconditioners: dict[str, np.ndarray | None] = {}
        for name, _ in self.lora_params:
            values = np.asarray(train_grads[name], dtype=np.float32)
            hessian = (values.T @ values) / max(float(values.shape[0]), 1.0)
            hessian += float(self.cfg.damping) * np.eye(hessian.shape[0], dtype=np.float32)
            preconditioners[name] = np.linalg.pinv(hessian, hermitian=True).astype(np.float32)
            LOGGER.info("  [Stage2] leaf %s preconditioner=%s", name, preconditioners[name].shape)
        return preconditioners

    def _build_eval_instance(
        self,
        *,
        state: np.ndarray,
        base_image: np.ndarray,
        wrist_image: Optional[np.ndarray],
        action_chunk: np.ndarray,
    ) -> dict[str, Any]:
        normalized_state = self.metadata.normalize_state(state)
        normalized_actions = self.metadata.normalize_actions(action_chunk)
        prompt_builder = PurePromptBuilder("openvla")
        current_action_string = self.action_tokenizer(normalized_actions[0])
        future_actions_string = "".join(self.action_tokenizer(normalized_actions[1:]))
        action_chunk_string = current_action_string + future_actions_string
        prompt_builder.add_turn(
            "human",
            f"What action should the robot take to {(self.cfg.task_description or self.finetune_cfg.task_description or self.metadata.default_task_description).lower()}?",
        )
        prompt_builder.add_turn("gpt", action_chunk_string)
        input_ids = self.processor.tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
        labels = list(input_ids)
        labels[: -(len(action_chunk_string) + 1)] = -100

        sample = {
            "pixel_values": self.processor.image_processor.apply_transform(Image.fromarray(base_image)),
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "dataset_name": self.metadata.dataset_name,
            "actions": normalized_actions.astype(np.float32),
        }
        if self.finetune_cfg.num_images_in_input > 1 and wrist_image is not None:
            sample["pixel_values_wrist"] = self.processor.image_processor.apply_transform(Image.fromarray(wrist_image))
        if self.finetune_cfg.use_proprio:
            sample["proprio"] = normalized_state.astype(np.float32)
        return sample

    def _build_eval_action_chunk(self, actions: np.ndarray, frame_index: int) -> np.ndarray:
        end = min(frame_index + self.finetune_cfg.action_chunk_size, actions.shape[0])
        chunk = np.asarray(actions[frame_index:end], dtype=np.float32)
        if chunk.shape[0] < self.finetune_cfg.action_chunk_size:
            pad = np.zeros(
                (self.finetune_cfg.action_chunk_size - chunk.shape[0], self.metadata.action_dim),
                dtype=np.float32,
            )
            chunk = np.concatenate([chunk, pad], axis=0)
        return chunk

    def _compute_eval_return_grad(self) -> dict[str, np.ndarray]:
        LOGGER.info("Stage 3: computing eval return gradient")
        episodes = self.eval_dataset.episodes
        if self.cfg.max_eval_episodes is not None:
            episodes = episodes[: int(self.cfg.max_eval_episodes)]
        rewards = np.asarray([episode.reward for episode in episodes], dtype=np.float32)
        centered_rewards = rewards - float(np.mean(rewards)) if rewards.size else rewards

        return_grads = {
            name: np.zeros((compressor.state.output_dim,), dtype=np.float32)
            for name, compressor in zip((name for name, _ in self.lora_params), self.compressors, strict=True)
        }
        progress_every = max(int(self.cfg.progress_every), 1)
        for episode_idx, episode in enumerate(episodes, start=1):
            episode_grads = {name: np.zeros_like(value) for name, value in return_grads.items()}
            episode_length = episode.length
            if self.cfg.max_eval_steps_per_episode is not None:
                episode_length = min(episode_length, int(self.cfg.max_eval_steps_per_episode))

            for start in range(0, episode_length, int(self.cfg.stage2_batch_size)):
                end = min(start + int(self.cfg.stage2_batch_size), episode_length)
                instances = []
                for frame_index in range(start, end):
                    base_image = np.asarray(episode.observations["base_images"][frame_index], dtype=np.uint8)
                    wrist_image = None
                    if frame_index < len(episode.observations["wrist_images"]):
                        wrist_image = episode.observations["wrist_images"][frame_index]
                        if wrist_image is not None:
                            wrist_image = np.asarray(wrist_image, dtype=np.uint8)
                    instances.append(
                        self._build_eval_instance(
                            state=np.asarray(episode.observations["state"][frame_index], dtype=np.float32),
                            base_image=base_image,
                            wrist_image=wrist_image,
                            action_chunk=self._build_eval_action_chunk(episode.actions, frame_index),
                        )
                    )
                batch = self.collator(instances)
                compressed = self._compress_grads(
                    self._compute_lora_grads(batch, scale=float(end - start))
                )
                for name, vector in compressed.items():
                    episode_grads[name] += vector

            weight = float(centered_rewards[episode_idx - 1]) if centered_rewards.size else 0.0
            for name in return_grads:
                return_grads[name] += weight * episode_grads[name]
            if episode_idx == 1 or episode_idx % progress_every == 0 or episode_idx == len(episodes):
                LOGGER.info("  [Stage3] episode %d/%d", episode_idx, len(episodes))

        if episodes:
            for name in return_grads:
                return_grads[name] /= float(len(episodes))
        return return_grads

    def _score(
        self,
        *,
        train_grads: dict[str, np.ndarray],
        preconditioners: dict[str, np.ndarray | None],
        eval_grad: dict[str, np.ndarray],
    ) -> np.ndarray:
        scores = np.zeros((next(iter(train_grads.values())).shape[0],), dtype=np.float32)
        for name, _ in self.lora_params:
            target = eval_grad[name] if preconditioners[name] is None else preconditioners[name] @ eval_grad[name]
            scores -= train_grads[name] @ np.asarray(target, dtype=np.float32)
        return scores

    def _aggregate_windows(self, sample_scores: np.ndarray) -> dict[str, np.ndarray]:
        records = self.train_dataset.sample_records[: sample_scores.shape[0]]
        episode_indices = np.asarray([record.episode_index for record in records], dtype=np.int64)
        frame_indices = np.asarray([record.frame_index for record in records], dtype=np.int64)
        global_indices = np.asarray([record.global_index for record in records], dtype=np.int64)

        sample_window_indices = np.full(sample_scores.shape[0], -1, dtype=np.int64)
        sample_window_local_indices = np.full(sample_scores.shape[0], -1, dtype=np.int64)

        window_episode_indices: list[int] = []
        window_local_indices: list[int] = []
        window_frame_starts: list[int] = []
        window_frame_ends: list[int] = []
        window_sample_counts: list[int] = []
        window_is_partial: list[bool] = []
        window_influence_scores: list[float] = []
        window_sum_scores: list[float] = []
        window_mean_scores: list[float] = []

        window_idx = 0
        for episode_index in sorted(np.unique(episode_indices).tolist()):
            episode_sample_ids = np.where(episode_indices == episode_index)[0]
            ordered = episode_sample_ids[np.argsort(frame_indices[episode_sample_ids], kind="stable")]
            local_window_idx = 0
            for start in range(0, len(ordered), self.window_stride):
                end = min(start + self.cfg.window_size, len(ordered))
                if end <= start:
                    continue
                current = ordered[start:end]
                current_scores = np.asarray(sample_scores[current], dtype=np.float32)
                current_frames = frame_indices[current]
                sample_window_indices[current] = window_idx
                sample_window_local_indices[current] = local_window_idx
                window_episode_indices.append(int(episode_index))
                window_local_indices.append(int(local_window_idx))
                window_frame_starts.append(int(np.min(current_frames)))
                window_frame_ends.append(int(np.max(current_frames)))
                window_sample_counts.append(int(current.shape[0]))
                window_is_partial.append(bool(current.shape[0] < self.cfg.window_size))
                window_sum_scores.append(float(np.sum(current_scores)))
                window_mean_scores.append(float(np.mean(current_scores)))
                window_influence_scores.append(_reduce(current_scores, self.cfg.window_aggregation))
                local_window_idx += 1
                window_idx += 1

        mean_scores = np.asarray(window_mean_scores, dtype=np.float32)
        sum_scores = np.asarray(window_sum_scores, dtype=np.float32)
        if self.cfg.quality_score == "mean":
            window_quality_scores = mean_scores
        elif self.cfg.quality_score == "sum":
            window_quality_scores = sum_scores
        elif self.cfg.quality_score == "zscore_sum":
            window_quality_scores = _zscore(sum_scores)
        else:
            window_quality_scores = _zscore(mean_scores)

        return {
            "window_indices": np.arange(len(window_episode_indices), dtype=np.int64),
            "window_episode_indices": np.asarray(window_episode_indices, dtype=np.int64),
            "window_local_indices": np.asarray(window_local_indices, dtype=np.int64),
            "window_frame_starts": np.asarray(window_frame_starts, dtype=np.int64),
            "window_frame_ends": np.asarray(window_frame_ends, dtype=np.int64),
            "window_sample_counts": np.asarray(window_sample_counts, dtype=np.int64),
            "window_is_partial": np.asarray(window_is_partial, dtype=np.bool_),
            "window_influence_scores": np.asarray(window_influence_scores, dtype=np.float32),
            "window_sum_influence_scores": sum_scores,
            "window_mean_influence_scores": mean_scores,
            "window_quality_scores": np.asarray(window_quality_scores, dtype=np.float32),
            "sample_window_indices": sample_window_indices,
            "sample_window_local_indices": sample_window_local_indices,
            "sample_episode_indices": episode_indices,
            "sample_frame_indices": frame_indices,
            "sample_dataset_sample_indices": global_indices,
        }

    def _aggregate_episodes(self, sample_scores: np.ndarray, window_data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        episode_indices = np.asarray(window_data["sample_episode_indices"], dtype=np.int64)
        unique_episode_indices = np.asarray(sorted(np.unique(episode_indices).tolist()), dtype=np.int64)
        episode_influence_scores = np.zeros((len(unique_episode_indices),), dtype=np.float32)
        episode_mean_influence_scores = np.zeros((len(unique_episode_indices),), dtype=np.float32)
        episode_quality_scores = np.zeros((len(unique_episode_indices),), dtype=np.float32)
        episode_sample_counts = np.zeros((len(unique_episode_indices),), dtype=np.int64)
        episode_window_counts = np.zeros((len(unique_episode_indices),), dtype=np.int64)
        for out_idx, episode_index in enumerate(unique_episode_indices.tolist()):
            sample_ids = np.where(episode_indices == episode_index)[0]
            episode_values = np.asarray(sample_scores[sample_ids], dtype=np.float32)
            episode_influence_scores[out_idx] = float(np.sum(episode_values))
            episode_mean_influence_scores[out_idx] = float(np.mean(episode_values))
            episode_sample_counts[out_idx] = int(sample_ids.shape[0])
            window_ids = np.where(window_data["window_episode_indices"] == episode_index)[0]
            episode_window_counts[out_idx] = int(window_ids.shape[0])
            if window_ids.size > 0:
                episode_quality_scores[out_idx] = float(np.mean(window_data["window_quality_scores"][window_ids]))
        return {
            "episode_indices": unique_episode_indices,
            "episode_influence_scores": episode_influence_scores,
            "episode_mean_influence_scores": episode_mean_influence_scores,
            "episode_quality_scores": _zscore(episode_quality_scores),
            "episode_quality_raw_scores": episode_quality_scores,
            "episode_sample_counts": episode_sample_counts,
            "episode_window_counts": episode_window_counts,
        }

    def run(self) -> dict[str, Any]:
        cache_train = self.cache_dir / f"{self.output_path.stem}_stage1_train_grads.pkl"
        cache_precond = self.cache_dir / f"{self.output_path.stem}_stage2_preconditioners.pkl"
        cache_eval = self.cache_dir / f"{self.output_path.stem}_stage3_eval_return_grad.pkl"

        if cache_train.exists():
            with cache_train.open("rb") as f:
                train_grads = pickle.load(f)
        else:
            train_grads = self._cache_train_grads()
            with cache_train.open("wb") as f:
                pickle.dump(train_grads, f, protocol=pickle.HIGHEST_PROTOCOL)

        if cache_precond.exists():
            with cache_precond.open("rb") as f:
                preconditioners = pickle.load(f)
        else:
            preconditioners = self._compute_preconditioners(train_grads)
            with cache_precond.open("wb") as f:
                pickle.dump(preconditioners, f, protocol=pickle.HIGHEST_PROTOCOL)

        if cache_eval.exists():
            with cache_eval.open("rb") as f:
                eval_grad = pickle.load(f)
        else:
            eval_grad = self._compute_eval_return_grad()
            with cache_eval.open("wb") as f:
                pickle.dump(eval_grad, f, protocol=pickle.HIGHEST_PROTOCOL)

        sample_scores = self._score(train_grads=train_grads, preconditioners=preconditioners, eval_grad=eval_grad)
        window_data = self._aggregate_windows(sample_scores)
        episode_data = self._aggregate_episodes(sample_scores, window_data)

        result = {
            "config": dataclasses.asdict(self.cfg),
            "score_level": "window",
            "window_size": int(self.cfg.window_size),
            "window_stride": int(self.window_stride),
            "window_aggregation": self.cfg.window_aggregation,
            "quality_score": self.cfg.quality_score,
            "influence_scores": np.asarray(window_data["window_influence_scores"], dtype=np.float32),
            "sample_influence_scores": np.asarray(sample_scores, dtype=np.float32),
            "window_influence_scores": np.asarray(window_data["window_influence_scores"], dtype=np.float32),
            "window_sum_influence_scores": np.asarray(window_data["window_sum_influence_scores"], dtype=np.float32),
            "window_mean_influence_scores": np.asarray(window_data["window_mean_influence_scores"], dtype=np.float32),
            "window_quality_scores": np.asarray(window_data["window_quality_scores"], dtype=np.float32),
            "episode_influence_scores": np.asarray(episode_data["episode_influence_scores"], dtype=np.float32),
            "episode_mean_influence_scores": np.asarray(episode_data["episode_mean_influence_scores"], dtype=np.float32),
            "episode_quality_scores": np.asarray(episode_data["episode_quality_scores"], dtype=np.float32),
            "episode_quality_raw_scores": np.asarray(episode_data["episode_quality_raw_scores"], dtype=np.float32),
            "train_grads": train_grads,
            "eval_return_grad": eval_grad,
            "leaf_dims": {compressor.state.name: compressor.state.output_dim for compressor in self.compressors},
            "metadata": {
                "leaf_names": np.asarray([name for name, _ in self.lora_params], dtype=object),
                "leaf_shapes": np.asarray([tuple(param.shape) for _, param in self.lora_params], dtype=object),
                "episode_rewards": self.eval_dataset.episode_rewards,
                "episode_lengths": self.eval_dataset.episode_lengths,
                "train_window_indices": np.asarray(window_data["window_indices"], dtype=np.int64),
                "train_window_episode_indices": np.asarray(window_data["window_episode_indices"], dtype=np.int64),
                "train_window_local_indices": np.asarray(window_data["window_local_indices"], dtype=np.int64),
                "train_window_frame_starts": np.asarray(window_data["window_frame_starts"], dtype=np.int64),
                "train_window_frame_ends": np.asarray(window_data["window_frame_ends"], dtype=np.int64),
                "train_window_sample_counts": np.asarray(window_data["window_sample_counts"], dtype=np.int64),
                "train_window_is_partial": np.asarray(window_data["window_is_partial"], dtype=np.bool_),
                "train_sample_window_indices": np.asarray(window_data["sample_window_indices"], dtype=np.int64),
                "train_sample_window_local_indices": np.asarray(window_data["sample_window_local_indices"], dtype=np.int64),
                "train_sample_episode_indices": np.asarray(window_data["sample_episode_indices"], dtype=np.int64),
                "train_sample_frame_indices": np.asarray(window_data["sample_frame_indices"], dtype=np.int64),
                "train_dataset_sample_indices": np.asarray(window_data["sample_dataset_sample_indices"], dtype=np.int64),
                "train_episode_indices": np.asarray(episode_data["episode_indices"], dtype=np.int64),
                "train_episode_sample_counts": np.asarray(episode_data["episode_sample_counts"], dtype=np.int64),
                "train_episode_window_counts": np.asarray(episode_data["episode_window_counts"], dtype=np.int64),
                "train_score_aggregation": self.cfg.window_aggregation,
                "total_dataset_samples": int(len(self.train_dataset)),
                "finetune_config": dataclasses.asdict(self.finetune_cfg),
            },
        }
        if self.cfg.save_preconditioners:
            result["preconditioners"] = preconditioners

        with self.output_path.open("wb") as f:
            pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
        LOGGER.info(
            "Saved window-level GraSS result to %s (samples=%d, windows=%d, episodes=%d)",
            self.output_path,
            int(sample_scores.shape[0]),
            int(result["window_influence_scores"].shape[0]),
            int(result["episode_influence_scores"].shape[0]),
        )
        return result


def parse_args() -> WindowInfluenceConfig:
    parser = argparse.ArgumentParser(
        description="GraSS-style window influence attribution for OpenVLA LoRA checkpoints.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--eval-rollouts-dir", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--base-vla-path", type=str, default=None)
    parser.add_argument("--task-description", type=str, default=None)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-episodes", type=int, default=None)
    parser.add_argument("--max-eval-steps-per-episode", type=int, default=None)
    parser.add_argument("--stage1-microbatch-size", type=int, default=1)
    parser.add_argument("--stage2-batch-size", type=int, default=16)
    parser.add_argument("--progress-every", type=int, default=200)
    parser.add_argument("--sparsification", type=str, default="random_mask-32*32")
    parser.add_argument("--projection", type=str, default="sjlt-1024")
    parser.add_argument("--hessian", type=str, default="eFIM", choices=("eFIM", "Identity"))
    parser.add_argument("--damping", type=float, default=1e-4)
    parser.add_argument("--save-preconditioners", action="store_true")
    parser.add_argument("--stage-cache-prefix", type=Path, default=None)
    parser.add_argument("--window-size", type=int, default=50)
    parser.add_argument("--window-stride", type=int, default=None)
    parser.add_argument("--window-aggregation", type=str, choices=("sum", "mean"), default="sum")
    parser.add_argument(
        "--quality-score",
        type=str,
        choices=("mean", "zscore_mean", "sum", "zscore_sum"),
        default="zscore_mean",
    )
    args = parser.parse_args()
    return WindowInfluenceConfig(
        checkpoint_dir=args.checkpoint_dir,
        eval_rollouts_dir=args.eval_rollouts_dir,
        output_path=args.output_path,
        dataset_root=args.dataset_root,
        base_vla_path=args.base_vla_path,
        task_description=args.task_description,
        seed=args.seed,
        max_train_samples=args.max_train_samples,
        max_eval_episodes=args.max_eval_episodes,
        max_eval_steps_per_episode=args.max_eval_steps_per_episode,
        stage1_microbatch_size=args.stage1_microbatch_size,
        stage2_batch_size=args.stage2_batch_size,
        progress_every=args.progress_every,
        sparsification=args.sparsification,
        projection=args.projection,
        hessian=args.hessian,
        damping=args.damping,
        save_preconditioners=args.save_preconditioners,
        stage_cache_prefix=args.stage_cache_prefix,
        window_size=args.window_size,
        window_stride=args.window_stride,
        window_aggregation=args.window_aggregation,
        quality_score=args.quality_score,
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", force=True)
    runner = WindowInfluenceRunner(parse_args())
    runner.run()


if __name__ == "__main__":
    main()
