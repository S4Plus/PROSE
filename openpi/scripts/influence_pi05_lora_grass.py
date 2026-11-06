#!/usr/bin/env python3
"""GraSS-style influence attribution for OpenPI05 LoRA fine-tuning.

This script is intentionally independent from `scripts/influence_pi05_lora.py`.
It reuses OpenPI model / config / data-loading primitives, but implements a new
GraSS-style pipeline:

1. Build the OpenPI05 LoRA model and load a checkpoint.
2. Restrict gradients to LoRA trainable parameters only.
3. Compress per-sample train gradients with a two-stage block projector:
   - Stage 1: factorized sparsification over each parameter matrix
   - Stage 2: non-factorized projection on the flattened intermediate block
4. Estimate a compressed-space empirical Fisher preconditioner per parameter leaf.
5. Compute an eval return gradient from rollout chunks in the same compressed space.
6. Score train samples with:
      score_i = - <H^{-1} g_eval, g_train_i>
7. By default, aggregate sample scores into episode-level scores by summation.

The implementation is a practical OpenPI port of the GraSS approximation. It
does not monkey-patch JAX internals to avoid materializing gradients like the
PyTorch GraSS codebase does; instead, it applies GraSS-style two-stage
compression to LoRA parameter-gradient blocks after they are computed.
CUDA_VISIBLE_DEVICES=4 nohup uv run scripts/influence_pi05_lora_grass.py \
    --config-name pi05_can_mh_lora \
    --checkpoint-dir /workspace/liushuo/cupid/openpi/checkpoints/pi05_can_mh_lora/can_mh_delta_joint/10000 \
    --eval-rollouts-dir /workspace/liushuo/cupid/mimicgen/eval/tmp/can_mh_pi05_63_76.2/pkl \
    --output-path /workspace/liushuo/cupid/openpi/tmp/can_mh_grass_all/influence_grass.pkl \
    --sparsification random_mask-32*32 \
    --projection sjlt-1024 \
    --hessian eFIM \
    --damping 1e-4 \
    --stage1-microbatch-size 1 \
    --stage2-batch-size 32 \
    --progress-every 1 \
    > /workspace/liushuo/cupid/openpi/tmp/can_mh_grass_all/influence_grass_gpu3.log 2>&1 &


 CUDA_VISIBLE_DEVICES=2 nohup uv run python scripts/influence_pi05_lora_grass_leaf_subset.py \
    --config-name pi05_can_mh_lora \
    --checkpoint-dir /workspace/liushuo/cupid/openpi/checkpoints/pi05_can_mh_lora/can_mh_delta_joint/10000 \
    --eval-rollouts-dir /workspace/liushuo/cupid/mimicgen/eval/tmp/can_mh_pi05_63_76.2/pkl \
    --output-path /workspace/liushuo/cupid/openpi/tmp/can_mh_grass_action_last8/influence_grass.pkl \
    --score-level episode \
    --leaf-name-regex '.*action_expert.*' \
    --last-n-leaves 8 \
    --sparsification random_mask-32*32 \
    --projection sjlt-1024 \
    --hessian eFIM \
    --damping 1e-4 \
    --stage1-microbatch-size 1 \
    --stage2-batch-size 32 \
    --progress-every 1 \
    > /workspace/liushuo/cupid/openpi/tmp/can_mh_grass_action_last8/influence_grass.log 2>&1 &

"""

from __future__ import annotations

import argparse
import base64
import collections
import dataclasses
import io
import logging
import math
import multiprocessing
import pathlib
import pickle
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from PIL import Image
import torch

import openpi.models.model as _model
import openpi.policies.arm_prompt as arm_prompt_policy
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.sharding as sharding
import openpi.transforms as _transforms

LOGGER = logging.getLogger("influence_pi05_lora_grass")
MAX_DENSE_PROJECTION_BYTES = int(2 * 1024**3)


def _leaf_value(leaf: Any) -> Any:
    return getattr(leaf, "value", leaf)


def _replace_leaf_value(template: Any, value: Any) -> Any:
    replace_fn = getattr(template, "replace", None)
    if callable(replace_fn):
        return replace_fn(value=value)
    return value


def _path_to_name(path: Sequence[Any]) -> str:
    return "/".join(str(part) for part in path)


def _tree_paths_and_leaves(tree: Any) -> List[Tuple[Tuple[Any, ...], Any]]:
    flat, _ = jax.tree_util.tree_flatten_with_path(tree)
    normalized = []
    for path, leaf in flat:
        key_path = tuple(getattr(entry, "key", entry) for entry in path)
        normalized.append((key_path, leaf))
    return normalized


def _drop_none_entries(tree: Any) -> Any:
    if isinstance(tree, dict):
        return {k: _drop_none_entries(v) for k, v in tree.items() if v is not None}
    return tree


@dataclass
class ProjectionSpec:
    proj_type: str
    dims: Tuple[int, ...]


def _parse_sparsification(spec: str) -> ProjectionSpec:
    spec = spec.strip().lower()
    if spec == "identity":
        return ProjectionSpec("identity", (-1, -1))
    if "-" not in spec:
        raise ValueError(f"Invalid sparsification spec: {spec}")
    proj_type, dims = spec.split("-", 1)
    if "*" not in dims:
        raise ValueError(
            "Sparsification must use factorized dims like random_mask-128*128.",
        )
    d1, d2 = dims.split("*", 1)
    return ProjectionSpec(proj_type, (int(d1), int(d2)))


def _parse_projection(spec: str) -> ProjectionSpec:
    spec = spec.strip().lower()
    if spec == "identity":
        return ProjectionSpec("identity", (-1,))
    if "-" not in spec:
        raise ValueError(f"Invalid projection spec: {spec}")
    proj_type, dim = spec.split("-", 1)
    return ProjectionSpec(proj_type, (int(dim),))


@dataclass
class GrassConfig:
    config_name: str
    checkpoint_dir: str
    eval_rollouts_dir: str
    output_path: str
    score_level: str = "episode"

    batch_size: int = 8
    num_workers: int = 0
    seed: int = 0

    max_train_batches: Optional[int] = None
    max_eval_episodes: Optional[int] = None
    max_eval_steps_per_episode: Optional[int] = None
    stage1_microbatch_size: int = 1
    stage2_batch_size: int = 32
    progress_every: int = 1

    sparsification: str = "random_mask-32*32"
    projection: str = "sjlt-1024"
    hessian: str = "eFIM"
    damping: float = 1e-4

    save_preconditioners: bool = False
    stage_cache_prefix: Optional[str] = None


@dataclass
class EvalEpisode:
    observations: Dict[str, Any]
    actions: np.ndarray
    reward: float
    length: int
    executed_steps: np.ndarray | None = None


class EvalRolloutsDataset:
    def __init__(self, rollouts_dir: pathlib.Path, action_horizon: int = 1):
        self.rollouts_dir = pathlib.Path(rollouts_dir)
        if not self.rollouts_dir.exists():
            raise ValueError(f"Rollouts directory does not exist: {self.rollouts_dir}")
        self.action_horizon = max(int(action_horizon), 1)

        files = sorted(self.rollouts_dir.glob("episode_*.pkl"))
        if not files:
            files = sorted(self.rollouts_dir.glob("*.pkl"))
        if not files:
            raise ValueError(f"No rollout pickle files found in {self.rollouts_dir}")

        episodes: List[EvalEpisode] = []
        for file in files:
            try:
                with open(file, "rb") as f:
                    data = pickle.load(f)
                if isinstance(data, pd.DataFrame):
                    episode = self._load_from_pandas(data)
                elif isinstance(data, dict):
                    episode = self._load_from_dict(data)
                else:
                    raise TypeError(f"Unsupported rollout payload type: {type(data)}")
                episodes.append(episode)
            except Exception as exc:
                LOGGER.warning("Failed to load rollout file %s: %s", file, exc)
        if not episodes:
            raise ValueError(f"Failed to parse any rollouts in {self.rollouts_dir}")
        self.episodes = episodes

    @staticmethod
    def _expand_future_action_chunks(actions: np.ndarray, action_horizon: int) -> np.ndarray:
        actions = np.asarray(actions, dtype=np.float32)
        if actions.ndim != 2:
            raise ValueError(f"Expected 2D actions, got shape {actions.shape}")
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

    def _normalize_actions(self, actions: Any) -> np.ndarray:
        arr = np.asarray(actions, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[None, :]
        if arr.ndim == 2:
            return self._expand_future_action_chunks(arr, self.action_horizon)
        if arr.ndim == 3:
            if arr.shape[1] == self.action_horizon:
                return arr
            if arr.shape[1] == 1:
                return self._expand_future_action_chunks(arr[:, 0, :], self.action_horizon)
            if arr.shape[1] < self.action_horizon:
                pad = np.repeat(arr[:, -1:, :], self.action_horizon - arr.shape[1], axis=1)
                return np.concatenate([arr, pad], axis=1)
            return arr[:, : self.action_horizon, :]
        return arr

    @staticmethod
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
            if np.issubdtype(arr.dtype, np.floating):
                if arr.size > 0 and float(np.nanmax(arr)) <= 1.0 and float(np.nanmin(arr)) >= 0.0:
                    arr = arr * 255.0
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        return arr

    @staticmethod
    def _first_present(mapping: Mapping[str, Any], keys: Sequence[str]) -> Any:
        for key in keys:
            if key in mapping:
                return mapping[key]
        return None

    @classmethod
    def _to_numpy_tree(cls, value: Any) -> Any:
        if isinstance(value, dict):
            return {k: cls._to_numpy_tree(v) for k, v in value.items()}
        return np.asarray(value)

    @classmethod
    def _slice_step_tree(cls, tree: Any, index: int, leading_dim: int) -> Any:
        if isinstance(tree, dict):
            return {k: cls._slice_step_tree(v, index, leading_dim) for k, v in tree.items()}
        arr = np.asarray(tree)
        if arr.ndim > 0 and arr.shape[0] == leading_dim:
            return arr[index]
        return arr

    @classmethod
    def _stack_tree(cls, values: List[Any]) -> Any:
        if not values:
            raise ValueError("Cannot stack an empty tree")
        first = values[0]
        if isinstance(first, dict):
            return {k: cls._stack_tree([value[k] for value in values]) for k in first}
        return np.stack([np.asarray(value) for value in values], axis=0)

    @staticmethod
    def _extract_reward(data: Dict[str, Any], fallback: float = 0.0) -> float:
        reward = data.get("reward", data.get("success", fallback))
        if isinstance(reward, (list, np.ndarray)):
            return float(np.sum(reward))
        return float(reward)

    @staticmethod
    def _all_image_keys_have_length(images: Any, expected_length: int) -> bool:
        if not isinstance(images, dict) or not images:
            return False
        for value in images.values():
            arr = np.asarray(value)
            if arr.ndim == 0 or arr.shape[0] != expected_length:
                return False
        return True

    def _looks_like_query_repeated_rollout(self, data: Dict[str, Any]) -> bool:
        raw_observations = data.get("observations")
        metadata = data.get("metadata")
        if not isinstance(raw_observations, dict) or not isinstance(metadata, dict):
            return False

        action_chunks = metadata.get("action_chunks")
        if not isinstance(action_chunks, (list, tuple, np.ndarray)) or len(action_chunks) == 0:
            return False

        num_steps = len(action_chunks)
        try:
            actions = np.asarray(data["actions"])
        except Exception:
            return False
        if actions.ndim == 0 or actions.shape[0] != num_steps:
            return False

        states = np.asarray(raw_observations.get("state"))
        if states.ndim == 0 or states.shape[0] != num_steps:
            return False

        return self._all_image_keys_have_length(raw_observations.get("images"), num_steps)

    def _looks_like_per_inference_image_rollout(self, data: Dict[str, Any]) -> bool:
        raw_observations = data.get("observations")
        metadata = data.get("metadata")
        if not isinstance(raw_observations, dict) or not isinstance(metadata, dict):
            return False
        if metadata.get("image_storage") != "per_inference":
            return False

        action_chunks = metadata.get("action_chunks")
        if not isinstance(action_chunks, (list, tuple, np.ndarray)) or len(action_chunks) == 0:
            return False

        states = np.asarray(raw_observations.get("state"))
        if states.ndim == 0 or states.shape[0] == 0:
            return False

        return self._all_image_keys_have_length(raw_observations.get("images"), len(action_chunks))

    def _load_from_per_inference_image_dict(self, data: Dict[str, Any]) -> EvalEpisode:
        raw_observations = data["observations"]
        observations_np = self._to_numpy_tree(raw_observations)
        metadata = data.get("metadata", {})
        action_chunks_raw = metadata.get("action_chunks")
        if action_chunks_raw is None:
            raise ValueError("per-inference rollout payload is missing metadata.action_chunks")

        states = np.asarray(observations_np["state"], dtype=np.float32)
        images = observations_np.get("images", {})
        action_chunks = self._normalize_actions(
            np.stack([np.asarray(v, dtype=np.float32) for v in action_chunks_raw], axis=0)
        )

        executed_actions = np.asarray(data.get("actions", []), dtype=np.float32)
        num_executed = int(executed_actions.shape[0]) if executed_actions.ndim > 0 else int(states.shape[0])
        chunk_exec_steps = int(metadata.get("execution", {}).get("chunk_exec_steps") or self.action_horizon)
        chunk_exec_steps = max(chunk_exec_steps, 1)
        num_queries = min(
            int(action_chunks.shape[0]),
            int(math.ceil(max(num_executed, 1) / chunk_exec_steps)),
        )

        prompt = metadata.get("prompt")
        raw_prompt = observations_np.get("prompt")
        query_observations: List[Dict[str, Any]] = []
        query_actions: List[np.ndarray] = []
        executed_steps: List[int] = []
        for query_idx in range(num_queries):
            start = query_idx * chunk_exec_steps
            if start >= states.shape[0]:
                break

            sample: Dict[str, Any] = {"state": states[start]}
            if isinstance(images, dict):
                sample["images"] = {
                    key: np.asarray(value)[query_idx]
                    for key, value in images.items()
                    if np.asarray(value).ndim > 0 and np.asarray(value).shape[0] > query_idx
                }

            if prompt is not None:
                sample["prompt"] = np.asarray(prompt, dtype=np.str_)
            elif raw_prompt is not None:
                prompt_arr = np.asarray(raw_prompt)
                if prompt_arr.ndim > 0 and prompt_arr.shape[0] == states.shape[0]:
                    sample["prompt"] = prompt_arr[start]
                elif prompt_arr.ndim > 0 and prompt_arr.shape[0] > query_idx:
                    sample["prompt"] = prompt_arr[query_idx]
                else:
                    sample["prompt"] = prompt_arr

            valid_steps = min(chunk_exec_steps, max(num_executed - start, 0))
            if valid_steps <= 0:
                continue
            query_observations.append(sample)
            query_actions.append(np.asarray(action_chunks[query_idx], dtype=np.float32))
            executed_steps.append(int(valid_steps))

        if not query_observations:
            raise ValueError("per-inference rollout payload did not contain any executed policy queries")

        reward = self._extract_reward(data)
        stacked_observations = self._stack_tree(query_observations)
        stacked_actions = np.stack(query_actions, axis=0).astype(np.float32)
        executed_steps_np = np.asarray(executed_steps, dtype=np.int32)
        return EvalEpisode(
            observations=stacked_observations,
            actions=stacked_actions,
            reward=reward,
            length=len(query_actions),
            executed_steps=executed_steps_np,
        )

    def _infer_query_ranges(self, observations: Dict[str, Any], action_chunks: np.ndarray) -> List[Tuple[int, int]]:
        num_steps = int(action_chunks.shape[0])
        images = observations.get("images", {})
        ranges: List[Tuple[int, int]] = []
        start = 0

        for index in range(1, num_steps):
            same_chunk = bool(np.allclose(action_chunks[index - 1], action_chunks[index]))
            same_images = True
            if isinstance(images, dict):
                for value in images.values():
                    arr = np.asarray(value)
                    if arr.ndim == 0 or arr.shape[0] != num_steps:
                        continue
                    if not np.array_equal(arr[index - 1], arr[index]):
                        same_images = False
                        break
            if same_chunk and same_images:
                continue
            ranges.append((start, index))
            start = index

        ranges.append((start, num_steps))
        return ranges

    def _load_from_query_repeated_dict(self, data: Dict[str, Any]) -> EvalEpisode:
        raw_observations = data["observations"]
        observations_np = self._to_numpy_tree(raw_observations)
        metadata = data.get("metadata", {})
        action_chunks_raw = metadata.get("action_chunks")
        if action_chunks_raw is None:
            raise ValueError("query-level rollout payload is missing metadata.action_chunks")

        action_chunks = self._normalize_actions(np.stack([np.asarray(v, dtype=np.float32) for v in action_chunks_raw], axis=0))
        query_ranges = self._infer_query_ranges(observations_np, action_chunks)
        total_steps = int(action_chunks.shape[0])
        prompt = metadata.get("prompt")

        query_observations: List[Dict[str, Any]] = []
        query_actions: List[np.ndarray] = []
        executed_steps: List[int] = []
        for start, end in query_ranges:
            sample = self._slice_step_tree(observations_np, start, total_steps)
            if prompt is not None and "prompt" not in sample:
                sample["prompt"] = np.asarray(prompt, dtype=np.str_)
            query_observations.append(sample)
            query_actions.append(np.asarray(action_chunks[start], dtype=np.float32))
            executed_steps.append(int(end - start))

        reward = self._extract_reward(data)
        stacked_observations = self._stack_tree(query_observations)
        stacked_actions = np.stack(query_actions, axis=0).astype(np.float32)
        executed_steps_np = np.asarray(executed_steps, dtype=np.int32)
        return EvalEpisode(
            observations=stacked_observations,
            actions=stacked_actions,
            reward=reward,
            length=len(query_actions),
            executed_steps=executed_steps_np,
        )

    def _load_from_rows_dict(self, data: Dict[str, Any]) -> EvalEpisode:
        rows = data.get("rows")
        if not isinstance(rows, list) or not rows:
            raise ValueError("rows payload is empty")
        policy_calls = data.get("policy_calls")

        states: List[np.ndarray] = []
        actions: List[np.ndarray] = []
        base_images: List[np.ndarray | None] = []
        wrist_images: List[np.ndarray | None] = []
        right_wrist_images: List[np.ndarray | None] = []
        base_masks: List[bool] = []
        wrist_masks: List[bool] = []
        right_wrist_masks: List[bool] = []
        prompts: List[str | None] = []
        arm_ids: List[np.ndarray | None] = []
        policy_call_image_map: Dict[int, Dict[str, Any]] = {}
        policy_call_prompt_map: Dict[int, str] = {}
        policy_call_arm_id_map: Dict[int, Any] = {}

        if isinstance(policy_calls, list):
            for call in policy_calls:
                if not isinstance(call, dict):
                    continue
                call_id = call.get("call_id")
                request = call.get("request")
                images = request.get("images") if isinstance(request, dict) else None
                prompt = request.get("prompt") if isinstance(request, dict) else None
                arm_id = request.get("arm_id") if isinstance(request, dict) else None
                if call_id is None or not isinstance(images, dict):
                    if call_id is None:
                        continue
                    try:
                        call_id = int(call_id)
                    except (TypeError, ValueError):
                        continue
                else:
                    try:
                        call_id = int(call_id)
                    except (TypeError, ValueError):
                        continue
                    policy_call_image_map[call_id] = images
                if prompt is not None:
                    policy_call_prompt_map[call_id] = prompt
                if arm_id is not None:
                    policy_call_arm_id_map[call_id] = arm_id

        for row in rows:
            if not isinstance(row, dict):
                raise ValueError("rows contains non-dict entries")

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

            prompt = None
            arm_id = None
            policy_input = row.get("policy_input")
            if isinstance(policy_input, dict):
                prompt = policy_input.get("prompt")
                arm_id = policy_input.get("arm_id")

            images_map = None
            if isinstance(row_obs, dict):
                if isinstance(row_obs.get("image"), dict):
                    images_map = row_obs["image"]
                elif isinstance(row_obs.get("images"), dict):
                    images_map = row_obs["images"]
            if images_map is None:
                if isinstance(policy_input, dict) and isinstance(policy_input.get("images"), dict):
                    images_map = policy_input["images"]
            if images_map is None:
                policy_call_id = row.get("policy_call_id")
                try:
                    if policy_call_id is not None:
                        policy_call_id = int(policy_call_id)
                        images_map = policy_call_image_map.get(policy_call_id)
                        if prompt is None:
                            prompt = policy_call_prompt_map.get(policy_call_id)
                        if arm_id is None:
                            arm_id = policy_call_arm_id_map.get(policy_call_id)
                except (TypeError, ValueError):
                    images_map = None

            prompts.append(None if prompt is None else str(prompt))
            arm_ids.append(None if arm_id is None else np.asarray(arm_id, dtype=np.float32))

            if not isinstance(images_map, dict):
                base_images.append(None)
                wrist_images.append(None)
                right_wrist_images.append(None)
                base_masks.append(False)
                wrist_masks.append(False)
                right_wrist_masks.append(False)
                continue

            base_raw = self._first_present(
                images_map,
                (
                    "observation.images.agentview",
                    "observation.images.cam_high",
                    "agentview",
                    "base_0_rgb",
                    "cam_high",
                ),
            )
            wrist_raw = self._first_present(
                images_map,
                (
                    "observation.images.wrist",
                    "observation.images.cam_left_wrist",
                    "observation.images.robot0_eye_in_hand",
                    "wrist",
                    "left_wrist_0_rgb",
                    "cam_left_wrist",
                ),
            )
            right_wrist_raw = self._first_present(
                images_map,
                (
                    "observation.images.cam_right_wrist",
                    "right_wrist_0_rgb",
                    "cam_right_wrist",
                ),
            )
            base_img = self._decode_image(base_raw)
            wrist_img = self._decode_image(wrist_raw)
            right_wrist_img = self._decode_image(right_wrist_raw)
            base_images.append(base_img)
            wrist_images.append(wrist_img)
            right_wrist_images.append(right_wrist_img)
            base_masks.append(base_img is not None)
            wrist_masks.append(wrist_img is not None)
            right_wrist_masks.append(right_wrist_img is not None)

        states_np = np.stack(states, axis=0).astype(np.float32)
        actions_np = self._normalize_actions(np.stack(actions, axis=0))

        base_template = next((img for img in base_images if img is not None), None)
        wrist_template = next((img for img in wrist_images if img is not None), None)
        right_wrist_template = next((img for img in right_wrist_images if img is not None), None)
        if base_template is None and wrist_template is None and right_wrist_template is None:
            raise ValueError("No usable images in rows payload")
        if base_template is None and wrist_template is not None:
            base_template = np.zeros_like(wrist_template)
        elif base_template is None and right_wrist_template is not None:
            base_template = np.zeros_like(right_wrist_template)
        if wrist_template is None and base_template is not None:
            wrist_template = np.zeros_like(base_template)
        elif wrist_template is None and right_wrist_template is not None:
            wrist_template = np.zeros_like(right_wrist_template)
        if right_wrist_template is None and base_template is not None:
            right_wrist_template = np.zeros_like(base_template)
        elif right_wrist_template is None and wrist_template is not None:
            right_wrist_template = np.zeros_like(wrist_template)

        base_stack = np.stack(
            [img if img is not None else np.zeros_like(base_template) for img in base_images],
            axis=0,
        )
        wrist_stack = np.stack(
            [img if img is not None else np.zeros_like(wrist_template) for img in wrist_images],
            axis=0,
        )
        right_wrist_stack = np.stack(
            [img if img is not None else np.zeros_like(right_wrist_template) for img in right_wrist_images],
            axis=0,
        )

        reward = data.get("reward")
        if reward is None:
            row_rewards = [row.get("reward") for row in rows if row.get("reward") is not None]
            if row_rewards:
                reward = float(sum(float(v) for v in row_rewards))
            else:
                reward = float(data.get("success", rows[-1].get("success", 0.0)))
        elif isinstance(reward, (list, np.ndarray)):
            reward = float(np.sum(reward))
        else:
            reward = float(reward)

        observations = {
            "state": states_np,
            "image": {
                "base_0_rgb": base_stack,
                "left_wrist_0_rgb": wrist_stack,
                "right_wrist_0_rgb": right_wrist_stack,
            },
            "image_mask": {
                "base_0_rgb": np.asarray(base_masks, dtype=np.bool_),
                "left_wrist_0_rgb": np.asarray(wrist_masks, dtype=np.bool_),
                "right_wrist_0_rgb": np.asarray(right_wrist_masks, dtype=np.bool_),
            },
        }
        if any(prompt is not None for prompt in prompts):
            fallback_prompt = next((prompt for prompt in prompts if prompt is not None), "")
            observations["prompt"] = np.asarray(
                [prompt if prompt is not None else fallback_prompt for prompt in prompts],
                dtype=np.str_,
            )
        if any(arm_id is not None for arm_id in arm_ids):
            fallback_arm_id = next((arm_id for arm_id in arm_ids if arm_id is not None), np.asarray([0], dtype=np.float32))
            observations["arm_id"] = np.stack(
                [arm_id if arm_id is not None else fallback_arm_id for arm_id in arm_ids],
                axis=0,
            ).astype(np.float32)
        return EvalEpisode(observations=observations, actions=actions_np, reward=reward, length=len(actions_np))

    def _load_from_dict(self, data: Dict[str, Any]) -> EvalEpisode:
        if "rows" in data:
            return self._load_from_rows_dict(data)
        if self._looks_like_per_inference_image_rollout(data):
            return self._load_from_per_inference_image_dict(data)
        if self._looks_like_query_repeated_rollout(data):
            return self._load_from_query_repeated_dict(data)

        raw_observations = data.get("observations", {})
        if not isinstance(raw_observations, dict):
            raise ValueError("rollout dict payload must contain a dict observations field")
        observations = self._to_numpy_tree(raw_observations)
        actions = self._normalize_actions(data["actions"])
        reward = self._extract_reward(data)
        return EvalEpisode(observations=observations, actions=actions, reward=reward, length=len(actions))

    def _load_from_pandas(self, frame: pd.DataFrame) -> EvalEpisode:
        obs_dict: Dict[str, Any] = {}
        for col in frame.columns:
            if not col.startswith("observation."):
                continue
            key = col.replace("observation.", "")
            values = frame[col].values
            if len(values) > 0 and isinstance(values[0], np.ndarray):
                obs_dict[key] = np.stack([np.asarray(v) for v in values], axis=0)
            else:
                obs_dict[key] = np.asarray(values)

        if "action" in frame.columns:
            actions = np.stack([np.asarray(v) for v in frame["action"].values], axis=0)
        elif "actions" in frame.columns:
            actions = np.stack([np.asarray(v) for v in frame["actions"].values], axis=0)
        else:
            raise ValueError("No action column found")
        actions = self._normalize_actions(actions)

        if "reward" in frame.columns:
            reward = float(frame["reward"].sum())
        elif "next.reward" in frame.columns:
            reward = float(frame["next.reward"].sum())
        elif "success" in frame.columns:
            reward = float(frame["success"].iloc[-1])
        elif "next.success" in frame.columns:
            reward = float(frame["next.success"].iloc[-1])
        else:
            reward = 0.0
        return EvalEpisode(observations=obs_dict, actions=actions, reward=reward, length=len(frame))

    @property
    def episode_rewards(self) -> np.ndarray:
        return np.asarray([ep.reward for ep in self.episodes], dtype=np.float32)

    @property
    def episode_lengths(self) -> np.ndarray:
        return np.asarray([ep.length for ep in self.episodes], dtype=np.int64)


class TrainDataLoader:
    def __init__(
        self,
        data_config: _config.DataConfig,
        dataset: _data_loader.Dataset,
        batch_size: int,
        num_workers: int,
        *,
        shuffle: bool = False,
        seed: int = 0,
    ) -> None:
        if jax.process_count() > 1:
            raise NotImplementedError("GraSS train data loading with multiple processes is not supported.")

        if len(dataset) == 0:
            raise ValueError("Train dataset is empty.")

        generator = torch.Generator()
        generator.manual_seed(seed)
        mp_context = None
        if num_workers > 0:
            mp_context = multiprocessing.get_context("spawn")
        self._data_config = data_config
        self._torch_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            multiprocessing_context=mp_context,
            persistent_workers=num_workers > 0,
            collate_fn=_data_loader._collate_fn,  # noqa: SLF001
            worker_init_fn=_data_loader._worker_init_fn,  # noqa: SLF001
            drop_last=False,
            generator=generator,
        )
        self._num_batches = len(self._torch_loader)

    def data_config(self) -> _config.DataConfig:
        return self._data_config

    def __iter__(self):
        for batch in self._torch_loader:
            yield _model.Observation.from_dict(batch), batch["actions"]


@dataclass
class DenseProjector:
    matrix: jnp.ndarray

    def project(self, x: jnp.ndarray) -> jnp.ndarray:
        return x @ self.matrix


@dataclass
class CountSketchProjector:
    indices: jnp.ndarray
    signs: jnp.ndarray
    scale: float
    output_dim: int

    def project(self, x: jnp.ndarray) -> jnp.ndarray:
        def _single(vec: jnp.ndarray) -> jnp.ndarray:
            out = jnp.zeros((self.output_dim,), dtype=vec.dtype)
            out = out.at[self.indices].add(vec * self.signs)
            return out * self.scale

        if x.ndim == 1:
            return _single(x)
        return jax.vmap(_single)(x)


def _build_1d_projector(
    input_dim: int,
    output_dim: int,
    proj_type: str,
    seed: int,
) -> DenseProjector | CountSketchProjector | None:
    if proj_type == "identity" or output_dim < 0 or output_dim >= input_dim:
        return None

    rng = jax.random.key(seed)
    if proj_type == "normal":
        matrix = jax.random.normal(rng, (input_dim, output_dim), dtype=jnp.float32)
        matrix = matrix / jnp.sqrt(float(output_dim))
        return DenseProjector(matrix=matrix)

    if proj_type in {"sjlt", "countsketch"}:
        idx_rng, sign_rng = jax.random.split(rng)
        indices = jax.random.randint(idx_rng, (input_dim,), 0, output_dim)
        signs = jax.random.randint(sign_rng, (input_dim,), 0, 2).astype(jnp.float32) * 2.0 - 1.0
        scale = 1.0 / math.sqrt(float(output_dim))
        return CountSketchProjector(indices=indices, signs=signs, scale=scale, output_dim=output_dim)

    if proj_type == "random_mask":
        if output_dim <= 0:
            return None
        choice = np.random.default_rng(seed).choice(input_dim, size=min(output_dim, input_dim), replace=False)
        choice = np.sort(choice)
        matrix = np.zeros((input_dim, len(choice)), dtype=np.float32)
        matrix[choice, np.arange(len(choice))] = math.sqrt(input_dim / max(len(choice), 1))
        return DenseProjector(matrix=jnp.asarray(matrix))

    raise ValueError(f"Unsupported projector type: {proj_type}")


@dataclass
class LeafCompressionState:
    name: str
    original_shape: Tuple[int, ...]
    rows: int
    cols: int
    stage1_type: str
    stage2_type: str
    row_projector: DenseProjector | CountSketchProjector | None
    col_projector: DenseProjector | CountSketchProjector | None
    stage2_projector: DenseProjector | CountSketchProjector | None
    output_dim: int


class LeafCompressor:
    def __init__(
        self,
        name: str,
        shape: Sequence[int],
        sparsification: ProjectionSpec,
        projection: ProjectionSpec,
        seed: int,
    ) -> None:
        original_shape = tuple(int(dim) for dim in shape)
        if not original_shape:
            rows, cols = 1, 1
        elif len(original_shape) == 1:
            rows, cols = int(original_shape[0]), 1
        else:
            rows = int(np.prod(original_shape[:-1]))
            cols = int(original_shape[-1])

        row_dim_target, col_dim_target = sparsification.dims
        row_dim = rows if row_dim_target < 0 else min(row_dim_target, rows)
        col_dim = cols if col_dim_target < 0 else min(col_dim_target, cols)

        row_seed = seed
        col_seed = seed + 1
        stage2_seed = seed + 2

        row_projector = _build_1d_projector(rows, row_dim, sparsification.proj_type, row_seed)
        col_projector = _build_1d_projector(cols, col_dim, sparsification.proj_type, col_seed)
        inter_dim = row_dim * col_dim

        final_dim_target = projection.dims[0]
        final_dim = inter_dim if final_dim_target < 0 else min(final_dim_target, inter_dim)
        stage2_projector = _build_1d_projector(inter_dim, final_dim, projection.proj_type, stage2_seed)

        if stage2_projector is None:
            final_dim = inter_dim

        self.state = LeafCompressionState(
            name=name,
            original_shape=original_shape,
            rows=rows,
            cols=cols,
            stage1_type=sparsification.proj_type,
            stage2_type=projection.proj_type,
            row_projector=row_projector,
            col_projector=col_projector,
            stage2_projector=stage2_projector,
            output_dim=final_dim,
        )

    def compress(self, grad: jnp.ndarray) -> jnp.ndarray:
        state = self.state

        if grad.ndim == len(state.original_shape):
            grad = jnp.expand_dims(grad, axis=0)
            squeeze = True
        else:
            squeeze = False

        batch = grad.shape[0]
        grad = grad.reshape((batch, state.rows, state.cols))

        if state.row_projector is not None:
            if isinstance(state.row_projector, DenseProjector):
                grad = jnp.einsum("brc,rk->bkc", grad, state.row_projector.matrix)
            else:
                def _project_rows(matrix: jnp.ndarray) -> jnp.ndarray:
                    cols = matrix.shape[-1]
                    out = jnp.zeros((state.row_projector.output_dim, cols), dtype=matrix.dtype)
                    update = matrix * state.row_projector.signs[:, None]
                    out = out.at[state.row_projector.indices].add(update)
                    return out * state.row_projector.scale
                grad = jax.vmap(_project_rows)(grad)

        if state.col_projector is not None:
            if isinstance(state.col_projector, DenseProjector):
                grad = jnp.einsum("brc,ck->brk", grad, state.col_projector.matrix)
            else:
                def _project_cols(matrix: jnp.ndarray) -> jnp.ndarray:
                    rows = matrix.shape[0]
                    out = jnp.zeros((rows, state.col_projector.output_dim), dtype=matrix.dtype)
                    update = matrix * state.col_projector.signs[None, :]
                    out = out.at[:, state.col_projector.indices].add(update)
                    return out * state.col_projector.scale
                grad = jax.vmap(_project_cols)(grad)

        grad = grad.reshape((batch, -1))
        if state.stage2_projector is not None:
            grad = state.stage2_projector.project(grad)

        return grad[0] if squeeze else grad


class GrassOpenPIInfluence:
    def __init__(self, cfg: GrassConfig):
        self.cfg = cfg
        self.train_cfg = _config.get_config(cfg.config_name)
        if not bool(getattr(self.train_cfg.model, "pi05", False)):
            raise ValueError(f"Config '{cfg.config_name}' is not a PI05 config.")

        self.sparsification = _parse_sparsification(cfg.sparsification)
        self.projection = _parse_projection(cfg.projection)
        self._init_model()
        self._init_train_data()
        self._init_eval_data()
        self._init_compressors()
        self._build_grad_kernels()

        self.output_path = pathlib.Path(cfg.output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        if cfg.stage_cache_prefix:
            self.cache_dir = pathlib.Path(cfg.stage_cache_prefix)
        else:
            self.cache_dir = self.output_path.parent / f"{self.output_path.stem}.grass_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _init_model(self) -> None:
        rng = jax.random.key(self.cfg.seed)
        self.model = self.train_cfg.model.create(rng)
        self.graphdef, self.state = nnx.split(self.model)

        ckpt_path = pathlib.Path(self.cfg.checkpoint_dir) / "params"
        if not ckpt_path.exists():
            raise ValueError(f"Checkpoint does not exist: {ckpt_path}")
        ckpt_params = _model.restore_params(str(ckpt_path))
        self.state.replace_by_pure_dict(ckpt_params)
        self.model = nnx.merge(self.graphdef, self.state)

        trainable_state = nnx.state(self.model).filter(self.train_cfg.trainable_filter)
        self.trainable_template = trainable_state
        self.trainable_values = jax.tree.map(_leaf_value, trainable_state)

        path_leaf_pairs = _tree_paths_and_leaves(self.trainable_values)
        self.leaf_names = [_path_to_name(path) for path, _ in path_leaf_pairs]
        self.leaf_shapes = [tuple(int(dim) for dim in leaf.shape) for _, leaf in path_leaf_pairs]

        if not self.leaf_names:
            raise ValueError("No trainable LoRA parameters were found.")
        LOGGER.info("Loaded %d trainable LoRA parameter leaves.", len(self.leaf_names))

    def _init_train_data(self) -> None:
        data_config = self.train_cfg.data.create(self.train_cfg.assets_dirs, self.train_cfg.model)
        if data_config.rlds_data_dir is not None:
            self.train_loader = _data_loader.create_data_loader(
                self.train_cfg,
                sharding=None,
                shuffle=False,
                num_batches=None,
                skip_norm_stats=False,
                framework="jax",
            )
            return

        dataset = _data_loader.create_torch_dataset(
            data_config,
            self.train_cfg.model.action_horizon,
            self.train_cfg.model,
        )
        dataset = _data_loader.transform_dataset(dataset, data_config, skip_norm_stats=False)
        self.train_loader = TrainDataLoader(
            data_config,
            dataset,
            batch_size=int(self.train_cfg.batch_size),
            num_workers=int(self.train_cfg.num_workers),
            shuffle=False,
            seed=int(self.train_cfg.seed),
        )

    def _init_eval_data(self) -> None:
        self.eval_dataset = EvalRolloutsDataset(
            pathlib.Path(self.cfg.eval_rollouts_dir),
            action_horizon=int(self.train_cfg.model.action_horizon),
        )
        self.eval_canonical_transforms = self._build_eval_canonical_transforms()
        self.eval_input_transforms = self._build_eval_input_transforms()
        self.eval_prompt_transform = arm_prompt_policy.InjectArmIdIntoPrompt()

    def _init_compressors(self) -> None:
        compressors: List[LeafCompressor] = []
        total_dim = 0
        for idx, (name, shape) in enumerate(zip(self.leaf_names, self.leaf_shapes, strict=True)):
            compressor = LeafCompressor(
                name=name,
                shape=shape,
                sparsification=self.sparsification,
                projection=self.projection,
                seed=self.cfg.seed + idx * 97 + 1,
            )
            compressors.append(compressor)
            total_dim += compressor.state.output_dim
        self.compressors = compressors
        self.total_proj_dim = total_dim
        LOGGER.info(
            "Initialized %d GraSS leaf compressors, total compressed dim=%d.",
            len(compressors),
            total_dim,
        )

    def _build_grad_kernels(self) -> None:
        def single_sample_loss(values_pytree, obs_i, act_i, rng):
            model = self._apply_trainable_values(values_pytree)
            model.train()
            obs_batched = _drop_none_entries(jax.tree.map(lambda x: jnp.expand_dims(x, axis=0), obs_i))
            act_batched = jnp.expand_dims(act_i, axis=0)
            obs = _model.Observation.from_dict(obs_batched)
            loss = model.compute_loss(rng, obs, act_batched, train=True)
            return jnp.mean(loss)

        def chunk_loss(values_pytree, batch, rng):
            model = self._apply_trainable_values(values_pytree)
            model.train()
            action_mask = batch.get("eval_action_mask")
            obs_batch = _drop_none_entries({k: v for k, v in batch.items() if k not in ("actions", "eval_action_mask")})
            obs = _model.Observation.from_dict(obs_batch)
            loss = model.compute_loss(rng, obs, batch["actions"], train=True)
            if action_mask is not None:
                action_mask = jnp.asarray(action_mask, dtype=loss.dtype)
                masked_loss = loss * action_mask
                valid_steps = jnp.maximum(jnp.sum(action_mask, axis=-1), 1.0)
                return jnp.sum(jnp.sum(masked_loss, axis=-1) / valid_steps)
            return jnp.sum(jnp.mean(loss, axis=-1))

        self._train_batch_grad = jax.jit(
            jax.vmap(jax.grad(single_sample_loss), in_axes=(None, 0, 0, None)),
        )
        self._eval_chunk_grad = jax.jit(jax.grad(chunk_loss))

    def _apply_trainable_values(self, values_pytree):
        model = nnx.merge(self.graphdef, self.state)
        updated_state = jax.tree.map(_replace_leaf_value, self.trainable_template, values_pytree)
        nnx.update(model, updated_state)
        return model

    def _infer_train_num_batches(self) -> int:
        if self.cfg.max_train_batches is not None:
            if int(self.cfg.max_train_batches) <= 0:
                raise ValueError("--max-train-batches must be > 0")
            return int(self.cfg.max_train_batches)

        num_batches = getattr(self.train_loader, "_num_batches", None)
        if num_batches is not None and int(num_batches) > 0:
            return int(num_batches)

        loader_impl = getattr(self.train_loader, "_data_loader", None)
        if loader_impl is not None:
            num_batches = getattr(loader_impl, "_num_batches", None)
            if num_batches is not None and int(num_batches) > 0:
                return int(num_batches)
            torch_loader = getattr(loader_impl, "torch_loader", None)
            if torch_loader is not None:
                return int(len(torch_loader))
        raise RuntimeError("Unable to infer train batch count. Please set --max-train-batches.")

    def _normalize_train_batch(self, batch: Any) -> Dict[str, Any]:
        if isinstance(batch, dict):
            if "actions" not in batch:
                raise KeyError("Train batch missing actions")
            return batch
        if isinstance(batch, tuple) and len(batch) == 2:
            obs, actions = batch
            if isinstance(obs, _model.Observation):
                batch_dict = obs.to_dict()
            elif isinstance(obs, dict):
                batch_dict = dict(obs)
            else:
                raise TypeError(f"Unsupported observation type: {type(obs)}")
            batch_dict["actions"] = actions
            return batch_dict
        raise TypeError(f"Unsupported train batch type: {type(batch)}")

    def _build_eval_input_transforms(self) -> List[Any]:
        data_cfg = self.train_loader.data_config()
        transforms: List[Any] = []
        for transform in data_cfg.data_transforms.inputs:
            if isinstance(transform, (_transforms.DeltaActions, _transforms.SubsampleActions)):
                transforms.append(transform)

        norm_stats = {}
        if data_cfg.repo_id != "fake":
            if data_cfg.norm_stats is None:
                raise ValueError("Eval preprocessing requires norm_stats from the train config.")
            norm_stats = data_cfg.norm_stats

        transforms.append(_transforms.Normalize(norm_stats, use_quantiles=data_cfg.use_quantile_norm))
        transforms.extend(data_cfg.model_transforms.inputs)
        return transforms

    def _build_eval_canonical_transforms(self) -> List[Any]:
        data_cfg = self.train_loader.data_config()
        transforms: List[Any] = []
        transforms.extend(data_cfg.repack_transforms.inputs)
        for transform in data_cfg.data_transforms.inputs:
            if isinstance(transform, (_transforms.DeltaActions, _transforms.SubsampleActions)):
                continue
            transforms.append(transform)
        return transforms

    def _clone_eval_tree(self, tree: Any) -> Any:
        if isinstance(tree, dict):
            return {k: self._clone_eval_tree(v) for k, v in tree.items()}
        return np.asarray(tree)

    def _slice_eval_tree(self, tree: Any, index: int, leading_dim: int) -> Any:
        if isinstance(tree, dict):
            return {k: self._slice_eval_tree(v, index, leading_dim) for k, v in tree.items()}
        arr = np.asarray(tree)
        if arr.ndim == 0:
            return arr
        if arr.shape[0] == leading_dim:
            return arr[index]
        return arr

    def _stack_eval_trees(self, trees: List[Any]) -> Any:
        if not trees:
            raise ValueError("Cannot stack empty eval trees")
        first = trees[0]
        if isinstance(first, dict):
            return {k: self._stack_eval_trees([tree[k] for tree in trees]) for k in first}
        if first is None:
            return None
        return np.stack([np.asarray(tree) for tree in trees], axis=0)

    def _canonicalize_eval_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        if "image" in sample:
            return sample
        for transform in self.eval_canonical_transforms:
            sample = transform(sample)
        return sample

    def _prepare_eval_batch(
        self,
        observations: Dict[str, Any],
        actions: Any,
        executed_steps: Any | None = None,
    ) -> Dict[str, Any]:
        observations = self._clone_eval_tree(observations)
        actions = np.asarray(actions, dtype=np.float32)
        batch_size = int(actions.shape[0])
        action_horizon = int(actions.shape[1]) if actions.ndim >= 2 else 1
        if executed_steps is None:
            executed_steps_np = np.full((batch_size,), action_horizon, dtype=np.int32)
        else:
            executed_steps_np = np.asarray(executed_steps, dtype=np.int32).reshape(batch_size)
        samples = []
        action_masks = []
        for i in range(batch_size):
            sample = self._slice_eval_tree(observations, i, batch_size)
            sample["actions"] = np.asarray(actions[i], dtype=np.float32)
            sample = self._canonicalize_eval_sample(sample)
            if "prompt" in sample and "arm_id" in sample:
                sample = self.eval_prompt_transform(sample)
            for transform in self.eval_input_transforms:
                sample = transform(sample)
            samples.append(sample)
            valid_steps = int(np.clip(executed_steps_np[i], 0, action_horizon))
            mask = np.zeros((action_horizon,), dtype=np.float32)
            mask[:valid_steps] = 1.0
            action_masks.append(mask)

        batch = self._stack_eval_trees(samples)
        batch["state"] = np.asarray(batch["state"], dtype=np.float32)
        batch["actions"] = np.asarray(batch["actions"], dtype=np.float32)
        batch["image"] = {k: np.asarray(v, dtype=np.float32) for k, v in batch["image"].items()}
        batch["image_mask"] = {k: np.asarray(v, dtype=np.bool_) for k, v in batch["image_mask"].items()}
        batch["eval_action_mask"] = np.stack(action_masks, axis=0).astype(np.float32)

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
    def _iter_slices(length: int, chunk_size: int) -> Iterator[Tuple[int, int]]:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")
        for start in range(0, length, chunk_size):
            yield start, min(start + chunk_size, length)

    def _eval_subset(self) -> Tuple[List[EvalEpisode], jnp.ndarray, np.ndarray]:
        episodes = self.eval_dataset.episodes
        if self.cfg.max_eval_episodes is not None:
            if int(self.cfg.max_eval_episodes) <= 0:
                raise ValueError("--max-eval-episodes must be > 0")
            episodes = episodes[: int(self.cfg.max_eval_episodes)]

        max_steps = self.cfg.max_eval_steps_per_episode
        if max_steps is not None and int(max_steps) <= 0:
            raise ValueError("--max-eval-steps-per-episode must be > 0")

        lengths = np.asarray(
            [
                min(ep.length, int(max_steps)) if max_steps is not None else ep.length
                for ep in episodes
            ],
            dtype=np.int64,
        )
        rewards = jnp.asarray([ep.reward for ep in episodes], dtype=jnp.float32)
        return episodes, rewards, lengths

    def _compress_leaf_batch(self, batched_grad_pytree: Any) -> Dict[str, np.ndarray]:
        compressed: Dict[str, np.ndarray] = {}
        for compressor, (_, leaf_grad) in zip(
            self.compressors,
            _tree_paths_and_leaves(batched_grad_pytree),
            strict=True,
        ):
            compressed[compressor.state.name] = np.asarray(compressor.compress(leaf_grad), dtype=np.float32)
        return compressed

    def _compress_leaf_single(self, grad_pytree: Any) -> Dict[str, np.ndarray]:
        compressed: Dict[str, np.ndarray] = {}
        for compressor, (_, leaf_grad) in zip(
            self.compressors,
            _tree_paths_and_leaves(grad_pytree),
            strict=True,
        ):
            compressed[compressor.state.name] = np.asarray(compressor.compress(leaf_grad), dtype=np.float32)
        return compressed

    def _cache_train_grads(self) -> Dict[str, np.ndarray]:
        LOGGER.info(
            "Stage 1: caching GraSS-compressed train gradients (microbatch_size=%d)",
            int(self.cfg.stage1_microbatch_size),
        )
        total_batches = self._infer_train_num_batches()
        values = self.trainable_values
        rng = jax.random.key(self.cfg.seed + 100)
        microbatch_size = int(self.cfg.stage1_microbatch_size)
        if microbatch_size <= 0:
            raise ValueError("--stage1-microbatch-size must be > 0")

        buffers = {name: [] for name in self.leaf_names}
        stage_start = time.perf_counter()
        progress_every = max(int(self.cfg.progress_every), 1)

        train_iter = iter(self.train_loader)
        for batch_idx in range(total_batches):
            try:
                batch = self._normalize_train_batch(next(train_iter))
            except StopIteration:
                LOGGER.warning("Train loader ended early at batch %d/%d", batch_idx + 1, total_batches)
                break
            obs_batch = _drop_none_entries({k: v for k, v in batch.items() if k != "actions"})
            actions = batch["actions"]
            batch_size = int(actions.shape[0])
            for micro_idx, (start, end) in enumerate(self._iter_slices(batch_size, microbatch_size)):
                obs_micro = jax.tree.map(lambda x, s=start, e=end: x[s:e], obs_batch)
                act_micro = actions[start:end]
                micro_rng = jax.random.fold_in(rng, micro_idx)
                grads = self._train_batch_grad(values, obs_micro, act_micro, micro_rng)
                compressed = self._compress_leaf_batch(grads)
                for name, arr in compressed.items():
                    buffers[name].append(arr)
            rng = jax.random.fold_in(rng, batch_idx + 1)

            done = batch_idx + 1
            if done == 1 or done % progress_every == 0 or done == total_batches:
                elapsed = time.perf_counter() - stage_start
                LOGGER.info(
                    "  [Stage1] batch %d/%d done, elapsed %.2fs",
                    done,
                    total_batches,
                    elapsed,
                )

        leaf_state_map = self._leaf_state_map()
        stacked = {
            name: (
                np.concatenate(parts, axis=0)
                if parts
                else np.zeros((0, leaf_state_map[name].output_dim), dtype=np.float32)
            )
            for name, parts in buffers.items()
        }
        total_samples = next(iter(stacked.values())).shape[0]
        LOGGER.info("Stage 1 complete: %d train samples cached.", total_samples)
        return stacked

    def _compute_preconditioners(self, train_grads: Dict[str, np.ndarray]) -> Dict[str, np.ndarray | None]:
        LOGGER.info("Stage 2: computing compressed-space preconditioners")
        if self.cfg.hessian.lower() == "identity":
            return {name: None for name in self.leaf_names}

        preconditioners: Dict[str, np.ndarray | None] = {}
        for name in self.leaf_names:
            x = np.asarray(train_grads[name], dtype=np.float32)
            if x.shape[1] == 0:
                preconditioners[name] = None
                continue
            hessian = (x.T @ x) / max(float(x.shape[0]), 1.0)
            damping = float(self.cfg.damping)
            hessian = hessian + damping * np.eye(hessian.shape[0], dtype=np.float32)
            preconditioners[name] = np.linalg.pinv(hessian, hermitian=True).astype(np.float32)
            LOGGER.info("  [Stage2] leaf %s preconditioner shape=%s", name, hessian.shape)
        return preconditioners

    def _compute_eval_return_grad(self) -> Dict[str, np.ndarray]:
        LOGGER.info("Stage 3: computing GraSS-compressed eval return gradient")
        episodes, rewards, lengths = self._eval_subset()
        if not episodes:
            LOGGER.warning("No eval episodes selected; returning zero eval gradient.")
            return {
                name: np.zeros((state.output_dim,), dtype=np.float32)
                for name, state in self._leaf_state_map().items()
            }

        centered_rewards = rewards - jnp.mean(rewards)
        values = self.trainable_values
        stage2_batch_size = int(self.cfg.stage2_batch_size)
        base_rng = jax.random.key(self.cfg.seed + 200)
        total_chunks = int(sum(math.ceil(length / stage2_batch_size) for length in lengths))
        done_chunks = 0
        progress_every = max(int(self.cfg.progress_every), 1)
        stage_start = time.perf_counter()

        return_grads = {
            name: np.zeros((compressor.state.output_dim,), dtype=np.float32)
            for name, compressor in zip(self.leaf_names, self.compressors, strict=True)
        }

        for ep_idx, episode in enumerate(episodes):
            episode_grads = {name: np.zeros_like(vec) for name, vec in return_grads.items()}
            episode_length = int(lengths[ep_idx])
            for chunk_idx, (start, end) in enumerate(self._iter_slices(episode_length, stage2_batch_size)):
                obs_t = jax.tree.map(lambda x, s=start, e=end: x[s:e], episode.observations)
                act_t = episode.actions[start:end]
                exec_t = None if episode.executed_steps is None else episode.executed_steps[start:end]
                batch_t = self._prepare_eval_batch(obs_t, act_t, exec_t)
                chunk_rng = jax.random.fold_in(jax.random.fold_in(base_rng, ep_idx), chunk_idx)
                grad_pytree = self._eval_chunk_grad(values, batch_t, chunk_rng)
                compressed = self._compress_leaf_single(grad_pytree)
                for name, vec in compressed.items():
                    episode_grads[name] += vec
                done_chunks += 1

            weight = float(centered_rewards[ep_idx])
            for name in return_grads:
                return_grads[name] += weight * episode_grads[name]

            done_eps = ep_idx + 1
            if done_eps == 1 or done_eps % progress_every == 0 or done_eps == len(episodes):
                elapsed = time.perf_counter() - stage_start
                LOGGER.info(
                    "  [Stage3] episode %d/%d done, chunks %d/%d, elapsed %.2fs",
                    done_eps,
                    len(episodes),
                    done_chunks,
                    total_chunks,
                    elapsed,
                )

        for name in return_grads:
            return_grads[name] /= max(float(len(episodes)), 1.0)
        return return_grads

    def _score(self, train_grads: Dict[str, np.ndarray], preconditioners: Dict[str, np.ndarray | None], eval_grad: Dict[str, np.ndarray]) -> np.ndarray:
        LOGGER.info("Stage 4: scoring train-sample influences")
        total_samples = next(iter(train_grads.values())).shape[0]
        scores = np.zeros((total_samples,), dtype=np.float32)
        for name in self.leaf_names:
            leaf_train = np.asarray(train_grads[name], dtype=np.float32)
            leaf_eval = np.asarray(eval_grad[name], dtype=np.float32)
            precond = preconditioners[name]
            target = leaf_eval if precond is None else precond @ leaf_eval
            scores -= leaf_train @ target
        return scores

    def _get_train_sample_metadata(self, total_samples: int) -> Dict[str, np.ndarray | int]:
        data_cfg = self.train_loader.data_config()
        if data_cfg.rlds_data_dir is not None:
            raise NotImplementedError("Episode-level train scores are not supported for RLDS datasets.")

        dataset = _data_loader.create_torch_dataset(
            data_cfg,
            self.train_cfg.model.action_horizon,
            self.train_cfg.model,
        )
        total_dataset_samples = int(len(dataset))
        if total_samples > total_dataset_samples:
            raise ValueError(
                f"Scored sample count {total_samples} exceeds dataset size {total_dataset_samples}. "
                "This script currently expects Stage1 to iterate over at most one full dataset pass."
            )

        dataset_positions = np.arange(total_samples, dtype=np.int64)
        dataset_sample_indices = np.empty((total_samples,), dtype=np.int64)
        episode_indices = np.empty((total_samples,), dtype=np.int64)

        for i, dataset_pos in enumerate(dataset_positions.tolist()):
            item = dataset[dataset_pos]
            if "episode_index" not in item:
                raise KeyError("Train dataset item does not contain episode_index; cannot aggregate to episode scores.")
            episode_indices[i] = int(np.asarray(item["episode_index"]).reshape(()))
            if "index" in item:
                dataset_sample_indices[i] = int(np.asarray(item["index"]).reshape(()))
            else:
                dataset_sample_indices[i] = dataset_pos

        return {
            "dataset_positions": dataset_positions,
            "dataset_sample_indices": dataset_sample_indices,
            "episode_indices": episode_indices,
            "total_dataset_samples": total_dataset_samples,
            "usable_epoch_samples": total_dataset_samples,
        }

    def _aggregate_scores_by_episode(self, sample_scores: np.ndarray) -> Dict[str, np.ndarray | int]:
        metadata = self._get_train_sample_metadata(int(sample_scores.shape[0]))
        episode_indices = np.asarray(metadata["episode_indices"], dtype=np.int64)

        episode_buckets: dict[int, list[int]] = collections.defaultdict(list)
        for sample_idx, episode_idx in enumerate(episode_indices.tolist()):
            episode_buckets[episode_idx].append(sample_idx)

        unique_episode_indices = np.asarray(sorted(episode_buckets), dtype=np.int64)
        episode_scores = np.zeros((len(unique_episode_indices),), dtype=np.float32)
        episode_mean_scores = np.zeros((len(unique_episode_indices),), dtype=np.float32)
        episode_sample_counts = np.zeros((len(unique_episode_indices),), dtype=np.int64)

        for out_idx, episode_idx in enumerate(unique_episode_indices.tolist()):
            sample_ids = np.asarray(episode_buckets[episode_idx], dtype=np.int64)
            values = np.asarray(sample_scores[sample_ids], dtype=np.float32)
            episode_scores[out_idx] = float(np.sum(values))
            episode_mean_scores[out_idx] = float(np.mean(values))
            episode_sample_counts[out_idx] = int(sample_ids.size)

        return {
            "episode_indices": unique_episode_indices,
            "episode_scores": episode_scores,
            "episode_mean_scores": episode_mean_scores,
            "episode_sample_counts": episode_sample_counts,
            "sample_episode_indices": episode_indices,
            "dataset_sample_indices": np.asarray(metadata["dataset_sample_indices"], dtype=np.int64),
            "dataset_positions": np.asarray(metadata["dataset_positions"], dtype=np.int64),
            "total_dataset_samples": int(metadata["total_dataset_samples"]),
            "usable_epoch_samples": int(metadata["usable_epoch_samples"]),
        }

    def _leaf_state_map(self) -> Dict[str, LeafCompressionState]:
        return {compressor.state.name: compressor.state for compressor in self.compressors}

    def run(self) -> Dict[str, Any]:
        cache_train = self.cache_dir / f"{self.output_path.stem}_stage1_train_grads.pkl"
        cache_precond = self.cache_dir / f"{self.output_path.stem}_stage2_preconditioners.pkl"
        cache_eval = self.cache_dir / f"{self.output_path.stem}_stage3_eval_return_grad.pkl"

        if cache_train.exists():
            with open(cache_train, "rb") as f:
                train_grads = pickle.load(f)
            LOGGER.info("Loaded cached train grads from %s", cache_train)
        else:
            train_grads = self._cache_train_grads()
            with open(cache_train, "wb") as f:
                pickle.dump(train_grads, f, protocol=pickle.HIGHEST_PROTOCOL)

        if cache_precond.exists():
            with open(cache_precond, "rb") as f:
                preconditioners = pickle.load(f)
            LOGGER.info("Loaded cached preconditioners from %s", cache_precond)
        else:
            preconditioners = self._compute_preconditioners(train_grads)
            with open(cache_precond, "wb") as f:
                pickle.dump(preconditioners, f, protocol=pickle.HIGHEST_PROTOCOL)

        if cache_eval.exists():
            with open(cache_eval, "rb") as f:
                eval_grad = pickle.load(f)
            LOGGER.info("Loaded cached eval return gradient from %s", cache_eval)
        else:
            eval_grad = self._compute_eval_return_grad()
            with open(cache_eval, "wb") as f:
                pickle.dump(eval_grad, f, protocol=pickle.HIGHEST_PROTOCOL)

        sample_scores = self._score(train_grads, preconditioners, eval_grad)
        score_level = self.cfg.score_level.lower()
        if score_level == "episode":
            episode_score_data = self._aggregate_scores_by_episode(sample_scores)
            influence_scores = np.asarray(episode_score_data["episode_scores"], dtype=np.float32)
            LOGGER.info(
                "Aggregated %d sample scores into %d episode scores.",
                int(sample_scores.shape[0]),
                int(influence_scores.shape[0]),
            )
        elif score_level == "sample":
            episode_score_data = None
            influence_scores = sample_scores
        else:
            raise ValueError(f"Unsupported score level: {self.cfg.score_level}")

        result = {
            "config": dataclasses.asdict(self.cfg),
            "score_level": score_level,
            "influence_scores": influence_scores,
            "sample_influence_scores": sample_scores,
            "train_grads": train_grads,
            "eval_return_grad": eval_grad,
            "leaf_dims": {name: state.output_dim for name, state in self._leaf_state_map().items()},
            "metadata": {
                "leaf_names": np.asarray(self.leaf_names, dtype=object),
                "leaf_shapes": np.asarray(self.leaf_shapes, dtype=object),
                "episode_rewards": self.eval_dataset.episode_rewards,
                "episode_lengths": self.eval_dataset.episode_lengths,
            },
        }
        if episode_score_data is not None:
            result["episode_influence_scores"] = np.asarray(episode_score_data["episode_scores"], dtype=np.float32)
            result["episode_mean_influence_scores"] = np.asarray(
                episode_score_data["episode_mean_scores"], dtype=np.float32
            )
            result["metadata"]["train_episode_indices"] = np.asarray(
                episode_score_data["episode_indices"], dtype=np.int64
            )
            result["metadata"]["train_episode_sample_counts"] = np.asarray(
                episode_score_data["episode_sample_counts"], dtype=np.int64
            )
            result["metadata"]["train_sample_episode_indices"] = np.asarray(
                episode_score_data["sample_episode_indices"], dtype=np.int64
            )
            result["metadata"]["train_dataset_sample_indices"] = np.asarray(
                episode_score_data["dataset_sample_indices"], dtype=np.int64
            )
            result["metadata"]["train_dataset_positions"] = np.asarray(
                episode_score_data["dataset_positions"], dtype=np.int64
            )
            result["metadata"]["train_score_aggregation"] = "sum"
            result["metadata"]["total_dataset_samples"] = int(episode_score_data["total_dataset_samples"])
            result["metadata"]["usable_epoch_samples"] = int(episode_score_data["usable_epoch_samples"])
        if self.cfg.save_preconditioners:
            result["preconditioners"] = preconditioners

        with open(self.output_path, "wb") as f:
            pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
        LOGGER.info("Saved GraSS influence result to %s", self.output_path)
        return result


def parse_args() -> GrassConfig:
    parser = argparse.ArgumentParser(
        description="GraSS-style OpenPI05 LoRA influence attribution",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config-name", required=True, help="OpenPI training config name")
    parser.add_argument("--checkpoint-dir", required=True, help="Checkpoint directory containing params/")
    parser.add_argument("--eval-rollouts-dir", required=True, help="Directory containing eval rollout PKLs")
    parser.add_argument("--output-path", required=True, help="Output pickle path")
    parser.add_argument(
        "--score-level",
        choices=("episode", "sample"),
        default="episode",
        help="Final output score granularity. Episode-level sums sample scores within each episode.",
    )

    parser.add_argument("--batch-size", type=int, default=8, help="Unused placeholder for parity with other scripts")
    parser.add_argument("--num-workers", type=int, default=0, help="Unused placeholder for parity with other scripts")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--max-train-batches", type=int, default=None, help="Max Stage1 train batches")
    parser.add_argument("--max-eval-episodes", type=int, default=None, help="Max eval episodes for Stage3")
    parser.add_argument(
        "--max-eval-steps-per-episode",
        type=int,
        default=None,
        help="Max steps per eval episode for Stage3",
    )
    parser.add_argument(
        "--stage1-microbatch-size",
        type=int,
        default=1,
        help="Per-sample grad microbatch size for Stage1; lower uses less memory",
    )
    parser.add_argument("--stage2-batch-size", type=int, default=32, help="Eval rollout chunk size")
    parser.add_argument("--progress-every", type=int, default=1, help="Progress logging interval")

    parser.add_argument(
        "--sparsification",
        type=str,
        default="random_mask-32*32",
        help="GraSS stage-1 factorized compression, e.g. random_mask-32*32",
    )
    parser.add_argument(
        "--projection",
        type=str,
        default="sjlt-1024",
        help="GraSS stage-2 projection, e.g. sjlt-1024 or identity",
    )
    parser.add_argument(
        "--hessian",
        type=str,
        default="eFIM",
        choices=("eFIM", "Identity"),
        help="Compressed-space Hessian approximation",
    )
    parser.add_argument("--damping", type=float, default=1e-4, help="Preconditioner damping")
    parser.add_argument("--save-preconditioners", action="store_true", help="Include preconditioners in final output")
    parser.add_argument("--stage-cache-prefix", type=str, default=None, help="Optional stage cache directory")

    args = parser.parse_args()
    return GrassConfig(
        config_name=args.config_name,
        checkpoint_dir=args.checkpoint_dir,
        eval_rollouts_dir=args.eval_rollouts_dir,
        output_path=args.output_path,
        score_level=args.score_level,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        max_train_batches=args.max_train_batches,
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
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        force=True,
    )
    cfg = parse_args()
    runner = GrassOpenPIInfluence(cfg)
    runner.run()


if __name__ == "__main__":
    main()
