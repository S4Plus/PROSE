#!/usr/bin/env python3
"""
Serve a fine-tuned OpenVLA policy over the OpenPI-style websocket protocol.
"""

from __future__ import annotations

import argparse
import asyncio
import functools
import http
import io
import logging
import sys
import time
import threading
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import msgpack
import numpy as np
import torch
from PIL import Image
import websockets
import websockets.asyncio.server as ws_server
import websockets.frames
from transformers import BatchEncoding

from experiments.robot.openvla_utils import normalize_proprio
from prismatic.vla.lerobot_openvla import LeRobotDatasetMetadata, load_checkpoint_bundle
import prismatic.vla.constants as vla_constants

LOGGER = logging.getLogger("serve_policy_openvla")


def _pack_array(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        if obj.dtype.kind in ("V", "O", "c"):
            raise ValueError(f"Unsupported dtype for msgpack: {obj.dtype}")
        return {
            b"__ndarray__": True,
            b"data": obj.tobytes(),
            b"dtype": obj.dtype.str,
            b"shape": obj.shape,
        }
    if isinstance(obj, np.generic):
        return {
            b"__npgeneric__": True,
            b"data": obj.item(),
            b"dtype": obj.dtype.str,
        }
    return obj


def _unpack_array(obj: Mapping[Any, Any]) -> Any:
    if b"__ndarray__" in obj:
        return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"])
    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])
    return obj


Packer = functools.partial(msgpack.Packer, default=_pack_array)
packb = functools.partial(msgpack.packb, default=_pack_array)
unpackb = functools.partial(msgpack.unpackb, object_hook=_unpack_array)


def _decode_image(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        image = value
    elif isinstance(value, (bytes, bytearray)):
        image = np.asarray(Image.open(io.BytesIO(value)).convert("RGB"))
    elif isinstance(value, str):
        if value.startswith("data:image"):
            value = value.split(",", 1)[-1]
        import base64

        image = np.asarray(Image.open(io.BytesIO(base64.b64decode(value))).convert("RGB"))
    else:
        raise TypeError(f"Unsupported image payload type: {type(value)}")

    if image.ndim != 3:
        raise ValueError(f"Expected HWC image, got shape {image.shape}")
    if image.shape[0] in (1, 3) and image.shape[-1] not in (1, 3):
        image = np.transpose(image, (1, 2, 0))
    if image.dtype != np.uint8:
        if np.issubdtype(image.dtype, np.floating) and image.size > 0 and float(np.nanmax(image)) <= 1.0:
            image = image * 255.0
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def _first_present(mapping: Mapping[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


@dataclass
class Args:
    checkpoint: Path
    host: str = "0.0.0.0"
    port: int = 8000
    base_vla_path: Optional[str] = None
    default_prompt: Optional[str] = None
    dataset_root: Optional[Path] = None


def resolve_checkpoint_dir(path_str: Path | str) -> Path:
    root = Path(path_str).expanduser().resolve()
    if root.is_file():
        root = root.parent

    def _is_checkpoint_dir(path: Path) -> bool:
        return (
            path.is_dir()
            and (path / "lerobot_finetune_config.json").is_file()
            and (path / "lora_adapter").is_dir()
        )

    if _is_checkpoint_dir(root):
        return root

    sibling_candidates: list[tuple[int, Path]] = []
    if root.parent.exists():
        pattern = f"{root.name}--*_chkpt"
        for child in root.parent.glob(pattern):
            if not _is_checkpoint_dir(child):
                continue
            try:
                step = int(child.name[len(root.name) + 2 : -len("_chkpt")])
            except Exception:
                step = -1
            sibling_candidates.append((step, child))
    if sibling_candidates:
        sibling_candidates.sort(key=lambda item: (item[0], item[1].name))
        return sibling_candidates[-1][1]

    child_candidates: list[tuple[int, Path]] = []
    if root.is_dir():
        for child in root.iterdir():
            if not _is_checkpoint_dir(child) or not child.name.endswith("_chkpt"):
                continue
            stem = child.name[: -len("_chkpt")]
            step = -1
            if "--" in stem:
                maybe_step = stem.rsplit("--", 1)[-1]
                if maybe_step.isdigit():
                    step = int(maybe_step)
            child_candidates.append((step, child))
    if child_candidates:
        child_candidates.sort(key=lambda item: (item[0], item[1].name))
        return child_candidates[-1][1]

    raise FileNotFoundError(f"Could not resolve OpenVLA checkpoint dir from '{root}'.")


class OpenVLAWebsocketPolicyServer:
    def __init__(self, args: Args):
        self.args = args
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.input_dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        self.checkpoint_dir = resolve_checkpoint_dir(args.checkpoint)
        bundle = load_checkpoint_bundle(self.checkpoint_dir, device=self.device, base_vla_path=args.base_vla_path)
        self.finetune_cfg = bundle["cfg"]
        self.processor = bundle["processor"]
        self.vla = bundle["vla"]
        self.action_head = bundle["action_head"]
        self.proprio_projector = bundle["proprio_projector"]
        self.metadata: LeRobotDatasetMetadata = bundle["metadata"]
        self.lock = threading.Lock()
        if args.dataset_root is not None and args.dataset_root != self.metadata.dataset_root:
            self.metadata = LeRobotDatasetMetadata(
                args.dataset_root,
                dataset_name=self.finetune_cfg.dataset_name,
                task_description_override=self.finetune_cfg.task_description,
                norm_stats_path=self.finetune_cfg.norm_stats_path,
            )

        self.default_prompt = (
            args.default_prompt
            or self.finetune_cfg.task_description
            or self.metadata.default_task_description
            or self.finetune_cfg.dataset_name
        )
        self.training_prompt = (
            self.finetune_cfg.task_description
            or self.metadata.default_task_description
            or self.finetune_cfg.dataset_name
        )
        self._warned_prompt_mismatches: set[str] = set()
        self.unnorm_key = self.metadata.dataset_name
        self._maybe_override_action_stats_from_dataset()
        self._set_norm_stats()
        self._patch_openvla_constants()
        self.vla.eval()
        self.action_head.eval()
        if self.proprio_projector is not None:
            # `predict_action()` casts proprio inputs to the vision embedding dtype.
            # Keep the projector in the same dtype to avoid bf16/fp32 matmul mismatches.
            self.proprio_projector = self.proprio_projector.to(device=self.device, dtype=self.input_dtype)
            self.proprio_projector.eval()

        self.image_keys = ["observation.images.agentview"]
        if int(self.finetune_cfg.num_images_in_input) > 1:
            self.image_keys.append("observation.images.wrist")
        self.expect_state = bool(self.finetune_cfg.use_proprio and self.proprio_projector is not None)

        self.server_metadata = {
            "policy_type": "openvla",
            "protocol": "websocket-msgpack-numpy",
            "image_keys": self.image_keys,
            "expect_state": self.expect_state,
            "state_dim": int(self.metadata.state_dim) if self.expect_state else None,
            "state_names": self._feature_names("observation.state") if self.expect_state else None,
            "action_dim": int(self.metadata.action_dim),
            "action_names": self._feature_names("action"),
            "action_horizon": int(self.finetune_cfg.action_chunk_size),
            "num_images_in_input": int(self.finetune_cfg.num_images_in_input),
            "default_prompt": self.default_prompt,
            "checkpoint_dir": str(self.checkpoint_dir),
            "dataset_name": self.metadata.dataset_name,
            "train_primary_image_key": self.metadata.primary_image_key,
            "train_wrist_image_key": self.metadata.wrist_image_key,
        }
        self._log_io_alignment_summary()

    def _feature_names(self, feature_key: str) -> Optional[list[str]]:
        feature = self.metadata.info.get("features", {}).get(feature_key)
        if not isinstance(feature, Mapping):
            return None
        names = feature.get("names")
        if not isinstance(names, Sequence) or isinstance(names, (str, bytes)):
            return None
        out = [str(name) for name in names]
        return out if out else None

    def _log_io_alignment_summary(self) -> None:
        state_names = self.server_metadata.get("state_names")
        action_names = self.server_metadata.get("action_names")
        LOGGER.info(
            "OpenVLA IO schema for %s: state_dim=%s state_names=%s action_dim=%s action_names=%s",
            self.metadata.dataset_name,
            self.server_metadata.get("state_dim"),
            state_names,
            self.server_metadata.get("action_dim"),
            action_names,
        )
        if isinstance(state_names, list) and isinstance(action_names, list):
            joint_pairs = []
            first_non_joint_idx = 0
            for idx, (state_name, action_name) in enumerate(zip(state_names, action_names)):
                if "joint" not in state_name or "joint" not in action_name:
                    first_non_joint_idx = idx
                    break
                if state_name != action_name:
                    first_non_joint_idx = idx
                    break
                joint_pairs.append(f"{state_name}->{action_name}")
                first_non_joint_idx = idx + 1

            if joint_pairs:
                LOGGER.info("OpenVLA joint-aligned dimensions: %s", joint_pairs)

            gripper_mappings = []
            for idx in range(first_non_joint_idx, min(len(state_names), len(action_names))):
                gripper_mappings.append(f"{state_names[idx]}->{action_names[idx]}")
            if gripper_mappings:
                LOGGER.info(
                    "OpenVLA non-isomorphic state/action dimensions (expected for gripper schema): %s",
                    gripper_mappings,
                )

            if len(state_names) > len(action_names):
                LOGGER.info("OpenVLA state-only dimensions: %s", state_names[len(action_names) :])
            elif len(action_names) > len(state_names):
                LOGGER.info("OpenVLA action-only dimensions: %s", action_names[len(state_names) :])

    def _load_action_arrays_from_dataset(self) -> Optional[np.ndarray]:
        try:
            import pyarrow.parquet as pq
        except Exception as exc:
            LOGGER.warning("Skipping action-stat override because pyarrow is unavailable: %s", exc)
            return None

        action_batches = []
        for row in self.metadata.episodes:
            episode_index = int(row["episode_index"])
            parquet_path = self.metadata.parquet_path(episode_index)
            if not parquet_path.is_file():
                LOGGER.warning("Missing parquet while recomputing action stats: %s", parquet_path)
                return None
            table = pq.read_table(parquet_path, columns=["action"])
            actions = np.asarray(table.column("action").to_pylist(), dtype=np.float32)
            if actions.ndim != 2 or actions.shape[1] != int(self.metadata.action_dim):
                LOGGER.warning(
                    "Unexpected action shape %s while recomputing action stats from %s",
                    tuple(actions.shape),
                    parquet_path,
                )
                return None
            action_batches.append(actions)

        if not action_batches:
            return None
        return np.concatenate(action_batches, axis=0)

    def _action_stats_look_mismatched(self, action_stats: Mapping[str, Any]) -> bool:
        q01 = np.asarray(action_stats.get("q01"), dtype=np.float32)
        q99 = np.asarray(action_stats.get("q99"), dtype=np.float32)
        if q01.ndim != 1 or q99.ndim != 1 or q01.size != q99.size or q01.size < 2:
            return False

        joint_q_span = float(max(np.max(np.abs(q01[:-1])), np.max(np.abs(q99[:-1]))))
        state_stats = self.metadata.dataset_statistics[self.unnorm_key]["proprio"]
        state_q01 = np.asarray(state_stats.get("q01"), dtype=np.float32)
        state_q99 = np.asarray(state_stats.get("q99"), dtype=np.float32)
        if state_q01.ndim != 1 or state_q99.ndim != 1 or state_q01.size < q01.size - 1:
            return False

        state_joint_span = float(max(np.max(np.abs(state_q01[:-2])), np.max(np.abs(state_q99[:-2]))))
        # For absolute-joint datasets, action[:7] should live on the same scale as joint positions.
        # If action stats are much smaller than proprio stats, the saved normalization is likely wrong.
        return state_joint_span > 1.5 and joint_q_span < 1.5 and joint_q_span < 0.5 * state_joint_span

    def _maybe_override_action_stats_from_dataset(self) -> None:
        dataset_stats = self.metadata.dataset_statistics[self.unnorm_key]
        action_stats = dataset_stats["action"]
        if not self._action_stats_look_mismatched(action_stats):
            return

        actions = self._load_action_arrays_from_dataset()
        if actions is None or actions.size == 0:
            LOGGER.warning("Detected suspicious action stats but failed to load dataset actions for override.")
            return

        new_action_stats = {
            "mean": np.mean(actions, axis=0, dtype=np.float32).astype(np.float32),
            "std": np.std(actions, axis=0, dtype=np.float32).astype(np.float32),
            "q01": np.quantile(actions, 0.01, axis=0).astype(np.float32),
            "q99": np.quantile(actions, 0.99, axis=0).astype(np.float32),
        }
        LOGGER.warning(
            "Overriding suspicious checkpoint action stats for %s using dataset parquet values. "
            "old_q01=%s old_q99=%s new_q01=%s new_q99=%s",
            self.unnorm_key,
            np.asarray(action_stats.get("q01"), dtype=np.float32).tolist(),
            np.asarray(action_stats.get("q99"), dtype=np.float32).tolist(),
            new_action_stats["q01"].tolist(),
            new_action_stats["q99"].tolist(),
        )
        dataset_stats["action"] = new_action_stats

    def _patch_openvla_constants(self) -> None:
        action_dim = int(self.metadata.action_dim)
        proprio_dim = int(self.metadata.state_dim)
        action_chunk_size = int(self.finetune_cfg.action_chunk_size)

        vla_constants.ACTION_DIM = action_dim
        vla_constants.PROPRIO_DIM = proprio_dim
        vla_constants.NUM_ACTIONS_CHUNK = action_chunk_size

        module_names = set()

        def _collect_module_names(obj: Any) -> None:
            if obj is None:
                return
            cls = getattr(obj, "__class__", None)
            module_name = getattr(cls, "__module__", None)
            if isinstance(module_name, str) and module_name:
                module_names.add(module_name)

        _collect_module_names(self.vla)
        _collect_module_names(getattr(self.vla, "base_model", None))
        _collect_module_names(getattr(getattr(self.vla, "base_model", None), "model", None))
        _collect_module_names(getattr(self.vla, "model", None))
        _collect_module_names(self.action_head)
        _collect_module_names(self.proprio_projector)

        for module_name, module in sys.modules.items():
            if "modeling_prismatic" in module_name:
                module_names.add(module_name)
            elif module_name.endswith(".action_heads") or module_name.endswith(".constants"):
                module_names.add(module_name)

        patched_modules = []
        for module_name in module_names:
            module = sys.modules.get(module_name)
            if module is None:
                continue
            changed = False
            if hasattr(module, "ACTION_DIM"):
                setattr(module, "ACTION_DIM", action_dim)
                changed = True
            if hasattr(module, "PROPRIO_DIM"):
                setattr(module, "PROPRIO_DIM", proprio_dim)
                changed = True
            if hasattr(module, "NUM_ACTIONS_CHUNK"):
                setattr(module, "NUM_ACTIONS_CHUNK", action_chunk_size)
                changed = True
            if changed:
                patched_modules.append(module_name)

        LOGGER.info(
            "Patched OpenVLA constants: ACTION_DIM=%d PROPRIO_DIM=%d NUM_ACTIONS_CHUNK=%d modules=%s",
            action_dim,
            proprio_dim,
            action_chunk_size,
            sorted(patched_modules),
        )

    def _set_norm_stats(self) -> None:
        norm_stats = self.metadata.dataset_statistics
        visited: set[int] = set()

        def _apply(obj: Any) -> None:
            if obj is None:
                return
            obj_id = id(obj)
            if obj_id in visited:
                return
            visited.add(obj_id)

            try:
                setattr(obj, "norm_stats", norm_stats)
            except Exception:
                pass

            cfg = getattr(obj, "config", None)
            if cfg is not None:
                try:
                    setattr(cfg, "norm_stats", norm_stats)
                except Exception:
                    pass

            for attr in ("base_model", "model", "module"):
                try:
                    child = getattr(obj, attr, None)
                except Exception:
                    child = None
                if child is not None and child is not obj:
                    _apply(child)

        _apply(self.vla)

        action_stats = norm_stats[self.unnorm_key]["action"]
        LOGGER.info(
            "Loaded action norm stats for %s: q01=%s q99=%s",
            self.unnorm_key,
            np.asarray(action_stats.get("q01")).tolist() if "q01" in action_stats else None,
            np.asarray(action_stats.get("q99")).tolist() if "q99" in action_stats else None,
        )

    def _manual_unnormalize_actions(self, normalized_actions: np.ndarray) -> np.ndarray:
        action_stats = self.metadata.dataset_statistics[self.unnorm_key]["action"]
        action_low = np.asarray(action_stats["q01"], dtype=np.float32)
        action_high = np.asarray(action_stats["q99"], dtype=np.float32)
        return (
            0.5 * (np.asarray(normalized_actions, dtype=np.float32) + 1.0)
            * (action_high - action_low + 1e-8)
            + action_low
        ).astype(np.float32)

    def _looks_like_normalized_actions(self, actions: np.ndarray) -> bool:
        actions = np.asarray(actions, dtype=np.float32)
        if actions.size == 0 or actions.ndim < 2:
            return False
        action_stats = self.metadata.dataset_statistics[self.unnorm_key]["action"]
        q01 = np.asarray(action_stats["q01"], dtype=np.float32)
        q99 = np.asarray(action_stats["q99"], dtype=np.float32)
        stats_span = float(max(np.max(np.abs(q01)), np.max(np.abs(q99))))
        action_span = float(np.max(np.abs(actions)))
        return action_span <= 1.25 and stats_span > 1.5

    def _extract_images(self, payload: Mapping[str, Any]) -> tuple[np.ndarray, Optional[np.ndarray]]:
        images = payload.get("images", payload)
        if not isinstance(images, Mapping):
            raise ValueError("Incoming payload must contain an `images` mapping.")

        primary = _first_present(
            images,
            (
                "observation.images.agentview",
                "agentview",
                "observation.images.head_camera",
                "head_camera",
            ),
        )
        wrist = _first_present(
            images,
            (
                "observation.images.wrist",
                "observation.images.robot0_eye_in_hand",
                "wrist",
                "robot0_eye_in_hand",
            ),
        )
        if primary is None:
            raise KeyError("Missing primary image. Expected `observation.images.agentview`.")
        return _decode_image(primary), None if wrist is None else _decode_image(wrist)

    def _extract_state(self, payload: Mapping[str, Any]) -> Optional[np.ndarray]:
        raw = payload.get("state", payload.get("observation.state"))
        if raw is None:
            return None
        state = np.asarray(raw, dtype=np.float32).reshape(-1)
        if state.size != self.metadata.state_dim:
            raise ValueError(f"State dim mismatch: got {state.size}, expected {self.metadata.state_dim}")
        return state

    def _build_inputs(
        self,
        *,
        prompt: str,
        primary_image: np.ndarray,
        wrist_image: Optional[np.ndarray],
    ) -> BatchEncoding:
        primary = Image.fromarray(primary_image).convert("RGB")
        inputs = self.processor(prompt, primary).to(self.device, dtype=self.input_dtype)
        if wrist_image is not None and int(self.finetune_cfg.num_images_in_input) > 1:
            wrist = Image.fromarray(wrist_image).convert("RGB")
            wrist_inputs = self.processor(prompt, wrist).to(self.device, dtype=self.input_dtype)
            inputs["pixel_values"] = torch.cat([inputs["pixel_values"], wrist_inputs["pixel_values"]], dim=1)
        return inputs

    def reset(self) -> None:
        # Stateless policy server today, but keep an explicit reset command so the
        # client can mirror ACT/OpenPI behavior without special-casing server state.
        return None

    def infer(self, payload: Mapping[str, Any]) -> Dict[str, Any]:
        primary_image, wrist_image = self._extract_images(payload)
        state = self._extract_state(payload)
        prompt = str(payload.get("prompt", self.default_prompt))
        prompt_norm = prompt.strip().lower()
        training_prompt_norm = str(self.training_prompt).strip().lower()
        if prompt_norm != training_prompt_norm and prompt_norm not in self._warned_prompt_mismatches:
            LOGGER.warning(
                "Incoming prompt differs from training prompt. incoming=%r training=%r. "
                "Single-task OpenVLA checkpoints are often sensitive to prompt wording mismatches.",
                prompt,
                self.training_prompt,
            )
            self._warned_prompt_mismatches.add(prompt_norm)
        chunk_size = int(payload.get("chunk_size", self.finetune_cfg.action_chunk_size))
        if chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")
        if self.expect_state and state is None:
            raise ValueError("Policy expects `state` but none was provided.")

        model_prompt = f"In: What action should the robot take to {prompt.lower()}?\nOut:"
        inputs = self._build_inputs(prompt=model_prompt, primary_image=primary_image, wrist_image=wrist_image)

        proprio = None
        if self.proprio_projector is not None and state is not None:
            proprio_stats = self.metadata.dataset_statistics[self.unnorm_key]["proprio"]
            proprio = normalize_proprio(np.asarray(state, dtype=np.float32), proprio_stats)
            proprio = np.asarray(proprio, dtype=np.float32).reshape(1, -1)

        with self.lock:
            with torch.inference_mode():
                infer_start = time.monotonic()
                actions, _ = self.vla.predict_action(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    pixel_values=inputs["pixel_values"],
                    unnorm_key=self.unnorm_key,
                    proprio=proprio,
                    proprio_projector=self.proprio_projector,
                    action_head=self.action_head,
                    use_film=False,
                )
                infer_ms = (time.monotonic() - infer_start) * 1000.0

        action_array = np.asarray(actions, dtype=np.float32)
        if action_array.ndim == 1:
            action_array = action_array.reshape(1, -1)
        if self._looks_like_normalized_actions(action_array):
            LOGGER.warning(
                "Model output still appears normalized (max_abs=%.4f); applying manual action unnormalization fallback.",
                float(np.max(np.abs(action_array))),
            )
            action_array = self._manual_unnormalize_actions(action_array)
        action_array = action_array[:chunk_size]
        return {
            "action": action_array,
            "shape": list(action_array.shape),
            "action_dim": int(action_array.shape[-1]) if action_array.ndim >= 1 else int(self.metadata.action_dim),
            "server_timing": {"infer_ms": infer_ms},
        }

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self) -> None:
        async with ws_server.serve(
            self._handler,
            self.args.host,
            self.args.port,
            compression=None,
            max_size=None,
            process_request=_health_check,
        ) as server:
            LOGGER.info("Serving OpenVLA policy on ws://%s:%d", self.args.host, self.args.port)
            await server.serve_forever()

    async def _handler(self, websocket: ws_server.ServerConnection) -> None:
        LOGGER.info("Connection from %s opened", websocket.remote_address)
        packer = Packer()
        await websocket.send(packer.pack(self.server_metadata))
        prev_total_time = None

        while True:
            try:
                start_time = time.monotonic()
                request = unpackb(await websocket.recv())
                if not isinstance(request, Mapping):
                    raise TypeError(f"Expected dict request, got {type(request)}")
                if request.get("__command__") == "reset":
                    self.reset()
                    response = {"status": "ok"}
                else:
                    response = self.infer(request)
                if prev_total_time is not None:
                    response.setdefault("server_timing", {})["prev_total_ms"] = prev_total_time * 1000.0
                await websocket.send(packer.pack(response))
                prev_total_time = time.monotonic() - start_time
            except websockets.ConnectionClosed:
                LOGGER.info("Connection from %s closed", websocket.remote_address)
                break
            except Exception:
                tb = traceback.format_exc()
                LOGGER.error(tb)
                await websocket.send(tb)
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise


def _health_check(
    connection: ws_server.ServerConnection,
    request: ws_server.Request,
) -> ws_server.Response | None:
    if request.path in ("/health", "/healthz"):
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    return None


def parse_args() -> Args:
    parser = argparse.ArgumentParser(description="Serve a fine-tuned OpenVLA checkpoint over websocket.")
    parser.add_argument("--checkpoint", "--checkpoint-dir", dest="checkpoint", type=Path, required=True)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--base-vla-path", type=str, default=None)
    parser.add_argument("--default-prompt", type=str, default=None)
    parser.add_argument("--dataset-root", type=Path, default=None)
    args = parser.parse_args()
    return Args(
        checkpoint=args.checkpoint,
        host=args.host,
        port=args.port,
        base_vla_path=args.base_vla_path,
        default_prompt=args.default_prompt,
        dataset_root=args.dataset_root,
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, force=True)
    server = OpenVLAWebsocketPolicyServer(parse_args())
    server.serve_forever()


if __name__ == "__main__":
    main()
