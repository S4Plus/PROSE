from __future__ import annotations

"""TRAK pipeline for LeRobot diffusion policies.

This script mirrors the behaviour of ``train_trak_diffusion.py`` but is tailored
for LeRobot diffusion-policy checkpoints saved as ``model.safetensors``.  It
loads the policy from the provided directory, featurizes the training dataset,
and scores evaluation rollouts stored in an ``episodes`` folder.

  python train_trak_lerobot.py \
    --exp_name train_trak_lerobot_dp \
    --pretrained_dir /root/lerobot_2/outputs2/pretrained_model \
    --eval_dir /root/2_250_pkl/train_trak_lerobot_dp \
    --num_timesteps 1 \
    --batch_size 1 \
    --proj_dim 512 \
    --proj_max_batch_size 1\
    --use_half_precision True


python - <<'PY'
from safetensors.torch import load_file
import torch

src = "/root/lerobot_2.1/outputs2/pretrained_model/model.safetensors"  # 输入 safetensors 文件
dst = "model.ckpt"         # 输出 ckpt 文件

state = load_file(src)
torch.save(state, dst)
print(f"saved {dst}")
PY

"""

# Allow execution from any working directory without installing this checkout.
import sys as _prose_sys
from pathlib import Path as _ProsePath
_prose_sys.path.insert(0, str(_ProsePath(__file__).resolve().parents[2]))


from typing import Dict, Iterable, List, Optional, Tuple, TextIO

import sys


def _enable_line_buffering(stream: TextIO) -> None:
    """Best-effort line buffering without touching raw file descriptors."""

    try:
        stream.reconfigure(line_buffering=True)
    except (AttributeError, ValueError):  # pragma: no cover - depends on host Python
        pass


_enable_line_buffering(sys.stdout)
_enable_line_buffering(sys.stderr)

import os

# Reduce cuda memory fragmentation.
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import glob
import inspect
import json
import logging
import pickle
import random
import pathlib

import click
import numpy as np
import torch
import trak
from safetensors.torch import load_file as load_safetensors
from torch.utils.data import DataLoader, IterableDataset

from prose.compat.hf_datasets import ensure_hf_list_feature

LEROBOT_REPO = pathlib.Path(os.environ.get("LEROBOT_REPO", str(pathlib.Path(__file__).resolve().parents[2] / "lerobot_2.1"))).expanduser()
LEROBOT_SRC = LEROBOT_REPO / "src"
if LEROBOT_SRC.exists():
    sys.path.append(str(LEROBOT_SRC))

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

from third_party.trak.trak.modelout_functions import AbstractModelOutput
from third_party.trak.trak.gradient_computers import AbstractGradientComputer


LOGGER = logging.getLogger("train_trak_lerobot_dp")
IMAGE_KEYS = ("head_camera", "right_wrist_camera")
STATE_ALIASES = ("state", "joint_state")
DEFAULT_EVAL_DIR = str((pathlib.Path.home() / "2_250_pkl" / "train_trak_lerobot_dp").expanduser())


def get_parameter_names(model: torch.nn.Module, keys: Iterable[str]) -> List[str]:
    """Return parameter names containing any of the provided substrings."""

    key_list = list(keys)
    param_names = list(dict(model.named_parameters()).keys())
    return sorted([name for name in param_names if any(k in name for k in key_list) and "dummy" not in name])


def _ensure_hf_cache_env(root: pathlib.Path) -> None:
    """Keep Hugging Face caches under the dataset root to avoid permission issues."""

    os.environ.setdefault("HF_HOME", str(root))
    os.environ.setdefault("HF_DATASETS_CACHE", str(root / "hf_datasets_cache"))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(root / "hub"))


def _patch_lerobot_imports() -> None:
    """Patch missing soft dependencies used by the vendored LeRobot code.

    The environment we're running in might not ship optional packages such as
    ``draccus`` or the newer ``huggingface_hub.errors`` / ``HF_HOME`` symbols.
    Creating lightweight fallbacks keeps the LeRobot modules importable for
    inference without pulling additional wheels during execution.
    """

    import types
    import contextlib
    from pathlib import Path

    # Minimal stub for draccus (only pieces touched during config loading).
    if "draccus" not in sys.modules:
        draccus = types.ModuleType("draccus")
        draccus.CHOICE_TYPE_KEY = "choice_type"

        @contextlib.contextmanager
        def config_type(_fmt: str):
            yield

        draccus.config_type = config_type
        draccus.dump = lambda obj, f, indent=None: json.dump(obj, f, indent=indent)
        draccus.load = lambda _typ, f: json.load(f)
        draccus.parse = lambda *_, **__: None

        class ChoiceRegistry:
            @classmethod
            def get_choice_name(cls, choice):
                return getattr(choice, "__name__", str(choice))

            @classmethod
            def get_choice_type_name(cls, choice):
                return getattr(choice, "__name__", str(choice))

        draccus.ChoiceRegistry = ChoiceRegistry
        sys.modules["draccus"] = draccus

        choice_types = types.ModuleType("draccus.choice_types")
        choice_types.ChoiceRegistry = ChoiceRegistry
        sys.modules["draccus.choice_types"] = choice_types

    # Backfill huggingface_hub.errors for older hub versions.
    try:  # pragma: no cover - defensive shim
        import huggingface_hub.errors as _  # noqa: F401
    except ImportError:
        import huggingface_hub.utils as hf_utils

        errors_mod = types.ModuleType("huggingface_hub.errors")
        for name in [
            "HfHubHTTPError",
            "EntryNotFoundError",
            "RepositoryNotFoundError",
            "RevisionNotFoundError",
            "GatedRepoError",
            "LocalEntryNotFoundError",
            "LocalTokenNotFoundError",
            "HFValidationError",
            "BadRequestError",
        ]:
            setattr(errors_mod, name, getattr(hf_utils, name, Exception))
        sys.modules["huggingface_hub.errors"] = errors_mod

    # Missing constants in older huggingface_hub.
    # import huggingface_hub.constants as hf_constants

    # if not hasattr(hf_constants, "HF_HOME"):
    #     hf_constants.HF_HOME = Path.home() / ".cache" / "huggingface"
    # if not hasattr(hf_constants, "HUGGINGFACE_HUB_CACHE"):
    #     hf_constants.HUGGINGFACE_HUB_CACHE = hf_constants.HF_HOME / "hub"
    # if not hasattr(hf_constants, "HF_DATASETS_CACHE"):
    #     hf_constants.HF_DATASETS_CACHE = hf_constants.HF_HOME / "datasets"


def load_lerobot_policy(pretrained_dir: pathlib.Path, device: torch.device):
    """Load a LeRobot policy (diffusion or ACT) from a ``model.safetensors`` directory."""

    _patch_lerobot_imports()

    # Ensure vendored code is on the python path.
    repo_root = pathlib.Path(__file__).resolve().parents[2]
    lerobot_root = repo_root / "preprocess_data"
    if lerobot_root.exists():
        sys.path.append(str(lerobot_root))

    from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
    from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
    from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.act.modeling_act import ACTPolicy

    with open(pretrained_dir / "config.json", "r") as f:
        cfg_dict = json.load(f)

    model_type = cfg_dict.get("type", "diffusion")

    # Remove registry metadata and hydrate enums/nested dataclasses expected by DiffusionConfig.
    cfg_dict.pop("type", None)

    if "normalization_mapping" in cfg_dict:
        cfg_dict["normalization_mapping"] = {
            key: NormalizationMode(value) if not isinstance(value, NormalizationMode) else value
            for key, value in cfg_dict["normalization_mapping"].items()
        }

    def _build_features(raw_features):
        features = {}
        for key, ft in raw_features.items():
            if isinstance(ft, PolicyFeature):
                features[key] = ft
                continue
            features[key] = PolicyFeature(type=FeatureType(ft["type"]), shape=tuple(ft["shape"]))
        return features

    if "input_features" in cfg_dict:
        cfg_dict["input_features"] = _build_features(cfg_dict["input_features"])
    if "output_features" in cfg_dict:
        cfg_dict["output_features"] = _build_features(cfg_dict["output_features"])

    if isinstance(cfg_dict.get("down_dims"), list):
        cfg_dict["down_dims"] = tuple(cfg_dict["down_dims"])

    if model_type == "act":
        act_sig = inspect.signature(ACTConfig)
        valid_keys = set(act_sig.parameters.keys())
        filtered_cfg = {k: v for k, v in cfg_dict.items() if k in valid_keys}
        dropped = sorted(set(cfg_dict.keys()) - valid_keys)
        if dropped:
            LOGGER.warning("Dropping unsupported ACTConfig keys: %s", dropped)
        config = ACTConfig(**filtered_cfg)
        config.device = str(device)

        # Optional dataset stats if metadata exists.
        metadata_root = pretrained_dir / "metadata"
        if metadata_root.exists():
            _ensure_hf_cache_env(metadata_root)
            ensure_hf_list_feature()
            meta = LeRobotDatasetMetadata(
                repo_id=config.repo_id or "local/lerobot",
                root=metadata_root,
                force_cache_sync=False,
            )
            dataset_stats = meta.stats
        else:
            dataset_stats = None

        policy = ACTPolicy(config=config, dataset_stats=dataset_stats)
    else:
        diff_sig = inspect.signature(DiffusionConfig)
        valid_keys = set(diff_sig.parameters.keys())
        filtered_cfg = {k: v for k, v in cfg_dict.items() if k in valid_keys}
        dropped = sorted(set(cfg_dict.keys()) - valid_keys)
        if dropped:
            LOGGER.warning("Dropping unsupported DiffusionConfig keys: %s", dropped)
        config = DiffusionConfig(**filtered_cfg)
        config.device = str(device)
        policy = DiffusionPolicy(config)

    state_dict = load_safetensors(pretrained_dir / "model.safetensors")
    missing, unexpected = policy.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        LOGGER.warning("State dict load: missing=%s unexpected=%s", missing, unexpected)

    policy.to(device)
    policy.eval()
    return policy, config


def _infer_repo_id(dataset_root: pathlib.Path, cfg_repo_id: Optional[str]) -> str:
    """Infer repo_id from config or dataset path (fallback to folder name)."""

    if cfg_repo_id:
        return cfg_repo_id
    parts = dataset_root.parts
    if len(parts) >= 2:
        return "/".join(parts[-2:])
    return dataset_root.name


def build_lerobot_dataset(train_dataset_path: pathlib.Path, cfg: DiffusionConfig) -> LeRobotDataset:
    """Load a LeRobot dataset with delta timestamps matching the policy config."""

    dataset_root = train_dataset_path.expanduser()
    if dataset_root.is_file():
        dataset_root = dataset_root.parent

    _ensure_hf_cache_env(dataset_root)
    ensure_hf_list_feature()
    repo_id = _infer_repo_id(dataset_root, getattr(cfg, "repo_id", None))
    metadata = LeRobotDatasetMetadata(repo_id=repo_id, root=dataset_root, force_cache_sync=False)
    fps = metadata.fps

    delta_ts = {
        "observation.state": [i / fps for i in cfg.observation_delta_indices],
        "action": [i / fps for i in cfg.action_delta_indices],
    }
    for cam in ("observation.images.head_camera", "observation.images.right_wrist_camera"):
        delta_ts[cam] = [i / fps for i in cfg.observation_delta_indices]

    return LeRobotDataset(
        repo_id=repo_id,
        root=dataset_root,
        delta_timestamps=delta_ts,
        download_videos=False,
    )


class LerobotTraKDataset(torch.utils.data.Dataset):
    """Thin wrapper that reshapes LeRobot samples for TRAK consumption."""

    def __init__(self, dataset: LeRobotDataset):
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.dataset[idx]
        obs = {}

        head_key = "observation.images.head_camera"
        wrist_key = "observation.images.right_wrist_camera"

        if head_key in sample:
            obs["head_camera"] = sample[head_key]
        if wrist_key in sample:
            obs["right_wrist_camera"] = sample[wrist_key]
        if "observation.state" in sample:
            obs["state"] = sample["observation.state"]

        return {"obs": obs, "action": sample["action"]}


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
        import pandas as pd

        with open(path, "rb") as f:
            return pickle.load(f)

    @staticmethod
    def _infer_episode_success(episode: "pd.DataFrame", path: str) -> bool:
        """Infer success from dataframe contents or filename suffix."""

        first_row = episode.iloc[0].to_dict() if len(episode) > 0 else {}
        if "success" in first_row:
            return bool(first_row["success"])

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
                yield sample[0] if len(sample) == 1 else sample


def _tensor_from_array(array: np.ndarray | torch.Tensor) -> torch.Tensor:
    tensor = torch.as_tensor(array)
    if tensor.dtype == torch.uint8:
        tensor = tensor.float() / 255.0
    else:
        tensor = tensor.float()
    return tensor


def _normalize_image_tensor(tensor: torch.Tensor, batch_size: int, n_obs_steps: int) -> torch.Tensor:
    """Ensure image tensor shape is (B, T, C, H, W)."""

    if tensor.dim() == 3:  # H, W, C
        tensor = tensor.permute(2, 0, 1).unsqueeze(0).unsqueeze(0)
    elif tensor.dim() == 4:
        if tensor.shape[0] == batch_size and tensor.shape[1] in (3, 4):  # B, C, H, W
            tensor = tensor.unsqueeze(1)
        elif tensor.shape[0] != batch_size and tensor.shape[1] in (3, 4):  # C, H, W
            tensor = tensor.unsqueeze(0).unsqueeze(0)
        elif tensor.shape[0] == batch_size:  # B, T, H, W -> assume already time stacked
            tensor = tensor.unsqueeze(2)  # add channel dim placeholder
        else:  # fallback: treat leading dim as time
            tensor = tensor.unsqueeze(0)
    elif tensor.dim() == 5:
        if tensor.shape[0] != batch_size:
            tensor = tensor.unsqueeze(0)

    # At this point expect (B, T?, C, H, W)
    if tensor.dim() != 5:
        raise ValueError(f"Unexpected image tensor shape {tuple(tensor.shape)}")

    # Some datasets store channels last; permute into (B, T, C, H, W).
    if tensor.shape[2] not in (3, 4) and tensor.shape[-1] in (3, 4):
        tensor = tensor.permute(0, 1, 4, 2, 3)

    if tensor.shape[0] != batch_size:
        tensor = tensor.expand(batch_size, *tensor.shape[1:])

    if tensor.shape[1] < n_obs_steps:
        pad = tensor[:, -1:, ...].expand(batch_size, n_obs_steps - tensor.shape[1], *tensor.shape[2:])
        tensor = torch.cat([tensor, pad], dim=1)
    elif tensor.shape[1] > n_obs_steps:
        tensor = tensor[:, -n_obs_steps:, ...]
    return tensor


def _normalize_state_tensor(tensor: torch.Tensor, batch_size: int, n_obs_steps: int) -> torch.Tensor:
    """Ensure state tensor shape is (B, T, D)."""

    if tensor.dim() == 1:
        tensor = tensor.unsqueeze(0).unsqueeze(0)
    elif tensor.dim() == 2:
        if tensor.shape[0] == batch_size:
            tensor = tensor.unsqueeze(1)
        else:
            tensor = tensor.unsqueeze(0)
    elif tensor.dim() == 3:
        if tensor.shape[0] != batch_size:
            tensor = tensor.unsqueeze(0)
    else:
        raise ValueError(f"Unexpected state tensor shape {tuple(tensor.shape)}")

    if tensor.shape[0] != batch_size:
        tensor = tensor.expand(batch_size, *tensor.shape[1:])

    if tensor.shape[1] < n_obs_steps:
        pad = tensor[:, -1:, :].expand(batch_size, n_obs_steps - tensor.shape[1], tensor.shape[2])
        tensor = torch.cat([tensor, pad], dim=1)
    elif tensor.shape[1] > n_obs_steps:
        tensor = tensor[:, -n_obs_steps:, :]
    return tensor


def prepare_batch(
    batch: Dict[str, torch.Tensor],
    n_obs_steps: int,
    horizon: int,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """Format heterogeneous batch dictionaries into model-ready tensors."""

    action = _tensor_from_array(batch["action"]).to(device)
    if action.dim() == 2:
        action = action.unsqueeze(0)

    batch_size = action.shape[0]
    if action.shape[1] < horizon:
        pad = action[:, -1:, :].expand(batch_size, horizon - action.shape[1], action.shape[2])
        action = torch.cat([action, pad], dim=1)
    elif action.shape[1] > horizon:
        action = action[:, :horizon, :]

    obs = batch.get("obs", {})
    formatted_obs: Dict[str, torch.Tensor] = {}
    for key, value in obs.items():
        mapped_key = "state" if key in STATE_ALIASES else key
        tensor = _tensor_from_array(value).to(device)
        if mapped_key in IMAGE_KEYS:
            formatted_obs[mapped_key] = _normalize_image_tensor(tensor, batch_size, n_obs_steps)
        elif mapped_key == "state":
            formatted_obs[mapped_key] = _normalize_state_tensor(tensor, batch_size, n_obs_steps)

    return {
        "action": action,
        "obs": formatted_obs,
    }


class LerobotDiffusionModelOutput(AbstractModelOutput):
    """DDPM-style loss for LeRobot diffusion policies."""

    LOSS_FNS = {"ddpm", "square", "average"}

    def __init__(self, loss_fn: str, image_keys: Tuple[str, ...]):
        super().__init__()
        assert loss_fn in self.LOSS_FNS
        self.loss_fn = loss_fn
        self.image_keys = image_keys

    def _global_condition(self, model, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        feats = []
        if "state" in obs:
            feats.append(obs["state"].reshape(obs["state"].shape[0], -1))

        images = [obs[k] for k in self.image_keys if k in obs]
        if images:
            stacked = torch.stack(images, dim=2)  # B, T, N, C, H, W
            B, T, N, C, H, W = stacked.shape
            flat = stacked.reshape(B * T * N, C, H, W)
            encoder = model.diffusion.rgb_encoder
            if isinstance(encoder, torch.nn.ModuleList):
                # Split per camera if separate encoders are used.
                encoded_list = []
                for cam_idx, enc in enumerate(encoder):
                    cam_flat = flat[cam_idx :: N]
                    encoded_list.append(enc(cam_flat))
                encoded = torch.stack(encoded_list, dim=1)
                encoded = encoded.reshape(B, T, N, -1)
            else:
                encoded = encoder(flat)
                encoded = encoded.reshape(B, T, N, -1)
            feats.append(encoded.reshape(B, -1))

        if len(feats) == 0:
            raise ValueError("No conditioning features found in observation.")
        return torch.cat(feats, dim=-1)

    def _ddpm_loss(
        self,
        model,
        obs: Dict[str, torch.Tensor],
        action: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        global_cond = self._global_condition(model, obs)
        noise = torch.randn_like(action)
        noisy_trajectory = model.diffusion.noise_scheduler.add_noise(action, noise, timesteps)
        pred = model.diffusion.unet(noisy_trajectory, timesteps, global_cond=global_cond)

        if model.diffusion.noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif model.diffusion.noise_scheduler.config.prediction_type == "sample":
            target = action
        else:
            raise ValueError(
                f"Unsupported prediction type {model.diffusion.noise_scheduler.config.prediction_type}"
            )

        if self.loss_fn == "average":
            loss = (pred * (pred - target)).reshape(pred.shape[0], -1).mean(dim=1)
        elif self.loss_fn == "square":
            loss = (pred ** 2).reshape(pred.shape[0], -1).mean(dim=1)
        else:  # ddpm
            loss = ((pred - target) ** 2).reshape(pred.shape[0], -1).mean(dim=1)
        return loss

    def get_output(
        self,
        model: torch.nn.Module,
        weights,
        buffers,
        batch: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        action = batch["action"]
        timesteps = batch["timesteps"]
        obs = batch["obs"]

        losses = []
        for i in range(timesteps.shape[1]):
            t = timesteps[:, i]
            losses.append(self._ddpm_loss(model, obs, action, t))

        return torch.stack(losses, dim=1).mean(dim=1)

    def get_out_to_loss_grad(self, model, weights, buffers, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.ones(batch["action"].shape[0], device=batch["action"].device).unsqueeze(-1)


class LerobotDiffusionGradientComputer(AbstractGradientComputer):
    """Iterative per-sample gradients using autograd (no torch.func dependency)."""

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
        self.logger = logging.getLogger("LerobotDiffusionGradientComputer")

        params = list(model.named_parameters())
        if grad_wrt is not None:
            grad_wrt = set(grad_wrt)
            params = [(n, p) for n, p in params if n in grad_wrt]
            missing = grad_wrt - {n for n, _ in params}
            if missing:
                self.logger.warning("Requested grad_wrt params not found: %s", sorted(missing))

        filtered_params = []
        dropped = []
        for name, param in params:
            if param.requires_grad:
                filtered_params.append((name, param))
            else:
                dropped.append(name)
        if dropped:
            self.logger.debug("Skipping params without grad: %s", sorted(dropped))
        params = filtered_params

        self.params = params
        self.param_tensors = [p for _, p in params]
        self.grad_dim = sum(p.numel() for p in self.param_tensors)

    def load_model_params(self, model) -> None:  # pragma: no cover - no-op for iterative mode
        return None

    def compute_per_sample_grad(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        losses = self.modelout_fn.get_output(self.model, None, None, batch)
        batch_size = losses.shape[0]
        grads = torch.zeros(batch_size, self.grad_dim, device=self.device, dtype=self.dtype)
        self.logger.info("Computing per-sample gradients for batch_size=%d", batch_size)
        log_every = max(1, batch_size // 5)

        for i in range(batch_size):
            grad_tensors = torch.autograd.grad(
                losses[i],
                self.param_tensors,
                retain_graph=i < batch_size - 1,
                allow_unused=True,
            )
            flattened = []
            for grad_tensor, param in zip(grad_tensors, self.param_tensors):
                if grad_tensor is None:
                    flattened.append(torch.zeros_like(param).reshape(-1))  # Param unused in this sample
                else:
                    flattened.append(grad_tensor.reshape(-1))
            grads[i] = torch.cat(flattened).to(self.dtype)
            if (i + 1) % log_every == 0 or i == batch_size - 1:
                self.logger.info("Finished gradients for %d/%d samples", i + 1, batch_size)

        return grads

    def compute_loss_grad(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.modelout_fn.get_out_to_loss_grad(self.model, None, None, batch)


@click.command()
# Experiment params.
@click.option("--exp_name", type=str, required=True)
@click.option("--eval_dir", type=str, default=DEFAULT_EVAL_DIR)
@click.option("--pretrained_dir", type=str, required=True)
@click.option("--train_dataset_path", type=str, default=None)
@click.option("--model_keys", type=str, default=None)
# TRAK params.
@click.option("--model_id", type=int, default=0)
@click.option("--proj_dim", type=int, default=2048)
@click.option("--proj_max_batch_size", type=int, default=32)
@click.option("--lambda_reg", type=float, default=0.0)
@click.option("--use_half_precision", type=bool, default=False)
# Task params.
@click.option("--loss_fn", type=str, default="ddpm")
@click.option("--num_timesteps", type=int, default=8)
# Other params.
@click.option("--batch_size", type=int, default=4)
@click.option("--num_workers", type=int, default=4)
@click.option("--device", type=str, default="cuda:0")
@click.option("--overwrite", type=bool, default=False)
@click.option("--seed", type=int, default=0)
# Optional.
@click.option("--finalize_scores", type=bool, default=True)
def main(
    # Experiment params.
    exp_name: str,
    eval_dir: str,
    pretrained_dir: str,
    train_dataset_path: Optional[str],
    model_keys: Optional[str],
    # TRAK params.
    model_id: int,
    proj_dim: int,
    proj_max_batch_size: int,
    lambda_reg: float,
    use_half_precision: bool,
    # Task params.
    loss_fn: str,
    num_timesteps: int,
    # Other params.
    batch_size: int,
    num_workers: int,
    device: str,
    overwrite: bool,
    seed: int,
    # Optional.
    finalize_scores: bool,
):
    # Set random seed.
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    device = torch.device(device)

    pretrained_dir = pathlib.Path(pretrained_dir).expanduser()
    if pretrained_dir.is_file():
        pretrained_dir = pretrained_dir.parent

    # Load policy checkpoint.
    policy, cfg = load_lerobot_policy(pretrained_dir, device)
    grad_wrt = None
    if model_keys:
        grad_wrt = get_parameter_names(policy, model_keys.split(","))

    # Training set (no shuffle).
    default_dataset_root = pathlib.Path.home() / ".cache" / "huggingface" / "lerobot" / "ls2" / "sim_Belt_Pick_Bottle_lerobot"
    resolved_dataset_path = pathlib.Path(train_dataset_path or default_dataset_root)
    base_train_set = build_lerobot_dataset(resolved_dataset_path, cfg)
    train_set = LerobotTraKDataset(base_train_set)
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        num_workers=num_workers,
        persistent_workers=False,
        pin_memory=True,
        shuffle=False,
    )

    # Construct computers.
    task = LerobotDiffusionModelOutput(loss_fn=loss_fn, image_keys=IMAGE_KEYS)
    save_dir = pathlib.Path(eval_dir) / exp_name
    if save_dir.exists() and overwrite:
        import shutil

        shutil.rmtree(save_dir)
    traker = trak.TRAKer(
        model=policy,
        task=task,
        train_set_size=len(train_set),
        gradient_computer=LerobotDiffusionGradientComputer,
        device=device,
        grad_wrt=grad_wrt,
        proj_dim=proj_dim,
        proj_max_batch_size=proj_max_batch_size,
        lambda_reg=lambda_reg,
        save_dir=str(save_dir),
        use_half_precision=use_half_precision,
    )
    traker.load_checkpoint(policy.state_dict(), model_id=model_id)

    noise_steps = policy.diffusion.noise_scheduler.config.num_train_timesteps

    # Featurize training set.
    for batch in train_loader:
        formatted = prepare_batch(batch, cfg.n_obs_steps, cfg.horizon, device)
        formatted["timesteps"] = torch.randint(noise_steps, (formatted["action"].shape[0], num_timesteps), device=device)
        traker.featurize(formatted, num_samples=formatted["action"].shape[0])

    traker.finalize_features(model_ids=[model_id])

    # Load evaluation set (no shuffle).
    eval_path = pathlib.Path(eval_dir) / "episodes"
    test_set = EvalPklEpisodeDataset(
        dataset_path=eval_path,
        exec_horizon=1,
        sample_history=0,
    )
    test_set_size = len(test_set)

    # Scoring test set.
    traker.start_scoring_checkpoint(
        exp_name="all_episodes",
        checkpoint=policy.state_dict(),
        model_id=model_id,
        num_targets=test_set_size,
    )
    for batch in DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0):
        formatted = prepare_batch(batch, cfg.n_obs_steps, cfg.horizon, device)
        formatted["timesteps"] = torch.randint(noise_steps, (formatted["action"].shape[0], num_timesteps), device=device)
        traker.score(batch=formatted, num_samples=formatted["action"].shape[0])

    print(f"Completed featurizing traker for {pretrained_dir}")
    print(f"Results saved to {save_dir}")

    if finalize_scores:
        scores = np.array(
            traker.finalize_scores(
                exp_name="all_episodes",
                model_ids=[model_id],
                allow_skip=False,
            )
        )
        assert scores.shape == (len(train_set), test_set_size)
        print(f"Completed scoring for traker at {save_dir}")


if __name__ == "__main__":
    main()
