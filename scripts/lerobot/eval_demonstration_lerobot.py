from __future__ import annotations

"""Demonstration-scoring pipeline for LeRobot diffusion policies.

This script mirrors ``eval_demonstration_scores.py`` but avoids importing the
``diffusion_policy`` package.  It reads policy checkpoints saved as
``model.safetensors``, consumes metadata directly from LeRobot datasets, and
expects auxiliary evaluation artifacts (policy losses, TRAK features, etc.)
to follow the same mmap layout used throughout the CUPID repo.


 python eval_demonstration_lerobot.py \
    --exp_name lerobot_eval \
    --eval_dir /root/2_250_pkl/train_trak_lerobot_dp \
    --pretrained_dir /root/lerobot_2/outputs2/pretrained_model \
    --train_dataset_path /root/.cache/huggingface/lerobot/ls2/sim_Belt_Pick_Bottle_lerobot \
    --result_date 2024-05-30 \
    --overwrite True \
    --device cuda:0 \
    --seed 0 \
    --use_half_precision False \
    --compute_holdout False \
    --policy_embedding_dim 1536 \
    --state_embedding_load_key default_image \
    --eval_online_trak_influence True
"""

# Allow execution from any working directory without installing this checkout.
import sys as _prose_sys
from pathlib import Path as _ProsePath
_prose_sys.path.insert(0, str(_ProsePath(__file__).resolve().parents[2]))


from collections import defaultdict
from copy import deepcopy
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import glob
import inspect
import json
import os
import pathlib
import pickle
import random
import sys

# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode="w", buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode="w", buffering=1)

import click
import numpy as np
import torch
from numpy.lib.format import open_memmap
from safetensors.torch import load_file as load_safetensors
from torch.utils.data import IterableDataset

from prose.compat.hf_datasets import ensure_hf_list_feature


def _resolve_lerobot_repo() -> pathlib.Path:
    """Find the lerobot source tree used for policy definitions."""

    env_path = os.environ.get("LEROBOT_REPO")
    return pathlib.Path(env_path).expanduser() if env_path else pathlib.Path(__file__).resolve().parents[2] / "lerobot_2.1"


LEROBOT_REPO = _resolve_lerobot_repo()
LEROBOT_SRC = LEROBOT_REPO / "src"
if LEROBOT_SRC.exists() and str(LEROBOT_SRC) not in sys.path:
    sys.path.append(str(LEROBOT_SRC))

from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata  # noqa: E402
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig  # noqa: E402
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy  # noqa: E402


# --------------------------------------------------------------------------
# Helper utilities shared with train_trak_lerobot.py
# --------------------------------------------------------------------------


def _ensure_hf_cache_env(root: pathlib.Path) -> None:
    """Keep Hugging Face caches under the dataset root to avoid permission issues."""

    os.environ.setdefault("HF_HOME", str(root))
    os.environ.setdefault("HF_DATASETS_CACHE", str(root / "hf_datasets_cache"))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(root / "hub"))


def _patch_lerobot_imports() -> None:
    """Patch missing soft dependencies used by the vendored LeRobot code."""

    import contextlib
    import types

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


def load_lerobot_policy(
    pretrained_dir: pathlib.Path,
    device: torch.device,
) -> Tuple[DiffusionPolicy, DiffusionConfig]:
    """Load a LeRobot diffusion policy from a ``model.safetensors`` directory."""

    _patch_lerobot_imports()
    checkpoints_dir = pathlib.Path(pretrained_dir)
    with open(checkpoints_dir / "config.json", "r") as f:
        cfg_dict = json.load(f)

    cfg_dict.pop("type", None)
    try:
        config = DiffusionConfig(**cfg_dict)
    except TypeError:
        sig = inspect.signature(DiffusionConfig)
        valid_keys = set(sig.parameters.keys())
        filtered = {k: v for k, v in cfg_dict.items() if k in valid_keys}
        missing = sorted(set(cfg_dict) - valid_keys)
        if missing:
            print(f"Warning: dropping unsupported config keys: {missing}")
        config = DiffusionConfig(**filtered)

    metadata_root = checkpoints_dir / "metadata"
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

    state_dict = load_safetensors(checkpoints_dir / "model.safetensors")
    policy = DiffusionPolicy(config=config, dataset_stats=dataset_stats)
    missing, unexpected = policy.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f"State dict load: missing={missing} unexpected={unexpected}")
    policy.to(device)
    policy.eval()
    return policy, config


# --------------------------------------------------------------------------
# Dataset metadata helpers.
# --------------------------------------------------------------------------


def _read_episode_lengths(dataset_root: pathlib.Path) -> List[int]:
    dataset_root = dataset_root.expanduser()
    if dataset_root.is_file():
        dataset_root = dataset_root.parent
    meta_file = dataset_root / "meta" / "episodes.jsonl"
    lengths: List[int] = []
    with open(meta_file, "r") as f:
        for line in f:
            entry = json.loads(line)
            lengths.append(int(entry["length"]))
    return lengths


def _build_metadata_from_lengths(
    lengths: List[int],
    success_mask: Optional[List[bool]] = None,
) -> Dict[str, Any]:
    ep_idxs: List[np.ndarray] = []
    start_idx = 0
    for length in lengths:
        ep_idxs.append(np.arange(start_idx, start_idx + length, dtype=np.int64))
        start_idx += length

    metadata: Dict[str, Any] = {
        "num_eps": len(lengths),
        "num_samples": int(start_idx),
        "ep_lens": np.array(lengths, dtype=np.int64),
        "ep_idxs": ep_idxs,
    }
    if success_mask is not None:
        metadata["success_mask"] = np.array(success_mask, dtype=bool)
    return metadata


class EvalPklEpisodeDataset(IterableDataset):
    """Lightweight episode reader for pkl rollouts without metadata.yaml."""

    def __init__(self, dataset_path: pathlib.Path, exec_horizon: int, sample_history: int):
        super().__init__()
        self.dataset_path = dataset_path
        self.exec_horizon = exec_horizon
        self.sample_history = sample_history

        self.episode_files = sorted(p for p in glob.glob(str(dataset_path / "*.pkl")))
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
    def _load_episode(path: str):
        import pandas as pd

        with open(path, "rb") as f:
            return pickle.load(f)

    @staticmethod
    def _infer_episode_success(episode, path: str) -> bool:
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


# --------------------------------------------------------------------------
# Error utilities (ported from diffusion_policy.common.error_util)
# --------------------------------------------------------------------------


def mean_of_mean_influence(scores_ij: np.ndarray, is_success: bool) -> float:
    return scores_ij.mean(axis=1).mean()


def mean_of_mean_influence_success(scores_ij: np.ndarray, is_success: bool) -> float:
    return scores_ij.mean(axis=1).mean() if is_success else 0.0


def sum_of_sum_influence(scores_ij: np.ndarray, is_success: bool) -> float:
    return scores_ij.sum(axis=1).sum()


def sum_of_sum_influence_success(scores_ij: np.ndarray, is_success: bool) -> float:
    return scores_ij.sum(axis=1).sum() if is_success else 0.0


def min_of_max_influence(scores_ij: np.ndarray, is_success: bool) -> float:
    s = scores_ij.max(axis=1).min()
    return -s if is_success else s


def max_of_min_influence(scores_ij: np.ndarray, is_success: bool) -> float:
    s = scores_ij.min(axis=1).max()
    return s if is_success else -s


def compute_l2_distance(
    x: np.ndarray,
    y: np.ndarray,
    normalization: Optional[str] = "uniform",
) -> np.ndarray:
    assert x.ndim == 2 and y.ndim == 2
    if normalization == "uniform":
        min_val = np.minimum(x.min(axis=0), y.min(axis=0))
        max_val = np.maximum(x.max(axis=0), y.max(axis=0))
        range_val = max_val - min_val + 1e-12
        x = (x - min_val) / range_val
        y = (y - min_val) / range_val
    elif normalization == "gaussian":
        mean = np.mean(np.vstack([x, y]), axis=0)
        std = np.std(np.vstack([x, y]), axis=0) + 1e-12
        x = (x - mean) / std
        y = (y - mean) / std
    elif normalization == "unit":
        x_norm = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
        y_norm = np.linalg.norm(y, axis=1, keepdims=True) + 1e-12
        x = x / x_norm
        y = y / y_norm

    x_norm_sq = np.sum(x**2, axis=1)[:, None]
    y_norm_sq = np.sum(y**2, axis=1)[None, :]
    cross_term = 2 * np.dot(x, y.T)
    return np.sqrt(np.maximum(x_norm_sq + y_norm_sq - cross_term, 0))


def compute_mahal_distance(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    assert x.ndim == 2 and y.ndim == 2
    cov = np.cov(y.T) + np.eye(y.shape[1]) * 1e-12
    inv_cov = np.linalg.inv(cov)
    mean = np.mean(y, axis=0)
    x_centered = x - mean
    y_centered = y - mean
    left = x_centered @ inv_cov
    dists = np.zeros((x.shape[0], y.shape[0]))
    dists += np.sum(left * x_centered, axis=1)[:, None]
    dists += np.sum((y_centered @ inv_cov) * y_centered, axis=1)[None, :]
    dists -= 2 * (left @ y_centered.T)
    return np.sqrt(np.maximum(dists, 0))


def compute_cosine_error(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    assert x.ndim == 2 and y.ndim == 2
    x_norm = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    y_norm = np.linalg.norm(y, axis=1, keepdims=True) + 1e-12
    return -1.0 * (x / x_norm) @ (y / y_norm).T


EMBEDDING_ERROR_FNS: Dict[str, Callable[[np.ndarray, np.ndarray], np.ndarray]] = {
    "l2": compute_l2_distance,
    "cosine": compute_cosine_error,
}


def topk_embedding_score(
    data_embeddings: np.ndarray,
    test_embeddings: np.ndarray,
    error_fn: str = "l2",
    k: int = 5,
    leave_one_out: bool = False,
    **kwargs: Any,
) -> np.ndarray:
    scores = EMBEDDING_ERROR_FNS[error_fn](test_embeddings, data_embeddings, **kwargs)
    assert scores.shape == (len(test_embeddings), len(data_embeddings))
    if leave_one_out:
        assert data_embeddings.shape == test_embeddings.shape
        scores += np.diag([np.inf] * data_embeddings.shape[0])
    partitioned_scores = np.partition(scores, k, axis=1)[:, :k]
    return np.mean(partitioned_scores, axis=1)


def mahal_embedding_score(
    data_embeddings: np.ndarray,
    test_embeddings: np.ndarray,
) -> np.ndarray:
    assert data_embeddings.ndim == 2 and test_embeddings.ndim == 2
    cov = np.cov(data_embeddings.T) + np.eye(data_embeddings.shape[1]) * 1e-12
    inv_cov = np.linalg.inv(cov)
    mean = np.mean(data_embeddings, axis=0)
    centered = test_embeddings - mean
    proj = centered @ inv_cov
    mahals = np.sum(centered * proj, axis=1)
    return np.sqrt(np.maximum(mahals, 0))


def compute_test_embedding_scores(
    data_embeddings: np.ndarray,
    test_embeddings: np.ndarray,
    method: str,
    method_kwargs: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    if method == "topk":
        assert method_kwargs is not None
        return topk_embedding_score(
            data_embeddings=data_embeddings,
            test_embeddings=test_embeddings,
            **method_kwargs,
        )
    elif method == "mahal":
        return mahal_embedding_score(
            data_embeddings=data_embeddings,
            test_embeddings=test_embeddings,
        )
    else:
        raise ValueError(f"Embedding score method {method} is not supported.")


def sample_to_trajectory_scores(
    sample_scores: np.ndarray,
    num_eps: int,
    ep_idxs: List[np.ndarray],
    ep_lens: np.ndarray,
    aggr_fn: Callable[[np.ndarray], float] = np.mean,
    return_dtype: type = np.float32,
) -> np.ndarray:
    assert len(sample_scores) == ep_lens.sum()
    traj_scores = np.zeros(num_eps, dtype=return_dtype)
    for i, ep_idx, ep_len in zip(range(num_eps), ep_idxs, ep_lens):
        scores_i: np.ndarray = sample_scores[ep_idx]
        assert scores_i.shape == (ep_len,)
        traj_scores[i] = aggr_fn(scores_i)
    return traj_scores


def pairwise_sample_to_trajectory_scores(
    pairwise_sample_scores: np.ndarray,
    num_test_eps: int,
    num_train_eps: int,
    test_ep_idxs: List[np.ndarray],
    train_ep_idxs: List[np.ndarray],
    test_ep_lens: np.ndarray,
    train_ep_lens: np.ndarray,
    success_mask: np.ndarray,
    aggr_fn: Callable[[np.ndarray, bool], float] = mean_of_mean_influence,
    return_dtype: type = np.float32,
) -> np.ndarray:
    assert pairwise_sample_scores.shape == (test_ep_lens.sum(), train_ep_lens.sum())
    traj_scores = np.zeros((num_test_eps, num_train_eps), dtype=return_dtype)
    for i, test_idx in enumerate(test_ep_idxs):
        for j, train_idx in enumerate(train_ep_idxs):
            sample_scores_ij = pairwise_sample_scores[np.ix_(test_idx, train_idx)]
            assert sample_scores_ij.shape == (test_ep_lens[i], train_ep_lens[j])
            traj_scores[i, j] = aggr_fn(sample_scores_ij, success_mask[i])
    return traj_scores


def compute_demo_quality_scores(
    traj_scores: np.ndarray,
    success_mask: np.ndarray,
    metric: str = "net",
) -> Optional[np.ndarray]:
    assert traj_scores.shape[0] == len(success_mask)
    has_succ = np.any(success_mask == True)  # noqa: E712
    has_fail = np.any(success_mask == False)  # noqa: E712
    if metric == "net":
        succ_scores = compute_demo_quality_scores(traj_scores, success_mask, metric="succ")
        fail_scores = compute_demo_quality_scores(traj_scores, success_mask, metric="fail")
        succ_scores = succ_scores if succ_scores is not None else 0.0
        fail_scores = fail_scores if fail_scores is not None else 0.0
        return succ_scores + fail_scores
    elif metric == "succ":
        return traj_scores[success_mask].sum(axis=0) if has_succ else None
    elif metric == "fail":
        return -traj_scores[~success_mask].sum(axis=0) if has_fail else None
    else:
        raise ValueError(f"Quality metric {metric} is not supported.")


# --------------------------------------------------------------------------
# Results utilities (ported from diffusion_policy.common.results_util)
# --------------------------------------------------------------------------


DEMO_RESULT_KEYS = [
    "offline_policy_loss",
    "offline_action_diversity",
    "offline_state_diversity",
    "online_state_similarity",
    "online_demo_score",
    "online_trak_influence",
]


LOAD_POLICY_LOSS_KWARGS = {
    "default_diffusion": {
        "dtype": np.float16,
        "seed": 0,
        "exp_type": "diffusion",
        "num_timesteps": 64,
    },
}


LOAD_ACTION_VARIANCE_KWARGS = {
    "default_diffusion": {
        "dtype": np.float16,
        "seed": 0,
        "exp_type": "diffusion",
        "num_timesteps": 32,
    },
}


LOAD_STATE_EMBEDDING_KWARGS = {
    "default_state": {
        "embedding_names": [],
        "embedding_dims": [],
        "dtype": np.float16,
        "seed": 0,
    },
    "default_image": {
        "embedding_names": ["dinov2"],
        "embedding_dims": [384],
        "dtype": np.float16,
        "seed": 0,
    },
}


LOAD_DEMO_SCORE_KWARGS = {
    "default": {
        "dtype": np.float16,
        "seed": 0,
    },
}


LOAD_TRAK_KWARGS = {
    "default_diffusion": {
        "proj_dim": 4000,
        "lambda_reg": 0.0,
        "num_ckpts": 1,
        "trak_exp_name": "all_episodes",
        "dtype": np.float32,
        "seed": 0,
        "exp_type": "diffusion",
        "loss_fn": "square",
        "num_timesteps": 64,
    },
}


def check_mmap(array: np.ndarray) -> None:
    if np.any(np.isnan(array)):
        raise ValueError("Array contains NaNs.")
    if np.any(np.isinf(array)):
        raise ValueError("Array contains infs.")


def load_mmap(
    filename: pathlib.Path,
    shape: Tuple[int, ...],
    dtype: type,
    mode: str = "r",
) -> np.ndarray:
    array = np.array(open_memmap(filename=filename, shape=shape, dtype=dtype, mode=mode))
    check_mmap(array)
    return array


def get_policy_losses(
    eval_dir: pathlib.Path,
    train_set_size: int,
    test_set_size: int,
    exp_date: str,
    return_dtype: type,
    load_train: bool,
    load_test: bool,
    **kwargs,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    exp_name = f"{exp_date}_policy_loss-seed={kwargs['seed']}"
    if kwargs.get("exp_type") == "diffusion":
        exp_name = f"{exp_name}-num_timesteps={kwargs.get('num_timesteps')}"
    exp_dir = eval_dir / exp_name
    train_scores = (
        load_mmap(exp_dir / "train_set_loss.mmap", (train_set_size, 1), kwargs["dtype"]).squeeze().astype(return_dtype)
        if load_train
        else None
    )
    test_scores = (
        load_mmap(exp_dir / "test_set_loss.mmap", (test_set_size, 1), kwargs["dtype"]).squeeze().astype(return_dtype)
        if load_test
        else None
    )
    return train_scores, test_scores


def get_action_variances(
    eval_dir: pathlib.Path,
    train_set_size: int,
    test_set_size: int,
    exp_date: str,
    return_dtype: type,
    load_train: bool,
    load_test: bool,
    **kwargs,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    exp_name = f"{exp_date}_action_variance-seed={kwargs['seed']}"
    if kwargs.get("exp_type") == "diffusion":
        exp_name = f"{exp_name}-num_timesteps={kwargs.get('num_timesteps')}"
    exp_dir = eval_dir / exp_name
    train_scores = (
        load_mmap(exp_dir / "train_set_var.mmap", (train_set_size, 1), kwargs["dtype"]).squeeze().astype(return_dtype)
        if load_train
        else None
    )
    test_scores = (
        load_mmap(exp_dir / "test_set_var.mmap", (test_set_size, 1), kwargs["dtype"]).squeeze().astype(return_dtype)
        if load_test
        else None
    )
    return train_scores, test_scores


def get_state_embeddings(
    eval_dir: pathlib.Path,
    train_set_size: int,
    test_set_size: int,
    embedding_dims: List[int],
    embedding_names: List[str],
    exp_date: str,
    return_dtype: type,
    load_train: bool,
    load_test: bool,
    **kwargs,
) -> Dict[str, Tuple[Optional[np.ndarray], Optional[np.ndarray]]]:
    exp_name = f"{exp_date}_embeddings-seed={kwargs['seed']}"
    exp_dir = eval_dir / exp_name
    results: Dict[str, Tuple[Optional[np.ndarray], Optional[np.ndarray]]] = {}
    for emb_dim, emb_name in zip(embedding_dims, embedding_names):
        results[emb_name] = (
            load_mmap(exp_dir / f"train_set_{emb_name}_emb.mmap", (train_set_size, emb_dim), kwargs["dtype"])
            .squeeze()
            .astype(return_dtype)
            if load_train
            else None,
            load_mmap(exp_dir / f"test_set_{emb_name}_emb.mmap", (test_set_size, emb_dim), kwargs["dtype"])
            .squeeze()
            .astype(return_dtype)
            if load_test
            else None,
        )
    return results


def get_demo_score_scores(
    eval_dir: pathlib.Path,
    train_set_size: int,
    test_set_size: int,
    exp_date: str,
    return_dtype: type,
    load_train: bool,
    load_test: bool,
    **kwargs,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    exp_name = f"{exp_date}_demo_score-seed={kwargs['seed']}"
    exp_dir = eval_dir / exp_name
    train_scores = (
        load_mmap(exp_dir / "train_set_demo_scores.mmap", (train_set_size, 1), kwargs["dtype"])
        .squeeze()
        .astype(return_dtype)
        if load_train
        else None
    )
    test_scores = (
        load_mmap(exp_dir / "test_set_demo_scores.mmap", (test_set_size, 1), kwargs["dtype"])
        .squeeze()
        .astype(return_dtype)
        if load_test
        else None
    )
    return train_scores, test_scores


def get_trak_scores(
    eval_dir: pathlib.Path,
    train_set_size: int,
    test_set_size: int,
    exp_date: str,
    return_dtype: type,
    **kwargs,
) -> np.ndarray:
    exp_name = (
        f"{exp_date}_trak_results-proj_dim={kwargs['proj_dim']}"
        f"-lambda_reg={kwargs['lambda_reg']}-num_ckpts={kwargs['num_ckpts']}-seed={kwargs['seed']}"
    )
    if kwargs.get("exp_type") == "diffusion":
        exp_name = f"{exp_name}-loss_fn={kwargs['loss_fn']}-num_timesteps={kwargs['num_timesteps']}"
    exp_dir = eval_dir / exp_name
    scores_path = "/root/2_250_pkl/train_trak_lerobot_dp/scores/all_episodes.mmap"
    scores = load_mmap(scores_path, (train_set_size, test_set_size), kwargs["dtype"])
    return scores.T.astype(return_dtype)


# --------------------------------------------------------------------------
# Demonstration scoring methods (adapted from eval_demonstration_scores)
# --------------------------------------------------------------------------


TEST_EMBEDDING_SCORE_FNS: Dict[str, Dict[str, Any]] = {
    "mahal": {"method": "mahal"},
    "top1_l2": {"method": "topk", "method_kwargs": {"error_fn": "l2", "k": 1}},
    "top5_l2": {"method": "topk", "method_kwargs": {"error_fn": "l2", "k": 5}},
    "top10_l2": {"method": "topk", "method_kwargs": {"error_fn": "l2", "k": 10}},
    "top1_cosine": {"method": "topk", "method_kwargs": {"error_fn": "cosine", "k": 1}},
    "top5_cosine": {"method": "topk", "method_kwargs": {"error_fn": "cosine", "k": 5}},
    "top10_cosine": {"method": "topk", "method_kwargs": {"error_fn": "cosine", "k": 10}},
}


PAIRWISE_EMBEDDING_SCORE_FNS: Dict[str, Callable[[np.ndarray, np.ndarray], np.ndarray]] = {
    "l2": compute_l2_distance,
    "cosine": compute_cosine_error,
}


def get_rollout_idx(
    metric: str,
    num_rollouts: Union[str, int],
    success_mask: np.ndarray,
    random_choice: bool = False,
) -> Optional[np.ndarray]:
    valid_idx = np.arange(len(success_mask))
    if isinstance(num_rollouts, str):
        assert num_rollouts == "all"
        return valid_idx
    if metric == "succ":
        valid_idx = valid_idx[success_mask]
    elif metric == "fail":
        valid_idx = valid_idx[~success_mask]
    if num_rollouts > len(valid_idx):
        return None
    if not random_choice:
        return valid_idx[:num_rollouts]
    return np.random.choice(valid_idx, num_rollouts, replace=False)


def get_test_embedding_score_fns(embedding_name: str) -> List[str]:
    if embedding_name == "dinov2":
        return ["top5_cosine", "top10_cosine"]
    elif embedding_name == "policy":
        return ["mahal", "top5_l2", "top10_l2"]
    else:
        raise ValueError(f"Unsupported embedding {embedding_name}.")


def get_pairwise_embedding_score_fns(embedding_name: str) -> List[str]:
    if embedding_name == "dinov2":
        return ["cosine"]
    elif embedding_name == "policy":
        return ["l2"]
    else:
        raise ValueError(f"Unsupported embedding {embedding_name}.")


def offline_policy_loss_method(
    load_key: str,
    eval_dir: str,
    exp_date: str,
    compute_holdout: bool,
    return_dtype: type,
    train_set_metadata: Dict[str, Any],
    holdout_set_metadata: Dict[str, Any],
    test_set_metadata: Dict[str, Any],
) -> Dict[str, np.ndarray]:
    train_holdout_loss, _ = get_policy_losses(
        eval_dir=pathlib.Path(eval_dir),
        train_set_size=train_set_metadata["num_samples"] + holdout_set_metadata["num_samples"],
        test_set_size=test_set_metadata["num_samples"],
        exp_date=exp_date,
        return_dtype=return_dtype,
        load_train=True,
        load_test=False,
        **LOAD_POLICY_LOSS_KWARGS[load_key],
    )
    assert isinstance(train_holdout_loss, np.ndarray)
    result = {
        "train": sample_to_trajectory_scores(
            sample_scores=train_holdout_loss[: train_set_metadata["num_samples"]],
            num_eps=train_set_metadata["num_eps"],
            ep_idxs=train_set_metadata["ep_idxs"],
            ep_lens=train_set_metadata["ep_lens"],
            aggr_fn=np.mean,
            return_dtype=return_dtype,
        )
    }
    if compute_holdout and holdout_set_metadata["num_samples"] > 0:
        result["holdout"] = sample_to_trajectory_scores(
            sample_scores=train_holdout_loss[train_set_metadata["num_samples"] :],
            num_eps=holdout_set_metadata["num_eps"],
            ep_idxs=holdout_set_metadata["ep_idxs"],
            ep_lens=holdout_set_metadata["ep_lens"],
            aggr_fn=np.mean,
            return_dtype=return_dtype,
        )
    return result


def offline_action_diversity_method(
    load_key: str,
    eval_dir: str,
    exp_date: str,
    compute_holdout: bool,
    return_dtype: type,
    train_set_metadata: Dict[str, Any],
    holdout_set_metadata: Dict[str, Any],
    test_set_metadata: Dict[str, Any],
) -> Dict[str, np.ndarray]:
    train_holdout_var, _ = get_action_variances(
        eval_dir=pathlib.Path(eval_dir),
        train_set_size=train_set_metadata["num_samples"] + holdout_set_metadata["num_samples"],
        test_set_size=test_set_metadata["num_samples"],
        exp_date=exp_date,
        return_dtype=return_dtype,
        load_train=True,
        load_test=False,
        **LOAD_ACTION_VARIANCE_KWARGS[load_key],
    )
    assert isinstance(train_holdout_var, np.ndarray)
    result = {
        "train": sample_to_trajectory_scores(
            sample_scores=train_holdout_var[: train_set_metadata["num_samples"]],
            num_eps=train_set_metadata["num_eps"],
            ep_idxs=train_set_metadata["ep_idxs"],
            ep_lens=train_set_metadata["ep_lens"],
            aggr_fn=np.mean,
            return_dtype=return_dtype,
        )
    }
    if compute_holdout and holdout_set_metadata["num_samples"] > 0:
        result["holdout"] = sample_to_trajectory_scores(
            sample_scores=train_holdout_var[train_set_metadata["num_samples"] :],
            num_eps=holdout_set_metadata["num_eps"],
            ep_idxs=holdout_set_metadata["ep_idxs"],
            ep_lens=holdout_set_metadata["ep_lens"],
            aggr_fn=np.mean,
            return_dtype=return_dtype,
        )
    return result


def offline_state_diversity_method(
    load_key: str,
    eval_dir: str,
    exp_date: str,
    compute_holdout: bool,
    return_dtype: type,
    train_set_metadata: Dict[str, Any],
    holdout_set_metadata: Dict[str, Any],
    test_set_metadata: Dict[str, Any],
    policy_embedding_dim: int,
) -> Dict[str, Dict[str, np.ndarray]]:
    load_kwargs = deepcopy(LOAD_STATE_EMBEDDING_KWARGS[load_key])
    load_kwargs["embedding_names"].append("policy")
    load_kwargs["embedding_dims"].append(policy_embedding_dim)
    state_embeddings = get_state_embeddings(
        eval_dir=pathlib.Path(eval_dir),
        train_set_size=train_set_metadata["num_samples"] + holdout_set_metadata["num_samples"],
        test_set_size=test_set_metadata["num_samples"],
        exp_date=exp_date,
        return_dtype=return_dtype,
        load_train=True,
        load_test=False,
        **load_kwargs,
    )
    result = defaultdict(dict)
    for embedding_name in load_kwargs["embedding_names"]:
        train_holdout_embeddings = state_embeddings[embedding_name][0]
        if train_holdout_embeddings is None:
            continue
        train_embeddings = train_holdout_embeddings[: train_set_metadata["num_samples"]]
        for score_fn in get_test_embedding_score_fns(embedding_name):
            mask = np.zeros(train_set_metadata["num_samples"], dtype=bool)
            sample_scores = np.zeros(train_set_metadata["num_samples"], dtype=return_dtype)
            for ep_idx, ep_len in zip(train_set_metadata["ep_idxs"], train_set_metadata["ep_lens"]):
                mask[ep_idx] = True
                ep_embeddings = train_embeddings[mask]
                non_ep_embeddings = train_embeddings[~mask]
                sample_scores[ep_idx] = compute_test_embedding_scores(
                    data_embeddings=non_ep_embeddings,
                    test_embeddings=ep_embeddings,
                    **TEST_EMBEDDING_SCORE_FNS[score_fn],
                )
                mask[:] = False
            exp_key = f"{embedding_name}-{score_fn}"
            result["train"][exp_key] = sample_to_trajectory_scores(
                sample_scores=sample_scores,
                num_eps=train_set_metadata["num_eps"],
                ep_idxs=train_set_metadata["ep_idxs"],
                ep_lens=train_set_metadata["ep_lens"],
                aggr_fn=np.mean,
                return_dtype=return_dtype,
            )
            print(f"Completed: Offline State Diversity [Train] [{exp_key}].")
    if compute_holdout and holdout_set_metadata["num_samples"] > 0:
        for embedding_name in load_kwargs["embedding_names"]:
            train_holdout_embeddings = state_embeddings[embedding_name][0]
            if train_holdout_embeddings is None:
                continue
            train_embeddings = train_holdout_embeddings[: train_set_metadata["num_samples"]]
            holdout_embeddings = train_holdout_embeddings[train_set_metadata["num_samples"] :]
            for score_fn in get_test_embedding_score_fns(embedding_name):
                sample_scores = compute_test_embedding_scores(
                    data_embeddings=train_embeddings,
                    test_embeddings=holdout_embeddings,
                    **TEST_EMBEDDING_SCORE_FNS[score_fn],
                )
                exp_key = f"{embedding_name}-{score_fn}"
                result["holdout"][exp_key] = sample_to_trajectory_scores(
                    sample_scores=sample_scores,
                    num_eps=holdout_set_metadata["num_eps"],
                    ep_idxs=holdout_set_metadata["ep_idxs"],
                    ep_lens=holdout_set_metadata["ep_lens"],
                    aggr_fn=np.mean,
                    return_dtype=return_dtype,
                )
                print(f"Completed: Offline State Diversity [Holdout] [{exp_key}].")
    return result


def online_state_similarity_method(
    load_key: str,
    eval_dir: str,
    exp_date: str,
    compute_holdout: bool,
    return_dtype: type,
    train_set_metadata: Dict[str, Any],
    holdout_set_metadata: Dict[str, Any],
    test_set_metadata: Dict[str, Any],
    policy_embedding_dim: int,
) -> Dict[str, Dict[str, np.ndarray]]:
    load_kwargs = deepcopy(LOAD_STATE_EMBEDDING_KWARGS[load_key])
    load_kwargs["embedding_names"].append("policy")
    load_kwargs["embedding_dims"].append(policy_embedding_dim)
    state_embeddings = get_state_embeddings(
        eval_dir=pathlib.Path(eval_dir),
        train_set_size=train_set_metadata["num_samples"] + holdout_set_metadata["num_samples"],
        test_set_size=test_set_metadata["num_samples"],
        exp_date=exp_date,
        return_dtype=return_dtype,
        load_train=True,
        load_test=True,
        **load_kwargs,
    )
    metrics = ["succ", "fail", "net"]
    aggr_fns = {
        "mean_of_mean": mean_of_mean_influence,
        "mean_of_mean_success": mean_of_mean_influence_success,
        "sum_of_sum": sum_of_sum_influence,
        "sum_of_sum_success": sum_of_sum_influence_success,
    }
    test_rollout_fractions = {
        "succ": ["all"],
        "fail": ["all"],
        "net": [0.01, 0.05, 0.10, 0.25, 0.5, 1.00, "all"],
    }
    result = defaultdict(dict)

    def routine(split: str) -> None:
        demo_meta = train_set_metadata if split == "train" else holdout_set_metadata
        for embedding_name in load_kwargs["embedding_names"]:
            demo_embeddings_all = state_embeddings[embedding_name][0]
            test_embeddings = state_embeddings[embedding_name][1]
            if demo_embeddings_all is None or test_embeddings is None:
                continue
            test_embeddings = test_embeddings[: test_set_metadata["num_samples"]]
            if split == "train":
                demo_embeddings = demo_embeddings_all[: train_set_metadata["num_samples"]]
            else:
                demo_embeddings = demo_embeddings_all[train_set_metadata["num_samples"] :]
            for score_fn in get_pairwise_embedding_score_fns(embedding_name):
                pairwise_scores = -1 * PAIRWISE_EMBEDDING_SCORE_FNS[score_fn](test_embeddings, demo_embeddings)
                traj_scores = pairwise_sample_to_trajectory_scores(
                    pairwise_sample_scores=pairwise_scores,
                    num_test_eps=test_set_metadata["num_eps"],
                    num_train_eps=demo_meta["num_eps"],
                    test_ep_idxs=test_set_metadata["ep_idxs"],
                    train_ep_idxs=demo_meta["ep_idxs"],
                    test_ep_lens=test_set_metadata["ep_lens"],
                    train_ep_lens=demo_meta["ep_lens"],
                    success_mask=test_set_metadata["success_mask"],
                    aggr_fn=aggr_fns["mean_of_mean"],
                    return_dtype=return_dtype,
                )
                for aggr_fn_name, aggr_fn in aggr_fns.items():
                    traj_scores = pairwise_sample_to_trajectory_scores(
                        pairwise_sample_scores=pairwise_scores,
                        num_test_eps=test_set_metadata["num_eps"],
                        num_train_eps=demo_meta["num_eps"],
                        test_ep_idxs=test_set_metadata["ep_idxs"],
                        train_ep_idxs=demo_meta["ep_idxs"],
                        test_ep_lens=test_set_metadata["ep_lens"],
                        train_ep_lens=demo_meta["ep_lens"],
                        success_mask=test_set_metadata["success_mask"],
                        aggr_fn=aggr_fn,
                        return_dtype=return_dtype,
                    )
                    for metric in metrics:
                        for fraction in test_rollout_fractions[metric]:
                            num_rollouts = fraction if isinstance(fraction, str) else int(
                                test_set_metadata["num_eps"] * fraction
                            )
                            rollout_idx = get_rollout_idx(
                                metric=metric,
                                num_rollouts=num_rollouts,
                                success_mask=test_set_metadata["success_mask"],
                                random_choice=False,
                            )
                            if rollout_idx is None:
                                continue
                            exp_key = f"{embedding_name}-{score_fn}-{aggr_fn_name}-{metric}-{num_rollouts}"
                            result[split][exp_key] = compute_demo_quality_scores(
                                traj_scores=traj_scores[rollout_idx, :],
                                success_mask=test_set_metadata["success_mask"][rollout_idx],
                                metric=metric,
                            )
                            print(f"Completed: Online State Similarity [{split.title()}] [{exp_key}].")

    routine("train")
    if compute_holdout and holdout_set_metadata["num_samples"] > 0:
        routine("holdout")
    return result


def online_demo_score_method(
    load_key: str,
    eval_dir: str,
    exp_date: str,
    compute_holdout: bool,
    return_dtype: type,
    train_set_metadata: Dict[str, Any],
    holdout_set_metadata: Dict[str, Any],
    test_set_metadata: Dict[str, Any],
) -> Dict[str, np.ndarray]:
    train_holdout_scores, _ = get_demo_score_scores(
        eval_dir=pathlib.Path(eval_dir),
        train_set_size=train_set_metadata["num_samples"] + holdout_set_metadata["num_samples"],
        test_set_size=test_set_metadata["num_samples"],
        exp_date=exp_date,
        return_dtype=return_dtype,
        load_train=True,
        load_test=False,
        **LOAD_DEMO_SCORE_KWARGS[load_key],
    )
    assert isinstance(train_holdout_scores, np.ndarray)
    result = {
        "train": sample_to_trajectory_scores(
            sample_scores=train_holdout_scores[: train_set_metadata["num_samples"]],
            num_eps=train_set_metadata["num_eps"],
            ep_idxs=train_set_metadata["ep_idxs"],
            ep_lens=train_set_metadata["ep_lens"],
            aggr_fn=np.mean,
            return_dtype=return_dtype,
        )
    }
    if compute_holdout and holdout_set_metadata["num_samples"] > 0:
        result["holdout"] = sample_to_trajectory_scores(
            sample_scores=train_holdout_scores[train_set_metadata["num_samples"] :],
            num_eps=holdout_set_metadata["num_eps"],
            ep_idxs=holdout_set_metadata["ep_idxs"],
            ep_lens=holdout_set_metadata["ep_lens"],
            aggr_fn=np.mean,
            return_dtype=return_dtype,
        )
    return result


def online_trak_influence_method(
    load_key: str,
    eval_dir: str,
    exp_date: str,
    compute_holdout: bool,
    return_dtype: type,
    train_set_metadata: Dict[str, Any],
    holdout_set_metadata: Dict[str, Any],
    test_set_metadata: Dict[str, Any],
) -> Dict[str, Dict[str, np.ndarray]]:
    train_holdout_scores = get_trak_scores(
        eval_dir=pathlib.Path(eval_dir),
        train_set_size=train_set_metadata["num_samples"] + holdout_set_metadata["num_samples"],
        test_set_size=test_set_metadata["num_samples"],
        exp_date=exp_date,
        return_dtype=return_dtype,
        **LOAD_TRAK_KWARGS[load_key],
    )
    metrics = ["succ", "fail", "net"]
    aggr_fns = {
        "mean_of_mean": mean_of_mean_influence,
        "mean_of_mean_success": mean_of_mean_influence_success,
        "sum_of_sum": sum_of_sum_influence,
        "sum_of_sum_success": sum_of_sum_influence_success,
        "min_of_max": min_of_max_influence,
        "max_of_min": max_of_min_influence,
    }
    test_rollout_fractions = {
        "succ": ["all"],
        "fail": ["all"],
        "net": [0.01, 0.05, 0.10, 0.25, 0.5, 1.00, "all"],
    }
    result = defaultdict(dict)

    def routine(split: str) -> None:
        demo_meta = train_set_metadata if split == "train" else holdout_set_metadata
        if demo_meta["num_eps"] == 0:
            return
        if split == "train":
            test_demo_scores = train_holdout_scores[:, : train_set_metadata["num_samples"]]
        else:
            test_demo_scores = train_holdout_scores[:, train_set_metadata["num_samples"] :]
        traj_scores = pairwise_sample_to_trajectory_scores(
            pairwise_sample_scores=test_demo_scores,
            num_test_eps=test_set_metadata["num_eps"],
            num_train_eps=demo_meta["num_eps"],
            test_ep_idxs=test_set_metadata["ep_idxs"],
            train_ep_idxs=demo_meta["ep_idxs"],
            test_ep_lens=test_set_metadata["ep_lens"],
            train_ep_lens=demo_meta["ep_lens"],
            success_mask=test_set_metadata["success_mask"],
            aggr_fn=aggr_fns["mean_of_mean"],
            return_dtype=return_dtype,
        )
        for aggr_fn_name, aggr_fn in aggr_fns.items():
            traj_scores = pairwise_sample_to_trajectory_scores(
                pairwise_sample_scores=test_demo_scores,
                num_test_eps=test_set_metadata["num_eps"],
                num_train_eps=demo_meta["num_eps"],
                test_ep_idxs=test_set_metadata["ep_idxs"],
                train_ep_idxs=demo_meta["ep_idxs"],
                test_ep_lens=test_set_metadata["ep_lens"],
                train_ep_lens=demo_meta["ep_lens"],
                success_mask=test_set_metadata["success_mask"],
                aggr_fn=aggr_fn,
                return_dtype=return_dtype,
            )
            for metric in metrics:
                for fraction in test_rollout_fractions[metric]:
                    num_rollouts = fraction if isinstance(fraction, str) else int(
                        test_set_metadata["num_eps"] * fraction
                    )
                    rollout_idx = get_rollout_idx(
                        metric=metric,
                        num_rollouts=num_rollouts,
                        success_mask=test_set_metadata["success_mask"],
                        random_choice=False,
                    )
                    if rollout_idx is None:
                        continue
                    exp_key = f"{aggr_fn_name}-{metric}-{num_rollouts}"
                    result[split][exp_key] = compute_demo_quality_scores(
                        traj_scores=traj_scores[rollout_idx, :],
                        success_mask=test_set_metadata["success_mask"][rollout_idx],
                        metric=metric,
                    )
                    print(f"Completed: Online TRAK Influence [{split.title()}] [{exp_key}].")

    routine("train")
    if compute_holdout and holdout_set_metadata["num_samples"] > 0:
        routine("holdout")
    return result


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _infer_policy_embedding_dim(policy: DiffusionPolicy) -> Optional[int]:
    """Approximate the dimensionality of policy state embeddings."""

    try:
        config = policy.config
        dim = config.robot_state_feature.shape[0]
        if config.image_features:
            if isinstance(policy.diffusion.rgb_encoder, torch.nn.ModuleList):
                feat_dim = policy.diffusion.rgb_encoder[0].feature_dim
            else:
                feat_dim = policy.diffusion.rgb_encoder.feature_dim
            dim += feat_dim * len(config.image_features)
        if config.env_state_feature:
            dim += config.env_state_feature.shape[0]
        return dim * config.n_obs_steps
    except Exception:
        return None


def _load_dataset_metadata(dataset_path: pathlib.Path) -> Dict[str, Any]:
    lengths = _read_episode_lengths(dataset_path)
    return _build_metadata_from_lengths(lengths)


@click.command()
@click.option("--exp_name", type=str, required=True)
@click.option("--eval_dir", type=str, required=True)
@click.option("--pretrained_dir", type=str, required=True)
@click.option("--train_dataset_path", type=str, required=True)
@click.option("--holdout_dataset_path", type=str, default=None)
@click.option("--result_date", type=str, required=True)
@click.option("--overwrite", type=bool, default=False)
@click.option("--device", type=str, default="cuda:0")
@click.option("--seed", type=int, default=0)
@click.option("--use_half_precision", type=bool, default=False)
@click.option("--compute_holdout", type=bool, default=False)
@click.option("--policy_embedding_dim", type=int, default=None)
@click.option("--state_embedding_load_key", type=str, default="default_image")
@click.option("--eval_offline_policy_loss", type=bool, required=False)
@click.option("--eval_offline_action_diversity", type=bool, required=False)
@click.option("--eval_offline_state_diversity", type=bool, required=False)
@click.option("--eval_online_state_similarity", type=bool, required=False)
@click.option("--eval_online_demo_score", type=bool, required=False)
@click.option("--eval_online_trak_influence", type=bool, required=False)
def main(
    exp_name: str,
    eval_dir: str,
    pretrained_dir: str,
    train_dataset_path: str,
    holdout_dataset_path: Optional[str],
    result_date: str,
    overwrite: bool,
    device: str,
    seed: int,
    use_half_precision: bool,
    compute_holdout: bool,
    policy_embedding_dim: Optional[int],
    state_embedding_load_key: str,
    eval_offline_policy_loss: bool,
    eval_offline_action_diversity: bool,
    eval_offline_state_diversity: bool,
    eval_online_state_similarity: bool,
    eval_online_demo_score: bool,
    eval_online_trak_influence: bool,
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    device = torch.device(device)

    pretrained_dir = pathlib.Path(pretrained_dir).expanduser()
    policy, cfg = load_lerobot_policy(pretrained_dir, device)
    if policy_embedding_dim is None:
        policy_embedding_dim = _infer_policy_embedding_dim(policy)

    train_set_metadata = _load_dataset_metadata(pathlib.Path(train_dataset_path))
    if compute_holdout:
        if holdout_dataset_path is None:
            raise ValueError("Provide --holdout_dataset_path when compute_holdout=True")
        holdout_set_metadata = _load_dataset_metadata(pathlib.Path(holdout_dataset_path))
    else:
        holdout_set_metadata = _build_metadata_from_lengths([])

    test_set = EvalPklEpisodeDataset(
        dataset_path=pathlib.Path(eval_dir) / "episodes",
        exec_horizon=1,
        sample_history=0,
    )
    test_set_metadata = _build_metadata_from_lengths(
        test_set.episode_lengths,
        success_mask=test_set.episode_successes,
    )

    del test_set

    save_dir = pathlib.Path(eval_dir) / exp_name
    save_dir.mkdir(exist_ok=True, parents=True)
    return_dtype = np.float16 if use_half_precision else np.float32

    def save_result(result: Dict[str, Any], result_name: str) -> None:
        save_path = save_dir / f"{result_name}.pkl"
        if save_path.exists() and not overwrite:
            raise ValueError(f"Output path {save_path} already exists.")
        with open(save_path, "wb") as f:
            pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Results saved [{result_name}]")

    if eval_offline_policy_loss:
        save_result(
            offline_policy_loss_method(
                load_key="default_diffusion",
                eval_dir=eval_dir,
                exp_date=result_date,
                compute_holdout=compute_holdout,
                return_dtype=return_dtype,
                train_set_metadata=train_set_metadata,
                holdout_set_metadata=holdout_set_metadata,
                test_set_metadata=test_set_metadata,
            ),
            "offline_policy_loss",
        )

    if eval_offline_action_diversity:
        save_result(
            offline_action_diversity_method(
                load_key="default_diffusion",
                eval_dir=eval_dir,
                exp_date=result_date,
                compute_holdout=compute_holdout,
                return_dtype=return_dtype,
                train_set_metadata=train_set_metadata,
                holdout_set_metadata=holdout_set_metadata,
                test_set_metadata=test_set_metadata,
            ),
            "offline_action_diversity",
        )

    if eval_offline_state_diversity:
        if policy_embedding_dim is None:
            raise ValueError("policy_embedding_dim is required for offline_state_diversity.")
        save_result(
            offline_state_diversity_method(
                load_key=state_embedding_load_key,
                eval_dir=eval_dir,
                exp_date=result_date,
                compute_holdout=compute_holdout,
                return_dtype=return_dtype,
                train_set_metadata=train_set_metadata,
                holdout_set_metadata=holdout_set_metadata,
                test_set_metadata=test_set_metadata,
                policy_embedding_dim=policy_embedding_dim,
            ),
            "offline_state_diversity",
        )

    if eval_online_state_similarity:
        if policy_embedding_dim is None:
            raise ValueError("policy_embedding_dim is required for online_state_similarity.")
        save_result(
            online_state_similarity_method(
                load_key=state_embedding_load_key,
                eval_dir=eval_dir,
                exp_date=result_date,
                compute_holdout=compute_holdout,
                return_dtype=return_dtype,
                train_set_metadata=train_set_metadata,
                holdout_set_metadata=holdout_set_metadata,
                test_set_metadata=test_set_metadata,
                policy_embedding_dim=policy_embedding_dim,
            ),
            "online_state_similarity",
        )

    if eval_online_demo_score:
        save_result(
            online_demo_score_method(
                load_key="default",
                eval_dir=eval_dir,
                exp_date=result_date,
                compute_holdout=compute_holdout,
                return_dtype=return_dtype,
                train_set_metadata=train_set_metadata,
                holdout_set_metadata=holdout_set_metadata,
                test_set_metadata=test_set_metadata,
            ),
            "online_demo_score",
        )

    if eval_online_trak_influence:
        save_result(
            online_trak_influence_method(
                load_key="default_diffusion",
                eval_dir=eval_dir,
                exp_date=result_date,
                compute_holdout=compute_holdout,
                return_dtype=return_dtype,
                train_set_metadata=train_set_metadata,
                holdout_set_metadata=holdout_set_metadata,
                test_set_metadata=test_set_metadata,
            ),
            "online_trak_influence",
        )

    print(f"Experiment Complete: All results saved to {save_dir}.")


if __name__ == "__main__":
    main()
