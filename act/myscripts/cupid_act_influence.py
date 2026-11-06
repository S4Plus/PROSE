from __future__ import annotations

"""End-to-end CUPID-ACT influence computation utilities.

This module follows the four-stage pipeline outlined in
``act/cupid_act_influence.md`` and stitches together the existing ACT helpers
(``train_trak_act.py`` + ``act/scoring_utils.py``).  It supports:

1. Shadow training + gradient caching with pseudo Huber losses.
2. Rollout processing with reward logging and eval-time action noise injection.
3. LiSSA-based inverse-HVP solving using stochastic Hessian-vector products.
4. Influence scoring plus optional serialization for downstream curation.


 CUDA_VISIBLE_DEVICES=0 python /workspace/liushuo/cupid/act/myscripts/cupid_act_influence.py \
    --pretrained_dir /workspace/liushuo/cupid/3_model/can_mh/checkpoints/100000/pretrained_model \
    --train_dataset_path /workspace/liushuo/cupid/datasets/lerobot/can_mh \
    --eval_rollout_dir /workspace/liushuo/cupid/mimicgen/eval/tmp/can_mh_act_63_73/episodes \
    --output_path /workspace/liushuo/cupid/1_infulenece_score/can_mh/score_episode_window.pkl \
    --stage_cache_prefix /workspace/liushuo/cupid/1_infulenece_score/can_mh/score \
    --aggregate_mode episode \
    --window_size 50 \
    --window_stride 50


 CUDA_VISIBLE_DEVICES=0 nohup python /workspace/liushuo/cupid/act/myscripts/cupid_act_influence.py \
    --pretrained_dir /workspace/liushuo/cupid/3_model/square_mh/checkpoints/100000/pretrained_model \
    --train_dataset_path /workspace/liushuo/cupid/datasets/lerobot/square_mh \
    --eval_rollout_dir /workspace/liushuo/cupid/mimicgen/eval/tmp/square_mh/base_46_37.0/pkl \
    --output_path /workspace/liushuo/cupid/1_infulenece_score/square_mh/score_episode_window.pkl \
    --stage_cache_prefix /workspace/liushuo/cupid/1_infulenece_score/square_mh/score \
    --aggregate_mode episode \
    --window_size 50 \
    --window_stride 50\
    > logs/cupid_act_influence_square_mh.log 2>&1 &

python ./act/cupid_act_influence.py \
    --pretrained_dir /workspace/liushuo/cupid/outputs_new/random_40/retrain_random_1.0/checkpoints/200000/pretrained_model \
    --train_dataset_path /ext_workspace/liushuo/Static_Pick_Bottle_400 \
    --eval_rollout_dir /ext_workspace/liushuo/new_test/eval_belt_pick_bottle/episodes \
    --output_path ./new_test/score



python ./act/cupid_act_influence.py \
    --pretrained_dir /workspace/liushuo/cupid/act/swk/outputs/CB81/checkpoints/060000/pretrained_model \
    --train_dataset_path /ext_workspace/liushuo/swk/CB81 \
    --eval_rollout_dir /workspace/liushuo/cupid/act/swk/rollout/rollout_fail\
    --output_path ./act/swk/score


CUDA_VISIBLE_DEVICES=6 \
nohup python ./act/cupid_act_influence.py \
  --pretrained_dir /workspace/liushuo/cupid/act/swk/outputs/CB81/checkpoints/060000/pretrained_model \
  --train_dataset_path /ext_workspace/liushuo/swk/CB81 \
  --eval_rollout_dir /workspace/liushuo/cupid/act/swk/rollout/rollout_fail \
  --output_path ./act/swk/score \
> logs/cupid_act_influence_gpu6_$(date +%Y%m%d_%H%M%S).log 2>&1 &



nohup python ./act/cupid_act_influence.py \
    --pretrained_dir /workspace/liushuo/cupid/mimicgen/eval/outputs/square_d1/checkpoints/200000/pretrained_model \
    --train_dataset_path /workspace/liushuo/cupid/datasets/square_d1 \
    --eval_rollout_dir /workspace/liushuo/cupid/mimicgen/eval/act/eval_rollouts_act\
    --output_path ./act/mimicgen/score \
> logs/cupid_act_influence.log 2>&1 &




nohup python ./act/cupid_act_influence.py \
    --pretrained_dir /workspace/liushuo/cupid/outputs/hammer_cleanup_d0_core/checkpoints/100000/pretrained_model \
    --train_dataset_path /workspace/liushuo/cupid/datasets/hammer_cleanup_d0_core\
    --eval_rollout_dir /workspace/liushuo/cupid/mimicgen/eval/hammer_cleanup_D0/episodes\
    --output_path ./act/mimicgen/hammer_cleanup_D0/score \
> logs/cupid_act_influence.log 2>&1 &


CUDA_VISIBLE_DEVICES=7 \
nohup python cupid_act_influence.py \
    --pretrained_dir /workspace/liushuo/cupid/3_model/can_mh/checkpoints/100000/pretrained_model \
    --train_dataset_path /workspace/liushuo/cupid/datasets/lerobot/can_mh\
    --eval_rollout_dir /workspace/liushuo/cupid/mimicgen/eval/tmp/can_63_73/episodes\
    --output_path /workspace/liushuo/cupid/1_infulenece_score/can_mh/score \
    > /workspace/liushuo/cupid/3_model/logs/cupid_act_influence_can_mh.log 2>&1 &



CUDA_VISIBLE_DEVICES=3 nohup python /workspace/liushuo/cupid/act/myscripts/cupid_act_influence.py \
    --pretrained_dir /workspace/liushuo/cupid/outputs/stack_cube_mh_act/checkpoints/100000/pretrained_model \
    --train_dataset_path /workspace/liushuo/dataset/stack_cube_mh \
    --eval_rollout_dir /workspace/liushuo/eval \
    --output_path /workspace/liushuo/cupid/1_infulenece_score/stack_cube_mh/score_episode_window.pkl \
    --stage_cache_prefix /workspace/liushuo/cupid/1_infulenece_score/stack_cube_mh/score \
    --aggregate_mode episode \
    --window_size 50 \
    --window_stride 50\
    > logs/cupid_act_influence_stack_cube_mh.log 2>&1 &

"""

import argparse
import dataclasses
import logging
import math
import pathlib
import pickle
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
import sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from act.train_trak_act import (
    ACTModelOutput,
    ActTraKDataset,
    EvalPklEpisodeDataset,
    build_lerobot_dataset,
    load_act_policy,
    prepare_act_batch,
)


LOGGER = logging.getLogger("cupid_act_influence")
MAX_PROJECTION_BYTES = int(2 * 1024 ** 3)


# -----------------------------------------------------------------------------
# Configuration containers
# -----------------------------------------------------------------------------


@dataclass
class CupidActOutputPaths:
    """Resolved filesystem layout for influence artifacts."""

    result_path: pathlib.Path
    output_dir: pathlib.Path
    stage_dir: pathlib.Path
    stage_prefix: str


@dataclass
class CupidActInfluenceConfig:
    """User-facing configuration for the CUPID-ACT calculator."""

    pretrained_dir: str
    train_dataset_path: str
    eval_rollout_dir: str
    output_path: str
    holdout_dataset_path: Optional[str] = None
    device: str = "cuda:0"
    batch_size: int = 1
    hvp_batch_size: int = 4
    num_workers: int = 0
    proj_dim: int = 4096
    noise_sigma: float = 0.1
    lissa_depth: int = 8
    lissa_samples: int = 4
    lissa_scale: float = 10.0
    lissa_damping: float = 0.01
    seed: int = 0
    save_projection: bool = False
    stage_cache_prefix: Optional[str] = None
    reward_columns: Optional[str] = None
    reward_weights: Optional[str] = None
    reward_reduce: str = "sum"
    aggregate_mode: str = "episode"
    window_size: int = 0
    window_stride: int = 0
    window_weight_temperature: float = 1.0
    window_weight_clip_min: float = 0.2
    window_weight_clip_max: float = 5.0
    window_drop_quantile: float = 0.0
    window_drop_threshold: Optional[float] = None
    bootstrap_samples: int = 1
    bootstrap_seed: int = 0
    save_bootstrap_scores: bool = False
    save_cupid_quality_cache: bool = False
    cupid_quality_cache_path: Optional[str] = None


@dataclass
class CupidActInfluenceResult:
    """Serialized influence artifacts."""

    train_projections: np.ndarray
    reward_gradient: np.ndarray
    inverse_hvp: np.ndarray
    influence_scores: np.ndarray
    projection_matrix: Optional[np.ndarray]
    projection_state: Optional[Dict[str, np.ndarray]]
    metadata: Dict[str, np.ndarray]


# -----------------------------------------------------------------------------
# Helper utilities
# -----------------------------------------------------------------------------


def set_random_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def _parse_csv(value: Optional[str]) -> Optional[List[str]]:
    if value is None:
        return None
    items = [item.strip() for item in value.split(",") if item.strip()]
    return items or None


def _parse_csv_floats(value: Optional[str]) -> Optional[List[float]]:
    if value is None:
        return None
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        return None
    return [float(item) for item in items]


class ParameterVectorizer:
    """Flatten/unflatten helpers for model parameters."""

    def __init__(self, params: Iterable[torch.nn.Parameter]):
        self.params: List[torch.nn.Parameter] = [p for p in params if p.requires_grad]
        self.shapes = [p.shape for p in self.params]
        self.numels = [p.numel() for p in self.params]
        self.total_dim = int(sum(self.numels))

    @property
    def device(self) -> torch.device:
        return self.params[0].device if self.params else torch.device("cpu")

    def flatten(self, tensors: Sequence[torch.Tensor]) -> torch.Tensor:
        flats = []
        for tensor in tensors:
            flats.append(tensor.reshape(-1))
        return torch.cat(flats)

    def unflatten(self, vector: torch.Tensor) -> List[torch.Tensor]:
        assert vector.numel() == self.total_dim
        slices: List[torch.Tensor] = []
        offset = 0
        for numel, shape in zip(self.numels, self.shapes):
            slices.append(vector[offset : offset + numel].view(shape))
            offset += numel
        return slices

    def materialize_grads(self, grads: Sequence[Optional[torch.Tensor]]) -> List[torch.Tensor]:
        """Replace missing gradients with all-zero tensors matching each parameter."""
        filled: List[torch.Tensor] = []
        for param, grad in zip(self.params, grads):
            if grad is None:
                filled.append(torch.zeros_like(param))
            else:
                filled.append(grad)
        return filled


class CountSketchProjector:
    """Memory-efficient random projection implemented via CountSketch."""

    def __init__(self, input_dim: int, proj_dim: int, device: torch.device, seed: int):
        if proj_dim <= 0:
            raise ValueError("proj_dim must be positive for CountSketch projections.")
        self.input_dim = input_dim
        self.proj_dim = proj_dim
        self.device = device
        self.scale = 1.0 / math.sqrt(float(proj_dim))
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
        self.indices = torch.randint(
            low=0,
            high=proj_dim,
            size=(input_dim,),
            generator=generator,
            device=device,
            dtype=torch.int64,
        )
        self.signs = torch.randint(
            low=0,
            high=2,
            size=(input_dim,),
            generator=generator,
            device=device,
            dtype=torch.float32,
        )
        self.signs = self.signs * 2.0 - 1.0

    def project(self, vector: torch.Tensor) -> torch.Tensor:
        assert vector.numel() == self.input_dim
        projected = torch.zeros(self.proj_dim, device=vector.device, dtype=vector.dtype)
        projected.scatter_add_(0, self.indices, vector * self.signs)
        return projected * self.scale

    def state_dict(self) -> Dict[str, np.ndarray]:
        return {
            "indices": self.indices.detach().cpu().numpy(),
            "signs": self.signs.detach().cpu().numpy(),
            "scale": np.asarray(self.scale, dtype=np.float32),
            "proj_dim": np.asarray(self.proj_dim, dtype=np.int64),
        }


def _episode_rewards(
    dataset: EvalPklEpisodeDataset,
    reward_columns: Optional[Sequence[str]] = None,
    reward_weights: Optional[Sequence[float]] = None,
    reward_reduce: str = "sum",
) -> np.ndarray:
    """Derive per-episode returns from rollout PKLs."""

    def _bool_to_reward(value: object) -> Optional[float]:
        if value is None:
            return None
        if isinstance(value, (bool, np.bool_)):
            return 1.0 if bool(value) else -1.0
        scalar = np.asarray(value)
        if scalar.ndim == 0 and scalar.dtype.kind in {"b"}:
            return 1.0 if bool(scalar.item()) else -1.0
        return None

    def _lookup_nested(container: object, *path: str) -> object:
        cur = container
        for key in path:
            if not isinstance(cur, dict) or key not in cur:
                return None
            cur = cur[key]
        return cur

    def _reduce(values: object) -> float:
        arr = np.asarray(values, dtype=np.float64)
        finite = np.isfinite(arr)
        if not finite.any():
            return float("nan")
        arr = arr[finite]
        if arr.ndim == 0:
            return float(arr)
        if reward_reduce == "last":
            return float(arr[-1])
        if reward_reduce == "mean":
            return float(arr.mean())
        return float(arr.sum())

    def _pick_first_finite(*values: object) -> float:
        for value in values:
            bool_reward = _bool_to_reward(value)
            if bool_reward is not None:
                return bool_reward
            reduced = _reduce(value)
            if np.isfinite(reduced):
                return float(reduced)
            scalar = np.asarray(value, dtype=np.float64)
            if scalar.ndim == 0 and np.isfinite(scalar):
                return float(scalar)
        return 0.0

    rewards: List[float] = []
    for episode_idx, episode in enumerate(dataset._episodes):
        if reward_columns:
            total = 0.0
            cols = set(getattr(episode, "columns", []))
            last_row = episode.iloc[-1].to_dict() if len(episode) > 0 else {}
            for idx, col in enumerate(reward_columns):
                weight = reward_weights[idx] if reward_weights is not None else 1.0
                value = None
                if cols and col in cols:
                    value = _reduce(episode[col])
                elif col in last_row:
                    value = _reduce(last_row[col])
                if value is not None and np.isfinite(value):
                    total += weight * float(value)
            rewards.append(float(total))
            continue

        reward = 0.0
        last_row = episode.iloc[-1].to_dict() if len(episode) > 0 else {}
        if hasattr(episode, "columns") and "reward" in episode.columns:
            reward = _reduce(episode["reward"])
        if not np.isfinite(reward) and hasattr(episode, "columns") and "next.reward" in episode.columns:
            reward = _reduce(episode["next.reward"])
        if not np.isfinite(reward) and hasattr(episode, "columns") and "success" in episode.columns:
            reward = _pick_first_finite(episode["success"].iloc[-1])
        if not np.isfinite(reward) and hasattr(episode, "columns") and "next.success" in episode.columns:
            reward = _pick_first_finite(episode["next.success"].iloc[-1])
        if not np.isfinite(reward):
            reward = _pick_first_finite(
                last_row.get("next.reward"),
                last_row.get("reward"),
                last_row.get("success"),
                last_row.get("next.success"),
                _lookup_nested(last_row, "meta", "reward"),
                _lookup_nested(last_row, "meta", "episode_reward"),
                _lookup_nested(last_row, "meta", "success"),
                _lookup_nested(last_row, "meta", "eval_success"),
                _lookup_nested(last_row, "episode_info", "success"),
            )
        if not np.isfinite(reward) and hasattr(dataset, "episode_successes"):
            reward = _pick_first_finite(dataset.episode_successes[episode_idx])
        rewards.append(reward)
    return np.asarray(rewards, dtype=np.float64)


def _ensure_parent(path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# Influence calculator
# -----------------------------------------------------------------------------


class CupidActInfluenceCalculator:
    def __init__(self, cfg: CupidActInfluenceConfig):
        self.cfg = cfg
        set_random_seed(cfg.seed)
        self.device = torch.device(cfg.device)
        self.paths = self._resolve_output_paths(cfg.output_path, cfg.stage_cache_prefix)
        # Legacy base retained for backwards-compatible cache loading.
        self.legacy_stage_base = pathlib.Path(cfg.stage_cache_prefix or cfg.output_path).expanduser()
        LOGGER.info(
            "Output layout resolved: result=%s, stage_dir=%s (prefix=%s)",
            self.paths.result_path,
            self.paths.stage_dir,
            self.paths.stage_prefix,
        )
        self.policy, self.policy_cfg = load_act_policy(pathlib.Path(cfg.pretrained_dir), self.device)
        self.policy.eval()
        self.loss_model = ACTModelOutput()
        self.vectorizer = ParameterVectorizer(self.policy.parameters())
        self.reward_columns = _parse_csv(cfg.reward_columns)
        self.reward_weights = _parse_csv_floats(cfg.reward_weights)
        if self.reward_columns and self.reward_weights:
            if len(self.reward_columns) != len(self.reward_weights):
                raise ValueError("reward_weights must have the same length as reward_columns.")
        self.reward_reduce = cfg.reward_reduce
        self.aggregate_mode = cfg.aggregate_mode.lower().strip()
        if self.aggregate_mode not in {"episode", "window", "none"}:
            raise ValueError("aggregate_mode must be one of: episode, window, none.")
        self.window_size = max(int(cfg.window_size), 0)
        self.window_stride = int(cfg.window_stride) if int(cfg.window_stride) > 0 else self.window_size
        self.window_weight_temperature = float(cfg.window_weight_temperature)
        self.window_weight_clip_min = float(cfg.window_weight_clip_min)
        self.window_weight_clip_max = float(cfg.window_weight_clip_max)
        self.window_drop_quantile = float(cfg.window_drop_quantile)
        self.window_drop_threshold = cfg.window_drop_threshold
        self.bootstrap_samples = max(int(cfg.bootstrap_samples), 1)
        self.bootstrap_seed = int(cfg.bootstrap_seed)
        self.projection: Optional[torch.Tensor] = None
        self.projector: Optional[CountSketchProjector] = None
        self.train_episode_indices: Optional[np.ndarray] = None
        if cfg.proj_dim > 0:
            bytes_required = int(self.vectorizer.total_dim) * int(cfg.proj_dim) * 4
            if bytes_required <= MAX_PROJECTION_BYTES:
                try:
                    self.projection = torch.randn(
                        self.vectorizer.total_dim, cfg.proj_dim, device=self.device
                    ) / math.sqrt(cfg.proj_dim)
                except RuntimeError as exc:
                    if "out of memory" not in str(exc).lower():
                        raise
                    LOGGER.warning(
                        "Dense projection allocation failed with OOM (%s). Falling back to CountSketch.", exc
                    )
                    if self.device.type == "cuda":
                        torch.cuda.empty_cache()
            if self.projection is None:
                LOGGER.info(
                    "Using CountSketch projection with proj_dim=%d (dense would require %.2f GiB).",
                    cfg.proj_dim,
                    bytes_required / (1024 ** 3),
                )
                self.projector = CountSketchProjector(
                    self.vectorizer.total_dim,
                    cfg.proj_dim,
                    self.device,
                    cfg.seed,
                )
        self.train_loader = self._build_train_loader(shuffle=False, batch_size=cfg.batch_size)
        self.train_episode_indices = self._extract_episode_indices(self.train_loader.dataset)
        if self.train_episode_indices is None:
            LOGGER.warning(
                "Could not infer train episode indices; influence scores will remain per-sample instead of per-episode."
            )
        else:
            LOGGER.info(
                "Aggregating train influence scores over %d episodes.",
                int(self.train_episode_indices.max()) + 1,
            )
        hvp_loader = self._build_train_loader(shuffle=True, batch_size=max(1, cfg.hvp_batch_size))
        self.hvp_loader_iter = self._cycle_loader(hvp_loader)
        self.eval_dataset = EvalPklEpisodeDataset(
            dataset_path=pathlib.Path(cfg.eval_rollout_dir),
            exec_horizon=1,
            sample_history=0,
        )
        self.eval_loader = DataLoader(self.eval_dataset, batch_size=1, shuffle=False, num_workers=0)
        eval_rewards = _episode_rewards(
            self.eval_dataset,
            reward_columns=self.reward_columns,
            reward_weights=self.reward_weights,
            reward_reduce=self.reward_reduce,
        )
        if eval_rewards.size > 0 and np.allclose(eval_rewards, eval_rewards[0]):
            LOGGER.warning(
                "Eval rewards are constant (value=%s, count=%d). Stage 2 return gradient may collapse to zeros. "
                "Check reward extraction compatibility with rollout PKLs.",
                float(eval_rewards[0]),
                eval_rewards.size,
            )
        self.eval_rewards = torch.tensor(eval_rewards, device=self.device, dtype=torch.float32)

    # ------------------------------------------------------------------
    # Data loaders
    # ------------------------------------------------------------------

    def _build_train_loader(self, shuffle: bool, batch_size: int) -> DataLoader:
        base_dataset = build_lerobot_dataset(pathlib.Path(self.cfg.train_dataset_path), self.policy_cfg)
        image_keys = self.policy_cfg.image_features.keys() if self.policy_cfg.image_features else []
        wrapped = ActTraKDataset(base_dataset, image_keys)
        return DataLoader(
            wrapped,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            persistent_workers=False,
        )

    def _extract_episode_indices(self, dataset: ActTraKDataset) -> Optional[np.ndarray]:
        base = getattr(dataset, "dataset", None)
        if base is None or not hasattr(base, "hf_dataset"):
            return None
        hf_dataset = base.hf_dataset
        if hf_dataset is None:
            return None
        try:
            column = hf_dataset["episode_index"]
        except KeyError:
            return None
        if len(column) == 0:
            return None
        first = column[0]
        if torch.is_tensor(first):
            stacked = torch.stack(column).view(-1).cpu().numpy()
            return stacked.astype(np.int64, copy=False)
        try:
            return np.asarray(column, dtype=np.int64).reshape(-1)
        except Exception:
            return None

    def _cycle_loader(self, loader: DataLoader) -> Iterator[Dict[str, torch.Tensor]]:
        while True:
            for batch in loader:
                yield batch

    # ------------------------------------------------------------------
    # Core routines
    # ------------------------------------------------------------------

    def run(self) -> CupidActInfluenceResult:
        LOGGER.info("Stage 1: caching ACT gradients with pseudo Huber loss.")
        projected_train = self._get_or_cache_train_gradients()
        LOGGER.info("Stage 2: collecting rollout gradients + returns.")
        return_grad = self._get_or_compute_return_gradient()
        LOGGER.info("Stage 3: solving LiSSA inverse-HVP.")
        influence_scores: np.ndarray
        score_metadata: Dict[str, np.ndarray] = {}
        if self.bootstrap_samples > 1:
            if return_grad.ndim != 2:
                raise RuntimeError("Expected bootstrap return gradients to have shape (B, D).")
            scores_list: List[np.ndarray] = []
            inv_hvp_ref: Optional[torch.Tensor] = None
            base_metadata: Dict[str, np.ndarray] = {}
            for b in range(self.bootstrap_samples):
                inv_hvp = self._get_or_compute_inverse_hvp(return_grad[b], cache_suffix=f"boot{b}")
                scores, meta = self._score_influences(projected_train, inv_hvp)
                scores_list.append(scores)
                if inv_hvp_ref is None:
                    inv_hvp_ref = inv_hvp
                    base_metadata = meta
            stacked = np.stack(scores_list, axis=0)
            influence_scores = stacked.mean(axis=0)
            score_metadata = base_metadata
            score_metadata["influence_scores_std"] = stacked.std(axis=0)
            score_metadata["bootstrap_samples"] = np.asarray(self.bootstrap_samples, dtype=np.int64)
            if self.cfg.save_bootstrap_scores:
                score_metadata["bootstrap_influence_scores"] = stacked
            inv_hvp = inv_hvp_ref if inv_hvp_ref is not None else self._get_or_compute_inverse_hvp(return_grad[0])
            return_grad = return_grad.mean(dim=0)
        else:
            inv_hvp = self._get_or_compute_inverse_hvp(return_grad)
            LOGGER.info("Stage 4: scoring train samples via gradient dot-products.")
            influence_scores, score_metadata = self._score_influences(projected_train, inv_hvp)
        projection_state = None
        if self.projector is not None and self.cfg.save_projection:
            projection_state = self.projector.state_dict()
        result = CupidActInfluenceResult(
            train_projections=projected_train,
            reward_gradient=return_grad.cpu().numpy(),
            inverse_hvp=inv_hvp.cpu().numpy(),
            influence_scores=influence_scores,
            projection_matrix=self.projection.detach().cpu().numpy() if self.projection is not None else None,
            projection_state=projection_state,
            metadata={
                "episode_rewards": self.eval_rewards.cpu().numpy(),
                "episode_lengths": np.asarray(self.eval_dataset.episode_lengths, dtype=np.int64),
                **score_metadata,
            },
        )
        self._save_result(result)
        if self.cfg.save_cupid_quality_cache:
            cache_path = self._resolve_cupid_quality_cache_path()
            self._save_cupid_quality_cache(projected_train, cache_path)
        return result

    def _resolve_cupid_quality_cache_path(self) -> pathlib.Path:
        if self.cfg.cupid_quality_cache_path:
            return pathlib.Path(self.cfg.cupid_quality_cache_path).expanduser()
        result_path = self.paths.result_path
        return result_path.with_suffix(".cupid_quality_episode_pairs.npz")

    def _cache_train_gradients(self) -> np.ndarray:
        cached: List[np.ndarray] = []
        total = len(self.train_loader.dataset) if hasattr(self.train_loader, "dataset") else None
        processed = 0
        next_log = None
        if total:
            interval = max(total // 20, 1)
            next_log = interval
        for batch in self.train_loader:
            formatted = self._format_batch(batch, apply_noise=False)
            grads = self._batch_gradients(formatted)
            for grad in grads:
                cached.append(self._project_gradient(grad))
            processed += len(grads)
            if total and next_log is not None and processed >= next_log:
                LOGGER.info("Stage 1 progress: cached %d/%d train samples.", min(processed, total), total)
                next_log += interval
        if total and processed < total:
            LOGGER.info("Stage 1 progress: cached %d/%d train samples.", processed, total)
        return np.stack(cached, axis=0)

    def _get_or_cache_train_gradients(self) -> np.ndarray:
        cache_path = self._stage_path("stage1_train_projections", ".npy")
        if cache_path.exists():
            LOGGER.info("Stage 1 cache hit: loading %s", cache_path)
            with cache_path.open("rb") as f:
                cached = np.load(f, allow_pickle=False)
            LOGGER.info("Stage 1 cache load success: %s samples projected.", cached.shape[0])
            return cached
        LOGGER.info("Stage 1 cache miss: %s not found, recomputing.", cache_path)
        projected_train = self._cache_train_gradients()
        self._save_numpy(cache_path, projected_train)
        LOGGER.info("Stage 1 cache saved to %s (shape=%s).", cache_path, projected_train.shape)
        return projected_train

    def _compute_return_gradient_single(self, rewards: torch.Tensor) -> torch.Tensor:
        lengths = self.eval_dataset.episode_lengths
        if len(lengths) == 0:
            return torch.zeros(self.vectorizer.total_dim, device=self.device, dtype=torch.float32)

        centered_rewards = rewards - rewards.mean()
        scale = centered_rewards.shape[0]
        return_grad = torch.zeros(self.vectorizer.total_dim, device=self.device, dtype=torch.float32)
        episode_grad = torch.zeros_like(return_grad)
        ep_idx = 0
        steps_remaining = lengths[ep_idx]
        for grad in self._iter_eval_gradients():
            episode_grad += grad
            steps_remaining -= 1
            if steps_remaining == 0:
                return_grad += centered_rewards[ep_idx] * episode_grad
                ep_idx += 1
                if ep_idx >= scale:
                    break
                episode_grad.zero_()
                steps_remaining = lengths[ep_idx]
        return return_grad / max(scale, 1)

    def _compute_return_gradients(self) -> torch.Tensor:
        if self.bootstrap_samples <= 1:
            return self._compute_return_gradient_single(self.eval_rewards)

        lengths = self.eval_dataset.episode_lengths
        if len(lengths) == 0:
            return torch.zeros(
                (self.bootstrap_samples, self.vectorizer.total_dim),
                device=self.device,
                dtype=torch.float32,
            )

        rewards = self.eval_rewards.detach().cpu().numpy()
        num_episodes = rewards.shape[0]
        rng = np.random.default_rng(self.bootstrap_seed or self.cfg.seed)
        counts = np.zeros((self.bootstrap_samples, num_episodes), dtype=np.int64)
        for b in range(self.bootstrap_samples):
            indices = rng.integers(0, num_episodes, size=num_episodes)
            counts[b] = np.bincount(indices, minlength=num_episodes)
        counts_sum = np.maximum(counts.sum(axis=1, keepdims=True), 1.0)
        mean_rewards = (counts * rewards[None, :]).sum(axis=1, keepdims=True) / counts_sum

        return_grads = torch.zeros(
            (self.bootstrap_samples, self.vectorizer.total_dim),
            device=self.device,
            dtype=torch.float32,
        )
        episode_grad = torch.zeros(self.vectorizer.total_dim, device=self.device, dtype=torch.float32)
        ep_idx = 0
        steps_remaining = lengths[ep_idx]
        for grad in self._iter_eval_gradients():
            episode_grad += grad
            steps_remaining -= 1
            if steps_remaining == 0:
                coeff = counts[:, ep_idx] * (rewards[ep_idx] - mean_rewards[:, 0])
                coeff = torch.as_tensor(coeff, device=self.device, dtype=torch.float32)
                if coeff.any():
                    return_grads += coeff[:, None] * episode_grad[None, :]
                ep_idx += 1
                if ep_idx >= num_episodes:
                    break
                episode_grad.zero_()
                steps_remaining = lengths[ep_idx]
        return return_grads / float(max(num_episodes, 1))

    def _get_or_compute_return_gradient(self) -> torch.Tensor:
        cache_path = self._stage_path("stage2_return_gradient", ".pt")
        if self.bootstrap_samples > 1:
            cache_path = self._stage_path("stage2_return_gradients", ".pt")
        if cache_path.exists():
            LOGGER.info("Stage 2 cache hit: loading %s", cache_path)
            tensor = torch.load(cache_path, map_location=self.device)
            LOGGER.info("Stage 2 cache load success: tensor shape %s.", tuple(tensor.shape))
            return tensor
        LOGGER.info("Stage 2 cache miss: %s not found, recomputing.", cache_path)
        return_grad = self._compute_return_gradients()
        self._save_tensor(cache_path, return_grad)
        LOGGER.info("Stage 2 cache saved to %s (shape=%s).", cache_path, tuple(return_grad.shape))
        return return_grad

    def _solve_inverse_hvp(self, vector: torch.Tensor) -> torch.Tensor:
        assert vector.shape[0] == self.vectorizer.total_dim
        estimate = torch.zeros_like(vector)
        for sample_idx in range(self.cfg.lissa_samples):
            cur = vector.clone()
            for _ in range(self.cfg.lissa_depth):
                hvp = self._stochastic_hvp(cur)
                cur = vector + (1.0 - self.cfg.lissa_damping) * cur - hvp / self.cfg.lissa_scale
            estimate += cur
            LOGGER.debug("LiSSA sample %d/%d complete.", sample_idx + 1, self.cfg.lissa_samples)
        return estimate / float(self.cfg.lissa_samples)

    def _get_or_compute_inverse_hvp(self, vector: torch.Tensor, cache_suffix: Optional[str] = None) -> torch.Tensor:
        stem = "stage3_inverse_hvp" if cache_suffix is None else f"stage3_inverse_hvp_{cache_suffix}"
        cache_path = self._stage_path(stem, ".pt")
        if cache_path.exists():
            LOGGER.info("Stage 3 cache hit: loading %s", cache_path)
            tensor = torch.load(cache_path, map_location=self.device)
            LOGGER.info("Stage 3 cache load success: tensor shape %s.", tuple(tensor.shape))
            return tensor
        LOGGER.info("Stage 3 cache miss: %s not found, recomputing.", cache_path)
        inv_hvp = self._solve_inverse_hvp(vector)
        self._save_tensor(cache_path, inv_hvp)
        LOGGER.info("Stage 3 cache saved to %s (shape=%s).", cache_path, tuple(inv_hvp.shape))
        return inv_hvp

    def _score_influences(
        self, projected_train: np.ndarray, inv_hvp: torch.Tensor
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        train_tensor = torch.from_numpy(projected_train).to(self.device)
        if self.projection is not None:
            inv_proj = torch.matmul(inv_hvp, self.projection)
        elif self.projector is not None:
            inv_proj = self.projector.project(inv_hvp)
        else:
            inv_proj = inv_hvp
        scores = -torch.matmul(train_tensor, inv_proj)
        sample_scores = scores.detach().cpu().numpy()
        metadata: Dict[str, np.ndarray] = {}
        episode_scores: Optional[np.ndarray] = None
        if self.train_episode_indices is not None and len(sample_scores) == len(self.train_episode_indices):
            episode_scores = self._aggregate_episode_scores(sample_scores)
            metadata["episode_influence_scores"] = episode_scores
        if self.window_size > 0 and self.train_episode_indices is not None:
            window_scores, window_meta = self._compute_window_scores(sample_scores, self.train_episode_indices)
            metadata.update(window_meta)
            window_weights, window_drop = self._compute_window_weights_and_drop(window_scores)
            metadata["window_weights"] = window_weights
            metadata["window_drop_mask"] = window_drop.astype(np.int64)
            if self.aggregate_mode == "window":
                return window_scores, metadata
        if self.aggregate_mode == "episode" and episode_scores is not None:
            return episode_scores, metadata
        if self.aggregate_mode == "none":
            return sample_scores, metadata
        if self.train_episode_indices is not None:
            LOGGER.warning(
                "Train episode indices (len=%d) do not match sample scores (len=%d); skipping aggregation.",
                len(self.train_episode_indices),
                len(sample_scores),
            )
        return sample_scores, metadata

    # ------------------------------------------------------------------
    # Gradient helpers
    # ------------------------------------------------------------------

    def _format_batch(self, batch: Dict[str, torch.Tensor], apply_noise: bool = False) -> Dict[str, torch.Tensor]:
        formatted = prepare_act_batch(batch, self.policy_cfg, self.device)
        if apply_noise and self.cfg.noise_sigma > 0:
            formatted = dict(formatted)
            formatted["action"] = formatted["action"] + torch.randn_like(formatted["action"]) * self.cfg.noise_sigma
        return formatted

    def _batch_gradients(self, batch: Dict[str, torch.Tensor]) -> List[torch.Tensor]:
        losses = self.loss_model.get_output(self.policy, None, None, batch).view(-1)
        grads: List[torch.Tensor] = []
        for idx, loss in enumerate(losses):
            retain_graph = idx + 1 < losses.shape[0]
            self.policy.zero_grad(set_to_none=True)
            grad_tensors = torch.autograd.grad(
                loss,
                self.vectorizer.params,
                retain_graph=retain_graph,
                allow_unused=True,
            )
            materialized = self.vectorizer.materialize_grads(grad_tensors)
            grads.append(self.vectorizer.flatten(materialized).detach())
        return grads

    def _iter_eval_gradients(self) -> Iterator[torch.Tensor]:
        for batch in self.eval_loader:
            formatted = self._format_batch(batch, apply_noise=True)
            for grad in self._batch_gradients(formatted):
                yield grad

    def _stochastic_hvp(self, vector: torch.Tensor) -> torch.Tensor:
        batch = next(self.hvp_loader_iter)
        formatted = self._format_batch(batch, apply_noise=False)
        losses = self.loss_model.get_output(self.policy, None, None, formatted)
        loss = losses.mean()
        self.policy.zero_grad(set_to_none=True)
        grad_tensors = torch.autograd.grad(
            loss,
            self.vectorizer.params,
            create_graph=True,
            allow_unused=True,
        )
        grad_tensors = self.vectorizer.materialize_grads(grad_tensors)
        vec_tensors = self.vectorizer.unflatten(vector)
        grad_dot = torch.zeros(1, device=self.device)
        for grad, vec in zip(grad_tensors, vec_tensors):
            grad_dot = grad_dot + (grad * vec).sum()
        hvp_tensors = torch.autograd.grad(
            grad_dot, self.vectorizer.params, retain_graph=False, allow_unused=True
        )
        hvp_tensors = self.vectorizer.materialize_grads(hvp_tensors)
        return self.vectorizer.flatten(hvp_tensors).detach()

    def _project_gradient(self, grad: torch.Tensor) -> np.ndarray:
        if self.projection is None and self.projector is None:
            return grad.cpu().numpy()
        if self.projection is not None:
            projected = torch.matmul(grad, self.projection)
        else:
            projected = self.projector.project(grad)
        return projected.detach().cpu().numpy()

    def _aggregate_episode_scores(self, sample_scores: np.ndarray) -> np.ndarray:
        episode_ids = self.train_episode_indices
        assert episode_ids is not None
        max_episode = int(episode_ids.max())
        minlength = max_episode + 1
        episode_sums = np.bincount(episode_ids, weights=sample_scores, minlength=minlength)
        episode_counts = np.bincount(episode_ids, minlength=minlength)
        safe_counts = np.maximum(episode_counts, 1)
        aggregated = episode_sums / safe_counts
        aggregated[episode_counts == 0] = 0.0
        return aggregated

    def _compute_window_scores(
        self, sample_scores: np.ndarray, episode_ids: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        window_size = int(self.window_size)
        stride = int(self.window_stride) if self.window_stride > 0 else window_size
        window_scores: List[float] = []
        window_episode: List[int] = []
        window_start: List[int] = []
        window_end: List[int] = []

        max_episode = int(episode_ids.max())
        for ep in range(max_episode + 1):
            mask = episode_ids == ep
            count = int(mask.sum())
            if count == 0:
                continue
            ep_scores = sample_scores[mask]
            if count < window_size:
                continue
            for s in range(0, count - window_size + 1, stride):
                e = s + window_size
                window_scores.append(float(np.mean(ep_scores[s:e])))
                window_episode.append(ep)
                window_start.append(s)
                window_end.append(e)

        metadata = {
            "window_influence_scores": np.asarray(window_scores, dtype=np.float64),
            "window_episode_index": np.asarray(window_episode, dtype=np.int64),
            "window_start": np.asarray(window_start, dtype=np.int64),
            "window_end": np.asarray(window_end, dtype=np.int64),
            "window_size": np.asarray(window_size, dtype=np.int64),
            "window_stride": np.asarray(stride, dtype=np.int64),
        }
        return metadata["window_influence_scores"], metadata

    def _compute_window_weights_and_drop(
        self, window_scores: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        if window_scores.size == 0:
            return window_scores, np.zeros_like(window_scores, dtype=bool)

        mean = float(np.nanmean(window_scores))
        std = float(np.nanstd(window_scores)) + 1e-8
        scaled = (window_scores - mean) / std
        temperature = max(self.window_weight_temperature, 1e-6)
        weights = np.exp(scaled / temperature)

        clip_min = float(self.window_weight_clip_min)
        clip_max = float(self.window_weight_clip_max)
        if clip_min > 0 or clip_max > 0:
            weights = np.clip(weights, clip_min, clip_max)
        if np.all(np.isfinite(weights)) and weights.mean() > 0:
            weights = weights / weights.mean()

        drop_mask = np.zeros_like(window_scores, dtype=bool)
        if self.window_drop_quantile > 0:
            q = float(np.clip(self.window_drop_quantile, 0.0, 1.0))
            threshold = float(np.nanquantile(window_scores, q))
            drop_mask |= window_scores <= threshold
        if self.window_drop_threshold is not None:
            drop_mask |= window_scores <= float(self.window_drop_threshold)
        return weights.astype(np.float64), drop_mask

    def _stage_path(self, stem: str, extension: str) -> pathlib.Path:
        candidate = self.paths.stage_dir / f"{self.paths.stage_prefix}.{stem}{extension}"
        legacy = self._legacy_stage_path(stem, extension)
        if legacy.exists() and not candidate.exists():
            LOGGER.info("Stage cache fallback to legacy path %s", legacy)
            return legacy
        return candidate

    def _legacy_stage_path(self, stem: str, extension: str) -> pathlib.Path:
        """Backward compatible stage cache path using the previous naming scheme."""

        base = self.legacy_stage_base
        if base.is_dir():
            directory = base
            filename = pathlib.Path(self.cfg.output_path).expanduser().name
            filename_base = pathlib.Path(filename).stem if pathlib.Path(filename).suffix else filename
        else:
            directory = base.parent if str(base.parent) != "" else pathlib.Path(".")
            filename_base = base.name
        return directory / f"{filename_base}.{stem}{extension}"

    def _save_numpy(self, path: pathlib.Path, array: np.ndarray) -> None:
        _ensure_parent(path)
        with path.open("wb") as f:
            np.save(f, array, allow_pickle=False)

    def _save_tensor(self, path: pathlib.Path, tensor: torch.Tensor) -> None:
        _ensure_parent(path)
        torch.save(tensor.detach().cpu(), path)

    def _project_inverse_hvp(self, inv_hvp: torch.Tensor) -> torch.Tensor:
        if self.projection is not None:
            return torch.matmul(inv_hvp, self.projection)
        if self.projector is not None:
            return self.projector.project(inv_hvp)
        return inv_hvp

    def _build_train_episode_layout(
        self, projected_train: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.train_episode_indices is None:
            raise RuntimeError("CUPID-QUALITY cache requires train episode indices.")
        if len(projected_train) != len(self.train_episode_indices):
            raise RuntimeError(
                "Projected train gradients and train episode indices have different lengths: "
                f"{len(projected_train)} vs {len(self.train_episode_indices)}"
            )
        order = np.argsort(self.train_episode_indices, kind="stable")
        sorted_episode_ids = np.asarray(self.train_episode_indices[order], dtype=np.int64)
        unique_eps, counts = np.unique(sorted_episode_ids, return_counts=True)
        starts = np.concatenate(
            [np.asarray([0], dtype=np.int64), np.cumsum(counts[:-1], dtype=np.int64)]
        )
        return order.astype(np.int64), unique_eps.astype(np.int64), starts.astype(np.int64)

    def _compute_cupid_quality_episode_pair_contrib(
        self, projected_train: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        order, train_episode_ids, starts = self._build_train_episode_layout(projected_train)
        sorted_train_proj = np.asarray(projected_train[order], dtype=np.float32)
        train_tensor = torch.from_numpy(sorted_train_proj).to(self.device)

        episode_lengths = np.asarray(self.eval_dataset.episode_lengths, dtype=np.int64)
        num_eval_eps = int(episode_lengths.shape[0])
        num_train_eps = int(train_episode_ids.shape[0])
        contrib = np.full((num_eval_eps, num_train_eps), np.nan, dtype=np.float32)
        if num_eval_eps == 0 or num_train_eps == 0:
            return train_episode_ids, episode_lengths, contrib

        tau_best_of_worst = np.full(num_train_eps, -np.inf, dtype=np.float64)
        tau_worst_of_best = np.full(num_train_eps, np.inf, dtype=np.float64)

        current_episode = 0
        remaining_in_episode = int(episode_lengths[0])
        for grad in self._iter_eval_gradients():
            inv_hvp = self._solve_inverse_hvp(grad)
            inv_proj = self._project_inverse_hvp(inv_hvp)
            row_scores = -torch.matmul(train_tensor, inv_proj).detach().cpu().numpy()

            step_mins = np.minimum.reduceat(row_scores, starts)
            step_maxs = np.maximum.reduceat(row_scores, starts)
            tau_best_of_worst = np.maximum(tau_best_of_worst, step_mins)
            tau_worst_of_best = np.minimum(tau_worst_of_best, step_maxs)

            remaining_in_episode -= 1
            if remaining_in_episode > 0:
                continue

            contrib[current_episode] = (tau_best_of_worst - tau_worst_of_best).astype(np.float32)
            current_episode += 1
            if current_episode >= num_eval_eps:
                break
            tau_best_of_worst.fill(-np.inf)
            tau_worst_of_best.fill(np.inf)
            remaining_in_episode = int(episode_lengths[current_episode])
            if current_episode % 5 == 0:
                LOGGER.info(
                    "CUPID-QUALITY cache progress: processed %d/%d eval episodes.",
                    current_episode,
                    num_eval_eps,
                )
        return train_episode_ids, episode_lengths, contrib

    def _save_cupid_quality_cache(
        self, projected_train: np.ndarray, cache_path: pathlib.Path
    ) -> None:
        if cache_path.exists():
            LOGGER.info("CUPID-QUALITY cache already exists, skip recompute: %s", cache_path)
            return
        LOGGER.info("Saving CUPID-QUALITY sidecar cache to %s", cache_path)
        train_episode_ids, eval_episode_lengths, episode_pair_contrib = (
            self._compute_cupid_quality_episode_pair_contrib(projected_train)
        )
        eval_episode_ids = np.arange(int(eval_episode_lengths.shape[0]), dtype=np.int64)
        _ensure_parent(cache_path)
        np.savez_compressed(
            cache_path,
            format_version=np.asarray("episode_pair_contrib_v1"),
            source_score_pkl=np.asarray(str(self.paths.result_path)),
            train_episode_ids=train_episode_ids.astype(np.int64),
            eval_episode_ids=eval_episode_ids,
            eval_episode_lengths=eval_episode_lengths.astype(np.int64),
            episode_pair_contrib=episode_pair_contrib.astype(np.float32),
        )
        LOGGER.info(
            "Saved CUPID-QUALITY sidecar cache to %s (shape=%s).",
            cache_path,
            episode_pair_contrib.shape,
        )

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def _save_result(self, result: CupidActInfluenceResult) -> None:
        payload = {
            "config": dataclasses.asdict(self.cfg),
            "influence_scores": result.influence_scores,
            "train_projections": result.train_projections,
            "return_gradient": result.reward_gradient,
            "inverse_hvp": result.inverse_hvp,
            "metadata": result.metadata,
        }
        payload["train"] = {
            "mean_of_mean-succ-all": result.influence_scores,
            "influence_scores": result.influence_scores,
        }
        if result.projection_matrix is not None and self.cfg.save_projection:
            payload["projection_matrix"] = result.projection_matrix
        if result.projection_state is not None and self.cfg.save_projection:
            payload["projection_state"] = result.projection_state
        output_path = self.paths.result_path
        _ensure_parent(output_path)
        with output_path.open("wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        LOGGER.info("Saved CUPID-ACT influence result to %s", output_path)

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------

    def _resolve_output_paths(
        self, output_path: str, stage_cache_prefix: Optional[str]
    ) -> CupidActOutputPaths:
        """Normalize result/stage paths to a predictable directory layout.

        - `output_path` can be a file (with or without suffix) or a directory.
        - If a directory is provided, the result is written as <dir>/<dir>.pkl.
        - Stage caches default to <output_dir>/<basename>.* unless
          `stage_cache_prefix` is supplied.
        """

        output_raw = pathlib.Path(output_path).expanduser()
        trailing_slash = str(output_path).endswith("/")

        if output_raw.is_dir() or trailing_slash:
            output_dir = output_raw
            base_name = output_raw.name or "score"
            result_path = output_dir / f"{base_name}.pkl"
        elif output_raw.suffix:
            output_dir = output_raw.parent if str(output_raw.parent) != "" else pathlib.Path(".")
            base_name = output_raw.stem
            result_path = output_raw
        else:
            output_dir = output_raw.parent if str(output_raw.parent) != "" else pathlib.Path(".")
            base_name = output_raw.name or "score"
            result_path = output_raw.with_suffix(".pkl")

        if stage_cache_prefix is None:
            stage_dir = output_dir
            stage_prefix = base_name
        else:
            stage_base = pathlib.Path(stage_cache_prefix).expanduser()
            stage_is_dir = stage_base.suffix == "" or stage_base.is_dir() or str(stage_cache_prefix).endswith("/")
            if stage_is_dir:
                stage_dir = stage_base
                stage_prefix = base_name
            else:
                stage_dir = stage_base.parent if str(stage_base.parent) != "" else pathlib.Path(".")
                stage_prefix = stage_base.stem

        return CupidActOutputPaths(
            result_path=result_path,
            output_dir=output_dir,
            stage_dir=stage_dir,
            stage_prefix=stage_prefix,
        )


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CUPID-ACT influence calculator.")
    parser.add_argument("--pretrained_dir", type=str, required=True)
    parser.add_argument("--train_dataset_path", type=str, required=True)
    parser.add_argument("--eval_rollout_dir", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--hvp_batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--proj_dim", type=int, default=4096)
    parser.add_argument("--noise_sigma", type=float, default=0.1)
    parser.add_argument("--lissa_depth", type=int, default=8)
    parser.add_argument("--lissa_samples", type=int, default=4)
    parser.add_argument("--lissa_scale", type=float, default=10.0)
    parser.add_argument("--lissa_damping", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save_projection", action="store_true")
    parser.add_argument(
        "--reward_columns",
        type=str,
        default=None,
        help="Comma-separated reward/success columns for hierarchical returns.",
    )
    parser.add_argument(
        "--reward_weights",
        type=str,
        default=None,
        help="Comma-separated weights aligned with reward_columns.",
    )
    parser.add_argument(
        "--reward_reduce",
        type=str,
        default="sum",
        choices=("sum", "mean", "last"),
        help="Reduction over per-step reward columns.",
    )
    parser.add_argument(
        "--aggregate_mode",
        type=str,
        default="episode",
        help="Aggregation mode for influence scores: episode, window, none.",
    )
    parser.add_argument("--window_size", type=int, default=0, help="Window size for segment-level scores.")
    parser.add_argument(
        "--window_stride",
        type=int,
        default=0,
        help="Stride for window scores (default: window_size).",
    )
    parser.add_argument(
        "--window_weight_temperature",
        type=float,
        default=1.0,
        help="Temperature for window score reweighting.",
    )
    parser.add_argument(
        "--window_weight_clip_min",
        type=float,
        default=0.2,
        help="Minimum clip value for window weights.",
    )
    parser.add_argument(
        "--window_weight_clip_max",
        type=float,
        default=5.0,
        help="Maximum clip value for window weights.",
    )
    parser.add_argument(
        "--window_drop_quantile",
        type=float,
        default=0.0,
        help="Quantile of lowest window scores to drop (0 disables).",
    )
    parser.add_argument(
        "--window_drop_threshold",
        type=float,
        default=None,
        help="Drop windows with scores <= this threshold.",
    )
    parser.add_argument(
        "--bootstrap_samples",
        type=int,
        default=1,
        help="Number of bootstrap influence estimates for uncertainty.",
    )
    parser.add_argument("--bootstrap_seed", type=int, default=0)
    parser.add_argument("--save_bootstrap_scores", action="store_true")
    parser.add_argument(
        "--save-cupid-quality-cache",
        dest="save_cupid_quality_cache",
        action="store_true",
        default=False,
        help="保存供 cupid_quality 直接复用的 episode-pair 贡献 sidecar。默认关闭。",
    )
    parser.add_argument(
        "--no-save-cupid-quality-cache",
        dest="save_cupid_quality_cache",
        action="store_false",
        help="关闭 cupid_quality sidecar 缓存保存。",
    )
    parser.add_argument(
        "--cupid-quality-cache-path",
        type=str,
        default=None,
        help="可选：显式指定 cupid_quality sidecar 路径；默认写到 score.pkl 同目录同名前缀。",
    )
    parser.add_argument(
        "--stage_cache_prefix",
        type=str,
        default=None,
        help="Optional path prefix for stage cache artifacts; defaults to the output directory.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    cfg = CupidActInfluenceConfig(
        pretrained_dir=args.pretrained_dir,
        train_dataset_path=args.train_dataset_path,
        eval_rollout_dir=args.eval_rollout_dir,
        output_path=args.output_path,
        device=args.device,
        batch_size=args.batch_size,
        hvp_batch_size=args.hvp_batch_size,
        num_workers=args.num_workers,
        proj_dim=args.proj_dim,
        noise_sigma=args.noise_sigma,
        lissa_depth=args.lissa_depth,
        lissa_samples=args.lissa_samples,
        lissa_scale=args.lissa_scale,
        lissa_damping=args.lissa_damping,
        seed=args.seed,
        save_projection=args.save_projection,
        stage_cache_prefix=args.stage_cache_prefix,
        reward_columns=args.reward_columns,
        reward_weights=args.reward_weights,
        reward_reduce=args.reward_reduce,
        aggregate_mode=args.aggregate_mode,
        window_size=args.window_size,
        window_stride=args.window_stride,
        window_weight_temperature=args.window_weight_temperature,
        window_weight_clip_min=args.window_weight_clip_min,
        window_weight_clip_max=args.window_weight_clip_max,
        window_drop_quantile=args.window_drop_quantile,
        window_drop_threshold=args.window_drop_threshold,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
        save_bootstrap_scores=args.save_bootstrap_scores,
        save_cupid_quality_cache=args.save_cupid_quality_cache,
        cupid_quality_cache_path=args.cupid_quality_cache_path,
    )
    calculator = CupidActInfluenceCalculator(cfg)
    calculator.run()


if __name__ == "__main__":
    main()
