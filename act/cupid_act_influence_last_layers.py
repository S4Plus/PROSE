from __future__ import annotations

"""CUPID-ACT influence with parameter-subset selection.

This script keeps the original influence-function pipeline unchanged, but
restricts gradient/HVP computation to a parameter subset (e.g., the last N
layers) to reduce compute and memory.

Examples (nohup + multi-GPU):

GPU 1 (last 4 layers):
CUDA_VISIBLE_DEVICES=1 nohup python /workspace/liushuo/cupid/act/cupid_act_influence_last_layers.py \
  --pretrained_dir /workspace/liushuo/cupid/3_model/can_mh/checkpoints/100000/pretrained_model \
  --train_dataset_path /workspace/liushuo/cupid/datasets/lerobot/can_mh \
  --eval_rollout_dir /workspace/liushuo/cupid/mimicgen/eval/tmp/can_1_63_73/episodes \
  --output_path /workspace/liushuo/cupid/1_infulenece_score/can_mh/score_last4 \
  --param-scope last_n_layers \
  --last-n-layers 4 \
  > /workspace/liushuo/cupid/3_model/logs/cupid_act_influence_last4_gpu1.log 2>&1 &

GPU 2 (last 8 layers):
CUDA_VISIBLE_DEVICES=2 nohup python /workspace/liushuo/cupid/act/cupid_act_influence_last_layers.py \
  --pretrained_dir /workspace/liushuo/cupid/3_model/can_mh/checkpoints/100000/pretrained_model \
  --train_dataset_path /workspace/liushuo/cupid/datasets/lerobot/can_mh \
  --eval_rollout_dir /workspace/liushuo/cupid/mimicgen/eval/tmp/can_1_63_73/episodes \
  --output_path /workspace/liushuo/cupid/1_infulenece_score/can_mh/score_last8 \
  --param-scope last_n_layers \
  --last-n-layers 8 \
  > /workspace/liushuo/cupid/3_model/logs/cupid_act_influence_last8_gpu2.log 2>&1 &

GPU 3 (last 12 layers):
CUDA_VISIBLE_DEVICES=3 nohup python /workspace/liushuo/cupid/act/cupid_act_influence_last_layers.py \
  --pretrained_dir /workspace/liushuo/cupid/3_model/can_mh/checkpoints/100000/pretrained_model \
  --train_dataset_path /workspace/liushuo/cupid/datasets/lerobot/can_mh \
  --eval_rollout_dir /workspace/liushuo/cupid/mimicgen/eval/tmp/can_1_63_73/episodes \
  --output_path /workspace/liushuo/cupid/1_infulenece_score/can_mh/score_last12 \
  --param-scope last_n_layers \
  --last-n-layers 12 \
  > /workspace/liushuo/cupid/3_model/logs/cupid_act_influence_last12_gpu3.log 2>&1 &
"""

import argparse
import logging
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch

from cupid_act_influence import (
    MAX_PROJECTION_BYTES,
    CountSketchProjector,
    CupidActInfluenceCalculator,
    CupidActInfluenceConfig,
    CupidActInfluenceResult,
    LOGGER,
    ParameterVectorizer,
    _parse_csv,
)


@dataclass
class CupidActLastLayerConfig(CupidActInfluenceConfig):
    """Extension of the base config with parameter-scope controls."""

    param_scope: str = "last_n_layers"
    last_n_layers: int = 8
    param_keys: Optional[str] = None
    auto_stage_suffix: bool = True
    print_selected_params: bool = False


class CupidActLastLayerInfluenceCalculator(CupidActInfluenceCalculator):
    """Influence calculator that supports restricting parameters for gradients/HVP."""

    def __init__(self, cfg: CupidActLastLayerConfig):
        self.last_cfg = cfg
        if cfg.auto_stage_suffix and cfg.stage_cache_prefix is None:
            cfg.stage_cache_prefix = self._derive_stage_prefix(cfg)

        super().__init__(cfg)

        self._full_param_count, self._full_param_dim = self._count_trainable_params()
        selected_param_names, selected_layers = self._select_parameter_names(cfg)
        selected_set = set(selected_param_names)
        selected_params = [
            p for n, p in self.policy.named_parameters() if p.requires_grad and n in selected_set
        ]
        if not selected_params:
            raise ValueError("No parameters selected for influence computation.")

        self.selected_param_names = selected_param_names
        self.selected_layer_names = selected_layers
        self.vectorizer = ParameterVectorizer(selected_params)
        self._rebuild_projection()

        if cfg.print_selected_params:
            for name in self.selected_param_names:
                LOGGER.info("[selected-param] %s", name)

        ratio = float(self.vectorizer.total_dim) / max(float(self._full_param_dim), 1.0)
        LOGGER.info(
            "Parameter scope=%s, selected %d params in %d layers, dim=%d (full=%d, ratio=%.4f).",
            cfg.param_scope,
            len(self.selected_param_names),
            len(self.selected_layer_names),
            int(self.vectorizer.total_dim),
            int(self._full_param_dim),
            ratio,
        )

    def _derive_stage_prefix(self, cfg: CupidActLastLayerConfig) -> str:
        if cfg.param_scope == "last_n_layers":
            tag = f"last{int(cfg.last_n_layers)}"
        elif cfg.param_scope == "keys":
            keys = _parse_csv(cfg.param_keys) or []
            safe = "_".join(k.replace("/", "_").replace(".", "_") for k in keys) or "keys"
            tag = f"keys_{safe}"
        else:
            tag = "all_params"
        return f"{cfg.output_path}.scope_{tag}"

    def _count_trainable_params(self) -> Tuple[int, int]:
        count = 0
        dim = 0
        for _, p in self.policy.named_parameters():
            if not p.requires_grad:
                continue
            count += 1
            dim += int(p.numel())
        return count, dim

    @staticmethod
    def _layer_name(param_name: str) -> str:
        if "." not in param_name:
            return param_name
        return param_name.rsplit(".", 1)[0]

    def _all_trainable_named_params(self) -> List[Tuple[str, torch.nn.Parameter]]:
        return [(n, p) for n, p in self.policy.named_parameters() if p.requires_grad]

    def _select_parameter_names(
        self, cfg: CupidActLastLayerConfig
    ) -> Tuple[List[str], List[str]]:
        named = self._all_trainable_named_params()
        if not named:
            raise ValueError("Policy has no trainable parameters.")

        scope = cfg.param_scope.strip().lower()
        if scope not in {"all", "last_n_layers", "keys"}:
            raise ValueError("param_scope must be one of: all, last_n_layers, keys.")

        if scope == "all":
            selected = [n for n, _ in named]
            layers = list(dict.fromkeys(self._layer_name(n) for n in selected))
            return selected, layers

        if scope == "keys":
            keys = _parse_csv(cfg.param_keys)
            if not keys:
                raise ValueError("param_scope=keys requires --param-keys.")
            selected = [n for n, _ in named if any(k in n for k in keys)]
            if not selected:
                raise ValueError(f"No params matched keys: {keys}")
            layers = list(dict.fromkeys(self._layer_name(n) for n in selected))
            return selected, layers

        # scope == "last_n_layers"
        n_layers = int(cfg.last_n_layers)
        if n_layers <= 0:
            raise ValueError("--last-n-layers must be > 0 when param_scope=last_n_layers.")
        ordered_layers: List[str] = []
        seen = set()
        for n, _ in named:
            layer = self._layer_name(n)
            if layer not in seen:
                ordered_layers.append(layer)
                seen.add(layer)
        n_layers = min(n_layers, len(ordered_layers))
        selected_layers = ordered_layers[-n_layers:]
        selected_layer_set = set(selected_layers)
        selected = [n for n, _ in named if self._layer_name(n) in selected_layer_set]
        if not selected:
            raise ValueError(f"No params found in last {n_layers} layers.")
        return selected, selected_layers

    def _rebuild_projection(self) -> None:
        self.projection = None
        self.projector = None
        proj_dim = int(self.cfg.proj_dim)
        if proj_dim <= 0:
            return

        bytes_required = int(self.vectorizer.total_dim) * proj_dim * 4
        if bytes_required <= MAX_PROJECTION_BYTES:
            try:
                self.projection = torch.randn(
                    self.vectorizer.total_dim, proj_dim, device=self.device
                ) / math.sqrt(proj_dim)
            except RuntimeError as exc:
                if "out of memory" not in str(exc).lower():
                    raise
                LOGGER.warning(
                    "Dense projection allocation failed with OOM (%s). Falling back to CountSketch.",
                    exc,
                )
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
        if self.projection is None:
            LOGGER.info(
                "Using CountSketch projection with proj_dim=%d (dense would require %.2f GiB).",
                proj_dim,
                bytes_required / (1024 ** 3),
            )
            self.projector = CountSketchProjector(
                self.vectorizer.total_dim,
                proj_dim,
                self.device,
                int(self.cfg.seed),
            )

    def _save_result(self, result: CupidActInfluenceResult) -> None:
        result.metadata = dict(result.metadata)
        result.metadata["selected_param_names"] = np.asarray(self.selected_param_names, dtype=object)
        result.metadata["selected_layer_names"] = np.asarray(self.selected_layer_names, dtype=object)
        result.metadata["selected_param_dim"] = np.asarray(int(self.vectorizer.total_dim), dtype=np.int64)
        result.metadata["full_param_dim"] = np.asarray(int(self._full_param_dim), dtype=np.int64)
        result.metadata["selected_param_ratio"] = np.asarray(
            float(self.vectorizer.total_dim) / max(float(self._full_param_dim), 1.0),
            dtype=np.float64,
        )
        super()._save_result(result)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CUPID-ACT influence calculator (parameter-subset / last-layer mode)."
    )
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
        "--stage_cache_prefix",
        type=str,
        default=None,
        help="Optional path prefix for stage cache artifacts; defaults to output directory.",
    )
    parser.add_argument(
        "--param-scope",
        type=str,
        default="last_n_layers",
        choices=("all", "last_n_layers", "keys"),
        help="Parameter scope for gradient/HVP: all params, last N layers, or key match.",
    )
    parser.add_argument(
        "--last-n-layers",
        type=int,
        default=8,
        help="Number of last layers to keep when --param-scope=last_n_layers.",
    )
    parser.add_argument(
        "--param-keys",
        type=str,
        default=None,
        help="Comma-separated substrings for --param-scope=keys.",
    )
    parser.add_argument(
        "--no-auto-stage-suffix",
        action="store_true",
        help="Disable automatic stage-cache suffix for parameter scope.",
    )
    parser.add_argument(
        "--print-selected-params",
        action="store_true",
        help="Print all selected parameter names before computation.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    cfg = CupidActLastLayerConfig(
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
        param_scope=args.param_scope,
        last_n_layers=args.last_n_layers,
        param_keys=args.param_keys,
        auto_stage_suffix=not args.no_auto_stage_suffix,
        print_selected_params=args.print_selected_params,
    )
    calculator = CupidActLastLayerInfluenceCalculator(cfg)
    calculator.run()


if __name__ == "__main__":
    main()
