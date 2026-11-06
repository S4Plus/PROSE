from __future__ import annotations

"""CUPID-ACT influence with strict-topology parameter-subset selection.

This script keeps the original influence-function pipeline unchanged, but
restricts gradient/HVP computation to a parameter subset (e.g., the last N
layers) to reduce compute and memory.

Examples (nohup + multi-GPU):

GPU 1 (strict last 4 layers):
CUDA_VISIBLE_DEVICES=4 nohup python /workspace/liushuo/cupid/act/cupid_act_influence_strict_last_layers.py \
  --pretrained_dir /workspace/liushuo/cupid/3_model/can_mh/checkpoints/100000/pretrained_model \
  --train_dataset_path /workspace/liushuo/cupid/datasets/lerobot/can_mh \
  --eval_rollout_dir /workspace/liushuo/cupid/mimicgen/eval/tmp/can_mh_1_63_73/episodes \
  --output_path /workspace/liushuo/cupid/1_infulenece_score/can_mh_strict/score_last4 \
  --param-scope last_n_layers \
  --last-n-layers 4 \
  > /workspace/liushuo/cupid/3_model/logs/cupid_act_influence_last4_gpu1.log 2>&1 &

GPU 2 (strict last 8 layers):
CUDA_VISIBLE_DEVICES=5 nohup python /workspace/liushuo/cupid/act/cupid_act_influence_strict_last_layers.py \
  --pretrained_dir /workspace/liushuo/cupid/3_model/can_mh/checkpoints/100000/pretrained_model \
  --train_dataset_path /workspace/liushuo/cupid/datasets/lerobot/can_mh \
  --eval_rollout_dir /workspace/liushuo/cupid/mimicgen/eval/tmp/can_mh_1_63_73/episodes \
  --output_path /workspace/liushuo/cupid/1_infulenece_score/can_mh_strict/score_last8 \
  --param-scope last_n_layers \
  --last-n-layers 8 \
  > /workspace/liushuo/cupid/3_model/logs/cupid_act_influence_last8_gpu2.log 2>&1 &

GPU 3 (strict last 12 layers):
CUDA_VISIBLE_DEVICES=6 nohup python /workspace/liushuo/cupid/act/cupid_act_influence_strict_last_layers.py \
  --pretrained_dir /workspace/liushuo/cupid/3_model/can_mh/checkpoints/100000/pretrained_model \
  --train_dataset_path /workspace/liushuo/cupid/datasets/lerobot/can_mh \
  --eval_rollout_dir /workspace/liushuo/cupid/mimicgen/eval/tmp/can_mh_1_63_73/episodes \
  --output_path /workspace/liushuo/cupid/1_infulenece_score/can_mh_strict/score_last12 \
  --param-scope last_n_layers \
  --last-n-layers 12 \
  > /workspace/liushuo/cupid/3_model/logs/cupid_act_influence_last12_gpu3.log 2>&1 &




GPU 1 (strict last 4 layers):
CUDA_VISIBLE_DEVICES=4 nohup python /workspace/liushuo/cupid/act/cupid_act_influence_strict_last_layers.py \
  --pretrained_dir /workspace/liushuo/cupid/3_model/can_ph_model/random/random42_1.0/checkpoints/100000/pretrained_model \
  --train_dataset_path /workspace/liushuo/cupid/datasets/lerobot/can_ph \
  --eval_rollout_dir /workspace/liushuo/cupid/mimicgen/eval/tmp/can_ph/random/1.0/episodes\
  --output_path /workspace/liushuo/cupid/1_infulenece_score/can_ph_strict/score_last4 \
  --param-scope last_n_layers \
  --last-n-layers 4 \
  > /workspace/liushuo/cupid/3_model/logs/cupid_act_influence_last4_gpu1.log 2>&1 &

GPU 2 (strict last 8 layers):
CUDA_VISIBLE_DEVICES=5 nohup python /workspace/liushuo/cupid/act/cupid_act_influence_strict_last_layers.py \
  --pretrained_dir /workspace/liushuo/cupid/3_model/can_ph_model/random/random42_1.0/checkpoints/100000/pretrained_model \
  --train_dataset_path /workspace/liushuo/cupid/datasets/lerobot/can_ph \
  --eval_rollout_dir /workspace/liushuo/cupid/mimicgen/eval/tmp/can_ph/random/1.0/episodes \
  --output_path /workspace/liushuo/cupid/1_infulenece_score/can_ph_strict/score_last8 \
  --param-scope last_n_layers \
  --last-n-layers 8 \
  > /workspace/liushuo/cupid/3_model/logs/cupid_act_influence_last8_gpu2.log 2>&1 &

GPU 3 (strict last 12 layers):
CUDA_VISIBLE_DEVICES=6 nohup python /workspace/liushuo/cupid/act/cupid_act_influence_strict_last_layers.py \
  --pretrained_dir /workspace/liushuo/cupid/3_model/can_ph_model/random/random42_1.0/checkpoints/100000/pretrained_model \
  --train_dataset_path /workspace/liushuo/cupid/datasets/lerobot/can_ph \
  --eval_rollout_dir /workspace/liushuo/cupid/mimicgen/eval/tmp/can_ph/random/1.0/episodes \
  --output_path /workspace/liushuo/cupid/1_infulenece_score/can_ph_strict/score_last12 \
  --param-scope last_n_layers \
  --last-n-layers 12 \
  > /workspace/liushuo/cupid/3_model/logs/cupid_act_influence_last12_gpu3.log 2>&1 &
"""

import argparse
import logging
import math
import re
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

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
    """Influence calculator that supports strict-topology parameter selection."""

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
            tag = f"strict_last{int(cfg.last_n_layers)}"
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

    @staticmethod
    def _decoder_inner_rank(inner_layer: str) -> Tuple[int, str]:
        # Reverse decoder dataflow (closest-to-output first).
        ordered_prefix = (
            "norm3",
            "linear2",
            "linear1",
            "norm2",
            "multihead_attn.out_proj",
            "multihead_attn",
            "norm1",
            "self_attn.out_proj",
            "self_attn",
        )
        for idx, prefix in enumerate(ordered_prefix):
            if inner_layer == prefix or inner_layer.startswith(prefix + "."):
                return idx, inner_layer
        return len(ordered_prefix), inner_layer

    def _strict_input_to_output_layers(self, named: List[Tuple[str, torch.nn.Parameter]]) -> List[str]:
        ordered_layers: List[str] = []
        first_seen_idx = {}
        for n, _ in named:
            layer = self._layer_name(n)
            if layer not in first_seen_idx:
                first_seen_idx[layer] = len(ordered_layers)
                ordered_layers.append(layer)

        def output_to_input_key(layer: str) -> Tuple[int, int, int, str]:
            if layer == "model.action_head":
                return (0, 0, 0, layer)
            if layer == "model.decoder.norm":
                return (1, 0, 0, layer)
            decoder_match = re.match(r"^model\.decoder\.layers\.(\d+)\.(.+)$", layer)
            if decoder_match:
                layer_idx = int(decoder_match.group(1))
                inner = decoder_match.group(2)
                inner_rank, inner_name = self._decoder_inner_rank(inner)
                # Larger decoder index is closer to output.
                return (2, -layer_idx, inner_rank, inner_name)
            if layer == "model.decoder_pos_embed":
                return (3, 0, 0, layer)
            if layer == "model.encoder_1d_feature_pos_embed":
                return (4, 0, 0, layer)
            if layer == "model.encoder_img_feat_input_proj":
                return (5, 0, 0, layer)
            if layer == "model.encoder_latent_input_proj":
                return (6, 0, 0, layer)
            if layer == "model.encoder_robot_state_input_proj":
                return (7, 0, 0, layer)
            # Fallback: later-registered layers are treated as more output-side.
            return (100, -int(first_seen_idx[layer]), 0, layer)

        output_to_input = sorted(ordered_layers, key=output_to_input_key)
        return list(reversed(output_to_input))

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
        ordered_layers = self._strict_input_to_output_layers(named)
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
        result.metadata["layer_selection_rule"] = "strict_topology_last_n_layers"
        result.metadata["layer_selection_order"] = "input_to_output"
        result.metadata["selected_param_names"] = np.asarray(self.selected_param_names, dtype=object)
        result.metadata["selected_layer_names"] = np.asarray(self.selected_layer_names, dtype=object)
        result.metadata["selected_param_dim"] = np.asarray(int(self.vectorizer.total_dim), dtype=np.int64)
        result.metadata["full_param_dim"] = np.asarray(int(self._full_param_dim), dtype=np.int64)
        result.metadata["selected_param_ratio"] = np.asarray(
            float(self.vectorizer.total_dim) / max(float(self._full_param_dim), 1.0),
            dtype=np.float64,
        )
        super()._save_result(result)

    def run(self) -> CupidActInfluenceResult:
        total_start = time.perf_counter()

        LOGGER.info("Stage 1: caching ACT gradients with pseudo Huber loss.")
        t0 = time.perf_counter()
        projected_train = self._get_or_cache_train_gradients()
        stage1_sec = time.perf_counter() - t0
        LOGGER.info("Stage 1 done in %.2fs.", stage1_sec)

        LOGGER.info("Stage 2: collecting rollout gradients + returns.")
        t0 = time.perf_counter()
        return_grad = self._get_or_compute_return_gradient()
        stage2_sec = time.perf_counter() - t0
        LOGGER.info("Stage 2 done in %.2fs.", stage2_sec)

        LOGGER.info("Stage 3: solving LiSSA inverse-HVP.")
        influence_scores: np.ndarray
        score_metadata: Dict[str, np.ndarray] = {}
        stage3_sec = 0.0
        stage4_sec = 0.0
        if self.bootstrap_samples > 1:
            if return_grad.ndim != 2:
                raise RuntimeError("Expected bootstrap return gradients to have shape (B, D).")
            scores_list: List[np.ndarray] = []
            inv_hvp_ref: Optional[torch.Tensor] = None
            base_metadata: Dict[str, np.ndarray] = {}
            for b in range(self.bootstrap_samples):
                t3 = time.perf_counter()
                inv_hvp = self._get_or_compute_inverse_hvp(return_grad[b], cache_suffix=f"boot{b}")
                stage3_sec += time.perf_counter() - t3

                t4 = time.perf_counter()
                scores, meta = self._score_influences(projected_train, inv_hvp)
                stage4_sec += time.perf_counter() - t4

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
            t3 = time.perf_counter()
            inv_hvp = self._get_or_compute_inverse_hvp(return_grad)
            stage3_sec = time.perf_counter() - t3
            LOGGER.info("Stage 4: scoring train samples via gradient dot-products.")
            t4 = time.perf_counter()
            influence_scores, score_metadata = self._score_influences(projected_train, inv_hvp)
            stage4_sec = time.perf_counter() - t4
        LOGGER.info("Stage 3 done in %.2fs.", stage3_sec)
        LOGGER.info("Stage 4 done in %.2fs.", stage4_sec)

        post_start = time.perf_counter()
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
        post_sec = time.perf_counter() - post_start
        total_sec = time.perf_counter() - total_start
        result.metadata = dict(result.metadata)
        result.metadata["timing_stage1_sec"] = np.asarray(stage1_sec, dtype=np.float64)
        result.metadata["timing_stage2_sec"] = np.asarray(stage2_sec, dtype=np.float64)
        result.metadata["timing_stage3_sec"] = np.asarray(stage3_sec, dtype=np.float64)
        result.metadata["timing_stage4_sec"] = np.asarray(stage4_sec, dtype=np.float64)
        result.metadata["timing_post_build_sec"] = np.asarray(post_sec, dtype=np.float64)
        result.metadata["timing_total_pre_save_sec"] = np.asarray(total_sec, dtype=np.float64)
        self._save_result(result)
        LOGGER.info(
            "Timing summary (sec): stage1=%.2f stage2=%.2f stage3=%.2f stage4=%.2f post_build=%.2f total_pre_save=%.2f",
            stage1_sec,
            stage2_sec,
            stage3_sec,
            stage4_sec,
            post_sec,
            total_sec,
        )
        return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CUPID-ACT influence calculator (strict-topology last-layer mode)."
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
        help="Parameter scope for gradient/HVP: all params, strict-topology last N layers, or key match.",
    )
    parser.add_argument(
        "--last-n-layers",
        type=int,
        default=8,
        help="Number of strict-topology last layers to keep when --param-scope=last_n_layers.",
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
