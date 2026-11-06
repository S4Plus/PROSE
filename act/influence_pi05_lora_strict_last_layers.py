#!/usr/bin/env python3
"""OpenPI LoRA Influence Calculator with strict-topology last-N selection.

This script keeps the original OpenPI pipeline in
`/workspace/liushuo/cupid/openpi/scripts/influence_pi05_lora.py` unchanged,
while restricting gradients / HVP to a parameter subset (strict last N layers).

Examples:
  python /workspace/liushuo/cupid/act/influence_pi05_lora_strict_last_layers.py \
      --config-name pi05_square_d1_lora \
      --checkpoint-dir /workspace/liushuo/cupid/openpi/checkpoints/pi05_square_d1_lora/exp1/20000 \
      --eval-rollouts-dir /path/to/eval_rollouts \
      --output-path /path/to/influence_last4.pkl \
      --param-scope last_n_layers \
      --last-n-layers 4
"""

from __future__ import annotations

import argparse
import logging
import pathlib
import pickle
import re
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

OPENPI_ROOT = pathlib.Path("/workspace/liushuo/cupid/openpi").resolve()
OPENPI_SRC = OPENPI_ROOT / "src"
OPENPI_SCRIPTS = OPENPI_ROOT / "scripts"

for p in (OPENPI_SRC, OPENPI_SCRIPTS):
    p_str = str(p)
    if p_str not in sys.path:
        sys.path.insert(0, p_str)

import flax.nnx as nnx
from flax.nnx import filterlib
import jax
import jax.numpy as jnp
import numpy as np

from influence_pi05_lora import (
    CountSketchProjector,
    InfluenceCalculator,
    InfluenceConfig,
    LOGGER,
    MAX_PROJECTION_BYTES,
    ParameterVectorizer,
)


def _parse_csv(value: Optional[str]) -> Optional[List[str]]:
    if value is None:
        return None
    items = [item.strip() for item in value.split(",") if item.strip()]
    return items or None


@dataclass
class InfluenceLastLayerConfig(InfluenceConfig):
    """Extension of the base config with parameter-scope controls."""

    param_scope: str = "last_n_layers"
    last_n_layers: int = 4
    param_keys: Optional[str] = None
    auto_stage_suffix: bool = True
    print_selected_params: bool = False


class InfluenceLastLayerCalculator(InfluenceCalculator):
    """Influence calculator with strict-topology last-N parameter selection."""

    def __init__(self, cfg: InfluenceLastLayerConfig):
        self.last_cfg = cfg

        if cfg.auto_stage_suffix and cfg.stage_cache_prefix is None:
            cfg.stage_cache_prefix = self._derive_stage_cache_dir(cfg)

        # Avoid allocating projection for full trainable params first.
        proj_dim = int(cfg.proj_dim)
        cfg.proj_dim = 0
        super().__init__(cfg)
        cfg.proj_dim = proj_dim
        self.cfg.proj_dim = proj_dim

        full_flat = self.trainable_template.flat_state()
        self._full_param_count = len(full_flat)
        self._full_param_dim = int(sum(self._leaf_numel(v) for v in full_flat.values()))

        selected_paths, selected_param_names, selected_layer_names = self._select_parameter_paths(cfg, full_flat)
        if not selected_paths:
            raise ValueError("No parameters selected for influence computation.")

        all_params = nnx.state(self.model)
        path_filter = filterlib.PathIn(*selected_paths)
        selected_state = all_params.filter(nnx.All(self.train_cfg.trainable_filter, path_filter))
        selected_flat = selected_state.flat_state()
        if not selected_flat:
            raise ValueError("Selected parameter filter produced an empty trainable state.")

        self.trainable_template = selected_state
        selected_values = jax.tree.map(lambda p: p.value, self.trainable_template)
        self.vectorizer = ParameterVectorizer(selected_values)
        self.flat_params = self.vectorizer.flatten(selected_values)

        self.selected_param_names = selected_param_names
        self.selected_layer_names = selected_layer_names

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

    @staticmethod
    def _leaf_numel(leaf: Any) -> int:
        value = getattr(leaf, "value", leaf)
        size = getattr(value, "size", None)
        if size is None:
            return int(np.size(value))
        return int(size)

    @staticmethod
    def _path_to_name(path: Tuple[Any, ...]) -> str:
        return "/".join(str(part) for part in path)

    @staticmethod
    def _layer_name(path: Tuple[Any, ...]) -> str:
        if len(path) <= 1:
            return "/".join(str(part) for part in path)
        return "/".join(str(part) for part in path[:-1])

    def _derive_stage_cache_dir(self, cfg: InfluenceLastLayerConfig) -> str:
        if cfg.param_scope == "last_n_layers":
            tag = f"strict_last{int(cfg.last_n_layers)}"
        elif cfg.param_scope == "keys":
            keys = _parse_csv(cfg.param_keys) or []
            safe = "_".join(k.replace("/", "_").replace(".", "_") for k in keys) or "keys"
            tag = f"keys_{safe}"
        else:
            tag = "all_params"

        output = pathlib.Path(cfg.output_path).expanduser()
        if output.suffix:
            base = output.stem
            parent = output.parent if str(output.parent) != "" else pathlib.Path(".")
        else:
            base = output.name or "influence_scores"
            parent = output.parent if str(output.parent) != "" else pathlib.Path(".")
        return str(parent / f"{base}.scope_{tag}")

    def _select_parameter_paths(
        self,
        cfg: InfluenceLastLayerConfig,
        flat_state: Dict[Tuple[Any, ...], Any],
    ) -> Tuple[List[Tuple[Any, ...]], List[str], List[str]]:
        items = list(flat_state.items())
        if not items:
            raise ValueError("No trainable parameters found in model state.")

        scope = cfg.param_scope.strip().lower()
        if scope not in {"all", "last_n_layers", "keys"}:
            raise ValueError("param_scope must be one of: all, last_n_layers, keys.")

        entries: List[Tuple[Tuple[Any, ...], str, str]] = []
        for path, _ in items:
            name = self._path_to_name(path)
            layer = self._layer_name(path)
            entries.append((path, name, layer))

        if scope == "all":
            selected = entries
            selected_layers = list(dict.fromkeys(layer for _, _, layer in selected))
            return [p for p, _, _ in selected], [n for _, n, _ in selected], selected_layers

        if scope == "keys":
            keys = _parse_csv(cfg.param_keys)
            if not keys:
                raise ValueError("param_scope=keys requires --param-keys.")
            selected = [entry for entry in entries if any(k in entry[1] for k in keys)]
            if not selected:
                raise ValueError(f"No params matched keys: {keys}")
            selected_layers = list(dict.fromkeys(layer for _, _, layer in selected))
            return [p for p, _, _ in selected], [n for _, n, _ in selected], selected_layers

        # scope == "last_n_layers" (strict topology order)
        n_layers = int(cfg.last_n_layers)
        if n_layers <= 0:
            raise ValueError("--last-n-layers must be > 0 when param_scope=last_n_layers.")

        ordered_layers = self._strict_input_to_output_layers(entries)

        n_layers = min(n_layers, len(ordered_layers))
        selected_layers = ordered_layers[-n_layers:]
        selected_layer_set = set(selected_layers)
        selected = [entry for entry in entries if entry[2] in selected_layer_set]
        if not selected:
            raise ValueError(f"No params found in last {n_layers} layers.")
        return [p for p, _, _ in selected], [n for _, n, _ in selected], selected_layers

    @staticmethod
    def _decoder_inner_rank(inner: str) -> Tuple[int, str]:
        """Order submodules inside one transformer block from output-side to input-side."""
        # Output-side first: down_proj/o_proj ... input-side last: q_proj.
        ordered_prefix = (
            "mlp/down_proj",
            "self_attn/o_proj",
            "mlp/up_proj",
            "mlp/gate_proj",
            "self_attn/v_proj",
            "self_attn/k_proj",
            "self_attn/q_proj",
            "post_attention_layernorm",
            "input_layernorm",
        )
        for idx, prefix in enumerate(ordered_prefix):
            if inner == prefix or inner.startswith(prefix + "/"):
                return idx, inner
        return len(ordered_prefix), inner

    def _strict_input_to_output_layers(
        self, entries: List[Tuple[Tuple[Any, ...], str, str]]
    ) -> List[str]:
        """Build strict layer order (input -> output) using model-topology heuristics."""
        ordered_layers: List[str] = []
        first_seen_idx: Dict[str, int] = {}
        for _, _, layer in entries:
            if layer not in first_seen_idx:
                first_seen_idx[layer] = len(ordered_layers)
                ordered_layers.append(layer)

        def output_to_input_key(layer: str) -> Tuple[int, int, int, str]:
            lower = layer.lower()

            # Final action heads are closest to output.
            if any(k in lower for k in ("action_head", "action_proj", "policy_head", "actor_head")):
                return (0, 0, 0, lower)

            # Final model norm is typically right before output head.
            if lower.endswith("/norm") or "/model/norm" in lower or "final_norm" in lower:
                return (1, 0, 0, lower)

            # Transformer blocks: higher index is closer to output.
            m = re.search(r"/layers/(\d+)(?:/|$)", layer)
            if m:
                block_idx = int(m.group(1))
                inner = layer[m.end() :].lstrip("/")
                inner_rank, inner_name = self._decoder_inner_rank(inner)
                return (2, -block_idx, inner_rank, inner_name)

            # Mid/late modules (projectors) before embeddings.
            if any(k in lower for k in ("projector", "proj")):
                return (3, 0, 0, lower)

            # Embeddings / input-side towers.
            if any(k in lower for k in ("embed", "embedding", "vision_tower", "backbone", "encoder")):
                return (6, 0, 0, lower)

            # Fallback: later registered layers are treated as slightly more output-side.
            return (50, -int(first_seen_idx[layer]), 0, lower)

        output_to_input = sorted(ordered_layers, key=output_to_input_key)
        return list(reversed(output_to_input))

    def _rebuild_projection(self) -> None:
        self.projection_matrix = None
        self.projector = None

        proj_dim = int(self.cfg.proj_dim)
        if proj_dim <= 0:
            return

        bytes_required = int(self.vectorizer.total_dim) * proj_dim * 4
        if bytes_required <= MAX_PROJECTION_BYTES:
            try:
                proj_rng = jax.random.key(int(self.cfg.seed) + 1)
                self.projection_matrix = jax.random.normal(
                    proj_rng,
                    (self.vectorizer.total_dim, proj_dim),
                    dtype=jnp.float32,
                ) / jnp.sqrt(float(proj_dim))
            except Exception as exc:
                LOGGER.warning("Dense projection allocation failed (%s). Falling back to CountSketch.", exc)

        if self.projection_matrix is None:
            LOGGER.info(
                "Using CountSketch projection with proj_dim=%d (dense would require %.2f GiB).",
                proj_dim,
                bytes_required / (1024 ** 3),
            )
            self.projector = CountSketchProjector(self.vectorizer.total_dim, proj_dim, int(self.cfg.seed))

    def run(self) -> Dict[str, Any]:
        result = super().run()
        metadata = dict(result.get("metadata", {}))
        metadata["layer_selection_rule"] = np.asarray("strict_topology_last_n_layers", dtype=object)
        metadata["layer_selection_order"] = np.asarray("input_to_output", dtype=object)
        metadata["selected_param_names"] = np.asarray(self.selected_param_names, dtype=object)
        metadata["selected_layer_names"] = np.asarray(self.selected_layer_names, dtype=object)
        metadata["selected_param_dim"] = np.asarray(int(self.vectorizer.total_dim), dtype=np.int64)
        metadata["full_param_dim"] = np.asarray(int(self._full_param_dim), dtype=np.int64)
        metadata["selected_param_ratio"] = np.asarray(
            float(self.vectorizer.total_dim) / max(float(self._full_param_dim), 1.0),
            dtype=np.float64,
        )
        result["metadata"] = metadata

        # Re-save with parameter-scope metadata.
        with open(self.output_path, "wb") as f:
            pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
        LOGGER.info("Saved result with parameter-scope metadata: %s", self.output_path)
        return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="OpenPI LoRA influence calculator (strict-topology last-layer mode).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument("--config-name", required=True, help="Training config name (e.g. pi05_square_d1_lora)")
    parser.add_argument("--checkpoint-dir", required=True, help="Checkpoint directory")
    parser.add_argument("--eval-rollouts-dir", required=True, help="Eval rollouts directory")
    parser.add_argument("--output-path", default="./influence_scores.pkl", help="Output path")

    parser.add_argument("--batch-size", type=int, default=8, help="Training batch size")
    parser.add_argument("--hvp-batch-size", type=int, default=4, help="HVP batch size")
    parser.add_argument("--num-workers", type=int, default=0, help="Data loader workers")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")

    parser.add_argument("--proj-dim", type=int, default=4096, help="Projection dimension")
    parser.add_argument("--lissa-depth", type=int, default=8, help="LiSSA depth")
    parser.add_argument("--lissa-samples", type=int, default=4, help="LiSSA samples")
    parser.add_argument("--lissa-scale", type=float, default=10.0, help="LiSSA scale")
    parser.add_argument("--lissa-damping", type=float, default=0.01, help="LiSSA damping")

    parser.add_argument("--save-projection", action="store_true", help="Save projection matrix/state")
    parser.add_argument("--stage-cache-prefix", help="Optional stage cache directory")

    parser.add_argument(
        "--param-scope",
        type=str,
        default="last_n_layers",
        choices=("all", "last_n_layers", "keys"),
        help="Parameter scope for gradients/HVP.",
    )
    parser.add_argument(
        "--last-n-layers",
        type=int,
        default=4,
        help="Number of strict-topology final layers to keep when --param-scope=last_n_layers.",
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
        help="Print selected parameter names before computation.",
    )

    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    args = parse_args()

    cfg = InfluenceLastLayerConfig(
        config_name=args.config_name,
        checkpoint_dir=args.checkpoint_dir,
        eval_rollouts_dir=args.eval_rollouts_dir,
        output_path=args.output_path,
        batch_size=args.batch_size,
        hvp_batch_size=args.hvp_batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        proj_dim=args.proj_dim,
        lissa_depth=args.lissa_depth,
        lissa_samples=args.lissa_samples,
        lissa_scale=args.lissa_scale,
        lissa_damping=args.lissa_damping,
        save_projection=args.save_projection,
        stage_cache_prefix=args.stage_cache_prefix,
        param_scope=args.param_scope,
        last_n_layers=args.last_n_layers,
        param_keys=args.param_keys,
        auto_stage_suffix=not args.no_auto_stage_suffix,
        print_selected_params=args.print_selected_params,
    )

    calculator = InfluenceLastLayerCalculator(cfg)
    calculator.run()


if __name__ == "__main__":
    main()
