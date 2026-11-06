#!/usr/bin/env python3
"""Build an influence graph: eval outputs -> train trajectories.

Modes:
- global: reuse score.pkl influence (no extra compute).
- per-episode: recompute per-eval-episode influence via LiSSA (compute-heavy).
"""
from __future__ import annotations

import argparse
import csv
import json
import pathlib
import pickle
import sys
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch

sys.path.append("/workspace/liushuo/cupid/act")
from cupid_act_influence import CupidActInfluenceCalculator, CupidActInfluenceConfig


def _ensure_numpy_core_aliases() -> None:
    core = getattr(np, "core", None)
    if core is None:
        return
    sys.modules.setdefault("numpy._core", core)
    if hasattr(core, "numeric"):
        sys.modules.setdefault("numpy._core.numeric", core.numeric)
    if hasattr(core, "multiarray"):
        sys.modules.setdefault("numpy._core.multiarray", core.multiarray)


def _load_payload(path: pathlib.Path) -> Dict:
    _ensure_numpy_core_aliases()
    with path.open("rb") as f:
        return pickle.load(f)


def _load_config(payload: Dict, override_eval_dir: Optional[str], override_device: Optional[str]) -> CupidActInfluenceConfig:
    if "config" not in payload:
        raise ValueError("score.pkl missing config; cannot reconstruct model/dataset settings.")
    cfg_dict = dict(payload["config"])
    if override_eval_dir is not None:
        cfg_dict["eval_rollout_dir"] = override_eval_dir
    if override_device is not None:
        cfg_dict["device"] = override_device
    # Force episode aggregation for train influence graph.
    cfg_dict["aggregate_mode"] = "episode"
    return CupidActInfluenceConfig(**cfg_dict)


def _resolve_global_scores(payload: Dict) -> np.ndarray:
    metadata = payload.get("metadata") or {}
    if isinstance(metadata, dict) and "episode_influence_scores" in metadata:
        scores = np.asarray(metadata["episode_influence_scores"])
    else:
        scores = np.asarray(payload.get("influence_scores"))
    if scores is None:
        raise ValueError("score.pkl missing influence_scores.")
    return scores


def _episode_gradients(calc: CupidActInfluenceCalculator) -> Tuple[List[torch.Tensor], np.ndarray]:
    lengths = list(calc.eval_dataset.episode_lengths)
    if not lengths:
        return [], np.asarray([], dtype=np.float32)

    rewards = calc.eval_rewards.detach().cpu().numpy()
    grads: List[torch.Tensor] = []
    episode_grad = torch.zeros(calc.vectorizer.total_dim, device=calc.device, dtype=torch.float32)
    ep_idx = 0
    steps_remaining = lengths[ep_idx]
    for grad in calc._iter_eval_gradients():
        episode_grad += grad
        steps_remaining -= 1
        if steps_remaining == 0:
            grads.append(episode_grad.clone())
            ep_idx += 1
            if ep_idx >= len(lengths):
                break
            episode_grad.zero_()
            steps_remaining = lengths[ep_idx]
    return grads, rewards


def _compute_train_scores(
    calc: CupidActInfluenceCalculator,
    train_tensor: torch.Tensor,
    inv_hvp: torch.Tensor,
) -> np.ndarray:
    if calc.projection is not None:
        inv_proj = torch.matmul(inv_hvp, calc.projection)
    elif calc.projector is not None:
        inv_proj = calc.projector.project(inv_hvp)
    else:
        inv_proj = inv_hvp
    scores = -torch.matmul(train_tensor, inv_proj)
    scores_np = scores.detach().cpu().numpy()
    if calc.train_episode_indices is None:
        return scores_np
    return calc._aggregate_episode_scores(scores_np)


def _select_topk(
    scores: np.ndarray,
    topk: int,
    rank_abs: bool,
    min_quantile: float,
    min_abs: Optional[float],
) -> List[int]:
    if scores.size == 0:
        return []
    if rank_abs:
        key = np.abs(scores)
    else:
        key = scores
    threshold = -np.inf
    if min_quantile > 0:
        threshold = np.quantile(key, min_quantile)
    if min_abs is not None:
        threshold = max(threshold, float(min_abs))
    order = np.argsort(key)[::-1]
    selected: List[int] = []
    for idx in order:
        if key[idx] < threshold:
            continue
        selected.append(int(idx))
        if len(selected) >= topk:
            break
    return selected


def _write_csv(path: pathlib.Path, rows: Iterable[Dict[str, object]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build influence graph from CUPID-ACT outputs.")
    parser.add_argument("--score-pkl", type=pathlib.Path, required=True, help="Path to score.pkl")
    parser.add_argument(
        "--mode",
        type=str,
        default="global",
        choices=("global", "per-episode"),
        help="global: reuse score.pkl; per-episode: recompute per eval episode.",
    )
    parser.add_argument(
        "--eval-rollout-dir",
        type=str,
        default=None,
        help="Override eval_rollout_dir (defaults to config in score.pkl).",
    )
    parser.add_argument("--output-dir", type=pathlib.Path, required=True, help="Output directory.")
    parser.add_argument("--top-k", type=int, default=20, help="Top-K train episodes per output.")
    parser.add_argument(
        "--rank-abs",
        action="store_true",
        default=True,
        help="Rank by absolute influence (default).",
    )
    parser.add_argument(
        "--rank-signed",
        action="store_true",
        help="Rank by signed influence (largest positive first).",
    )
    parser.add_argument(
        "--min-quantile",
        type=float,
        default=0.05,
        help="Drop edges below this quantile of |influence|.",
    )
    parser.add_argument(
        "--min-abs",
        type=float,
        default=None,
        help="Minimum absolute influence threshold (applied with min-quantile).",
    )
    parser.add_argument(
        "--episode-weight",
        type=str,
        default="centered_reward",
        choices=("centered_reward", "reward", "none"),
        help="How to weight per-episode gradients.",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Limit the number of eval episodes (for faster runs).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override device (e.g., cuda:0, cpu).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.rank_signed:
        rank_abs = False
    else:
        rank_abs = True

    payload = _load_payload(args.score_pkl)
    edges: List[Dict[str, object]] = []
    nodes: List[Dict[str, object]] = []

    if args.mode == "global":
        scores = _resolve_global_scores(payload)
        output_id = "eval_global"
        selected = _select_topk(
            scores=scores,
            topk=args.top_k,
            rank_abs=rank_abs,
            min_quantile=args.min_quantile,
            min_abs=args.min_abs,
        )
        for rank, train_idx in enumerate(selected, start=1):
            val = float(scores[train_idx])
            edges.append(
                {
                    "output_id": output_id,
                    "train_episode_id": f"train_ep_{train_idx:06d}",
                    "influence": val,
                    "abs_influence": abs(val),
                    "sign": "pos" if val >= 0 else "neg",
                    "rank": rank,
                    "output_reward": 0.0,
                    "output_weight": 0.0,
                }
            )
        nodes.append({"node_id": output_id, "type": "output", "reward": 0.0, "weight": 0.0})
        num_train = scores.shape[0]
        for ep in range(num_train):
            nodes.append({"node_id": f"train_ep_{ep:06d}", "type": "train"})
    else:
        cfg = _load_config(payload, args.eval_rollout_dir, args.device)
        calc = CupidActInfluenceCalculator(cfg)

        # Prefer cached train projections from score.pkl if present.
        projected_train = payload.get("train_projections")
        if isinstance(projected_train, np.ndarray):
            train_proj = projected_train
        else:
            train_proj = calc._get_or_cache_train_gradients()

        train_tensor = torch.from_numpy(train_proj).to(calc.device)

        episode_grads, rewards = _episode_gradients(calc)
        if args.max_episodes is not None:
            episode_grads = episode_grads[: args.max_episodes]
            rewards = rewards[: args.max_episodes]

        if args.episode_weight == "reward":
            weights = rewards
        elif args.episode_weight == "none":
            weights = np.ones_like(rewards, dtype=np.float32)
        else:
            weights = rewards - rewards.mean() if rewards.size > 0 else rewards

        for idx, (grad, weight) in enumerate(zip(episode_grads, weights)):
            output_id = f"eval_ep_{idx:06d}"
            g = grad * float(weight)
            if not torch.isfinite(g).all():
                print(f"[warn] skip {output_id}: non-finite gradient")
                continue
            inv_hvp = calc._solve_inverse_hvp(g)
            scores = _compute_train_scores(calc, train_tensor, inv_hvp)

            selected = _select_topk(
                scores=scores,
                topk=args.top_k,
                rank_abs=rank_abs,
                min_quantile=args.min_quantile,
                min_abs=args.min_abs,
            )

            for rank, train_idx in enumerate(selected, start=1):
                val = float(scores[train_idx])
                edges.append(
                    {
                        "output_id": output_id,
                        "train_episode_id": f"train_ep_{train_idx:06d}",
                        "influence": val,
                        "abs_influence": abs(val),
                        "sign": "pos" if val >= 0 else "neg",
                        "rank": rank,
                        "output_reward": float(rewards[idx]) if rewards.size > idx else 0.0,
                        "output_weight": float(weight),
                    }
                )

            nodes.append(
                {
                    "node_id": output_id,
                    "type": "output",
                    "reward": float(rewards[idx]) if rewards.size > idx else 0.0,
                    "weight": float(weight),
                }
            )
            if (idx + 1) % 5 == 0:
                print(f"processed {idx + 1}/{len(episode_grads)} outputs")

        # Train nodes
        if calc.train_episode_indices is not None:
            num_train = int(calc.train_episode_indices.max()) + 1
        else:
            num_train = train_proj.shape[0]
        for ep in range(num_train):
            nodes.append({"node_id": f"train_ep_{ep:06d}", "type": "train"})

    args.output_dir.mkdir(parents=True, exist_ok=True)
    edges_path = args.output_dir / "influence_graph_edges.csv"
    nodes_path = args.output_dir / "influence_graph_nodes.csv"
    summary_path = args.output_dir / "influence_graph_summary.json"

    _write_csv(
        edges_path,
        edges,
        [
            "output_id",
            "train_episode_id",
            "influence",
            "abs_influence",
            "sign",
            "rank",
            "output_reward",
            "output_weight",
        ],
    )
    _write_csv(nodes_path, nodes, ["node_id", "type", "reward", "weight"])

    summary = {
        "score_pkl": str(args.score_pkl),
        "eval_rollout_dir": payload.get("config", {}).get("eval_rollout_dir"),
        "num_outputs": len([n for n in nodes if n.get("type") == "output"]),
        "num_train_nodes": num_train,
        "top_k": args.top_k,
        "rank_abs": rank_abs,
        "min_quantile": args.min_quantile,
        "min_abs": args.min_abs,
        "episode_weight": args.episode_weight,
        "mode": args.mode,
    }
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved edges to {edges_path}")
    print(f"Saved nodes to {nodes_path}")
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
