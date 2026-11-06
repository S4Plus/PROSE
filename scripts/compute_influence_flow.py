"""End-to-end influence workflow for CUPID.

This script stitches together the failure clustering output, CUPID influence
scores, 2D visualization, and a light-weight influence-gap based prescription.

Example:
python scripts/compute_influence_flow.py \
  --cluster_file eval_belt_pick_bottle/failure_clusters.json \
  --score_file eval_belt_pick_bottle/new_score/score.pkl \
  --output_dir outputs/influence_flow
"""

import argparse
import json
import math
import pickle
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader

from act.cupid_act_influence import (
    CupidActInfluenceCalculator,
    CupidActInfluenceConfig,
    EvalPklEpisodeDataset,
    _episode_rewards,
)

try:
    import umap  # type: ignore

    HAS_UMAP = True
except Exception:
    HAS_UMAP = False


def _load_json(path: Path) -> Dict[str, object]:
    with path.open("r") as f:
        return json.load(f)


def _read_episode_lengths(train_root: Path) -> List[int]:
    lengths: List[int] = []
    meta_path = train_root / "meta" / "episodes.jsonl"
    with meta_path.open("r") as f:
        for line in f:
            record = json.loads(line)
            lengths.append(int(record["length"]))
    return lengths


def _episode_indices_from_lengths(lengths: Sequence[int]) -> np.ndarray:
    ids: List[int] = []
    for ep_id, length in enumerate(lengths):
        ids.extend([ep_id] * int(length))
    return np.asarray(ids, dtype=np.int64)


def _filter_eval_dataset(dataset: EvalPklEpisodeDataset, keep_names: Iterable[str]) -> EvalPklEpisodeDataset:
    keep = set(keep_names)
    mask = [Path(p).name in keep for p in dataset.episode_files]
    dataset.episode_files = [p for p, m in zip(dataset.episode_files, mask) if m]
    dataset._episodes = [ep for ep, m in zip(dataset._episodes, mask) if m]  # type: ignore[attr-defined]
    dataset._episode_lengths = [l for l, m in zip(dataset._episode_lengths, mask) if m]  # type: ignore[attr-defined]
    dataset._episode_successes = [s for s, m in zip(dataset._episode_successes, mask) if m]  # type: ignore[attr-defined]
    dataset._length = sum(dataset._episode_lengths)  # type: ignore[attr-defined]
    return dataset


def _aggregate_episode_means(sample_array: np.ndarray, episode_ids: np.ndarray) -> np.ndarray:
    max_ep = int(episode_ids.max())
    accum = np.zeros((max_ep + 1, sample_array.shape[1]), dtype=np.float32)
    counts = np.zeros(max_ep + 1, dtype=np.int64)
    for sample, ep in zip(sample_array, episode_ids):
        accum[int(ep)] += sample
        counts[int(ep)] += 1
    counts = np.maximum(counts, 1)
    accum /= counts[:, None]
    return accum


def _compute_episode_projected_grads(
    calc: CupidActInfluenceCalculator, dataset: EvalPklEpisodeDataset
) -> np.ndarray:
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    episode_means: List[np.ndarray] = []
    ep_idx = 0
    steps_remaining = dataset._episode_lengths[ep_idx]  # type: ignore[attr-defined]
    cur: List[np.ndarray] = []

    for batch in loader:
        formatted = calc._format_batch(batch)
        for grad in calc._batch_gradients(formatted):
            proj = calc._project_gradient(grad)
            cur.append(proj)
            steps_remaining -= 1
            if steps_remaining == 0:
                episode_means.append(np.stack(cur).mean(axis=0))
                cur = []
                ep_idx += 1
                if ep_idx >= len(dataset._episode_lengths):  # type: ignore[attr-defined]
                    break
                steps_remaining = dataset._episode_lengths[ep_idx]  # type: ignore[attr-defined]
    return np.stack(episode_means, axis=0)


def _fit_2d(features: np.ndarray, seed: int = 0) -> np.ndarray:
    if HAS_UMAP:
        reducer = umap.UMAP(  # type: ignore[name-defined]
            n_components=2,
            random_state=seed,
            min_dist=0.1,
            n_neighbors=min(30, max(5, features.shape[0] // 10)),
        )
        return reducer.fit_transform(features)
    perplexity = min(30, max(5, (features.shape[0] - 1) // 3))
    tsne = TSNE(
        n_components=2,
        init="random",
        random_state=seed,
        perplexity=perplexity,
        learning_rate="auto",
    )
    return tsne.fit_transform(features)


def _visualize(
    coords: np.ndarray,
    train_count: int,
    influences: np.ndarray,
    failure_labels: List[str],
    failure_clusters: List[int],
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    train_xy = coords[:train_count]
    fail_xy = coords[train_count:]

    plt.figure(figsize=(8, 6))
    vmin = float(np.min(influences))
    vmax = float(np.max(influences))
    if vmax < vmin:
        vmin, vmax = vmax, vmin
    if math.isclose(vmin, vmax, rel_tol=1e-6, abs_tol=1e-9):
        vmax = vmin + 1e-3
    if vmin < 0.0 < vmax:
        norm = mcolors.TwoSlopeNorm(vcenter=0.0, vmin=vmin, vmax=vmax)
    else:
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    plt.scatter(train_xy[:, 0], train_xy[:, 1], c=influences, cmap="coolwarm", norm=norm, s=18, alpha=0.85, label="train")

    if len(failure_labels) > 0:
        clusters = sorted(set(failure_clusters))
        cmap = plt.get_cmap("tab10")
        for cl in clusters:
            idxs = [i for i, c in enumerate(failure_clusters) if c == cl]
            if not idxs:
                continue
            pts = fail_xy[idxs]
            plt.scatter(
                pts[:, 0],
                pts[:, 1],
                c=[cmap(cl % 10)],
                edgecolor="k",
                s=60,
                marker="o",
                label=f"fail cluster {cl}",
            )
        for pt, name in zip(fail_xy, failure_labels):
            plt.annotate(name.replace(".pkl", ""), (pt[0], pt[1]), fontsize=6, alpha=0.7)

    plt.legend()
    plt.title("Influence landscape (red=harming, blue=helping)")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()


def _topk(scores: np.ndarray, k: int = 10) -> List[Tuple[int, float]]:
    if scores.size == 0:
        return []
    k = min(k, scores.shape[0])
    order = np.argsort(scores)[::-1][:k]
    return [(int(i), float(scores[i])) for i in order]


def _normalize_rows(arr: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-8
    return arr / denom


def _prescription(neighbor_scores: np.ndarray, global_mean_abs: float, neg_thresh: float, zero_thresh: float) -> Dict[str, object]:
    if neighbor_scores.size == 0:
        return {
            "type": "exploratory",
            "reason": "No nearby train episodes; treat as OOD.",
            "instruction": "Collect demonstrations that mirror the failure viewpoint and approach, especially unseen perspectives.",
        }
    mean_signed = float(np.mean(neighbor_scores))
    mean_abs = float(np.mean(np.abs(neighbor_scores)))
    neg_frac = float(np.mean(neighbor_scores < -neg_thresh)) if neighbor_scores.size > 0 else 0.0
    gap = mean_abs - global_mean_abs

    if neg_frac >= 0.5 and mean_signed < -neg_thresh:
        return {
            "type": "corrective",
            "gap": gap,
            "reason": f"{neg_frac:.2f} of neighbors are strongly harmful.",
            "instruction": "Re-collect crisp, unambiguous demonstrations in this state region; avoid background shortcuts.",
        }
    if mean_abs < zero_thresh:
        return {
            "type": "exploratory",
            "gap": gap,
            "reason": "Neighbor influences are near zero; likely under-covered.",
            "instruction": "Add trajectories that match the failure pose/approach to fill missing coverage.",
        }
    return {
        "type": "consolidating",
        "gap": gap,
        "reason": "Mixed or low-confidence influence; improve signal-to-noise.",
        "instruction": "Collect a diverse set of successful attempts around this failure to reduce variance.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute CUPID influence flow + prescriptions.")
    parser.add_argument("--cluster_file", type=Path, required=True, help="failure_clusters.json from cluster_failures.py")
    parser.add_argument("--score_file", type=Path, required=True, help="CUPID score.pkl (stage4 output)")
    parser.add_argument("--output_dir", type=Path, required=True, help="Directory for plots and JSON summaries")
    parser.add_argument("--stage1_path", type=Path, default=None, help="Optional stage1_train_projections.npy override")
    parser.add_argument("--eval_dir", type=Path, default=None, help="Override eval rollout dir (defaults to score config)")
    parser.add_argument("--pretrained_dir", type=Path, default="/workspace/liushuo/cupid/pretrained_model", help="Override policy weights (defaults to score config)")
    parser.add_argument("--train_dataset_path", type=Path, default=None, help="Override train dataset (defaults to score config)")
    parser.add_argument("--include_noise_cluster", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--neighbor_k", type=int, default=5, help="Top-K nearest train episodes per failure (cosine)")
    parser.add_argument("--neg_threshold", type=float, default=0.05, help="Influence magnitude to treat as harmful")
    parser.add_argument("--zero_threshold", type=float, default=0.02, help="Abs influence threshold for zero-density gap")
    parser.add_argument("--device", type=str, default=None, help="Device override (cpu/cuda:0)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    with args.score_file.open("rb") as f:
        score_payload = pickle.load(f)

    cfg = score_payload["config"]
    train_dataset_path = args.train_dataset_path or Path(cfg["train_dataset_path"])
    eval_dir = args.eval_dir or Path(cfg["eval_rollout_dir"])
    pretrained_dir = args.pretrained_dir or Path(cfg["pretrained_dir"])
    device = torch.device(args.device or cfg.get("device", "cpu"))

    stage1_path = args.stage1_path or (args.score_file.parent / "score.stage1_train_projections.npy")
    if stage1_path.exists():
        train_projections = np.load(stage1_path, allow_pickle=False)
    else:
        train_projections = score_payload["train_projections"]

    episode_lengths = _read_episode_lengths(train_dataset_path)
    episode_ids = _episode_indices_from_lengths(episode_lengths)
    train_episode_feats = _aggregate_episode_means(train_projections, episode_ids)

    calc_cfg = CupidActInfluenceConfig(**cfg)
    calc_cfg.device = str(device)
    calc_cfg.pretrained_dir = str(pretrained_dir)
    calc_cfg.train_dataset_path = str(train_dataset_path)
    calc_cfg.eval_rollout_dir = str(eval_dir)
    calc = CupidActInfluenceCalculator(calc_cfg)
    calc.device = device
    calc.policy.to(device)

    cluster_data = _load_json(args.cluster_file)
    label_map: Dict[str, int] = {k: int(v["cluster"]) for k, v in cluster_data.get("clusters", {}).items()}
    clusters = sorted(set(label_map.values()))
    if not args.include_noise_cluster:
        clusters = [c for c in clusters if c != -1]

    if (eval_dir / "metadata.yaml").exists() or list(eval_dir.glob("ep*.pkl")):
        failures_dir = eval_dir
    else:
        failures_dir = eval_dir / "episodes"

    summary: Dict[str, object] = {
        "config": cfg,
        "train_dataset_path": str(train_dataset_path),
        "eval_dir": str(eval_dir),
        "clusters": {},
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for cl in clusters:
        keep = [name for name, c in label_map.items() if c == cl]
        if len(keep) == 0:
            continue

        dataset = EvalPklEpisodeDataset(dataset_path=failures_dir, exec_horizon=1, sample_history=0)
        dataset = _filter_eval_dataset(dataset, keep)
        calc.eval_dataset = dataset
        calc.eval_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
        calc.eval_rewards = torch.tensor(_episode_rewards(dataset), device=device, dtype=torch.float32)

        return_grad = calc._compute_return_gradient()
        inv_hvp = calc._solve_inverse_hvp(return_grad)
        scores = calc._score_influences(train_projections, inv_hvp)
        global_mean_abs = float(np.mean(np.abs(scores)))

        failure_proj = _compute_episode_projected_grads(calc, dataset)
        norm_train = _normalize_rows(train_episode_feats)
        norm_fail = _normalize_rows(failure_proj)
        sim = norm_fail @ norm_train.T
        neighbor_idx = np.argsort(sim, axis=1)[:, ::-1][:, : max(1, args.neighbor_k)]
        neighbor_scores = scores[neighbor_idx]

        prescription = _prescription(neighbor_scores.flatten(), global_mean_abs, args.neg_threshold, args.zero_threshold)

        feats = np.concatenate([train_episode_feats, failure_proj], axis=0)
        coords = _fit_2d(feats, seed=args.seed)
        plot_path = args.output_dir / f"cluster_{cl}_influence.png"
        _visualize(
            coords,
            train_episode_feats.shape[0],
            scores,
            failure_labels=keep,
            failure_clusters=[cl] * len(keep),
            output_path=plot_path,
        )

        summary["clusters"][str(cl)] = {
            "failures": keep,
            "prescription": prescription,
            "top_harmful_train_episodes": _topk(scores, k=10),
            "neighbor_indices": neighbor_idx.tolist(),
            "return_gradient_norm": float(return_grad.norm().item()),
            "plot": str(plot_path),
        }

    with (args.output_dir / "influence_flow_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote influence workflow outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
