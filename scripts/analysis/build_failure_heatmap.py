
# Allow execution from any working directory without installing this checkout.
import sys as _prose_sys
from pathlib import Path as _ProsePath
_prose_sys.path.insert(0, str(_ProsePath(__file__).resolve().parents[2]))

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

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
    import umap

    HAS_UMAP = True
except Exception:  # pragma: no cover - optional dependency
    HAS_UMAP = False


def _load_json(path: Path):
    with path.open("r") as f:
        return json.load(f)


def _read_episode_lengths(train_root: Path) -> List[int]:
    """Parse HF-style episodes.jsonl to recover per-episode lengths."""

    lengths: List[int] = []
    meta_path = train_root / "meta" / "episodes.jsonl"
    with meta_path.open("r") as f:
        for line in f:
            record = json.loads(line)
            lengths.append(int(record["length"]))
    return lengths


def _episode_indices_from_lengths(lengths: Sequence[int]) -> np.ndarray:
    """Map sample index -> episode id given per-episode lengths."""

    ids: List[int] = []
    for ep_id, length in enumerate(lengths):
        ids.extend([ep_id] * int(length))
    return np.asarray(ids, dtype=np.int64)


def _filter_eval_dataset(
    dataset: EvalPklEpisodeDataset, keep_names: Iterable[str]
) -> EvalPklEpisodeDataset:
    """Restrict an EvalPklEpisodeDataset to a whitelist of filenames."""

    keep = set(keep_names)
    mask = [Path(p).name in keep for p in dataset.episode_files]
    dataset.episode_files = [p for p, m in zip(dataset.episode_files, mask) if m]
    dataset._episodes = [ep for ep, m in zip(dataset._episodes, mask) if m]  # type: ignore[attr-defined]
    dataset._episode_lengths = [l for l, m in zip(dataset._episode_lengths, mask) if m]  # type: ignore[attr-defined]
    dataset._episode_successes = [s for s, m in zip(dataset._episode_successes, mask) if m]  # type: ignore[attr-defined]
    dataset._length = sum(dataset._episode_lengths)  # type: ignore[attr-defined]
    return dataset


def _aggregate_episode_means(sample_array: np.ndarray, episode_ids: np.ndarray) -> np.ndarray:
    """Average per-sample features into per-episode features."""

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
    """Project per-episode gradients into the same subspace as train_projections."""

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


def _compute_constant_return_gradient(calc: CupidActInfluenceCalculator, reward_value: float) -> torch.Tensor:
    """Compute return gradient using constant reward per episode (no centering)."""

    lengths = calc.eval_dataset.episode_lengths
    if len(lengths) == 0:
        return torch.zeros(calc.vectorizer.total_dim, device=calc.device, dtype=torch.float32)

    return_grad = torch.zeros(calc.vectorizer.total_dim, device=calc.device, dtype=torch.float32)
    episode_grad = torch.zeros_like(return_grad)
    ep_idx = 0
    steps_remaining = lengths[ep_idx]
    for grad in calc._iter_eval_gradients():
        episode_grad += grad
        steps_remaining -= 1
        if steps_remaining == 0:
            return_grad += reward_value * episode_grad
            ep_idx += 1
            if ep_idx >= len(lengths):
                break
            episode_grad.zero_()
            steps_remaining = lengths[ep_idx]
    return return_grad / max(len(lengths), 1)


def _fit_2d(features: np.ndarray, seed: int = 0) -> np.ndarray:
    """Run UMAP (preferred) or t-SNE to 2D."""

    if HAS_UMAP:
        reducer = umap.UMAP(
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


def _smooth_grid(grid: np.ndarray, sigma: float) -> np.ndarray:
    """Separable Gaussian smoothing (fallback when scipy is unavailable)."""

    if sigma <= 0.0 or grid.size == 0:
        return grid

    radius = max(1, int(3 * sigma))
    ax = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-(ax**2) / (2.0 * sigma * sigma))
    kernel /= max(kernel.sum(), 1e-8)

    smoothed = grid.copy()
    # Convolve over x then y.
    smoothed = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode="same"), 0, smoothed)
    smoothed = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode="same"), 1, smoothed)
    return smoothed


def _categorize(grid: np.ndarray, neg_thresh: Optional[float], pos_thresh: Optional[float]):
    """
    Label each grid cell:
    2 -> 极度不自信 (very unconfident / harmful influence)
    1 -> 不怎么自信 (somewhat unconfident / neutral)
    0 -> 很自信 (confident / helpful)
    -1 -> 无数据 (empty bin)
    """

    valid = grid[~np.isnan(grid)]
    if valid.size == 0:
        return np.full_like(grid, -1, dtype=np.int32), 0.0, 0.0

    if neg_thresh is None or pos_thresh is None:
        q_low, q_high = np.quantile(valid, [0.33, 0.66])
        neg_thresh = q_low if neg_thresh is None else neg_thresh
        pos_thresh = q_high if pos_thresh is None else pos_thresh
        if pos_thresh <= neg_thresh:
            pos_thresh = neg_thresh + 1e-6

    labels = np.full_like(grid, -1, dtype=np.int32)
    labels[grid >= pos_thresh] = 2
    mid_mask = (grid > neg_thresh) & (grid < pos_thresh)
    labels[mid_mask] = 1
    labels[grid <= neg_thresh] = 0
    labels[np.isnan(grid)] = -1
    return labels, float(neg_thresh), float(pos_thresh)


def _grid_cell_centers(edges: Tuple[np.ndarray, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Compute 1D center coordinates for each grid bin along x and y."""

    x_edges, y_edges = edges
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    return x_centers, y_centers


def _select_cells_for_collection(
    grid: np.ndarray,
    edges: Tuple[np.ndarray, np.ndarray],
    budget: Optional[int],
    threshold: Optional[float],
    prefer_low: bool,
) -> List[Dict]:
    """
    Pick grid cells for new data collection given a score heatmap.

    - prefer_low=True matches the user story: lower scores mean the model is unsure.
    - threshold filters cells first; budget optionally caps how many to return.
    """

    valid_mask = ~np.isnan(grid)
    if not valid_mask.any():
        return []

    scores = grid[valid_mask]
    idxs = np.argwhere(valid_mask)
    if threshold is not None:
        keep_mask = scores <= threshold if prefer_low else scores >= threshold
        scores = scores[keep_mask]
        idxs = idxs[keep_mask]
        if scores.size == 0:
            return []

    order = np.argsort(scores) if prefer_low else np.argsort(-scores)
    idxs = idxs[order]
    scores = scores[order]
    if budget is not None and budget > 0:
        idxs = idxs[:budget]
        scores = scores[:budget]

    x_centers, y_centers = _grid_cell_centers(edges)
    x_sizes = edges[0][1:] - edges[0][:-1]
    y_sizes = edges[1][1:] - edges[1][:-1]

    selected: List[Dict] = []
    for (ix, iy), score in zip(idxs, scores):
        selected.append(
            {
                "index": [int(ix), int(iy)],
                "score": float(score),
                "center": [float(x_centers[int(ix)]), float(y_centers[int(iy)])],
                "size": [float(x_sizes[int(ix)]), float(y_sizes[int(iy)])],
            }
        )
    return selected


def _sample_points_within_cells(
    cells: Sequence[Dict],
    samples_per_cell: int,
    jitter_frac: float,
    seed: Optional[int],
) -> List[Dict]:
    """Generate physical sampling points using cell centers plus uniform jitter."""

    if samples_per_cell <= 0 or len(cells) == 0:
        return []
    rng = np.random.default_rng(seed)
    frac = max(0.0, jitter_frac)
    samples: List[Dict] = []
    for cell in cells:
        cx, cy = cell["center"]
        sx, sy = cell["size"]
        half_jitter_x = 0.5 * sx * frac
        half_jitter_y = 0.5 * sy * frac
        for _ in range(samples_per_cell):
            dx = rng.uniform(-half_jitter_x, half_jitter_x)
            dy = rng.uniform(-half_jitter_y, half_jitter_y)
            samples.append(
                {
                    "from_cell": cell["index"],
                    "score": cell["score"],
                    "position": [float(cx + dx), float(cy + dy)],
                }
            )
    return samples


def _build_heatmap(coords: np.ndarray, scores: np.ndarray, grid_size: int, sigma: float):
    """Bin scores onto a 2D grid in the embedding space and smooth."""

    if coords.shape[0] == 0:
        raise ValueError("No coordinates provided for heatmap.")
    x_min, x_max = float(coords[:, 0].min()), float(coords[:, 0].max())
    y_min, y_max = float(coords[:, 1].min()), float(coords[:, 1].max())
    margin_x = 0.05 * max(abs(x_min), abs(x_max), 1.0)
    margin_y = 0.05 * max(abs(y_min), abs(y_max), 1.0)
    x_min, x_max = x_min - margin_x, x_max + margin_x
    y_min, y_max = y_min - margin_y, y_max + margin_y

    heat_sum, x_edges, y_edges = np.histogram2d(
        coords[:, 0],
        coords[:, 1],
        bins=grid_size,
        range=[[x_min, x_max], [y_min, y_max]],
        weights=scores,
    )
    heat_count, _, _ = np.histogram2d(
        coords[:, 0],
        coords[:, 1],
        bins=grid_size,
        range=[[x_min, x_max], [y_min, y_max]],
    )
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_grid = heat_sum / np.maximum(heat_count, 1e-8)
        mean_grid[heat_count == 0] = np.nan

    smoothed = _smooth_grid(mean_grid, sigma=sigma)
    return smoothed, (x_edges, y_edges)


def _visualize_heatmaps(
    raw_grid: np.ndarray,
    labels: np.ndarray,
    edges: Tuple[np.ndarray, np.ndarray],
    train_coords: np.ndarray,
    fail_coords: np.ndarray,
    output_dir: Path,
) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    x_edges, y_edges = edges
    extent = [x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]]

    plt.figure(figsize=(8, 6))
    vmin = float(np.nanmin(raw_grid)) if not np.isnan(raw_grid).all() else -1.0
    vmax = float(np.nanmax(raw_grid)) if not np.isnan(raw_grid).all() else 1.0
    if math.isclose(vmin, vmax, rel_tol=1e-6, abs_tol=1e-9):
        vmax = vmin + 1e-3
    norm = mcolors.TwoSlopeNorm(vcenter=0.0, vmin=vmin, vmax=vmax) if vmin < 0 < vmax else mcolors.Normalize(vmin=vmin, vmax=vmax)
    plt.imshow(raw_grid.T, origin="lower", extent=extent, cmap="seismic", norm=norm, aspect="auto")
    plt.scatter(train_coords[:, 0], train_coords[:, 1], s=12, c="k", alpha=0.35, label="train episodes")
    if len(fail_coords) > 0:
        plt.scatter(fail_coords[:, 0], fail_coords[:, 1], s=120, edgecolor="k", facecolor="none", label="failures")
    plt.colorbar(label="Influence (harmful → helpful)")
    plt.title("Failure-driven Influence Heatmap")
    plt.legend()
    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "failure_heatmap_raw.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 6))
    cmap = mcolors.ListedColormap(["#1fb141", "#ffb100", "#d62728", "#b8b8b8"])
    bounds = [-1.5, -0.5, 0.5, 1.5, 2.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    plt.imshow(labels.T, origin="lower", extent=extent, cmap=cmap, norm=norm, aspect="auto")
    plt.scatter(train_coords[:, 0], train_coords[:, 1], s=12, c="k", alpha=0.35)
    if len(fail_coords) > 0:
        plt.scatter(fail_coords[:, 0], fail_coords[:, 1], s=120, edgecolor="k", facecolor="none")
    handles = [
        plt.Rectangle((0, 0), 1, 1, color="#1fb141"),
        plt.Rectangle((0, 0), 1, 1, color="#ffb100"),
        plt.Rectangle((0, 0), 1, 1, color="#d62728"),
        plt.Rectangle((0, 0), 1, 1, color="#b8b8b8"),
    ]
    plt.legend(handles, ["Confident (helpful)", "Uncertain", "Highly uncertain (harmful)", "No data"], loc="upper right")
    plt.title("Confidence Classes from Failure Influence")
    plt.tight_layout()
    plt.savefig(output_dir / "failure_heatmap_classes.png", dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Build a heatmap from failure rollouts and influence scores.")
    parser.add_argument("--cluster_file", type=Path, default=None, help="Optional failure_clusters.json; if absent, use all fail episodes.")
    parser.add_argument("--score_file", type=Path, required=True, help="CUPID-ACT score.pkl path")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--stage1_path", type=Path, default=None, help="Optional override for stage1_train_projections.npy")
    parser.add_argument("--eval_dir", type=Path, default=None, help="Eval rollout dir; defaults to score config.")
    parser.add_argument("--pretrained_dir", type=Path, default=None, help="Override policy path; defaults to score config.")
    parser.add_argument("--train_dataset_path", type=Path, default=None, help="Override train dataset; defaults to score config.")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--grid_size", type=int, default=80, help="Number of bins per axis for the heatmap.")
    parser.add_argument("--sigma", type=float, default=1.5, help="Smoothing sigma (in grid bins).")
    parser.add_argument("--neg_thresh", type=float, default=None, help="Confidence threshold for helpful cells (optional).")
    parser.add_argument("--pos_thresh", type=float, default=None, help="Unconfidence threshold for harmful cells (optional).")
    parser.add_argument(
        "--collection_budget",
        type=int,
        default=None,
        help="Optional: max number of grid cells to collect data from (after thresholding).",
    )
    parser.add_argument(
        "--collection_threshold",
        type=float,
        default=None,
        help="Optional: treat cells below this score as uncertain (override budget ranking).",
    )
    parser.add_argument(
        "--uncertain_direction",
        type=str,
        choices=["low", "high"],
        default="low",
        help="Interpretation of the heatmap: 'low' means low scores are bad and should be collected.",
    )
    parser.add_argument(
        "--samples_per_cell",
        type=int,
        default=3,
        help="How many physical samples to draw per selected grid cell.",
    )
    parser.add_argument(
        "--cell_jitter_frac",
        type=float,
        default=0.35,
        help="Jitter range as a fraction of cell width/height when sampling inside a grid cell.",
    )
    args = parser.parse_args()

    import pickle

    with args.score_file.open("rb") as f:
        score_payload = pickle.load(f)

    cfg = score_payload["config"]
    train_dataset_path = args.train_dataset_path or Path(cfg["train_dataset_path"])
    eval_dir = args.eval_dir or Path(cfg["eval_rollout_dir"])
    pretrained_dir = args.pretrained_dir or Path(cfg["pretrained_dir"])
    device = torch.device(args.device or cfg.get("device", "cpu"))

    # Resolve episodes dir.
    if (eval_dir / "metadata.yaml").exists() or list(eval_dir.glob("ep*.pkl")):
        failures_dir = eval_dir
    else:
        failures_dir = eval_dir / "episodes"

    # Load train projections (stage1).
    stage1_path = args.stage1_path or (args.score_file.parent / "score.stage1_train_projections.npy")
    if stage1_path.exists():
        train_projections = np.load(stage1_path, allow_pickle=False)
    else:
        train_projections = score_payload["train_projections"]

    # Episode indices -> per-episode features.
    episode_lengths = _read_episode_lengths(train_dataset_path)
    episode_ids = _episode_indices_from_lengths(episode_lengths)
    train_episode_feats = _aggregate_episode_means(train_projections, episode_ids)

    # Build calculator with same config.
    calc_cfg = CupidActInfluenceConfig(**cfg)
    calc_cfg.device = str(device)
    calc_cfg.pretrained_dir = str(pretrained_dir)
    calc_cfg.train_dataset_path = str(train_dataset_path)
    calc_cfg.eval_rollout_dir = str(failures_dir)
    calc = CupidActInfluenceCalculator(calc_cfg)
    calc.device = device
    calc.policy.to(device)

    # Load evaluation episodes and choose failure subset.
    dataset = EvalPklEpisodeDataset(dataset_path=failures_dir, exec_horizon=1, sample_history=0)
    all_episode_names = [Path(p).name for p in dataset.episode_files]
    if args.cluster_file and args.cluster_file.exists():
        cluster_data = _load_json(args.cluster_file)
        failure_names = cluster_data.get("failure_episodes", list(cluster_data.get("clusters", {}).keys()))
    else:
        failure_names = [n for n in all_episode_names if "fail" in n.lower()]
    if len(failure_names) == 0:
        failure_names = all_episode_names  # fallback: use all episodes if no fail tags
    dataset = _filter_eval_dataset(dataset, failure_names)
    calc.eval_dataset = dataset
    calc.eval_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    calc.eval_rewards = torch.tensor(_episode_rewards(dataset), device=device, dtype=torch.float32)

    # Influence scores: treat failures as reward=-1 to highlight harmful regions.
    return_grad = _compute_constant_return_gradient(calc, reward_value=-1.0)
    inv_hvp = calc._solve_inverse_hvp(return_grad)
    scores = calc._score_influences(train_projections, inv_hvp)

    # Failure projections for visualization.
    failure_proj = _compute_episode_projected_grads(calc, dataset)

    # 2D embedding.
    feats = np.concatenate([train_episode_feats, failure_proj], axis=0)
    coords = _fit_2d(feats, seed=args.seed)
    train_coords = coords[: train_episode_feats.shape[0]]
    fail_coords = coords[train_episode_feats.shape[0] :]

    # Heatmap + classes.
    heat_grid, edges = _build_heatmap(train_coords, scores, grid_size=args.grid_size, sigma=args.sigma)
    labels, neg_thresh, pos_thresh = _categorize(heat_grid, args.neg_thresh, args.pos_thresh)
    _visualize_heatmaps(heat_grid, labels, edges, train_coords, fail_coords, args.output_dir)

    collection_cells: List[Dict] = []
    collection_samples: List[Dict] = []
    if args.collection_budget is not None or args.collection_threshold is not None:
        collection_cells = _select_cells_for_collection(
            heat_grid,
            edges,
            budget=args.collection_budget,
            threshold=args.collection_threshold,
            prefer_low=args.uncertain_direction == "low",
        )
        collection_samples = _sample_points_within_cells(
            collection_cells,
            samples_per_cell=args.samples_per_cell,
            jitter_frac=args.cell_jitter_frac,
            seed=args.seed,
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "neg_thresh": neg_thresh,
        "pos_thresh": pos_thresh,
        "grid_size": args.grid_size,
        "sigma": args.sigma,
        "train_dataset_path": str(train_dataset_path),
        "eval_dir": str(eval_dir),
        "failure_reward": -1.0,
        "num_failure_episodes": len(failure_names),
        "collection_budget": args.collection_budget,
        "collection_threshold": args.collection_threshold,
        "uncertain_direction": args.uncertain_direction,
        "samples_per_cell": args.samples_per_cell,
        "cell_jitter_frac": args.cell_jitter_frac,
        "selected_cells": len(collection_cells),
        "planned_samples": len(collection_samples),
    }
    with (args.output_dir / "failure_heatmap_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    np.savez_compressed(
        args.output_dir / "failure_heatmap_arrays.npz",
        heat_grid=heat_grid,
        labels=labels,
        x_edges=edges[0],
        y_edges=edges[1],
    )
    if collection_samples:
        plan = {
            "cells": collection_cells,
            "samples": collection_samples,
            "edges": {"x_edges": edges[0].tolist(), "y_edges": edges[1].tolist()},
        }
        with (args.output_dir / "collection_plan.json").open("w") as f:
            json.dump(plan, f, indent=2)

    print(f"Saved failure influence heatmaps to {args.output_dir}")


if __name__ == "__main__":
    main()
