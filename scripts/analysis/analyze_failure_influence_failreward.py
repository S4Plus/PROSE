
# Allow execution from any working directory without installing this checkout.
import sys as _prose_sys
from pathlib import Path as _ProsePath
_prose_sys.path.insert(0, str(_ProsePath(__file__).resolve().parents[2]))

import argparse
import json
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
    import umap

    HAS_UMAP = True
except Exception:
    HAS_UMAP = False


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
    ids: List[int] = []
    for ep_id, length in enumerate(lengths):
        ids.extend([ep_id] * int(length))
    return np.asarray(ids, dtype=np.int64)


def _filter_eval_dataset(dataset: EvalPklEpisodeDataset, keep_names: Iterable[str]) -> EvalPklEpisodeDataset:
    """Restrict EvalPklEpisodeDataset to a whitelist of filenames."""

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
    if HAS_UMAP:
        reducer = umap.UMAP(
            n_components=2,
            random_state=seed,
            min_dist=0.1,
            n_neighbors=min(30, max(5, features.shape[0] // 10)),
            metric="cosine",
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
    train_coords: np.ndarray,
    influences: np.ndarray,
    fail_coords: np.ndarray,
    fail_labels: List[str],
    fail_clusters: List[int],
    neighbor_lines: List[Tuple[int, List[int]]],
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    vmin = float(np.min(influences))
    vmax = float(np.max(influences))
    if vmax < vmin:
        vmin, vmax = vmax, vmin
    if abs(vmax - vmin) < 1e-9:
        vmax = vmin + 1e-3
    norm = mcolors.TwoSlopeNorm(vcenter=0.0, vmin=vmin, vmax=vmax) if vmin < 0 < vmax else mcolors.Normalize(vmin=vmin, vmax=vmax)

    plt.figure(figsize=(9, 6))
    plt.scatter(train_coords[:, 0], train_coords[:, 1], c=influences, cmap="coolwarm", norm=norm, s=18, alpha=0.9, label="train")

    if len(fail_coords) > 0:
        cmap = plt.get_cmap("tab10")
        clusters = sorted(set(fail_clusters))
        for cl in clusters:
            idxs = [i for i, c in enumerate(fail_clusters) if c == cl]
            if not idxs:
                continue
            pts = fail_coords[idxs]
            plt.scatter(
                pts[:, 0],
                pts[:, 1],
                c=[cmap(cl % 10)],
                edgecolor="k",
                s=60,
                marker="o",
                label=f"fail cluster {cl}",
            )
        for pt, name in zip(fail_coords, fail_labels):
            plt.annotate(name.replace(".pkl", ""), (pt[0], pt[1]), fontsize=6, alpha=0.7)
        # Draw neighbor lines.
        for f_idx, neigh_idxs in neighbor_lines:
            fp = fail_coords[f_idx]
            for n_idx in neigh_idxs:
                tp = train_coords[n_idx]
                plt.plot([fp[0], tp[0]], [fp[1], tp[1]], color="gray", alpha=0.3, linewidth=0.7)

    plt.legend()
    plt.title("Train vs failure (reward=-1 influences)")
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


def main():
    parser = argparse.ArgumentParser(description="Influence scores w/ failure reward = -1.")
    parser.add_argument("--cluster_file", type=Path, required=True, help="failure_clusters.json")
    parser.add_argument("--score_file", type=Path, required=True, help="CUPID-ACT score.pkl path")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--top_train", type=int, default=200, help="Plot top-N train episodes by |influence|.")
    parser.add_argument("--neighbor_k", type=int, default=5, help="High-dim cosine nearest neighbors from failures to train.")
    args = parser.parse_args()

    import pickle

    with args.score_file.open("rb") as f:
        score_payload = pickle.load(f)

    cfg = score_payload["config"]
    train_dataset_path = Path(cfg["train_dataset_path"])
    eval_dir = Path(cfg["eval_rollout_dir"])
    pretrained_dir = Path(cfg["pretrained_dir"])
    device = torch.device(args.device or cfg.get("device", "cpu"))

    # Resolve episodes dir.
    if (eval_dir / "metadata.yaml").exists() or list(eval_dir.glob("ep*.pkl")):
        failures_dir = eval_dir
    else:
        failures_dir = eval_dir / "episodes"

    # Load train projections (stage1).
    stage1_path = args.score_file.parent / "score.stage1_train_projections.npy"
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

    # Use all failure episodes in cluster file.
    with args.cluster_file.open("r") as f:
        cluster_data = json.load(f)
    failure_names = cluster_data.get("failure_episodes", list(cluster_data.get("clusters", {}).keys()))
    dataset = EvalPklEpisodeDataset(dataset_path=failures_dir, exec_horizon=1, sample_history=0)
    dataset = _filter_eval_dataset(dataset, failure_names)
    calc.eval_dataset = dataset
    calc.eval_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    # Set all failure rewards to -1 (no centering).
    calc.eval_rewards = torch.full((len(dataset._episode_lengths),), -1.0, device=device, dtype=torch.float32)  # type: ignore[attr-defined]

    return_grad = _compute_constant_return_gradient(calc, reward_value=-1.0)
    inv_hvp = calc._solve_inverse_hvp(return_grad)
    scores = calc._score_influences(train_projections, inv_hvp)

    # Failure projections for visualization.
    failure_proj = _compute_episode_projected_grads(calc, dataset)

    # Select train subset: top-N by |influence| plus neighbors of failures in high-dim cosine space.
    abs_scores = np.abs(scores)
    top_n = min(args.top_train, len(abs_scores))
    top_idx = np.argsort(abs_scores)[::-1][:top_n]

    def _norm_rows(arr: np.ndarray) -> np.ndarray:
        denom = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-8
        return arr / denom

    norm_train = _norm_rows(train_episode_feats)
    norm_fail = _norm_rows(failure_proj)
    sim = norm_fail @ norm_train.T  # cosine similarity
    neighbor_lines: List[Tuple[int, List[int]]] = []
    neighbor_idx: List[int] = []
    for fi in range(sim.shape[0]):
        order = np.argsort(sim[fi])[::-1][: max(1, args.neighbor_k)]
        neighbor_lines.append((fi, order.tolist()))
        neighbor_idx.extend(order.tolist())

    keep_train = np.unique(np.concatenate([top_idx, np.asarray(neighbor_idx, dtype=np.int64)]))
    train_subset = train_episode_feats[keep_train]
    scores_subset = scores[keep_train]

    feats = np.concatenate([train_subset, failure_proj], axis=0)
    coords = _fit_2d(feats, seed=args.seed)
    train_coords = coords[: train_subset.shape[0]]
    fail_coords = coords[train_subset.shape[0] :]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # Map clusters for failure labels.
    with args.cluster_file.open("r") as f:
        cluster_data = json.load(f)
    label_map: Dict[str, int] = {k: int(v["cluster"]) for k, v in cluster_data.get("clusters", {}).items()}
    fail_clusters = [label_map.get(name, -1) for name in failure_names]

    # Remap neighbor indices to the subset index space.
    subset_index_map: Dict[int, int] = {orig: i for i, orig in enumerate(keep_train.tolist())}
    remapped_lines: List[Tuple[int, List[int]]] = []
    for fi, neighs in neighbor_lines:
        remapped = [subset_index_map[n] for n in neighs if n in subset_index_map]
        remapped_lines.append((fi, remapped))

    _visualize(
        train_coords,
        scores_subset,
        fail_coords,
        failure_names,
        fail_clusters,
        remapped_lines,
        args.output_dir / "train_influence_failreward.png",
    )

    summary = {
        "failure_reward": -1.0,
        "top_harmful_train_episodes": _topk(scores, k=20),
        "scores_per_episode": scores.tolist(),
        "return_gradient_norm": float(return_grad.norm().item()),
    }
    with (args.output_dir / "cluster_influence_failreward.json").open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved failure-reward influence summary to {args.output_dir}")


if __name__ == "__main__":
    main()
