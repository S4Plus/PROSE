
# Allow execution from any working directory without installing this checkout.
import sys as _prose_sys
from pathlib import Path as _ProsePath
_prose_sys.path.insert(0, str(_ProsePath(__file__).resolve().parents[2]))

import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from act.train_trak_act import load_act_policy


def _load_failure_episodes(eval_dir: Path) -> List[Path]:
    """Return sorted episode paths filtered to only failures."""

    ep_dir = eval_dir / "episodes"
    meta_path = ep_dir / "metadata.yaml"
    metadata = yaml.safe_load(meta_path.read_text())
    success_mask = metadata["episode_successes"]

    ep_files = sorted(ep_dir.glob("ep*.pkl"))
    if len(ep_files) != len(success_mask):
        raise ValueError(
            f"Episode count ({len(ep_files)}) and success_mask length "
            f"({len(success_mask)}) mismatch."
        )
    return [p for p, is_success in zip(ep_files, success_mask) if not is_success]


def _load_episode_split(eval_dir: Path) -> Tuple[List[Path], List[Path]]:
    """Split evaluation episodes into success and failure sets."""

    ep_dir = eval_dir / "episodes"
    meta_path = ep_dir / "metadata.yaml"
    metadata = yaml.safe_load(meta_path.read_text())
    success_mask = metadata["episode_successes"]

    ep_files = sorted(ep_dir.glob("ep*.pkl"))
    if len(ep_files) != len(success_mask):
        raise ValueError(
            f"Episode count ({len(ep_files)}) and success_mask length "
            f"({len(success_mask)}) mismatch."
        )

    successes: List[Path] = []
    failures: List[Path] = []
    for path, is_success in zip(ep_files, success_mask):
        (successes if is_success else failures).append(path)
    return successes, failures


def _prepare_batch(row: pd.Series, device: torch.device) -> Dict[str, torch.Tensor]:
    """Convert one dataframe row into an ACTPolicy-friendly batch."""

    obs = row["obs"]

    def _img_to_tensor(img: np.ndarray) -> torch.Tensor:
        # Convert HWC uint8 -> CHW float in [0, 1].
        tensor = torch.as_tensor(img, dtype=torch.float32, device=device)
        if tensor.ndim == 3 and tensor.shape[-1] == 3:
            tensor = tensor.permute(2, 0, 1)
        tensor = tensor / 255.0
        return tensor.unsqueeze(0)

    batch = {
        "observation.images.head_camera": _img_to_tensor(obs["head_camera"]),
        "observation.images.right_wrist_camera": _img_to_tensor(obs["right_wrist_camera"]),
        "observation.state": torch.as_tensor(
            obs["joint_state"], dtype=torch.float32, device=device
        ).unsqueeze(0),
    }
    return batch


def _extract_end_effector(
    obs: Dict[str, object], ee_key: Optional[str], ee_components: int
) -> Optional[torch.Tensor]:
    """Retrieve and standardize end-effector features."""

    if not ee_key:
        return None
    if ee_key not in obs:
        return None

    arr = np.asarray(obs[ee_key])
    if arr.size == 0:
        return None

    flat = torch.as_tensor(arr, dtype=torch.float32).reshape(-1)
    if flat.numel() < ee_components:
        padded = torch.zeros(ee_components, dtype=torch.float32)
        padded[: flat.numel()] = flat
        flat = padded
    else:
        flat = flat[:ee_components]
    return flat


@torch.no_grad()
def _encode_step(
    policy: torch.nn.Module, batch: Dict[str, torch.Tensor]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run one step through ACT encoder and return pooled tokens + raw state."""

    encoder_out: List[torch.Tensor] = []

    def _hook(_, __, output):
        encoder_out.append(output.detach().cpu())

    handle = policy.model.encoder.register_forward_hook(_hook)
    _ = policy.predict_action_chunk(batch)  # forward pass
    handle.remove()

    if not encoder_out:
        raise RuntimeError("Encoder hook did not capture any output.")

    tokens = encoder_out.pop()  # (seq, batch, dim)
    pooled = tokens.mean(dim=0).squeeze(0)  # (dim_model,)
    state = batch["observation.state"].cpu().squeeze(0)  # (7,)
    return pooled, state


@torch.no_grad()
def compute_failure_embeddings(
    failure_paths: Sequence[Path],
    pretrained_dir: Path,
    device: torch.device,
    *,
    include_joint_state: bool = True,
    ee_key: Optional[str] = None,
    ee_components: int = 3,
) -> Tuple[np.ndarray, List[str]]:
    """Embed all failure episodes and return (N, D) array + episode names."""

    policy, _ = load_act_policy(pretrained_dir, device)
    policy.to(device)
    policy.eval()

    embeddings: List[np.ndarray] = []
    names: List[str] = []

    for ep_path in failure_paths:
        with open(ep_path, "rb") as f:
            episode = pickle.load(f)

        step_embs: List[torch.Tensor] = []
        step_states: List[torch.Tensor] = []
        step_ee: List[torch.Tensor] = []
        for _, row in episode.iterrows():
            batch = _prepare_batch(row, device)
            pooled_tokens, state = _encode_step(policy, batch)
            step_embs.append(pooled_tokens)
            step_states.append(state)
            ee_vec = _extract_end_effector(row["obs"], ee_key, ee_components)
            if ee_vec is not None:
                step_ee.append(ee_vec)

        enc_mean = torch.stack(step_embs).mean(dim=0)

        parts: List[torch.Tensor] = [enc_mean]
        if include_joint_state:
            parts.append(torch.stack(step_states).mean(dim=0))
        if ee_key:
            if step_ee:
                parts.append(torch.stack(step_ee).mean(dim=0))
            else:
                parts.append(torch.zeros(ee_components, dtype=torch.float32))

        embeddings.append(torch.cat(parts, dim=0).numpy())
        names.append(ep_path.name)

    return np.vstack(embeddings), names


def cluster_failures(
    embeddings: np.ndarray, eps: float = 0.8, min_samples: int = 2
) -> Tuple[np.ndarray, PCA]:
    """Run DBSCAN on PCA-compressed embeddings with a few fallbacks."""

    scaler = StandardScaler()
    scaled = scaler.fit_transform(embeddings)

    n_components = min(max(2, embeddings.shape[0]), min(16, scaled.shape[1]))
    pca = PCA(n_components=n_components, random_state=0)
    reduced = pca.fit_transform(scaled)

    for multiplier in (1.0, 1.5, 2.0, 3.0):
        cur_eps = eps * multiplier
        labels = DBSCAN(eps=cur_eps, min_samples=min_samples).fit_predict(reduced)
        if not (labels == -1).all():
            return labels, pca

    # If still all noise, return last attempt (all -1) for transparency.
    return labels, pca


def main():
    parser = argparse.ArgumentParser(description="Cluster failure rollouts with ACT embeddings.")
    parser.add_argument("--eval_dir", type=Path, default=Path("eval_belt_pick_bottle"))
    parser.add_argument("--pretrained_dir", type=Path, default=Path("pretrained_model"))
    parser.add_argument("--eps", type=float, default=0.8)
    parser.add_argument("--min_samples", type=int, default=2)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--ee_key", type=str, default=None, help="Obs key for end-effector pose.")
    parser.add_argument("--ee_components", type=int, default=3, help="How many EE dims to keep.")
    parser.add_argument(
        "--include_joint_state",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Append mean joint_state to embeddings.",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("eval_belt_pick_bottle/failure_clusters.json")
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    successes, failures = _load_episode_split(args.eval_dir)

    if failures:
        embeddings, names = compute_failure_embeddings(
            failures,
            args.pretrained_dir,
            device,
            include_joint_state=args.include_joint_state,
            ee_key=args.ee_key,
            ee_components=args.ee_components,
        )
        labels, pca = cluster_failures(embeddings, eps=args.eps, min_samples=args.min_samples)
    else:
        names = []
        labels = np.array([], dtype=int)
        pca = None

    summary: Dict[str, Dict[str, object]] = {}
    for name, label in zip(names, labels):
        summary[name] = {"cluster": int(label), "is_noise": bool(label == -1)}

    cluster_sizes: Dict[int, int] = {}
    for label in labels:
        cluster_sizes[int(label)] = cluster_sizes.get(int(label), 0) + 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(
            {
                "eps": args.eps,
                "min_samples": args.min_samples,
                "pca_components": getattr(pca, "n_components_", 0),
                "include_joint_state": args.include_joint_state,
                "ee_key": args.ee_key,
                "ee_components": args.ee_components,
                "success_episodes": [p.name for p in successes],
                "failure_episodes": [p.name for p in failures],
                "clusters": summary,
                "cluster_sizes": cluster_sizes,
            },
            f,
            indent=2,
        )

    print("Outcome split:")
    print(f"  Successes: {len(successes)}")
    print(f"  Failures: {len(failures)}")
    if summary:
        print("\nCluster assignments:")
        for name, label in summary.items():
            print(f"  {name}: cluster={label['cluster']}")
        print(f"\nCluster sizes: {cluster_sizes}")
    else:
        print("  No failure episodes to cluster.")
    print(f"\nSaved details to {args.output}")


if __name__ == "__main__":
    main()
