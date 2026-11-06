#!/usr/bin/env python
"""
Train ACT with CUPID-ACT scores using:
1) episode filtering based on bad-window ratio
2) episode reweighting based on window weights or influence scores

Example:

CUDA_VISIBLE_DEVICES=7 nohup python /workspace/liushuo/cupid/act/train_act_cupid_reweight.py \
    --dataset.repo_id ls/hammer_cleanup \
    --dataset.root /workspace/liushuo/cupid/datasets/hammer_cleanup_d0_core \
    --output_dir /workspace/liushuo/cupid/outputs/hammer_cleanup_d0_core_cupid_1\
    --job_name hammer_cleanup \
    --policy.type=act \
    --policy.repo_id ls/hammer_cleanup \
    --score-pkl /workspace/liushuo/cupid/1_infulenece_score/hammer_cleanup_D0/score.pkl \
    --drop-weight-quantile 0.2 \
    --weight-mode window_mean \
    --use-weighted-sampler \
    --segment-mask \
    --segment-weight \
    > ./train2_hammer_cleanup_400_act.log 2>&1 &


CUDA_VISIBLE_DEVICES=7 nohup python /workspace/liushuo/cupid/act/train_act_cupid_reweight.py \
    --dataset.repo_id ls/hammer_cleanup \
    --dataset.root /workspace/liushuo/cupid/datasets/hammer_cleanup_d0_core \
    --output_dir /workspace/liushuo/cupid/3_model/hammer_cleanup_d0_core_cupid\
    --job_name hammer_cleanup \
    --policy.type=act \
    --policy.repo_id ls/hammer_cleanup \
    --score-pkl /workspace/liushuo/cupid/1_infulenece_score/hammer_cleanup_D0_8/score.pkl \
    --drop-weight-quantile 0.2 \
    --weight-mode window_mean \
    --use-weighted-sampler \
    --segment-mask \
    --segment-weight \
    > /workspace/liushuo/cupid/3_model/logs/HammerCleanup_8_act.log 2>&1 &

python /workspace/liushuo/cupid/act/train_act_cupid_reweight.py \
  --dataset-path /workspace/liushuo/cupid/datasets/hammer_cleanup_d0_core \
  --score-pkl /workspace/liushuo/cupid/1_infulenece_score/hammer_cleanup_D0/score.pkl \
  --output-dir /workspace/liushuo/cupid/outputs/hammer_cleanup_d0_core_cupid \
  --drop-bad-ratio 0.3 \
  --weight-mode window_mean \
  --use-weighted-sampler \
  --max-steps 100000

CUDA_VISIBLE_DEVICES=7 nohup python /workspace/liushuo/cupid/act/train_act_cupid_reweight.py \
    --dataset.repo_id ls/can_mh \
    --dataset.root /workspace/liushuo/cupid/datasets/lerobot/can_mh \
    --output_dir /workspace/liushuo/cupid/3_model/can_mh_0.1\
    --job_name can_mh \
    --policy.type=act \
    --policy.repo_id ls/can_mh \
    --score-pkl /workspace/liushuo/cupid/1_infulenece_score/can_mh/score.pkl \
    --drop-weight-quantile 0.1 \
    --weight-mode window_mean \
    --use-weighted-sampler \
    --segment-mask \
    --segment-weight \
    > /workspace/liushuo/cupid/3_model/logs/can_mh_0.1_act.log 2>&1 &

CUDA_VISIBLE_DEVICES=7 nohup python /workspace/liushuo/cupid/act/train_act_cupid_reweight.py \
    --dataset.repo_id ls/can_mh \
    --dataset.root /workspace/liushuo/cupid/datasets/lerobot/can_mh \
    --output_dir /workspace/liushuo/cupid/3_model/can_mh_0.2\
    --job_name can_mh \
    --policy.type=act \
    --policy.repo_id ls/can_mh \
    --score-pkl /workspace/liushuo/cupid/1_infulenece_score/can_mh/score.pkl \
    --drop-weight-quantile 0.2 \
    --weight-mode window_mean \
    --use-weighted-sampler \
    --segment-mask \
    --segment-weight \
    > /workspace/liushuo/cupid/3_model/logs/can_mh_0.2_act.log 2>&1 &


CUDA_VISIBLE_DEVICES=6 nohup python /workspace/liushuo/cupid/act/train_act_cupid_reweight.py \
    --dataset.repo_id ls/can_mh \
    --dataset.root /workspace/liushuo/cupid/datasets/lerobot/can_mh \
    --output_dir /workspace/liushuo/cupid/3_model/can_mh_0.3\
    --job_name can_mh \
    --policy.type=act \
    --policy.repo_id ls/can_mh \
    --score-pkl /workspace/liushuo/cupid/1_infulenece_score/can_mh/score.pkl \
    --drop-weight-quantile 0.3 \
    --weight-mode window_mean \
    --use-weighted-sampler \
    --segment-mask \
    --segment-weight \
    > /workspace/liushuo/cupid/3_model/logs/can_mh_0.3_act.log 2>&1 &
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import sys
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import torch
from torch import nn


REPO_ROOT = Path("/workspace/liushuo/cupid")
LEROBOT_DIR = REPO_ROOT / "lerobot_2.1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ACT with CUPID-ACT reweighting.")
    parser.add_argument("--dataset-path", type=Path, default=None, help="LeRobot dataset root directory.")
    parser.add_argument("--dataset.root", dest="dataset_root", type=Path, default=None, help="LeRobot dataset root.")
    parser.add_argument("--dataset.repo_id", dest="dataset_repo_id", type=str, default=None, help="Dataset repo id.")
    parser.add_argument("--score-pkl", type=Path, required=True, help="CUPID-ACT score.pkl output.")
    parser.add_argument("--output-dir", type=Path, required=False)
    parser.add_argument("--output_dir", dest="output_dir_alias", type=Path, default=None, help="Alias for --output-dir.")
    parser.add_argument("--job_name", type=str, default=None, help="Optional job name (lerobot-train compatible).")
    parser.add_argument("--policy.type", dest="policy_type", type=str, default="act", help="Policy type.")
    parser.add_argument("--policy.repo_id", dest="policy_repo_id", type=str, default=None, help="Policy repo id.")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=100000)
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument("--save-every", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--grad-clip-norm", type=float, default=10.0)

    parser.add_argument("--drop-bad-ratio", type=float, default=0.3, help="Drop episodes with bad-window ratio >= this.")
    parser.add_argument(
        "--drop-weight-quantile",
        type=float,
        default=0.0,
        help="Drop lowest quantile of episode weights after bad-ratio filtering (0 disables).",
    )
    parser.add_argument(
        "--weight-mode",
        type=str,
        default="window_mean",
        choices=("window_mean", "influence", "none"),
        help="How to compute episode weights.",
    )
    parser.add_argument(
        "--use-weighted-sampler",
        action="store_true",
        help="Use WeightedRandomSampler for episode reweighting.",
    )
    parser.add_argument(
        "--segment-mask",
        action="store_true",
        help="Mask bad windows by assigning zero sample weight to frames inside dropped windows.",
    )
    parser.add_argument(
        "--segment-weight",
        action="store_true",
        help="Apply window-level weights to frame sampling weights when available.",
    )
    parser.add_argument(
        "--pretrained-dir",
        type=Path,
        default=None,
        help="Optional pretrained ACT checkpoint to initialize weights.",
    )

    # ACT hyper-params
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument("--n-action-steps", type=int, default=50)
    parser.add_argument("--vision-backbone", type=str, default="resnet18")
    parser.add_argument(
        "--pretrained-backbone",
        type=str,
        default="ResNet18_Weights.IMAGENET1K_V1",
        help="Pass None to disable backbone pretraining.",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--backbone-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    return parser.parse_args()


def _infer_repo_id(dataset_root: Path) -> str:
    parts = dataset_root.parts
    if len(parts) >= 2:
        return "/".join(parts[-2:])
    return dataset_root.name


def _load_influence_payload(score_pkl: Path) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    with score_pkl.open("rb") as f:
        payload = pickle.load(f)
    scores = None
    if isinstance(payload, dict):
        if isinstance(payload.get("influence_scores"), Iterable):
            scores = np.asarray(payload["influence_scores"], dtype=float)
        if scores is None and isinstance(payload.get("train", {}), dict):
            train = payload["train"]
            if isinstance(train.get("influence_scores"), Iterable):
                scores = np.asarray(train["influence_scores"], dtype=float)
    if scores is None:
        raise ValueError(f"{score_pkl} does not contain influence_scores")
    metadata = {}
    if isinstance(payload, dict) and isinstance(payload.get("metadata"), dict):
        metadata = payload["metadata"]
    return scores, metadata


def _compute_episode_stats(
    scores: np.ndarray, metadata: Dict[str, np.ndarray]
) -> Tuple[np.ndarray, np.ndarray]:
    window_weights = metadata.get("window_weights")
    window_drop = metadata.get("window_drop_mask")
    window_ep = metadata.get("window_episode_index")
    if window_weights is None or window_drop is None or window_ep is None:
        num = scores.shape[0]
        return np.asarray(scores, dtype=float), np.zeros(num, dtype=float)

    window_weights = np.asarray(window_weights, dtype=float)
    window_drop = np.asarray(window_drop, dtype=float)
    window_ep = np.asarray(window_ep, dtype=int)
    max_ep = int(window_ep.max())
    ep_weights = np.zeros(max_ep + 1, dtype=float)
    ep_bad = np.zeros(max_ep + 1, dtype=float)
    for ep in range(max_ep + 1):
        mask = window_ep == ep
        if not mask.any():
            continue
        ep_weights[ep] = float(window_weights[mask].mean())
        ep_bad[ep] = float(window_drop[mask].mean())
    return ep_weights, ep_bad


def _build_segment_maps(metadata: Dict[str, np.ndarray]) -> Dict[int, Tuple[np.ndarray, np.ndarray]] | None:
    window_weights = metadata.get("window_weights")
    window_drop = metadata.get("window_drop_mask")
    window_ep = metadata.get("window_episode_index")
    window_start = metadata.get("window_start")
    window_end = metadata.get("window_end")
    if (
        window_weights is None
        or window_drop is None
        or window_ep is None
        or window_start is None
        or window_end is None
    ):
        return None

    window_weights = np.asarray(window_weights, dtype=float)
    window_drop = np.asarray(window_drop, dtype=bool)
    window_ep = np.asarray(window_ep, dtype=int)
    window_start = np.asarray(window_start, dtype=int)
    window_end = np.asarray(window_end, dtype=int)

    segment_maps: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    for ep in np.unique(window_ep):
        mask = window_ep == ep
        if not mask.any():
            continue
        max_end = int(np.max(window_end[mask]))
        if max_end <= 0:
            continue
        weight_sum = np.zeros(max_end, dtype=float)
        count = np.zeros(max_end, dtype=float)
        keep_mask = np.zeros(max_end, dtype=bool)
        for s, e, w, d in zip(
            window_start[mask], window_end[mask], window_weights[mask], window_drop[mask]
        ):
            s = max(int(s), 0)
            e = min(int(e), max_end)
            if e <= s:
                continue
            if d:
                continue
            weight_sum[s:e] += float(w)
            count[s:e] += 1.0
            keep_mask[s:e] = True
        frame_weight = np.where(count > 0, weight_sum / count, 0.0)
        segment_maps[int(ep)] = (frame_weight, keep_mask)
    return segment_maps


def _build_frame_indices(episode_indices: np.ndarray) -> np.ndarray:
    frame_idx = np.zeros_like(episode_indices, dtype=np.int64)
    last_ep = None
    counter = 0
    for i, ep in enumerate(episode_indices):
        if last_ep is None or ep != last_ep:
            counter = 0
        frame_idx[i] = counter
        counter += 1
        last_ep = ep
    return frame_idx


def _normalize_weights(weights: np.ndarray) -> np.ndarray:
    w = np.asarray(weights, dtype=float)
    w = w - np.nanmin(w)
    if np.allclose(w, 0):
        return np.ones_like(w)
    w = w / (np.nanmean(w) + 1e-8)
    return w


def _summarize_array(values: np.ndarray) -> Dict[str, float]:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return {"min": float("nan"), "max": float("nan"), "mean": float("nan"), "std": float("nan")}
    return {
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
        "mean": float(np.mean(vals)),
        "std": float(np.std(vals)),
    }


def _write_report(
    output_dir: Path,
    all_eps: list[int],
    keep_eps: list[int],
    drop_bad_ratio: float,
    drop_weight_quantile: float,
    ep_bad: np.ndarray,
    ep_weights: np.ndarray,
    weight_source: np.ndarray,
    sample_weights: np.ndarray | None,
    sampling_weights_episode: np.ndarray,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    keep_set = set(keep_eps)
    report = {
        "total_episodes": len(all_eps),
        "kept_episodes": len(keep_eps),
        "dropped_episodes": len(all_eps) - len(keep_eps),
        "drop_bad_ratio": drop_bad_ratio,
        "drop_weight_quantile": drop_weight_quantile,
        "bad_ratio_stats": _summarize_array(ep_bad),
        "weight_stats": _summarize_array(ep_weights),
        "weight_source_stats": _summarize_array(weight_source),
        "sample_weight_stats": _summarize_array(sample_weights) if sample_weights is not None else None,
        "sampling_weight_episode_stats": _summarize_array(sampling_weights_episode),
        "segment_mask_enabled": bool(sample_weights is not None),
    }
    report_path = output_dir / "cupid_reweight_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    csv_path = output_dir / "cupid_reweight_episodes.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "keep", "bad_ratio", "episode_weight", "weight_source", "sampling_weight"])
        for ep in all_eps:
            bad_val = ep_bad[ep] if ep < len(ep_bad) else float("nan")
            w_val = ep_weights[ep] if ep < len(ep_weights) else float("nan")
            ws_val = weight_source[ep] if ep < len(weight_source) else float("nan")
            sw_val = sampling_weights_episode[ep] if ep < len(sampling_weights_episode) else float("nan")
            writer.writerow([ep, int(ep in keep_set), bad_val, w_val, ws_val, sw_val])

    keep_list = output_dir / "cupid_reweight_keep.txt"
    drop_list = output_dir / "cupid_reweight_drop.txt"
    keep_list.write_text("\n".join(str(ep) for ep in keep_eps))
    drop_list.write_text("\n".join(str(ep) for ep in all_eps if ep not in keep_set))

    def _write_ranked(name: str, values: np.ndarray, topk: int = 50, reverse: bool = True) -> None:
        pairs = [(ep, float(values[ep])) for ep in all_eps if ep < len(values) and np.isfinite(values[ep])]
        pairs.sort(key=lambda x: x[1], reverse=reverse)
        out = output_dir / name
        lines = [f"{ep}\t{val:.6g}" for ep, val in pairs[:topk]]
        out.write_text("\n".join(lines))

    _write_ranked("cupid_top_weight_source.txt", weight_source, topk=50, reverse=True)
    _write_ranked("cupid_bottom_weight_source.txt", weight_source, topk=50, reverse=False)
    _write_ranked("cupid_top_sampling_weight.txt", sampling_weights_episode, topk=50, reverse=True)
    _write_ranked("cupid_bottom_sampling_weight.txt", sampling_weights_episode, topk=50, reverse=False)
    _write_ranked("cupid_top_bad_ratio.txt", ep_bad, topk=50, reverse=True)
    _write_ranked("cupid_bottom_bad_ratio.txt", ep_bad, topk=50, reverse=False)

    def _write_hist(name: str, values: np.ndarray, bins: int = 20) -> None:
        vals = np.asarray(values, dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            return
        hist, edges = np.histogram(vals, bins=bins)
        out = output_dir / name
        with out.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["bin_left", "bin_right", "count"])
            for i in range(len(hist)):
                writer.writerow([float(edges[i]), float(edges[i + 1]), int(hist[i])])

    _write_hist("cupid_hist_bad_ratio.csv", ep_bad)
    _write_hist("cupid_hist_weight_source.csv", weight_source)
    _write_hist("cupid_hist_sampling_weight.csv", sampling_weights_episode)
    if sample_weights is not None:
        _write_hist("cupid_hist_sample_weight.csv", sample_weights)


def main() -> None:
    args = parse_args()

    if args.output_dir_alias is not None:
        args.output_dir = args.output_dir_alias
    if args.output_dir is None:
        raise ValueError("Provide --output-dir or --output_dir.")

    dataset_root = args.dataset_root or args.dataset_path
    if dataset_root is None:
        raise ValueError("Provide --dataset-path or --dataset.root.")

    if args.policy_type and args.policy_type != "act":
        raise ValueError(f"Only ACT policy is supported, got policy.type={args.policy_type}.")

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

    sys.path.insert(0, str(LEROBOT_DIR / "src"))
    from lerobot.configs.types import FeatureType  # type: ignore
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata  # type: ignore
    from lerobot.datasets.utils import dataset_to_policy_features  # type: ignore
    from lerobot.policies.act.configuration_act import ACTConfig  # type: ignore
    from lerobot.policies.act.modeling_act import ACTPolicy  # type: ignore

    args.output_dir.mkdir(parents=True, exist_ok=True)

    import random
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    scores, metadata = _load_influence_payload(args.score_pkl)
    ep_weights, ep_bad = _compute_episode_stats(scores, metadata)

    repo_id = args.dataset_repo_id or _infer_repo_id(dataset_root)
    ds_meta = LeRobotDatasetMetadata(repo_id=repo_id, root=dataset_root, force_cache_sync=False)

    total_eps = len(ds_meta.episodes)
    all_eps = sorted(ds_meta.episodes.keys())
    keep_eps = [ep for ep in all_eps if ep < len(ep_bad) and ep_bad[ep] < args.drop_bad_ratio]

    weight_source = ep_weights if args.weight_mode != "none" else scores
    weight_source = np.asarray(weight_source, dtype=float)
    if args.drop_weight_quantile > 0:
        thresh = float(np.quantile(weight_source, args.drop_weight_quantile))
        keep_eps = [ep for ep in keep_eps if ep < len(weight_source) and weight_source[ep] > thresh]
        print(f"[filter] weight quantile threshold={thresh:.6g} (q={args.drop_weight_quantile})")

    if not keep_eps:
        raise RuntimeError("No episodes left after filtering. Lower drop thresholds.")

    policy_features = dataset_to_policy_features(ds_meta.features)
    output_features = {k: ft for k, ft in policy_features.items() if ft.type is FeatureType.ACTION}
    input_features = {k: ft for k, ft in policy_features.items() if k not in output_features}

    cfg = ACTConfig(
        input_features=input_features,
        output_features=output_features,
        chunk_size=args.chunk_size,
        n_action_steps=args.n_action_steps,
        vision_backbone=args.vision_backbone,
        pretrained_backbone_weights=None if args.pretrained_backbone == "None" else args.pretrained_backbone,
        optimizer_lr=args.learning_rate,
        optimizer_weight_decay=args.weight_decay,
        optimizer_lr_backbone=args.backbone_lr,
        dropout=args.dropout,
    )
    if args.policy_repo_id is not None:
        cfg.repo_id = args.policy_repo_id

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    if args.pretrained_dir:
        policy = ACTPolicy.from_pretrained(str(args.pretrained_dir))
        policy.to(device)
        policy.train()
    else:
        policy = ACTPolicy(cfg, dataset_stats=ds_meta.stats).train().to(device)

    delta_ts = {}
    if cfg.action_delta_indices is not None:
        delta_ts["action"] = [i / ds_meta.fps for i in cfg.action_delta_indices]
    if cfg.observation_delta_indices is not None:
        for key in ds_meta.image_keys:
            if key.startswith("observation."):
                delta_ts[key] = [i / ds_meta.fps for i in cfg.observation_delta_indices]
        if cfg.robot_state_feature is not None:
            delta_ts["observation.state"] = [i / ds_meta.fps for i in cfg.observation_delta_indices]

    dataset = LeRobotDataset(
        repo_id=repo_id,
        root=dataset_root,
        delta_timestamps=delta_ts or None,
        episodes=keep_eps,
        force_cache_sync=False,
        download_videos=False,
    )

    sampler = None
    sample_weights = None
    segment_maps = _build_segment_maps(metadata)
    if args.segment_mask or args.segment_weight:
        if segment_maps is None:
            print("[warn] segment_mask/segment_weight enabled but window metadata missing; falling back to episode weights.")
            segment_maps = None
    if args.weight_mode == "window_mean":
        base_weights = _normalize_weights(ep_weights)
    elif args.weight_mode == "influence":
        base_weights = _normalize_weights(scores)
    else:
        base_weights = np.ones_like(ep_weights)
    sampling_weights_episode = np.zeros(max(len(base_weights), total_eps), dtype=float)
    for ep in all_eps:
        if ep < len(base_weights):
            sampling_weights_episode[ep] = float(base_weights[ep])
    for ep in all_eps:
        if ep not in keep_eps:
            sampling_weights_episode[ep] = 0.0
    if args.use_weighted_sampler:
        ep_weight_map = {ep: float(base_weights[ep]) for ep in keep_eps if ep < len(base_weights)}
        sample_eps = np.asarray(dataset.hf_dataset["episode_index"], dtype=np.int64)
        if "frame_index" in dataset.hf_dataset.column_names:
            frame_idx = np.asarray(dataset.hf_dataset["frame_index"], dtype=np.int64)
        else:
            frame_idx = _build_frame_indices(sample_eps)
        sample_weights = np.asarray([ep_weight_map.get(int(ep), 1.0) for ep in sample_eps], dtype=np.float64)
        if segment_maps is not None and (args.segment_mask or args.segment_weight):
            for i, (ep, fi) in enumerate(zip(sample_eps, frame_idx)):
                mapping = segment_maps.get(int(ep))
                if mapping is None:
                    continue
                frame_weight, keep_mask = mapping
                if fi >= len(keep_mask):
                    continue
                if args.segment_mask and not keep_mask[fi]:
                    sample_weights[i] = 0.0
                    continue
                if args.segment_weight:
                    sample_weights[i] *= float(frame_weight[fi])
        sampler = torch.utils.data.WeightedRandomSampler(
            torch.from_numpy(sample_weights),
            num_samples=len(sample_weights),
            replacement=True,
        )

    _write_report(
        args.output_dir,
        all_eps,
        keep_eps,
        args.drop_bad_ratio,
        args.drop_weight_quantile,
        ep_bad,
        ep_weights,
        weight_source,
        sample_weights,
        sampling_weights_episode,
    )
    print(
        f"[filter] kept {len(keep_eps)}/{total_eps} episodes "
        f"(bad_ratio<{args.drop_bad_ratio}, weight_mode={args.weight_mode})"
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=device.type != "cpu",
        drop_last=False,
    )

    if hasattr(policy, "get_optim_params"):
        optim_params = policy.get_optim_params()
    else:
        optim_params = policy.parameters()
    optimizer = torch.optim.AdamW(
        optim_params,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    step = 0
    while step < args.max_steps:
        for batch in dataloader:
            batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            loss, loss_dict = policy.forward(batch)
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), max_norm=args.grad_clip_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            if step % args.log_every == 0:
                l1 = loss_dict.get("l1_loss", 0.0)
                kld = loss_dict.get("kld_loss", 0.0)
                print(
                    f"[step {step}] loss={loss.item():.4f} l1={float(l1):.4f} "
                    f"kld={float(kld):.4f} kept_eps={len(keep_eps)}/{total_eps}"
                )

            if args.save_every > 0 and step > 0 and step % args.save_every == 0:
                ckpt_dir = args.output_dir / f"step_{step}"
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                policy.save_pretrained(ckpt_dir)
                print(f"[checkpoint] saved to {ckpt_dir}")

            step += 1
            if step >= args.max_steps:
                break

    policy.save_pretrained(args.output_dir)
    print(f"[done] saved to {args.output_dir}")


if __name__ == "__main__":
    main()
