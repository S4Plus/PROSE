#!/usr/bin/env python
"""
根据 influence score 过滤低分 episode 后重新训练 ACT。

示例：
 nohup python libero/retrain_act.py \
  --score-pkl ./new_test20/score_libero20.pkl \
  --repo-id local/lerobot_random20 \
  --lerobot-home ./datasets \
  --output-dir ./output/act/libero20_drop5 \
  --max-steps 100000 \
  > ./logs_libero/libero20_drop5.log 2>&1 &
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import torch
from torch import nn


REPO_ROOT = Path(__file__).resolve().parents[1]
LEROBOT_DIR = REPO_ROOT / "lerobot_2.1"


def build_delta_timestamps(features, cfg, fps: int) -> Dict[str, List[float]]:
    """根据特征类型生成 delta_timestamps，ACT 仅对动作使用未来序列。"""
    from lerobot.configs.types import FeatureType  # type: ignore

    delta_timestamps: Dict[str, List[float]] = {}
    for key, ft in features.items():
        if ft.type is FeatureType.ACTION:
            delta_timestamps[key] = [i / fps for i in cfg.action_delta_indices]
        else:
            delta_timestamps[key] = [0.0]
    return delta_timestamps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="在过滤低分 episode 后重新训练 ACT")
    parser.add_argument(
        "--score-pkl",
        type=Path,
        required=True,
        help="score_libero*.pkl 路径，需包含 influence_scores",
    )
    parser.add_argument(
        "--drop-k",
        type=int,
        default=5,
        help="按 influence_scores 从低到高剔除的 episode 数量",
    )
    parser.add_argument("--repo-id", type=str, default="local/libero-act", help="LeRobot 数据集 ID")
    parser.add_argument(
        "--lerobot-home",
        type=Path,
        default=REPO_ROOT / "outputs" / "libero_lerobot",
        help="LEROBOT_HOME，需与预处理时保持一致",
    )
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "libero" / "checkpoints" / "act_retrain")
    parser.add_argument("--device", type=str, default="cuda", help="cuda 或 cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=100000, help="训练总步数")
    parser.add_argument("--log-every", type=int, default=20, help="打印日志的步数间隔")
    parser.add_argument("--save-every", type=int, default=20000, help="保存 checkpoint 的步数间隔")

    # ACT 关键超参
    parser.add_argument("--chunk-size", type=int, default=50, help="动作块长度（同时也决定 action delta 个数）")
    parser.add_argument("--n-action-steps", type=int, default=50, help="每次推理执行的动作步数")
    parser.add_argument("--vision-backbone", type=str, default="resnet18", help="torchvision resnet 主干")
    parser.add_argument(
        "--pretrained-backbone",
        type=str,
        default="ResNet18_Weights.IMAGENET1K_V1",
        help="ImageNet 预训练权重名称；若不想用预训练，传入 None",
    )

    # 优化器设置
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--backbone-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    return parser.parse_args()


def load_influence_scores(score_pkl: Path) -> np.ndarray:
    with score_pkl.open("rb") as f:
        payload = pickle.load(f)

    # 通用读取逻辑：优先顶层，其次 train/influence_scores
    def _get_scores(obj: Dict) -> np.ndarray | None:
        if isinstance(obj, dict) and isinstance(obj.get("influence_scores"), Iterable):
            return np.asarray(obj["influence_scores"], dtype=float)
        return None

    scores = _get_scores(payload)
    if scores is None:
        scores = _get_scores(payload.get("train", {}))
    if scores is None:
        raise ValueError(f"{score_pkl} 中未找到 influence_scores")
    return scores


def main() -> None:
    args = parse_args()

    # 关闭 Hub 网络访问，强制走本地缓存
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ["HF_LEROBOT_HOME"] = str(args.lerobot_home.resolve())

    # 延迟导入，确保使用上面设置的缓存目录
    sys.path.insert(0, str(LEROBOT_DIR / "src"))
    from lerobot.configs.types import FeatureType  # type: ignore  # noqa: E402
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata  # type: ignore  # noqa: E402
    from lerobot.datasets.utils import dataset_to_policy_features  # type: ignore  # noqa: E402
    from lerobot.policies.act.configuration_act import ACTConfig  # type: ignore  # noqa: E402
    from lerobot.policies.act.modeling_act import ACTPolicy  # type: ignore  # noqa: E402

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 读取分数并计算要保留的 episodes
    influence_scores = load_influence_scores(args.score_pkl)
    drop_k = max(0, min(args.drop_k, influence_scores.shape[0] - 1))
    drop_indices = set(np.argsort(influence_scores)[:drop_k].tolist())
    keep_episodes = [i for i in range(influence_scores.shape[0]) if i not in drop_indices]
    if not keep_episodes:
        raise ValueError("过滤后没有剩余的 episode，可适当调小 --drop-k")
    print(
        f"Loaded {len(influence_scores)} scores from {args.score_pkl}, "
        f"dropping {len(drop_indices)} lowest -> keep episodes {keep_episodes}"
    )

    dataset_dir = Path(os.environ["HF_LEROBOT_HOME"]) / args.repo_id
    if not dataset_dir.exists():
        raise FileNotFoundError(f"未找到已转换的数据集: {dataset_dir}")

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    # 读取数据集元信息并构建特征
    ds_meta = LeRobotDatasetMetadata(args.repo_id, root=dataset_dir, force_cache_sync=False)
    policy_features = dataset_to_policy_features(ds_meta.features)
    output_features = {k: ft for k, ft in policy_features.items() if ft.type is FeatureType.ACTION}
    input_features = {k: ft for k, ft in policy_features.items() if k not in output_features}

    # ACT 配置
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

    policy = ACTPolicy(cfg, dataset_stats=ds_meta.stats).train().to(device)

    delta_timestamps = build_delta_timestamps(policy_features, cfg, ds_meta.fps)
    dataset = LeRobotDataset(
        args.repo_id,
        delta_timestamps=delta_timestamps,
        root=dataset_dir,
        episodes=keep_episodes,
        force_cache_sync=False,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type != "cpu",
        drop_last=True,
    )

    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    step = 0
    while step < args.max_steps:
        for batch in dataloader:
            batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            loss, loss_dict = policy.forward(batch)
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            if step % args.log_every == 0:
                l1 = loss_dict.get("l1_loss", 0.0)
                kld = loss_dict.get("kld_loss", 0.0)
                print(
                    f"[step {step}] loss={loss.item():.4f} "
                    f"l1={float(l1):.4f} kld={float(kld):.4f} episodes={keep_episodes}"
                )

            if args.save_every > 0 and step > 0 and step % args.save_every == 0:
                ckpt_dir = args.output_dir / f"step_{step}"
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                policy.save_pretrained(ckpt_dir)
                print(f"💾 已保存 checkpoint 到 {ckpt_dir}")

            step += 1
            if step >= args.max_steps:
                break

    policy.save_pretrained(args.output_dir)
    print(f"✅ 训练完成，模型已保存到 {args.output_dir}")


if __name__ == "__main__":
    main()
