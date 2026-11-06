#!/usr/bin/env python
"""
在 RobotWin 的本地 LeRobot 数据集上训练 ACT，并将 `observation.env_state`
重映射为 ACT 可识别的 `observation.environment_state` 条件输入。

设计原则：
1. 不修改 upstream `lerobot` 代码。
2. 不改写原始数据集文件，只做内存级别的数据预处理与特征映射。
3. 兼容当前 beat_block_hammer_ur5_randomized_500 数据集的 task 元信息缺失问题。
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import pyarrow.parquet as pq
import torch
from torch import nn


REPO_ROOT = Path(__file__).resolve().parents[1]
LEROBOT_DIR = REPO_ROOT / "lerobot_2.1"
DEFAULT_DATASET_ROOT = Path("/workspace/liushuo/dataset/beat_block_hammer_ur5_randomized_500")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="在 RobotWin 数据集上训练 ACT，并注入 arm_id 条件输入")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="本地 LeRobot v2.1 数据集根目录",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default="local/beat_block_hammer_ur5_randomized_500",
        help="LeRobot 数据集 ID，仅用于元信息构造",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "robotwin" / "beat_block_hammer_ur5_randomized_500_act",
        help="模型与 checkpoint 输出目录",
    )
    parser.add_argument("--device", type=str, default="cuda", help="cuda 或 cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=100000, help="训练总步数")
    parser.add_argument("--log-every", type=int, default=100, help="打印日志的步数间隔")
    parser.add_argument("--save-every", type=int, default=10000, help="保存 checkpoint 的步数间隔")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--use-amp", action="store_true", help="启用 AMP 混合精度训练")
    parser.add_argument(
        "--pretrained-dir",
        type=Path,
        default=None,
        help="已有 ACT save_pretrained checkpoint，用于从已有权重继续训练",
    )
    parser.add_argument(
        "--resume-step",
        type=int,
        default=0,
        help="续训起始全局 step；仅用于日志与 checkpoint 命名，optimizer 会重新初始化",
    )

    # ACT 关键超参
    parser.add_argument("--chunk-size", type=int, default=50, help="动作块长度")
    parser.add_argument("--n-action-steps", type=int, default=50, help="每次推理执行的动作步数")
    parser.add_argument("--vision-backbone", type=str, default="resnet18", help="torchvision ResNet 主干")
    parser.add_argument(
        "--pretrained-backbone",
        type=str,
        default="ResNet18_Weights.IMAGENET1K_V1",
        help="ImageNet 预训练权重名称；若不想用预训练，传入 None",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--backbone-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    return parser.parse_args()


def build_delta_timestamps(features, cfg, fps: int) -> Dict[str, List[float]]:
    """根据数据集真实键名生成 delta_timestamps。"""
    from lerobot.configs.types import FeatureType  # type: ignore

    delta_timestamps: Dict[str, List[float]] = {}
    for key, ft in features.items():
        if ft.type is FeatureType.ACTION:
            delta_timestamps[key] = [i / fps for i in cfg.action_delta_indices]
        else:
            delta_timestamps[key] = [0.0]
    return delta_timestamps


def build_runtime_task_maps(dataset_root: Path) -> Tuple[dict[int, str], dict[str, int]]:
    """
    从 `episodes.jsonl` 与 parquet 里的 `task_index` 动态重建任务映射。

    这么做的原因是当前数据集的 `meta/tasks.jsonl` 不完整，直接走 LeRobotDataset
    会在 `__getitem__` 时因为缺失 task 索引而触发 KeyError。
    """
    meta_dir = dataset_root / "meta"
    episodes_path = meta_dir / "episodes.jsonl"
    if not episodes_path.exists():
        raise FileNotFoundError(f"缺少 episodes.jsonl: {episodes_path}")

    tasks_by_index: dict[int, str] = {}
    with episodes_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            episode = json.loads(line)
            episode_index = int(episode["episode_index"])
            tasks = episode.get("tasks", [])
            if len(tasks) != 1:
                raise ValueError(
                    f"期望每个 episode 只有 1 条任务描述，但 episode {episode_index} 得到 {len(tasks)} 条。"
                )

            parquet_path = dataset_root / "data" / f"chunk-{episode_index // 1000:03d}" / f"episode_{episode_index:06d}.parquet"
            if not parquet_path.exists():
                raise FileNotFoundError(f"缺少 episode parquet: {parquet_path}")

            table = pq.read_table(parquet_path, columns=["task_index"])
            task_index = int(table.column("task_index")[0].as_py())
            task_text = str(tasks[0])

            previous = tasks_by_index.get(task_index)
            if previous is not None and previous != task_text:
                raise ValueError(
                    f"task_index={task_index} 映射到多个任务文本，无法安全修复：{previous!r} vs {task_text!r}"
                )
            tasks_by_index[task_index] = task_text

    task_to_task_index = {task: task_index for task_index, task in tasks_by_index.items()}
    return tasks_by_index, task_to_task_index


def patch_dataset_task_maps(dataset, dataset_root: Path) -> None:
    """在内存中修复 dataset.meta.tasks，避免原始 tasks.jsonl 缺失导致取样失败。"""
    runtime_tasks, runtime_task_to_index = build_runtime_task_maps(dataset_root)
    dataset.meta.tasks = runtime_tasks
    dataset.meta.task_to_task_index = runtime_task_to_index


def remap_policy_features_for_env_condition(ds_meta_features: dict):
    """
    为 ACT 构造模型输入特征：
    - 保留原始 `observation.state`
    - 将 `observation.env_state` 改名为 `observation.environment_state`
      以触发 ACT 的 env token 路径
    """
    from lerobot.datasets.utils import dataset_to_policy_features  # type: ignore

    model_features = copy.deepcopy(ds_meta_features)
    if "observation.env_state" not in model_features:
        return dataset_to_policy_features(model_features)
    if "observation.environment_state" in model_features:
        raise KeyError("数据集已存在 observation.environment_state，当前脚本避免覆盖该键。")

    model_features["observation.environment_state"] = model_features.pop("observation.env_state")
    return dataset_to_policy_features(model_features)


def preprocess_batch(batch: dict) -> dict:
    """把原始 batch 中的 `observation.env_state` 映射到 ACT 需要的键名。"""
    processed = dict(batch)
    if "observation.env_state" in processed:
        processed["observation.environment_state"] = processed["observation.env_state"]

    for key, value in list(processed.items()):
        if not isinstance(value, torch.Tensor):
            continue

        # RobotWin 的单步观测会多带一个长度为 1 的时间维，这里统一压掉。
        if value.dim() == 5 and value.shape[1] == 1:
            processed[key] = value[:, 0]
        elif value.dim() == 3 and value.shape[1] == 1 and key in {
            "observation.state",
            "observation.env_state",
            "observation.environment_state",
        }:
            processed[key] = value[:, 0]

    return processed


def main() -> None:
    args = parse_args()

    if not args.dataset_root.exists():
        raise FileNotFoundError(f"未找到数据集目录: {args.dataset_root}")
    if args.output_dir.exists() and args.pretrained_dir is None:
        raise FileExistsError(f"输出目录已存在；如需续训请传入 --pretrained-dir：{args.output_dir}")
    if args.pretrained_dir is not None and not args.pretrained_dir.exists():
        raise FileNotFoundError(f"未找到 pretrained checkpoint: {args.pretrained_dir}")

    os.environ["HF_LEROBOT_HOME"] = str(args.dataset_root.parent.resolve())
    sys.path.insert(0, str(LEROBOT_DIR / "src"))

    from lerobot.configs.types import FeatureType  # type: ignore  # noqa: E402
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata  # type: ignore  # noqa: E402
    from lerobot.datasets.utils import dataset_to_policy_features  # type: ignore  # noqa: E402
    from lerobot.policies.act.configuration_act import ACTConfig  # type: ignore  # noqa: E402
    from lerobot.policies.act.modeling_act import ACTPolicy  # type: ignore  # noqa: E402

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=args.pretrained_dir is not None)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    ds_meta = LeRobotDatasetMetadata(args.repo_id, root=args.dataset_root)
    raw_policy_features = dataset_to_policy_features(ds_meta.features)
    model_policy_features = remap_policy_features_for_env_condition(ds_meta.features)

    output_features = {k: ft for k, ft in model_policy_features.items() if ft.type is FeatureType.ACTION}
    input_features = {k: ft for k, ft in model_policy_features.items() if k not in output_features}

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
        device=device.type,
        use_amp=args.use_amp,
    )

    if args.pretrained_dir is not None:
        policy = ACTPolicy.from_pretrained(str(args.pretrained_dir)).train().to(device)
        print(
            f"[resume] 从 {args.pretrained_dir} 加载权重，"
            f"从 global step {args.resume_step} 继续计数",
            flush=True,
        )
    else:
        policy = ACTPolicy(cfg, dataset_stats=ds_meta.stats).train().to(device)

    delta_timestamps = build_delta_timestamps(raw_policy_features, cfg, ds_meta.fps)
    dataset = LeRobotDataset(args.repo_id, delta_timestamps=delta_timestamps, root=args.dataset_root)
    patch_dataset_task_maps(dataset, args.dataset_root)

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
    scaler = torch.amp.GradScaler(device.type, enabled=args.use_amp)

    step = max(int(args.resume_step), 0)
    while step < args.max_steps:
        for batch in dataloader:
            batch = preprocess_batch(batch)
            batch = {k: (v.to(device, non_blocking=device.type != "cpu") if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}

            with torch.autocast(device_type=device.type, enabled=args.use_amp):
                loss, loss_dict = policy.forward(batch)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            if step % args.log_every == 0:
                l1 = float(loss_dict.get("l1_loss", 0.0))
                kld = float(loss_dict.get("kld_loss", 0.0))
                print(f"[step {step}] loss={loss.item():.4f} l1={l1:.4f} kld={kld:.4f}", flush=True)

            if args.save_every > 0 and step > 0 and step % args.save_every == 0:
                ckpt_dir = args.output_dir / f"step_{step}"
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                policy.save_pretrained(ckpt_dir)
                torch.save({"step": step, "optimizer": optimizer.state_dict()}, ckpt_dir / "training_state.pt")
                print(f"[checkpoint] 已保存到 {ckpt_dir}", flush=True)

            step += 1
            if step >= args.max_steps:
                break

    policy.save_pretrained(args.output_dir)
    torch.save({"step": step, "optimizer": optimizer.state_dict()}, args.output_dir / "training_state.pt")
    print(f"[done] 训练完成，模型已保存到 {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
