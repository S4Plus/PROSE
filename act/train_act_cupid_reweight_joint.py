#!/usr/bin/env python
"""
Train ACT with CUPID-ACT scores using:
1) episode filtering based on bad-window ratio
2) episode reweighting based on window weights or influence scores
质量和影响力统一查看
Example:
质量评估由以下分量加权得到（都归一到 0-1 后融合）：

  - 动作平滑度（相邻动作差分越小越好）
  - 状态平滑度（相邻状态差分越小越好）
  - 轨迹长度一致性（接近中位长度越好）
  - 动作幅值惩罚（可选，默认权重 0）
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


CUDA_VISIBLE_DEVICES=5 nohup python /workspace/liushuo/cupid/act/train_act_cupid_reweight_joint.py \
    --dataset.repo_id ls/can_mh \
    --dataset.root /workspace/liushuo/cupid/datasets/lerobot/can_mh \
    --output_dir /workspace/liushuo/cupid/3_model/can_mh_joint_eval_0.2 \
    --job_name can_mh_joint_eval \
    --policy.type=act \
    --policy.repo_id ls/can_mh \
    --score-pkl /workspace/liushuo/cupid/1_infulenece_score/can_mh/score.pkl \
    --filter-score-mode joint \
    --joint-quality-source episode_eval \
    --joint-drop-quantile 0.2 \
    --joint-quality-power 1.0 \
    --joint-influence-power 1.0 \
    --quality-eval-action-weight 0.5 \
    --quality-eval-state-weight 0.3 \
    --quality-eval-length-weight 0.2 \
    --quality-eval-energy-weight 0.0 \
    --weight-mode window_mean \
    --use-weighted-sampler \
    --segment-mask \
    --segment-weight \
    > /workspace/liushuo/cupid/3_model/logs/can_mh_joint_eval_0.2_act.log 2>&1 &

"""
from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pyarrow.parquet as pq
import torch
from torch import nn


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from prose.paths import lerobot_root

LEROBOT_DIR = lerobot_root()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ACT with CUPID-ACT reweighting + joint-score filtering.")
    parser.add_argument("--dataset-path", type=Path, default=None, help="LeRobot dataset root directory.")
    parser.add_argument("--dataset.root", dest="dataset_root", type=Path, default=None, help="LeRobot dataset root.")
    parser.add_argument("--dataset.repo_id", dest="dataset_repo_id", type=str, default=None, help="Dataset repo id.")
    parser.add_argument(
        "--score-pkl",
        type=Path,
        required=False,
        default=None,
        help=(
            "CUPID-ACT score.pkl output. Optional in quality-only usage "
            "(e.g., weight_mode=none and joint_influence_power=0)."
        ),
    )
    parser.add_argument(
        "--selection-npz",
        type=Path,
        default=None,
        help="Optional joint episode-window selection artifact produced by myscripts/filter_joint_episode_window.py.",
    )
    parser.add_argument(
        "--selection-override-episodes",
        action="store_true",
        help="Use episode_keep_mask from --selection-npz as the primary episode filter result.",
    )
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
        "--filter-score-mode",
        type=str,
        default="joint",
        choices=("joint", "weight"),
        help="Use joint quality*influence score or legacy weight score for quantile filtering.",
    )
    parser.add_argument(
        "--joint-drop-quantile",
        type=float,
        default=0.2,
        help="Drop lowest quantile by joint score after bad-ratio filtering (0 disables).",
    )
    parser.add_argument(
        "--joint-quality-source",
        type=str,
        default="episode_eval",
        choices=("episode_eval", "trajectory_eval", "bad_ratio", "operator_csv", "quality_csv"),
        help=(
            "How to build episode quality score for joint score. "
            "episode_eval=kinematic smoothness; trajectory_eval=task-trajectory structure/progress; "
            "quality_csv=read per-episode scores from eval_cupid_quality.py output."
        ),
    )
    parser.add_argument(
        "--quality-score-csv",
        type=Path,
        default=None,
        help="Path to quality_episodes.csv generated by eval_cupid_quality.py (episode,quality_score).",
    )
    parser.add_argument(
        "--quality-score-default",
        type=float,
        default=0.5,
        help="Fallback quality score when an episode is missing in --quality-score-csv.",
    )
    parser.add_argument(
        "--quality-eval-action-weight",
        type=float,
        default=0.5,
        help="Weight for action smoothness quality in episode_eval mode.",
    )
    parser.add_argument(
        "--quality-eval-state-weight",
        type=float,
        default=0.3,
        help="Weight for state smoothness quality in episode_eval mode.",
    )
    parser.add_argument(
        "--quality-eval-length-weight",
        type=float,
        default=0.2,
        help="Weight for length consistency quality in episode_eval mode.",
    )
    parser.add_argument(
        "--quality-eval-energy-weight",
        type=float,
        default=0.0,
        help="Weight for action magnitude penalty quality in episode_eval mode.",
    )
    parser.add_argument(
        "--quality-eval-length-scale",
        type=float,
        default=1.0,
        help="Scale for episode length deviation penalty (in MAD units).",
    )
    parser.add_argument(
        "--quality-eval-default",
        type=float,
        default=0.5,
        help="Fallback quality score when episode_eval cannot compute a score.",
    )
    parser.add_argument(
        "--quality-eval-jitter-mode",
        type=str,
        default="absolute",
        choices=("relative", "absolute"),
        help="Use relative jitter (diff norm / signal magnitude) or absolute diff norm in episode_eval mode.",
    )
    parser.add_argument(
        "--quality-eval-normalizer",
        type=str,
        default="minmax",
        choices=("quantile", "minmax"),
        help="How to map raw quality components to [0,1] in episode_eval mode.",
    )
    parser.add_argument(
        "--quality-eval-quantile-low",
        type=float,
        default=0.05,
        help="Lower quantile for robust quantile normalization in episode_eval mode.",
    )
    parser.add_argument(
        "--quality-eval-quantile-high",
        type=float,
        default=0.95,
        help="Upper quantile for robust quantile normalization in episode_eval mode.",
    )
    parser.add_argument(
        "--traj-eval-terminal-weight",
        type=float,
        default=0.25,
        help="Weight for terminal-state consistency score in trajectory_eval mode.",
    )
    parser.add_argument(
        "--traj-eval-efficiency-weight",
        type=float,
        default=0.2,
        help="Weight for state-path efficiency score in trajectory_eval mode.",
    )
    parser.add_argument(
        "--traj-eval-progress-weight",
        type=float,
        default=0.25,
        help="Weight for state displacement/progress score in trajectory_eval mode.",
    )
    parser.add_argument(
        "--traj-eval-state-smooth-weight",
        type=float,
        default=0.15,
        help="Weight for relative state smoothness score in trajectory_eval mode.",
    )
    parser.add_argument(
        "--traj-eval-action-smooth-weight",
        type=float,
        default=0.1,
        help="Weight for relative action smoothness score in trajectory_eval mode.",
    )
    parser.add_argument(
        "--traj-eval-length-weight",
        type=float,
        default=0.05,
        help="Weight for length consistency score in trajectory_eval mode.",
    )
    parser.add_argument(
        "--traj-eval-terminal-knn-k",
        type=int,
        default=15,
        help="K for kNN terminal consistency distance in trajectory_eval mode (>=2).",
    )
    parser.add_argument(
        "--traj-eval-monotonic-weight",
        type=float,
        default=0.2,
        help="Weight for monotonic-progress score in trajectory_eval mode.",
    )
    parser.add_argument(
        "--traj-eval-backtrack-weight",
        type=float,
        default=0.1,
        help="Weight for backtrack-ratio penalty score in trajectory_eval mode.",
    )
    parser.add_argument(
        "--traj-eval-stall-weight",
        type=float,
        default=0.1,
        help="Weight for stall-ratio penalty score in trajectory_eval mode.",
    )
    parser.add_argument(
        "--traj-eval-fusion",
        type=str,
        default="weighted_geomean",
        choices=("weighted_mean", "weighted_geomean"),
        help="How to fuse trajectory_eval component scores.",
    )
    parser.add_argument(
        "--traj-eval-min-terminal-score",
        type=float,
        default=0.35,
        help="If terminal score is below this, apply quality cap when enabled.",
    )
    parser.add_argument(
        "--traj-eval-min-progress-score",
        type=float,
        default=0.45,
        help="If progress score is below this, apply quality cap when enabled.",
    )
    parser.add_argument(
        "--traj-eval-gate-quality-cap",
        type=float,
        default=0.15,
        help="Cap quality score when terminal/progress gate is violated; <0 disables.",
    )
    parser.add_argument(
        "--operator-quality-csv",
        type=Path,
        default=None,
        help="CSV with operator and episode_range columns (e.g. can_mh_operator_influence_stats.csv).",
    )
    parser.add_argument(
        "--operator-quality-better",
        type=float,
        default=1.0,
        help="Quality score assigned to operators containing 'better'.",
    )
    parser.add_argument(
        "--operator-quality-okay",
        type=float,
        default=0.7,
        help="Quality score assigned to operators containing 'okay'.",
    )
    parser.add_argument(
        "--operator-quality-worse",
        type=float,
        default=0.4,
        help="Quality score assigned to operators containing 'worse'.",
    )
    parser.add_argument(
        "--operator-quality-default",
        type=float,
        default=1.0,
        help="Fallback quality score when an episode has no operator mapping.",
    )
    parser.add_argument(
        "--joint-quality-power",
        type=float,
        default=1.0,
        help="Exponent applied to quality score in joint score.",
    )
    parser.add_argument(
        "--joint-influence-power",
        type=float,
        default=1.0,
        help="Exponent applied to normalized influence score in joint score.",
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
    parser.add_argument(
        "--resume-step",
        type=int,
        default=None,
        help="Global step to resume from. If omitted, infer from checkpoint dir name or training_state.pt.",
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


def _infer_resume_step(pretrained_dir: Path | None, resume_step: int | None) -> int:
    if resume_step is not None:
        return max(int(resume_step), 0)
    if pretrained_dir is None:
        return 0
    state_path = pretrained_dir / "training_state.pt"
    if state_path.is_file():
        state = torch.load(state_path, map_location="cpu")
        if isinstance(state, dict) and "step" in state:
            return max(int(state["step"]), 0)
    match = re.fullmatch(r"step_(\d+)", pretrained_dir.name)
    if match:
        return int(match.group(1))
    return 0


def _infer_repo_id(dataset_root: Path) -> str:
    parts = dataset_root.parts
    if len(parts) >= 2:
        return "/".join(parts[-2:])
    return dataset_root.name


def _build_runtime_task_maps(dataset_root: Path) -> Tuple[Dict[int, str], Dict[str, int]]:
    """从 episodes.jsonl 与 parquet 重建 task 映射，修复脏 metadata。"""

    meta_dir = dataset_root / "meta"
    episodes_path = meta_dir / "episodes.jsonl"
    if not episodes_path.exists():
        raise FileNotFoundError(f"Missing episodes.jsonl: {episodes_path}")

    tasks_by_index: Dict[int, str] = {}
    with episodes_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            episode = json.loads(line)
            episode_index = int(episode["episode_index"])
            tasks = episode.get("tasks", [])
            if len(tasks) != 1:
                raise ValueError(
                    f"Expected exactly one task in episode {episode_index}, got {len(tasks)}."
                )

            parquet_path = (
                dataset_root
                / "data"
                / f"chunk-{episode_index // 1000:03d}"
                / f"episode_{episode_index:06d}.parquet"
            )
            if not parquet_path.exists():
                raise FileNotFoundError(f"Missing episode parquet: {parquet_path}")

            table = pq.read_table(parquet_path, columns=["task_index"])
            task_index = int(table.column("task_index")[0].as_py())
            task_text = str(tasks[0])

            previous = tasks_by_index.get(task_index)
            if previous is not None and previous != task_text:
                raise ValueError(
                    f"task_index={task_index} maps to multiple tasks: {previous!r} vs {task_text!r}"
                )
            tasks_by_index[task_index] = task_text

    task_to_task_index = {task: task_index for task_index, task in tasks_by_index.items()}
    return tasks_by_index, task_to_task_index


def _patch_dataset_task_maps_if_needed(dataset_or_meta, dataset_root: Path) -> None:
    """仅在 metadata 缺失 task 映射时补丁修复，避免 DataLoader worker KeyError。"""

    try:
        runtime_tasks, runtime_task_to_index = _build_runtime_task_maps(dataset_root)
    except Exception:
        return

    current_tasks = getattr(dataset_or_meta, "tasks", {})
    needs_patch = False
    for task_index, task_text in runtime_tasks.items():
        if current_tasks.get(task_index) != task_text:
            needs_patch = True
            break

    if not needs_patch:
        return

    dataset_or_meta.tasks = runtime_tasks
    dataset_or_meta.task_to_task_index = runtime_task_to_index


def _infer_env_state_dim_from_policy(policy: ACTPolicy) -> int:
    """从 ACT policy 配置中推断多任务环境状态 one-hot 维度。"""

    config = getattr(policy, "config", None)
    if config is None:
        return 0
    input_features = getattr(config, "input_features", None)
    if not isinstance(input_features, dict):
        return 0
    feature = input_features.get("observation.environment_state")
    if feature is None:
        return 0
    shape = getattr(feature, "shape", None)
    if shape is None:
        return 0
    if isinstance(shape, int):
        return int(shape)
    if len(shape) == 0:
        return 0
    return int(shape[-1])


def _format_env_state_tensor(
    env_state: torch.Tensor | np.ndarray | list | tuple,
    env_state_dim: int,
) -> torch.Tensor:
    """将环境条件量整理为 ACT 期望的 `(B, D)` 形状。"""

    tensor = env_state if isinstance(env_state, torch.Tensor) else torch.as_tensor(env_state)
    if tensor.dim() == 0:
        return tensor.reshape(1, 1).to(dtype=torch.float32)
    if tensor.dim() == 1:
        if env_state_dim == 1:
            return tensor.unsqueeze(-1).to(dtype=torch.float32)
        if env_state_dim > 0 and tensor.shape[0] == env_state_dim:
            return tensor.unsqueeze(0).to(dtype=torch.float32)
        if env_state_dim > 0 and tensor.numel() % env_state_dim == 0:
            return tensor.view(-1, env_state_dim).to(dtype=torch.float32)
        return tensor.unsqueeze(0).to(dtype=torch.float32)
    if tensor.dim() == 2:
        return tensor.to(dtype=torch.float32)
    if tensor.dim() == 3:
        return tensor[:, -1, :].to(dtype=torch.float32)
    raise ValueError(f"Unexpected env_state tensor rank {tensor.dim()} for shape {tuple(tensor.shape)}")


def _augment_batch_environment_state(
    batch: Dict[str, torch.Tensor],
    env_state_dim: int,
) -> Dict[str, torch.Tensor]:
    """优先复用真实 env_state；缺失时再从 task_index 构造 one-hot。"""

    if env_state_dim <= 0:
        return batch
    if "observation.environment_state" in batch:
        batch["observation.environment_state"] = _format_env_state_tensor(
            batch["observation.environment_state"],
            env_state_dim,
        ).to(device=batch["observation.environment_state"].device)
        return batch
    if "observation.env_state" in batch:
        formatted = _format_env_state_tensor(batch["observation.env_state"], env_state_dim)
        if isinstance(batch["observation.env_state"], torch.Tensor):
            formatted = formatted.to(device=batch["observation.env_state"].device)
        batch["observation.environment_state"] = formatted
        return batch
    if "task_index" not in batch:
        raise KeyError(
            "当前 ACT 模型需要 observation.environment_state，但 batch 中既没有该字段/observation.env_state，也没有 task_index 可用于构造 one-hot。"
        )

    task_index = batch["task_index"]
    if not isinstance(task_index, torch.Tensor):
        task_index = torch.as_tensor(task_index)
    task_index = task_index.to(dtype=torch.long).view(-1)

    env_state = torch.zeros((task_index.shape[0], env_state_dim), dtype=torch.float32, device=task_index.device)
    valid = (task_index >= 0) & (task_index < env_state_dim)
    if not torch.all(valid):
        bad_values = task_index[~valid].detach().cpu().tolist()
        raise ValueError(
            f"task_index 超出 environment_state 维度范围: dim={env_state_dim}, bad_task_index={bad_values[:8]}"
        )
    env_state.scatter_(1, task_index.unsqueeze(1), 1.0)
    batch["observation.environment_state"] = env_state
    return batch


def _ensure_writable_hf_cache(output_dir: Path) -> None:
    hf_home = output_dir / "hf_home"
    hf_datasets_cache = output_dir / "hf_datasets_cache"
    xdg_cache_home = hf_home / "xdg_cache"
    hf_home.mkdir(parents=True, exist_ok=True)
    hf_datasets_cache.mkdir(parents=True, exist_ok=True)
    xdg_cache_home.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(hf_home)
    os.environ["HF_DATASETS_CACHE"] = str(hf_datasets_cache)
    os.environ["HF_DATASETS_DOWNLOADED_DATASETS_PATH"] = str(hf_datasets_cache / "downloads")
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(hf_home / "hub")
    os.environ["XDG_CACHE_HOME"] = str(xdg_cache_home)
    try:
        import datasets

        datasets.config.HF_HOME = str(hf_home)
        datasets.config.HF_DATASETS_CACHE = str(hf_datasets_cache)
        datasets.config.DOWNLOADED_DATASETS_PATH = Path(os.environ["HF_DATASETS_DOWNLOADED_DATASETS_PATH"])
    except Exception:
        pass


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


def _load_selection_payload(selection_npz: Path) -> Dict[str, np.ndarray]:
    if not selection_npz.exists():
        raise FileNotFoundError(f"selection npz not found: {selection_npz}")
    payload = np.load(selection_npz, allow_pickle=False)
    required = {
        "episode_keep_mask",
        "window_episode",
        "window_start",
        "window_end",
        "window_keep",
        "window_contribution",
    }
    missing = sorted(required - set(payload.files))
    if missing:
        raise ValueError(f"{selection_npz} is missing arrays: {missing}")
    return {name: np.asarray(payload[name]) for name in payload.files}


def _merge_selection_into_metadata(
    metadata: Dict[str, np.ndarray],
    selection_payload: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    merged = dict(metadata)
    window_keep = np.asarray(selection_payload["window_keep"], dtype=np.int64).reshape(-1)
    contribution = np.asarray(selection_payload["window_contribution"], dtype=float).reshape(-1)
    if contribution.shape[0] != window_keep.shape[0]:
        raise ValueError("Selection window_contribution shape mismatch.")

    weights = np.maximum(contribution, 0.0)
    keep_mask = window_keep.astype(bool)
    if np.any(keep_mask):
        keep_mean = float(np.mean(weights[keep_mask]))
        if keep_mean > 1e-8:
            weights = weights / keep_mean
    weights[~keep_mask] = 0.0

    merged["window_episode_index"] = np.asarray(selection_payload["window_episode"], dtype=np.int64).reshape(-1)
    merged["window_start"] = np.asarray(selection_payload["window_start"], dtype=np.int64).reshape(-1)
    merged["window_end"] = np.asarray(selection_payload["window_end"], dtype=np.int64).reshape(-1)
    merged["window_drop_mask"] = (~keep_mask).astype(np.int64)
    merged["window_weights"] = weights.astype(np.float64)
    return merged


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


def _normalize_to_unit_interval(values: np.ndarray) -> np.ndarray:
    vals = np.asarray(values, dtype=float)
    out = np.zeros_like(vals, dtype=float)
    finite = np.isfinite(vals)
    if not finite.any():
        return out
    v = vals[finite]
    v_min = float(np.min(v))
    v_max = float(np.max(v))
    if np.isclose(v_min, v_max):
        out[finite] = 1.0
        return out
    out[finite] = (v - v_min) / (v_max - v_min)
    return np.clip(out, 0.0, 1.0)


def _quality_from_bad_ratio(ep_bad: np.ndarray) -> np.ndarray:
    return np.clip(1.0 - np.asarray(ep_bad, dtype=float), 0.0, None)


def _lower_better_to_score(
    values: np.ndarray,
    normalizer: str = "quantile",
    quantile_low: float = 0.05,
    quantile_high: float = 0.95,
) -> np.ndarray:
    vals = np.asarray(values, dtype=float)
    out = np.full_like(vals, np.nan, dtype=float)
    finite = np.isfinite(vals)
    if not finite.any():
        return out
    v = vals[finite]
    if normalizer == "quantile":
        q_low = float(np.clip(quantile_low, 0.0, 1.0))
        q_high = float(np.clip(quantile_high, 0.0, 1.0))
        if q_high < q_low:
            q_low, q_high = q_high, q_low
        v_min = float(np.quantile(v, q_low))
        v_max = float(np.quantile(v, q_high))
    else:
        v_min = float(np.min(v))
        v_max = float(np.max(v))
    if np.isclose(v_min, v_max):
        out[finite] = 1.0
        return out
    out[finite] = 1.0 - (v - v_min) / (v_max - v_min)
    return np.clip(out, 0.0, 1.0)


def _higher_better_to_score(
    values: np.ndarray,
    normalizer: str = "quantile",
    quantile_low: float = 0.05,
    quantile_high: float = 0.95,
) -> np.ndarray:
    vals = np.asarray(values, dtype=float)
    out = np.full_like(vals, np.nan, dtype=float)
    finite = np.isfinite(vals)
    if not finite.any():
        return out
    v = vals[finite]
    if normalizer == "quantile":
        q_low = float(np.clip(quantile_low, 0.0, 1.0))
        q_high = float(np.clip(quantile_high, 0.0, 1.0))
        if q_high < q_low:
            q_low, q_high = q_high, q_low
        v_min = float(np.quantile(v, q_low))
        v_max = float(np.quantile(v, q_high))
    else:
        v_min = float(np.min(v))
        v_max = float(np.max(v))
    if np.isclose(v_min, v_max):
        out[finite] = 1.0
        return out
    out[finite] = (v - v_min) / (v_max - v_min)
    return np.clip(out, 0.0, 1.0)


def _quality_from_episode_eval(
    dataset_root: Path,
    all_eps: list[int],
    action_weight: float,
    state_weight: float,
    length_weight: float,
    energy_weight: float,
    length_scale: float,
    default_value: float,
    jitter_mode: str,
    normalizer: str,
    quantile_low: float,
    quantile_high: float,
) -> np.ndarray:
    try:
        import pyarrow.parquet as pq
    except Exception as e:  # pragma: no cover
        raise RuntimeError("episode_eval requires pyarrow to read parquet episodes") from e

    num_episodes = max(all_eps) + 1 if all_eps else 0
    action_jitter = np.full(num_episodes, np.nan, dtype=float)
    state_jitter = np.full(num_episodes, np.nan, dtype=float)
    action_energy = np.full(num_episodes, np.nan, dtype=float)
    state_energy = np.full(num_episodes, np.nan, dtype=float)
    episode_len = np.full(num_episodes, np.nan, dtype=float)

    episode_files: Dict[int, Path] = {}
    for p in sorted((dataset_root / "data").glob("chunk-*/episode_*.parquet")):
        stem = p.stem
        if not stem.startswith("episode_"):
            continue
        ep_str = stem.split("_")[-1]
        if ep_str.isdigit():
            episode_files[int(ep_str)] = p

    for ep in all_eps:
        p = episode_files.get(int(ep))
        if p is None:
            continue
        schema = pq.read_schema(p)
        available = set(schema.names)
        cols = [c for c in ("action", "observation.state") if c in available]
        table = pq.read_table(p, columns=cols if cols else None)
        n = int(table.num_rows)
        episode_len[ep] = float(n)
        if n < 2:
            continue

        if "action" in table.column_names:
            a = np.asarray(table["action"].to_pylist(), dtype=float)
            if a.ndim == 2 and a.shape[0] >= 2:
                action_jitter[ep] = float(np.linalg.norm(np.diff(a, axis=0), axis=1).mean())
                action_energy[ep] = float(np.linalg.norm(a, axis=1).mean())

        if "observation.state" in table.column_names:
            s = np.asarray(table["observation.state"].to_pylist(), dtype=float)
            if s.ndim == 2 and s.shape[0] >= 2:
                state_jitter[ep] = float(np.linalg.norm(np.diff(s, axis=0), axis=1).mean())
                state_energy[ep] = float(np.linalg.norm(s, axis=1).mean())

    if jitter_mode == "relative":
        action_metric = action_jitter / np.maximum(action_energy, 1e-8)
        state_metric = state_jitter / np.maximum(state_energy, 1e-8)
    else:
        action_metric = action_jitter
        state_metric = state_jitter

    action_score = _lower_better_to_score(
        action_metric,
        normalizer=normalizer,
        quantile_low=quantile_low,
        quantile_high=quantile_high,
    )
    state_score = _lower_better_to_score(
        state_metric,
        normalizer=normalizer,
        quantile_low=quantile_low,
        quantile_high=quantile_high,
    )
    energy_score = _lower_better_to_score(
        action_energy,
        normalizer=normalizer,
        quantile_low=quantile_low,
        quantile_high=quantile_high,
    )

    length_score = np.full(num_episodes, np.nan, dtype=float)
    finite_len = np.isfinite(episode_len)
    if finite_len.any():
        lens = episode_len[finite_len]
        med = float(np.median(lens))
        mad = float(np.median(np.abs(lens - med)))
        denom = max(mad * max(length_scale, 1e-6), 1e-6)
        length_score[finite_len] = np.exp(-np.abs(lens - med) / denom)
        length_score = np.clip(length_score, 0.0, 1.0)

    comp_pairs = [
        (action_score, max(action_weight, 0.0)),
        (state_score, max(state_weight, 0.0)),
        (length_score, max(length_weight, 0.0)),
        (energy_score, max(energy_weight, 0.0)),
    ]
    quality = np.full(num_episodes, np.nan, dtype=float)
    for ep in all_eps:
        num = 0.0
        den = 0.0
        for comp, w in comp_pairs:
            if w <= 0.0 or ep >= len(comp):
                continue
            val = comp[ep]
            if np.isfinite(val):
                num += w * float(val)
                den += w
        if den > 0:
            quality[ep] = num / den

    missing = ~np.isfinite(quality)
    if missing.any():
        quality[missing] = float(default_value)
    return np.clip(quality, 0.0, None)


def _quality_from_trajectory_eval(
    dataset_root: Path,
    all_eps: list[int],
    terminal_weight: float,
    efficiency_weight: float,
    progress_weight: float,
    state_smooth_weight: float,
    action_smooth_weight: float,
    length_weight: float,
    terminal_knn_k: int,
    monotonic_weight: float,
    backtrack_weight: float,
    stall_weight: float,
    fusion: str,
    min_terminal_score: float,
    min_progress_score: float,
    gate_quality_cap: float,
    length_scale: float,
    default_value: float,
    normalizer: str,
    quantile_low: float,
    quantile_high: float,
) -> np.ndarray:
    try:
        import pyarrow.parquet as pq
    except Exception as e:  # pragma: no cover
        raise RuntimeError("trajectory_eval requires pyarrow to read parquet episodes") from e

    num_episodes = max(all_eps) + 1 if all_eps else 0
    terminal_dist = np.full(num_episodes, np.nan, dtype=float)
    state_efficiency = np.full(num_episodes, np.nan, dtype=float)
    state_progress = np.full(num_episodes, np.nan, dtype=float)
    state_rel_jitter = np.full(num_episodes, np.nan, dtype=float)
    state_monotonic = np.full(num_episodes, np.nan, dtype=float)
    state_backtrack = np.full(num_episodes, np.nan, dtype=float)
    state_stall = np.full(num_episodes, np.nan, dtype=float)
    action_rel_jitter = np.full(num_episodes, np.nan, dtype=float)
    episode_len = np.full(num_episodes, np.nan, dtype=float)
    final_state_map: Dict[int, np.ndarray] = {}

    episode_files: Dict[int, Path] = {}
    for p in sorted((dataset_root / "data").glob("chunk-*/episode_*.parquet")):
        stem = p.stem
        if not stem.startswith("episode_"):
            continue
        ep_str = stem.split("_")[-1]
        if ep_str.isdigit():
            episode_files[int(ep_str)] = p

    for ep in all_eps:
        p = episode_files.get(int(ep))
        if p is None:
            continue
        schema = pq.read_schema(p)
        available = set(schema.names)
        cols = [c for c in ("action", "observation.state") if c in available]
        table = pq.read_table(p, columns=cols if cols else None)
        n = int(table.num_rows)
        episode_len[ep] = float(n)
        if n < 2:
            continue

        if "observation.state" in table.column_names:
            s = np.asarray(table["observation.state"].to_pylist(), dtype=float)
            if s.ndim == 2 and s.shape[0] >= 2:
                ds = np.diff(s, axis=0)
                ds_norm = np.linalg.norm(ds, axis=1)
                path = float(np.linalg.norm(ds, axis=1).sum())
                disp = float(np.linalg.norm(s[-1] - s[0]))
                mag = float(np.linalg.norm(s, axis=1).mean())
                state_efficiency[ep] = disp / (path + 1e-8)
                state_progress[ep] = disp
                state_rel_jitter[ep] = float(np.linalg.norm(ds, axis=1).mean() / (mag + 1e-8))
                if disp > 1e-8:
                    direction = (s[-1] - s[0]) / (disp + 1e-8)
                    proj = ds @ direction
                    forward = float(np.sum(np.clip(proj, 0.0, None)))
                    backward = float(np.sum(np.clip(-proj, 0.0, None)))
                    total_dir = forward + backward
                    if total_dir > 1e-8:
                        state_monotonic[ep] = forward / total_dir
                        state_backtrack[ep] = backward / total_dir
                    else:
                        state_monotonic[ep] = 0.0
                        state_backtrack[ep] = 1.0
                else:
                    state_monotonic[ep] = 0.0
                    state_backtrack[ep] = 1.0

                stall_thr = max(0.1 * disp / max(int(s.shape[0]) - 1, 1), 1e-6)
                state_stall[ep] = float(np.mean(ds_norm <= stall_thr))
                final_state_map[ep] = s[-1]

        if "action" in table.column_names:
            a = np.asarray(table["action"].to_pylist(), dtype=float)
            if a.ndim == 2 and a.shape[0] >= 2:
                da = np.diff(a, axis=0)
                mag = float(np.linalg.norm(a, axis=1).mean())
                action_rel_jitter[ep] = float(np.linalg.norm(da, axis=1).mean() / (mag + 1e-8))

    if final_state_map:
        valid_eps = sorted(final_state_map.keys())
        final_states = np.stack([final_state_map[ep] for ep in valid_eps], axis=0)
        if len(valid_eps) == 1:
            terminal_dist[valid_eps[0]] = 0.0
        else:
            # Use kNN terminal distance to support multi-modal valid terminal states.
            k = int(max(1, min(int(terminal_knn_k), len(valid_eps) - 1)))
            dist_mat = np.linalg.norm(final_states[:, None, :] - final_states[None, :, :], axis=2)
            np.fill_diagonal(dist_mat, np.inf)
            nearest = np.partition(dist_mat, kth=k - 1, axis=1)[:, :k]
            d = nearest.mean(axis=1)
            for ep, dist in zip(valid_eps, d):
                terminal_dist[ep] = float(dist)

    terminal_score = _lower_better_to_score(
        terminal_dist,
        normalizer=normalizer,
        quantile_low=quantile_low,
        quantile_high=quantile_high,
    )
    efficiency_score = _higher_better_to_score(
        state_efficiency,
        normalizer=normalizer,
        quantile_low=quantile_low,
        quantile_high=quantile_high,
    )
    progress_score = _higher_better_to_score(
        state_progress,
        normalizer=normalizer,
        quantile_low=quantile_low,
        quantile_high=quantile_high,
    )
    monotonic_score = _higher_better_to_score(
        state_monotonic,
        normalizer=normalizer,
        quantile_low=quantile_low,
        quantile_high=quantile_high,
    )
    backtrack_score = _lower_better_to_score(
        state_backtrack,
        normalizer=normalizer,
        quantile_low=quantile_low,
        quantile_high=quantile_high,
    )
    stall_score = _lower_better_to_score(
        state_stall,
        normalizer=normalizer,
        quantile_low=quantile_low,
        quantile_high=quantile_high,
    )
    state_smooth_score = _lower_better_to_score(
        state_rel_jitter,
        normalizer=normalizer,
        quantile_low=quantile_low,
        quantile_high=quantile_high,
    )
    action_smooth_score = _lower_better_to_score(
        action_rel_jitter,
        normalizer=normalizer,
        quantile_low=quantile_low,
        quantile_high=quantile_high,
    )

    length_score = np.full(num_episodes, np.nan, dtype=float)
    finite_len = np.isfinite(episode_len)
    if finite_len.any():
        lens = episode_len[finite_len]
        med = float(np.median(lens))
        mad = float(np.median(np.abs(lens - med)))
        denom = max(mad * max(length_scale, 1e-6), 1e-6)
        length_score[finite_len] = np.exp(-np.abs(lens - med) / denom)
        length_score = np.clip(length_score, 0.0, 1.0)

    comp_pairs = [
        (terminal_score, max(terminal_weight, 0.0)),
        (efficiency_score, max(efficiency_weight, 0.0)),
        (progress_score, max(progress_weight, 0.0)),
        (monotonic_score, max(monotonic_weight, 0.0)),
        (backtrack_score, max(backtrack_weight, 0.0)),
        (stall_score, max(stall_weight, 0.0)),
        (state_smooth_score, max(state_smooth_weight, 0.0)),
        (action_smooth_score, max(action_smooth_weight, 0.0)),
        (length_score, max(length_weight, 0.0)),
    ]
    quality = np.full(num_episodes, np.nan, dtype=float)
    for ep in all_eps:
        den = 0.0
        if fusion == "weighted_geomean":
            num_log = 0.0
            for comp, w in comp_pairs:
                if w <= 0.0 or ep >= len(comp):
                    continue
                val = comp[ep]
                if np.isfinite(val):
                    num_log += w * float(np.log(max(float(val), 1e-8)))
                    den += w
            if den > 0:
                quality[ep] = float(np.exp(num_log / den))
        else:
            num = 0.0
            for comp, w in comp_pairs:
                if w <= 0.0 or ep >= len(comp):
                    continue
                val = comp[ep]
                if np.isfinite(val):
                    num += w * float(val)
                    den += w
            if den > 0:
                quality[ep] = num / den

    if gate_quality_cap >= 0.0:
        gate_mask = np.zeros(num_episodes, dtype=bool)
        if min_terminal_score >= 0.0:
            gate_mask |= np.isfinite(terminal_score) & (terminal_score < float(min_terminal_score))
        if min_progress_score >= 0.0:
            gate_mask |= np.isfinite(progress_score) & (progress_score < float(min_progress_score))
        quality[gate_mask] = np.minimum(quality[gate_mask], float(gate_quality_cap))

    missing = ~np.isfinite(quality)
    if missing.any():
        quality[missing] = float(default_value)
    return np.clip(quality, 0.0, None)


def _quality_from_operator_csv(
    csv_path: Path,
    num_episodes: int,
    better: float,
    okay: float,
    worse: float,
    default_value: float,
) -> np.ndarray:
    if not csv_path.exists():
        raise FileNotFoundError(f"operator quality csv not found: {csv_path}")

    quality = np.full(num_episodes, np.nan, dtype=float)
    pattern = re.compile(r"^\s*(\d+)\s*-\s*(\d+)\s*$")
    assigned = 0

    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            op = str(row.get("operator", "")).strip().lower()
            rng = str(row.get("episode_range", "")).strip()
            match = pattern.match(rng)
            if not match:
                continue
            start = int(match.group(1))
            end = int(match.group(2))
            if start > end:
                continue
            if "better" in op:
                score = better
            elif "okay" in op:
                score = okay
            elif "worse" in op:
                score = worse
            else:
                continue
            start = max(start, 0)
            end = min(end, num_episodes - 1)
            if end < start:
                continue
            quality[start : end + 1] = float(score)
            assigned += (end - start + 1)

    if assigned == 0:
        raise ValueError(f"No valid operator episode ranges parsed from {csv_path}")
    missing = ~np.isfinite(quality)
    if missing.any():
        quality[missing] = float(default_value)
    return np.clip(quality, 0.0, None)


def _quality_from_quality_csv(
    csv_path: Path,
    num_episodes: int,
    default_value: float,
) -> np.ndarray:
    if not csv_path.exists():
        raise FileNotFoundError(f"quality score csv not found: {csv_path}")

    quality = np.full(num_episodes, np.nan, dtype=float)
    parsed = 0
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        required_cols = {"episode", "quality_score"}
        if not required_cols.issubset(set(reader.fieldnames or [])):
            raise ValueError(
                f"{csv_path} must contain columns {sorted(required_cols)}, got {reader.fieldnames}"
            )
        for row in reader:
            ep_raw = row.get("episode")
            q_raw = row.get("quality_score")
            try:
                ep = int(ep_raw) if ep_raw is not None else -1
                q = float(q_raw) if q_raw is not None else float("nan")
            except (TypeError, ValueError):
                continue
            if ep < 0 or ep >= num_episodes:
                continue
            if not np.isfinite(q):
                continue
            quality[ep] = q
            parsed += 1

    if parsed == 0:
        raise ValueError(f"No valid rows parsed from {csv_path}")
    missing = ~np.isfinite(quality)
    if missing.any():
        quality[missing] = float(default_value)
    return np.clip(quality, 0.0, None)


def _compute_joint_scores(
    influence_source: np.ndarray,
    quality_score: np.ndarray,
    quality_power: float,
    influence_power: float,
) -> np.ndarray:
    quality = np.clip(np.asarray(quality_score, dtype=float), 0.0, None)
    influence = _normalize_to_unit_interval(influence_source)
    q = np.power(quality, quality_power)
    i = np.power(np.clip(influence, 0.0, None), influence_power)
    return q * i


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
    filter_score_mode: str,
    joint_drop_quantile: float,
    joint_quality_source: str,
    ep_bad: np.ndarray,
    ep_weights: np.ndarray,
    weight_source: np.ndarray,
    quality_score: np.ndarray,
    joint_score: np.ndarray,
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
        "filter_score_mode": filter_score_mode,
        "joint_drop_quantile": joint_drop_quantile,
        "joint_quality_source": joint_quality_source,
        "bad_ratio_stats": _summarize_array(ep_bad),
        "weight_stats": _summarize_array(ep_weights),
        "weight_source_stats": _summarize_array(weight_source),
        "quality_score_stats": _summarize_array(quality_score),
        "joint_score_stats": _summarize_array(joint_score),
        "sample_weight_stats": _summarize_array(sample_weights) if sample_weights is not None else None,
        "sampling_weight_episode_stats": _summarize_array(sampling_weights_episode),
        "segment_mask_enabled": bool(sample_weights is not None),
    }
    report_path = output_dir / "cupid_reweight_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    csv_path = output_dir / "cupid_reweight_episodes.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "episode",
                "keep",
                "bad_ratio",
                "quality_score",
                "episode_weight",
                "weight_source",
                "joint_score",
                "sampling_weight",
            ]
        )
        for ep in all_eps:
            bad_val = ep_bad[ep] if ep < len(ep_bad) else float("nan")
            q_val = quality_score[ep] if ep < len(quality_score) else float("nan")
            w_val = ep_weights[ep] if ep < len(ep_weights) else float("nan")
            ws_val = weight_source[ep] if ep < len(weight_source) else float("nan")
            js_val = joint_score[ep] if ep < len(joint_score) else float("nan")
            sw_val = sampling_weights_episode[ep] if ep < len(sampling_weights_episode) else float("nan")
            writer.writerow([ep, int(ep in keep_set), bad_val, q_val, w_val, ws_val, js_val, sw_val])

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
    _write_ranked("cupid_top_joint_score.txt", joint_score, topk=50, reverse=True)
    _write_ranked("cupid_bottom_joint_score.txt", joint_score, topk=50, reverse=False)
    _write_ranked("cupid_top_sampling_weight.txt", sampling_weights_episode, topk=50, reverse=True)
    _write_ranked("cupid_bottom_sampling_weight.txt", sampling_weights_episode, topk=50, reverse=False)
    _write_ranked("cupid_top_bad_ratio.txt", ep_bad, topk=50, reverse=True)
    _write_ranked("cupid_bottom_bad_ratio.txt", ep_bad, topk=50, reverse=False)
    _write_ranked("cupid_top_quality_score.txt", quality_score, topk=50, reverse=True)
    _write_ranked("cupid_bottom_quality_score.txt", quality_score, topk=50, reverse=False)

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
    _write_hist("cupid_hist_quality_score.csv", quality_score)
    _write_hist("cupid_hist_weight_source.csv", weight_source)
    _write_hist("cupid_hist_joint_score.csv", joint_score)
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
    _ensure_writable_hf_cache(args.output_dir)

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

    repo_id = args.dataset_repo_id or _infer_repo_id(dataset_root)
    ds_meta = LeRobotDatasetMetadata(repo_id=repo_id, root=dataset_root, force_cache_sync=False)
    _patch_dataset_task_maps_if_needed(ds_meta, dataset_root)

    total_eps = len(ds_meta.episodes)
    all_eps = sorted(ds_meta.episodes.keys())
    num_episodes = max(max(all_eps) + 1 if all_eps else 0, total_eps)

    requires_influence = (
        args.weight_mode in ("window_mean", "influence")
        or args.filter_score_mode == "weight"
        or float(args.joint_influence_power) > 0.0
        or args.segment_mask
        or args.segment_weight
    )
    if args.score_pkl is None:
        if requires_influence:
            raise ValueError(
                "--score-pkl is required when using influence-dependent settings "
                "(weight_mode window_mean/influence, filter_score_mode=weight, "
                "joint_influence_power>0, or segment_mask/segment_weight)."
            )
        # Quality-only fallback: neutral influence and no bad windows.
        scores = np.ones(num_episodes, dtype=float)
        ep_weights = np.ones(num_episodes, dtype=float)
        ep_bad = np.zeros(num_episodes, dtype=float)
        metadata: Dict[str, np.ndarray] = {}
        print("[info] score-pkl not provided; running in quality-only mode.")
    else:
        scores, metadata = _load_influence_payload(args.score_pkl)
        ep_weights, ep_bad = _compute_episode_stats(scores, metadata)

    selection_payload = None
    selection_keep_eps = None
    if args.selection_npz is not None:
        selection_payload = _load_selection_payload(args.selection_npz)
        metadata = _merge_selection_into_metadata(metadata, selection_payload)
        ep_weights, ep_bad = _compute_episode_stats(scores, metadata)
        episode_keep_mask = np.asarray(selection_payload["episode_keep_mask"], dtype=np.int64).reshape(-1)
        selection_keep_eps = [ep for ep in all_eps if ep < len(episode_keep_mask) and bool(episode_keep_mask[ep])]

    keep_eps = [ep for ep in all_eps if ep < len(ep_bad) and ep_bad[ep] < args.drop_bad_ratio]

    weight_source = ep_weights if args.weight_mode != "none" else scores
    weight_source = np.asarray(weight_source, dtype=float)
    if args.joint_quality_source == "episode_eval":
        quality_score = _quality_from_episode_eval(
            dataset_root=dataset_root,
            all_eps=all_eps,
            action_weight=args.quality_eval_action_weight,
            state_weight=args.quality_eval_state_weight,
            length_weight=args.quality_eval_length_weight,
            energy_weight=args.quality_eval_energy_weight,
            length_scale=args.quality_eval_length_scale,
            default_value=args.quality_eval_default,
            jitter_mode=args.quality_eval_jitter_mode,
            normalizer=args.quality_eval_normalizer,
            quantile_low=args.quality_eval_quantile_low,
            quantile_high=args.quality_eval_quantile_high,
        )
    elif args.joint_quality_source == "trajectory_eval":
        quality_score = _quality_from_trajectory_eval(
            dataset_root=dataset_root,
            all_eps=all_eps,
            terminal_weight=args.traj_eval_terminal_weight,
            efficiency_weight=args.traj_eval_efficiency_weight,
            progress_weight=args.traj_eval_progress_weight,
            state_smooth_weight=args.traj_eval_state_smooth_weight,
            action_smooth_weight=args.traj_eval_action_smooth_weight,
            length_weight=args.traj_eval_length_weight,
            terminal_knn_k=args.traj_eval_terminal_knn_k,
            monotonic_weight=args.traj_eval_monotonic_weight,
            backtrack_weight=args.traj_eval_backtrack_weight,
            stall_weight=args.traj_eval_stall_weight,
            fusion=args.traj_eval_fusion,
            min_terminal_score=args.traj_eval_min_terminal_score,
            min_progress_score=args.traj_eval_min_progress_score,
            gate_quality_cap=args.traj_eval_gate_quality_cap,
            length_scale=args.quality_eval_length_scale,
            default_value=args.quality_eval_default,
            normalizer=args.quality_eval_normalizer,
            quantile_low=args.quality_eval_quantile_low,
            quantile_high=args.quality_eval_quantile_high,
        )
    elif args.joint_quality_source == "operator_csv":
        if args.operator_quality_csv is None:
            raise ValueError("joint_quality_source=operator_csv requires --operator-quality-csv")
        quality_score = _quality_from_operator_csv(
            csv_path=args.operator_quality_csv,
            num_episodes=max(max(all_eps) + 1 if all_eps else 0, len(weight_source), len(ep_bad)),
            better=args.operator_quality_better,
            okay=args.operator_quality_okay,
            worse=args.operator_quality_worse,
            default_value=args.operator_quality_default,
        )
    elif args.joint_quality_source == "quality_csv":
        if args.quality_score_csv is None:
            raise ValueError("joint_quality_source=quality_csv requires --quality-score-csv")
        quality_score = _quality_from_quality_csv(
            csv_path=args.quality_score_csv,
            num_episodes=max(max(all_eps) + 1 if all_eps else 0, len(weight_source), len(ep_bad)),
            default_value=args.quality_score_default,
        )
    else:
        quality_score = _quality_from_bad_ratio(ep_bad)

    joint_score = _compute_joint_scores(
        influence_source=weight_source,
        quality_score=quality_score,
        quality_power=args.joint_quality_power,
        influence_power=args.joint_influence_power,
    )

    if args.filter_score_mode == "joint":
        if args.joint_drop_quantile > 0:
            keep_joint = np.asarray([joint_score[ep] for ep in keep_eps if ep < len(joint_score)], dtype=float)
            if keep_joint.size == 0:
                raise RuntimeError("No episodes left after bad-ratio filtering for joint score thresholding.")
            thresh = float(np.quantile(keep_joint, args.joint_drop_quantile))
            keep_eps = [ep for ep in keep_eps if ep < len(joint_score) and joint_score[ep] > thresh]
            print(f"[filter] joint quantile threshold={thresh:.6g} (q={args.joint_drop_quantile})")
    elif args.drop_weight_quantile > 0:
        keep_weights = np.asarray([weight_source[ep] for ep in keep_eps if ep < len(weight_source)], dtype=float)
        if keep_weights.size == 0:
            raise RuntimeError("No episodes left after bad-ratio filtering for weight thresholding.")
        thresh = float(np.quantile(keep_weights, args.drop_weight_quantile))
        keep_eps = [ep for ep in keep_eps if ep < len(weight_source) and weight_source[ep] > thresh]
        print(f"[filter] weight quantile threshold={thresh:.6g} (q={args.drop_weight_quantile})")

    if selection_keep_eps is not None:
        selection_keep_set = set(selection_keep_eps)
        if args.selection_override_episodes:
            keep_eps = [ep for ep in all_eps if ep in selection_keep_set]
            print(f"[filter] overriding episode filter with selection-npz ({len(keep_eps)} episodes kept)")
        else:
            keep_eps = [ep for ep in keep_eps if ep in selection_keep_set]
            print(f"[filter] intersected with selection-npz ({len(keep_eps)} episodes kept)")

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
    _patch_dataset_task_maps_if_needed(dataset.meta, dataset_root)

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
        args.filter_score_mode,
        args.joint_drop_quantile,
        args.joint_quality_source,
        ep_bad,
        ep_weights,
        weight_source,
        quality_score,
        joint_score,
        sample_weights,
        sampling_weights_episode,
    )
    print(
        f"[filter] kept {len(keep_eps)}/{total_eps} episodes "
        f"(bad_ratio<{args.drop_bad_ratio}, filter_score_mode={args.filter_score_mode}, "
        f"quality_source={args.joint_quality_source}, weight_mode={args.weight_mode})"
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

    step = _infer_resume_step(args.pretrained_dir, args.resume_step)
    if args.pretrained_dir is not None:
        state_path = args.pretrained_dir / "training_state.pt"
        if state_path.is_file():
            state = torch.load(state_path, map_location="cpu")
            optimizer_state = state.get("optimizer") if isinstance(state, dict) else None
            if optimizer_state is not None:
                optimizer.load_state_dict(optimizer_state)
                print(f"[resume] loaded optimizer state from {state_path}")
    if step > 0:
        print(f"[resume] starting from global step {step}")

    env_state_dim = _infer_env_state_dim_from_policy(policy)
    if env_state_dim > 0:
        print(f"[multitask] policy expects observation.environment_state, dim={env_state_dim}")

    while step < args.max_steps:
        for batch in dataloader:
            batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            batch = _augment_batch_environment_state(batch, env_state_dim)
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
                torch.save({"step": step, "optimizer": optimizer.state_dict()}, ckpt_dir / "training_state.pt")
                print(f"[checkpoint] saved to {ckpt_dir}")

            step += 1
            if step >= args.max_steps:
                break

    policy.save_pretrained(args.output_dir)
    torch.save({"step": step, "optimizer": optimizer.state_dict()}, args.output_dir / "training_state.pt")
    print(f"[done] saved to {args.output_dir}")


if __name__ == "__main__":
    main()
