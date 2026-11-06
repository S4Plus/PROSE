#!/usr/bin/env python
"""
在验证数据上搜索插值权重，合并多个 ACT checkpoint（Model Soup）。

支持三种模式：
- uniform：等权平均
- greedy：贪心 Soup（依次尝试加入新模型，只有验证 loss 下降才保留）
- grid：最多 3 个模型，按给定步长在 simplex 上网格搜索
- learned：梯度下降优化权重（lambda），冻结模型参数，仅训练插值系数

用法示例：
python libero/merge_act_soup.py \
  --checkpoints output/act/select5 output/act/random20 \
  --method learned \
  --learn-steps 50 \
  --learn-lr 1e-2 \
  --repo-id local/lerobot_random5  \
  --lerobot-home ./datasets \
  --out ./output/act/soup_select25 \
  --max-batches 20 \
  --batch-size 8
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
from safetensors.torch import load_file
from torch.nn.utils.stateless import functional_call


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


def load_state_dict_from_checkpoint(path: Path) -> Dict[str, torch.Tensor]:
    """从 checkpoint 目录读取 state dict（优先 safetensors）。"""
    if not path.is_dir():
        raise FileNotFoundError(f"Checkpoint 目录不存在: {path}")

    safetensors_path = path / "model.safetensors"
    pt_path = path / "pytorch_model.bin"

    if safetensors_path.exists():
        state = load_file(safetensors_path)
    elif pt_path.exists():
        state = torch.load(pt_path, map_location="cpu")
    else:
        raise FileNotFoundError(f"{path} 下未找到 model.safetensors 或 pytorch_model.bin")

    return {k: v.cpu() for k, v in state.items()}


def average_state_dict(states: Sequence[Dict[str, torch.Tensor]], weights: Sequence[float]) -> Dict[str, torch.Tensor]:
    """按给定权重线性插值参数。"""
    if len(states) != len(weights):
        raise ValueError("states 和 weights 长度不一致")
    weights = [float(w) for w in weights]
    total = sum(weights)
    if total <= 0:
        raise ValueError("权重和必须为正")
    weights = [w / total for w in weights]

    merged: Dict[str, torch.Tensor] = {}
    keys = states[0].keys()
    for key in keys:
        acc = None
        for state, w in zip(states, weights):
            tensor = state[key]
            acc = tensor.mul(w) if acc is None else acc.add(tensor, alpha=w)
        merged[key] = acc
    return merged


def move_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}


def evaluate_policy(policy, dataloader, device: torch.device, max_batches: int) -> float:
    """在验证集上计算平均 loss。"""
    policy.eval()
    total_loss = 0.0
    n = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            batch = move_to_device(batch, device)
            try:
                loss, _ = policy.forward(batch)
            except TypeError as e:
                # 某些检查点配置 use_vae=True 但返回 None，会在 KL 处触发 NoneType 报错，降级为仅 L1 评估
                if "NoneType" in str(e):
                    orig = policy.config.use_vae
                    policy.config.use_vae = False
                    loss, _ = policy.forward(batch)
                    policy.config.use_vae = orig
                else:
                    raise
            total_loss += loss.item()
            n += 1
            if max_batches > 0 and n >= max_batches:
                break
    policy.train()
    return total_loss / max(1, n)


def greedy_soup(
    base_state: Dict[str, torch.Tensor],
    other_states: List[Dict[str, torch.Tensor]],
    policy,
    dataloader,
    device: torch.device,
    max_batches: int,
) -> Tuple[Dict[str, torch.Tensor], List[int]]:
    """贪心加入模型，只在验证 loss 更优时保留。"""
    selected = [base_state]
    selected_ids = [0]

    def set_and_eval(states: List[Dict[str, torch.Tensor]]) -> float:
        merged = average_state_dict(states, [1.0] * len(states))
        policy.load_state_dict(merged)
        return evaluate_policy(policy, dataloader, device, max_batches)

    best_loss = set_and_eval(selected)
    print(f"[greedy] 起始模型 loss={best_loss:.6f}")

    for idx, state in enumerate(other_states, start=1):
        candidate_states = selected + [state]
        merged = average_state_dict(candidate_states, [1.0] * len(candidate_states))
        policy.load_state_dict(merged)
        loss = evaluate_policy(policy, dataloader, device, max_batches)
        print(f"[greedy] 尝试加入模型 {idx} -> loss={loss:.6f}")
        if loss <= best_loss:
            selected.append(state)
            selected_ids.append(idx)
            best_loss = loss
            print(f"[greedy] 接受模型 {idx}，新的 loss={best_loss:.6f}")
        else:
            print(f"[greedy] 拒绝模型 {idx}")

    merged = average_state_dict(selected, [1.0] * len(selected))
    return merged, selected_ids


def grid_soup(
    states: List[Dict[str, torch.Tensor]],
    policy,
    dataloader,
    device: torch.device,
    max_batches: int,
    step: float,
) -> Tuple[Dict[str, torch.Tensor], List[float]]:
    """针对最多 3 个模型，在 simplex 上网格搜索权重。"""
    n = len(states)
    if n == 1:
        return states[0], [1.0]
    if n > 3:
        raise ValueError("grid 模式仅支持最多 3 个模型")

    best_loss = float("inf")
    best_w: List[float] = []

    def try_weights(ws: List[float]):
        nonlocal best_loss, best_w
        merged = average_state_dict(states, ws)
        policy.load_state_dict(merged)
        loss = evaluate_policy(policy, dataloader, device, max_batches)
        if loss < best_loss:
            best_loss = loss
            best_w = ws

    step = float(step)
    if n == 2:
        alpha = 0.0
        while alpha <= 1.0 + 1e-6:
            try_weights([alpha, 1 - alpha])
            alpha += step
    elif n == 3:
        a = 0.0
        while a <= 1.0 + 1e-6:
            b = 0.0
            while b <= 1.0 - a + 1e-6:
                c = 1.0 - a - b
                try_weights([a, b, c])
                b += step
            a += step

    merged = average_state_dict(states, best_w)
    print(f"[grid] 最优权重 {best_w}，loss={best_loss:.6f}")
    return merged, best_w


def uniform_soup(states: List[Dict[str, torch.Tensor]]) -> Tuple[Dict[str, torch.Tensor], List[float]]:
    w = [1.0] * len(states)
    merged = average_state_dict(states, w)
    return merged, [1.0 / len(states)] * len(states)


def learned_soup(
    states: List[Dict[str, torch.Tensor]],
    base_policy,
    dataloader,
    device: torch.device,
    max_batches: int,
    steps: int,
    lr: float,
) -> Tuple[Dict[str, torch.Tensor], List[float]]:
    """梯度下降学习权重（lambda），冻结模型参数，仅优化线性组合系数。"""
    base_state = base_policy.state_dict()
    param_keys = set(states[0].keys())
    if any(set(st.keys()) != param_keys for st in states[1:]):
        raise ValueError("所有 checkpoint 参数键必须一致，才能进行 learned soup")

    # 预取验证 batch，加速并保证稳定性
    cached_batches = []
    for idx, batch in enumerate(dataloader):
        cached_batches.append(move_to_device(batch, device))
        if max_batches > 0 and len(cached_batches) >= max_batches:
            break
    if not cached_batches:
        raise ValueError("验证集为空，无法优化权重")

    raw_weights = torch.zeros(len(states), device=device, requires_grad=True)
    optimizer = torch.optim.Adam([raw_weights], lr=lr)

    best_loss = float("inf")
    best_w = None

    def build_params(weights: torch.Tensor) -> Dict[str, torch.Tensor]:
        params: Dict[str, torch.Tensor] = {}
        for key, buf in base_state.items():
            if key in param_keys:
                tensors = [st[key].to(device) for st in states]
                merged = sum(w * t for w, t in zip(weights, tensors))
                params[key] = merged
            else:
                params[key] = buf.to(device)
        return params

    for step in range(steps):
        optimizer.zero_grad()
        lambdas = torch.softmax(raw_weights, dim=0)
        params = build_params(lambdas)

        total = 0.0
        for batch in cached_batches:
            try:
                loss, _ = functional_call(base_policy, params, (batch,), strict=False)
            except TypeError as e:
                if "NoneType" in str(e):
                    # 若 VAE 输出为空导致报错，强制关闭 VAE 分支，仅用 L1 loss
                    orig_vae = base_policy.config.use_vae
                    base_policy.config.use_vae = False
                    base_policy.model.config.use_vae = False
                    loss, _ = functional_call(base_policy, params, (batch,), strict=False)
                    base_policy.config.use_vae = orig_vae
                    base_policy.model.config.use_vae = orig_vae
                else:
                    raise
            total = total + loss
        loss = total / len(cached_batches)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            loss_val = loss.item()
            print(f"[learned] step {step} loss={loss_val:.6f} weights={lambdas.detach().cpu().tolist()}")
            if loss_val < best_loss:
                best_loss = loss_val
                best_w = lambdas.detach().cpu().tolist()

    if best_w is None:
        best_w = torch.softmax(raw_weights.detach(), dim=0).cpu().tolist()
    merged_state = average_state_dict(states, best_w)
    return merged_state, best_w


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="在验证集上搜索插值权重，合并 ACT 模型")
    p.add_argument("--checkpoints", nargs="+", required=True, help="多个 ACT checkpoint 目录（支持 step_*）")
    p.add_argument("--method", choices=["uniform", "greedy", "grid", "learned"], default="greedy", help="插值搜索方式")
    p.add_argument("--grid-step", type=float, default=0.2, help="grid 模式的步长（仅 n<=3）")
    p.add_argument("--learn-steps", type=int, default=50, help="learned 模式的梯度步数")
    p.add_argument("--learn-lr", type=float, default=1e-2, help="learned 模式的学习率")
    p.add_argument("--disable-vae", action="store_true", help="推理/合并时关闭 VAE（use_vae=False）")
    p.add_argument("--repo-id", type=str, default="local/libero-act", help="验证用数据集 ID")
    p.add_argument(
        "--lerobot-home",
        type=Path,
        default=REPO_ROOT / "outputs" / "libero_lerobot",
        help="LEROBOT_HOME 路径，需与数据预处理一致",
    )
    p.add_argument("--device", type=str, default="cuda", help="cuda 或 cpu")
    p.add_argument("--batch-size", type=int, default=16, help="验证 batch size")
    p.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    p.add_argument("--max-batches", type=int, default=20, help="验证批次数上限（加速搜索，<=0 表示全量）")
    p.add_argument("--out", type=Path, required=True, help="合并后模型输出目录")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ["HF_LEROBOT_HOME"] = str(args.lerobot_home.resolve())

    sys.path.insert(0, str(LEROBOT_DIR / "src"))
    from lerobot.configs.types import FeatureType  # type: ignore  # noqa: E402
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata  # type: ignore  # noqa: E402
    from lerobot.datasets.utils import dataset_to_policy_features  # type: ignore  # noqa: E402
    from lerobot.policies.act.modeling_act import ACTPolicy  # type: ignore  # noqa: E402

    ckpt_paths = [Path(p) for p in args.checkpoints]
    print(f"加载 {len(ckpt_paths)} 个 checkpoint: {ckpt_paths}")

    # 读取 state dict
    states = [load_state_dict_from_checkpoint(p) for p in ckpt_paths]

    # 用第一个 checkpoint 初始化模型与配置
    base_policy = ACTPolicy.from_pretrained(ckpt_paths[0])
    if args.disable_vae:
        base_policy.config.use_vae = False
        base_policy.model.config.use_vae = False

    dataset_dir = Path(os.environ["HF_LEROBOT_HOME"]) / args.repo_id
    if not dataset_dir.exists():
        raise FileNotFoundError(f"未找到验证数据集: {dataset_dir}")

    # 构建验证集
    ds_meta = LeRobotDatasetMetadata(args.repo_id, root=dataset_dir, force_cache_sync=False)
    policy_features = dataset_to_policy_features(ds_meta.features)
    output_features = {k: ft for k, ft in policy_features.items() if ft.type is FeatureType.ACTION}
    input_features = {k: ft for k, ft in policy_features.items() if k not in output_features}
    delta_timestamps = build_delta_timestamps(policy_features, base_policy.config, ds_meta.fps)

    val_dataset = LeRobotDataset(
        args.repo_id,
        delta_timestamps=delta_timestamps,
        root=dataset_dir,
        episodes=None,
        force_cache_sync=False,
    )
    dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device != "cpu",
        drop_last=True,
    )

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    base_policy.to(device)

    merged_state: Dict[str, torch.Tensor]
    used_weights: List[float] | List[int]
    if args.method == "uniform":
        merged_state, used_weights = uniform_soup(states)
        print(f"[uniform] 使用等权重 {used_weights}")
    elif args.method == "greedy":
        merged_state, selected_ids = greedy_soup(states[0], states[1:], base_policy, dataloader, device, args.max_batches)
        used_weights = [1.0] * len(selected_ids)
        print(f"[greedy] 选择的模型索引: {selected_ids}")
    elif args.method == "learned":
        merged_state, used_weights = learned_soup(
            states,
            base_policy,
            dataloader,
            device,
            args.max_batches,
            args.learn_steps,
            args.learn_lr,
        )
        print(f"[learned] 最优权重: {used_weights}")
    else:
        merged_state, used_weights = grid_soup(states, base_policy, dataloader, device, args.max_batches, args.grid_step)

    # 保存合并模型
    base_policy.load_state_dict(merged_state)
    if args.disable_vae:
        base_policy.config.use_vae = False
        base_policy.model.config.use_vae = False
    args.out.mkdir(parents=True, exist_ok=True)
    base_policy.save_pretrained(args.out)
    print(f"✅ 已保存合并模型到 {args.out}")
    print(f"使用权重/选择: {used_weights}")


if __name__ == "__main__":
    main()
