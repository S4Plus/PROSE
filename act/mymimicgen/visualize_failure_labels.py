#!/usr/bin/env python3
"""
根据 failure_labels_pos.csv 可视化轨迹分布与阶段失败占比。

输入 CSV 需包含列：label, pkl_path（默认来自 classify_failures_from_positions.py）
Pkl 默认键：raw_obs[*]["SquareNut_pos"] / raw_obs[*]["peg1_pos"]

输出：
  - <prefix>_nut_xy.png       初始 nut XY 覆盖
  - <prefix>_peg_xy.png       初始 peg XY 覆盖
  - <prefix>_delta_xy.png     初始 nut-peg XY 相对位移覆盖
  - <prefix>_r_hist.png       初始 XY 距离分布
  - <prefix>_label_bar.png    阶段成功/失败柱状图
"""

from __future__ import annotations

import argparse
import csv
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

# 若 CSV 中 label 为中文，可在此映射为英文以避免字体缺失警告
LABEL_MAP = {
    "抓取失败": "grasp_fail",
    "搬运掉落": "drop_during_transport",
    "插入失败": "insertion_fail",
    "成功": "success",
    "失败": "failure",
}


def normalize_label(raw: str) -> str:
    """Best-effort将中文或带空格的标签映射为英文，默认返回原值。"""
    raw = raw.strip()
    if not raw:
        return raw
    if raw in LABEL_MAP:
        return LABEL_MAP[raw]
    # 子串匹配
    if "抓" in raw:
        return "grasp_fail"
    if "搬" in raw or "掉" in raw:
        return "drop_during_transport"
    if "插" in raw or "对齐" in raw:
        return "insertion_fail"
    if "成" in raw:
        return "success"
    if "败" in raw:
        return "failure"
    return raw


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="可视化 failure_labels_pos.csv 的分布覆盖与失败占比。")
    p.add_argument("--labels-csv", type=Path, required=True, help="failure_labels_pos.csv 路径")
    p.add_argument("--outfile-prefix", type=Path, default=None, help="输出前缀（默认同 CSV 目录下的 prefix）")
    p.add_argument("--nut-key", type=str, default="SquareNut_pos", help="nut 位置键名")
    p.add_argument("--peg-key", type=str, default="peg1_pos", help="peg 位置键名")
    p.add_argument("--limit", type=int, default=None, help="最多读取多少条（调试可用）")
    p.add_argument(
        "--value-column",
        type=str,
        default=None,
        help="可选：CSV 中用于加权的列名（如 influence/score），默认使用计数。",
    )
    p.add_argument(
        "--score-pkl",
        type=Path,
        default=None,
        help="可选：score.pkl 路径，若提供则按 episode_index 读取影响分数作为权重。",
    )
    p.add_argument("--gridsize", type=int, default=20, help="热力图网格分辨率 (default: 20)")
    return p.parse_args()


def load_initial_xy(pkl_path: Path, nut_key: str, peg_key: str) -> Tuple[np.ndarray, np.ndarray]:
    with pkl_path.open("rb") as f:
        data = pickle.load(f)
    raw = data.get("raw_obs") or data.get("raw") or []
    if not raw:
        raise ValueError("raw_obs 缺失")
    nut0 = np.asarray(raw[0][nut_key])[:2]
    peg0 = np.asarray(raw[0][peg_key])[:2]
    return nut0, peg0


def plot_hexbin(
    xy: np.ndarray, title: str, outfile: Path, gridsize: int = 20, weights: np.ndarray | None = None
) -> None:
    """将 XY 平面划分为 gridsize x gridsize 网格，显示计数或“平均权重”热力图。"""
    plt.figure(figsize=(5, 4.2))
    
    if weights is not None:
        # 计算每个格子的权重总和与计数
        w_valid = np.where(np.isfinite(weights), weights, 0.0)
        mask = np.isfinite(weights).astype(float)
        sum_w, xedges, yedges = np.histogram2d(xy[:, 0], xy[:, 1], bins=gridsize, weights=w_valid)
        cnt, _, _ = np.histogram2d(xy[:, 0], xy[:, 1], bins=gridsize, weights=mask)
        
        with np.errstate(invalid="ignore", divide="ignore"):
            hist = sum_w / cnt
        hist[cnt == 0] = np.nan  # 无样本的格子设为 nan 显示为空白
        cmap = "coolwarm"
        cbar_label = "mean(influence/score)"
    else:
        # 仅显示计数
        hist, xedges, yedges = np.histogram2d(xy[:, 0], xy[:, 1], bins=gridsize)
        cmap = "YlOrRd"
        cbar_label = "count"

    # 绘制背景网格热力图
    # shading='flat' 配合 xedges, yedges 绘制格子
    mesh = plt.pcolormesh(xedges, yedges, hist.T, cmap=cmap, shading="flat")
    cb = plt.colorbar(mesh, label=cbar_label)
    
    # 在热力图上方叠加原始散点，帮助识别数据密度
    plt.scatter(xy[:, 0], xy[:, 1], c="black", s=1, alpha=0.15, label="samples")

    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.title(title)
    plt.gca().set_facecolor("#fcfcfc")
    plt.tight_layout()
    outfile.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outfile, dpi=220, facecolor="white")
    plt.close()
    print(f"saved {outfile}")


def plot_hist(values: np.ndarray, title: str, outfile: Path, bins: int = 40) -> None:
    plt.figure(figsize=(5, 4))
    plt.hist(values, bins=bins, color="#ffb703", edgecolor="#fb8500", alpha=0.9)
    plt.xlabel("distance (m)")
    plt.ylabel("count")
    plt.title(title)
    plt.tight_layout()
    outfile.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outfile, dpi=220, facecolor="white")
    plt.close()
    print(f"saved {outfile}")


def plot_bar(counts: Dict[str, int], outfile: Path, title: str = "Failure breakdown") -> None:
    labels = list(counts.keys())
    vals = [counts[k] for k in labels]
    plt.figure(figsize=(6, 4))
    plt.bar(labels, vals, color="#ffb703", edgecolor="#d1495b")
    plt.ylabel("count")
    plt.title(title)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    outfile.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outfile, dpi=220, facecolor="white")
    plt.close()
    print(f"saved {outfile}")


def read_labels(
    csv_path: Path,
    limit: int | None,
    value_col: str | None,
    score_map: Dict[int, float] | None,
) -> List[Tuple[str, Path, float | None]]:
    rows: List[Tuple[str, Path, float | None]] = []
    with csv_path.open("r") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if limit is not None and i >= limit:
                break
            raw_label = row.get("label", "").strip()
            label = normalize_label(raw_label)
            pkl_raw = row.get("pkl_path", "").strip()
            if not pkl_raw:
                continue
            pkl_path = Path(pkl_raw)
            # 只接受文件路径
            if not pkl_path.is_file():
                print(f"[warn] skip non-file pkl_path: {pkl_raw}")
                continue
            value = None
            if value_col and value_col in row and row[value_col].strip():
                try:
                    value = float(row[value_col])
                except ValueError:
                    value = None
            # 若 score_map 提供，则按 episode_index 优先使用
            if score_map is not None:
                ep_raw = row.get("episode_index")
                if ep_raw is not None:
                    try:
                        ep_idx = int(ep_raw)
                        if ep_idx in score_map:
                            value = score_map[ep_idx]
                    except ValueError:
                        pass
            rows.append((label, pkl_path, value))
    return rows


def load_scores(score_pkl: Path) -> Dict[int, float]:
    """从 score.pkl 读取 influence/score，返回 {episode_index: score}，兼容 numpy._core 前缀。"""

    def _load_with_remap(path: Path):
        try:
            with path.open("rb") as f:
                return pickle.load(f)
        except ModuleNotFoundError as e:
            # 兼容 numpy._core.* 的旧前缀
            class RenamingUnpickler(pickle.Unpickler):
                def find_class(self, module, name):  # type: ignore[override]
                    if module.startswith("numpy._core"):
                        module = module.replace("numpy._core", "numpy.core", 1)
                    return super().find_class(module, name)

            with path.open("rb") as f:
                return RenamingUnpickler(f).load()

    data = _load_with_remap(score_pkl)
    scores = {}
    # 支持两种常见结构：list of dict 或 dict of arrays
    if isinstance(data, list):
        for item in data:
            if not isinstance(item, dict):
                continue
            ep = item.get("episode_index")
            if ep is None:
                continue
            val = item.get("score") or item.get("influence") or item.get("value")
            if val is None:
                continue
            try:
                scores[int(ep)] = float(val)
            except Exception:
                continue
    elif isinstance(data, dict):
        # 兼容用户提供的 score.pkl 结构: {'influence_scores': array(...)}
        target_key = None
        for k in ["influence_scores", "score", "influence", "value"]:
            if k in data:
                target_key = k
                break
        
        if target_key:
            vals = data[target_key]
            eps = data.get("episode_index")
            try:
                if eps is not None:
                    for ep, v in zip(eps, vals):
                        scores[int(ep)] = float(v)
                else:
                    # 如果没有显式的 episode_index，假设索引就是 ID
                    for i, v in enumerate(vals):
                        scores[i] = float(v)
            except Exception as e:
                print(f"[warn] Failed to parse dict scores: {e}")
    return scores


def main() -> None:
    args = parse_args()
    csv_path = args.labels_csv.expanduser().resolve()
    if not csv_path.is_file():
        raise SystemExit(f"CSV 不存在: {csv_path}")
    score_map = load_scores(args.score_pkl.expanduser().resolve()) if args.score_pkl else None
    rows = read_labels(csv_path, args.limit, args.value_column, score_map)
    if not rows:
        raise SystemExit("CSV 中无有效行")

    # 输出前缀
    if args.outfile_prefix:
        prefix = args.outfile_prefix.expanduser().resolve()
    else:
        prefix = csv_path.with_suffix("")

    nut_list = []
    peg_list = []
    label_counts: Dict[str, int] = {}
    weights: List[float] = []
    weight_valid: List[bool] = []

    for label, pkl_path, val in rows:
        label_counts[label] = label_counts.get(label, 0) + 1
        try:
            nut0, peg0 = load_initial_xy(pkl_path, args.nut_key, args.peg_key)
            nut_list.append(nut0)
            peg_list.append(peg0)
            weights.append(val if val is not None else np.nan)
            weight_valid.append(val is not None)
        except Exception as e:
            print(f"[warn] {pkl_path.name}: {e}")
            continue

    if not nut_list:
        raise SystemExit("未能解析任何位姿")

    nut_xy = np.stack(nut_list)
    peg_xy = np.stack(peg_list)
    delta_xy = nut_xy - peg_xy
    r = np.linalg.norm(delta_xy, axis=1)
    w = None
    if weight_valid and any(weight_valid):
        w = np.asarray(weights)

    plot_hexbin(nut_xy, "Nut XY (start)", prefix.parent / f"{prefix.name}_nut_xy.png", weights=w, gridsize=args.gridsize)
    plot_hexbin(peg_xy, "Peg XY", prefix.parent / f"{prefix.name}_peg_xy.png", weights=w, gridsize=args.gridsize)
    plot_hexbin(delta_xy, "Nut-Peg ΔXY", prefix.parent / f"{prefix.name}_delta_xy.png", weights=w, gridsize=args.gridsize)
    plot_hist(r, "ΔXY distance", prefix.parent / f"{prefix.name}_r_hist.png")
    plot_bar(label_counts, prefix.parent / f"{prefix.name}_label_bar.png", title="Failure breakdown")


if __name__ == "__main__":
    main()
