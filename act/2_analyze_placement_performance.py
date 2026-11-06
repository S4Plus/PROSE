#!/usr/bin/env python3
import argparse
import pickle
import sys
from pathlib import Path
from typing import List, Tuple

import h5py
import numpy as np
import pandas as pd

# Official Environment Limits
LIMITS = {
    "nut": {"x": (-0.115, 0.115), "y": (-0.255, 0.255)},
    "peg": {"x": (-0.1, 0.3), "y": (-0.2, 0.2)},
}


def load_scores(score_pkl: Path) -> np.ndarray:
    """Load influence scores from score.pkl."""
    print(f"Loading scores from {score_pkl}...")
    sys.modules.setdefault("numpy._core", np.core)
    sys.modules.setdefault("numpy._core.numeric", np.core.numeric)
    with score_pkl.open("rb") as f:
        payload = pickle.load(f)
    if isinstance(payload, dict) and "influence_scores" in payload:
        return np.asarray(payload["influence_scores"])
    if isinstance(payload, dict) and "train" in payload and "influence_scores" in payload["train"]:
        return np.asarray(payload["train"]["influence_scores"])
    raise ValueError("Could not find influence_scores in pickle.")


def quat_to_yaw(q) -> float:
    """Z-axis (yaw) extraction from quaternion."""
    if len(q) == 4:
        x, y, z, w = q
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        return float(np.arctan2(siny_cosp, cosy_cosp))
    return 0.0


def safe_zscore(series: pd.Series) -> pd.Series:
    std = series.std(ddof=0)
    if std == 0 or np.isnan(std):
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - series.mean()) / std


def build_edges_from_data(series: pd.Series, bins: int, pad_ratio: float = 0.02) -> np.ndarray:
    vals = series.dropna().to_numpy()
    if vals.size == 0:
        return np.array([-1.0, 1.0])
    min_val = float(np.min(vals))
    max_val = float(np.max(vals))
    if np.isclose(min_val, max_val):
        min_val -= 1e-3
        max_val += 1e-3
    else:
        pad = max((max_val - min_val) * pad_ratio, 1e-6)
        min_val -= pad
        max_val += pad
    return np.linspace(min_val, max_val, bins + 1)


def build_grid_stats(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    alpha: float,
) -> pd.DataFrame:
    tmp = df[[x_col, y_col, "score"]].copy()
    tmp["x_bin"] = pd.cut(tmp[x_col], x_edges, labels=False, include_lowest=True)
    tmp["y_bin"] = pd.cut(tmp[y_col], y_edges, labels=False, include_lowest=True)
    tmp = tmp.dropna(subset=["x_bin", "y_bin"])
    tmp["x_bin"] = tmp["x_bin"].astype(int)
    tmp["y_bin"] = tmp["y_bin"].astype(int)
    stats = tmp.groupby(["x_bin", "y_bin"], observed=False)["score"].agg(["mean", "count"]).reset_index()
    if stats.empty:
        return stats
    stats["priority"] = safe_zscore(stats["mean"]) + alpha * safe_zscore(1.0 / stats["count"])
    stats["x_low"] = stats["x_bin"].apply(lambda i: float(x_edges[i]))
    stats["x_high"] = stats["x_bin"].apply(lambda i: float(x_edges[i + 1]))
    stats["y_low"] = stats["y_bin"].apply(lambda i: float(y_edges[i]))
    stats["y_high"] = stats["y_bin"].apply(lambda i: float(y_edges[i + 1]))
    stats["x_center"] = (stats["x_low"] + stats["x_high"]) / 2.0
    stats["y_center"] = (stats["y_low"] + stats["y_high"]) / 2.0
    return stats


def list_empty_bins(x_edges: np.ndarray, y_edges: np.ndarray, stats: pd.DataFrame) -> List[Tuple[int, int]]:
    total_x = len(x_edges) - 1
    total_y = len(y_edges) - 1
    existing = set(zip(stats["x_bin"].astype(int), stats["y_bin"].astype(int)))
    empty = []
    for i in range(total_x):
        for j in range(total_y):
            if (i, j) not in existing:
                empty.append((i, j))
    return empty


def clamp_center(cx: float, cy: float, obj_type: str, box_size: float) -> Tuple[float, float]:
    half = box_size / 2.0
    lim_x = LIMITS[obj_type]["x"]
    lim_y = LIMITS[obj_type]["y"]
    cx = float(np.clip(cx, lim_x[0] + half, lim_x[1] - half))
    cy = float(np.clip(cy, lim_y[0] + half, lim_y[1] - half))
    return cx, cy


def format_box(cx: float, cy: float, obj_type: str, box_size: float) -> str:
    half = box_size / 2.0
    cx, cy = clamp_center(cx, cy, obj_type, box_size)
    return f"X:[{cx - half:.3f},{cx + half:.3f}] Y:[{cy - half:.3f},{cy + half:.3f}]"


def box_from_points(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    obj_type: str,
    box_size: float,
    fallback_center: Tuple[float, float] = None,
) -> str:
    if df.empty:
        if fallback_center is None:
            return "N/A"
        return format_box(fallback_center[0], fallback_center[1], obj_type, box_size)
    cx = float(df[x_col].mean())
    cy = float(df[y_col].mean())
    if not np.isfinite(cx) or not np.isfinite(cy):
        if fallback_center is None:
            return "N/A"
        return format_box(fallback_center[0], fallback_center[1], obj_type, box_size)
    return format_box(cx, cy, obj_type, box_size)


def subset_by_bin(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    x_low: float,
    x_high: float,
    y_low: float,
    y_high: float,
) -> pd.DataFrame:
    eps = 1e-12
    return df[
        (df[x_col] >= x_low - eps)
        & (df[x_col] <= x_high + eps)
        & (df[y_col] >= y_low - eps)
        & (df[y_col] <= y_high + eps)
    ]


def allocate_quota(total: int, priorities: List[float], min_per_bin: int) -> List[int]:
    n = len(priorities)
    if n == 0 or total <= 0:
        return []
    min_per_bin = max(0, min_per_bin)
    max_min = total // n
    if min_per_bin > max_min:
        min_per_bin = max_min
    alloc = np.full(n, min_per_bin, dtype=int)
    remaining = total - alloc.sum()
    if remaining <= 0:
        return alloc.tolist()
    weights = np.asarray(priorities, dtype=float)
    weights = weights - np.min(weights)
    if np.allclose(weights, 0):
        weights = np.ones_like(weights)
    weights = weights / np.sum(weights)
    extra = np.floor(remaining * weights).astype(int)
    alloc += extra
    leftover = total - alloc.sum()
    if leftover > 0:
        order = np.argsort(-weights)
        for i in range(leftover):
            alloc[order[i % n]] += 1
    return alloc.tolist()


def format_range(low: float, high: float, decimals: int = 3) -> str:
    return f"[{low:.{decimals}f},{high:.{decimals}f}]"


def append_top_bins_table(
    report: List[str],
    stats: pd.DataFrame,
    top_n: int,
    x_label: str,
    y_label: str,
    decimals: int = 3,
) -> pd.DataFrame:
    if stats.empty:
        report.append("该网格无可用数据。")
        return stats
    top = stats.sort_values("priority", ascending=False).head(top_n).reset_index(drop=True)
    report.append(f"| 序号 | {x_label} | {y_label} | 均值 | 样本数 | 优先级 |")
    report.append("| :--- | :--- | :--- | :--- | :--- | :--- |")
    for i, row in top.iterrows():
        x_rng = format_range(row["x_low"], row["x_high"], decimals)
        y_rng = format_range(row["y_low"], row["y_high"], decimals)
        report.append(
            f"| {i + 1} | {x_rng} | {y_rng} | {row['mean']:.2e} | {int(row['count'])} | {row['priority']:.2f} |"
        )
    return top


def append_empty_bins_table(
    report: List[str],
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    stats: pd.DataFrame,
    x_label: str,
    y_label: str,
    max_rows: int = 6,
    decimals: int = 3,
) -> None:
    empty_bins = list_empty_bins(x_edges, y_edges, stats)
    total_bins = (len(x_edges) - 1) * (len(y_edges) - 1)
    report.append(f"空网格数: {len(empty_bins)} / {total_bins}")
    if not empty_bins:
        return
    report.append(f"| {x_label} | {y_label} |")
    report.append("| :--- | :--- |")
    for i, j in empty_bins[:max_rows]:
        x_rng = format_range(float(x_edges[i]), float(x_edges[i + 1]), decimals)
        y_rng = format_range(float(y_edges[j]), float(y_edges[j + 1]), decimals)
        report.append(f"| {x_rng} | {y_rng} |")


def parse_weights(weights: str) -> Tuple[float, float, float]:
    parts = [float(x.strip()) for x in weights.split(",") if x.strip()]
    if len(parts) != 3:
        raise ValueError("weights 需要三个以逗号分隔的值: abs,rel,yaw")
    if any(p < 0 for p in parts):
        raise ValueError("weights 不能为负数")
    total = sum(parts)
    if total == 0:
        raise ValueError("weights 总和需大于 0")
    return parts[0] / total, parts[1] / total, parts[2] / total


def generate_report(
    df: pd.DataFrame,
    output_path: Path,
    total_episodes: int,
    alpha: float,
    abs_bins: int,
    rel_bins: int,
    yaw_bins: int,
    dist_bins: int,
    top_bins: int,
    box_size: float,
    min_per_bin: int,
    weights: Tuple[float, float, float],
    warnings: List[str],
) -> None:
    report: List[str] = []
    report.append("# 放置数据采集分析 v2")

    scored = df.dropna(subset=["score"]).copy()
    report.append("\n## 1. 数据概览")
    report.append(f"- 总样本数: {len(df)}")
    report.append(f"- 有分数样本数: {len(scored)}")
    if len(scored) > 0:
        report.append(f"- 平均分: {scored['score'].mean():.4e}")
        report.append(f"- P90 分数: {scored['score'].quantile(0.90):.4e}")
    else:
        report.append("- 平均分: N/A")
        report.append("- P90 分数: N/A")
    if warnings:
        report.append("\n告警:")
        for w in warnings:
            report.append(f"- {w}")

    if scored.empty:
        report.append("\n没有可用的评分样本，跳过优先级分析。")
        with output_path.open("w", encoding="utf-8") as f:
            f.write("\n".join(report))
        print(f"报告已生成: {output_path}")
        return

    report.append("\n## 2. 相关性（绝对值）")
    feat_cols = [
        "nut_x",
        "nut_y",
        "nut_yaw",
        "peg_x",
        "peg_y",
        "rel_dx",
        "rel_dy",
        "rel_dist",
    ]
    valid_cols = [c for c in feat_cols if c in scored.columns]
    corrs = scored[valid_cols + ["score"]].corr()["score"].abs().sort_values(ascending=False).drop("score")
    report.append("| 特征 | 相关系数 |")
    report.append("| :--- | :--- |")
    for idx, val in corrs.items():
        report.append(f"| {idx} | {val:.4f} |")

    report.append("\n## 3. 优先级地图")

    nut_x_edges = np.linspace(LIMITS["nut"]["x"][0], LIMITS["nut"]["x"][1], abs_bins + 1)
    nut_y_edges = np.linspace(LIMITS["nut"]["y"][0], LIMITS["nut"]["y"][1], abs_bins + 1)
    peg_x_edges = np.linspace(LIMITS["peg"]["x"][0], LIMITS["peg"]["x"][1], abs_bins + 1)
    peg_y_edges = np.linspace(LIMITS["peg"]["y"][0], LIMITS["peg"]["y"][1], abs_bins + 1)

    report.append("\n### 3.1 螺母绝对位置（固定网格）")
    nut_stats = build_grid_stats(scored, "nut_x", "nut_y", nut_x_edges, nut_y_edges, alpha)
    nut_top = append_top_bins_table(report, nut_stats, top_bins, "螺母X", "螺母Y")
    append_empty_bins_table(report, nut_x_edges, nut_y_edges, nut_stats, "螺母X", "螺母Y")

    report.append("\n### 3.2 Peg 绝对位置（固定网格）")
    peg_stats = build_grid_stats(scored, "peg_x", "peg_y", peg_x_edges, peg_y_edges, alpha)
    peg_top = append_top_bins_table(report, peg_stats, top_bins, "Peg X", "Peg Y")
    append_empty_bins_table(report, peg_x_edges, peg_y_edges, peg_stats, "Peg X", "Peg Y")

    report.append("\n### 3.3 相对位移 dx/dy（nut - peg）")
    rel_x_edges = build_edges_from_data(scored["rel_dx"], rel_bins)
    rel_y_edges = build_edges_from_data(scored["rel_dy"], rel_bins)
    rel_stats = build_grid_stats(scored, "rel_dx", "rel_dy", rel_x_edges, rel_y_edges, alpha)
    rel_top = append_top_bins_table(report, rel_stats, top_bins, "dX", "dY")
    append_empty_bins_table(report, rel_x_edges, rel_y_edges, rel_stats, "dX", "dY")

    report.append("\n## 4. Yaw + 距离优先级")
    yaw_edges = np.linspace(-np.pi, np.pi, yaw_bins + 1)
    dist_max = float(scored["rel_dist"].max())
    if not np.isfinite(dist_max) or dist_max <= 0:
        dist_max = 1e-3
    dist_edges = np.linspace(0.0, dist_max, dist_bins + 1)
    yaw_stats = build_grid_stats(scored, "nut_yaw", "rel_dist", yaw_edges, dist_edges, alpha)
    yaw_top = yaw_stats.sort_values("priority", ascending=False).head(top_bins).reset_index(drop=True)
    report.append("| 序号 | Yaw(度) | 距离(m) | 均值 | 样本数 | 优先级 |")
    report.append("| :--- | :--- | :--- | :--- | :--- | :--- |")
    for i, row in yaw_top.iterrows():
        yaw_low = np.degrees(row["x_low"])
        yaw_high = np.degrees(row["x_high"])
        dist_rng = format_range(row["y_low"], row["y_high"], 3)
        yaw_rng = f"[{yaw_low:.1f},{yaw_high:.1f}]"
        report.append(
            f"| {i + 1} | {yaw_rng} | {dist_rng} | {row['mean']:.2e} | {int(row['count'])} | {row['priority']:.2f} |"
        )
    append_empty_bins_table(report, yaw_edges, dist_edges, yaw_stats, "Yaw(弧度)", "距离(m)")

    report.append("\n## 5. 采集蓝图（多盒子目标）")
    abs_weight, rel_weight, yaw_weight = weights
    abs_total = int(total_episodes * abs_weight)
    rel_total = int(total_episodes * rel_weight)
    yaw_total = total_episodes - abs_total - rel_total

    report.append(f"- 采集总条数: {total_episodes}")
    report.append(f"- 权重: 绝对={abs_weight:.2f}, 相对={rel_weight:.2f}, 角度={yaw_weight:.2f}")
    report.append(f"- 盒子尺寸: {box_size:.2f} m（已按边界裁剪）")

    report.append("\n### 5.1 任务A：螺母绝对位置网格")
    abs_priorities = nut_top["priority"].tolist() if not nut_top.empty else []
    abs_quota = allocate_quota(abs_total, abs_priorities, min_per_bin)
    report.append("| 序号 | 螺母网格 | 均值 | 样本数 | 优先级 | 配额 | 螺母盒子 | Peg 盒子 |")
    report.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
    for i, row in nut_top.iterrows():
        subset = subset_by_bin(scored, "nut_x", "nut_y", row["x_low"], row["x_high"], row["y_low"], row["y_high"])
        nut_center = (row["x_center"], row["y_center"])
        nut_box = box_from_points(subset, "nut_x", "nut_y", "nut", box_size, fallback_center=nut_center)
        peg_box = box_from_points(subset, "peg_x", "peg_y", "peg", box_size)
        x_rng = format_range(row["x_low"], row["x_high"], 3)
        y_rng = format_range(row["y_low"], row["y_high"], 3)
        report.append(
            f"| {i + 1} | {x_rng} {y_rng} | {row['mean']:.2e} | {int(row['count'])} | {row['priority']:.2f} | {abs_quota[i] if i < len(abs_quota) else 0} | {nut_box} | {peg_box} |"
        )

    report.append("\n### 5.2 任务B：相对 dx/dy 网格")
    rel_priorities = rel_top["priority"].tolist() if not rel_top.empty else []
    rel_quota = allocate_quota(rel_total, rel_priorities, min_per_bin)
    report.append("| 序号 | dX | dY | 均值 | 样本数 | 优先级 | 配额 | 螺母盒子 | Peg 盒子 |")
    report.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
    for i, row in rel_top.iterrows():
        subset = subset_by_bin(scored, "rel_dx", "rel_dy", row["x_low"], row["x_high"], row["y_low"], row["y_high"])
        nut_box = box_from_points(subset, "nut_x", "nut_y", "nut", box_size)
        peg_box = box_from_points(subset, "peg_x", "peg_y", "peg", box_size)
        dx_rng = format_range(row["x_low"], row["x_high"], 3)
        dy_rng = format_range(row["y_low"], row["y_high"], 3)
        report.append(
            f"| {i + 1} | {dx_rng} | {dy_rng} | {row['mean']:.2e} | {int(row['count'])} | {row['priority']:.2f} | {rel_quota[i] if i < len(rel_quota) else 0} | {nut_box} | {peg_box} |"
        )

    report.append("\n### 5.3 任务C：Yaw + 距离网格")
    yaw_priorities = yaw_top["priority"].tolist() if not yaw_top.empty else []
    yaw_quota = allocate_quota(yaw_total, yaw_priorities, min_per_bin)
    report.append("| 序号 | Yaw(度) | 距离(m) | 均值 | 样本数 | 优先级 | 配额 | 螺母盒子 | Peg 盒子 |")
    report.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
    for i, row in yaw_top.iterrows():
        subset = subset_by_bin(scored, "nut_yaw", "rel_dist", row["x_low"], row["x_high"], row["y_low"], row["y_high"])
        nut_box = box_from_points(subset, "nut_x", "nut_y", "nut", box_size)
        peg_box = box_from_points(subset, "peg_x", "peg_y", "peg", box_size)
        yaw_low = np.degrees(row["x_low"])
        yaw_high = np.degrees(row["x_high"])
        dist_rng = format_range(row["y_low"], row["y_high"], 3)
        yaw_rng = f"[{yaw_low:.1f},{yaw_high:.1f}]"
        report.append(
            f"| {i + 1} | {yaw_rng} | {dist_rng} | {row['mean']:.2e} | {int(row['count'])} | {row['priority']:.2f} | {yaw_quota[i] if i < len(yaw_quota) else 0} | {nut_box} | {peg_box} |"
        )

    report.append("\n## 6. 最差样本（按分数 Top 5）")
    top5 = scored.nlargest(5, "score")
    report.append("| 样本 | 螺母 (X,Y) | Peg (X,Y) | 分数 |")
    report.append("| :--- | :--- | :--- | :--- |")
    for _, row in top5.iterrows():
        report.append(
            f"| {row['episode']} | ({row['nut_x']:.3f},{row['nut_y']:.3f}) | ({row['peg_x']:.3f},{row['peg_y']:.3f}) | {row['score']:.4e} |"
        )

    with output_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(report))
    print(f"报告已生成: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--score-pkl", type=Path)
    parser.add_argument("--hdf5", type=Path)
    parser.add_argument("--total-episodes", type=int, default=200)
    parser.add_argument("--output", type=Path, default=Path("placement_analysis_v2.csv"))
    parser.add_argument("--report", type=Path, default=Path("collection_report_v2.md"))
    parser.add_argument("--abs-bins", type=int, default=10)
    parser.add_argument("--rel-bins", type=int, default=10)
    parser.add_argument("--yaw-bins", type=int, default=8)
    parser.add_argument("--dist-bins", type=int, default=6)
    parser.add_argument("--top-bins", type=int, default=6)
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--box-size", type=float, default=0.10)
    parser.add_argument("--min-per-bin", type=int, default=2)
    parser.add_argument("--weights", type=str, default="0.40,0.35,0.25")
    args = parser.parse_args()

    scores = None
    warnings: List[str] = []
    if args.score_pkl and args.score_pkl.exists():
        scores = load_scores(args.score_pkl)

    results = []
    demos_count = 0
    if args.hdf5 and args.hdf5.exists():
        with h5py.File(args.hdf5, "r") as f:
            demos = sorted(f["data"].keys())
            demos_count = len(demos)
            for i, d in enumerate(demos):
                obs = f["data"][d]["obs"]
                obj = obs["object"][0]
                nut_pos = obj[:3]
                nut_quat = obj[3:7] if len(obj) >= 7 else [0, 0, 0, 1]
                peg_pos = obj[-3:] if len(obj) >= 17 else obs.get("peg1_pos", [[0, 0, 0]])[0]
                score = scores[i] if scores is not None and i < len(scores) else np.nan
                results.append(
                    {
                        "episode": d,
                        "nut_x": nut_pos[0],
                        "nut_y": nut_pos[1],
                        "nut_z": nut_pos[2],
                        "nut_yaw": quat_to_yaw(nut_quat),
                        "peg_x": peg_pos[0],
                        "peg_y": peg_pos[1],
                        "peg_z": peg_pos[2],
                        "score": score,
                    }
                )

    if scores is not None and demos_count and len(scores) != demos_count:
        warnings.append(
            f"score 数量不匹配: scores={len(scores)} demos={demos_count}。"
            "按索引对齐，额外 demo 的 score 为 NaN。"
        )

    df = pd.DataFrame(results)
    if df.empty:
        print("未加载到数据，退出。")
        return

    df["rel_dx"] = df["nut_x"] - df["peg_x"]
    df["rel_dy"] = df["nut_y"] - df["peg_y"]
    df["rel_dist"] = np.sqrt(df["rel_dx"] ** 2 + df["rel_dy"] ** 2)
    df.to_csv(args.output, index=False)

    weights = parse_weights(args.weights)
    generate_report(
        df,
        args.report,
        args.total_episodes,
        args.alpha,
        args.abs_bins,
        args.rel_bins,
        args.yaw_bins,
        args.dist_bins,
        args.top_bins,
        args.box_size,
        args.min_per_bin,
        weights,
        warnings,
    )


if __name__ == "__main__":
    main()
