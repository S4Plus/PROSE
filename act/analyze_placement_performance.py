#!/usr/bin/env python3
import h5py
import argparse
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
import sys
from typing import Dict, List, Tuple

# Official Environment Limits
LIMITS = {
    "nut": {"x": (-0.115, 0.115), "y": (-0.255, 0.255)},
    "peg": {"x": (-0.1, 0.3), "y": (-0.2, 0.2)}
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

def quat_to_euler(q):
    """Z-axis (yaw) extraction from quaternion."""
    if len(q) == 4:
        x, y, z, w = q
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        return np.arctan2(siny_cosp, cosy_cosp)
    return 0

def find_clamped_10cm_box(df_hi: pd.DataFrame, x_col: str, y_col: str, obj_type: str) -> str:
    """Find a 10cm box centered on high-influence points, clamped to env limits."""
    if df_hi.empty: return "N/A"
    center_x = df_hi[x_col].mean()
    center_y = df_hi[y_col].mean()
    lim_x = LIMITS[obj_type]["x"]
    lim_y = LIMITS[obj_type]["y"]
    cx = np.clip(center_x, lim_x[0] + 0.05, lim_x[1] - 0.05)
    cy = np.clip(center_y, lim_y[0] + 0.05, lim_y[1] - 0.05)
    return f"X: [{cx-0.05:.3f}, {cx+0.05:.3f}], Y: [{cy-0.05:.3f}, {cy+0.05:.3f}]"

def generate_report(df: pd.DataFrame, output_path: Path, total_episodes: int):
    """Generate the ULTIMATE technical analysis report."""
    report = []
    report.append(f"# 💎 全维度空间性能综合情报 (Ultimate Placement Intelligence)")
    
    # 1. Dataset Overview
    report.append("\n## 1. 全局显著性指标")
    avg_score = df["score"].mean()
    p90_score = df["score"].quantile(0.90)
    report.append(f"- **分析样本量**: {len(df)}")
    report.append(f"- **平均复杂性指数**: {avg_score:.4e}")
    report.append(f"- **P90 难点阈值**: {p90_score:.4e}")
    report.append(f"- **Nut 物理边界**: X={LIMITS['nut']['x']}, Y={LIMITS['nut']['y']}")

    # 2. Correlation Analysis (The WHY)
    report.append("\n## 2. 空间决策敏感度分析 (Correlation)")
    feat_cols = ["nut_x", "nut_y", "nut_yaw", "peg_x", "peg_y", "rel_dist"]
    valid_cols = [c for c in feat_cols if c in df.columns]
    corrs = df[valid_cols + ["score"]].corr()["score"].abs().sort_values(ascending=False).drop("score")
    
    report.append("\n| 特征维度 | 相关系数 | 敏感度等级 | 采集决策依据 |")
    report.append("| :--- | :--- | :--- | :--- |")
    for idx, val in corrs.items():
        lvl = "🔴 极高" if val > 0.25 else "🟡 中等" if val > 0.1 else "⚪ 背景"
        desc = "侧向精度瓶颈" if "y" in idx else "直线覆盖不足" if "dist" in idx or "x" in idx else "旋转偏差敏感"
        report.append(f"| **{idx}** | {val:.4f} | {lvl} | {desc} |")

    # 3. 5x5 Heatmap (The WHERE)
    report.append("\n## 3. 5x5 空间难度与密度网格 (Landscapes)")
    report.append("表格格式: `平均分/样本数`。 🔥=高分点，📉=数据稀缺(<2%)。")
    x_lbls = ["极远", "远", "中", "近", "极近"]
    y_lbls = ["极左", "左", "中", "右", "极右"]
    df["nx_5"] = pd.qcut(df["nut_x"], 5, labels=x_lbls)
    df["ny_5"] = pd.qcut(df["nut_y"], 5, labels=y_lbls)
    g_m = df.pivot_table(index="nx_5", columns="ny_5", values="score", aggfunc="mean", observed=False)
    g_c = df.pivot_table(index="nx_5", columns="ny_5", values="score", aggfunc="count", observed=False)
    
    report.append("\n### [Nut] 绝对位置热力图")
    report.append("| X \\ Y | 极左 | 左 | 中 | 右 | 极右 |")
    report.append("| :--- | :--- | :--- | :--- | :--- | :--- |")
    for r in g_m.index:
        line = [f"**{r}**"]
        for c in g_m.columns:
            m, ct = g_m.loc[r,c], g_c.loc[r,c]
            tag = "🔥" if m > p90_score else ""
            tag += "📉" if ct < len(df)*0.02 else ""
            line.append(f"{m:.1e}/{ct}{tag}")
        report.append("| " + " | ".join(line) + " |")

    # 4. Rotational Sensitivity (The HOW - Orientation)
    report.append("\n## 4. 旋转角 (Yaw) 决策高风险区 (High-Risk Orientation)")
    report.append("分析螺母初始旋转角度对难度的影响。🚨=高风险区间（得分前15%）。")
    
    # Calculate bins and intervals
    df["yaw_bin"], bins = pd.cut(df["nut_yaw"], bins=8, labels=["-180°", "-135°", "-90°", "-45°", "0°", "45°", "90°", "135°"], retbins=True)
    yaw_m = df.groupby("yaw_bin", observed=False)["score"].mean()
    
    report.append("\n| 初始旋转角 | 弧度区间 (rad) | 角度区间 (deg) | 平均得分 | 状态 |")
    report.append("| :--- | :--- | :--- | :--- | :--- |")
    for i, (label, val) in enumerate(yaw_m.items()):
        low, high = bins[i], bins[i+1]
        is_risky = val > df["score"].quantile(0.85)
        tag = "🚨 **高风险**" if is_risky else "✅ 稳健"
        report.append(f"| **{label}** | `[{low:.2f}, {high:.2f}]` | `[{np.degrees(low):.1f}°, {np.degrees(high):.1f}°]` | {val:.1e} | {tag} |")
    
    # 5. High-Risk Summary (NEW)
    report.append("\n## 5. ⚠️ 核心高风险报警 (Critical Warnings)")
    max_yaw_bin = yaw_m.idxmax()
    report.append(f"- **[角度报警]**: 旋转角度在 `{max_yaw_bin}` 区间时表现最差，必须在此姿势下补 demo。")
    report.append(f"- **[位置报警]**: 螺母位于 `极远-极右` 格点时影响力最高，属 P0 级补强区。")

    # 6. Collection Manifesto (The ACTIONS)
    report.append(f"\n## 6. {total_episodes}条精密采集蓝图 (Collection Blueprint)")
    df_hi = df[df["score"] >= p90_score].copy()
    
    weights = {"TaskA": 0.40, "TaskB": 0.35, "TaskC": 0.25}
    counts = {k: int(total_episodes * v) for k, v in weights.items()}
    counts["TaskC"] += total_episodes - sum(counts.values())

    # Task A: Extreme Y
    df_z1 = df_hi[df_hi["nut_y"].abs() > 0.15].copy()
    report.append(f"\n### 🔴 任务 A：边缘对齐补强 ({counts['TaskA']}条)")
    report.append(f"- **Nut 摆放框 (10cm)**: {find_clamped_10cm_box(df_z1, 'nut_x', 'nut_y', 'nut')}")
    report.append(f"- **Peg 摆放框 (10cm)**: {find_clamped_10cm_box(df_z1, 'peg_x', 'peg_y', 'peg')}")

    # Task B: Long Distance
    df_z2 = df_hi[df_hi["rel_dist"] > df_hi["rel_dist"].median()].copy()
    report.append(f"\n### 🔵 任务 B：长距离交互 ({counts['TaskB']}条)")
    report.append(f"- **Nut 摆放框 (10cm)**: {find_clamped_10cm_box(df_z2, 'nut_x', 'nut_y', 'nut')}")
    report.append(f"- **Peg 摆放框 (10cm)**: {find_clamped_10cm_box(df_z2, 'peg_x', 'peg_y', 'peg')}")

    # Task C: Symmetry/Scarcity
    df_z3 = df_hi[df_hi["nut_y"] < -0.1].copy()
    report.append(f"\n### 🟢 任务 C：空间对称加固 ({counts['TaskC']}条)")
    report.append(f"- **Nut 摆放框 (10cm)**: {find_clamped_10cm_box(df_z3, 'nut_x', 'nut_y', 'nut')}")
    report.append(f"- **Peg 摆放框 (10cm)**: {find_clamped_10cm_box(df_z3, 'peg_x', 'peg_y', 'peg')}")

    # 5. Top 5 Blacklist
    report.append("\n## 5. 前 5 名“深渊级”案例参考 (Worst Cases)")
    top5 = df.nlargest(5, "score")
    report.append("\n| Episode | Nut (X, Y) | Peg (X, Y) | Score |")
    report.append("| :--- | :--- | :--- | :--- |")
    for _, row in top5.iterrows():
        report.append(f"| {row['episode']} | ({row['nut_x']:.3f}, {row['nut_y']:.3f}) | ({row['peg_x']:.3f}, {row['peg_y']:.3f}) | {row['score']:.4e} |")

    report_content = "\n".join(report)
    with output_path.open("w", encoding="utf-8") as f:
        f.write(report_content)
    print(f"\n[ULTIMATE REPORT] 综合版报告已生成: {output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--score-pkl", type=Path)
    parser.add_argument("--hdf5", type=Path)
    parser.add_argument("--total-episodes", type=int, default=200)
    parser.add_argument("--output", type=Path, default=Path("placement_analysis.csv"))
    parser.add_argument("--report", type=Path, default=Path("collection_report.md"))
    args = parser.parse_args()

    scores = None
    if args.score_pkl and args.score_pkl.exists():
        scores = load_scores(args.score_pkl)

    results = []
    if args.hdf5 and args.hdf5.exists():
        with h5py.File(args.hdf5, "r") as f:
            demos = sorted(f["data"].keys())
            for i, d in enumerate(demos):
                obs = f["data"][d]["obs"]
                obj = obs["object"][0]
                nut_pos, nut_quat = obj[:3], obj[3:7] if len(obj)>=7 else [0,0,0,1]
                peg_pos = obj[-3:] if len(obj)>=17 else obs.get("peg1_pos", [[0,0,0]])[0]
                results.append({
                    "episode": d, "nut_x": nut_pos[0], "nut_y": nut_pos[1], "nut_z": nut_pos[2],
                    "nut_yaw": quat_to_euler(nut_quat), "peg_x": peg_pos[0], "peg_y": peg_pos[1], "peg_z": peg_pos[2],
                    "score": scores[i] if scores is not None else np.nan
                })

    df = pd.DataFrame(results)
    if df.empty: return
    df["rel_dist"] = np.sqrt((df["nut_x"]-df["peg_x"])**2 + (df["nut_y"]-df["peg_y"])**2)
    df.to_csv(args.output, index=False)
    generate_report(df, args.report, args.total_episodes)

if __name__ == "__main__":
    main()
