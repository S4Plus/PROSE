#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import List

import numpy as np

try:
    import pyarrow.parquet as pq
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"pyarrow is required: {exc}")


def load_joint_names(dataset_root: Path) -> List[str]:
    info = json.loads((dataset_root / "meta" / "info.json").read_text())
    return info["features"]["action"]["names"][0]


def compute_metrics(dataset_root: Path) -> dict:
    parquet_paths = sorted((dataset_root / "data").rglob("episode_*.parquet"))
    if not parquet_paths:
        raise FileNotFoundError("No parquet files found.")

    sum_action = np.zeros(14)
    sum_action_abs = np.zeros(14)
    sum_action2 = np.zeros(14)
    min_action = np.full(14, np.inf)
    max_action = np.full(14, -np.inf)

    sum_vel_abs = np.zeros(14)
    min_vel = np.full(14, np.inf)
    max_vel = np.full(14, -np.inf)

    mae_sum = np.zeros(14)
    count = 0
    corr_sum_x = np.zeros(14)
    corr_sum_y = np.zeros(14)
    corr_sum_x2 = np.zeros(14)
    corr_sum_y2 = np.zeros(14)
    corr_sum_xy = np.zeros(14)

    for p in parquet_paths:
        t = pq.read_table(
            p, columns=["action", "observation.state", "observation.velocity"]
        )
        action = np.asarray(t["action"].to_pylist(), dtype=np.float64)
        state = np.asarray(t["observation.state"].to_pylist(), dtype=np.float64)
        vel = np.asarray(t["observation.velocity"].to_pylist(), dtype=np.float64)

        sum_action += action.sum(axis=0)
        sum_action_abs += np.abs(action).sum(axis=0)
        sum_action2 += (action ** 2).sum(axis=0)
        min_action = np.minimum(min_action, action.min(axis=0))
        max_action = np.maximum(max_action, action.max(axis=0))

        sum_vel_abs += np.abs(vel).sum(axis=0)
        min_vel = np.minimum(min_vel, vel.min(axis=0))
        max_vel = np.maximum(max_vel, vel.max(axis=0))

        mae_sum += np.abs(action - state).sum(axis=0)
        count += action.shape[0]

        corr_sum_x += action.sum(axis=0)
        corr_sum_y += state.sum(axis=0)
        corr_sum_x2 += (action ** 2).sum(axis=0)
        corr_sum_y2 += (state ** 2).sum(axis=0)
        corr_sum_xy += (action * state).sum(axis=0)

    count_total = max(count, 1)
    mean_action = sum_action / count_total
    mean_abs_action = sum_action_abs / count_total
    std_action = np.sqrt(np.maximum(sum_action2 / count_total - mean_action ** 2, 0.0))
    range_action = max_action - min_action
    mean_abs_vel = sum_vel_abs / count_total
    range_vel = max_vel - min_vel
    mae = mae_sum / count_total

    num = corr_sum_xy - (corr_sum_x * corr_sum_y / count_total)
    den = (corr_sum_x2 - (corr_sum_x ** 2) / count_total) * (
        corr_sum_y2 - (corr_sum_y ** 2) / count_total
    )
    den = np.sqrt(np.maximum(den, 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = np.where(den == 0, np.nan, num / den)

    return {
        "mean_abs_action": mean_abs_action,
        "std_action": std_action,
        "range_action": range_action,
        "mean_abs_vel": mean_abs_vel,
        "range_vel": range_vel,
        "mae": mae,
        "corr": corr,
    }


def save_csv(out_path: Path, joint_names: List[str], metrics: dict) -> None:
    headers = [
        "joint",
        "mean_abs_action",
        "std_action",
        "range_action",
        "mean_abs_vel",
        "range_vel",
        "action_state_mae",
        "action_state_corr",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for i, name in enumerate(joint_names):
            writer.writerow(
                [
                    name,
                    f"{metrics['mean_abs_action'][i]:.6f}",
                    f"{metrics['std_action'][i]:.6f}",
                    f"{metrics['range_action'][i]:.6f}",
                    f"{metrics['mean_abs_vel'][i]:.6f}",
                    f"{metrics['range_vel'][i]:.6f}",
                    f"{metrics['mae'][i]:.6f}",
                    f"{metrics['corr'][i]:.6f}",
                ]
            )


def plot_pair_bars(out_path: Path, joint_base: List[str], right_vals, left_vals, title, ylabel):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    x = np.arange(len(joint_base))
    width = 0.38
    plt.figure(figsize=(10, 4))
    plt.bar(x - width / 2, right_vals, width, label="right")
    plt.bar(x + width / 2, left_vals, width, label="left")
    plt.xticks(x, joint_base, rotation=45, ha="right")
    plt.title(title)
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare left/right action and observation statistics"
    )
    parser.add_argument(
        "--dataset",
        default="/workspace/liushuo/cupid/act/swk/six_object_pick_conveyor_first2",
        help="Dataset root directory",
    )
    parser.add_argument(
        "--out",
        default="/workspace/liushuo/cupid/act/swk/outputs/dataset_quality/six_object_pick_conveyor_first2",
        help="Output directory",
    )
    args = parser.parse_args()

    dataset_root = Path(args.dataset)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    joint_names = load_joint_names(dataset_root)
    metrics = compute_metrics(dataset_root)
    save_csv(out_dir / "left_right_metrics.csv", joint_names, metrics)

    right = slice(0, 7)
    left = slice(7, 14)
    joint_base = [name.replace("right_", "") for name in joint_names[:7]]

    plot_pair_bars(
        out_dir / "left_right_action_state_mae.png",
        joint_base,
        metrics["mae"][right],
        metrics["mae"][left],
        "Action vs state MAE (right vs left)",
        "MAE",
    )
    plot_pair_bars(
        out_dir / "left_right_action_state_corr.png",
        joint_base,
        metrics["corr"][right],
        metrics["corr"][left],
        "Action vs state correlation (right vs left)",
        "corr",
    )
    plot_pair_bars(
        out_dir / "left_right_action_range.png",
        joint_base,
        metrics["range_action"][right],
        metrics["range_action"][left],
        "Action range (right vs left)",
        "range",
    )
    plot_pair_bars(
        out_dir / "left_right_velocity_abs.png",
        joint_base,
        metrics["mean_abs_vel"][right],
        metrics["mean_abs_vel"][left],
        "Mean |velocity| (right vs left)",
        "mean |vel|",
    )

    print(f"Saved left/right comparison to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
