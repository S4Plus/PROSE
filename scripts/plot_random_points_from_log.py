#!/usr/bin/env python3
"""
Parse a training log, randomly pick a few points, and map their indices onto a plot.
"""
import argparse
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt


@dataclass
class Point:
    index: int
    step: int
    loss: float


STEP_LOSS_RE = re.compile(r"step:(?P<step>[\d\.]+K?)\b.*?loss:(?P<loss>[\d\.]+)")


def parse_step(raw: str) -> int:
    raw = raw.strip()
    if raw.endswith("K"):
        return int(float(raw[:-1]) * 1000)
    return int(float(raw))


def parse_log(path: Path) -> List[Point]:
    points: List[Point] = []
    with path.open() as f:
        for idx, line in enumerate(f):
            match = STEP_LOSS_RE.search(line)
            if not match:
                continue
            step = parse_step(match.group("step"))
            loss = float(match.group("loss"))
            points.append(Point(index=idx, step=step, loss=loss))
    return points


def plot_points(points: List[Point], selected: List[int], output: Path) -> None:
    steps = [p.step for p in points]
    losses = [p.loss for p in points]

    plt.figure(figsize=(11, 5))
    plt.plot(steps, losses, label="loss", linewidth=1.0, color="#3366cc")

    sel_steps = [points[i].step for i in selected]
    sel_losses = [points[i].loss for i in selected]
    plt.scatter(sel_steps, sel_losses, color="#d62728", s=35, label="random picks")

    for i, x, y in zip(selected, sel_steps, sel_losses):
        plt.annotate(str(i), (x, y), textcoords="offset points", xytext=(4, 4), fontsize=8)

    plt.xlabel("step")
    plt.ylabel("loss")
    plt.title("Random points mapped by index")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output, dpi=160)
    plt.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log",
        default="logs/train2_400_act_random_0.5.log",
        type=Path,
        help="Path to training log to parse.",
    )
    parser.add_argument("--num", type=int, default=20, help="How many points to sample.")
    parser.add_argument("--seed", type=int, default=0, help="Seed for reproducible sampling.")
    parser.add_argument(
        "--output",
        type=Path,
        help="Where to save the plot. Defaults to <log>.png next to the log file.",
    )
    args = parser.parse_args()

    log_path = args.log
    if not log_path.exists():
        print(f"Log file not found: {log_path}", file=sys.stderr)
        return 1

    points = parse_log(log_path)
    if not points:
        print("No step/loss entries found in the log.", file=sys.stderr)
        return 1

    rng = random.Random(args.seed)
    num = min(args.num, len(points))
    selected = rng.sample(range(len(points)), num)
    selected.sort()

    output_path = args.output or log_path.with_suffix(".png")
    plot_points(points, selected, output_path)

    print(f"Saved plot to: {output_path}")
    print("Highlighted indices (0-based in parsed points):")
    for i in selected:
        p = points[i]
        print(f"  {i}: line={p.index}, step={p.step}, loss={p.loss}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
