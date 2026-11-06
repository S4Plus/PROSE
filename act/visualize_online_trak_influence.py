from __future__ import annotations

import argparse
import pickle
import pathlib
from typing import Dict, Iterable
import sys

try:  # pragma: no cover - optional dependency
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover - dependency check
    raise SystemExit("Matplotlib is required. Install it via `pip install matplotlib`.") from exc

import numpy as np


def _ensure_numpy_core_aliases() -> None:
    """
    Legacy pickles may refer to numpy._core.* modules that are no longer exposed.
    Register aliases to numpy.core to keep loading working on newer numpy.
    """
    core = getattr(np, "core", None)
    if core is None:
        return
    sys.modules.setdefault("numpy._core", core)
    if hasattr(core, "numeric"):
        sys.modules.setdefault("numpy._core.numeric", core.numeric)
    if hasattr(core, "multiarray"):
        sys.modules.setdefault("numpy._core.multiarray", core.multiarray)


def _load_data(path: pathlib.Path) -> Dict[str, Dict[str, np.ndarray]]:
    _ensure_numpy_core_aliases()
    with path.open("rb") as f:
        raw = pickle.load(f)

    # CUPID-ACT influence files store influence scores directly instead of
    # split->metric dictionaries. Normalize them here for backward compatibility.
    if isinstance(raw, dict) and "influence_scores" in raw:
        influence_scores = np.asarray(raw["influence_scores"])
        result = {"train": {"influence_scores": influence_scores}}
        return result

    result: Dict[str, Dict[str, np.ndarray]] = {}
    for split, metrics in raw.items():
        result[split] = {}
        for metric_name, values in metrics.items():
            result[split][metric_name] = np.asarray(values)
    return result


def _list_metrics(data: Dict[str, Dict[str, np.ndarray]]) -> None:
    for split, metrics in data.items():
        print(f"[{split}]")
        for metric in metrics:
            arr = metrics[metric]
            print(f"  {metric} (shape={arr.shape})")


def _plot_metric(
    values: np.ndarray,
    split: str,
    metric_name: str,
    topk: int,
    output_path: pathlib.Path,
) -> None:
    idx = np.arange(values.shape[0])
    order = np.argsort(values)[::-1]
    top_idx = order[:topk]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)

    axes[0].plot(idx, values, marker="o", linewidth=1)
    axes[0].set_title(f"{metric_name} ({split})")
    axes[0].set_xlabel("Episode index")
    axes[0].set_ylabel("Influence score")

    axes[1].scatter(idx, values, s=10)
    mean_val = float(values.mean())
    axes[1].axhline(mean_val, color="red", linestyle="--", linewidth=1, label=f"mean={mean_val:.4f}")
    bottom_k = min(6, values.shape[0])
    worst_idx = np.argsort(values)[:bottom_k]
    worst_text = "\n".join(f"{i}: {values[i]:.4g}" for i in worst_idx)
    axes[1].text(
        0.98,
        0.02,
        f"Worst {bottom_k}:\n{worst_text}",
        transform=axes[1].transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
    )
    axes[1].set_title("Scores by index")
    axes[1].set_xlabel("Episode index")
    axes[1].set_ylabel("Influence score")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, linestyle="--", alpha=0.3)

    axes[2].barh(
        np.arange(len(top_idx)),
        values[top_idx][::-1],
        align="center",
    )
    axes[2].set_yticks(np.arange(len(top_idx)))
    axes[2].set_yticklabels(top_idx[::-1])
    axes[2].invert_yaxis()
    axes[2].set_xlabel("Influence score")
    axes[2].set_title(f"Top {len(top_idx)} episodes")

    fig.suptitle(f"Online TRAK Influence — {metric_name} ({split})")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved figure to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize online TRAK influence results.")
    parser.add_argument(
        "--pkl-path",
        type=pathlib.Path,
        required=True,
        help="Path to online_trak_influence.pkl",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Split to visualize (e.g., train, holdout).",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default=None,
        help="Metric key to visualize. Use --list-metrics to discover options.",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=20,
        help="Number of highest-scoring episodes to highlight.",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=None,
        help="Where to save the generated figure.",
    )
    parser.add_argument(
        "--list-metrics",
        action="store_true",
        help="Only list available metrics and exit.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = _load_data(args.pkl_path)

    if args.list_metrics:
        _list_metrics(data)
        return

    if args.split not in data:
        raise KeyError(f"Split '{args.split}' not found. Available: {list(data.keys())}")

    split_metrics = data[args.split]
    metric_name = args.metric or next(iter(split_metrics.keys()))
    if metric_name not in split_metrics:
        raise KeyError(f"Metric '{metric_name}' not in split '{args.split}'. Available: {list(split_metrics.keys())}")

    values = split_metrics[metric_name]
    output = args.output
    if output is None:
        sanitized = metric_name.replace("/", "_")
        output = args.pkl_path.with_name(f"{args.split}_{sanitized}.png")

    _plot_metric(values, args.split, metric_name, args.topk, output)


if __name__ == "__main__":
    main()
