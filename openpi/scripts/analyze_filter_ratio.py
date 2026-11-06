#!/usr/bin/env python3
"""Analyze score distributions and suggest adaptive filtering ratios.

Supports two input modes:
1. Existing score artifacts:
   - retrain_joint.py -> joint_selection_scores.npz
   - retrain_window_joint.py -> joint_window_selection_scores.npz
2. Direct joint analysis from raw sources:
   - influence score pickle + quality CSV / pickle

Heuristics:
- knee: elbow/knee on sorted score curve
- gmm2: 1D two-Gaussian mixture threshold
- changepoint: prefix/suffix mean-shift on sorted scores

Examples:
  uv run python scripts/analyze_filter_ratio.py \
    --score-npz /workspace/liushuo/cupid/openpi/tmp/can_mh_grass_all/joint_selection_scores.npz \
    --plot

  uv run python scripts/analyze_filter_ratio.py \
    --score-pkl /workspace/liushuo/cupid/openpi/tmp/can_mh_grass_all/influence_grass.pkl \
    --quality-score-csv /workspace/liushuo/cupid/3_model/quality_eval/can_mh_rank_fusion/quality_episodes.csv \
    --plot
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np


def _load_pickle_payload(score_pkl: Path) -> Any:
    """Load a pickle payload, keeping compatibility with older numpy module paths."""
    class _CompatUnpickler(pickle.Unpickler):
        def find_class(self, module: str, name: str) -> Any:
            if module.startswith("numpy._core"):
                module = module.replace("numpy._core", "numpy.core", 1)
            return super().find_class(module, name)

    try:
        with score_pkl.open("rb") as f:
            return pickle.load(f)
    except ModuleNotFoundError as e:
        if "numpy._core" not in str(e):
            raise
    except SystemError:
        pass

    importlib.import_module("numpy.core")
    importlib.import_module("numpy.core.numeric")
    with score_pkl.open("rb") as f:
        return _CompatUnpickler(f).load()


def _load_scores(score_npz: Path, score_field: str | None) -> tuple[np.ndarray, str, np.ndarray | None, str | None]:
    payload = np.load(score_npz)
    files = set(payload.files)

    if score_field is not None:
        if score_field not in files:
            raise ValueError(f"{score_npz} does not contain score field: {score_field}")
        scores = np.asarray(payload[score_field], dtype=np.float64).reshape(-1)
        indices = None
        index_field = None
        if "episode_indices" in files:
            indices = np.asarray(payload["episode_indices"], dtype=np.int64).reshape(-1)
            index_field = "episode_indices"
        elif "window_indices" in files:
            indices = np.asarray(payload["window_indices"], dtype=np.int64).reshape(-1)
            index_field = "window_indices"
        return scores, score_field, indices, index_field

    for candidate in ("joint_scores", "quality_scores", "influence_scores", "normalized_influence"):
        if candidate in files:
            scores = np.asarray(payload[candidate], dtype=np.float64).reshape(-1)
            indices = None
            if "episode_indices" in files:
                indices = np.asarray(payload["episode_indices"], dtype=np.int64).reshape(-1)
            elif "window_indices" in files:
                indices = np.asarray(payload["window_indices"], dtype=np.int64).reshape(-1)
            index_field = None
            if "episode_indices" in files:
                index_field = "episode_indices"
            elif "window_indices" in files:
                index_field = "window_indices"
            return scores, candidate, indices, index_field

    raise ValueError(f"{score_npz} does not contain a recognized score field.")


def _extract_episode_vector(payload: Any, field: str) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(payload, dict):
        raise ValueError("Score pickle payload must be a dict.")

    value_source: Any | None = None
    metadata = payload.get("metadata", {})
    if field in payload:
        value_source = payload[field]
    elif isinstance(metadata, dict) and field in metadata:
        value_source = metadata[field]
    elif isinstance(payload.get("train"), dict) and field in payload["train"]:
        value_source = payload["train"][field]
    if value_source is None:
        raise ValueError(f"Score file does not contain field: {field}")

    values = np.asarray(value_source, dtype=np.float64).reshape(-1)
    if values.size == 0:
        raise ValueError(f"Field {field} is empty.")

    if isinstance(metadata, dict) and "train_episode_indices" in metadata:
        episode_indices = np.asarray(metadata["train_episode_indices"], dtype=np.int64).reshape(-1)
    elif isinstance(metadata, dict) and "episode_lengths" in metadata:
        episode_lengths = metadata["episode_lengths"]
        if len(episode_lengths) != len(values):
            raise ValueError(
                f"Field {field} has {len(values)} values but metadata.episode_lengths has {len(episode_lengths)} entries."
            )
        episode_indices = np.arange(len(values), dtype=np.int64)
    else:
        raise ValueError(
            f"Field {field} requires metadata.train_episode_indices or metadata.episode_lengths in the score pickle."
        )

    if len(episode_indices) != len(values):
        raise ValueError(
            f"Field {field} has {len(values)} values but train_episode_indices has {len(episode_indices)} entries."
        )
    return episode_indices, values


def _load_quality_csv(
    csv_path: Path,
    *,
    episode_column: str,
    quality_column: str,
) -> tuple[np.ndarray, np.ndarray]:
    episode_indices: list[int] = []
    quality_scores: list[float] = []
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"Quality CSV has no header: {csv_path}")
        if episode_column not in reader.fieldnames:
            raise ValueError(f"Quality CSV {csv_path} is missing episode column: {episode_column}")
        if quality_column not in reader.fieldnames:
            raise ValueError(f"Quality CSV {csv_path} is missing quality column: {quality_column}")
        for row in reader:
            episode_indices.append(int(row[episode_column]))
            quality_scores.append(float(row[quality_column]))
    if not episode_indices:
        raise ValueError(f"Quality CSV is empty: {csv_path}")
    return np.asarray(episode_indices, dtype=np.int64), np.asarray(quality_scores, dtype=np.float64)


def _align_episode_vectors(
    influence_episode_indices: np.ndarray,
    influence_scores: np.ndarray,
    quality_episode_indices: np.ndarray,
    quality_scores: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    influence_map = {
        int(ep): float(score)
        for ep, score in zip(influence_episode_indices, influence_scores, strict=True)
    }
    quality_map = {
        int(ep): float(score)
        for ep, score in zip(quality_episode_indices, quality_scores, strict=True)
    }
    common_episode_indices = np.asarray(sorted(set(influence_map).intersection(quality_map)), dtype=np.int64)
    if common_episode_indices.size == 0:
        raise ValueError("Influence scores and quality scores have no overlapping episode indices.")
    aligned_influence = np.asarray([influence_map[int(ep)] for ep in common_episode_indices], dtype=np.float64)
    aligned_quality = np.asarray([quality_map[int(ep)] for ep in common_episode_indices], dtype=np.float64)
    return common_episode_indices, aligned_influence, aligned_quality


def _signed_power(values: np.ndarray, power: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if power < 0:
        raise ValueError("joint powers must be >= 0")
    return np.sign(values) * (np.abs(values) ** power)


def _normalize_01(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    vmin = float(np.min(values))
    vmax = float(np.max(values))
    if vmax - vmin < 1e-12:
        return np.zeros_like(values, dtype=np.float64)
    return (values - vmin) / (vmax - vmin)


def _knee_profile(sorted_scores: np.ndarray) -> tuple[int, np.ndarray]:
    n = int(sorted_scores.size)
    if n <= 2:
        return 0, np.zeros(n, dtype=np.float64)
    x = np.linspace(0.0, 1.0, n)
    y = _normalize_01(sorted_scores)
    start = np.array([x[0], y[0]], dtype=np.float64)
    end = np.array([x[-1], y[-1]], dtype=np.float64)
    line = end - start
    denom = np.linalg.norm(line)
    if denom < 1e-12:
        idx = max(0, n // 2 - 1)
        return idx, np.zeros(n, dtype=np.float64)
    points = np.stack([x, y], axis=1)
    rel = points - start[None, :]
    distance = np.abs(rel[:, 0] * line[1] - rel[:, 1] * line[0]) / denom
    idx = int(np.argmax(distance))
    return min(max(idx, 0), n - 1), distance


def _fit_gmm2(values: np.ndarray, max_iter: int = 200, tol: float = 1e-6) -> dict[str, np.ndarray | float]:
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    if x.size < 4:
        raise ValueError("gmm2 requires at least 4 scores.")

    q25, q75 = np.quantile(x, [0.25, 0.75])
    var0 = float(np.var(x))
    sigma0 = max(np.sqrt(var0), 1e-3)
    means = np.array([q25, q75], dtype=np.float64)
    sigmas = np.array([sigma0, sigma0], dtype=np.float64)
    weights = np.array([0.5, 0.5], dtype=np.float64)

    log_likelihood_prev = -np.inf
    for _ in range(max_iter):
        pdf = []
        for k in range(2):
            sigma = max(sigmas[k], 1e-6)
            coeff = 1.0 / (np.sqrt(2.0 * np.pi) * sigma)
            exp_term = np.exp(-0.5 * ((x - means[k]) / sigma) ** 2)
            pdf.append(weights[k] * coeff * exp_term)
        pdf = np.stack(pdf, axis=1)
        total = np.sum(pdf, axis=1, keepdims=True)
        total = np.maximum(total, 1e-12)
        resp = pdf / total

        Nk = np.sum(resp, axis=0)
        weights = Nk / x.size
        means = np.sum(resp * x[:, None], axis=0) / np.maximum(Nk, 1e-12)
        sigmas = np.sqrt(
            np.sum(resp * (x[:, None] - means[None, :]) ** 2, axis=0) / np.maximum(Nk, 1e-12)
        )
        sigmas = np.maximum(sigmas, 1e-6)

        log_likelihood = float(np.sum(np.log(total[:, 0])))
        if abs(log_likelihood - log_likelihood_prev) < tol:
            break
        log_likelihood_prev = log_likelihood

    order = np.argsort(means)
    means = means[order]
    sigmas = sigmas[order]
    weights = weights[order]
    return {"means": means, "sigmas": sigmas, "weights": weights, "log_likelihood": log_likelihood_prev}


def _gmm2_threshold(values: np.ndarray) -> tuple[float, dict[str, Any]]:
    params = _fit_gmm2(values)
    x = np.asarray(values, dtype=np.float64)
    grid = np.linspace(float(np.min(x)), float(np.max(x)), 4096)
    dens = []
    for mean, sigma, weight in zip(params["means"], params["sigmas"], params["weights"], strict=True):
        sigma = max(float(sigma), 1e-6)
        coeff = 1.0 / (np.sqrt(2.0 * np.pi) * sigma)
        dens.append(float(weight) * coeff * np.exp(-0.5 * ((grid - float(mean)) / sigma) ** 2))
    left = dens[0]
    right = dens[1]
    diff = left - right
    crossing = np.where(np.sign(diff[:-1]) != np.sign(diff[1:]))[0]
    if crossing.size > 0:
        idx = int(crossing[0])
        threshold = float(grid[idx])
    else:
        threshold = float(np.mean(params["means"]))
    return threshold, {
        "means": [float(v) for v in params["means"]],
        "sigmas": [float(v) for v in params["sigmas"]],
        "weights": [float(v) for v in params["weights"]],
        "log_likelihood": float(params["log_likelihood"]),
        "grid": grid,
        "left_density": left,
        "right_density": right,
    }


def _changepoint_profile(sorted_scores: np.ndarray) -> tuple[int, np.ndarray]:
    x = np.asarray(sorted_scores, dtype=np.float64)
    n = int(x.size)
    if n <= 4:
        idx = max(0, n // 2 - 1)
        return idx, np.zeros(n, dtype=np.float64)

    prefix_sum = np.cumsum(x)
    prefix_sq = np.cumsum(x * x)
    best_idx = 0
    best_score = -np.inf
    score_curve = np.zeros(n, dtype=np.float64)
    for idx in range(1, n - 1):
        n_left = idx + 1
        n_right = n - n_left
        left_mean = prefix_sum[idx] / n_left
        right_sum = prefix_sum[-1] - prefix_sum[idx]
        right_mean = right_sum / n_right

        left_var = max(prefix_sq[idx] / n_left - left_mean * left_mean, 0.0)
        right_sq = prefix_sq[-1] - prefix_sq[idx]
        right_var = max(right_sq / n_right - right_mean * right_mean, 0.0)
        score = abs(right_mean - left_mean) / (np.sqrt(left_var + right_var) + 1e-8)
        score_curve[idx] = score
        if score > best_score:
            best_score = score
            best_idx = idx
    return best_idx, score_curve


def _ratio_from_index(index: int, total: int) -> tuple[float, float]:
    drop_count = max(0, min(index + 1, total))
    drop_ratio = drop_count / float(total)
    keep_ratio = 1.0 - drop_ratio
    return keep_ratio, drop_ratio


def _bounded_candidates(center_keep_ratio: float) -> list[float]:
    candidates = sorted(
        {
            round(float(np.clip(center_keep_ratio + delta, 0.05, 1.0)), 3)
            for delta in (-0.1, -0.05, -0.02, 0.0, 0.02, 0.05, 0.1)
        }
    )
    return candidates


def _derive_joint_scores(
    *,
    score_pkl: Path,
    quality_score_pkl: Path | None,
    quality_score_csv: Path | None,
    episode_influence_field: str,
    episode_quality_field: str,
    quality_csv_episode_column: str,
    quality_csv_score_column: str,
    joint_quality_power: float,
    joint_influence_power: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    influence_payload = _load_pickle_payload(score_pkl)
    quality_payload = _load_pickle_payload(quality_score_pkl) if quality_score_pkl is not None else influence_payload

    episode_indices, influence_scores = _extract_episode_vector(influence_payload, episode_influence_field)
    if quality_score_csv is not None:
        quality_episode_indices, quality_scores = _load_quality_csv(
            quality_score_csv,
            episode_column=quality_csv_episode_column,
            quality_column=quality_csv_score_column,
        )
    else:
        quality_episode_indices, quality_scores = _extract_episode_vector(quality_payload, episode_quality_field)

    episode_indices, influence_scores, quality_scores = _align_episode_vectors(
        episode_indices,
        influence_scores,
        quality_episode_indices,
        quality_scores,
    )

    normalized_influence = _normalize_01(influence_scores)
    joint_scores = _signed_power(quality_scores, joint_quality_power) * (
        normalized_influence ** joint_influence_power
    )

    metadata = {
        "episode_indices": episode_indices,
        "influence_scores": influence_scores,
        "quality_scores": quality_scores,
        "normalized_influence": normalized_influence,
        "joint_quality_power": float(joint_quality_power),
        "joint_influence_power": float(joint_influence_power),
        "score_pkl": str(score_pkl),
        "quality_score_pkl": None if quality_score_pkl is None else str(quality_score_pkl),
        "quality_score_csv": None if quality_score_csv is None else str(quality_score_csv),
        "episode_influence_field": episode_influence_field,
        "episode_quality_field": episode_quality_field,
    }
    return joint_scores, episode_indices, metadata


def _analyze(scores: np.ndarray) -> dict[str, Any]:
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if scores.size == 0:
        raise ValueError("Cannot analyze an empty score array.")

    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]

    knee_idx, knee_distance_curve = _knee_profile(sorted_scores)
    knee_keep, knee_drop = _ratio_from_index(knee_idx, len(sorted_scores))

    cp_idx, changepoint_score_curve = _changepoint_profile(sorted_scores)
    cp_keep, cp_drop = _ratio_from_index(cp_idx, len(sorted_scores))

    gmm_threshold = None
    gmm_keep = None
    gmm_drop = None
    gmm_info: dict[str, Any] | None = None
    gmm_plot_info: dict[str, Any] | None = None
    try:
        gmm_threshold, gmm_info_full = _gmm2_threshold(scores)
        gmm_drop = float(np.mean(scores <= gmm_threshold))
        gmm_keep = 1.0 - gmm_drop
        gmm_plot_info = {
            "grid": np.asarray(gmm_info_full["grid"], dtype=np.float64),
            "left_density": np.asarray(gmm_info_full["left_density"], dtype=np.float64),
            "right_density": np.asarray(gmm_info_full["right_density"], dtype=np.float64),
        }
        gmm_info = {
            "means": gmm_info_full["means"],
            "sigmas": gmm_info_full["sigmas"],
            "weights": gmm_info_full["weights"],
            "log_likelihood": gmm_info_full["log_likelihood"],
        }
    except Exception as e:
        gmm_info = {"error": str(e)}

    keep_candidates = []
    keep_candidates.extend(_bounded_candidates(knee_keep))
    keep_candidates.extend(_bounded_candidates(cp_keep))
    if gmm_keep is not None:
        keep_candidates.extend(_bounded_candidates(gmm_keep))
    keep_candidates = sorted(set(float(x) for x in keep_candidates))

    center_values = [knee_keep, cp_keep]
    if gmm_keep is not None:
        center_values.append(gmm_keep)
    recommended_keep = float(np.median(center_values))
    recommended_keep = float(np.clip(recommended_keep, 0.05, 1.0))

    return {
        "summary": {
            "num_scores": int(scores.size),
            "score_min": float(np.min(scores)),
            "score_max": float(np.max(scores)),
            "score_mean": float(np.mean(scores)),
            "score_std": float(np.std(scores)),
            "recommended_keep_ratio": recommended_keep,
            "recommended_drop_ratio": 1.0 - recommended_keep,
            "candidate_keep_ratios": keep_candidates,
        },
        "knee": {
            "sorted_index": int(knee_idx),
            "threshold": float(sorted_scores[knee_idx]),
            "keep_ratio": float(knee_keep),
            "drop_ratio": float(knee_drop),
            "distance_at_knee": float(knee_distance_curve[knee_idx]) if knee_distance_curve.size else 0.0,
        },
        "changepoint": {
            "sorted_index": int(cp_idx),
            "threshold": float(sorted_scores[cp_idx]),
            "keep_ratio": float(cp_keep),
            "drop_ratio": float(cp_drop),
            "score_at_changepoint": float(changepoint_score_curve[cp_idx]) if changepoint_score_curve.size else 0.0,
        },
        "gmm2": {
            "threshold": None if gmm_threshold is None else float(gmm_threshold),
            "keep_ratio": None if gmm_keep is None else float(gmm_keep),
            "drop_ratio": None if gmm_drop is None else float(gmm_drop),
            "fit": gmm_info,
        },
        "gmm2_plot": gmm_plot_info,
        "sorted_scores": sorted_scores,
        "sorted_order": order,
        "knee_distance_curve": knee_distance_curve,
        "changepoint_score_curve": changepoint_score_curve,
    }


def _maybe_plot(
    output_png: Path,
    sorted_scores: np.ndarray,
    *,
    knee: dict[str, Any],
    changepoint: dict[str, Any],
    gmm2: dict[str, Any],
    gmm2_plot: dict[str, Any] | None,
    knee_distance_curve: np.ndarray,
    changepoint_score_curve: np.ndarray,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        raise RuntimeError("Plotting requires matplotlib.") from e

    x = np.arange(sorted_scores.size)
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    ax_sorted = axes[0, 0]
    ax_hist = axes[0, 1]
    ax_knee = axes[1, 0]
    ax_cp = axes[1, 1]

    ax_sorted.plot(x, sorted_scores, lw=2, label="sorted score")
    ax_sorted.axvline(knee["sorted_index"], color="tab:orange", ls="--", label=f"knee={knee['keep_ratio']:.3f}")
    ax_sorted.axvline(
        changepoint["sorted_index"],
        color="tab:green",
        ls="--",
        label=f"changepoint={changepoint['keep_ratio']:.3f}",
    )
    if gmm2["threshold"] is not None:
        gmm_idx = int(np.searchsorted(sorted_scores, gmm2["threshold"], side="right")) - 1
        gmm_idx = max(gmm_idx, 0)
        ax_sorted.axvline(gmm_idx, color="tab:red", ls="--", label=f"gmm2={gmm2['keep_ratio']:.3f}")
    ax_sorted.set_title("Sorted Score Curve")
    ax_sorted.set_xlabel("Sorted index")
    ax_sorted.set_ylabel("Score")
    ax_sorted.legend()

    ax_hist.hist(sorted_scores, bins=min(50, max(10, sorted_scores.size // 20)), color="steelblue", alpha=0.45)
    ax_hist.axvline(knee["threshold"], color="tab:orange", ls="--", label="knee")
    ax_hist.axvline(changepoint["threshold"], color="tab:green", ls="--", label="changepoint")
    if gmm2["threshold"] is not None:
        ax_hist.axvline(gmm2["threshold"], color="tab:red", ls="--", label="gmm2")
    if gmm2_plot is not None:
        grid = np.asarray(gmm2_plot["grid"], dtype=np.float64)
        left_density = np.asarray(gmm2_plot["left_density"], dtype=np.float64)
        right_density = np.asarray(gmm2_plot["right_density"], dtype=np.float64)
        ax_hist_twin = ax_hist.twinx()
        ax_hist_twin.plot(grid, left_density, color="tab:red", alpha=0.7, label="gmm-left")
        ax_hist_twin.plot(grid, right_density, color="tab:purple", alpha=0.7, label="gmm-right")
        ax_hist_twin.set_ylabel("Density")
        handles1, labels1 = ax_hist.get_legend_handles_labels()
        handles2, labels2 = ax_hist_twin.get_legend_handles_labels()
        ax_hist.legend(handles1 + handles2, labels1 + labels2, loc="upper left")
    else:
        ax_hist.legend()
    ax_hist.set_title("Histogram + GMM Components")
    ax_hist.set_xlabel("Score")
    ax_hist.set_ylabel("Count")

    ax_knee.plot(x, knee_distance_curve, color="tab:orange", lw=2)
    ax_knee.axvline(knee["sorted_index"], color="tab:orange", ls="--", label=f"knee={knee['sorted_index']}")
    ax_knee.set_title("Knee Distance Curve")
    ax_knee.set_xlabel("Sorted index")
    ax_knee.set_ylabel("Distance to baseline")
    ax_knee.legend()

    ax_cp.plot(x, changepoint_score_curve, color="tab:green", lw=2)
    ax_cp.axvline(
        changepoint["sorted_index"],
        color="tab:green",
        ls="--",
        label=f"changepoint={changepoint['sorted_index']}",
    )
    ax_cp.set_title("Changepoint Score Curve")
    ax_cp.set_xlabel("Sorted index")
    ax_cp.set_ylabel("Mean-shift score")
    ax_cp.legend()

    fig.tight_layout()
    fig.savefig(output_png, dpi=160)
    plt.close(fig)


def _default_report_prefix(args: argparse.Namespace) -> Path:
    if args.score_npz is not None:
        return args.score_npz.with_name(args.score_npz.stem)

    assert args.score_pkl is not None
    quality_source = args.quality_score_csv or args.quality_score_pkl or args.score_pkl
    assert quality_source is not None
    quality_tag = quality_source.stem
    if quality_source == args.score_pkl:
        return args.score_pkl.with_name(f"{args.score_pkl.stem}_ratio_analysis")
    return args.score_pkl.with_name(f"{args.score_pkl.stem}__qfrom_{quality_tag}_joint_ratio_analysis")


def _write_joint_npz(output_path: Path, metadata: dict[str, Any], joint_scores: np.ndarray) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        episode_indices=np.asarray(metadata["episode_indices"], dtype=np.int64),
        influence_scores=np.asarray(metadata["influence_scores"], dtype=np.float32),
        quality_scores=np.asarray(metadata["quality_scores"], dtype=np.float32),
        normalized_influence=np.asarray(metadata["normalized_influence"], dtype=np.float32),
        joint_scores=np.asarray(joint_scores, dtype=np.float32),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze score distributions and suggest adaptive filter ratios.")
    parser.add_argument(
        "--score-npz",
        type=Path,
        default=None,
        help="joint_selection_scores.npz or joint_window_selection_scores.npz",
    )
    parser.add_argument(
        "--score-pkl",
        type=Path,
        default=None,
        help="Influence score pickle for direct joint analysis.",
    )
    parser.add_argument(
        "--quality-score-pkl",
        type=Path,
        default=None,
        help="Optional quality score pickle for direct joint analysis.",
    )
    parser.add_argument(
        "--quality-score-csv",
        type=Path,
        default=None,
        help="Optional quality score CSV for direct joint analysis.",
    )
    parser.add_argument("--score-field", type=str, default=None, help="Optional score field. Defaults to joint_scores if present.")
    parser.add_argument(
        "--quality-csv-episode-column",
        type=str,
        default="episode",
        help="Episode index column in --quality-score-csv.",
    )
    parser.add_argument(
        "--quality-csv-score-column",
        type=str,
        default="quality_score",
        help="Quality score column in --quality-score-csv.",
    )
    parser.add_argument(
        "--episode-influence-field",
        type=str,
        default="episode_influence_scores",
        help="Field name for episode-level influence scores in --score-pkl.",
    )
    parser.add_argument(
        "--episode-quality-field",
        type=str,
        default="episode_quality_scores",
        help="Field name for episode-level quality scores in a quality pickle.",
    )
    parser.add_argument(
        "--joint-quality-power",
        type=float,
        default=1.0,
        help="Exponent applied to quality scores in direct joint analysis.",
    )
    parser.add_argument(
        "--joint-influence-power",
        type=float,
        default=1.0,
        help="Exponent applied to normalized influence scores in direct joint analysis.",
    )
    parser.add_argument("--output-json", type=Path, default=None, help="Optional JSON output path.")
    parser.add_argument("--plot", action="store_true", help="Also save a plot next to the JSON/report.")
    parser.add_argument("--output-png", type=Path, default=None, help="Optional PNG output path.")
    parser.add_argument(
        "--output-joint-npz",
        type=Path,
        default=None,
        help="Optional path to save derived joint scores when using direct joint analysis.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.score_npz is not None and args.score_pkl is not None:
        raise ValueError("Use either --score-npz or --score-pkl, not both.")
    if args.score_npz is None and args.score_pkl is None:
        raise ValueError("One of --score-npz or --score-pkl is required.")
    if args.quality_score_pkl is not None and args.quality_score_csv is not None:
        raise ValueError("Only one of --quality-score-pkl or --quality-score-csv can be set.")

    derived_joint_metadata: dict[str, Any] | None = None
    if args.score_npz is not None:
        scores, used_field, indices, index_field = _load_scores(args.score_npz, args.score_field)
        input_mode = "score_npz"
    else:
        scores, indices, derived_joint_metadata = _derive_joint_scores(
            score_pkl=args.score_pkl,
            quality_score_pkl=args.quality_score_pkl,
            quality_score_csv=args.quality_score_csv,
            episode_influence_field=args.episode_influence_field,
            episode_quality_field=args.episode_quality_field,
            quality_csv_episode_column=args.quality_csv_episode_column,
            quality_csv_score_column=args.quality_csv_score_column,
            joint_quality_power=args.joint_quality_power,
            joint_influence_power=args.joint_influence_power,
        )
        used_field = "joint_scores"
        index_field = "episode_indices"
        input_mode = "joint_sources"

    result = _analyze(scores)

    report = {
        "input_mode": input_mode,
        "score_npz": None if args.score_npz is None else str(args.score_npz),
        "score_pkl": None if args.score_pkl is None else str(args.score_pkl),
        "quality_score_pkl": None if args.quality_score_pkl is None else str(args.quality_score_pkl),
        "quality_score_csv": None if args.quality_score_csv is None else str(args.quality_score_csv),
        "score_field": used_field,
        "index_field": index_field,
        "summary": result["summary"],
        "knee": result["knee"],
        "changepoint": result["changepoint"],
        "gmm2": result["gmm2"],
    }
    if derived_joint_metadata is not None:
        report["joint_score_recipe"] = {
            "joint_quality_power": derived_joint_metadata["joint_quality_power"],
            "joint_influence_power": derived_joint_metadata["joint_influence_power"],
            "episode_influence_field": derived_joint_metadata["episode_influence_field"],
            "episode_quality_field": derived_joint_metadata["episode_quality_field"],
            "num_aligned_episodes": int(len(derived_joint_metadata["episode_indices"])),
        }

    output_json = args.output_json
    if output_json is None:
        output_json = _default_report_prefix(args).with_suffix(".json")
    output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    if args.score_npz is not None:
        print(f"Score file: {args.score_npz}")
    else:
        print(f"Influence score file: {args.score_pkl}")
        print(f"Quality score file: {args.quality_score_csv or args.quality_score_pkl or args.score_pkl}")
    print(f"Score field: {used_field}")
    print(f"Analysis report: {output_json}")
    print(f"Recommended keep ratio: {report['summary']['recommended_keep_ratio']:.4f}")
    print(f"Recommended drop ratio: {report['summary']['recommended_drop_ratio']:.4f}")
    print("Ratio semantics: keep_ratio = fraction kept, drop_ratio = fraction dropped.")
    print(f"Candidate keep ratios: {report['summary']['candidate_keep_ratios']}")
    print(
        "Knee: "
        f"threshold={report['knee']['threshold']:.6f}, keep={report['knee']['keep_ratio']:.4f}, "
        f"drop={report['knee']['drop_ratio']:.4f}"
    )
    print(
        "Changepoint: "
        f"threshold={report['changepoint']['threshold']:.6f}, keep={report['changepoint']['keep_ratio']:.4f}, "
        f"drop={report['changepoint']['drop_ratio']:.4f}"
    )
    if report["gmm2"]["threshold"] is None:
        print(f"GMM2: unavailable ({report['gmm2']['fit'].get('error', 'unknown error')})")
    else:
        print(
            "GMM2: "
            f"threshold={report['gmm2']['threshold']:.6f}, keep={report['gmm2']['keep_ratio']:.4f}, "
            f"drop={report['gmm2']['drop_ratio']:.4f}"
        )

    if args.plot:
        output_png = args.output_png or _default_report_prefix(args).with_suffix(".png")
        _maybe_plot(
            output_png,
            result["sorted_scores"],
            knee=result["knee"],
            changepoint=result["changepoint"],
            gmm2=result["gmm2"],
            gmm2_plot=result["gmm2_plot"],
            knee_distance_curve=result["knee_distance_curve"],
            changepoint_score_curve=result["changepoint_score_curve"],
        )
        print(f"Analysis plot: {output_png}")

    if args.output_joint_npz is not None:
        if derived_joint_metadata is None:
            raise ValueError("--output-joint-npz is only supported with --score-pkl direct joint analysis.")
        _write_joint_npz(args.output_joint_npz, derived_joint_metadata, scores)
        print(f"Derived joint score artifact: {args.output_joint_npz}")


if __name__ == "__main__":
    main()
