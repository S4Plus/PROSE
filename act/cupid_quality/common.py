from __future__ import annotations

import csv
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class PairScoreData:
    pair_scores: np.ndarray
    eval_episode_ids: np.ndarray
    train_episode_ids: np.ndarray
    eval_step_ids: np.ndarray | None = None
    train_step_ids: np.ndarray | None = None
    source_path: Path | None = None


def _ensure_1d_int(values: Any, *, name: str) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1D, got shape={arr.shape}")
    if arr.size == 0:
        return arr.astype(np.int64)
    return arr.astype(np.int64, copy=False)


def _ensure_2d_float(values: Any, *, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2D, got shape={arr.shape}")
    return arr


def lower_better_to_score(
    values: np.ndarray,
    *,
    normalizer: str = "quantile",
    quantile_low: float = 0.05,
    quantile_high: float = 0.95,
) -> np.ndarray:
    vals = np.asarray(values, dtype=np.float64)
    out = np.full_like(vals, np.nan, dtype=np.float64)
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


def higher_better_to_score(
    values: np.ndarray,
    *,
    normalizer: str = "quantile",
    quantile_low: float = 0.05,
    quantile_high: float = 0.95,
) -> np.ndarray:
    vals = np.asarray(values, dtype=np.float64)
    out = np.full_like(vals, np.nan, dtype=np.float64)
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


def summarize(values: np.ndarray) -> dict[str, float]:
    vals = np.asarray(values, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return {
            "count": 0,
            "min": float("nan"),
            "max": float("nan"),
            "mean": float("nan"),
            "std": float("nan"),
            "median": float("nan"),
        }
    return {
        "count": int(vals.size),
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
        "mean": float(np.mean(vals)),
        "std": float(np.std(vals)),
        "median": float(np.median(vals)),
    }


def to_json_ready(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): to_json_ready(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_json_ready(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def _extract_dict_payload(payload: dict[str, Any]) -> PairScoreData:
    score_keys = ("pair_scores", "psi_a_inf", "scores", "influence_scores")
    eval_episode_keys = ("eval_episode_ids", "eval_episode_index", "eval_episode")
    train_episode_keys = ("train_episode_ids", "train_episode_index", "train_episode")
    eval_step_keys = ("eval_step_ids", "eval_step_index", "eval_step")
    train_step_keys = ("train_step_ids", "train_step_index", "train_step")

    pair_scores = None
    eval_episode_ids = None
    train_episode_ids = None
    eval_step_ids = None
    train_step_ids = None

    for key in score_keys:
        if key in payload:
            pair_scores = payload[key]
            break
    for key in eval_episode_keys:
        if key in payload:
            eval_episode_ids = payload[key]
            break
    for key in train_episode_keys:
        if key in payload:
            train_episode_ids = payload[key]
            break
    for key in eval_step_keys:
        if key in payload:
            eval_step_ids = payload[key]
            break
    for key in train_step_keys:
        if key in payload:
            train_step_ids = payload[key]
            break

    if pair_scores is None:
        if {"config", "train_projections", "inverse_hvp"} <= set(payload.keys()):
            raise ValueError(
                "This looks like a standard CUPID score.pkl with aggregated influence outputs, "
                "but it does not contain state-action pair scores. "
                "Expected one of: pair_scores, psi_a_inf, scores, influence_scores "
                "with shape [num_eval_state_actions, num_train_state_actions]. "
                "Use cupid_quality/compute_cupid_quality_from_score.py for score.pkl-based reconstruction."
            )
        raise ValueError("Missing pair score matrix. Expected one of: pair_scores, psi_a_inf, scores, influence_scores")
    if eval_episode_ids is None:
        raise ValueError("Missing eval episode ids. Expected one of: eval_episode_ids, eval_episode_index, eval_episode")
    if train_episode_ids is None:
        raise ValueError("Missing train episode ids. Expected one of: train_episode_ids, train_episode_index, train_episode")

    pair_scores_arr = _ensure_2d_float(pair_scores, name="pair_scores")
    eval_episode_arr = _ensure_1d_int(eval_episode_ids, name="eval_episode_ids")
    train_episode_arr = _ensure_1d_int(train_episode_ids, name="train_episode_ids")

    if pair_scores_arr.shape[0] != eval_episode_arr.shape[0]:
        raise ValueError(
            "pair_scores rows must match eval_episode_ids length, "
            f"got {pair_scores_arr.shape[0]} vs {eval_episode_arr.shape[0]}"
        )
    if pair_scores_arr.shape[1] != train_episode_arr.shape[0]:
        raise ValueError(
            "pair_scores cols must match train_episode_ids length, "
            f"got {pair_scores_arr.shape[1]} vs {train_episode_arr.shape[0]}"
        )

    eval_step_arr = None if eval_step_ids is None else _ensure_1d_int(eval_step_ids, name="eval_step_ids")
    train_step_arr = None if train_step_ids is None else _ensure_1d_int(train_step_ids, name="train_step_ids")
    return PairScoreData(
        pair_scores=pair_scores_arr,
        eval_episode_ids=eval_episode_arr,
        train_episode_ids=train_episode_arr,
        eval_step_ids=eval_step_arr,
        train_step_ids=train_step_arr,
    )


def _load_npz(path: Path) -> PairScoreData:
    with np.load(path, allow_pickle=True) as payload:
        data = {key: payload[key] for key in payload.files}
    out = _extract_dict_payload(data)
    out.source_path = path
    return out


def _load_pkl(path: Path) -> PairScoreData:
    with path.open("rb") as f:
        payload = pickle.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"PKL payload must be dict, got {type(payload)!r}")
    out = _extract_dict_payload(payload)
    out.source_path = path
    return out


def _load_csv(path: Path) -> PairScoreData:
    rows: list[tuple[int, int, int, int, float]] = []
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        required = None
        score_fields = ("psi_a_inf", "score", "influence", "pair_score")
        eval_episode_fields = ("eval_episode", "eval_episode_id")
        eval_step_fields = ("eval_step", "eval_step_id")
        train_episode_fields = ("train_episode", "train_episode_id")
        train_step_fields = ("train_step", "train_step_id")

        for row in reader:
            if required is None:
                available = set(row)

                def _pick(candidates: tuple[str, ...]) -> str:
                    for candidate in candidates:
                        if candidate in available:
                            return candidate
                    raise ValueError(f"CSV {path} missing one of columns: {candidates}")

                required = {
                    "score": _pick(score_fields),
                    "eval_episode": _pick(eval_episode_fields),
                    "eval_step": _pick(eval_step_fields),
                    "train_episode": _pick(train_episode_fields),
                    "train_step": _pick(train_step_fields),
                }

            rows.append(
                (
                    int(row[required["eval_episode"]]),
                    int(row[required["eval_step"]]),
                    int(row[required["train_episode"]]),
                    int(row[required["train_step"]]),
                    float(row[required["score"]]),
                )
            )

    if not rows:
        raise ValueError(f"No rows found in {path}")

    eval_keys = sorted({(ep, step) for ep, step, _, _, _ in rows})
    train_keys = sorted({(ep, step) for _, _, ep, step, _ in rows})
    eval_key_to_idx = {key: idx for idx, key in enumerate(eval_keys)}
    train_key_to_idx = {key: idx for idx, key in enumerate(train_keys)}

    pair_scores = np.full((len(eval_keys), len(train_keys)), np.nan, dtype=np.float64)
    for eval_ep, eval_step, train_ep, train_step, score in rows:
        pair_scores[eval_key_to_idx[(eval_ep, eval_step)], train_key_to_idx[(train_ep, train_step)]] = score

    out = PairScoreData(
        pair_scores=pair_scores,
        eval_episode_ids=np.asarray([ep for ep, _ in eval_keys], dtype=np.int64),
        train_episode_ids=np.asarray([ep for ep, _ in train_keys], dtype=np.int64),
        eval_step_ids=np.asarray([step for _, step in eval_keys], dtype=np.int64),
        train_step_ids=np.asarray([step for _, step in train_keys], dtype=np.int64),
        source_path=path,
    )
    return out


def load_pair_score_data(path: str | Path, fmt: str = "auto") -> PairScoreData:
    src = Path(path).expanduser().resolve()
    if not src.exists():
        raise FileNotFoundError(f"Input pair score file not found: {src}")

    actual_fmt = fmt
    if actual_fmt == "auto":
        suffix = src.suffix.lower()
        if suffix == ".npz":
            actual_fmt = "npz"
        elif suffix in {".pkl", ".pickle"}:
            actual_fmt = "pkl"
        elif suffix == ".csv":
            actual_fmt = "csv"
        else:
            raise ValueError(f"Cannot infer format from suffix {src.suffix!r}; pass --input-format explicitly.")

    if actual_fmt == "npz":
        return _load_npz(src)
    if actual_fmt == "pkl":
        return _load_pkl(src)
    if actual_fmt == "csv":
        return _load_csv(src)
    raise ValueError(f"Unsupported input format: {actual_fmt}")
