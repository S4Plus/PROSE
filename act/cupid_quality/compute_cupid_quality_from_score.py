#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from compute_cupid_quality import write_outputs  # type: ignore
    from common import PairScoreData, higher_better_to_score  # type: ignore
else:
    from .compute_cupid_quality import write_outputs
    from .common import PairScoreData, higher_better_to_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reconstruct CUPID-QUALITY directly from score.pkl config + stage1 train projections."
    )
    parser.add_argument("--score-pkl", type=Path, required=True)
    parser.add_argument(
        "--stage1-train-projections",
        type=Path,
        default=None,
        help="Optional stage1 projection cache (.npy). If omitted, fall back to train_projections stored in score-pkl.",
    )
    parser.add_argument("--stage2-return-gradient", type=Path, default=None)
    parser.add_argument("--stage3-inverse-hvp", type=Path, default=None)
    parser.add_argument(
        "--episode-pair-cache",
        type=Path,
        default=None,
        help="可选：优先使用已保存的 cupid_quality episode-pair 贡献缓存，避免重复重建。",
    )
    parser.add_argument(
        "--save-episode-pair-cache",
        dest="save_episode_pair_cache",
        action="store_true",
        default=False,
        help="若未命中缓存，则在重建完成后把 episode-pair 贡献缓存落盘，供后续复用。默认关闭。",
    )
    parser.add_argument(
        "--no-save-episode-pair-cache",
        dest="save_episode_pair_cache",
        action="store_false",
        help="关闭 episode-pair 缓存落盘。",
    )
    parser.add_argument(
        "--episode-pair-cache-path",
        type=Path,
        default=None,
        help="显式指定缓存保存路径；默认与 score.pkl 同目录同名前缀。",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", type=str, default=None, help="Override device from score.pkl config.")
    parser.add_argument("--reduce-rollouts", type=str, default="mean", choices=("mean", "sum"))
    parser.add_argument("--normalize-by-trajectory-length", action="store_true")
    parser.add_argument("--normalizer", type=str, default="quantile", choices=("quantile", "minmax"))
    parser.add_argument("--quantile-low", type=float, default=0.05)
    parser.add_argument("--quantile-high", type=float, default=0.95)
    parser.add_argument("--drop-quantile", type=float, default=0.0)
    parser.add_argument("--max-eval-episodes", type=int, default=None)
    parser.add_argument("--max-eval-steps", type=int, default=None)
    parser.add_argument("--hf-home", type=Path, default=None, help="Writable HF_HOME for dataset/model caches.")
    parser.add_argument(
        "--hf-datasets-cache",
        type=Path,
        default=None,
        help="Writable HF_DATASETS_CACHE for parquet dataset locks/cache.",
    )
    return parser.parse_args()


def _load_score_payload(path: Path) -> dict:
    with path.open("rb") as f:
        payload = pickle.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"score.pkl must contain a dict payload, got {type(payload)!r}")
    if "config" not in payload:
        raise ValueError(f"{path} missing config")
    return payload


def _load_train_projections(payload: dict, stage1_train_projections: Path | None) -> np.ndarray:
    if stage1_train_projections is not None:
        return np.load(stage1_train_projections, allow_pickle=False)
    train_proj = payload.get("train_projections")
    if train_proj is None:
        raise ValueError(
            "Need --stage1-train-projections, or score-pkl must contain an inline train_projections array."
        )
    return np.asarray(train_proj, dtype=np.float32)


def _prepare_calculator(payload: dict, device_override: str | None):
    sys.path.insert(0, "/workspace/liushuo/cupid/act")
    from cupid_act_influence import CupidActInfluenceCalculator, CupidActInfluenceConfig  # type: ignore

    cfg_dict = dict(payload["config"])
    if device_override is not None:
        cfg_dict["device"] = device_override
    cfg_dict["aggregate_mode"] = "none"
    cfg = CupidActInfluenceConfig(**cfg_dict)
    return CupidActInfluenceCalculator(cfg)


def _default_episode_pair_cache_path(score_pkl: Path) -> Path:
    return score_pkl.expanduser().resolve().with_suffix(".cupid_quality_episode_pairs.npz")


def _resolve_episode_pair_cache_path(args: argparse.Namespace) -> Path:
    if getattr(args, "episode_pair_cache_path", None) is not None:
        return Path(args.episode_pair_cache_path).expanduser().resolve()
    return _default_episode_pair_cache_path(Path(args.score_pkl))


def _load_episode_pair_cache(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        required = {"episode_pair_contrib", "train_episode_ids", "eval_episode_lengths"}
        missing = sorted(required - set(payload.files))
        if missing:
            raise ValueError(f"{path} 缺少缓存字段: {missing}")
        contrib = np.asarray(payload["episode_pair_contrib"], dtype=np.float64)
        train_episode_ids = np.asarray(payload["train_episode_ids"], dtype=np.int64).reshape(-1)
        eval_episode_lengths = np.asarray(payload["eval_episode_lengths"], dtype=np.int64).reshape(-1)
    if contrib.ndim != 2:
        raise ValueError(f"{path} 中的 episode_pair_contrib 必须是 2D，当前 shape={contrib.shape}")
    if contrib.shape[0] != eval_episode_lengths.shape[0]:
        raise ValueError(
            f"{path} 中 eval_episode_lengths 与 contrib 行数不一致: "
            f"{eval_episode_lengths.shape[0]} vs {contrib.shape[0]}"
        )
    if contrib.shape[1] != train_episode_ids.shape[0]:
        raise ValueError(
            f"{path} 中 train_episode_ids 与 contrib 列数不一致: "
            f"{train_episode_ids.shape[0]} vs {contrib.shape[1]}"
        )
    return train_episode_ids, eval_episode_lengths, contrib


def _compute_quality_from_episode_pair_cache(
    args: argparse.Namespace,
    cache_path: Path,
) -> tuple[PairScoreData, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if args.max_eval_steps is not None:
        raise ValueError("episode-pair cache 不支持 --max-eval-steps；请去掉该参数或回退到逐步重建。")

    train_episode_ids, eval_episode_lengths, contrib = _load_episode_pair_cache(cache_path)
    if args.max_eval_episodes is not None:
        max_eval_eps = int(args.max_eval_episodes)
        eval_episode_lengths = eval_episode_lengths[:max_eval_eps]
        contrib = contrib[:max_eval_eps]

    contrib = np.asarray(contrib, dtype=np.float64)
    if bool(args.normalize_by_trajectory_length):
        denom = np.maximum(eval_episode_lengths.astype(np.float64), 1.0)
        contrib = contrib / denom[:, None]

    valid = np.isfinite(contrib)
    raw_sum = np.where(valid, contrib, 0.0).sum(axis=0)
    raw_count = valid.sum(axis=0).astype(np.int64)

    raw_quality = np.full(train_episode_ids.shape[0], np.nan, dtype=np.float64)
    if args.reduce_rollouts == "sum":
        raw_quality[raw_count > 0] = raw_sum[raw_count > 0]
    else:
        raw_quality[raw_count > 0] = raw_sum[raw_count > 0] / raw_count[raw_count > 0]

    quality_score = higher_better_to_score(
        raw_quality,
        normalizer=args.normalizer,
        quantile_low=float(args.quantile_low),
        quantile_high=float(args.quantile_high),
    )
    data = PairScoreData(
        pair_scores=np.empty((0, 0), dtype=np.float64),
        eval_episode_ids=np.arange(int(contrib.shape[0]), dtype=np.int64),
        train_episode_ids=train_episode_ids,
        source_path=cache_path,
    )
    print(f"[cupid_quality] loaded episode-pair cache: {cache_path}")
    return data, train_episode_ids, raw_quality, quality_score, raw_count


def _ensure_writable_hf_cache(args: argparse.Namespace) -> None:
    output_dir_value = getattr(args, "output_dir", None)
    if output_dir_value is None:
        output_dir_value = getattr(args, "quality_output_dir", None)
    if output_dir_value is None:
        raise ValueError("Need output_dir or quality_output_dir to place writable HF cache.")
    output_dir = Path(output_dir_value).expanduser().resolve()
    hf_home = Path(args.hf_home).expanduser().resolve() if args.hf_home is not None else (output_dir / "hf_home")
    hf_datasets_cache = (
        Path(args.hf_datasets_cache).expanduser().resolve()
        if args.hf_datasets_cache is not None
        else (output_dir / "hf_datasets_cache")
    )
    hf_home.mkdir(parents=True, exist_ok=True)
    hf_datasets_cache.mkdir(parents=True, exist_ok=True)
    xdg_cache_home = hf_home / "xdg_cache"
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
    print(f"[cupid_quality] HF_HOME={hf_home}")
    print(f"[cupid_quality] HF_DATASETS_CACHE={hf_datasets_cache}")


def _project_inverse_hvp(calc: CupidActInfluenceCalculator, inv_hvp: torch.Tensor) -> torch.Tensor:
    if calc.projection is not None:
        return torch.matmul(inv_hvp, calc.projection)
    if calc.projector is not None:
        return calc.projector.project(inv_hvp)
    return inv_hvp


def _build_train_episode_layout(
    train_projections: np.ndarray,
    episode_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    order = np.argsort(episode_ids, kind="stable")
    sorted_episode_ids = np.asarray(episode_ids[order], dtype=np.int64)
    unique_eps, counts = np.unique(sorted_episode_ids, return_counts=True)
    starts = np.concatenate([np.asarray([0], dtype=np.int64), np.cumsum(counts[:-1], dtype=np.int64)])
    return order, unique_eps.astype(np.int64), starts.astype(np.int64)


def _save_episode_pair_cache(
    cache_path: Path,
    *,
    score_pkl: Path,
    train_episode_ids: np.ndarray,
    eval_episode_lengths: np.ndarray,
    episode_pair_contrib: np.ndarray,
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        format_version=np.asarray("episode_pair_contrib_v1"),
        source_score_pkl=np.asarray(str(score_pkl)),
        train_episode_ids=np.asarray(train_episode_ids, dtype=np.int64),
        eval_episode_ids=np.arange(int(eval_episode_lengths.shape[0]), dtype=np.int64),
        eval_episode_lengths=np.asarray(eval_episode_lengths, dtype=np.int64),
        episode_pair_contrib=np.asarray(episode_pair_contrib, dtype=np.float32),
    )
    print(f"[cupid_quality] saved episode-pair cache -> {cache_path}")


def compute_cupid_quality_from_score(args: argparse.Namespace) -> tuple[PairScoreData, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cache_arg = getattr(args, "episode_pair_cache", None)
    auto_cache_path = _default_episode_pair_cache_path(Path(args.score_pkl))
    cache_path = None
    if cache_arg is not None:
        cache_path = Path(cache_arg).expanduser().resolve()
    elif auto_cache_path.exists():
        cache_path = auto_cache_path
    if cache_path is not None and cache_path.exists():
        return _compute_quality_from_episode_pair_cache(args, cache_path)

    payload = _load_score_payload(args.score_pkl)
    if args.stage2_return_gradient is not None:
        print(
            "[cupid_quality] note: stage2_return_gradient is a policy-level aggregate cache and is not used "
            "for exact state-action pair quality reconstruction."
        )
    if args.stage3_inverse_hvp is not None:
        print(
            "[cupid_quality] note: stage3_inverse_hvp is a policy-level aggregate cache and is not used "
            "for exact state-action pair quality reconstruction."
        )

    device_override = getattr(args, "device", None)
    if device_override is None:
        device_override = getattr(args, "reconstruct_device", None)
    _ensure_writable_hf_cache(args)
    print("[cupid_quality] initializing CupidActInfluenceCalculator")
    calc = _prepare_calculator(payload, device_override)
    print("[cupid_quality] calculator ready")
    if calc.train_episode_indices is None:
        raise RuntimeError("Failed to infer train episode indices from the training dataset.")

    train_proj = _load_train_projections(payload, args.stage1_train_projections)
    print(f"[cupid_quality] loaded train projections: shape={train_proj.shape}")
    order, train_episode_ids, starts = _build_train_episode_layout(train_proj, calc.train_episode_indices)
    sorted_train_proj = np.asarray(train_proj[order], dtype=np.float32)
    train_tensor = torch.from_numpy(sorted_train_proj).to(calc.device)

    episode_lengths = list(calc.eval_dataset.episode_lengths)
    if args.max_eval_episodes is not None:
        episode_lengths = episode_lengths[: int(args.max_eval_episodes)]
    print(f"[cupid_quality] eval episodes to process: {len(episode_lengths)}")

    num_train_eps = int(train_episode_ids.shape[0])
    raw_sum = np.zeros(num_train_eps, dtype=np.float64)
    raw_count = np.zeros(num_train_eps, dtype=np.int64)
    episode_pair_contrib = np.full((len(episode_lengths), num_train_eps), np.nan, dtype=np.float32)

    total_step_limit = None if args.max_eval_steps is None else int(args.max_eval_steps)
    processed_steps = 0

    current_episode = 0
    remaining_in_episode = episode_lengths[0] if episode_lengths else 0
    tau_best_of_worst = np.full(num_train_eps, -np.inf, dtype=np.float64)
    tau_worst_of_best = np.full(num_train_eps, np.inf, dtype=np.float64)

    for grad in calc._iter_eval_gradients():
        if current_episode >= len(episode_lengths):
            break
        if total_step_limit is not None and processed_steps >= total_step_limit:
            break

        inv_hvp = calc._solve_inverse_hvp(grad)
        inv_proj = _project_inverse_hvp(calc, inv_hvp)
        row_scores = -torch.matmul(train_tensor, inv_proj).detach().cpu().numpy()

        step_mins = np.minimum.reduceat(row_scores, starts)
        step_maxs = np.maximum.reduceat(row_scores, starts)

        tau_best_of_worst = np.maximum(tau_best_of_worst, step_mins)
        tau_worst_of_best = np.minimum(tau_worst_of_best, step_maxs)

        remaining_in_episode -= 1
        processed_steps += 1
        if remaining_in_episode == 0:
            contrib = tau_best_of_worst - tau_worst_of_best
            episode_pair_contrib[current_episode] = contrib.astype(np.float32)
            if args.normalize_by_trajectory_length:
                contrib = contrib / float(max(episode_lengths[current_episode], 1))
            raw_sum += contrib
            raw_count += 1

            current_episode += 1
            if current_episode >= len(episode_lengths):
                break
            remaining_in_episode = episode_lengths[current_episode]
            tau_best_of_worst.fill(-np.inf)
            tau_worst_of_best.fill(np.inf)

            if current_episode % 5 == 0:
                print(f"[cupid_quality] processed eval episodes: {current_episode}/{len(episode_lengths)}")

    raw_quality = np.full(num_train_eps, np.nan, dtype=np.float64)
    valid = raw_count > 0
    if args.reduce_rollouts == "sum":
        raw_quality[valid] = raw_sum[valid]
    else:
        raw_quality[valid] = raw_sum[valid] / raw_count[valid]

    quality_score = higher_better_to_score(
        raw_quality,
        normalizer=args.normalizer,
        quantile_low=float(args.quantile_low),
        quantile_high=float(args.quantile_high),
    )
    data = PairScoreData(
        pair_scores=np.empty((0, 0), dtype=np.float64),
        eval_episode_ids=np.empty((0,), dtype=np.int64),
        train_episode_ids=train_episode_ids,
        source_path=args.score_pkl,
    )
    if bool(getattr(args, "save_episode_pair_cache", False)):
        cache_path = _resolve_episode_pair_cache_path(args)
        if cache_path.exists():
            print(f"[cupid_quality] episode-pair cache already exists, skip overwrite: {cache_path}")
        else:
            _save_episode_pair_cache(
                cache_path,
                score_pkl=Path(args.score_pkl).expanduser().resolve(),
                train_episode_ids=train_episode_ids,
                eval_episode_lengths=np.asarray(episode_lengths, dtype=np.int64),
                episode_pair_contrib=episode_pair_contrib,
            )
    return data, train_episode_ids, raw_quality, quality_score, raw_count


def main() -> None:
    args = parse_args()
    data, episode_ids, raw_quality, quality_score, rollout_count = compute_cupid_quality_from_score(args)
    data.pair_scores = np.zeros((0, 0), dtype=np.float64)
    write_outputs(args, data, episode_ids, raw_quality, quality_score, rollout_count)
    print(f"[cupid_quality] train_episodes={len(episode_ids)}")
    print(f"[cupid_quality] outputs -> {args.output_dir}")


if __name__ == "__main__":
    main()
