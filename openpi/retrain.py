
#!/usr/bin/env python3
"""
Filter low-scoring training units using a CUPID/OpenPI score pickle and retrain on the remainder.

The script can work in two modes:
1. Episode mode: score count matches dataset episodes, materialize a filtered dataset.
2. Sample mode: score count refers to training samples, save sample indices and retrain
   against the original dataset using only those indices.

Usage (example):
  python retrain.py \
    --score-pkl /workspace/liushuo/cupid/mimicgen/eval/tmp/can_ckpt10000_abs/score/influence_openpi05_last2_0.pkl \
    --keep-ratio 0.8 \
    --prepare-only




  mkdir -p logs/can_mh


 UV_CACHE_DIR=./tmp/uv-cache uv run python retrain.py \
    --score-pkl /workspace/liushuo/cupid/mimicgen/eval/tmp/can_ckpt10000_abs/score/influence_openpi05_last2_0.pkl \
    --keep-ratio 0.8 \
    --output-indices /workspace/liushuo/cupid/datasets/lerobot/can_mh/can_mh_keep80_last2_0_sample_indices.npy \
    --prepare-only \
    --overwrite

  nohup bash -lc '
  CUDA_VISIBLE_DEVICES=0 \
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
  WANDB_MODE=disabled \
  UV_CACHE_DIR=./tmp/uv-cache \
  uv run scripts/train.py pi05_can_mh_lora \
    --exp-name can_mh_keep80_last2_0 \
    --data.sample-indices-path /workspace/liushuo/cupid/datasets/lerobot/can_mh/can_mh_keep80_last2_0_sample_indices.npy \
    --num-train-steps 10000 \
    --overwrite
  ' > logs/can_mh/can_mh_keep80_last2_0.log 2>&1 &





 UV_CACHE_DIR=./tmp/uv-cache uv run python retrain.py \
    --score-pkl /workspace/liushuo/cupid/mimicgen/eval/tmp/can_ckpt10000_abs/score/influence_openpi05_last2_0.pkl \
    --keep-ratio 0.9 \
    --output-indices /workspace/liushuo/cupid/datasets/lerobot/can_mh/can_mh_keep90_last2_0_sample_indices.npy \
    --prepare-only \
    --overwrite

  nohup bash -lc '
  CUDA_VISIBLE_DEVICES=1 \
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
  WANDB_MODE=disabled \
  UV_CACHE_DIR=./tmp/uv-cache \
  uv run scripts/train.py pi05_can_mh_lora \
    --exp-name can_mh_keep90_last2_0 \
    --data.sample-indices-path /workspace/liushuo/cupid/datasets/lerobot/can_mh/can_mh_keep90_last2_0_sample_indices.npy \
    --num-train-steps 10000 \
    --overwrite
  ' > logs/can_mh/can_mh_keep90_last2_0.log 2>&1 &


 UV_CACHE_DIR=./tmp/uv-cache uv run python retrain.py \
    --score-pkl /workspace/liushuo/cupid/mimicgen/eval/tmp/can_ckpt10000_abs/score/influence_openpi05_last2_0.pkl \
    --keep-ratio 0.7 \
    --output-indices /workspace/liushuo/cupid/datasets/lerobot/can_mh/can_mh_keep70_last2_0_sample_indices.npy \
    --prepare-only \
    --overwrite

  nohup bash -lc '
  CUDA_VISIBLE_DEVICES=1 \
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
  WANDB_MODE=disabled \
  UV_CACHE_DIR=./tmp/uv-cache \
  uv run scripts/train.py pi05_can_mh_lora \
    --exp-name can_mh_keep70_last2_0 \
    --data.sample-indices-path /workspace/liushuo/cupid/datasets/lerobot/can_mh/can_mh_keep70_last2_0_sample_indices.npy \
    --num-train-steps 10000 \
    --overwrite
  ' > logs/can_mh/can_mh_keep70_last2_0.log 2>&1 &
"""

from __future__ import annotations

import argparse
import dataclasses
import pickle
import shutil
import sys
import importlib
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def _load_score_payload(score_pkl: Path) -> Any:
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
        # Older pickles may reference pre-rename numpy internal modules.
        if "numpy._core" not in str(e):
            raise
    except SystemError:
        # Some environments crash when aliasing numpy internals through sys.modules.
        pass

    importlib.import_module("numpy.core")
    importlib.import_module("numpy.core.numeric")
    with score_pkl.open("rb") as f:
        return _CompatUnpickler(f).load()


def _load_scores(payload: Any, score_pkl: Path) -> np.ndarray:
    """Extract influence scores from a score pickle payload."""

    def _extract(obj: dict) -> np.ndarray | None:
        if isinstance(obj, dict) and "influence_scores" in obj:
            return np.asarray(obj["influence_scores"], dtype=float)
        return None

    scores = _extract(payload)
    if scores is None and isinstance(payload, dict):
        scores = _extract(payload.get("train", {}))
    if scores is None:
        raise ValueError(f"{score_pkl} does not contain influence_scores")
    return scores


def _select_keep_indices(scores: np.ndarray, keep_ratio: float) -> np.ndarray:
    """Return sorted indices to keep after dropping the lowest scores."""
    if not 0.0 < keep_ratio <= 1.0:
        raise ValueError("--keep-ratio must be in (0,1]")
    keep = max(1, int(len(scores) * keep_ratio))
    order = np.argsort(scores)[::-1]  # high to low
    keep_indices = np.sort(order[:keep])
    return keep_indices


def _extract_score_config_name(payload: Any) -> str | None:
    if not isinstance(payload, dict):
        return None
    config = payload.get("config")
    if not isinstance(config, dict):
        return None
    config_name = config.get("config_name")
    return str(config_name) if config_name else None


def _extract_scored_sample_indices(payload: Any) -> np.ndarray | None:
    if not isinstance(payload, dict):
        return None

    candidates = [
        payload.get("sample_indices"),
        payload.get("train_sample_indices"),
        payload.get("metadata", {}).get("sample_indices") if isinstance(payload.get("metadata"), dict) else None,
    ]
    for value in candidates:
        if value is None:
            continue
        indices = np.asarray(value, dtype=int).reshape(-1)
        if indices.size > 0:
            return indices
    return None


def _copy_file(src: Path | str, dst: Path | str) -> None:
    """Copy a single file, creating parent dirs."""
    src_path = Path(src)
    dst_path = Path(dst)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_path, dst_path)


def _materialize_subset(src_root: Path, dst_root: Path, keep_indices: np.ndarray, overwrite: bool) -> None:
    if dst_root.exists():
        if not overwrite:
            raise FileExistsError(f"{dst_root} already exists; pass --overwrite to reuse it")
        shutil.rmtree(dst_root)

    # Prepare fresh dirs.
    (dst_root / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
    (dst_root / "meta").mkdir(parents=True, exist_ok=True)
    # Remove any stale cache if present.
    if (dst_root / "hf_datasets_cache").exists():
        shutil.rmtree(dst_root / "hf_datasets_cache")
    # Copy norm stats if available.
    src_norm = src_root / "norm_stats.json"
    if src_norm.exists():
        _copy_file(src_norm, dst_root / "norm_stats.json")

    # Build mapping old_idx -> new_idx (contiguous).
    keep_indices = np.asarray(keep_indices, dtype=int)
    mapping = {old: new for new, old in enumerate(keep_indices)}

    # Copy/rewrite parquet/actions with new contiguous naming and record lengths from parquet.
    src_data = src_root / "data" / "chunk-000"
    dst_data = dst_root / "data" / "chunk-000"
    lengths: dict[int, int] = {}
    global_index = 0
    for old_idx, new_idx in mapping.items():
        old_stem = f"episode_{old_idx:06d}"
        new_stem = f"episode_{new_idx:06d}"
        src_parquet = src_data / f"{old_stem}.parquet"
        if src_parquet.exists():
            table = pq.read_table(src_parquet)
            n = table.num_rows
            # rewrite episode_index, frame_index, index, preserving column order/schema
            new_arrays = []
            for field in table.schema:
                name = field.name
                if name == "episode_index":
                    arr = pa.array([new_idx] * n, type=field.type)
                elif name == "frame_index":
                    arr = pa.array(range(n), type=field.type)
                elif name == "index":
                    arr = pa.array(range(global_index, global_index + n), type=field.type)
                else:
                    arr = table[name]
                new_arrays.append(arr)
            new_table = pa.Table.from_arrays(new_arrays, schema=table.schema)
            pq.write_table(new_table, dst_data / f"{new_stem}.parquet")
            lengths[new_idx] = n
            global_index += n
        # copy actions txt if present
        src_actions = src_data / f"{old_stem}_actions.txt"
        if src_actions.exists():
            _copy_file(src_actions, dst_data / f"{new_stem}_actions.txt")

    # Copy videos if present, renaming to new indices.
    src_videos = src_root / "videos" / "chunk-000"
    if src_videos.exists():
        for cam_dir in src_videos.iterdir():
            if not cam_dir.is_dir():
                continue
            dst_cam = dst_root / "videos" / "chunk-000" / cam_dir.name
            dst_cam.mkdir(parents=True, exist_ok=True)
            for old_idx, new_idx in mapping.items():
                src_file = cam_dir / f"episode_{old_idx:06d}.mp4"
                if src_file.exists():
                    _copy_file(src_file, dst_cam / f"episode_{new_idx:06d}.mp4")

    # Filter and rewrite meta.
    src_meta = src_root / "meta"
    dst_meta = dst_root / "meta"

    # episodes.jsonl
    episodes_path = src_meta / "episodes.jsonl"
    total_frames = 0
    if episodes_path.exists():
        out_lines = []
        import json

        with episodes_path.open() as f:
            for line in f:
                rec = line.strip()
                if not rec:
                    continue
                obj = json.loads(rec)
                old_idx = obj.get("episode_index")
                if old_idx in mapping:
                    new_idx = mapping[old_idx]
                    obj["episode_index"] = new_idx
                    if new_idx in lengths:
                        obj["length"] = lengths[new_idx]
                    total_frames += obj.get("length", 0)
                    out_lines.append(json.dumps(obj))
        (dst_meta / "episodes.jsonl").write_text("\n".join(out_lines) + ("\n" if out_lines else ""))
    else:
        total_frames = 0

    # episodes_stats.jsonl (preserve but fix episode_index only).
    stats_path = src_meta / "episodes_stats.jsonl"
    if stats_path.exists():
        out_lines = []
        import json

        with stats_path.open() as f:
            for line in f:
                rec = line.strip()
                if not rec:
                    continue
                obj = json.loads(rec)
                old_idx = obj.get("episode_index")
                if old_idx in mapping:
                    obj["episode_index"] = mapping[old_idx]
                    out_lines.append(json.dumps(obj))
        (dst_meta / "episodes_stats.jsonl").write_text("\n".join(out_lines) + ("\n" if out_lines else ""))

    # tasks.jsonl
    tasks_path = src_meta / "tasks.jsonl"
    if tasks_path.exists():
        shutil.copy2(tasks_path, dst_meta / "tasks.jsonl")

    # info.json
    info_path = src_meta / "info.json"
    if info_path.exists():
        import json

        info = json.loads(info_path.read_text())
        orig_total_episodes = info.get("total_episodes", len(mapping))
        info["total_episodes"] = len(mapping)
        info["total_frames"] = total_frames
        info["total_videos"] = (
            info.get("total_videos", 0) * len(mapping) // max(orig_total_episodes, 1)
        )
        info["chunks_size"] = len(mapping)
        info["splits"] = {"train": f"0:{len(mapping)}"}
        info["data_path"] = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
        info["video_path"] = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
        (dst_meta / "info.json").write_text(json.dumps(info, indent=4))


def _build_train_config(
    train_config_name: str,
    *,
    filtered_repo: Path | None,
    sample_indices_path: Path | None,
    exp_name: str,
    checkpoint_base_dir: Path,
):
    from openpi.training import config as _config

    cfg = _config.get_config(train_config_name)
    data_cfg = dataclasses.replace(
        cfg.data,
        repo_id=str(filtered_repo) if filtered_repo is not None else cfg.data.repo_id,
        sample_indices_path=str(sample_indices_path) if sample_indices_path is not None else cfg.data.sample_indices_path,
    )
    cfg = dataclasses.replace(
        cfg,
        data=data_cfg,
        exp_name=exp_name,
        checkpoint_base_dir=str(checkpoint_base_dir),
    )
    return cfg


def _infer_defaults_from_train_config(train_config_name: str) -> tuple[Path, str]:
    from openpi.training import config as _config

    cfg = _config.get_config(train_config_name)
    repo_id = cfg.data.repo_id
    if repo_id is None:
        raise ValueError(f"Train config {train_config_name} does not define a dataset repo_id")
    return Path(repo_id), cfg.exp_name


def _default_output_dataset(dataset_dir: Path, keep_ratio: float, score_pkl: Path) -> Path:
    keep_pct = int(round(keep_ratio * 100))
    score_tag = score_pkl.stem
    return dataset_dir.parent / f"{dataset_dir.name}_keep{keep_pct}_{score_tag}"


def _default_output_indices(dataset_dir: Path, keep_ratio: float, score_pkl: Path) -> Path:
    keep_pct = int(round(keep_ratio * 100))
    score_tag = score_pkl.stem
    return dataset_dir.parent / f"{dataset_dir.name}_keep{keep_pct}_{score_tag}_sample_indices.npy"


def _count_dataset_episodes(dataset_dir: Path) -> int:
    meta_info = dataset_dir / "meta" / "info.json"
    if meta_info.exists():
        import json

        info = json.loads(meta_info.read_text())
        total = info.get("total_episodes")
        if total is not None:
            return int(total)
    return len(list((dataset_dir / "data" / "chunk-000").glob("episode_*.parquet")))


def _resolve_score_unit(score_unit: str, score_count: int, dataset_dir: Path) -> str:
    if score_unit != "auto":
        return score_unit
    dataset_episodes = _count_dataset_episodes(dataset_dir)
    return "episode" if score_count == dataset_episodes else "sample"


def _save_sample_indices(path: Path, sample_indices: np.ndarray, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists; pass --overwrite to reuse it")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.asarray(sample_indices, dtype=np.int64))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter low-score data and retrain OpenPI pi05.")
    parser.add_argument("--score-pkl", type=Path, required=True, help="Path to score.pkl with influence_scores")
    parser.add_argument(
        "--keep-ratio",
        type=float,
        default=0.8,
        help="Fraction of highest-scoring episodes to keep (default: 0.8 == keep top 80%)",
    )
    parser.add_argument(
        "--score-unit",
        choices=("auto", "episode", "sample"),
        default="auto",
        help="Interpret scores as episode-level or sample-level. Default infers from dataset episode count.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=None,
        help="Original dataset root. If omitted, infer from score.pkl config metadata.",
    )
    parser.add_argument(
        "--output-dataset",
        type=Path,
        default=None,
        help="Filtered dataset output root. If omitted, derive from dataset-dir and keep-ratio.",
    )
    parser.add_argument(
        "--output-indices",
        type=Path,
        default=None,
        help="Output `.npy` path for kept sample indices in sample mode.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output dataset directory")
    parser.add_argument(
        "--train-config",
        type=str,
        default=None,
        help="Training config name defined in openpi.training.config. If omitted, infer from score.pkl.",
    )
    parser.add_argument(
        "--exp-name",
        type=str,
        default=None,
        help="Experiment name for checkpoints/logs. If omitted, derive from train-config and keep-ratio.",
    )
    parser.add_argument(
        "--checkpoint-base-dir",
        type=Path,
        default=Path("./checkpoints"),
        help="Base directory to store checkpoints",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Only build the filtered dataset, skip launching training",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    payload = _load_score_payload(args.score_pkl)
    scores = _load_scores(payload, args.score_pkl)
    inferred_train_config = args.train_config or _extract_score_config_name(payload)
    if args.dataset_dir is None:
        if inferred_train_config is None:
            raise ValueError("--dataset-dir is required when score.pkl does not contain config.config_name")
        dataset_dir, base_exp_name = _infer_defaults_from_train_config(inferred_train_config)
    else:
        dataset_dir = args.dataset_dir
        base_exp_name = inferred_train_config or "retrain"

    train_config_name = inferred_train_config
    if train_config_name is None and not args.prepare_only:
        raise ValueError("--train-config is required to launch training")

    keep_indices = _select_keep_indices(scores, args.keep_ratio)
    keep_pct = int(round(args.keep_ratio * 100))
    exp_name = args.exp_name or f"{(train_config_name or base_exp_name)}_keep{keep_pct}"
    score_unit = _resolve_score_unit(args.score_unit, len(scores), dataset_dir)
    print(
        f"Loaded {len(scores)} scores from {args.score_pkl}, "
        f"keep top {args.keep_ratio*100:.1f}% -> keep {len(keep_indices)} {score_unit}s."
    )
    print(f"Dataset: {dataset_dir}")

    sample_indices_path: Path | None = None
    filtered_repo: Path | None = None
    if score_unit == "episode":
        dataset_episodes = _count_dataset_episodes(dataset_dir)
        if len(scores) != dataset_episodes:
            raise ValueError(
                f"Episode mode requires one score per episode, but got {len(scores)} scores for {dataset_episodes} episodes"
            )
        output_dataset = args.output_dataset or _default_output_dataset(dataset_dir, args.keep_ratio, args.score_pkl)
        print(f"Output dataset: {output_dataset}")
        _materialize_subset(dataset_dir, output_dataset, keep_indices, overwrite=args.overwrite)
        print(f"Filtered dataset ready at {output_dataset}")
        filtered_repo = output_dataset
    else:
        output_indices = args.output_indices or _default_output_indices(dataset_dir, args.keep_ratio, args.score_pkl)
        scored_sample_indices = _extract_scored_sample_indices(payload)
        if scored_sample_indices is None:
            scored_sample_indices = np.arange(len(scores), dtype=int)
            print("Sample indices not found in score.pkl; inferring sequential dataset indices [0, N).")
        if len(scored_sample_indices) != len(scores):
            raise ValueError(
                f"sample index count {len(scored_sample_indices)} does not match score count {len(scores)}"
            )
        kept_sample_indices = np.asarray(scored_sample_indices, dtype=int)[keep_indices]
        print(f"Output sample indices: {output_indices}")
        _save_sample_indices(output_indices, kept_sample_indices, overwrite=args.overwrite)
        print(f"Filtered sample-index file ready at {output_indices}")
        sample_indices_path = output_indices

    if args.prepare_only:
        return

    cfg = _build_train_config(
        train_config_name,
        filtered_repo=filtered_repo,
        sample_indices_path=sample_indices_path,
        exp_name=exp_name,
        checkpoint_base_dir=args.checkpoint_base_dir,
    )
    print(
        f"Starting training with config={train_config_name}, exp_name={cfg.exp_name}, "
        f"dataset={filtered_repo or dataset_dir}, sample_indices={sample_indices_path}"
    )
    from scripts import train as train_script

    train_script.main(cfg)


if __name__ == "__main__":
    main()
