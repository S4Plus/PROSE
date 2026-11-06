#!/usr/bin/env python

#   CUDA_VISIBLE_DEVICES=4 nohup python /workspace/liushuo/cupid/act/cupid_quality/run_cupid_quality_train.py \
#     --score-pkl /workspace/liushuo/cupid/1_infulenece_score/can_mh/score.pkl \
#     --stage1-train-projections /workspace/liushuo/cupid/1_infulenece_score/can_mh/score.stage1_train_projections.npy \
#     --stage2-return-gradient /workspace/liushuo/cupid/1_infulenece_score/can_mh/score.stage2_return_gradient.pt \
#     --stage3-inverse-hvp /workspace/liushuo/cupid/1_infulenece_score/can_mh/score.stage3_inverse_hvp.pt \
#     --quality-output-dir /workspace/liushuo/cupid/act/cupid_quality/outputs/can_mh \
#     --dataset.root /workspace/liushuo/cupid/datasets/lerobot/can_mh \
#     --dataset.repo_id ls/can_mh \
#     --train-output-dir /workspace/liushuo/cupid/3_model/can_mh_cupid_quality \
#     --job-name can_mh_cupid_quality \
#     --normalize-by-trajectory-length \
#     --normalizer quantile \
#     --quantile-low 0.05 \
#     --quantile-high 0.95 \
#     --joint-drop-quantile 0.2 \
#     --joint-quality-power 1.0 \
#     --joint-influence-power 1.0 \
#     --weight-mode influence \
#     --use-weighted-sampler \
#     --max-steps 100000 \
#     > logs/can_mh/cupid_quality_train.log 2>&1 &




# CUDA_VISIBLE_DEVICES=3 nohup python /workspace/liushuo/cupid/act/cupid_quality/run_cupid_quality_train.py \
#     --score-pkl /workspace/liushuo/cupid/1_infulenece_score/square_mh/score_episode_window.pkl \
#     --quality-output-dir /workspace/liushuo/cupid/act/cupid_quality/outputs/square_mh \
#     --dataset.root /workspace/liushuo/cupid/datasets/lerobot/square_mh \
#     --dataset.repo_id ls/square_mh \
#     --train-output-dir /workspace/liushuo/cupid/3_model/square_mh_cupid_quality \
#     --job-name square_mh_cupid_quality \
#     --normalize-by-trajectory-length \
#     --normalizer quantile \
#     --quantile-low 0.05 \
#     --quantile-high 0.95 \
#     --joint-drop-quantile 0.2 \
#     --joint-quality-power 1.0 \
#     --joint-influence-power 1.0 \
#     --weight-mode influence \
#     --use-weighted-sampler \
#     --train-device cuda \
#     --reconstruct-device cuda \
#     --num-workers 4 \
#     --max-steps 100000 \
#     --log-every 20 \
#     --save-every 20000 \
#     --seed 1000 \
#     > logs/square_mh/cupid_quality_train.log 2>&1 &

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from common import load_pair_score_data  # type: ignore
    from compute_cupid_quality import compute_cupid_quality, write_outputs  # type: ignore
    from compute_cupid_quality_from_score import compute_cupid_quality_from_score  # type: ignore
else:
    from .common import load_pair_score_data
    from .compute_cupid_quality import compute_cupid_quality, write_outputs
    from .compute_cupid_quality_from_score import compute_cupid_quality_from_score


DEFAULT_TRAIN_SCRIPT = Path("/workspace/liushuo/cupid/act/train_act_cupid_reweight_joint.py")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute CUPID-QUALITY, filter, and retrain.")

    src = parser.add_argument_group("quality source")
    src.add_argument("--input-path", type=Path, default=None, help="Pair-score file (.npz/.pkl/.csv).")
    src.add_argument("--input-format", type=str, default="auto", choices=("auto", "npz", "pkl", "csv"))
    src.add_argument("--score-pkl", type=Path, default=None, help="Standard score.pkl for reconstruction mode.")
    src.add_argument("--stage1-train-projections", type=Path, default=None)
    src.add_argument("--stage2-return-gradient", type=Path, default=None)
    src.add_argument("--stage3-inverse-hvp", type=Path, default=None)
    src.add_argument("--episode-pair-cache", type=Path, default=None)
    src.add_argument(
        "--save-episode-pair-cache",
        dest="save_episode_pair_cache",
        action="store_true",
        default=False,
        help="若未命中缓存，则在重建完成后把 episode-pair 贡献缓存落盘，供后续复用。默认关闭。",
    )
    src.add_argument(
        "--no-save-episode-pair-cache",
        dest="save_episode_pair_cache",
        action="store_false",
        help="关闭 episode-pair 缓存落盘。",
    )
    src.add_argument("--episode-pair-cache-path", type=Path, default=None)
    src.add_argument("--reconstruct-device", type=str, default=None)
    src.add_argument("--max-eval-episodes", type=int, default=None)
    src.add_argument("--max-eval-steps", type=int, default=None)
    src.add_argument("--hf-home", type=Path, default=None)
    src.add_argument("--hf-datasets-cache", type=Path, default=None)

    quality = parser.add_argument_group("quality config")
    quality.add_argument("--quality-output-dir", type=Path, required=True)
    quality.add_argument("--reduce-rollouts", type=str, default="mean", choices=("mean", "sum"))
    quality.add_argument("--normalize-by-trajectory-length", action="store_true")
    quality.add_argument("--normalizer", type=str, default="quantile", choices=("quantile", "minmax"))
    quality.add_argument("--quantile-low", type=float, default=0.05)
    quality.add_argument("--quantile-high", type=float, default=0.95)
    quality.add_argument("--drop-quantile", type=float, default=0.0)

    train = parser.add_argument_group("train config")
    train.add_argument("--dataset.root", dest="dataset_root", type=Path, required=True)
    train.add_argument("--dataset.repo_id", dest="dataset_repo_id", type=str, required=True)
    train.add_argument("--train-output-dir", type=Path, required=True)
    train.add_argument("--job-name", type=str, default=None)
    train.add_argument("--python-bin", type=str, default=sys.executable)
    train.add_argument("--train-script", type=Path, default=DEFAULT_TRAIN_SCRIPT)
    train.add_argument("--train-device", type=str, default="cuda")
    train.add_argument("--batch-size", type=int, default=32)
    train.add_argument("--joint-drop-quantile", type=float, default=0.2)
    train.add_argument("--drop-bad-ratio", type=float, default=1.1)
    train.add_argument("--joint-quality-power", type=float, default=1.0)
    train.add_argument("--joint-influence-power", type=float, default=0.0)
    train.add_argument("--weight-mode", type=str, default="auto", choices=("auto", "none", "influence"))
    train.add_argument("--use-weighted-sampler", action="store_true")
    train.add_argument("--num-workers", type=int, default=0)
    train.add_argument("--max-steps", type=int, default=100000)
    train.add_argument("--log-every", type=int, default=20)
    train.add_argument("--save-every", type=int, default=20000)
    train.add_argument("--seed", type=int, default=1000)
    train.add_argument("--dry-run", action="store_true")
    train.add_argument("--skip-train", action="store_true")
    return parser.parse_args()


def _compute_quality(args: argparse.Namespace) -> Path:
    if args.input_path is not None:
        data = load_pair_score_data(args.input_path, fmt=args.input_format)
        episode_ids, raw_quality, quality_score, rollout_count = compute_cupid_quality(
            data,
            reduce_rollouts=args.reduce_rollouts,
            normalize_by_trajectory_length=bool(args.normalize_by_trajectory_length),
            normalizer=args.normalizer,
            quantile_low=float(args.quantile_low),
            quantile_high=float(args.quantile_high),
        )
    else:
        if args.score_pkl is None:
            raise ValueError("Either provide --input-path, or provide --score-pkl.")
        data, episode_ids, raw_quality, quality_score, rollout_count = compute_cupid_quality_from_score(args)

    write_outputs(
        argparse.Namespace(
            output_dir=args.quality_output_dir,
            normalize_by_trajectory_length=args.normalize_by_trajectory_length,
            reduce_rollouts=args.reduce_rollouts,
            drop_quantile=args.drop_quantile,
            input_path=args.input_path if args.input_path is not None else args.score_pkl,
            input_format=args.input_format,
            normalizer=args.normalizer,
            quantile_low=args.quantile_low,
            quantile_high=args.quantile_high,
        ),
        data,
        episode_ids,
        raw_quality,
        quality_score,
        rollout_count,
    )
    return args.quality_output_dir / "quality_episodes.csv"


def _resolved_weight_mode(args: argparse.Namespace) -> str:
    if args.weight_mode != "auto":
        return args.weight_mode
    if float(args.joint_influence_power) > 0.0:
        return "influence"
    return "none"


def _build_train_cmd(args: argparse.Namespace, quality_csv: Path) -> list[str]:
    cmd = [
        args.python_bin,
        str(args.train_script),
        "--dataset.root",
        str(args.dataset_root),
        "--dataset.repo_id",
        str(args.dataset_repo_id),
        "--output-dir",
        str(args.train_output_dir),
        "--job_name",
        str(args.job_name or args.train_output_dir.name),
        "--device",
        str(args.train_device),
        "--batch-size",
        str(args.batch_size),
        "--num-workers",
        str(args.num_workers),
        "--max-steps",
        str(args.max_steps),
        "--log-every",
        str(args.log_every),
        "--save-every",
        str(args.save_every),
        "--seed",
        str(args.seed),
        "--joint-quality-source",
        "quality_csv",
        "--quality-score-csv",
        str(quality_csv),
        "--filter-score-mode",
        "joint",
        "--joint-drop-quantile",
        str(args.joint_drop_quantile),
        "--joint-quality-power",
        str(args.joint_quality_power),
        "--joint-influence-power",
        str(args.joint_influence_power),
        "--drop-bad-ratio",
        str(args.drop_bad_ratio),
        "--weight-mode",
        _resolved_weight_mode(args),
    ]
    if float(args.joint_influence_power) > 0.0:
        if args.score_pkl is None:
            raise ValueError("--score-pkl is required when --joint-influence-power > 0.")
        cmd.extend(["--score-pkl", str(args.score_pkl)])
    if args.use_weighted_sampler:
        cmd.append("--use-weighted-sampler")
    return cmd


def main() -> None:
    args = parse_args()
    quality_csv = _compute_quality(args)
    print(f"[cupid_quality] quality csv -> {quality_csv}")
    if args.skip_train:
        return

    cmd = _build_train_cmd(args, quality_csv)
    print("[cupid_quality] train command:")
    print(" " + " ".join(shlex.quote(part) for part in cmd))
    if args.dry_run:
        return
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
