#!/usr/bin/env python3
"""Preview or run one ACT stage using a portable, recorded configuration."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
STAGES = ("train", "score", "select", "retrain")
KEYS = {
    "run_name", "dataset_root", "dataset_repo_id", "pretrained_dir", "rollout_dir", "output_root",
    "seed", "device", "steps", "batch_size", "num_workers", "window_size", "window_stride",
    "episode_drop_ratio",
}


def resolve_path(value: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Paths must be nonempty strings.")
    expanded = os.path.expandvars(value)
    if "$" in expanded:
        raise ValueError(f"Unresolved environment variable in {value!r}")
    path = Path(expanded).expanduser()
    return (ROOT / path).resolve() if not path.is_absolute() else path.resolve()


def load_config(path: Path) -> dict:
    config = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(config, dict):
        raise ValueError("The configuration must be a JSON object.")
    unknown = config.keys() - KEYS
    missing = KEYS - config.keys()
    if unknown or missing:
        raise ValueError(f"Unknown keys: {sorted(unknown)}; missing keys: {sorted(missing)}")
    name = config["run_name"]
    if not isinstance(name, str) or not name or name in (".", "..") or any(c in name for c in '/\\:'):
        raise ValueError("run_name must be a single directory name.")
    for key in ("dataset_repo_id", "device"):
        if not isinstance(config[key], str) or not config[key].strip():
            raise ValueError(f"{key} must be a nonempty string.")
    for key in ("steps", "batch_size", "window_size", "window_stride", "seed", "num_workers"):
        minimum = 0 if key in ("seed", "num_workers") else 1
        if type(config[key]) is not int or config[key] < minimum:
            raise ValueError(f"{key} must be an integer >= {minimum}.")
    ratio = config["episode_drop_ratio"]
    if type(ratio) not in (int, float) or not math.isfinite(ratio) or not 0 <= ratio < 1:
        raise ValueError("episode_drop_ratio must be in [0, 1).")
    for key in ("dataset_root", "pretrained_dir", "rollout_dir", "output_root"):
        config[key] = str(resolve_path(config[key]))
    return config


def artifacts(config: dict) -> dict[str, Path]:
    run = Path(config["output_root"]) / config["run_name"]
    return {
        "run": run,
        "baseline": run / "baseline",
        "score": run / "influence" / "score.pkl",
        "cache": run / "influence" / "score",
        "selection": run / "selection",
        "selection_npz": run / "selection" / "joint_episode_window_selection.npz",
        "retrain": run / "retrain",
    }


def build_command(config: dict, stage: str) -> list[str]:
    if stage not in STAGES:
        raise ValueError(f"Unknown stage: {stage}")
    paths = artifacts(config)
    dataset = config["dataset_root"]
    if stage == "train":
        return [
            sys.executable, "-m", "lerobot.scripts.train", "--policy.type=act",
            f"--dataset.root={dataset}", f"--dataset.repo_id={config['dataset_repo_id']}",
            f"--output_dir={paths['baseline']}", f"--job_name={config['run_name']}_baseline",
            f"--policy.device={config['device']}", "--policy.push_to_hub=false",
            "--wandb.enable=false", f"--seed={config['seed']}", f"--steps={config['steps']}",
            f"--batch_size={config['batch_size']}", f"--num_workers={config['num_workers']}",
        ]
    if stage == "score":
        return [
            sys.executable, str(ROOT / "act/myscripts/cupid_act_influence.py"),
            "--pretrained_dir", config["pretrained_dir"], "--train_dataset_path", dataset,
            "--eval_rollout_dir", config["rollout_dir"], "--output_path", str(paths["score"]),
            "--stage_cache_prefix", str(paths["cache"]), "--aggregate_mode", "episode",
            "--window_size", str(config["window_size"]), "--window_stride", str(config["window_stride"]),
            "--seed", str(config["seed"]), "--device", config["device"],
        ]
    if stage == "select":
        return [
            sys.executable, str(ROOT / "act/myscripts/filter_joint_episode_window.py"),
            "--dataset-root", dataset, "--score-pkl", str(paths["score"]),
            "--output-dir", str(paths["selection"]), "--quality-source", "trajectory_eval",
            "--episode-drop-ratio", str(config["episode_drop_ratio"]),
            "--window-size", str(config["window_size"]), "--window-stride", str(config["window_stride"]),
            "--budget-mode", "auto_marginal",
        ]
    return [
        sys.executable, str(ROOT / "act/train_act_cupid_reweight_joint.py"),
        "--dataset.root", dataset, "--dataset.repo_id", config["dataset_repo_id"],
        "--score-pkl", str(paths["score"]), "--selection-npz", str(paths["selection_npz"]),
        "--selection-override-episodes", "--output_dir", str(paths["retrain"]),
        "--job_name", f"{config['run_name']}_retrain", "--policy.type", "act",
        "--joint-drop-quantile", "0", "--drop-weight-quantile", "0", "--drop-bad-ratio", "1.1",
        "--joint-quality-source", "trajectory_eval", "--weight-mode", "influence",
        "--use-weighted-sampler", "--segment-mask", "--segment-weight",
        "--seed", str(config["seed"]), "--max-steps", str(config["steps"]),
        "--batch-size", str(config["batch_size"]), "--num-workers", str(config["num_workers"]),
        "--device", config["device"],
    ]


def preflight(config: dict, stage: str) -> list[str]:
    """Check only inputs needed by this stage; never create or alter files."""
    errors = []
    paths = artifacts(config)
    metadata = Path(config["dataset_root"]) / "meta/info.json"
    if not metadata.is_file():
        errors.append(f"Missing LeRobot dataset metadata: {metadata}")
    if stage == "score":
        checkpoint = Path(config["pretrained_dir"])
        for name in ("config.json", "model.safetensors"):
            if not (checkpoint / name).is_file():
                errors.append(f"Missing checkpoint file: {checkpoint / name}")
        rollouts = Path(config["rollout_dir"])
        if not rollouts.is_dir() or not any(rollouts.glob("*.pkl")):
            errors.append(f"No rollout .pkl files found directly in: {rollouts}")
    if stage in ("select", "retrain") and not paths["score"].is_file():
        errors.append(f"Missing influence scores: {paths['score']}")
    if stage == "retrain" and not paths["selection_npz"].is_file():
        errors.append(f"Missing selection result: {paths['selection_npz']}")
    if stage in ("train", "retrain") and paths["baseline" if stage == "train" else "retrain"].exists():
        errors.append("Training output already exists; use a new run_name/output_root to avoid overwriting it.")
    return errors


def execute(config: dict, stage: str, command: list[str]) -> int:
    paths = artifacts(config)
    attempt = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    log_dir = paths["run"] / "logs" / f"{stage}_{attempt}"
    log_dir.mkdir(parents=True)
    env = os.environ.copy()
    env.setdefault("WANDB_MODE", "disabled")
    lerobot = Path(env.get("LEROBOT_REPO", str(ROOT / "lerobot_2.1"))).expanduser().resolve()
    env["LEROBOT_REPO"] = str(lerobot)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(ROOT), str(lerobot / "src"), env.get("PYTHONPATH")]))
    try:
        revision = subprocess.check_output([
            "git", "-c", f"safe.directory={ROOT.as_posix()}", "-C", str(ROOT), "rev-parse", "HEAD"
        ], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    manifest = {
        "stage": stage, "started_utc": attempt, "git_revision": revision,
        "config": config, "command": command, "cwd": str(ROOT), "python_version": sys.version,
        "packages": sorted(f"{d.metadata['Name']}=={d.version}" for d in importlib.metadata.distributions()),
        "lerobot_repo": str(lerobot),
    }
    manifest_path = log_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Log: {log_dir / 'stdout.log'}", flush=True)
    try:
        with (log_dir / "stdout.log").open("w", encoding="utf-8") as log:
            result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
        code = result.returncode
    except OSError as exc:
        manifest["error"] = str(exc)
        code = 1
    manifest["returncode"] = code
    manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return code


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/reproduction/act.example.json")
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--check", action="store_true", help="Check stage inputs without launching a job.")
    parser.add_argument("--execute", action="store_true", help="Check inputs, then run the selected stage.")
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config.expanduser().resolve())
        command = build_command(config, args.stage)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(shlex.join(command))
    if args.check or args.execute:
        errors = preflight(config, args.stage)
        if errors:
            for error in errors:
                print(f"ERROR: {error}", file=sys.stderr)
            return 1
        print("Stage inputs are present. Runtime dependencies and GPU availability are checked by the framework.")
    if args.execute:
        return execute(config, args.stage, command)
    print("Preview only. Add --execute to run; add --check to validate local inputs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
