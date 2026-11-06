#!/usr/bin/env python3
"""
评估“9维：7维机械臂的绝对关节角度 + gripper qpos(2)”动作的新数据集训练出的 ACT 模型。

用法示例：
python mymimicgen/eval_random_act_joint.py \
  --policy-path /workspace/liushuo/cupid/outputs_new/retrain_cupid/ratio_drop_0.2/checkpoints/200000/pretrained_model\
  --env-task square_d1 \
  --n-episodes 100
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

FAILURE_TAGS: Dict[str, str] = {
    "success": "成功",
    "approach": "接近/对齐失败",
    "grasp": "抓取失败（抓空/时机/滑落/不稳）",
    "transport": "搬运/路径失败（碰撞/绕不过/拖拽偏移）",
    "place": "放置失败（放偏/提前松手/角度不对）",
    "recover": "恢复失败（不会回到正轨）",
    "ood": "分布外失败（光照/背景/姿态/尺度/遮挡差异）",
}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy-path", required=True, type=Path, help="Path to pretrained_model directory.")
    parser.add_argument("--env-task", default="square_d1", help="Task name (e.g., square_d1).")
    parser.add_argument(
        "--env-type",
        default="mimicgen",
        help="Env type for lerobot. Use mimicgen for joint-space control.",
    )
    parser.add_argument(
        "--env-control-mode",
        default="JOINT_POSITION",
        help="Controller type for robosuite (e.g., JOINT_POSITION for joint targets).",
    )
    parser.add_argument("--n-episodes", type=int, default=10, help="Number of rollout episodes.")
    parser.add_argument("--batch-size", type=int, default=1, help="Eval batch size.")
    parser.add_argument("--device", default="cuda", help="cuda / cpu.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Eval 输出目录；默认 outputs/eval_cli/<日期>/<时间>。",
    )
    parser.add_argument(
        "--label-failures",
        action="store_true",
        help="收集完 rollout 后交互式标注失败类型并写入 CSV。",
    )
    parser.add_argument(
        "--labels-file",
        type=Path,
        default=None,
        help="失败标签 CSV 路径；默认 <output-dir>/failure_labels.csv。",
    )
    parser.add_argument(
        "--lerobot-repo",
        type=Path,
        default=Path("/workspace/liushuo/cupid/lerobot_2.1"),
        help="Path to lerobot source tree (to set PYTHONPATH).",
    )
    parser.add_argument(
        "--mimicgen-repo",
        type=Path,
        default=Path("/workspace/liushuo/cupid/mimicgen"),
        help="Path to mimicgen source tree (for custom robosuite envs).",
    )
    parser.add_argument(
        "--python-bin",
        type=Path,
        default=Path(sys.executable),
        help="Python executable used to launch lerobot-eval.",
    )
    parser.add_argument(
        "--extra",
        nargs=argparse.REMAINDER,
        help="Additional args passed verbatim to lerobot-eval after a '--'.",
    )
    return parser.parse_args()


def prompt_failure_labels(output_dir: Path, labels_file: Path) -> None:
    """
    简单交互：逐条轨迹选择失败标签，写入 CSV。
    """
    episodes = sorted(output_dir.glob("rollout_ep*.pkl"))
    if not episodes:
        print(f"未找到 rollout 数据，目录下无 rollout_ep*.pkl: {output_dir}")
        return

    videos_dir = output_dir / "videos"

    labels_file.parent.mkdir(parents=True, exist_ok=True)
    write_header = not labels_file.exists()
    with labels_file.open("a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["episode_index", "label", "note", "pkl_path", "video_path"])

        print("\n==== 失败标签选项 ====")
        for key, desc in FAILURE_TAGS.items():
            print(f"{key:8s}: {desc}")
        print("按回车默认 success。")

        for ep_path in episodes:
            ep_idx_str = ep_path.stem.replace("rollout_ep", "")
            try:
                ep_idx = int(ep_idx_str)
            except ValueError:
                ep_idx = -1

            video_path: Optional[Path] = None
            if videos_dir.is_dir():
                matches = sorted(videos_dir.glob(f"eval_episode_{ep_idx}_*.mp4"))
                if matches:
                    video_path = matches[0]

            print(f"\n轨迹 {ep_idx_str} | 视频: {video_path.name if video_path else '无'}")
            label = input("请输入标签: ").strip()
            if not label:
                label = "success"
            while label not in FAILURE_TAGS:
                print(f"无效标签 {label}，可选: {', '.join(FAILURE_TAGS.keys())}")
                label = input("请重新输入标签: ").strip() or "success"
            note = input("备注(可空): ").strip()

            writer.writerow([ep_idx_str, label, note, str(ep_path), str(video_path or "")])

    print(f"\n已写入标签: {labels_file}")


def main() -> None:
    args = parse_args()
    policy_path = args.policy_path.expanduser().resolve()
    if not policy_path.exists():
        raise FileNotFoundError(f"policy-path not found: {policy_path}")

    # 固定输出目录，便于后续标注。默认放在 outputs/eval_cli/<日期>/<时间>
    if args.output_dir is None:
        now = dt.datetime.now()
        auto_dir = Path("outputs") / "eval_cli" / f"{now:%Y-%m-%d}" / f"{now:%H-%M-%S}"
        output_dir = auto_dir
    else:
        output_dir = args.output_dir
    output_dir = output_dir.expanduser().resolve()

    lerobot_src = args.lerobot_repo.expanduser().resolve() / "src"
    mimicgen_root = args.mimicgen_repo.expanduser().resolve()

    env = os.environ.copy()
    env.setdefault("NUMBA_CACHE_DIR", str(Path.home() / ".cache" / "numba"))
    try:
        Path(env["NUMBA_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)
    except OSError:
        pass
    env.setdefault("NUMBA_DISABLE_CACHING", "1")
    env.setdefault("NUMBA_DISABLE_JIT", "1")
    
    if lerobot_src.is_dir():
        env["PYTHONPATH"] = f"{lerobot_src}:{env.get('PYTHONPATH','')}"
    if mimicgen_root.is_dir():
        env["PYTHONPATH"] = f"{mimicgen_root}:{env.get('PYTHONPATH','')}"
    
    # 使用指定的 mimicgen 环境类型（8 维或 9 维），mimicgen_env.py 中的 step 函数会转换为底层控制信号。
    cmd: List[str] = [
        str(args.python_bin),
        "-m",
        "lerobot.scripts.eval",
        f"--policy.path={policy_path}",
        f"--env.type={args.env_type}",
        f"--env.task={args.env_task}",
        f"--env.control_mode={args.env_control_mode}",
        f"--eval.n_episodes={args.n_episodes}",
        f"--eval.batch_size={args.batch_size}",
        f"--policy.device={args.device}",
        f"--output_dir={output_dir}",
        "--policy.use_amp=false",
        "--eval.max_episodes_rendered=1000",
        "--eval.save_episode_data=true",
    ]
    
    if args.extra:
        cmd.extend(args.extra)

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)

    if args.label_failures:
        labels_file = (args.labels_file or (output_dir / "failure_labels.csv")).expanduser().resolve()
        prompt_failure_labels(output_dir, labels_file)


if __name__ == "__main__":
    main()
