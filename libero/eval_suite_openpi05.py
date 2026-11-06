#!/usr/bin/env python3
"""批量评估 LIBERO suite 中所有任务，底层通过 OpenPI05 websocket server。"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
CLIENT_SCRIPT = SCRIPT_DIR / "openpi05_client.py"
DEFAULT_LIBERO_ROOT = Path("/workspace/liushuo/mimiclabs/LIBERO")
DEFAULT_TASKS_JSONL = Path("/workspace/liushuo/cupid/datasets/local/libero_object_multitask_act/meta/tasks.jsonl")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="批量评估 LIBERO suite 的 OpenPI05 websocket client")
    parser.add_argument("--server-url", type=str, required=True, help="OpenPI05 websocket 地址，例如 ws://127.0.0.1:8000")
    parser.add_argument(
        "--suite",
        type=str,
        default="libero_object",
        choices=["libero_object", "libero_goal", "libero_spatial", "libero_10"],
        help="LIBERO suite 名称",
    )
    parser.add_argument("--episodes", type=int, default=10, help="每个 task 的 episode 数")
    parser.add_argument("--max-steps", type=int, default=400, help="每条 episode 的最大步数")
    parser.add_argument("--seed", type=int, default=7, help="起始随机种子")
    parser.add_argument("--image-size", type=int, default=224, help="发送给 OpenPI05 的图像尺寸")
    parser.add_argument("--render-resolution", type=int, default=256, help="LIBERO 渲染分辨率")
    parser.add_argument("--controller", type=str, default="JOINT_POSITION", help="环境控制器")
    parser.add_argument(
        "--joint-input-type",
        type=str,
        default="absolute",
        choices=["absolute", "delta"],
        help="policy 前 7 维 joint 动作语义",
    )
    parser.add_argument("--joint-delta-scale", type=float, default=0.05, help="JOINT_POSITION 的 delta 缩放上限")
    parser.add_argument("--control-freq", type=int, default=20, help="环境频率")
    parser.add_argument("--num-steps-wait", type=int, default=10, help="每条 episode 的稳定等待步数")
    parser.add_argument("--replan-steps", type=int, default=5, help="每次重规划后执行的动作步数")
    parser.add_argument("--timeout", type=float, default=30.0, help="预留的 websocket 超时参数")
    parser.add_argument("--render", type=int, default=0, help="每个 task 保存视频条数")
    parser.add_argument("--save-rollout-pkl", action="store_true", help="是否额外保存 rollout pkl，供 influence score 使用")
    parser.add_argument("--pkl-dir", type=Path, default=None, help="统一的 rollout pkl 输出目录")
    parser.add_argument(
        "--prompt-source",
        type=str,
        default="auto",
        choices=["auto", "language", "name", "tasks_jsonl"],
        help="prompt 来源",
    )
    parser.add_argument("--tasks-jsonl", type=Path, default=None, help="训练数据中的 meta/tasks.jsonl")
    parser.add_argument("--default-prompt", type=str, default=None, help="覆盖所有 task 的 prompt")
    parser.add_argument("--libero-root", type=Path, default=DEFAULT_LIBERO_ROOT, help="LIBERO 仓库根目录")
    parser.add_argument("--python-bin", type=Path, default=Path(sys.executable), help="用于启动 client 的 Python")
    parser.add_argument("--out-root", type=Path, required=True, help="汇总输出目录")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not CLIENT_SCRIPT.exists():
        raise FileNotFoundError(f"缺少 client 脚本: {CLIENT_SCRIPT}")

    args.out_root.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(args.libero_root))
    from libero.libero.benchmark import get_benchmark  # type: ignore

    benchmark = get_benchmark(args.suite.upper())(0)
    results: list[dict] = []

    tasks_jsonl = args.tasks_jsonl
    if tasks_jsonl is None and DEFAULT_TASKS_JSONL.exists():
        tasks_jsonl = DEFAULT_TASKS_JSONL

    for task_id in range(int(benchmark.n_tasks)):
        task = benchmark.get_task(task_id)
        task_out = args.out_root / f"{task_id:02d}_{task.name}"
        cmd = [
            str(args.python_bin),
            str(CLIENT_SCRIPT),
            f"--server-url={args.server_url}",
            f"--suite={args.suite}",
            f"--task-id={task_id}",
            f"--episodes={args.episodes}",
            f"--seed={args.seed + task_id * max(args.episodes, 1)}",
            f"--image-size={args.image_size}",
            f"--render-resolution={args.render_resolution}",
            f"--controller={args.controller}",
            f"--joint-input-type={args.joint_input_type}",
            f"--joint-delta-scale={args.joint_delta_scale}",
            f"--control-freq={args.control_freq}",
            f"--num-steps-wait={args.num_steps_wait}",
            f"--replan-steps={args.replan_steps}",
            f"--timeout={args.timeout}",
            f"--render={args.render}",
            f"--prompt-source={args.prompt_source}",
            f"--libero-root={args.libero_root}",
            f"--out-dir={task_out}",
        ]
        if args.save_rollout_pkl:
            cmd.append("--save-rollout-pkl")
        if args.max_steps is not None:
            cmd.append(f"--max-steps={args.max_steps}")
        if tasks_jsonl is not None:
            cmd.append(f"--tasks-jsonl={tasks_jsonl}")
        if args.default_prompt is not None:
            cmd.append(f"--default-prompt={args.default_prompt}")
        if args.pkl_dir is not None:
            cmd.append(f"--pkl-dir={args.pkl_dir}")

        print(f"[eval] task_id={task_id} task={task.name}")
        print(" ".join(cmd))
        subprocess.run(cmd, check=True)

        info_path = task_out / "eval_info.json"
        info = json.loads(info_path.read_text(encoding="utf-8"))
        results.append(
            {
                "task_id": task_id,
                "task_name": task.name,
                "successes": int(info["successes"]),
                "episodes": int(info["episodes"]),
                "success_rate": float(info["success_rate"]),
                "prompt": str(info["prompt"]),
                "output_dir": str(task_out),
            }
        )

    total_success = sum(item["successes"] for item in results)
    total_episodes = sum(item["episodes"] for item in results)
    macro_success_rate = sum(item["success_rate"] for item in results) / len(results) if results else 0.0
    micro_success_rate = total_success / total_episodes if total_episodes else 0.0

    summary = {
        "server_url": args.server_url,
        "suite": args.suite,
        "episodes_per_task": args.episodes,
        "num_tasks": len(results),
        "total_success": total_success,
        "total_episodes": total_episodes,
        "macro_success_rate": macro_success_rate,
        "micro_success_rate": micro_success_rate,
        "results": results,
    }

    (args.out_root / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    with (args.out_root / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["task_id", "task_name", "successes", "episodes", "success_rate", "prompt", "output_dir"],
        )
        writer.writeheader()
        writer.writerows(results)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
