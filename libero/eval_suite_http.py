#!/usr/bin/env python3
"""批量评估 LIBERO suite 中所有任务，底层通过 HTTP 请求远端 ACT server。"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
CLIENT_SCRIPT = SCRIPT_DIR / "client.py"
DEFAULT_LIBERO_ROOT = Path("/workspace/liushuo/mimiclabs/LIBERO")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="批量评估 LIBERO suite 的 HTTP client")
    parser.add_argument("--server-url", type=str, required=True, help="ACT server 地址")
    parser.add_argument(
        "--suite",
        type=str,
        default="libero_object",
        choices=["libero_object", "libero_goal", "libero_spatial", "libero_10"],
        help="LIBERO suite 名称",
    )
    parser.add_argument("--episodes", type=int, default=10, help="每个 task 的 episode 数")
    parser.add_argument("--max-steps", type=int, default=400, help="每条 episode 的最大步数")
    parser.add_argument("--seed", type=int, default=0, help="起始随机种子")
    parser.add_argument("--image-size", type=int, default=128, help="图像分辨率")
    parser.add_argument("--control-freq", type=int, default=20, help="环境频率")
    parser.add_argument("--num-steps-wait", type=int, default=10, help="每条 episode 的稳定等待步数")
    parser.add_argument("--render", type=int, default=0, help="每个 task 保存视频条数")
    parser.add_argument("--controller", type=str, default="JOINT_POSITION", help="环境控制器")
    parser.add_argument(
        "--joint-input-type",
        type=str,
        default="absolute",
        choices=["absolute", "delta"],
        help="policy 前 7 维 joint 动作的语义",
    )
    parser.add_argument("--joint-delta-scale", type=float, default=0.05, help="JOINT_POSITION 的 delta 缩放上限")
    parser.add_argument("--gripper-force-open-steps", type=int, default=0, help="前 N 个执行步强制张开夹爪")
    parser.add_argument(
        "--task-conditioning",
        type=str,
        default="auto",
        choices=["auto", "on", "off"],
        help="是否启用 task-conditioned 输入",
    )
    parser.add_argument("--task-map-json", type=Path, default=None, help="task_index_map.json 路径")
    parser.add_argument("--timeout", type=float, default=30.0, help="HTTP 超时")
    parser.add_argument("--chunk-size", type=int, default=20, help="覆盖 server 默认 chunk 长度")
    parser.add_argument("--libero-root", type=Path, default=DEFAULT_LIBERO_ROOT, help="LIBERO 仓库根目录")
    parser.add_argument("--python-bin", type=Path, default=Path(sys.executable), help="用于启动 client 的 Python")
    parser.add_argument("--out-root", type=Path, required=True, help="汇总输出目录")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not CLIENT_SCRIPT.exists():
        raise FileNotFoundError(f"缺少 client 脚本: {CLIENT_SCRIPT}")

    args.out_root.mkdir(parents=True, exist_ok=True)

    # 通过 benchmark API 获取任务数和名称，避免手写列表。
    sys.path.insert(0, str(args.libero_root))
    from libero.libero.benchmark import get_benchmark  # type: ignore

    benchmark = get_benchmark(args.suite.upper())(0)
    results: list[dict] = []

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
            f"--max-steps={args.max_steps}",
            f"--seed={args.seed + task_id * max(args.episodes, 1)}",
            f"--image-size={args.image_size}",
            f"--control-freq={args.control_freq}",
            f"--num-steps-wait={args.num_steps_wait}",
            f"--render={args.render}",
            f"--controller={args.controller}",
            f"--joint-input-type={args.joint_input_type}",
            f"--joint-delta-scale={args.joint_delta_scale}",
            f"--gripper-force-open-steps={args.gripper_force_open_steps}",
            f"--task-conditioning={args.task_conditioning}",
            f"--timeout={args.timeout}",
            f"--libero-root={args.libero_root}",
            f"--out-dir={task_out}",
        ]
        if args.task_map_json is not None:
            cmd.append(f"--task-map-json={args.task_map_json}")
        if args.chunk_size is not None:
            cmd.append(f"--chunk-size={args.chunk_size}")

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
            fieldnames=["task_id", "task_name", "successes", "episodes", "success_rate", "output_dir"],
        )
        writer.writeheader()
        writer.writerows(results)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
