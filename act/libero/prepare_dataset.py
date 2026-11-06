#!/usr/bin/env python
"""
    将 LIBERO 原始数据转换为 LeRobot 数据集，供 ACT 训练使用。

python libero/prepare_dataset.py \
    --data-dir /workspace/liushuo/cupid/datasets/bddl/object_20 \
    --lerobot-home ./datasets \
    --repo-id local/lerobot_random20

"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
LEROBOT_DIR = REPO_ROOT / "lerobot_2.1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="将 LIBERO 数据集转换为 LeRobot 数据集")
    parser.add_argument("--data-dir", type=Path, required=True, help="原始 LIBERO 数据目录（HDF5 或 RLDS）")
    parser.add_argument("--repo-id", type=str, default="local/libero-act", help="目标数据集 repo_id，例如 local/libero-act")
    parser.add_argument(
        "--lerobot-home",
        type=Path,
        default=REPO_ROOT / "outputs" / "libero_lerobot",
        help="LEROBOT_HOME，转换结果会写到此目录下的 <repo-id>",
    )
    parser.add_argument("--fps", type=int, default=20, help="数据帧率，影响 delta_timestamps 计算")
    parser.add_argument("--num-workers", type=int, default=4, help="并行处理线程数")
    parser.add_argument("--image-writer-processes", type=int, default=5, help="写视频/图片的进程数")
    parser.add_argument("--image-writer-threads", type=int, default=10, help="写视频/图片的线程数")
    parser.add_argument("--task-name", type=str, default="libero_task", help="HDF5 数据默认任务名称")
    parser.add_argument("--robot-type", type=str, default="panda", help="机器人类型元数据")
    parser.add_argument("--push-to-hub", action="store_true", help="转换后推送到 Hugging Face Hub")
    parser.add_argument("--private", action="store_true", help="推送到 Hub 时创建私有数据集")
    parser.add_argument("--tags", nargs="+", default=["libero", "lerobot", "act"], help="Hub 数据集标签")
    parser.add_argument("--license", type=str, default="apache-2.0", help="Hub 数据集许可证")
    parser.add_argument("--no-videos", action="store_true", help="使用逐帧图像而非视频存储")
    parser.add_argument("--keep-existing", action="store_true", help="若目录已存在则不清理旧数据")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # 先设置 HF_LEROBOT_HOME，再加载 lerobot 相关模块（其常量读取该环境变量）
    os.environ["HF_LEROBOT_HOME"] = str(args.lerobot_home.resolve())
    args.lerobot_home.mkdir(parents=True, exist_ok=True)

    # 确保可以直接 import converter 和 lerobot 依赖
    sys.path.insert(0, str(LEROBOT_DIR / "libero2lerobot"))
    sys.path.insert(0, str(LEROBOT_DIR / "src"))

    from libero_rlds_converter import UnifiedConverter  # type: ignore  # noqa: E402

    if not args.data_dir.exists():
        raise FileNotFoundError(f"data_dir 不存在: {args.data_dir}")

    hub_config = {"tags": args.tags, "private": args.private, "license": args.license}

    converter = UnifiedConverter(num_workers=args.num_workers)
    dataset = converter.convert_dataset(
        data_dir=args.data_dir,
        repo_id=args.repo_id,
        output_dir=args.lerobot_home,
        push_to_hub=args.push_to_hub,
        use_videos=not args.no_videos,
        robot_type=args.robot_type,
        fps=args.fps,
        task_name=args.task_name,
        hub_config=hub_config,
        image_writer_processes=args.image_writer_processes,
        image_writer_threads=args.image_writer_threads,
        clean_existing=not args.keep_existing,
    )

    target_dir = args.lerobot_home / args.repo_id
    print(f"✅ 数据转换完成，保存于: {target_dir}")
    print(f"   - 统计信息: {target_dir / 'meta' / 'stats.json'}")
    print(f"   - episodes: {dataset.num_episodes}")


if __name__ == "__main__":
    main()
