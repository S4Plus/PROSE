#!/usr/bin/env python3
"""
Libero 统一转换器运行脚本

支持自动识别RLDS和HDF5格式，提供完整的命令行接口。
"""

import argparse
import sys
import os
from pathlib import Path
import logging

def create_argument_parser():
    """创建命令行参数解析器，支持RLDS和HDF5格式"""
    parser = argparse.ArgumentParser(
        description="将Libero数据集（RLDS或HDF5格式）转换为LeRobot格式",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 自动检测格式并转换
  python run_converter.py \
    --data-dir /path/to/data \
    --repo-id username/dataset_name \
    --push-to-hub \
    --use-videos \
    --num-workers 4

  # HDF5格式，指定配置和任务名称
  python run_converter.py \
    --data-dir /path/to/hdf5/data \
    --repo-id username/hdf5_dataset \
    --config config.json \
    --task-name "pick_and_place" \
    --num-workers 8

  # RLDS格式（传统用法）
  python run_converter.py \
    --data-dir /path/to/rlds/data \
    --repo-id username/libero_dataset \
    --use-videos

支持的数据集: 
  RLDS: libero_10_no_noops, libero_goal_no_noops, 
        libero_object_no_noops, libero_spatial_no_noops
  HDF5: 任何包含trajectory.hdf5文件的目录结构
        """
    )
    
    # 必需参数
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="包含数据集的目录路径（自动检测RLDS或HDF5格式）"
    )
    parser.add_argument(
        "--repo-id", 
        type=str,
        required=True,
        help="输出数据集的仓库ID (格式: username/dataset_name)"
    )
    
    # 输出配置
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="本地输出目录 (默认: LEROBOT_HOME)"
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="转换后推送到Hugging Face Hub"
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="创建私有数据集"
    )
    
    # 数据格式
    parser.add_argument(
        "--use-videos",
        action="store_true",
        default=True,
        help="使用视频格式存储图像 (默认: True)"
    )
    parser.add_argument(
        "--robot-type",
        type=str,
        default="panda",
        help="机器人类型 (默认: panda)"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=20,
        help="帧率 (默认: 20)"
    )
    
    # HDF5特定参数
    parser.add_argument(
        "--config",
        type=str,
        help="HDF5格式的配置文件路径（可选）"
    )
    parser.add_argument(
        "--task-name",
        type=str,
        default="default_task",
        help="HDF5格式的任务名称 (默认: default_task)"
    )
    
    # 性能参数
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="并行处理的工作线程数 (默认: 2)"
    )
    parser.add_argument(
        "--image-writer-processes",
        type=int,
        default=2,
        help="图像写入进程数 (默认: 2)"
    )
    parser.add_argument(
        "--image-writer-threads",
        type=int,
        default=2,
        help="图像写入线程数 (默认: 2)"
    )
    
    # Hub配置
    parser.add_argument(
        "--license",
        type=str,
        default="apache-2.0",
        help="数据集许可证 (默认: apache-2.0)"
    )
    parser.add_argument(
        "--tags",
        nargs="+",
        default=["libero", "robotics", "lerobot"],
        help="数据集标签"
    )
    
    # 调试选项
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="详细日志输出"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="试运行模式（检查参数，不执行转换）"
    )
    parser.add_argument(
        "--force-format",
        type=str,
        choices=["rlds", "hdf5"],
        help="强制指定数据格式，跳过自动检测"
    )
    
    return parser


def main():
    """主函数"""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # 设置日志
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # 验证参数
    if not Path(args.data_dir).exists():
        logger.error(f"数据目录不存在: {args.data_dir}")
        return 1
    
    if "/" not in args.repo_id:
        logger.error(f"repo_id格式错误，应为 'username/dataset_name': {args.repo_id}")
        return 1
    
    logger.info("📋 转换配置:")
    logger.info(f"  数据源: {args.data_dir}")
    logger.info(f"  仓库ID: {args.repo_id}")
    logger.info(f"  输出目录: {args.output_dir or 'LEROBOT_HOME'}")
    logger.info(f"  并行线程数: {args.num_workers}")
    logger.info(f"  使用视频: {args.use_videos}")
    logger.info(f"  推送到Hub: {args.push_to_hub}")
    if args.config:
        logger.info(f"  配置文件: {args.config}")
    if args.task_name != "default_task":
        logger.info(f"  任务名称: {args.task_name}")
    
    if args.dry_run:
        logger.info("✅ 试运行完成，参数验证通过")
        return 0
    
    # 调用统一转换器
    try:
        from libero_rlds_converter import UnifiedConverter
        
        converter = UnifiedConverter(num_workers=args.num_workers)
        
        hub_config = {
            "tags": args.tags,
            "private": args.private,
            "license": args.license,
        }
        
        # 如果指定了强制格式，先临时修改检测器
        if args.force_format:
            logger.info(f"强制使用格式: {args.force_format}")
            original_detect = converter.detector.detect_format
            converter.detector.detect_format = lambda x: args.force_format
        
        dataset = converter.convert_dataset(
            data_dir=args.data_dir,
            repo_id=args.repo_id,
            output_dir=args.output_dir,
            push_to_hub=args.push_to_hub,
            use_videos=args.use_videos,
            robot_type=args.robot_type,
            fps=args.fps,
            task_name=args.task_name,
            config_path=args.config,
            hub_config=hub_config,
            image_writer_processes=args.image_writer_processes,
            image_writer_threads=args.image_writer_threads,
        )
        
        logger.info("✅ 转换完成!")
        logger.info(f"📊 数据集信息:")
        logger.info(f"  总episode数: {len(dataset)}")
        logger.info(f"  数据集路径: {dataset.root}")
        
        return 0
        
    except ImportError as e:
        logger.error(f"导入错误: {e}")
        logger.error("请确保已安装所需依赖:")
        logger.error("  pip install opencv-python h5py tqdm")
        logger.error("  pip install tensorflow tensorflow-datasets  # 用于RLDS支持")
        logger.error("  pip install lerobot  # 主要包")
        return 1
    except Exception as e:
        logger.error(f"转换失败: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())