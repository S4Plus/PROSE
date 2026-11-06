#!/usr/bin/env python3
"""
合并两个 mimicgen 格式的 HDF5（/data/demo_xxx ...）为一个新文件，保持 demo_* 分组结构。

用法：
  python mymimicgen/merge_hdf5_datasets.py \
    --input-a /workspace/liushuo/cupid/mimicgen/tmp/core_datasets/square/demo_src_square_task_D1_70/demo.hdf5 \
    --input-b /workspace/liushuo/cupid/mimicgen/tmp/core_datasets/square/demo_src_square_task_D1_80/demo.hdf5 \
    --output merged.hdf5 
    [--overwrite]

行为：
  - 读取每个输入文件下 /data 下的 demo_* 分组，按顺序复制到输出。
  - 输出 demo 名从 demo_0 开始连续编号，保留每个 demo 原有的 attr 与 datasets。
  - 若 output 已存在，需加 --overwrite。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="合并两个 mimicgen HDF5 数据集到一个文件。")
    p.add_argument("--input-a", type=Path, required=True, help="第一个 HDF5 文件路径")
    p.add_argument("--input-b", type=Path, required=True, help="第二个 HDF5 文件路径")
    p.add_argument("--output", type=Path, required=True, help="输出 HDF5 路径")
    p.add_argument("--overwrite", action="store_true", help="允许覆盖已存在的输出文件")
    return p.parse_args()


def list_demos(h5: h5py.File) -> list[str]:
    if "data" not in h5:
        raise KeyError("未找到顶层分组 /data")
    return sorted(h5["data"].keys())


def main() -> None:
    args = parse_args()
    out_path = args.output.expanduser().resolve()
    if out_path.exists() and not args.overwrite:
        sys.exit(f"输出文件已存在: {out_path}（若需覆盖请加 --overwrite）")

    in_a = args.input_a.expanduser().resolve()
    in_b = args.input_b.expanduser().resolve()
    for p in [in_a, in_b]:
        if not p.is_file():
            sys.exit(f"输入文件不存在: {p}")

    # 创建输出文件
    if out_path.exists():
        out_path.unlink()
    out_h5 = h5py.File(out_path, "w")
    out_data = out_h5.create_group("data")

    counter = 0
    def copy_from(src_path: Path) -> int:
        nonlocal counter
        copied = 0
        with h5py.File(src_path, "r") as src:
            demos = list_demos(src)
            for name in demos:
                src_grp = src["data"][name]
                dst_name = f"demo_{counter}"
                src.copy(src_grp, out_data, name=dst_name)
                counter += 1
                copied += 1
        return copied

    n_a = copy_from(in_a)
    n_b = copy_from(in_b)
    out_h5.close()

    print(f"完成合并: {out_path}")
    print(f"  from {in_a}: {n_a} demos")
    print(f"  from {in_b}: {n_b} demos")
    print(f"  total demos: {counter}")


if __name__ == "__main__":
    main()
