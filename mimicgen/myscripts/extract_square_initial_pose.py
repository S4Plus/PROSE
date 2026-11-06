#!/usr/bin/env python3
"""
从 MimicGen 方形螺母任务数据集中提取每个 episode 的初始 peg / nut 位姿（pos + yaw）。

默认读取 datagen_info/object_poses 下的 square_peg、square_nut 的首帧（索引 0），
写入指定的 txt 文件；若未指定输出文件则打印到终端。

用法示例：
python myscripts/extract_square_initial_pose.py \
    --input /workspace/liushuo/cupid/mimicgen/tmp/core_datasets/square/demo_src_square_task_D1/demo.hdf5 \
    --output /workspace/liushuo/cupid/mimicgen/tmp/core_datasets/square/init_poses.txt
"""

import argparse
import math
from pathlib import Path

import h5py
import numpy as np


def mat_to_pos_yaw(mat4):
    mat4 = np.asarray(mat4)
    pos = mat4[:3, 3]
    yaw = math.atan2(mat4[1, 0], mat4[0, 0])
    return pos, yaw


def format_pos(pos):
    return " ".join(f"{v:.6f}" for v in pos.tolist())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="源 HDF5 数据集路径")
    parser.add_argument("--output", help="输出 txt 文件路径（不填则直接打印）")
    args = parser.parse_args()

    h5_path = Path(args.input).expanduser().resolve()
    out_path = Path(args.output).expanduser().resolve() if args.output else None
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(h5_path, "r") as f:
        data_grp = f["data"]
        ep_keys = sorted(data_grp.keys())

        lines = []
        for ep_idx, ep_key in enumerate(ep_keys):
            grp = data_grp[ep_key]
            obj_group = grp["datagen_info"]["object_poses"]

            peg = obj_group["square_peg"][0]
            nut = obj_group["square_nut"][0]

            peg_pos, peg_yaw = mat_to_pos_yaw(peg)
            nut_pos, nut_yaw = mat_to_pos_yaw(nut)

            lines.append(f"episode_index: {ep_idx} (key: {ep_key})")
            lines.append(
                f"  square_peg pos: {format_pos(peg_pos)} | yaw_rad: {peg_yaw:.6f} | yaw_deg: {math.degrees(peg_yaw):.3f}"
            )
            lines.append(
                f"  square_nut pos: {format_pos(nut_pos)} | yaw_rad: {nut_yaw:.6f} | yaw_deg: {math.degrees(nut_yaw):.3f}"
            )
            lines.append("")  # blank line between episodes

    if out_path:
        out_path.write_text("\n".join(lines), encoding="utf-8")
        print(f"已写出到 {out_path}")
    else:
        print("\n".join(lines))


if __name__ == "__main__":
    main()
