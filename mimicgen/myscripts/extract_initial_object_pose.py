#!/usr/bin/env python3
"""
从 MimicGen / robomimic HDF5 数据集中提取每个 episode 的初始物体状态，写入 txt。

默认行为：
- 读取每个 episode 的 obs["object"][0]（若存在）。
- 可选保存状态向量的首帧 states[0]（--include-states）。

用法示例：
python myscripts/extract_initial_object_pose.py \
    --input datasets/source/square.hdf5 \
    --output tmp/initial_objects.txt \
    --include-states
"""

import argparse
from pathlib import Path
import h5py
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="源 HDF5 数据集路径")
    parser.add_argument("--output", required=True, help="输出 txt 路径")
    parser.add_argument("--include-states", action="store_true", help="同时写入 states[0]")
    args = parser.parse_args()

    h5_path = Path(args.input).expanduser().resolve()
    out_path = Path(args.output).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(h5_path, "r") as f, open(out_path, "w", encoding="utf-8") as out_f:
        data_grp = f["data"]
        ep_keys = sorted(data_grp.keys())

        for ep_idx, ep_key in enumerate(ep_keys):
            grp = data_grp[ep_key]
            obs = grp["obs"]

            out_f.write(f"episode_index: {ep_idx} (key: {ep_key})\n")

            if "object" in obs:
                obj0 = np.asarray(obs["object"][0]).ravel()
                out_f.write("  object[0]: " + " ".join(map(str, obj0.tolist())) + "\n")
            else:
                out_f.write("  object[0]: <missing>\n")

            if args.include_states and "states" in grp:
                s0 = np.asarray(grp["states"][0]).ravel()
                out_f.write("  states[0]: " + " ".join(map(str, s0.tolist())) + "\n")

            out_f.write("\n")

    print(f"已写出到 {out_path}")


if __name__ == "__main__":
    main()
