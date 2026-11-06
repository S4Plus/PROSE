#!/usr/bin/env python3
"""
Quickly inspect the structure of an HDF5 file.

Usage:
  python hdf5_tree.py --path ../mimicgen/datasets/source/square.hdf5 --max-children 5 --depth 3
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import h5py


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print a compact tree view of an HDF5 file.")
    parser.add_argument("--path", required=True, type=Path, help="Path to the .hdf5 file.")
    parser.add_argument("--depth", type=int, default=None, help="Max depth to traverse (None for full depth).")
    parser.add_argument(
        "--max-children",
        type=int,
        default=30,
        help="Limit number of children shown per group to keep output short.",
    )
    parser.add_argument(
        "--show-attrs",
        action="store_true",
        help="Also print attributes for groups/datasets (can be verbose).",
    )
    parser.add_argument(
        "--show-mimicgen-summary",
        action="store_true",
        help="If present, print MimicGen-specific summary: actions/qpos/gripper shapes and first row.",
    )
    return parser.parse_args()


def format_dataset(name: str, ds: h5py.Dataset) -> str:
    shape = ds.shape
    dtype = ds.dtype
    return f"{name}  shape={shape} dtype={dtype}"


def format_attrs(obj: h5py.Group | h5py.Dataset) -> Iterable[str]:
    for key, val in obj.attrs.items():
        try:
            text = repr(val)
        except Exception:
            text = "<unprintable>"
        yield f"@{key}={text}"


def walk(
    obj: h5py.Group | h5py.Dataset,
    name: str,
    prefix: str,
    depth: int | None,
    max_children: int,
    show_attrs: bool,
    out_lines: list[str],
) -> None:
    if isinstance(obj, h5py.Dataset):
        out_lines.append(prefix + format_dataset(name, obj))
        if show_attrs and obj.attrs:
            for attr in format_attrs(obj):
                out_lines.append(prefix + "  " + attr)
        return

    # Group
    children = sorted(obj.keys())
    summary = f"{name}/  children={len(children)}"
    out_lines.append(prefix + summary)
    if show_attrs and obj.attrs:
        for attr in format_attrs(obj):
            out_lines.append(prefix + "  " + attr)

    if depth is not None and depth <= 0:
        return

    next_depth = None if depth is None else depth - 1
    for idx, child in enumerate(children):
        if idx >= max_children:
            out_lines.append(prefix + f"  ... ({len(children) - max_children} more)")
            break
        walk(obj[child], child, prefix + "  ", next_depth, max_children, show_attrs, out_lines)


def main() -> None:
    args = parse_args()
    fpath = args.path.expanduser().resolve()
    if not fpath.is_file():
        raise FileNotFoundError(f"File not found: {fpath}")

    with h5py.File(fpath, "r") as f:
        lines: list[str] = []
        walk(f, fpath.name, prefix="", depth=args.depth, max_children=args.max_children, show_attrs=args.show_attrs, out_lines=lines)

        if args.show_mimicgen_summary:
            # Best-effort MimicGen summary: report actions/qpos/gripper shapes and first row.
            try:
                ep0 = f["data/demo_0"]
                obs0 = ep0["obs"]
                actions = ep0["actions"]
                joint = obs0["robot0_joint_pos"]
                gripper = obs0["robot0_gripper_qpos"]
                robot0_eef_pos=obs0["robot0_eef_pos"]
                robot0_eef_quat=obs0["robot0_eef_quat"]
                summary = [
                    f"actions shape={actions.shape} dtype={actions.dtype}",
                    f"robot0_joint_pos shape={joint.shape} dtype={joint.dtype}",
                    f"robot0_gripper_qpos shape={gripper.shape} dtype={gripper.dtype}",
                    f"first action: {actions[0],actions[1]}",
                    f"first joint_pos: {joint[0],joint[1]}",
                    f"first gripper_qpos: {gripper[0],gripper[1]}",
                    f"robot0_eef_pos:{robot0_eef_pos[0],robot0_eef_pos[1]}",
                    f"robot0_eef_quat:{robot0_eef_quat[0],robot0_eef_quat[1]}",
                ]
                lines.append("\n[MimicGen summary]")
                lines.extend(summary)
            except Exception as exc:  # pragma: no cover - best-effort helper
                lines.append(f"\n[MimicGen summary] failed: {exc}")

    print("\n".join(lines))


if __name__ == "__main__":
    main()
