#!/usr/bin/env python3
"""
给 rollout_ep*.pkl 文件名追加成功/失败后缀，便于后续筛选。

用法示例：
python mymimicgen/rename_rollouts_with_result.py \
  --rollout-dir /workspace/liushuo/cupid/outputs/eval_cli/2026-01-05/18-24-10
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path
from typing import Iterable, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Add success/fail suffix to rollout_ep*.pkl files.")
    parser.add_argument("--rollout-dir", type=Path, required=True, help="目录，包含 rollout_ep*.pkl。")
    parser.add_argument("--success-suffix", type=str, default="success", help="成功轨迹文件名后缀。")
    parser.add_argument("--fail-suffix", type=str, default="fail", help="失败轨迹文件名后缀。")
    parser.add_argument("--dry-run", action="store_true", help="只打印计划重命名，不实际修改。")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="目标文件已存在时是否覆盖；默认跳过该文件。",
    )
    return parser.parse_args()


def iter_rollouts(rollout_dir: Path) -> Iterable[Path]:
    """支持 rollout_ep*.pkl 与 rollout_raw_ep*.pkl 两种命名。"""
    paths = list(rollout_dir.glob("rollout_ep*.pkl"))
    paths += list(rollout_dir.glob("rollout_raw_ep*.pkl"))
    return sorted(paths)


def infer_success_flag(data: dict) -> bool:
    """
    兼容 lerobot rollout 的标记方式：
    - 优先使用 next.success（每步的 bool 序列）
    - 其次使用 success（若存在）
    - 若存储在 {"traj": {"success": ...}} 中也会尝试读取
    """
    def _extract(obj, key: str) -> bool | None:
        if key in obj:
            try:
                return bool(obj[key][-1])
            except Exception:
                return None
        return None

    flag = _extract(data, "next.success")
    if flag is not None:
        return flag
    flag = _extract(data, "success")
    if flag is not None:
        return flag

    if "traj" in data and isinstance(data["traj"], dict):
        traj = data["traj"]
        flag = _extract(traj, "next.success")
        if flag is not None:
            return flag
        flag = _extract(traj, "success")
        if flag is not None:
            return flag

    return False


def build_target_name(src: Path, success: bool, success_suffix: str, fail_suffix: str) -> Path:
    suffix = success_suffix if success else fail_suffix
    return src.with_name(f"{src.stem}_{suffix}{src.suffix}")


def rename_one(
    src: Path,
    dst: Path,
    dry_run: bool = False,
    overwrite: bool = False,
) -> Tuple[bool, str]:
    if dst.exists() and not overwrite:
        return False, f"skip (exists): {dst.name}"
    if dry_run:
        return True, f"would rename {src.name} -> {dst.name}"
    try:
        dst.unlink(missing_ok=True)
        src.rename(dst)
        return True, f"renamed {src.name} -> {dst.name}"
    except Exception as exc:  # pragma: no cover - defensive
        return False, f"failed {src.name} -> {dst.name}: {exc}"


def main() -> None:
    args = parse_args()
    rollout_dir = args.rollout_dir.expanduser().resolve()
    if not rollout_dir.is_dir():
        sys.exit(f"rollout 目录不存在: {rollout_dir}")

    rollouts = list(iter_rollouts(rollout_dir))
    if not rollouts:
        sys.exit(f"未找到 rollout_ep*.pkl: {rollout_dir}")

    ok, failed = 0, 0
    for pkl_path in rollouts:
        try:
            with pkl_path.open("rb") as f:
                data = pickle.load(f)
        except Exception as exc:  # pragma: no cover - I/O 确保报错信息
            failed += 1
            print(f"[{pkl_path.name}] 读取失败: {exc}")
            continue

        success_flag = infer_success_flag(data)
        target = build_target_name(
            pkl_path,
            success_flag,
            args.success_suffix,
            args.fail_suffix,
        )
        changed, msg = rename_one(
            pkl_path,
            target,
            dry_run=args.dry_run,
            overwrite=args.overwrite,
        )
        print(msg)
        if changed:
            ok += 1
        else:
            failed += 1

    print(f"完成。成功处理 {ok}，失败/跳过 {failed}。")


if __name__ == "__main__":
    main()
