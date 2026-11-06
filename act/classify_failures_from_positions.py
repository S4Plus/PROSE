#!/usr/bin/env python3
"""
按 nut/peg 的位置轨迹判别失败类型（3 类）：
  1) 抓取失败：nut z 几乎没抬起
  2) 抓取成功，搬运过程掉落：nut 抬起后在远离 peg 处出现明显下落
  3) 插入未对齐（卡顶或孔附近掉落）：nut 接近 peg 后失败

输入：包含 nut / peg 位置的 rollout_ep*.pkl 目录，或单个 pkl
默认键：raw_obs[*]["SquareNut_pos"] / raw_obs[*]["peg1_pos"]，可通过 --nut-key / --peg-key 指定。

输出：CSV（episode_index,label,note,pkl_path），末尾追加统计行。
可选中文标签（默认）：抓取失败 / 搬运失败 / 插入失败 / 成功 / 疑似成功 / 失败不确定
"""

from __future__ import annotations

import argparse
import glob
import pickle
from pathlib import Path
from typing import Tuple

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="基于 nut/peg 位置的失败分类（3 类）")
    p.add_argument("--rollout-dir", type=Path, help="包含 rollout_ep*.pkl 的目录；可与 --pkl 二选一")
    p.add_argument("--pkl", type=Path, help="单个 pkl 路径")
    p.add_argument("--outfile", type=Path, default=None, help="输出 CSV，默认 <rollout-dir>/failure_labels_pos.csv")
    p.add_argument("--nut-key", default="SquareNut_pos", help="nut 位置键名（raw_obs/observation）")
    p.add_argument("--peg-key", default="peg1_pos", help="peg 位置键名（raw_obs/observation）")
    # 阈值（单位：米）
    p.add_argument("--lift-thresh", type=float, default=0.02, help="nut 抬升量阈值，低于视为抓取失败")
    p.add_argument("--drop-delta", type=float, default=0.02, help="检测掉落的 z 突降阈值")
    p.add_argument(
        "--far-drop-dist",
        type=float,
        default=0.08,
        help="掉落时与 peg 的 XY 距离大于此值 → 搬运掉落（仅 XY）",
    )
    p.add_argument(
        "--near-peg-dist",
        type=float,
        default=0.05,
        help="与 peg 的 XY 距离小于此值视为进入插入阶段（仅 XY）",
    )
    p.add_argument(
        "--stuck-z-thresh",
        type=float,
        default=0.001,
        help="若最终 nut z 与初始差值超过此阈值，视为卡在 peg 上（优先判插入失败）",
    )
    p.add_argument(
        "--label-lang",
        choices=("zh", "en"),
        default="zh",
        help="标签语言：zh=中文，en=英文",
    )
    return p.parse_args()


def load_pickle(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def extract_traj(data, nut_key: str, peg_key: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    返回 (nut_pos[T,3], peg_pos[T,3])
    支持：
      - data["raw_obs"][t][nut_key]
      - data["observation"][nut_key]
      - data[nut_key]（顶层）
    """
    if "raw_obs" in data:
        raw = data["raw_obs"]
        nut = np.stack([r[nut_key] for r in raw])
        peg = np.stack([r[peg_key] for r in raw])
        return nut, peg

    if "observation" in data and isinstance(data["observation"], dict):
        obs = data["observation"]
        if nut_key in obs and peg_key in obs:
            return np.asarray(obs[nut_key]), np.asarray(obs[peg_key])

    if nut_key in data and peg_key in data:
        return np.asarray(data[nut_key]), np.asarray(data[peg_key])

    raise KeyError(f"未找到 nut_key={nut_key} 或 peg_key={peg_key}，请用 --nut-key/--peg-key 指定。")


def parse_ep_and_tag(path: Path) -> Tuple[int, str | None]:
    """
    支持 rollout_ep0001.pkl、rollout_ep0001_success.pkl、rollout_ep0001_failure.pkl。
    返回 (episode_index, tag)，tag ∈ {"success","failure",None}
    """
    stem = path.stem
    tag = None
    lowered = stem.lower()
    if lowered.endswith("_success"):
        tag = "success"
        stem = stem[: -len("_success")]
    elif lowered.endswith("_failure"):
        tag = "failure"
        stem = stem[: -len("_failure")]

    ep_idx = -1
    if "rollout_ep" in stem:
        tail = stem.split("rollout_ep", 1)[-1]
        try:
            ep_idx = int(tail)
        except ValueError:
            pass
    else:
        # fallback: 抓取最后的数字串
        digits = "".join(ch if ch.isdigit() else " " for ch in stem).split()
        if digits:
            try:
                ep_idx = int(digits[-1])
            except ValueError:
                ep_idx = -1
    return ep_idx, tag


def classify_episode(
    nut_pos: np.ndarray,
    peg_pos: np.ndarray,
    lift_thresh: float,
    drop_delta: float,
    far_drop_dist: float,
    near_peg_dist: float,
    stuck_z_thresh: float,
) -> Tuple[str, str]:
    """
    返回 (label, note)
    基于阶段的判定逻辑：
    1. Grasp: 是否抬起 (Lift)
    2. Transport: 抬起后是否进入 Peg 区域 (Insertion Phase)
    3. Drop Detection: 是否在过程中掉落。
       - 进入前掉落 -> drop_during_transport
       - 进入后失败/掉落 -> insertion_fail
    """
    z = nut_pos[:, 2]
    z_init = z[0]
    dist_xy = np.linalg.norm(nut_pos[:, :2] - peg_pos[:, :2], axis=1)
    
    # 1. 查找抬起时刻
    lift_idx = None
    for i in range(len(z)):
        if z[i] - z_init >= lift_thresh:
            lift_idx = i
            break
            
    if lift_idx is None:
        return "grasp_fail", f"max_lift={float(z.max() - z_init):.3f}"
        
    # 2. 查找首次进入 Peg 区域时刻 (Insertion Phase Start)
    insertion_idx = None
    for i in range(lift_idx, len(dist_xy)):
        if dist_xy[i] <= near_peg_dist:
            insertion_idx = i
            break
            
    # 3. 检查掉落或卡住
    # 寻找抬起后的最大高度及其位置，判定是否在中途发生了显著下降
    drop_idx = None
    curr_max_z = z[lift_idx]
    for i in range(lift_idx + 1, len(z)):
        if z[i] > curr_max_z:
            curr_max_z = z[i]
        elif curr_max_z - z[i] >= drop_delta:
            # 出现了显著下降
            drop_idx = i
            break
            
    # 判定最终状态
    final_z_diff = float(abs(z[-1] - z_init))
    final_dist = float(dist_xy[-1])
    
    # 如果最终状态是成功的（距离近且已经在低位），则直接判定为成功
    if final_dist <= near_peg_dist and final_z_diff < stuck_z_thresh:
        return "success_guess", f"final_dist={final_dist:.3f};z_diff={final_z_diff:.3f}"

    # 如果发生了掉落
    if drop_idx is not None:
        # 判定掉落发生的时机
        if insertion_idx is None or drop_idx < insertion_idx:
            # 还没到 Peg 就掉了
            return "drop_during_transport", f"drop_at_dist={float(dist_xy[drop_idx]):.3f};frame={drop_idx}"
        else:
            # 到了 Peg 附近掉了
            return "insertion_fail", f"drop_near_peg={float(dist_xy[drop_idx]):.3f};frame={drop_idx}"

    # 如果没检测到突发掉落，但最终没成功
    if insertion_idx is not None:
        # 进入了区域但最终没成功插入（可能是卡在上面，或者缓慢滑落到旁边）
        if final_z_diff > stuck_z_thresh:
            return "insertion_fail", f"stuck_on_top;final_z_diff={final_z_diff:.3f}"
        else:
            return "insertion_fail", f"missed_hole;final_dist={final_dist:.3f}"
            
    # 其他情况：抬起了，没掉落，但也没进 Peg 区域就结束了（可能是超时或路径规划问题）
    return "transport_fail", f"never_reached_peg;min_dist={float(dist_xy.min()):.3f}"


def map_label(label: str, lang: str) -> str:
    if lang == "en":
        return label
    mapping = {
        "grasp_fail": "抓取失败",
        "drop_during_transport": "搬运失败",
        "transport_fail": "搬运失败",
        "insertion_fail": "插入失败",
        "success": "成功",
        "success_guess": "疑似成功",
        "failure_uncertain": "失败不确定",
    }
    return mapping.get(label, label)


def failure_group(label: str) -> str | None:
    if label == "grasp_fail":
        return "grasp"
    if label in ("drop_during_transport", "transport_fail"):
        return "transport"
    if label == "insertion_fail":
        return "insertion"
    return None


def group_label(group: str, lang: str) -> str:
    if lang == "en":
        mapping = {"grasp": "grasp_fail", "transport": "transport_fail", "insertion": "insertion_fail"}
    else:
        mapping = {"grasp": "抓取失败", "transport": "搬运失败", "insertion": "插入失败"}
    return mapping.get(group, group)


def format_ratio(count: int, denom: int) -> str:
    if denom <= 0:
        return "0.0%"
    return f"{(count / denom) * 100.0:.1f}%"


def main() -> None:
    args = parse_args()
    if not args.rollout_dir and not args.pkl:
        raise SystemExit("需提供 --rollout-dir 或 --pkl")

    if args.rollout_dir:
        rollout_dir = args.rollout_dir.expanduser().resolve()
        paths = sorted(Path(rollout_dir).glob("rollout_*.pkl"))
        if not paths:
            raise SystemExit(f"目录下未找到 rollout_*.pkl: {rollout_dir}")
        outfile = args.outfile or (rollout_dir / "failure_labels_pos.csv")
    else:
        paths = [args.pkl.expanduser().resolve()]
        outfile = args.outfile or (paths[0].with_suffix(".csv"))

    rows = []
    raw_labels: list[str] = []
    for p in paths:
        ep_idx, file_tag = parse_ep_and_tag(p)
        ep_str = str(ep_idx if ep_idx >= 0 else p.stem)
        data = load_pickle(p)
        nut_pos, peg_pos = extract_traj(data, args.nut_key, args.peg_key)
        label, note = classify_episode(
            nut_pos,
            peg_pos,
            lift_thresh=args.lift_thresh,
            drop_delta=args.drop_delta,
            far_drop_dist=args.far_drop_dist,
            near_peg_dist=args.near_peg_dist,
            stuck_z_thresh=args.stuck_z_thresh,
        )

        # 若文件名携带 success/failure 后缀，优先采用文件标签：
        if file_tag == "success":
            label = "success"
            note = f"file_tag=success; {note}"
        elif file_tag == "failure":
            # 细分失败：若分类意外给出 success_guess，仍标记为 failure_uncertain
            if label == "success_guess":
                label = "failure_uncertain"
                note = f"file_tag=failure; classifier={note}"
            else:
                note = f"file_tag=failure; {note}"

        mapped_label = map_label(label, args.label_lang)
        raw_labels.append(label)
        rows.append((ep_idx, mapped_label, note, str(p)))
        print(f"ep {ep_str}: {mapped_label} | {note}")

    total = len(raw_labels)
    group_counts = {"grasp": 0, "transport": 0, "insertion": 0}
    for lbl in raw_labels:
        grp = failure_group(lbl)
        if grp:
            group_counts[grp] += 1
    failure_total = sum(group_counts.values())
    summary_label = "统计" if args.label_lang == "zh" else "summary"
    summary_note = f"total={total};failure_total={failure_total}"
    rows.append(("", summary_label, summary_note, ""))
    for grp in ("grasp", "transport", "insertion"):
        cnt = group_counts[grp]
        label_out = group_label(grp, args.label_lang)
        note = f"count={cnt};fail_ratio={format_ratio(cnt, failure_total)};all_ratio={format_ratio(cnt, total)}"
        rows.append(("", label_out, note, ""))

    outfile = outfile.expanduser().resolve()
    outfile.parent.mkdir(parents=True, exist_ok=True)
    with outfile.open("w") as f:
        f.write("episode_index,label,note,pkl_path\n")
        for r in rows:
            f.write(f"{r[0]},{r[1]},{r[2]},{r[3]}\n")
    print(f"\n已写入: {outfile}，共 {len(rows)} 条")


if __name__ == "__main__":
    main()
