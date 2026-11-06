#!/usr/bin/env python
"""
批量跑 lerobot-eval，按网格修改 bbq_sauce 位置。

- x 范围 [0.035, 0.065]，y 范围 [-0.115, -0.085]
- 默认 3x3 网格中心点
- 每个位置运行若干 episode，输出独立目录（含 eval_info、rollouts、视频）

用法示例：/workspace/liushuo/cupid/output/act/libero20_drop5/step_60000
    python libero/eval_test_n.py \
        --policy ./output/act/random25 \
        --task pick_up_the_bbq_sauce_and_place_it_in_the_basket_one \
        --bddl-root LIBERO/bddl/object\
        --episodes 5 \
        --render 1 \
        --out-root outputs/eval_grid_random25
    python libero/eval_test_n.py \
        --policy ./output/act/libero20_drop5 \
        --task pick_up_the_bbq_sauce_and_place_it_in_the_basket_one \
        --bddl-root LIBERO/bddl/object\
        --episodes 1 \
        --render 1 \
        --out-root outputs/eval_grid_libero20_drop5

    python libero/eval_test_n.py \
        --policy ./output/act/select25 \
        --task pick_up_the_bbq_sauce_and_place_it_in_the_basket_one \
        --bddl-root LIBERO/bddl/object\
        --episodes 1 \
        --render 1 \
        --out-root outputs/eval_grid_select25
    python libero/eval_test_n.py \
        --policy ./output/act/soup_select25 \
        --task pick_up_the_bbq_sauce_and_place_it_in_the_basket_one \
        --bddl-root LIBERO/bddl/object\
        --episodes 1 \
        --render 1 \
        --out-root outputs/eval_grid_soup_select25

    python libero/eval_test_n.py \
        --policy ./output/act/select5 \
        --task pick_up_the_bbq_sauce_and_place_it_in_the_basket_one \
        --bddl-root LIBERO/bddl/object\
        --episodes 1 \
        --render 1 \
        --out-root outputs/eval_grid_select5

    python libero/eval_test_n.py \
        --policy ./output/act/libero20_drop5 \
        --task pick_up_the_bbq_sauce_and_place_it_in_the_basket_one \
        --bddl-root LIBERO/bddl/object\
        --episodes 5 \
        --render 1 \
        --out-root outputs/eval_grid_libero20_drop5
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import List, Tuple


def generate_positions(x_min: float, x_max: float, y_min: float, y_max: float, n: int) -> List[Tuple[float, float]]:
    """返回网格中心点 + 边框点（含角点），3x3 时总计 21 个（9 中心 + 12 边框）。"""
    step_x = (x_max - x_min) / n
    step_y = (y_max - y_min) / n
    xs = [x_min + i * step_x for i in range(n + 1)]
    ys = [y_min + j * step_y for j in range(n + 1)]

    # 边框点
    pts: List[Tuple[float, float]] = []
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            if i in (0, n) or j in (0, n):  # 边框（包括四个角）
                pts.append((round(x, 6), round(y, 6)))

    # 中心点（n x n）
    xs_center = [x_min + (i + 0.5) * step_x for i in range(n)]
    ys_center = [y_min + (j + 0.5) * step_y for j in range(n)]
    pts += [(round(x, 6), round(y, 6)) for x in xs_center for y in ys_center]

    # 去重并排序
    pts = sorted(set(pts), key=lambda p: (p[0], p[1]))
    return pts


def run_eval(
    policy_path: Path,
    task: str,
    bddl_root: Path,
    basket_xy: Tuple[float, float],
    bbqsauce_xy: Tuple[float, float],
    bbqsauce_yaw: float,
    episodes: int,
    batch_size: int,
    render_eps: int,
    out_root: Path,
) -> Tuple[int, int]:
    x, y = bbqsauce_xy
    suffix = f"x{x:.3f}_y{y:.3f}"
    tmp_dir = out_root / f"tmp_{suffix}"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    out_root.mkdir(parents=True, exist_ok=True)
    '''
    lerobot-eval      \
    --policy.path=./output/act/select25  \
    --env.type=libero   \
    --env.task=pick_up_the_bbq_sauce_and_place_it_in_the_basket_one  \
    --env.bddl_root=LIBERO/bddl/object \
    --eval.n_episodes=5    \
    --eval.batch_size=1       \
    --eval.save_episode_data=true   \
    --eval.max_episodes_rendered=5 \
    --env.fixed_object_xy='{"basket_1_main":[0.0,0.26],"bbq_sauce_1_main":[0.035, -0.115]}' \
    --env.fixed_object_yaw='{"bbq_sauce_1_main":1.57}' 
    --env.state_dim=8
    '''
    fixed_xy = {"bbq_sauce_1_main": [x, y,0.034999999999999996], "basket_1_main": [0.0,0.26,0.034999999999999996]}
    fixed_yaw = {"bbq_sauce_1_main": bbqsauce_yaw,"basket_1_main":1.57}
    print(fixed_xy)
    cmd = [
        "lerobot-eval",
        f"--policy.path={policy_path}",
        "--env.type=libero",
        f"--env.task={task}",
        f"--env.bddl_root={bddl_root}",
        f"--env.fixed_object_xy={json.dumps(fixed_xy)}",
        f"--env.fixed_object_yaw={json.dumps(fixed_yaw)}",
        f"--eval.n_episodes={episodes}",
        f"--eval.batch_size={batch_size}",
        "--eval.save_episode_data=true",
        f"--eval.max_episodes_rendered={render_eps}",
        f"--output_dir={tmp_dir}",
        # "--env.state_dim=8" ,
    ]

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)

    successes = 0
    total = 0
    eval_info_path = tmp_dir / "eval_info.json"
    if eval_info_path.exists():
        with open(eval_info_path, "r") as f:
            info = json.load(f)
        per_episode = info.get("per_episode", [])
        successes = sum(1 for ep in per_episode if bool(ep.get("success", False)))
        total = len(per_episode)

    # 追加坐标后缀输出到 out_root
    eval_info = tmp_dir / "eval_info.json"
    if eval_info.exists():
        eval_info.rename(out_root / f"eval_info_{suffix}.json")

    for pkl in tmp_dir.glob("rollout_ep*.pkl"):
        pkl.rename(out_root / f"{pkl.stem}_{suffix}{pkl.suffix}")

    videos_dir = tmp_dir / "videos"
    if videos_dir.exists():
        for mp4 in videos_dir.glob("*.mp4"):
            mp4.rename(out_root / f"{mp4.stem}_{suffix}{mp4.suffix}")

    # 清理临时目录
    shutil.rmtree(tmp_dir, ignore_errors=True)
    return successes, total


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Batch eval with grid positions for bbq sauce.")
    p.add_argument("--policy", required=True, type=Path, help="Path to policy checkpoint dir")
    p.add_argument("--task", required=True, type=str, help="LIBERO task name")
    p.add_argument("--bddl-root", required=True, type=Path, help="Path to bddl root")
    p.add_argument("--episodes", type=int, default=5, help="Episodes per position")
    p.add_argument("--batch-size", type=int, default=1, help="Eval batch size")
    p.add_argument("--render", type=int, default=5, help="Max episodes rendered per position")
    p.add_argument("--out-root", type=Path, default=Path("outputs/eval_grid"), help="Root output directory")
    p.add_argument("--bbq-yaw", type=float, default=1.57, help="Fixed yaw for bbq sauce")
    p.add_argument("--basket-xy", type=float, nargs=2, default=(0.0, 0.26), help="Basket XY")
    p.add_argument("--x-range", type=float, nargs=2, default=(0.035, 0.065), help="bbq sauce x range")
    p.add_argument("--y-range", type=float, nargs=2, default=(-0.115, -0.085), help="bbq sauce y range")
    p.add_argument("--grid", type=int, default=3, help="Grid splits per axis (e.g., 3 for 3x3)")
    return p.parse_args()


def main():
    args = parse_args()
    positions = generate_positions(args.x_range[0], args.x_range[1], args.y_range[0], args.y_range[1], args.grid)
    # print(positions)
    total_success = 0
    total_episodes = 0
    for xy in positions:
        succ, tot = run_eval(
            policy_path=args.policy,
            task=args.task,
            bddl_root=args.bddl_root,
            basket_xy=tuple(args.basket_xy),
            bbqsauce_xy=xy,
            bbqsauce_yaw=args.bbq_yaw,
            episodes=args.episodes,
            batch_size=args.batch_size,
            render_eps=args.render,
            out_root=args.out_root,
        )
        total_success += succ
        total_episodes += tot

    if total_episodes > 0:
        success_rate = total_success / total_episodes
        print(f"Overall success: {total_success}/{total_episodes} ({success_rate*100:.1f}%)")
    else:
        print("No episodes recorded; success rate unavailable.")


if __name__ == "__main__":
    main()
