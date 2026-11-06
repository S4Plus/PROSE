#!/usr/bin/env python3
"""
自定义评估脚本：在运行策略时同时保存原始环境观测（含物体/末端位姿等），便于后续自动失败分类。

保存内容（每条 episode 一个 pkl）：
- actions, rewards, dones, successes
- observation_state（policy 输入的 9 维 state）
- raw: robot0_eef_pos/quat/vel, robot0_gripper_qpos/qvel, object 等原始 obs
- 可选存视频（同 lerobot eval）

示例：
python eval_with_raw_logging.py \
  --policy-path outputs/random_40act_core/retrain2_random_1.0/checkpoints/200000/pretrained_model \
  --env-task square_d1 \
  --n-episodes 10 \
  --output-dir outputs/eval_raw/$(date +%F/%H-%M-%S)
"""

from __future__ import annotations

import argparse
import copy
import os
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.envs.configs import MimicGenRobosuiteEnv
from lerobot.envs.factory import make_env
from lerobot.envs.utils import add_envs_task, preprocess_observation
from lerobot.policies.factory import make_policy
from lerobot.utils.io_utils import write_video
from lerobot.utils.utils import get_safe_torch_device


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--policy-path", type=Path, required=True, help="pretrained_model 目录")
    p.add_argument("--env-task", type=str, default="square_d1", help="任务名（mimicgen robosuite 变体）")
    p.add_argument("--env-control-mode", type=str, default="JOINT_POSITION", help="控制模式")
    p.add_argument("--n-episodes", type=int, default=10, help="rollout 次数")
    p.add_argument("--max-steps", type=int, default=200, help="每条 episode 最多步数")
    p.add_argument("--device", type=str, default="cuda", help="cuda / cpu")
    p.add_argument("--output-dir", type=Path, default=None, help="输出目录，默认 outputs/eval_raw/<date>/<time>")
    p.add_argument("--save-video", action="store_true", help="保存视频到 <output-dir>/videos")
    p.add_argument("--video-fps", type=int, default=20, help="写视频的 fps")
    p.add_argument("--lerobot-repo", type=Path, default=Path("/workspace/liushuo/cupid/lerobot_2.1"))
    p.add_argument("--mimicgen-repo", type=Path, default=Path("/workspace/liushuo/cupid/mimicgen"))
    return p.parse_args()


def collect_episode(
    env,
    policy,
    device: torch.device,
) -> Tuple[dict, List[dict], List[np.ndarray]]:
    """
    单条 episode rollout，返回：
    - traj: dict 包含 action/reward/done/success/obs_state
    - raw_list: 每步原始 obs 字典列表（来自 env.envs[0]._last_obs）
    - frames: 渲染图像列表（可为空）
    """
    policy.reset()
    obs, info = env.reset()
    raw_list: List[dict] = [copy.deepcopy(env.envs[0]._last_obs)]
    frames: List[np.ndarray] = []

    obs_states = []
    actions = []
    rewards = []
    dones = []
    successes = []

    done = np.array([False])

    step = 0
    while not done[0] and step < env.call("_max_episode_steps")[0]:
        obs_proc = preprocess_observation(obs)
        obs_for_policy = {
            k: v.to(device, non_blocking=device.type == "cuda") for k, v in obs_proc.items() if k.startswith("observation")
        }
        obs_for_policy = add_envs_task(env, obs_for_policy)

        with torch.inference_mode():
            action = policy.select_action(obs_for_policy)
        action_np = action.to("cpu").numpy()

        obs, reward, terminated, truncated, info = env.step(action_np)
        # 渲染单帧（仅在 save-video 时使用）
        try:
            frame = env.envs[0].render()
            if frame is not None:
                frames.append(frame)
        except Exception:
            pass

        # success from final_info if available
        success = False
        if "final_info" in info and info["final_info"][0] is not None:
            success = bool(info["final_info"][0].get("is_success", False))

        done = terminated | truncated | done

        obs_states.append(obs_proc["observation.state"][0].cpu().numpy())
        actions.append(action_np[0])
        rewards.append(float(reward[0]))
        successes.append(success)
        dones.append(bool(done[0]))

        raw_list.append(copy.deepcopy(env.envs[0]._last_obs))
        step += 1

    traj = {
        "observation.state": np.stack(obs_states, axis=0),
        "action": np.stack(actions, axis=0),
        "reward": np.array(rewards),
        "done": np.array(dones),
        "success": np.array(successes),
    }
    return traj, raw_list, frames


def main() -> None:
    args = parse_args()
    policy_path = args.policy_path.expanduser().resolve()
    if not policy_path.exists():
        raise FileNotFoundError(f"policy-path not found: {policy_path}")

    # 设置 PYTHONPATH
    env = os.environ
    lerobot_src = args.lerobot_repo.expanduser().resolve() / "src"
    mimicgen_root = args.mimicgen_repo.expanduser().resolve()
    if lerobot_src.is_dir():
        env["PYTHONPATH"] = f"{lerobot_src}:{env.get('PYTHONPATH','')}"
    if mimicgen_root.is_dir():
        env["PYTHONPATH"] = f"{mimicgen_root}:{env.get('PYTHONPATH','')}"

    # 输出目录
    if args.output_dir is None:
        from datetime import datetime

        now = datetime.now()
        output_dir = Path("outputs") / "eval_raw" / f"{now:%Y-%m-%d}" / f"{now:%H-%M-%S}"
    else:
        output_dir = args.output_dir
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # 构造 env config
    env_cfg = MimicGenRobosuiteEnv(
        task=args.env_task,
        control_mode=args.env_control_mode,
        render=False,
        episode_length=args.max_steps,
    )
    vec_env = make_env(env_cfg, n_envs=1, use_async_envs=False)

    # 加载 policy
    policy_cfg = PreTrainedConfig.from_pretrained(policy_path)
    policy_cfg.pretrained_path = policy_path
    policy = make_policy(cfg=policy_cfg, env_cfg=env_cfg)
    device = get_safe_torch_device(args.device, log=True)
    policy.to(device)
    policy.eval()

    summary = []
    video_dir = output_dir / "videos" if args.save_video else None
    if video_dir:
        video_dir.mkdir(parents=True, exist_ok=True)
    for ep in range(args.n_episodes):
        traj, raw_list, frames = collect_episode(vec_env, policy, device)
        success_flag = bool(traj["success"][-1]) if len(traj["success"]) else False
        ep_suffix = "success" if success_flag else "failure"
        ep_path = output_dir / f"rollout_raw_ep{ep:04d}_{ep_suffix}.pkl"
        with ep_path.open("wb") as f:
            pickle.dump({"traj": traj, "raw_obs": raw_list}, f)
        summary.append({"ep": ep, "success": success_flag, "steps": len(traj["action"])})
        if video_dir and frames:
            suffix = "success" if success_flag else "fail"
            vid_path = video_dir / f"ep{ep:04d}_{suffix}.mp4"
            try:
                write_video(str(vid_path), np.stack(frames), args.video_fps)
            except Exception as e:
                print(f"[{ep}] 写视频失败: {e}")
        print(f"[{ep}] saved {ep_path.name}, success={summary[-1]['success']}, steps={summary[-1]['steps']}")

    # 写 summary
    with (output_dir / "summary.txt").open("w") as f:
        for s in summary:
            f.write(f"{s['ep']},{s['success']},{s['steps']}\n")
    print(f"done. saved to {output_dir}")


if __name__ == "__main__":
    main()
