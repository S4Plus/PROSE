#!/usr/bin/env python
"""
Evaluate a pretrained LeRobot ACT policy in Libero's simulation loop.

Example:


lerobot-eval      \
    --policy.path=./output/act/random  \
    --env.type=libero   \
    --env.task=pick_up_the_bbq_sauce_and_place_it_in_the_basket_one  \
    --env.bddl_root=LIBERO/bddl/object \
    --eval.n_episodes=5    \
    --eval.batch_size=1       \
    --eval.save_episode_data=true   \
    --eval.max_episodes_rendered=5 \
    --env.fixed_object_xy='{"basket_1_main":[0.0,0.26],"bbq_sauce_1_main":[0.05974137467540093, -0.0995890719413247]}' \
    --env.fixed_object_yaw='{"bbq_sauce_1_main":1.57}' \
    --env.state_dim=8





python -m lerobot.scripts.eval_libero \
    --policy_path /workspace/liushuo/cupid/outputs_new/random_40/retrain_random_0.8/checkpoints/002000/pretrained_model \
    --benchmark libero_object \
    --task_id 0 \
    --device cuda \
    --n_eval 20 \
    --obs_mapping '{"agentview_image":"observation.images.agentview","robot0_eye_in_hand_image":"observation.images.wrist"}'
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from libero.libero import get_libero_path
from libero.libero.benchmark import get_benchmark
from libero.libero.envs import DummyVectorEnv, OffScreenRenderEnv
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.utils.random_utils import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate LeRobot ACT policy on Libero tasks.")
    parser.add_argument("--policy_path", type=str, required=True, help="Path to ACT pretrained_model folder.")
    parser.add_argument(
        "--benchmark",
        type=str,
        required=True,
        choices=["libero_10", "libero_spatial", "libero_object", "libero_goal"],
    )
    parser.add_argument("--task_id", type=int, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--n_eval", type=int, default=20, help="Number of evaluation episodes.")
    parser.add_argument("--max_steps", type=int, default=400, help="Max env steps per episode.")
    parser.add_argument(
        "--obs_mapping",
        type=str,
        required=True,
        help="JSON string or path mapping Libero obs keys to ACT obs keys.",
    )
    parser.add_argument("--camera_height", type=int, default=128)
    parser.add_argument("--camera_width", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save_stats", type=str, default=None, help="Optional path to save success_rate json.")
    return parser.parse_args()


def load_mapping(mapping_arg: str) -> dict:
    """Allow json string or file path for obs_mapping."""
    if os.path.exists(mapping_arg):
        with open(mapping_arg) as f:
            return json.load(f)
    return json.loads(mapping_arg)


def obs_dict_to_act_batch(obs: dict, mapping: dict, device: torch.device) -> dict:
    """Convert Libero observation dict to the keys/shapes ACT expects."""
    batch = {}
    for src, dst in mapping.items():
        if src not in obs:
            raise KeyError(f"Observation key '{src}' missing from env obs.")
        arr = obs[src]
        if arr.ndim == 3:  # HWC image
            tensor = torch.from_numpy(arr).permute(2, 0, 1).float()  # CHW
            tensor = tensor.unsqueeze(0).unsqueeze(0)  # 1 x 1 x C x H x W
        elif arr.ndim == 1:  # state vector
            tensor = torch.from_numpy(arr).float().unsqueeze(0).unsqueeze(0)  # 1 x 1 x D
        else:
            raise ValueError(f"Unsupported obs shape for key {src}: {arr.shape}")
        batch[dst] = tensor.to(device)
    return batch


def evaluate(args: argparse.Namespace) -> float:
    device = torch.device(args.device)
    mapping = load_mapping(args.obs_mapping)

    policy = ACTPolicy.from_pretrained(args.policy_path).to(device)
    policy.eval()
    policy.reset()

    benchmark = get_benchmark(args.benchmark.upper())(0)
    task = benchmark.get_task(args.task_id)

    env_args = {
        "bddl_file_name": os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file),
        "camera_heights": args.camera_height,
        "camera_widths": args.camera_width,
    }

    env = DummyVectorEnv([lambda: OffScreenRenderEnv(**env_args)])
    init_states_path = os.path.join(get_libero_path("init_states"), task.problem_folder, task.init_states_file)
    init_states = torch.load(init_states_path)

    set_seed(args.seed)

    action_dim = policy.config.output_shapes.get("action", [7])[-1]
    num_success = 0

    for ep in range(args.n_eval):
        env.reset()
        init_state = init_states[ep % len(init_states)]
        obs = env.set_init_state(init_state)

        # simulate a few steps with zero actions to stabilize physics (match Libero evaluate)
        dummy = np.zeros((1, action_dim))
        for _ in range(5):
            obs, _, _, _ = env.step(dummy)

        done = False
        steps = 0
        policy.reset()

        while steps < args.max_steps and not done:
            steps += 1
            batch = obs_dict_to_act_batch(obs[0], mapping, device)
            with torch.no_grad():
                action = policy.select_action(batch).cpu().numpy()
            if action.ndim == 1:
                action = action[None, :]
            obs, _, done_flags, _ = env.step(action)
            done = bool(done_flags[0])

        num_success += int(done)

    success_rate = num_success / args.n_eval
    if args.save_stats:
        Path(args.save_stats).parent.mkdir(parents=True, exist_ok=True)
        with open(args.save_stats, "w") as f:
            json.dump({"success_rate": success_rate}, f, indent=2)
    return success_rate


def main():
    args = parse_args()
    sr = evaluate(args)
    print(f"Success rate: {sr * 100:.1f}% over {args.n_eval} episodes.")


if __name__ == "__main__":
    main()
