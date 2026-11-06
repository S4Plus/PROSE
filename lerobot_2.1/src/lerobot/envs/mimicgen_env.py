#!/usr/bin/env python

# Minimal MimicGen / robosuite single-arm environment wrapper to plug into LeRobot eval.
from __future__ import annotations

import copy
from dataclasses import asdict
from typing import Any, Dict

import gymnasium as gym
import numpy as np
import os
import sys
from pathlib import Path

# Prefer the MimicGen vendored robosuite over the pip-installed one
_ROBOSUITE_VENDORED = Path(__file__).resolve().parents[4] / "mimicgen" / "third_party" / "robosuite"
if _ROBOSUITE_VENDORED.is_dir():
    sys.path.insert(0, str(_ROBOSUITE_VENDORED))

import robosuite
from gymnasium import spaces
from robosuite.controllers import load_controller_config

from lerobot.envs.configs import MimicGenRobosuiteEnv


class MimicGenGymEnv(gym.Env):
    """
    Wrap a robosuite single-arm task (e.g., Square) with MimicGen-style observations into a Gym API.
    Observations:
      pixels: dict with agentview / wrist uint8 images (84x84x3)
      agent_pos: np.ndarray shape (9,) = absolute qpos7 + gripper qpos2
    Action:
      - 9-dim :absolute qpos7 + gripper qpos2
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, cfg: MimicGenRobosuiteEnv):
        super().__init__()
        self.cfg = copy.deepcopy(cfg)
        self._abs_pose_mode = False  # whether to treat incoming actions as absolute pose targets
        # Ensure numba cache is writable to avoid import errors in robosuite.
        os.environ.setdefault("NUMBA_CACHE_DIR", str(Path.home() / ".cache" / "numba"))
        self._make_env()
        # Expose max episode steps for vector env helpers
        self._max_episode_steps = self.cfg.episode_length

        cam_h = self.cfg.camera_size[1]
        cam_w = self.cfg.camera_size[0]
        self.observation_space = spaces.Dict(
            {
                "pixels": spaces.Dict(
                    {
                        "agentview": spaces.Box(low=0, high=255, shape=(cam_h, cam_w, 3), dtype=np.uint8),
                        "wrist": spaces.Box(low=0, high=255, shape=(cam_h, cam_w, 3), dtype=np.uint8),
                    }
                ),
                "agent_pos": spaces.Box(low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32),
            }
        )
        if "action" in self.cfg.features:
            act_shape = self.cfg.features["action"].shape
            self.action_space = spaces.Box(low=-np.inf, high=np.inf, shape=act_shape, dtype=np.float32)
        else:
            self.action_space = spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32)

    def _make_env(self) -> None:
        camera_names = list(self.cfg.camera_names)
        # Map custom convenience modes to underlying robosuite controllers
        #用绝对关节角度控制
        ctrl_mode = self.cfg.control_mode

        controller_config = load_controller_config(default_controller=ctrl_mode)
        # Force joint controllers to interpret inputs as absolute joint targets
        if str(ctrl_mode).upper().startswith("JOINT"):
            controller_config["control_delta"] = False

        task = (self.cfg.task or "").lower()
        # If available, use MimicGen custom variants (Square_D1, etc.)
        if task in {"square_d0", "square_d1", "square_d2", "coffee_d0", "coffee_d1", "coffee_d2", "coffee"}:
            import sys
            from pathlib import Path
            root = Path(__file__).resolve().parents[4] / "mimicgen"
            if root.exists() and str(root) not in sys.path:
                sys.path.append(str(root))
            import mimicgen
            
            #场景案例导入
            from mimicgen.envs.robosuite import nut_assembly as mg_na  # type: ignore
            from mimicgen.envs.robosuite import coffee as mg_coffee  # type: ignore

            cls_map = {
                "square_d0": mg_na.Square_D0,
                "square_d1": mg_na.Square_D1,
                "square_d2": mg_na.Square_D2,
                "coffee": mg_coffee.Coffee,
                "coffee_d0": mg_coffee.Coffee_D0,
                "coffee_d1": mg_coffee.Coffee_D1,
                "coffee_d2": mg_coffee.Coffee_D2,
            }
            if task not in cls_map:
                raise ValueError(f"Unsupported MimicGen task: {self.cfg.task}")
            env_cls = cls_map[task]#映射到对应的环境
            self.env = env_cls(
                robots=self.cfg.robots,
                controller_configs=controller_config,
                has_renderer=self.cfg.render,
                has_offscreen_renderer=True,
                control_freq=self.cfg.fps,
                horizon=self.cfg.episode_length,
                use_camera_obs=True,
                camera_names=camera_names,
                camera_heights=self.cfg.camera_size[1],
                camera_widths=self.cfg.camera_size[0],
                reward_shaping=True,
            )
        else:
            raise ValueError(f"Unsupported MimicGen task: {self.cfg.task}")
            # Fallback to vanilla robosuite tasks (e.g., NutAssemblySquare)
            # self.env = robosuite.make(
            #     self.cfg.task,
            #     robots=self.cfg.robots,
            #     controller_configs=controller_config,
            #     has_renderer=self.cfg.render,
            #     has_offscreen_renderer=True,
            #     control_freq=self.cfg.fps,
            #     horizon=self.cfg.episode_length,
            #     use_camera_obs=True,
            #     camera_names=camera_names,
            #     camera_heights=self.cfg.camera_size[1],
            #     camera_widths=self.cfg.camera_size[0],
            #     reward_shaping=True,
            # )

    def reset(self, *, seed: int | None = None, options: Dict[str, Any] | None = None):
        if seed is not None:
            # Best-effort seeding: robosuite envs don't always expose seed/reset(seed=...) APIs
            np.random.seed(seed)
        obs = self.env.reset()
        self._hide_collision_meshes()
        self._last_obs = obs  # cache for action conversion that needs current gripper qpos
        return self._convert_obs(obs), {}

    def step(self, action: np.ndarray):
        """
        Execute one step in the environment.
        绝对关节角度7+2
        """
        # If 9-D joint+gripper qpos is provided, map gripper qpos to a scalar command
        if action.shape[-1] == 9:
            joint_target = action[:7]
            target_gripper_qpos = action[7:9]
            current_qpos = None
            if hasattr(self, "_last_obs") and self._last_obs is not None:
                current_qpos = self._last_obs.get("robot0_gripper_qpos")
            if current_qpos is None:
                try:
                    current_qpos = np.array(self.env._observables["gripper_qpos"].obs)
                except Exception:
                    current_qpos = np.zeros_like(target_gripper_qpos)
            current_qpos = np.array(current_qpos)
            delta_qpos = target_gripper_qpos - current_qpos
            # Heuristic: if delta small, keep current grip; otherwise use sign for strong open/close
            delta_mean = delta_qpos.mean()
            if abs(delta_mean) < 1e-3:
                gripper_cmd = 0.0
            else:
                gripper_cmd = float(np.clip(delta_mean * 50.0, -1.0, 1.0))
            action = np.concatenate([joint_target, [gripper_cmd]])
        elif action.shape[-1] == 8:
            # assume 7 joints + scalar gripper cmd; pass-through
            pass
        obs, reward, done, info = self.env.step(action)
        self._last_obs = obs
        truncated = False  # robosuite uses done only
        # Ensure success flag for eval pipeline expectations
        info = info or {}
        # Prefer the environment's own success signal if available; otherwise fall back to any
        # provided flag. Some robosuite tasks (e.g., NutAssemblySquare) don't populate info["success"]
        # so we explicitly query `_check_success` when present.
        success = info.get("success", False)
        if hasattr(self.env, "_check_success"):
            try:
                success = bool(self.env._check_success())
            except Exception:
                # If the environment throws during success check, keep existing flag.
                success = bool(success)
        info["is_success"] = bool(success)
        return self._convert_obs(obs), reward, done, truncated, info

    def _hide_collision_meshes(self) -> None:
        """Make collision geoms transparent to avoid double-rendering."""
        try:
            model = self.env.sim.model
            if not hasattr(model, "geom_group") or not hasattr(model, "geom_rgba"):
                return
            collision = model.geom_group == 0
            if np.any(collision):
                model.geom_rgba[collision, 3] = 0.0
        except Exception:
            return

    def _convert_obs(self, obs: dict) -> dict:
        # Robosuite returns images as agentview_image etc.
        pix = {
            # Flip vertically to match the orientation expected by training data / renderer.
            "agentview": np.flipud(obs["agentview_image"]),
            "wrist": np.flipud(obs["robot0_eye_in_hand_image"]),
        }
        joint_pos = obs.get("robot0_joint_pos")
        if joint_pos is None:
            # Some configs expose cos/sin instead of raw joint pos
            sin_term = obs.get("robot0_joint_pos_sin")
            cos_term = obs.get("robot0_joint_pos_cos")
            if sin_term is not None and cos_term is not None:
                joint_pos = np.arctan2(sin_term, cos_term)
            else:
                # Fallback: read from robot state or zeros
                try:
                    joint_pos = np.array(self.env.robots[0]._joint_positions)
                except Exception:
                    joint_pos = np.zeros(7, dtype=np.float32)
        # use scalar gripper (mean) to align with 8-dim state
        # grip_scalar = float(np.mean(obs["robot0_gripper_qpos"]))
        grip_joint=obs["robot0_gripper_qpos"]

        agent_pos = np.concatenate([joint_pos,grip_joint], axis=-1).astype(np.float32)
        # print(agent_pos)
        return {"pixels": pix, "agent_pos": agent_pos}

    def render(self):
        """
        Return a visualization frame. If no on-screen viewer, render offscreen and
        horizontally stack agentview + wrist for clarity.
        """
        if self.env.viewer is None and self.env.sim is not None:
            w, h = self.cfg.camera_size
            imgs = []
            for cam in self.cfg.camera_names[:2]:
                img = self.env.sim.render(camera_name=cam, height=h, width=w)
                img = np.flipud(img)  # mujoco offscreen renders are upside-down
                imgs.append(img)
            # if len(imgs) == 2:
            #     return np.concatenate(imgs, axis=1)
            return imgs[0]
        return self.env.render()

    def close(self) -> None:
        self.env.close()

    @property
    def task(self) -> str:
        return self.cfg.task or ""

    @property
    def task_description(self) -> str:
        return self.task
