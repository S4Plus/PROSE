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
_ROBOSUITE_VENDORED = Path("/workspace/liushuo/cupid/mimicgen/third_party/robosuite")
if _ROBOSUITE_VENDORED.is_dir():
    sys.path.insert(0, str(_ROBOSUITE_VENDORED))

# Ensure numba cache is configured before robosuite import to prevent cache errors.
_NUMBA_CACHE = Path.home() / ".cache" / "numba"
os.environ.setdefault("NUMBA_CACHE_DIR", str(_NUMBA_CACHE))
_NUMBA_CACHE.mkdir(parents=True, exist_ok=True)
# Disable numba JIT caching for robosuite to avoid import-time failures in read-only envs.
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

import robosuite
from gymnasium import spaces
from robosuite.controllers import load_controller_config

from lerobot.envs.configs import MimicGenRobosuiteEnv
from openpi_client.runtime import environment as _client_env


class MimicGenGymEnv(gym.Env, _client_env.Environment):
    """
    Wrap a robosuite single-arm task (e.g., Square) with MimicGen-style observations into a Gym API.
    Observations:
      pixels: dict with agentview / wrist uint8 images (84x84x3)
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, cfg: MimicGenRobosuiteEnv):
        super().__init__()
        self.cfg = copy.deepcopy(cfg)
        # Treat incoming actions as absolute joint targets (matches square_d1_core actions).
        self._abs_pose_mode = True
        # Optional channel swap for renderers that output BGR.
        self._bgr_to_rgb = bool(getattr(self.cfg, "bgr_to_rgb", False))
        # Some renderers are already upright; allow disabling vertical flip.
        self._flip_vertical = bool(getattr(self.cfg, "flip_vertical", True))
        # Stop the episode early once success is achieved.
        self._terminate_on_success = bool(getattr(self.cfg, "terminate_on_success", False))
        self._last_info: dict = {}
        # Ensure numba cache is writable to avoid import errors in robosuite.
        os.environ.setdefault("NUMBA_CACHE_DIR", str(Path.home() / ".cache" / "numba"))
        self._make_env()
        # Expose max episode steps for vector env helpers
        self._max_episode_steps = self.cfg.episode_length

        cam_h = self.cfg.camera_size[1]
        cam_w = self.cfg.camera_size[0]
        # Match square_d1_core dataset keys so outputs can be passed directly to policy.
        self.observation_space = spaces.Dict(
            {
                "observation.images.agentview": spaces.Box(low=0, high=255, shape=(cam_h, cam_w, 3), dtype=np.uint8),
                "observation.images.wrist": spaces.Box(low=0, high=255, shape=(cam_h, cam_w, 3), dtype=np.uint8),
                "observation.state": spaces.Box(low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32),
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
        ctrl_mode ="JOINT_POSITION"

        controller_config = load_controller_config(default_controller=ctrl_mode)
        # Force joint controllers to interpret inputs as absolute joint targets
        if str(ctrl_mode).upper().startswith("JOINT"):
            controller_config["control_delta"] = False

        task = (self.cfg.task or "").lower()
        common_kwargs = dict(
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

        # If available, use MimicGen custom variants (Square_D1, Coffee_D1, etc.).
        if task in {"square_d0", "square_d1", "square_d2", "coffee_d0", "coffee_d1", "coffee_d2", "coffee"}:
            import sys
            from pathlib import Path
            root = Path(__file__).resolve().parents[3] / "mimicgen"
            # Prefer local mimicgen repo over any installed package to match assets.
            if root.exists() and str(root) not in sys.path:
                sys.path.insert(0, str(root))
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
            self.env = env_cls(**common_kwargs)
        elif task in {"can", "pickplacecan", "pick_place_can"}:
            # Robomimic can task (PickPlaceCan) from vanilla robosuite.
            self.env = robosuite.make("PickPlaceCan", **common_kwargs)
        else:
            raise ValueError(f"Unsupported MimicGen task: {self.cfg.task}")

    def reset(self, *, seed: int | None = None, options: Dict[str, Any] | None = None):
        if seed is not None:
            # Best-effort seeding: robosuite envs don't always expose seed/reset(seed=...) APIs
            np.random.seed(seed)
        obs = self.env.reset()
        self._hide_collision_meshes()
        obs = self._refresh_observations(obs)
        self._last_obs = obs  # cache for action conversion that needs current gripper qpos
        self._done = False
        self._truncated = False
        self._last_info = {}
        return self._convert_obs(obs), {}

    # --- openpi_client runtime.Environment API ---
    def get_observation(self) -> dict:
        if self._last_obs is None:
            self.reset()
        return self._convert_obs(self._last_obs)

    def apply_action(self, action):
        if self.is_episode_complete():
            # Avoid stepping terminated episodes, which can raise in robosuite.
            return

        act_arr = self._extract_action(action).astype(np.float32, copy=False)
        obs, reward, done, info = self.env.step(act_arr)
        # Update cached observation and done flags so Runtime can stop the episode.
        self._post_step_update(obs, reward, done, info)

    def is_episode_complete(self) -> bool:
        return bool(getattr(self, "_done", False) or getattr(self, "_truncated", False))

    # def step(self, action: np.ndarray):
    #     """
    #     Execute one step in the environment.
    #     绝对关节角度7+1
    #     """
    #     obs, reward, done, info = self.env.step(action)
    #     return self._post_step_update(obs, reward, done, info)

    def _extract_action(self, action) -> np.ndarray:
        """Extract numpy action array from policy output."""
        if isinstance(action, dict):
            if "actions" in action:
                action = action["actions"]
            elif "action" in action:
                action = action["action"]
        action = np.asarray(action)
        if action.ndim > 1:
            action = action[0]
        return action

    def _post_step_update(self, obs, reward, done, info):
        """Shared bookkeeping after stepping the robosuite env."""
        info = info or {}
        truncated = bool(info.get("TimeLimit.truncated", False))
        self._last_obs = obs
        self._done = bool(done)
        self._truncated = bool(truncated)
        self._last_info = info

        # Ensure success flag for eval pipeline expectations
        success = info.get("success", False)
        if hasattr(self.env, "_check_success"):
            try:
                success = bool(self.env._check_success())
            except Exception:
                # If the environment throws during success check, keep existing flag.
                success = bool(success)
        info["is_success"] = bool(success)
        if self._terminate_on_success and success:
            self._done = True

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

    def _refresh_observations(self, obs: dict) -> dict:
        """Warm up offscreen rendering to avoid a corrupted first frame."""
        sim = getattr(self.env, "sim", None)
        if sim is None:
            return obs
        try:
            cam_names = list(getattr(self.cfg, "camera_names", ()))
            if cam_names:
                w, h = self.cfg.camera_size
                for cam in cam_names:
                    sim.render(camera_name=cam, height=h, width=w)
            get_obs = getattr(self.env, "_get_observations", None)
            if callable(get_obs):
                return get_obs(force_update=True)
        except Exception:
            return obs
        return obs

    # def _pad_action(self, action: np.ndarray) -> np.ndarray:
    #     """Pad / trim action to env action_dim (default 8) and snap gripper cmd."""
    #     action = np.asarray(action).reshape(-1)
    #     target = self.action_space.shape[0]
    #     if action.size != target:
    #         import logging

    #         logging.warning(f"Action size {action.size} != expected {target}, padding/trimming.")
    #     if action.size < target:
    #         action = np.pad(action, (0, target - action.size))
    #     elif action.size > target:
    #         action = action[:target]
    #     # Snap gripper command to {-1, 0, 1} for decisive open/close.
    #     grip = action[-1]
    #     if abs(grip) < 0.2:
    #         grip = 0.0
    #     else:
    #         grip = float(np.sign(grip))
    #     action = action.astype(np.float32)
    #     action[-1] = grip
    #     return action

    def _convert_obs(self, obs: dict) -> dict:
        # Robosuite returns images as agentview_image etc.
        # Flip vertically to match the orientation expected by training data / renderer.
        agentview = obs["agentview_image"]
        wrist = obs["robot0_eye_in_hand_image"]
        if self._flip_vertical:
            agentview = np.flipud(agentview)
            wrist = np.flipud(wrist)
        if self._bgr_to_rgb:
            agentview = agentview[..., ::-1]
            wrist = wrist[..., ::-1]
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
        grip_joint = obs["robot0_gripper_qpos"]
        agent_pos = np.concatenate([joint_pos, grip_joint], axis=-1).astype(np.float32)

        return {
            "observation.images.agentview": agentview,
            "observation.images.wrist": wrist,
            "observation.state": agent_pos,
            "state": agent_pos,
        }

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
                if self._flip_vertical:
                    img = np.flipud(img)  # mujoco offscreen renders are upside-down
                imgs.append(img)
            # if len(imgs) == 2:
            #     return np.concatenate(imgs, axis=1)
            return imgs[0]
        return self.env.render()

    def close(self) -> None:
        self.env.close()

    # --- Metrics helpers ---
    def get_episode_metrics(self) -> dict:
        """Return success flag for the last completed episode."""
        info = getattr(self, "_last_info", {}) or {}
        return {"is_success": bool(info.get("is_success", False))}

    @property
    def task(self) -> str:
        return self.cfg.task or ""

    @property
    def task_description(self) -> str:
        return self.task
