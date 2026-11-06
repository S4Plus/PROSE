
'''
      lerobot-eval \
        --policy.path=libero/checkpoints_20/act\
        --env.type=libero \
        --env.task=pick_up_the_bbq_sauce_and_place_it_in_the_basket_one \
        --env.bddl_root=LIBERO/bddl/object --env.image_size=256 \
        --env.state_dim=8 \
        --eval.n_episodes=5 \
        --eval.batch_size=1 \
        --eval.save_episode_data=true \
        --eval.max_episodes_rendered=5 \
        --env.image_size=256 \
        --env.state_dim=8 \
       --env.fixed_object_xy='{"basket_1_main":[0.0,0.26],"bbq_sauce_1_main":[0.05974137467540093, -0.0995890719413247]}' \
       --env.fixed_object_yaw='{"bbq_sauce_1_main":1.57}'



       #bbq_sauce_1_main', 'basket_1_main'
'''

from __future__ import annotations

import numpy as np

import gymnasium as gym
from gymnasium import spaces

from lerobot.envs.configs import LiberoEnvConfig


class LiberoGymEnv(gym.Env):
    """Thin Gym wrapper around LIBERO's robosuite environment.

    The wrapper adapts raw LIBERO observations into LeRobot-compatible keys:
    - images -> {"pixels": {"front": ..., "wrist": ...}}
    - state  -> {"agent_pos": state}
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, cfg: LiberoEnvConfig):
        super().__init__()
        try:
            import sys 
            sys.path.append("/workspace/liushuo/cupid/LIBERO")
            from libero.libero.envs.env_wrapper import OffScreenRenderEnv
        except ImportError as exc:  # pragma: no cover - import guard
            raise ModuleNotFoundError(
                "Missing LIBERO package. Ensure `LIBERO` is on PYTHONPATH and dependencies are installed."
            ) from exc

        self.cfg = cfg
        self._env = OffScreenRenderEnv(
            bddl_file_name=str(cfg.bddl_file),
            camera_names=list(cfg.camera_names),
            camera_heights=cfg.image_size,
            camera_widths=cfg.image_size,
            control_freq=cfg.fps,
            horizon=cfg.horizon,
            use_camera_obs=True,
            has_renderer=False,
            has_offscreen_renderer=True,
        )
        self._last_frame: np.ndarray | None = None
        self._fixed_object_xy = cfg.fixed_object_xy
        self._fixed_object_yaw = cfg.fixed_object_yaw

        self.task_description = getattr(self._env, "language_instruction", "")
        self.task = getattr(self._env, "problem_name", cfg.task or "")

        # libero / robosuite envs don't expose a gym action_space.
        # Build a Box space from action_spec if available, otherwise fall back to [-1, 1]^action_dim.
        action_space = getattr(self._env.env, "action_space", None)
        if action_space is None:
            spec = getattr(self._env.env, "action_spec", None)
            if callable(spec):
                low, high = spec()
            elif isinstance(spec, (tuple, list)) and len(spec) == 2:
                low, high = spec
            else:
                dim = getattr(self._env.env, "action_dim", None)
                if dim is None:
                    raise AttributeError(
                        "Underlying LIBERO env has no action_space/action_spec/action_dim"
                    )
                low = -np.ones(dim, dtype=np.float32)
                high = np.ones(dim, dtype=np.float32)
            action_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.action_space = action_space
        # Horizon info for Gym/Gymnasium compatibility
        self._max_episode_steps = getattr(self._env.env, "horizon", cfg.horizon)
        # Minimal spec stub with max_episode_steps to satisfy gym code paths
        self.spec = type("spec", (), {"max_episode_steps": self._max_episode_steps})
        self.observation_space = spaces.Dict(
            {
                "pixels": spaces.Dict(
                    {
                        "front": spaces.Box(low=0, high=255, shape=(cfg.image_size, cfg.image_size, 3), dtype=np.uint8),
                        "wrist": spaces.Box(low=0, high=255, shape=(cfg.image_size, cfg.image_size, 3), dtype=np.uint8),
                    }
                ),
                "agent_pos": spaces.Box(low=-np.inf, high=np.inf, shape=(cfg.state_dim,), dtype=np.float32),
            }
        )

    # def _resize_if_needed(self, img: np.ndarray) -> np.ndarray:
    #     h, w = img.shape[:2]
    #     if h == self.cfg.image_size and w == self.cfg.image_size:
    #         return img

    #     try:  # pragma: no cover - requires cv2 at runtime
    #         import cv2

    #         return cv2.resize(img, (self.cfg.image_size, self.cfg.image_size))
    #     except Exception:
    #         # Fallback to center crop/pad if OpenCV is unavailable
    #         crop_h = min(h, self.cfg.image_size)
    #         crop_w = min(w, self.cfg.image_size)
    #         start_h = (h - crop_h) // 2
    #         start_w = (w - crop_w) // 2
    #         cropped = img[start_h : start_h + crop_h, start_w : start_w + crop_w]
    #         pad_h = self.cfg.image_size - crop_h
    #         pad_w = self.cfg.image_size - crop_w
    #         return np.pad(
    #             cropped,
    #             ((pad_h // 2, pad_h - pad_h // 2), (pad_w // 2, pad_w - pad_w // 2), (0, 0)),
    #             mode="edge",
    #         )

    def _extract_image(self, obs: dict, keys: list[str]) -> np.ndarray:
        for key in keys:
            if key in obs:
                img = obs[key]
                if img.dtype != np.uint8:
                    img = np.clip(img, 0, 255).astype(np.uint8)
                # return self._resize_if_needed(img)
                return img
        raise KeyError(f"None of {keys} found in observation keys {list(obs.keys())}")

    def _extract_state(self, obs: dict) -> np.ndarray:
        for key in self.cfg.state_keys:
            if key in obs:
                state = np.asarray(obs[key], dtype=np.float32).reshape(-1)
                target_dim = getattr(self.cfg, "state_dim", None)
                if target_dim:
                    # Pad or truncate to the configured state_dim so vectorized env reset doesn't fail.
                    if state.shape[0] < target_dim:
                        pad = np.zeros(target_dim - state.shape[0], dtype=np.float32)
                        state = np.concatenate([state, pad], axis=0)
                    elif state.shape[0] > target_dim:
                        state = state[:target_dim]
                return state
        raise KeyError(f"None of {self.cfg.state_keys} found in state keys {list(obs.keys())}")

    def _format_obs(self, raw_obs: dict) -> dict:
        front = self._extract_image(raw_obs, [self.cfg.front_obs_key, "agentview_image", "frontview_image"])
        wrist = self._extract_image(raw_obs, [self.cfg.wrist_obs_key, "robot0_eye_in_hand_image", "wrist_image"])
        # Align camera orientation by flipping vertically (robosuite images come upside-down).
        front = np.flipud(front)
        wrist = np.flipud(wrist)
        state = self._extract_state(raw_obs)
        return {"pixels": {"front": front, "wrist": wrist}, "agent_pos": state}

    def _apply_fixed_object_pose(self):
        if not (self._fixed_object_xy or self._fixed_object_yaw):
            return
        sim = self._env.sim
        names = set(self._fixed_object_xy) | set(self._fixed_object_yaw)
        for name in names:
            xy = self._fixed_object_xy.get(name)
            yaw = self._fixed_object_yaw.get(name)
            try:
                body_id = sim.model.body_name2id(name)
            except KeyError:
                continue

            quat = None if yaw is None else np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)], dtype=np.float64)
            xy_arr = None if xy is None else np.asarray(xy, dtype=np.float64)

            # Prefer updating free joint qpos so pose stays consistent (including z).
            if sim.model.body_jntnum[body_id] > 0:
                jnt_id = sim.model.body_jntadr[body_id]
                jnt_type = sim.model.jnt_type[jnt_id]
                if jnt_type == 0 and sim.model.jnt_qposadr[jnt_id] + 7 <= sim.model.nq:  # free joint
                    qpos_adr = sim.model.jnt_qposadr[jnt_id]
                    qpos = sim.data.qpos[qpos_adr : qpos_adr + 7].copy()
                    if xy_arr is not None:
                        qpos[0:2] = xy_arr[:2]
                        if xy_arr.size >= 3:
                            qpos[2] = xy_arr[2]
                    if quat is not None:
                        qpos[3:] = quat
                    sim.data.qpos[qpos_adr : qpos_adr + 7] = qpos
                    print("成功修改坐标", name, qpos[:3])
                    if quat is not None:
                        print("成功修改quat", quat, name, qpos[3:])
                    continue

            # Fallback: edit body pos/quat directly.
            if xy_arr is not None:
                pos = sim.model.body_pos[body_id].copy()
                pos[:2] = xy_arr[:2]
                if xy_arr.size >= 3:
                    pos[2] = xy_arr[2]
                sim.model.body_pos[body_id] = pos
                print("成功修改坐标", name, sim.model.body_pos[body_id])
            if quat is not None:
                sim.model.body_quat[body_id] = quat
                print("成功修改quat", quat, name, sim.model.body_quat[body_id])

        sim.forward()

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self._env.seed(seed)
        raw_obs = self._env.reset()
        # Optionally override initial object positions/orientations before producing observations.
        self._apply_fixed_object_pose()
        raw_obs = self._env.regenerate_obs_from_state(self._env.get_sim_state())
        # Debug: print current positions after applying fixed pose
        # MuJoCo stores runtime poses in sim.data; model poses stay at defaults for free joints.
        for name in set(self._fixed_object_xy) | set(self._fixed_object_yaw):
            try:
                sim = self._env.sim
                bid = sim.model.body_name2id(name)
                pos = sim.data.body_xpos[bid].copy()
                quat = sim.data.body_xquat[bid].copy()
                print(f"[debug] {name} pose after reset: pos={pos}, quat={quat}")
            except Exception:
                continue
        obs = self._format_obs(raw_obs)
        self._last_frame = obs["pixels"]["front"]
        info = {"task_description": self.task_description, "task": self.task}
        return obs, info

    def step(self, action):
        try:
            raw_obs, reward, done, info = self._env.step(action)
        except ValueError as exc:
            # robosuite raises if step is called after termination
            if "terminated episode" in str(exc):
                raw_obs = self._env.reset()
                self._apply_fixed_object_pose()
                raw_obs = self._env.regenerate_obs_from_state(self._env.get_sim_state())
                obs = self._format_obs(raw_obs)
                info = {
                    "task_description": self.task_description,
                    "task": self.task,
                    "auto_reset": True,
                }
                return obs, 0.0, True, False, info
            raise
        obs = self._format_obs(raw_obs)
        self._last_frame = obs["pixels"]["front"]
        info = info or {}
        info.setdefault("task_description", self.task_description)
        info.setdefault("task", self.task)
        terminated = bool(done)
        truncated = False
        return obs, reward, terminated, truncated, info

    def render(self):
        # Prefer the latest front camera frame from the last step/reset.
        if self._last_frame is not None:
            return self._last_frame
        # Fallback to any visualization frames if available.
        raw_obs = getattr(self._env, "_visualizations", None)
        if raw_obs and isinstance(raw_obs, dict) and len(raw_obs) > 0:
            return list(raw_obs.values())[0]
        return None

    def close(self):
        self._env.close()
