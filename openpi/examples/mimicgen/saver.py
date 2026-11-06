import logging
import pathlib
import pickle

import imageio
import numpy as np
from openpi_client.runtime import subscriber as _subscriber
from typing_extensions import override


class VideoSaver(_subscriber.Subscriber):
    """Saves episode data."""

    def __init__(self, out_dir: pathlib.Path, subsample: int = 1) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        self._out_dir = out_dir
        self._images: list[np.ndarray] = []
        self._subsample = subsample

    @override
    def on_episode_start(self) -> None:
        self._images = []

    @override
    def on_step(self, observation: dict, action: dict) -> None:
        im = np.asarray(observation["observation.images.agentview"])
        if im.ndim == 3 and im.shape[0] in (1, 3):
            im = np.transpose(im, (1, 2, 0))  # CHW -> HWC
        if im.ndim != 3 or im.shape[2] not in (1, 3, 4):
            logging.warning(f"Skipping frame with unexpected shape {im.shape}")
            return
        self._images.append(im)

    @override
    def on_episode_end(self, metrics: dict | None = None) -> None:
        if not self._images:
            logging.warning("No frames captured; skipping video save.")
            return
        success = bool(metrics.get("is_success")) if metrics else False
        status_suffix = "success" if success else "failure"

        existing = list(self._out_dir.glob("out_[0-9]*.mp4"))
        indices = []
        for p in existing:
            parts = p.stem.split("_")
            if len(parts) >= 2:
                try:
                    indices.append(int(parts[1]))
                except ValueError:
                    continue
        next_idx = max(indices, default=-1) + 1
        out_path = self._out_dir / f"out_{next_idx}_{status_suffix}.mp4"

        logging.info(f"Saving video to {out_path}")
        imageio.mimwrite(
            out_path,
            [np.asarray(x) for x in self._images[:: self._subsample]],
            fps=50 // max(1, self._subsample),
        )


class ActionSaver(_subscriber.Subscriber):
    """Saves per-step actions for each episode."""

    def __init__(self, out_dir: pathlib.Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        self._out_dir = out_dir
        self._actions: list[np.ndarray] = []

    @override
    def on_episode_start(self) -> None:
        self._actions = []

    @override
    def on_step(self, observation: dict, action: dict) -> None:
        action_arr = self._to_array(action)
        if action_arr is not None:
            self._actions.append(action_arr)

    @override
    def on_episode_end(self, metrics: dict | None = None) -> None:
        if not self._actions:
            logging.warning("No actions captured; skipping action save.")
            return

        try:
            actions = np.stack(self._actions)
        except ValueError:
            shapes = {tuple(a.shape) for a in self._actions}
            logging.warning("Inconsistent action shapes %s; skipping action save.", shapes)
            return

        existing = list(self._out_dir.glob("actions_[0-9]*.txt"))
        next_idx = max([int(p.stem.split("_")[1]) for p in existing], default=-1) + 1
        out_path = self._out_dir / f"actions_{next_idx}.txt"

        logging.info(f"Saving actions to {out_path}")
        np.savetxt(out_path, actions, fmt="%.6f")

    def _to_array(self, action: dict | np.ndarray | list) -> np.ndarray | None:
        """Convert an action container into a flat numpy array."""
        if action is None:
            return None
        if isinstance(action, dict):
            if "actions" in action:
                action = action["actions"]
            elif "action" in action:
                action = action["action"]
        try:
            arr = np.asarray(action)
        except Exception:
            logging.warning("Could not convert action %s to ndarray", type(action))
            return None

        if arr.ndim > 1:
            arr = arr.reshape(-1)
        return arr


class RolloutSaver(_subscriber.Subscriber):
    """Saves raw rollout data to rollout_raw_ep*.pkl for failure labeling."""

    def __init__(self, out_dir: pathlib.Path, env: object) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        self._out_dir = out_dir
        self._env = env
        self._raw_obs: list[dict] = []
        self._actions: list[np.ndarray] = []

    @override
    def on_episode_start(self) -> None:
        self._raw_obs = []
        self._actions = []
        raw = self._get_raw_obs()
        if raw is not None:
            self._raw_obs.append(raw)

    @override
    def on_step(self, observation: dict, action: dict) -> None:
        action_arr = self._to_array(action)
        if action_arr is not None:
            self._actions.append(action_arr)
        raw = self._get_raw_obs()
        if raw is not None:
            self._raw_obs.append(raw)

    @override
    def on_episode_end(self, metrics: dict | None = None) -> None:
        if not self._raw_obs:
            logging.warning("No raw obs captured; skipping rollout save.")
            return

        success = bool(metrics.get("is_success")) if metrics else False
        status_suffix = "success" if success else "failure"

        existing = list(self._out_dir.glob("rollout_raw_ep*.pkl"))
        indices = []
        for p in existing:
            stem = p.stem
            if stem.startswith("rollout_raw_ep"):
                tail = stem.replace("rollout_raw_ep", "", 1).split("_", 1)[0]
                try:
                    indices.append(int(tail))
                except ValueError:
                    continue
        next_idx = max(indices, default=-1) + 1
        out_path = self._out_dir / f"rollout_raw_ep{next_idx:04d}_{status_suffix}.pkl"

        payload = {"raw_obs": self._raw_obs}
        if self._actions:
            try:
                payload["traj"] = {"action": np.stack(self._actions)}
            except ValueError:
                payload["traj"] = {"action": np.asarray(self._actions, dtype=object)}
        if metrics:
            payload["metrics"] = dict(metrics)

        logging.info("Saving rollout to %s", out_path)
        with out_path.open("wb") as f:
            pickle.dump(payload, f)

    def _get_raw_obs(self) -> dict | None:
        raw = getattr(self._env, "_last_obs", None)
        if raw is None or not isinstance(raw, dict):
            return None
        copied = {}
        for k, v in raw.items():
            if isinstance(v, np.ndarray):
                copied[k] = v.copy()
            else:
                copied[k] = v
        return copied

    def _to_array(self, action: dict | np.ndarray | list) -> np.ndarray | None:
        if action is None:
            return None
        if isinstance(action, dict):
            if "actions" in action:
                action = action["actions"]
            elif "action" in action:
                action = action["action"]
        try:
            arr = np.asarray(action)
        except Exception:
            logging.warning("Could not convert action %s to ndarray", type(action))
            return None
        if arr.ndim > 1:
            arr = arr.reshape(-1)
        return arr
