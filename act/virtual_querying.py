"""
Virtual Querying utilities for choosing bottle placement without a simulator.

This module provides a light-weight skeleton to:
1) generate candidate poses on a grid,
2) build proxy observations from a live RGB-D frame or low-dim state,
3) score candidates via ensemble variance or gradient norms,
4) smooth the score heatmap and sample with temperature-softmax.

Fill in ObservationBuilder callbacks with your robot's camera intrinsics and
table plane, and plug your policy ensemble into ScoreModel.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Optional, Sequence

import numpy as np

try:
    # Preferred for fast Gaussian smoothing if available.
    from scipy.ndimage import gaussian_filter
except Exception:  # pragma: no cover - optional dependency
    gaussian_filter = None


@dataclass
class CandidateGrid:
    """Stores candidate poses p_i = (x, y, theta) and optional validity mask."""

    poses_xytheta: np.ndarray  # shape (N, 3)
    valid_mask: Optional[np.ndarray] = None  # shape (N,)

    @classmethod
    def from_ranges(
        cls,
        xs: Sequence[float],
        ys: Sequence[float],
        thetas: Sequence[float],
        mask_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ) -> "CandidateGrid":
        """Generate a dense grid of poses and apply an optional boolean mask_fn."""
        grid = np.array(np.meshgrid(xs, ys, thetas, indexing="xy"))
        poses = np.stack([grid[0].ravel(), grid[1].ravel(), grid[2].ravel()], axis=-1)
        valid = mask_fn(poses) if mask_fn else None
        return cls(poses_xytheta=poses, valid_mask=valid)

    def filtered(self) -> np.ndarray:
        """Return poses after masking invalid entries (if provided)."""
        if self.valid_mask is None:
            return self.poses_xytheta
        return self.poses_xytheta[self.valid_mask.astype(bool)]


@dataclass
class ObservationBuilder:
    """
    Builds proxy observations without a simulator.

    Customize `pose_to_roi` to crop RGB-D patches using camera intrinsics +
    a plane assumption, or return low-dim states if you skip vision.
    """

    camera_intrinsics: Optional[np.ndarray] = None  # 3x3
    camera_extrinsics: Optional[np.ndarray] = None  # 4x4
    plane_height: float = 0.0  # table plane in world frame
    crop_size: int = 64

    def pose_to_roi(
        self, pose_xytheta: np.ndarray, rgb: np.ndarray, depth: Optional[np.ndarray]
    ) -> np.ndarray:
        """
        Map (x, y, theta) to an image-space crop.

        Replace this stub with your projection math:
        - project world (x, y, plane_height) to pixel using intrinsics/extrinsics
        - rotate crop by theta if needed
        - handle out-of-frame by returning None or a zero patch
        """
        raise NotImplementedError("Implement pose_to_roi for your camera setup.")

    def build_state(
        self,
        pose_xytheta: np.ndarray,
        env_features: Optional[np.ndarray] = None,
        rgb: Optional[np.ndarray] = None,
        depth: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Returns an observation tensor for scoring.

        - If using vision, call pose_to_roi to get a patch.
        - If using low-dim, concatenate (x, y, theta) with env_features.
        """
        if rgb is not None:
            patch = self.pose_to_roi(pose_xytheta, rgb, depth)
            return patch
        if env_features is None:
            env_features = np.zeros(0, dtype=np.float32)
        return np.concatenate([pose_xytheta.astype(np.float32), env_features], axis=-1)


class ScoreModel:
    """
    Wraps ensemble-variance and gradient-norm scoring.

    Provide callables that map observation -> trajectory or gradient embedding.
    """

    def __init__(
        self,
        ensemble_predict: Callable[[np.ndarray], Sequence[np.ndarray]],
        gradient_embed: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ) -> None:
        self.ensemble_predict = ensemble_predict
        self.gradient_embed = gradient_embed

    def ensemble_variance(self, obs: np.ndarray) -> float:
        """
        Compute variance of action trajectories across ensemble members.

        ensemble_predict should return a sequence of arrays shaped (T, A) or similar.
        """
        trajectories = self.ensemble_predict(obs)
        traj_stack = np.stack(trajectories, axis=0)
        mean_traj = traj_stack.mean(axis=0)
        variance = np.mean(np.linalg.norm(traj_stack - mean_traj, axis=-1) ** 2)
        return float(variance)

    def gradient_norm(self, obs: np.ndarray) -> float:
        """
        Compute ||g||^2 of pseudo-label gradient embedding.

        gradient_embed should return a 1D array; requires gradient head support.
        """
        if self.gradient_embed is None:
            raise ValueError("gradient_embed is not set.")
        g = self.gradient_embed(obs)
        return float(np.linalg.norm(g) ** 2)


@dataclass
class HeatmapSampler:
    """Applies Gaussian smoothing and temperature-softmax sampling on scores."""

    grid_shape: tuple[int, int, int]  # (len(xs), len(ys), len(thetas))
    sigma: float = 1.0
    temperature: float = 1.0

    def smooth(self, scores: np.ndarray) -> np.ndarray:
        """Gaussian smooth scores over the reshaped grid."""
        volume = scores.reshape(self.grid_shape)
        if gaussian_filter:
            smoothed = gaussian_filter(volume, sigma=self.sigma)
        else:
            # Fallback: simple mean filter over neighbors.
            smoothed = self._mean_filter(volume)
        return smoothed.reshape(-1)

    def softmax(self, scores: np.ndarray) -> np.ndarray:
        """Temperature-scaled softmax."""
        scaled = scores / max(self.temperature, 1e-6)
        scaled -= scaled.max()  # numerical stability
        exp_scores = np.exp(scaled)
        prob = exp_scores / np.clip(exp_scores.sum(), a_min=1e-9, a_max=None)
        return prob

    def sample(self, probs: np.ndarray, rng: Optional[np.random.Generator] = None) -> int:
        """Sample an index from the probability distribution."""
        rng = rng or np.random.default_rng()
        return int(rng.choice(len(probs), p=probs))

    @staticmethod
    def _mean_filter(volume: np.ndarray) -> np.ndarray:
        """Tiny 3D mean filter fallback if scipy is unavailable."""
        padded = np.pad(volume, 1, mode="edge")
        acc = np.zeros_like(volume)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    acc += padded[1 + dx : 1 + dx + volume.shape[0], 1 + dy : 1 + dy + volume.shape[1], 1 + dz : 1 + dz + volume.shape[2]]
        return acc / 27.0


def select_next_pose(
    grid: CandidateGrid,
    obs_builder: ObservationBuilder,
    scorer: ScoreModel,
    sampler: HeatmapSampler,
    rgb: Optional[np.ndarray],
    depth: Optional[np.ndarray],
    env_features: Optional[np.ndarray] = None,
    use_gradient: bool = False,
) -> tuple[np.ndarray, float, np.ndarray]:
    """
    End-to-end helper: build observations, score, smooth, sample, and return pose.

    Returns: (chosen_pose, chosen_score, prob_dist)
    """
    poses = grid.filtered()
    raw_scores = []
    for pose in poses:
        obs = obs_builder.build_state(pose, env_features=env_features, rgb=rgb, depth=depth)
        if use_gradient:
            score = scorer.gradient_norm(obs)
        else:
            score = scorer.ensemble_variance(obs)
        raw_scores.append(score)
    raw_scores = np.array(raw_scores, dtype=np.float32)
    smoothed = sampler.smooth(raw_scores)
    probs = sampler.softmax(smoothed)
    idx = sampler.sample(probs)
    return poses[idx], float(raw_scores[idx]), probs
