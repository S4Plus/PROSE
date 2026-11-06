"""
Example scaffold for coarse heatmap sampling without simulation.

This follows the workflow described in docs/virtual_querying.md:
1) build a coarse grid over the workspace,
2) construct low-dim observations from (x, y, theta) + robot state,
3) score via ensemble variance,
4) smooth and sample with temperature-softmax.

Fill in:
- your_ensemble_predict_fn: maps observation -> list of trajectories
- robot_state_vec: latest low-dim robot state/features
"""
from __future__ import annotations

import numpy as np

from act.virtual_querying import CandidateGrid, ObservationBuilder, ScoreModel, HeatmapSampler


def coarse_heatmap_select(
    xs: np.ndarray,
    ys: np.ndarray,
    thetas: list[float],
    robot_state_vec: np.ndarray,
    ensemble_predict_fn,
    sigma: float = 1.0,
    temperature: float = 2.0,
) -> tuple[np.ndarray, float, np.ndarray]:
    """
    Build a coarse heatmap over (x, y, theta) and sample a pose.

    Returns (chosen_pose, raw_score_at_pose, prob_dist).
    """
    grid = CandidateGrid.from_ranges(xs, ys, thetas)
    obs_builder = ObservationBuilder()  # low-dim only
    scorer = ScoreModel(ensemble_predict=ensemble_predict_fn)
    sampler = HeatmapSampler(grid_shape=(len(xs), len(ys), len(thetas)), sigma=sigma, temperature=temperature)

    raw_scores = []
    poses = grid.filtered()
    for pose in poses:
        obs = obs_builder.build_state(pose, env_features=robot_state_vec)
        score = scorer.ensemble_variance(obs)
        raw_scores.append(score)

    raw_scores = np.array(raw_scores, dtype=np.float32)
    smoothed = sampler.smooth(raw_scores)
    probs = sampler.softmax(smoothed)
    idx = sampler.sample(probs)
    return poses[idx], float(raw_scores[idx]), probs


if __name__ == "__main__":  # pragma: no cover - usage example
    # Example placeholders; replace with real functions/data.
    def your_ensemble_predict_fn(obs):
        # Dummy ensemble of 3 trajectories, each shape (T=2, A=2).
        return [
            np.array([[0.1, 0.2], [0.0, 0.1]]),
            np.array([[0.0, 0.1], [0.1, 0.0]]),
            np.array([[0.2, 0.0], [0.0, 0.2]]),
        ]

    robot_state_vec = np.array([0.0, 0.0], dtype=np.float32)
    xs = np.linspace(0.2, 0.6, 20)
    ys = np.linspace(-0.2, 0.2, 20)
    thetas = [0.0]

    pose, score, probs = coarse_heatmap_select(xs, ys, thetas, robot_state_vec, your_ensemble_predict_fn)
    print("Chosen pose:", pose, "score:", score)
