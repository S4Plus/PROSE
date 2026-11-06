import logging
import pathlib

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
        im = observation["images"]["cam_high"]  # [C, H, W]
        im = np.transpose(im, (1, 2, 0))  # [H, W, C]
        self._images.append(im)

    @override
    def on_episode_end(self, metrics: dict | None = None) -> None:
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
