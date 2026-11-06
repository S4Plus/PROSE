"""Runtime patch that applies a frame-level keep mask to LeRobotDataset."""

from __future__ import annotations

import json
import os
from pathlib import Path

from hf_datasets_compat import ensure_hf_list_feature


ENV_KEY = "CUPID_FRAME_SELECTION_PATH"


def _load_selection() -> dict | None:
    path_str = os.environ.get(ENV_KEY)
    if not path_str:
        return None
    path = Path(path_str).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"{ENV_KEY} points to a missing file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _patch_lerobot_dataset() -> None:
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except Exception:
        return

    if getattr(LeRobotDataset, "_cupid_frame_mask_patch_applied", False):
        return

    selection = _load_selection()
    if selection is None:
        return

    selected_frame_indices = [int(v) for v in selection.get("selected_frame_indices", [])]
    selected_episode_ids = [int(v) for v in selection.get("selected_episode_ids", [])]
    dataset_root = Path(selection.get("dataset_root", "")).expanduser().resolve() if selection.get("dataset_root") else None

    orig_init = LeRobotDataset.__init__
    orig_len = LeRobotDataset.__len__
    orig_getitem = LeRobotDataset.__getitem__
    orig_num_frames = LeRobotDataset.num_frames
    orig_num_episodes = LeRobotDataset.num_episodes

    def _selection_applies(self) -> bool:
        if not hasattr(self, "_cupid_selected_frame_indices"):
            return False
        if dataset_root is None:
            return True
        return Path(self.root).expanduser().resolve() == dataset_root

    def patched_init(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        if dataset_root is not None and Path(self.root).expanduser().resolve() != dataset_root:
            return
        self._cupid_selected_frame_indices = selected_frame_indices
        self._cupid_selected_episode_ids = selected_episode_ids

    def patched_len(self):
        if _selection_applies(self):
            return len(self._cupid_selected_frame_indices)
        return orig_len(self)

    def patched_getitem(self, idx):
        if _selection_applies(self):
            real_idx = int(self._cupid_selected_frame_indices[int(idx)])
            return orig_getitem(self, real_idx)
        return orig_getitem(self, idx)

    def patched_num_frames(self):
        if _selection_applies(self):
            return len(self._cupid_selected_frame_indices)
        return orig_num_frames.fget(self)

    def patched_num_episodes(self):
        if _selection_applies(self):
            return len(self._cupid_selected_episode_ids)
        return orig_num_episodes.fget(self)

    LeRobotDataset.__init__ = patched_init  # type: ignore[assignment]
    LeRobotDataset.__len__ = patched_len  # type: ignore[assignment]
    LeRobotDataset.__getitem__ = patched_getitem  # type: ignore[assignment]
    LeRobotDataset.num_frames = property(patched_num_frames)  # type: ignore[assignment]
    LeRobotDataset.num_episodes = property(patched_num_episodes)  # type: ignore[assignment]
    LeRobotDataset._cupid_frame_mask_patch_applied = True  # type: ignore[attr-defined]


def apply() -> None:
    ensure_hf_list_feature()
    _patch_lerobot_dataset()
