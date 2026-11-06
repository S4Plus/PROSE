"""Runtime patches for random_train.py subprocesses."""

from __future__ import annotations

from typing import Dict, List, Tuple

from prose.compat.hf_datasets import ensure_hf_list_feature


def _patch_subset_episode_indexing() -> None:
    """Allow non-contiguous episode subsets by remapping episode_index->position."""

    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except Exception:
        return

    if getattr(LeRobotDataset, "_cupid_subset_patch_applied", False):
        return

    def _patched_get_query_indices(
        self, idx: int, ep_idx: int
    ) -> Tuple[Dict[str, List[int | bool]], Dict[str, List[bool]]]:
        # Map original episode index to its position within the selected subset.
        if self.episodes is not None:
            if not hasattr(self, "_episode_pos_map"):
                self._episode_pos_map = {ep: i for i, ep in enumerate(self.episodes)}
            mapped_ep_idx = self._episode_pos_map.get(ep_idx)
            if mapped_ep_idx is None:
                raise IndexError(f"Episode index {ep_idx} not found in selected subset.")
        else:
            mapped_ep_idx = ep_idx

        ep_start = self.episode_data_index["from"][mapped_ep_idx]
        ep_end = self.episode_data_index["to"][mapped_ep_idx]
        query_indices = {
            key: [max(ep_start.item(), min(ep_end.item() - 1, idx + delta)) for delta in delta_idx]
            for key, delta_idx in self.delta_indices.items()
        }
        padding = {
            f"{key}_is_pad": [  # Pad values outside of current episode range
                (idx + delta < ep_start.item()) | (idx + delta >= ep_end.item()) for delta in delta_idx
            ]
            for key, delta_idx in self.delta_indices.items()
        }
        # Match original return type (BoolTensor) to keep downstream code unchanged.
        import torch

        padding = {k: torch.BoolTensor(v) for k, v in padding.items()}
        return query_indices, padding

    LeRobotDataset._get_query_indices = _patched_get_query_indices  # type: ignore[attr-defined]
    LeRobotDataset._cupid_subset_patch_applied = True  # type: ignore[attr-defined]


def apply() -> None:
    """Entry point imported by the training subprocess."""

    ensure_hf_list_feature()
    _patch_subset_episode_indexing()
