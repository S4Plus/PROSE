import numpy as np
import pytest

import openpi.shared.normalize as normalize
from openpi.training import checkpoints


def test_load_norm_stats_falls_back_to_single_available_asset(tmp_path):
    expected = {
        "state": normalize.NormStats(mean=np.array([1.0]), std=np.array([2.0])),
        "actions": normalize.NormStats(mean=np.array([3.0]), std=np.array([4.0])),
    }
    normalize.save(tmp_path / "can_mh_joint_keep90_last4_csvq", expected)

    loaded = checkpoints.load_norm_stats(tmp_path, "can_mh")

    np.testing.assert_allclose(loaded["state"].mean, expected["state"].mean)
    np.testing.assert_allclose(loaded["state"].std, expected["state"].std)
    np.testing.assert_allclose(loaded["actions"].mean, expected["actions"].mean)
    np.testing.assert_allclose(loaded["actions"].std, expected["actions"].std)


def test_load_norm_stats_raises_when_multiple_fallback_assets_exist(tmp_path):
    expected = {"state": normalize.NormStats(mean=np.array([1.0]), std=np.array([2.0]))}
    normalize.save(tmp_path / "asset_a", expected)
    normalize.save(tmp_path / "asset_b", expected)

    with pytest.raises(FileNotFoundError, match="Multiple fallback candidates exist"):
        checkpoints.load_norm_stats(tmp_path, "missing_asset")
