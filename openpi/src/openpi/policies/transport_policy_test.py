import numpy as np

from openpi.models import model as _model
from openpi.policies import transport_policy


def _make_image(fill_value: int) -> np.ndarray:
    return np.full((3, 84, 84), fill_value, dtype=np.uint8)


def test_transport_policy_supports_four_camera_layout():
    repack = transport_policy.TransportRepack()
    inputs = transport_policy.TransportInputs(model_type=_model.ModelType.PI05)

    sample = {
        "observation": {
            "images": {
                "shouldercamera0": _make_image(1),
                "shouldercamera1": _make_image(2),
                "robot0_eye_in_hand": _make_image(3),
                "robot1_eye_in_hand": _make_image(4),
            },
            "state": np.zeros(18, dtype=np.float32),
        },
        "action": np.zeros((20, 16), dtype=np.float32),
    }

    repacked = repack(sample)
    transformed = inputs(repacked)

    assert tuple(transformed["image"].keys()) == (
        "base_0_rgb",
        "base_1_rgb",
        "left_wrist_0_rgb",
        "right_wrist_0_rgb",
    )
    assert tuple(transformed["image_mask"].values()) == (
        np.True_,
        np.True_,
        np.True_,
        np.True_,
    )
    assert transformed["image"]["base_0_rgb"].shape == (84, 84, 3)
    assert transformed["image"]["base_1_rgb"].shape == (84, 84, 3)


def test_transport_policy_keeps_backward_compatibility_with_agentview():
    repack = transport_policy.TransportRepack()
    inputs = transport_policy.TransportInputs(model_type=_model.ModelType.PI05)

    sample = {
        "observation": {
            "images": {
                "agentview": _make_image(5),
                "robot0_eye_in_hand": _make_image(6),
            },
            "state": np.zeros(18, dtype=np.float32),
        },
    }

    repacked = repack(sample)
    transformed = inputs(repacked)

    assert tuple(transformed["image"].keys()) == (
        "base_0_rgb",
        "base_1_rgb",
        "left_wrist_0_rgb",
        "right_wrist_0_rgb",
    )
    assert tuple(transformed["image_mask"].values()) == (
        np.True_,
        np.False_,
        np.True_,
        np.False_,
    )
