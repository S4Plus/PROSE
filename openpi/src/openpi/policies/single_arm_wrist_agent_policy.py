import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def _parse_image(image) -> np.ndarray:
    """将图像统一转成 uint8 HWC 格式。"""
    img = np.asarray(image)
    if np.issubdtype(img.dtype, np.floating):
        img = (255 * img).astype(np.uint8)
    if img.ndim == 3 and img.shape[0] in (1, 3):
        img = einops.rearrange(img, "c h w -> h w c")
    return img


def _first_present(flat: dict, *keys: str):
    """兼容 `a.b.c` 与 `a/b/c` 两种展开键名。"""
    for key in keys:
        candidates = (key, key.replace(".", "/")) if "." in key else (key, key.replace("/", "."))
        for candidate in candidates:
            if candidate in flat:
                return candidate, flat[candidate]
    return None, None


@dataclasses.dataclass(frozen=True)
class SingleArmWristAgentRepack(transforms.DataTransformFn):
    """把单臂 wrist+agent 相机 LeRobot 样本重排为 OpenPI 下游统一结构。"""

    def __call__(self, data: dict) -> dict:
        flat = transforms.flatten_dict(data)
        _, wrist_image = _first_present(
            flat,
            "observation.images.wrist_camera",
            "observation.images.left_wrist",
            "images.wrist_camera",
            "images.left_wrist",
        )
        _, agent_image = _first_present(
            flat,
            "observation.images.agent_camera",
            "observation.images.agent_view",
            "images.agent_camera",
            "images.agent_view",
        )
        _, state = _first_present(flat, "observation.state", "state")
        _, actions = _first_present(flat, "action", "actions")
        _, prompt = _first_present(flat, "prompt")

        missing = []
        if wrist_image is None:
            missing.append("observation.images.wrist_camera")
        if agent_image is None:
            missing.append("observation.images.agent_camera")
        if state is None:
            missing.append("observation.state")
        if missing:
            raise KeyError(f"single_arm_wrist_agent 数据缺少字段: {', '.join(missing)}。")

        repacked = {
            "images": {
                "wrist_camera": wrist_image,
                "agent_camera": agent_image,
            },
            "state": state,
        }
        if actions is not None:
            repacked["actions"] = actions
        if prompt is not None:
            repacked["prompt"] = prompt
        return repacked


@dataclasses.dataclass(frozen=True)
class SingleArmWristAgentInputs(transforms.DataTransformFn):
    """单臂 wrist+agent 相机数据的模型输入变换。"""

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        wrist_image = _parse_image(data["images"]["wrist_camera"])
        agent_image = _parse_image(data["images"]["agent_camera"])
        zeros_image = np.zeros_like(wrist_image)

        match self.model_type:
            case _model.ModelType.PI0 | _model.ModelType.PI05:
                names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
                image_values = (agent_image, wrist_image, zeros_image)
                image_masks = (np.True_, np.True_, np.False_)
            case _model.ModelType.PI0_FAST:
                names = ("base_0_rgb", "base_1_rgb", "wrist_0_rgb")
                image_values = (agent_image, zeros_image, wrist_image)
                image_masks = (np.True_, np.False_, np.True_)
            case _:
                raise ValueError(f"Unsupported model type: {self.model_type}")

        inputs = {
            "state": np.asarray(data["state"]),
            "image": dict(zip(names, image_values, strict=True)),
            "image_mask": dict(zip(names, image_masks, strict=True)),
            "prompt": data.get("prompt", "pick up the blue ring and fit it onto the bearing."),
        }
        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"])
        return inputs


@dataclasses.dataclass(frozen=True)
class SingleArmWristAgentOutputs(transforms.DataTransformFn):
    """单臂 7 维关节 + 1 维 gripper command 的输出变换。"""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :8])}
