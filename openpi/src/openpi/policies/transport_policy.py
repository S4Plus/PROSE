import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def _parse_image(image) -> np.ndarray:
    """将图像转换为 uint8 的 HWC 格式。"""
    img = np.asarray(image)
    if np.issubdtype(img.dtype, np.floating):
        img = (255 * img).astype(np.uint8)
    if img.ndim == 3 and img.shape[0] in (1, 3):
        img = einops.rearrange(img, "c h w -> h w c")
    return img


def _first_present(flat: dict, *keys: str):
    """返回第一个存在的键和值，用于兼容旧版和新版相机命名。"""
    for key in keys:
        candidates = (key, key.replace(".", "/")) if "." in key else (key, key.replace("/", "."))
        for candidate in candidates:
            if candidate in flat:
                return candidate, flat[candidate]
    return None, None


@dataclasses.dataclass(frozen=True)
class TransportRepack(transforms.DataTransformFn):
    """把 LeRobot transport_mh 样本整理成下游变换统一使用的结构。"""

    def __call__(self, data: dict) -> dict:
        flat = transforms.flatten_dict(data)
        _, base_0 = _first_present(flat, "observation.images.shouldercamera0", "observation.images.agentview")
        _, base_1 = _first_present(flat, "observation.images.shouldercamera1")
        _, left_wrist = _first_present(flat, "observation.images.robot0_eye_in_hand")
        _, right_wrist = _first_present(flat, "observation.images.robot1_eye_in_hand")
        _, state = _first_present(flat, "observation.state")

        if base_0 is None:
            raise KeyError(
                "transport_mh 数据缺少主视角图像。期望 observation.images.shouldercamera0 或 observation.images.agentview。"
            )
        if left_wrist is None:
            raise KeyError("transport_mh 数据缺少 observation.images.robot0_eye_in_hand。")
        if state is None:
            raise KeyError("transport_mh 数据缺少 observation.state。")

        repacked = {
            "images": {
                "base_0": base_0,
                "left_wrist": left_wrist,
            },
            "state": state,
        }
        if base_1 is not None:
            repacked["images"]["base_1"] = base_1
        if right_wrist is not None:
            repacked["images"]["right_wrist"] = right_wrist
        if "action" in flat:
            repacked["actions"] = flat["action"]
        if "prompt" in flat:
            repacked["prompt"] = flat["prompt"]
        return repacked


@dataclasses.dataclass(frozen=True)
class TransportInputs(transforms.DataTransformFn):
    """transport_mh 的模型输入变换。"""

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        images = data["images"] if "images" in data else {}
        base_0_key = "base_0" if "base_0" in images else "agentview"
        left_wrist_key = "left_wrist" if "left_wrist" in images else "wrist"
        base_0_image = _parse_image(images[base_0_key])
        left_wrist_image = _parse_image(images[left_wrist_key])
        has_base_1 = "base_1" in images
        has_right_wrist = "right_wrist" in images
        base_1_image = _parse_image(images["base_1"]) if has_base_1 else np.zeros_like(base_0_image)
        right_wrist_image = (
            _parse_image(images["right_wrist"]) if has_right_wrist else np.zeros_like(base_0_image)
        )
        zeros_image = np.zeros_like(base_0_image)

        match self.model_type:
            case _model.ModelType.PI0 | _model.ModelType.PI05:
                names = ("base_0_rgb", "base_1_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
                image_values = (base_0_image, base_1_image, left_wrist_image, right_wrist_image)
                image_masks = (np.True_, np.bool_(has_base_1), np.True_, np.bool_(has_right_wrist))
            case _model.ModelType.PI0_FAST:
                names = ("base_0_rgb", "base_1_rgb", "wrist_0_rgb")
                image_values = (base_0_image, base_1_image if has_base_1 else zeros_image, left_wrist_image)
                image_masks = (np.True_, np.bool_(has_base_1), np.True_)
            case _:
                raise ValueError(f"Unsupported model type: {self.model_type}")

        inputs = {
            "state": np.asarray(data["state"]),
            "image": dict(zip(names, image_values, strict=True)),
            "image_mask": dict(zip(names, image_masks, strict=True)),
            "prompt": data.get("prompt", "transport"),
        }
        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"])
        return inputs


@dataclasses.dataclass(frozen=True)
class TransportOutputs(transforms.DataTransformFn):
    """transport_mh 的模型输出变换。"""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :16])}
