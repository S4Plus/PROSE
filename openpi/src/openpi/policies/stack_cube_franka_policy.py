import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def _parse_image(image) -> np.ndarray:
    """将图像统一转成 uint8 的 HWC 格式。"""
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
class StackCubeFrankaRepack(transforms.DataTransformFn):
    """把单臂 Franka stack_cube 的 LeRobot 样本重排为 OpenPI 下游统一结构。"""

    def __call__(self, data: dict) -> dict:
        flat = transforms.flatten_dict(data)
        _, fixed_image = _first_present(flat, "observation.images.fixed_camera")
        _, wrist_image = _first_present(flat, "observation.images.wrist_camera")
        _, state = _first_present(flat, "observation.state")
        _, actions = _first_present(flat, "action", "actions")
        _, prompt = _first_present(flat, "prompt")

        if fixed_image is None:
            raise KeyError("stack_cube_franka 数据缺少 observation.images.fixed_camera。")
        if wrist_image is None:
            raise KeyError("stack_cube_franka 数据缺少 observation.images.wrist_camera。")
        if state is None:
            raise KeyError("stack_cube_franka 数据缺少 observation.state。")

        repacked = {
            "images": {
                "fixed_camera": fixed_image,
                "wrist_camera": wrist_image,
            },
            "state": state,
        }
        if actions is not None:
            repacked["actions"] = actions
        if prompt is not None:
            repacked["prompt"] = prompt
        return repacked


@dataclasses.dataclass(frozen=True)
class StackCubeFrankaInputs(transforms.DataTransformFn):
    """stack_cube 单臂 Franka 数据的模型输入变换。"""

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        images = data["images"]
        base_image = _parse_image(images["fixed_camera"])
        wrist_image = _parse_image(images["wrist_camera"])
        zeros_image = np.zeros_like(base_image)

        match self.model_type:
            case _model.ModelType.PI0 | _model.ModelType.PI05:
                names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
                image_values = (base_image, wrist_image, zeros_image)
                image_masks = (np.True_, np.True_, np.False_)
            case _model.ModelType.PI0_FAST:
                names = ("base_0_rgb", "base_1_rgb", "wrist_0_rgb")
                image_values = (base_image, zeros_image, wrist_image)
                image_masks = (np.True_, np.False_, np.True_)
            case _:
                raise ValueError(f"Unsupported model type: {self.model_type}")

        inputs = {
            "state": np.asarray(data["state"]),
            "image": dict(zip(names, image_values, strict=True)),
            "image_mask": dict(zip(names, image_masks, strict=True)),
            "prompt": data.get("prompt", "stack cube"),
        }
        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"])
        return inputs


@dataclasses.dataclass(frozen=True)
class SingleWristFrankaRepack(transforms.DataTransformFn):
    """把单 wrist 相机 Franka LeRobot 样本重排为 OpenPI 下游统一结构。"""

    def __call__(self, data: dict) -> dict:
        flat = transforms.flatten_dict(data)
        _, wrist_image = _first_present(flat, "observation.images.wrist_camera")
        _, state = _first_present(flat, "observation.state")
        _, actions = _first_present(flat, "action", "actions")
        _, prompt = _first_present(flat, "prompt")

        if wrist_image is None:
            raise KeyError("single_wrist_franka 数据缺少 observation.images.wrist_camera。")
        if state is None:
            raise KeyError("single_wrist_franka 数据缺少 observation.state。")

        repacked = {
            "images": {
                "wrist_camera": wrist_image,
            },
            "state": state,
        }
        if actions is not None:
            repacked["actions"] = actions
        if prompt is not None:
            repacked["prompt"] = prompt
        return repacked


@dataclasses.dataclass(frozen=True)
class SingleWristFrankaInputs(transforms.DataTransformFn):
    """单 wrist 相机 Franka 数据的模型输入变换。"""

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        wrist_image = _parse_image(data["images"]["wrist_camera"])
        zeros_image = np.zeros_like(wrist_image)

        match self.model_type:
            case _model.ModelType.PI0 | _model.ModelType.PI05:
                names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
                image_values = (zeros_image, wrist_image, zeros_image)
                image_masks = (np.False_, np.True_, np.False_)
            case _model.ModelType.PI0_FAST:
                names = ("base_0_rgb", "base_1_rgb", "wrist_0_rgb")
                image_values = (zeros_image, zeros_image, wrist_image)
                image_masks = (np.False_, np.False_, np.True_)
            case _:
                raise ValueError(f"Unsupported model type: {self.model_type}")

        inputs = {
            "state": np.asarray(data["state"]),
            "image": dict(zip(names, image_values, strict=True)),
            "image_mask": dict(zip(names, image_masks, strict=True)),
            "prompt": data.get("prompt", "stack cube"),
        }
        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"])
        return inputs


@dataclasses.dataclass(frozen=True)
class DualArmFrankaRepack(transforms.DataTransformFn):
    """把双臂三相机 Franka LeRobot 样本重排为 OpenPI 下游统一结构。"""

    def __call__(self, data: dict) -> dict:
        flat = transforms.flatten_dict(data)
        _, agent_view = _first_present(flat, "observation.images.agent_view")
        _, left_wrist = _first_present(flat, "observation.images.left_wrist")
        _, right_wrist = _first_present(flat, "observation.images.right_wrist")
        _, state = _first_present(flat, "observation.state")
        _, actions = _first_present(flat, "action", "actions")
        _, prompt = _first_present(flat, "prompt")

        missing = []
        if agent_view is None:
            missing.append("observation.images.agent_view")
        if left_wrist is None:
            missing.append("observation.images.left_wrist")
        if right_wrist is None:
            missing.append("observation.images.right_wrist")
        if state is None:
            missing.append("observation.state")
        if missing:
            raise KeyError(f"dual_arm_franka 数据缺少字段: {', '.join(missing)}。")

        repacked = {
            "images": {
                "agent_view": agent_view,
                "left_wrist": left_wrist,
                "right_wrist": right_wrist,
            },
            "state": state,
        }
        if actions is not None:
            repacked["actions"] = actions
        if prompt is not None:
            repacked["prompt"] = prompt
        return repacked


@dataclasses.dataclass(frozen=True)
class DualArmFrankaInputs(transforms.DataTransformFn):
    """双臂三相机 Franka 数据的模型输入变换。"""

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        agent_view = _parse_image(data["images"]["agent_view"])
        left_wrist = _parse_image(data["images"]["left_wrist"])
        right_wrist = _parse_image(data["images"]["right_wrist"])

        match self.model_type:
            case _model.ModelType.PI0 | _model.ModelType.PI05:
                names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
                image_values = (agent_view, left_wrist, right_wrist)
                image_masks = (np.True_, np.True_, np.True_)
            case _model.ModelType.PI0_FAST:
                names = ("base_0_rgb", "base_1_rgb", "wrist_0_rgb")
                image_values = (agent_view, left_wrist, right_wrist)
                image_masks = (np.True_, np.True_, np.True_)
            case _:
                raise ValueError(f"Unsupported model type: {self.model_type}")

        inputs = {
            "state": np.asarray(data["state"]),
            "image": dict(zip(names, image_values, strict=True)),
            "image_mask": dict(zip(names, image_masks, strict=True)),
            "prompt": data.get("prompt", "dual arm manipulation"),
        }
        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"])
        return inputs


@dataclasses.dataclass(frozen=True)
class DualArmFrankaOutputs(transforms.DataTransformFn):
    """双臂 Franka 数据的模型输出变换。"""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :16])}


@dataclasses.dataclass(frozen=True)
class StackCubeFrankaOutputs(transforms.DataTransformFn):
    """stack_cube 单臂 Franka 数据的模型输出变换。"""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :8])}
