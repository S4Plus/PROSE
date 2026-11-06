import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def _parse_image(image) -> np.ndarray:
    """Convert image to uint8 HWC."""
    img = np.asarray(image)
    if np.issubdtype(img.dtype, np.floating):
        img = (255 * img).astype(np.uint8)
    if img.ndim == 3 and img.shape[0] in (1, 3):
        img = einops.rearrange(img, "c h w -> h w c")
    return img


@dataclasses.dataclass(frozen=True)
class SquareRepack(transforms.DataTransformFn):
    """Repack raw dataset sample into a structure expected by downstream transforms."""

    def __call__(self, data: dict) -> dict:
        flat = transforms.flatten_dict(data)
        if "observation.images.wrist" in flat:
            wrist_image = flat["observation.images.wrist"]
        elif "observation.images.robot0_eye_in_hand" in flat:
            # Compatibility path for MimicGen-style datasets.
            wrist_image = flat["observation.images.robot0_eye_in_hand"]
        else:
            raise KeyError(
                "Expected wrist image key 'observation.images.wrist' or "
                "'observation.images.robot0_eye_in_hand' in dataset sample."
            )

        repacked = {
            "images": {
                "agentview": flat["observation.images.agentview"],
                "wrist": wrist_image,
            },
            "state": flat["observation.state"],
        }
        if "action" in flat:
            repacked["actions"] = flat["action"]
        if "prompt" in flat:
            repacked["prompt"] = flat["prompt"]
        return repacked


@dataclasses.dataclass(frozen=True)
class SquareInputs(transforms.DataTransformFn):
    """Inputs for the square_d1_core dataset."""

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        # print(data.keys())
        if "images" in data:
            images = data["images"]
        else:
            images = {
                "agentview": data["observation.images.agentview"],
                "wrist": data["observation.images.wrist"],
            }

        base_image = _parse_image(images["agentview"])
        wrist_image = _parse_image(images["wrist"])
        # base_image=data["observation.images.agentview"]
        # wrist_image=data["observation.images.wrist"]
        zeros_image = np.zeros_like(base_image)

        match self.model_type:
            case _model.ModelType.PI0 | _model.ModelType.PI05:
                names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
                images = (base_image, wrist_image, zeros_image)
                image_masks = (np.True_, np.True_, np.False_)
            case _model.ModelType.PI0_FAST:
                names = ("base_0_rgb", "base_1_rgb", "wrist_0_rgb")
                images = (base_image, zeros_image, wrist_image)
                # Only agent and wrist views are present in the dataset, so ignore the extra slot.
                image_masks = (np.True_, np.False_, np.True_)
            case _:
                raise ValueError(f"Unsupported model type: {self.model_type}")

        inputs = {
            "state": np.asarray(data["state"] if "state" in data else data["observation.state"]),
            "image": dict(zip(names, images, strict=True)),
            "image_mask": dict(zip(names, image_masks, strict=True)),
        }

        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"])
        inputs["prompt"]="square_d1"
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        
        # print(inputs)
        return inputs


@dataclasses.dataclass(frozen=True)
class SquareOutputs(transforms.DataTransformFn):
    """Outputs for the square_d1_core dataset."""

    def __call__(self, data: dict) -> dict:
        # Only return the first 8 dims (7 joints + gripper cmd).
        # print(data["actions"].shape) （20,32）裁减前8维action
        # breakpoint()
        return {"actions": np.asarray(data["actions"][:, :8])}
