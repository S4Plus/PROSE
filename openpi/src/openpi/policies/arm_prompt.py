import dataclasses
import re

import numpy as np

from openpi import transforms


@dataclasses.dataclass(frozen=True)
class InjectArmIdIntoPrompt(transforms.DataTransformFn):
    """将 arm_id 转成左右臂提示词，并注入到 prompt 前缀。"""

    left_arm_id: int = 0
    right_arm_id: int = 1
    strict_conflict: bool = True

    def __call__(self, data: dict) -> dict:
        if "prompt" not in data:
            raise ValueError("缺少 prompt，无法注入 arm_id 条件。")
        if "arm_id" not in data:
            raise ValueError("缺少 arm_id，无法注入 prompt。")

        prompt = data["prompt"]
        if not isinstance(prompt, str):
            prompt = prompt.item()

        arm_id = int(np.asarray(data["arm_id"]).reshape(-1)[0])
        if arm_id == self.left_arm_id:
            arm_name = "left"
        elif arm_id == self.right_arm_id:
            arm_name = "right"
        else:
            raise ValueError(f"未知 arm_id: {arm_id}")

        lowered = prompt.lower()
        mentioned_arms = []
        if "left arm" in lowered:
            mentioned_arms.append("left")
        if "right arm" in lowered:
            mentioned_arms.append("right")

        if self.strict_conflict and mentioned_arms and arm_name not in mentioned_arms:
            raise ValueError(f"prompt 与 arm_id 冲突: prompt={prompt!r}, arm_id={arm_id}")

        # 删除已有左右臂描述，统一用稳定模板，减少语言噪声。
        body = re.sub(r"\b(?:use|using|with)\s+the\s+(?:left|right)\s+arm\b[, ]*", "", prompt, flags=re.IGNORECASE)
        body = re.sub(r"\b(?:use|using|with)\s+(?:left|right)\s+arm\b[, ]*", "", body, flags=re.IGNORECASE)
        body = re.sub(r"\s+", " ", body).strip(" .,")

        data["prompt"] = f"Use the {arm_name} arm. {body}"
        data.pop("arm_id", None)
        return data
