import pytest

from openpi.policies import arm_prompt


def test_inject_arm_id_into_prompt_adds_stable_prefix():
    transform = arm_prompt.InjectArmIdIntoPrompt()

    result = transform({"prompt": "Pick up the hammer and hit the block.", "arm_id": [1]})

    assert result["prompt"] == "Use the right arm. Pick up the hammer and hit the block"
    assert "arm_id" not in result


def test_inject_arm_id_into_prompt_detects_conflict():
    transform = arm_prompt.InjectArmIdIntoPrompt(strict_conflict=True)

    with pytest.raises(ValueError, match="prompt 与 arm_id 冲突"):
        transform({"prompt": "Use the left arm to pick up the hammer.", "arm_id": [1]})
