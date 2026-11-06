"""
Fine-tune OpenVLA with LoRA on a LeRobot-format dataset.
"""

import draccus

from prismatic.vla.lerobot_openvla import OpenVLALeRobotConfig, finetune_openvla_lerobot


@draccus.wrap()
def main(cfg: OpenVLALeRobotConfig) -> None:
    finetune_openvla_lerobot(cfg)


if __name__ == "__main__":
    main()
