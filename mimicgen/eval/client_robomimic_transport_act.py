#!/usr/bin/env python3
"""
Transport 专用的 robomimic ACT 评测入口。

这个入口只服务双臂 transport 任务，把它和单臂任务常用的通用 client 分开。

示例:
    python client_robomimic_transport_act.py \
        --controller JOINT_POSITION \
        --joint_input_type absolute \
        --joint_control_delta false \
        --test_init_mode random \
        --n_rollouts 50
"""

from __future__ import annotations

import sys
from typing import List, Optional

import client_robomimic_act as _act


TRANSPORT_ENV_ALIASES = {"transport", "twoarmtransport"}
DEFAULT_TRANSPORT_CAMERA_NAMES = "shouldercamera0,shouldercamera1,robot0_eye_in_hand,robot1_eye_in_hand"


def _get_flag_value(argv: List[str], flag: str) -> Optional[str]:
    prefix = flag + "="
    for idx, token in enumerate(argv):
        if token == flag and idx + 1 < len(argv):
            return argv[idx + 1]
        if token.startswith(prefix):
            return token[len(prefix) :]
    return None


def _ensure_transport_defaults(argv: List[str]) -> List[str]:
    out = list(argv)

    env_name = _get_flag_value(out, "--env_name")
    if env_name is not None and str(env_name).lower() not in TRANSPORT_ENV_ALIASES:
        raise SystemExit("client_robomimic_transport_act.py 只支持 --env_name transport。")
    if env_name is None:
        out.extend(["--env_name", "transport"])

    if _get_flag_value(out, "--camera_names") is None:
        out.extend(["--camera_names", DEFAULT_TRANSPORT_CAMERA_NAMES])

    return out


def main() -> None:
    sys.argv = [sys.argv[0], *_ensure_transport_defaults(sys.argv[1:])]
    _act.main()


if __name__ == "__main__":
    main()
