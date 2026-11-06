import dataclasses
import json
import logging
import pathlib
import time

import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy
from PIL import Image
import tyro


LOGGER = logging.getLogger(__name__)


@dataclasses.dataclass
class Args:
    """stack_cube_franka 专用 websocket client 参数。"""

    host: str = "127.0.0.1"
    port: int = 8003
    api_key: str | None = None
    prompt: str = "stack cube"
    state: str = "0,0,0,0,0,0,0,0"
    fixed_image: pathlib.Path | None = None
    wrist_image: pathlib.Path | None = None
    image_size: int = 224
    num_requests: int = 1
    sleep_s: float = 0.0
    seed: int = 0


def _parse_state(text: str) -> np.ndarray:
    """把命令行里的逗号分隔 state 解析成 float32 向量。"""
    values = [float(item.strip()) for item in text.split(",") if item.strip()]
    state = np.asarray(values, dtype=np.float32)
    if state.shape != (8,):
        raise ValueError(f"stack_cube_franka 期望 8 维 state（7 维关节角 + 1 维 gripper），实际收到 {state.shape}")
    return state


def _load_image(path: pathlib.Path, image_size: int) -> np.ndarray:
    """从磁盘读取图像并转成模型推理使用的 uint8 HWC 格式。"""
    with Image.open(path) as image:
        rgb = np.asarray(image.convert("RGB"))
    rgb = image_tools.convert_to_uint8(rgb)
    rgb = image_tools.resize_with_pad(rgb, image_size, image_size)
    return np.asarray(rgb, dtype=np.uint8)


def _random_image(image_size: int, rng: np.random.Generator) -> np.ndarray:
    """生成一张随机图，便于验证链路而非验证策略质量。"""
    return rng.integers(0, 256, size=(image_size, image_size, 3), dtype=np.uint8)


def _build_observation(args: Args) -> dict:
    """构造 stack_cube_franka inference 所需 observation。"""
    rng = np.random.default_rng(args.seed)
    fixed_image = _load_image(args.fixed_image, args.image_size) if args.fixed_image else _random_image(args.image_size, rng)
    wrist_image = _load_image(args.wrist_image, args.image_size) if args.wrist_image else _random_image(args.image_size, rng)
    return {
        "images": {
            "fixed_camera": fixed_image,
            "wrist_camera": wrist_image,
        },
        "state": _parse_state(args.state),
        "prompt": args.prompt,
    }


def main(args: Args) -> None:
    client = websocket_client_policy.WebsocketClientPolicy(
        host=args.host,
        port=args.port,
        api_key=args.api_key,
    )
    LOGGER.info("Server metadata: %s", client.get_server_metadata())

    observation = _build_observation(args)
    LOGGER.info(
        "Observation ready: state_shape=%s fixed_image_shape=%s wrist_image_shape=%s prompt=%s",
        observation["state"].shape,
        observation["images"]["fixed_camera"].shape,
        observation["images"]["wrist_camera"].shape,
        observation["prompt"],
    )

    for step in range(args.num_requests):
        start_time = time.time()
        response = client.infer(observation)
        elapsed_ms = 1000.0 * (time.time() - start_time)

        actions = np.asarray(response["actions"])
        summary = {
            "step": step,
            "client_infer_ms": round(elapsed_ms, 2),
            "actions_shape": list(actions.shape),
            "first_action": actions[0].round(6).tolist() if actions.ndim == 2 and len(actions) > 0 else [],
            "policy_timing": response.get("policy_timing", {}),
            "server_timing": response.get("server_timing", {}),
        }
        print(json.dumps(summary, ensure_ascii=False))

        if args.sleep_s > 0 and step + 1 < args.num_requests:
            time.sleep(args.sleep_s)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
