#!/usr/bin/env python3
from __future__ import annotations

"""
RobotWin ACT HTTP 推理服务。

服务职责：
1. 加载 RobotWin ACT checkpoint。
2. 根据 checkpoint 的 config.json 自动解析图像 / 状态 / 环境条件输入规格。
3. 提供 /health、/infer、/reset 三个 HTTP 接口。

默认兼容模型：
/workspace/liushuo/cupid/act/outputs/robotwin/beat_block_hammer_ur5_randomized_500_act
"""

import argparse
import base64
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np
import torch

from lerobot.policies.act.modeling_act import ACTPolicy


DEFAULT_CHECKPOINT = Path(
    "/workspace/liushuo/cupid/act/outputs/robotwin/beat_block_hammer_ur5_randomized_500_act"
)

ARM_ID_TO_VALUE = {
    "left": 0.0,
    "right": 1.0,
}


def resolve_checkpoint(path_str: Optional[str]) -> Path:
    """解析 checkpoint 目录，兼容根目录、step_xxx 子目录与显式 pretrained 目录。"""
    root = Path(path_str).expanduser() if path_str else DEFAULT_CHECKPOINT

    if root.is_file():
        return root.parent

    if not root.is_dir():
        raise FileNotFoundError(f"未找到 checkpoint 路径: {root}")

    if (root / "config.json").is_file() and (root / "model.safetensors").is_file():
        return root

    if (root / "pretrained_model").is_dir():
        return root / "pretrained_model"

    step_candidates: list[tuple[int, Path]] = []
    for child in root.iterdir():
        if not child.is_dir() or not child.name.startswith("step_"):
            continue
        suffix = child.name.removeprefix("step_")
        if not suffix.isdigit():
            continue
        if (child / "config.json").is_file() and (child / "model.safetensors").is_file():
            step_candidates.append((int(suffix), child))

    if step_candidates:
        step_candidates.sort(key=lambda item: item[0])
        return step_candidates[-1][1]

    raise FileNotFoundError(f"无法从 '{root}' 解析出有效 ACT checkpoint。")


def load_model_config(checkpoint: Path) -> Dict[str, object]:
    config_path = checkpoint / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"缺少配置文件: {config_path}")
    return json.loads(config_path.read_text(encoding="utf-8"))


def decode_image_rgb(encoded: str) -> np.ndarray:
    raw = base64.b64decode(encoded)
    np_arr = np.frombuffer(raw, np.uint8)
    bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("图像解码失败。")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def rgb_to_tensor(rgb: np.ndarray, width: int, height: int, device: torch.device) -> torch.Tensor:
    resized = cv2.resize(rgb, (width, height))
    tensor = torch.from_numpy(resized).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    return tensor.to(device)


def resolve_arm_id_value(arm_id: object) -> float:
    """将 arm_id 解析为数据集约定的 env_state 数值。"""
    if isinstance(arm_id, str):
        normalized = arm_id.strip().lower()
        if normalized in ARM_ID_TO_VALUE:
            return ARM_ID_TO_VALUE[normalized]
        if normalized in {"0", "0.0"}:
            return 0.0
        if normalized in {"1", "1.0"}:
            return 1.0
    elif isinstance(arm_id, (int, float)):
        value = float(arm_id)
        if value in (0.0, 1.0):
            return value

    raise ValueError("arm_id 仅支持 left/right/0/1。")


class InferenceEngine:
    """封装 ACT policy 的线程安全推理逻辑。"""

    def __init__(self, checkpoint: Path, device: Optional[str], default_chunk_size: Optional[int]) -> None:
        self.checkpoint = checkpoint
        self.model_cfg = load_model_config(checkpoint)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        input_features = self.model_cfg.get("input_features", {})
        output_features = self.model_cfg.get("output_features", {})
        if not isinstance(input_features, dict) or not isinstance(output_features, dict):
            raise ValueError("config.json 中 input_features/output_features 格式非法。")

        self.image_specs: Dict[str, Dict[str, int]] = {}
        self.state_dim: Optional[int] = None
        self.env_state_dim: Optional[int] = None
        self.action_dim: Optional[int] = None

        for key, spec in input_features.items():
            if not isinstance(spec, dict):
                continue
            feature_type = spec.get("type")
            shape = spec.get("shape")

            if feature_type == "VISUAL":
                if not isinstance(shape, list) or len(shape) != 3:
                    raise ValueError(f"图像特征 {key} 的 shape 非法: {shape}")
                channels, height, width = [int(x) for x in shape]
                self.image_specs[key] = {
                    "channels": channels,
                    "height": height,
                    "width": width,
                }
            elif key == "observation.state":
                if not isinstance(shape, list) or len(shape) != 1:
                    raise ValueError(f"状态特征 {key} 的 shape 非法: {shape}")
                self.state_dim = int(shape[0])
            elif key == "observation.environment_state":
                if not isinstance(shape, list) or len(shape) != 1:
                    raise ValueError(f"环境特征 {key} 的 shape 非法: {shape}")
                self.env_state_dim = int(shape[0])

        action_spec = output_features.get("action")
        if isinstance(action_spec, dict):
            action_shape = action_spec.get("shape")
            if isinstance(action_shape, list) and len(action_shape) == 1:
                self.action_dim = int(action_shape[0])

        policy = ACTPolicy.from_pretrained(str(checkpoint))
        policy.to(self.device)
        policy.eval()
        policy.reset()
        self.policy = policy
        self.lock = threading.Lock()

        self.image_keys = list(self.image_specs.keys())
        self.expect_state = self.state_dim is not None
        self.expect_env_state = self.env_state_dim is not None
        self.default_chunk_size = (
            int(default_chunk_size)
            if default_chunk_size is not None
            else int(self.model_cfg.get("n_action_steps", self.model_cfg.get("chunk_size", 1)))
        )

    def reset(self) -> None:
        with self.lock:
            self.policy.reset()

    def _parse_state(self, payload: Dict[str, object]) -> Optional[torch.Tensor]:
        if not self.expect_state:
            return None

        state = payload.get("state")
        if state is None:
            raise ValueError("当前模型要求提供 state。")
        state_arr = np.asarray(state, dtype=np.float32)
        if state_arr.ndim != 1 or self.state_dim is None or state_arr.shape[0] != self.state_dim:
            raise ValueError(f"state 维度错误，期望 {(self.state_dim,)}, 实际 {tuple(state_arr.shape)}")
        return torch.from_numpy(state_arr).unsqueeze(0).to(self.device)

    def _parse_env_state(self, payload: Dict[str, object]) -> Optional[torch.Tensor]:
        if not self.expect_env_state:
            return None

        env_state = payload.get("env_state")
        if env_state is None:
            arm_id = payload.get("arm_id")
            if self.env_state_dim == 1 and arm_id is not None:
                env_state_arr = np.asarray([resolve_arm_id_value(arm_id)], dtype=np.float32)
            else:
                raise ValueError("当前模型要求提供 env_state；若是 RobotWin arm_id 条件，可直接传 arm_id。")
        else:
            env_state_arr = np.asarray(env_state, dtype=np.float32)
            if env_state_arr.ndim != 1 or self.env_state_dim is None or env_state_arr.shape[0] != self.env_state_dim:
                raise ValueError(
                    f"env_state 维度错误，期望 {(self.env_state_dim,)}, 实际 {tuple(env_state_arr.shape)}"
                )

        return torch.from_numpy(env_state_arr).unsqueeze(0).to(self.device)

    def infer(self, payload: Dict[str, object]) -> Dict[str, object]:
        images = payload.get("images")
        if not isinstance(images, dict):
            raise ValueError("payload 必须包含 images 字段。")

        missing = [key for key in self.image_keys if key not in images]
        if missing:
            raise ValueError(f"缺少图像输入: {missing}")

        batch: Dict[str, torch.Tensor] = {}
        for key in self.image_keys:
            spec = self.image_specs[key]
            rgb = decode_image_rgb(str(images[key]))
            batch[key] = rgb_to_tensor(
                rgb=rgb,
                width=spec["width"],
                height=spec["height"],
                device=self.device,
            )

        state_tensor = self._parse_state(payload)
        if state_tensor is not None:
            batch["observation.state"] = state_tensor

        env_state_tensor = self._parse_env_state(payload)
        if env_state_tensor is not None:
            batch["observation.environment_state"] = env_state_tensor

        chunk_size = payload.get("chunk_size", self.default_chunk_size)
        if not isinstance(chunk_size, int) or chunk_size <= 0:
            raise ValueError("chunk_size 必须是正整数。")

        with self.lock:
            with torch.no_grad():
                action_chunk = self.policy.predict_action_chunk(batch).squeeze(0)

        action_chunk = action_chunk[:chunk_size].detach().cpu().float().numpy()
        if self.action_dim is not None and action_chunk.ndim == 2 and action_chunk.shape[1] != self.action_dim:
            raise ValueError(
                f"模型输出维度错误，期望 action_dim={self.action_dim}，实际 {tuple(action_chunk.shape)}"
            )

        return {
            "action": action_chunk.tolist(),
            "shape": list(action_chunk.shape),
            "action_dim": self.action_dim,
        }


class RobotWinHTTPServer(ThreadingHTTPServer):
    def __init__(self, server_address, handler_cls, engine: InferenceEngine):
        super().__init__(server_address, handler_cls)
        self.engine = engine


class RequestHandler(BaseHTTPRequestHandler):
    server_version = "RobotWinACTServer/0.1"

    def _send_json(self, status: int, payload: Dict[str, object]) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _read_json(self) -> Dict[str, object]:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length > 0 else b"{}"
        try:
            return json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"JSON 解析失败: {exc}") from exc

    def do_GET(self) -> None:  # noqa: N802
        if self.path != "/health":
            self._send_json(404, {"error": "Not found"})
            return

        engine: InferenceEngine = self.server.engine  # type: ignore[attr-defined]
        self._send_json(
            200,
            {
                "status": "ok",
                "checkpoint": str(engine.checkpoint),
                "device": str(engine.device),
                "image_keys": engine.image_keys,
                "image_specs": engine.image_specs,
                "expect_state": engine.expect_state,
                "state_dim": engine.state_dim,
                "expect_env_state": engine.expect_env_state,
                "env_state_dim": engine.env_state_dim,
                "env_state_semantics": "RobotWin arm_id, 0.0=left, 1.0=right" if engine.env_state_dim == 1 else None,
                "action_dim": engine.action_dim,
                "default_chunk_size": engine.default_chunk_size,
                "n_action_steps": engine.model_cfg.get("n_action_steps"),
            },
        )

    def do_POST(self) -> None:  # noqa: N802
        engine: InferenceEngine = self.server.engine  # type: ignore[attr-defined]

        try:
            if self.path == "/reset":
                engine.reset()
                self._send_json(200, {"status": "ok"})
                return

            if self.path != "/infer":
                self._send_json(404, {"error": "Not found"})
                return

            payload = self._read_json()
            start = time.time()
            result = engine.infer(payload)
            result["latency_ms"] = round((time.time() - start) * 1000.0, 2)
            self._send_json(200, result)
        except Exception as exc:  # noqa: BLE001
            self._send_json(400, {"error": str(exc)})

    def log_message(self, format: str, *args) -> None:  # noqa: A003
        return


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RobotWin ACT HTTP 推理服务")
    parser.add_argument("--host", default="0.0.0.0", help="监听地址")
    parser.add_argument("--port", type=int, default=8002, help="监听端口")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(DEFAULT_CHECKPOINT),
        help="ACT checkpoint 路径，可传根目录、step_xxx 或 pretrained_model",
    )
    parser.add_argument("--device", type=str, default=None, help="Torch device，例如 cuda:0")
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="默认返回动作步数；不传时跟随训练配置 n_action_steps",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = resolve_checkpoint(args.checkpoint)
    engine = InferenceEngine(
        checkpoint=checkpoint,
        device=args.device,
        default_chunk_size=args.chunk_size,
    )
    server = RobotWinHTTPServer((args.host, args.port), RequestHandler, engine)
    print(
        f"RobotWin ACT server listening on http://{args.host}:{args.port} "
        f"(checkpoint={checkpoint}, device={engine.device}, "
        f"state_dim={engine.state_dim}, env_state_dim={engine.env_state_dim}, "
        f"action_dim={engine.action_dim}, default_chunk={engine.default_chunk_size})"
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
