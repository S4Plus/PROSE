#!/usr/bin/env python3
"""LIBERO/ACT HTTP 推理服务。

用法示例：
    conda activate cupid
    python /workspace/liushuo/cupid/libero/server.py \
        --checkpoint /workspace/liushuo/cupid/output/act/libero_object_multitask \
        --host 0.0.0.0 \
        --port 8000 \
        --image-width 128 \
        --image-height 128 \
        --chunk-size 50
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
LEROBOT_SRC = REPO_ROOT / "lerobot_2.1" / "src"
if str(LEROBOT_SRC) not in sys.path:
    sys.path.insert(0, str(LEROBOT_SRC))

from lerobot.policies.act.modeling_act import ACTPolicy  # noqa: E402


def resolve_checkpoint(path_str: Optional[str]) -> Path:
    """解析 checkpoint 路径，兼容多种目录结构。"""
    if path_str:
        root = Path(path_str).expanduser()
    else:
        root = REPO_ROOT / "output" / "act" / "libero_object_multitask"

    if root.is_file():
        return root.parent

    if root.is_dir():
        if (root / "config.json").is_file() or (root / "model.safetensors").is_file():
            return root

        if (root / "pretrained_model").is_dir():
            return root / "pretrained_model"

        if (root / "checkpoints").is_dir():
            root = root / "checkpoints"

        candidates = []
        for child in root.iterdir():
            if not child.is_dir():
                continue
            if child.name.isdigit():
                pretrained = child / "pretrained_model"
                if pretrained.is_dir():
                    candidates.append((int(child.name), pretrained))
            elif child.name.startswith("step_") and (child / "config.json").is_file():
                try:
                    step = int(child.name.split("step_", 1)[1])
                except ValueError:
                    continue
                candidates.append((step, child))
        if candidates:
            candidates.sort()
            return candidates[-1][1]

    raise FileNotFoundError(f"无法从路径解析 ACT checkpoint: {root}")


def decode_image(encoded: str, width: int, height: int, device: torch.device) -> torch.Tensor:
    """将 base64 编码图片解码为模型输入 tensor。"""
    raw = base64.b64decode(encoded)
    np_arr = np.frombuffer(raw, np.uint8)
    bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("图片解码失败。")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (width, height))
    tensor = torch.from_numpy(resized).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    return tensor.to(device)


class InferenceEngine:
    """ACT 推理引擎，负责模型加载和单次推理。"""

    def __init__(self, checkpoint: Path, device: str, image_width: int, image_height: int, chunk_size: Optional[int]):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.image_width = int(image_width)
        self.image_height = int(image_height)

        policy = ACTPolicy.from_pretrained(str(checkpoint))
        policy.to(self.device)
        policy.eval()
        policy.reset()
        self.policy = policy
        self.lock = threading.Lock()

        self.image_keys = list(self.policy.config.image_features.keys())
        self.expect_state = self.policy.config.robot_state_feature is not None
        self.expect_env_state = self.policy.config.env_state_feature is not None
        self.state_dim = (
            int(self.policy.config.robot_state_feature.shape[0])
            if self.policy.config.robot_state_feature is not None
            else None
        )
        self.env_state_dim = (
            int(self.policy.config.env_state_feature.shape[0])
            if self.policy.config.env_state_feature is not None
            else None
        )
        self.action_dim = int(self.policy.config.output_features["action"].shape[0])
        self.default_chunk_size = int(chunk_size or self.policy.config.n_action_steps)

    def reset(self) -> None:
        with self.lock:
            self.policy.reset()

    def infer(self, payload: Dict[str, object]) -> Dict[str, object]:
        images = payload.get("images")
        if not isinstance(images, dict):
            raise ValueError("payload 必须包含 images 字段，且类型为 dict。")

        missing = [key for key in self.image_keys if key not in images]
        if missing:
            raise ValueError(f"缺少图像键: {missing}")

        state = payload.get("state", payload.get("observation.state"))
        env_state = payload.get("env_state", payload.get("observation.environment_state"))

        if self.expect_state and state is None:
            raise ValueError("模型需要 observation.state，但请求中未提供。")
        if self.expect_env_state and env_state is None:
            raise ValueError("模型需要 observation.environment_state，但请求中未提供。")

        batch: Dict[str, torch.Tensor] = {
            key: decode_image(str(images[key]), self.image_width, self.image_height, self.device)
            for key in self.image_keys
        }

        if state is not None:
            state_arr = np.asarray(state, dtype=np.float32).reshape(-1)
            batch["observation.state"] = torch.from_numpy(state_arr).unsqueeze(0).to(self.device)

        if env_state is not None:
            env_state_arr = np.asarray(env_state, dtype=np.float32).reshape(-1)
            batch["observation.environment_state"] = torch.from_numpy(env_state_arr).unsqueeze(0).to(self.device)

        chunk_size = payload.get("chunk_size", self.default_chunk_size)
        if not isinstance(chunk_size, int) or chunk_size <= 0:
            raise ValueError("chunk_size 必须是正整数。")

        with self.lock:
            with torch.no_grad():
                action_chunk = self.policy.predict_action_chunk(batch).squeeze(0)

        action_chunk = action_chunk[:chunk_size].detach().cpu().float().numpy()
        return {
            "action": action_chunk.tolist(),
            "shape": list(action_chunk.shape),
            "action_dim": self.action_dim,
            "chunk_size": int(action_chunk.shape[0]),
        }


class ACTHTTPServer(ThreadingHTTPServer):
    def __init__(self, server_address, handler_cls, engine: InferenceEngine):
        super().__init__(server_address, handler_cls)
        self.engine = engine


class RequestHandler(BaseHTTPRequestHandler):
    server_version = "LiberoACTServer/0.1"

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
            raise ValueError(f"非法 JSON: {exc}") from exc

    def do_GET(self) -> None:  # noqa: N802
        if self.path != "/health":
            self._send_json(404, {"error": "Not found"})
            return

        engine: InferenceEngine = self.server.engine  # type: ignore[attr-defined]
        self._send_json(
            200,
            {
                "status": "ok",
                "image_keys": engine.image_keys,
                "expect_state": engine.expect_state,
                "expect_env_state": engine.expect_env_state,
                "state_dim": engine.state_dim,
                "env_state_dim": engine.env_state_dim,
                "action_dim": engine.action_dim,
                "default_chunk_size": engine.default_chunk_size,
                "image_width": engine.image_width,
                "image_height": engine.image_height,
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
            result["latency_ms"] = round((time.time() - start) * 1000, 2)
            self._send_json(200, result)
        except Exception as exc:  # noqa: BLE001
            self._send_json(400, {"error": str(exc)})

    def log_message(self, format: str, *args) -> None:  # noqa: A003
        return


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LIBERO ACT HTTP 推理服务")
    parser.add_argument("--host", default="0.0.0.0", help="绑定地址")
    parser.add_argument("--port", type=int, default=8000, help="监听端口")
    parser.add_argument("--checkpoint", default=None, help="模型目录或 checkpoint 根目录")
    parser.add_argument("--device", default=None, help="torch device，默认自动选择")
    parser.add_argument("--image-width", type=int, default=128, help="输入图片宽度")
    parser.add_argument("--image-height", type=int, default=128, help="输入图片高度")
    parser.add_argument("--chunk-size", type=int, default=50, help="默认返回动作步数")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = resolve_checkpoint(args.checkpoint)
    engine = InferenceEngine(
        checkpoint=checkpoint,
        device=args.device,
        image_width=args.image_width,
        image_height=args.image_height,
        chunk_size=args.chunk_size,
    )
    server = ACTHTTPServer((args.host, args.port), RequestHandler, engine)
    print(
        f"ACT server listening on http://{args.host}:{args.port} "
        f"(checkpoint={checkpoint}, image_keys={engine.image_keys}, "
        f"state_dim={engine.state_dim}, env_state_dim={engine.env_state_dim}, action_dim={engine.action_dim})"
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
