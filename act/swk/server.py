#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, Optional
import threading

import cv2
import numpy as np
import torch

from lerobot.policies.act.modeling_act import ACTPolicy


def resolve_checkpoint(path_str: Optional[str]) -> Path:
    if path_str:
        root = Path(path_str).expanduser()
    else:
        root = Path(__file__).resolve().parent / "outputs" / "CB81"

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
            if child.is_dir() and child.name.isdigit():
                pretrained = child / "pretrained_model"
                if pretrained.is_dir():
                    candidates.append((int(child.name), pretrained))
        if candidates:
            candidates.sort()
            return candidates[-1][1]

    raise FileNotFoundError(f"Could not resolve ACT checkpoint from '{root}'.")


def decode_image(encoded: str, width: int, height: int, device: torch.device) -> torch.Tensor:
    raw = base64.b64decode(encoded)
    np_arr = np.frombuffer(raw, np.uint8)
    bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Failed to decode image bytes.")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (width, height))
    tensor = torch.from_numpy(resized).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    return tensor.to(device)


class InferenceEngine:
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
        self.default_chunk_size = chunk_size or self.policy.config.n_action_steps

    def reset(self) -> None:
        with self.lock:
            self.policy.reset()

    def infer(self, payload: Dict[str, object]) -> Dict[str, object]:
        images = payload.get("images")
        if not isinstance(images, dict):
            raise ValueError("Payload must include an 'images' object.")

        missing = [key for key in self.image_keys if key not in images]
        if missing:
            raise ValueError(f"Missing image keys: {missing}")

        state = payload.get("state")
        if self.expect_state and state is None:
            raise ValueError("Policy expects 'state' but none was provided.")

        image_tensors = {
            key: decode_image(str(images[key]), self.image_width, self.image_height, self.device)
            for key in self.image_keys
        }

        batch: Dict[str, torch.Tensor] = {}
        if state is not None:
            state_arr = np.asarray(state, dtype=np.float32)
            batch["observation.state"] = torch.from_numpy(state_arr).unsqueeze(0).to(self.device)
        batch.update(image_tensors)

        chunk_size = payload.get("chunk_size") or self.default_chunk_size
        if not isinstance(chunk_size, int) or chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer.")

        with self.lock:
            with torch.no_grad():
                action_chunk = self.policy.predict_action_chunk(batch).squeeze(0)
                # print(action_chunk.shape)

        action_chunk = action_chunk[:chunk_size].detach().cpu().float().numpy()
        return {
            "action": action_chunk.tolist(),
            "shape": list(action_chunk.shape),
        }


class ACTHTTPServer(ThreadingHTTPServer):
    def __init__(self, server_address, handler_cls, engine: InferenceEngine):
        super().__init__(server_address, handler_cls)
        self.engine = engine


class RequestHandler(BaseHTTPRequestHandler):
    server_version = "ACTServer/0.1"

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
            raise ValueError(f"Invalid JSON: {exc}") from exc

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler naming
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
                "default_chunk_size": engine.default_chunk_size,
                "image_width": engine.image_width,
                "image_height": engine.image_height,
            },
        )

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler naming
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

    def log_message(self, format: str, *args) -> None:  # noqa: A003 - BaseHTTPRequestHandler API
        return


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HTTP inference server for ACT checkpoints.")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host.")
    parser.add_argument("--port", type=int, default=8000, help="Bind port.")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Path to pretrained ACT directory or checkpoint root (default: swk/outputs/CB81).",
    )
    parser.add_argument("--device", default=None, help="Torch device string (default: auto).")
    parser.add_argument("--image-width", type=int, default=640, help="Resize width for incoming images.")
    parser.add_argument("--image-height", type=int, default=360, help="Resize height for incoming images.")
    parser.add_argument("--chunk-size", type=int, default=50, help="Default number of actions to return.")
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
        f"(checkpoint={checkpoint}, images={engine.image_keys})."
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
