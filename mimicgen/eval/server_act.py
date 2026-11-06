#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
import time
from io import BytesIO
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, Optional
import threading

import numpy as np
import torch
from PIL import Image

try:
    import cv2
except Exception:  # pragma: no cover - optional fallback to PIL
    cv2 = None

from lerobot.policies.act.modeling_act import ACTPolicy


IMAGE_KEY_ALIASES = {
    "observation.images.agentview": ["agentview_image", "agentview"],
    "observation.images.wrist": ["robot0_eye_in_hand_image", "robot0_eye_in_hand", "wrist"],
    "observation.images.robot0_eye_in_hand": [
        "robot0_eye_in_hand_image",
        "robot0_eye_in_hand",
        "wrist",
        "observation.images.wrist",
    ],
}


def _all_aliases(key: str) -> list[str]:
    aliases = [key]
    aliases.extend(IMAGE_KEY_ALIASES.get(key, []))
    return aliases


def _resolve_images(images: Dict[str, object], required_keys: list[str]) -> Dict[str, object]:
    resolved: Dict[str, object] = {}
    for required_key in required_keys:
        value = None
        for candidate in _all_aliases(required_key):
            if candidate in images:
                value = images[candidate]
                break
        if value is None:
            raise ValueError(f"Missing image key '{required_key}'. Got keys: {sorted(images.keys())}")
        resolved[required_key] = value
    return resolved


def resolve_checkpoint(path_str: Optional[str]) -> Path:
    if path_str:
        root = Path(path_str).expanduser()
    else:
        root = Path(__file__).resolve().parent / "outputs"

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
        try:
            for child in root.iterdir():
                if child.is_dir() and child.name.isdigit():
                    pretrained = child / "pretrained_model"
                    if pretrained.is_dir():
                        candidates.append((int(child.name), pretrained))
        except FileNotFoundError:
            candidates = []

        if candidates:
            candidates.sort()
            return candidates[-1][1]

        matches = []
        for path in root.rglob("pretrained_model"):
            if not path.is_dir():
                continue
            step = int(path.parent.name) if path.parent.name.isdigit() else -1
            matches.append((step, path))
        if matches:
            matches.sort()
            return matches[-1][1]

    raise FileNotFoundError(f"Could not resolve ACT checkpoint from '{root}'.")


def _strip_data_url(encoded: str) -> str:
    if encoded.startswith("data:"):
        parts = encoded.split(",", 1)
        if len(parts) == 2:
            return parts[1]
    return encoded


def decode_image(encoded: str, width: int, height: int, device: torch.device) -> torch.Tensor:
    raw = base64.b64decode(_strip_data_url(encoded))
    if cv2 is not None:
        np_arr = np.frombuffer(raw, np.uint8)
        bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("Failed to decode image bytes.")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (width, height))
    else:
        image = Image.open(BytesIO(raw)).convert("RGB")
        image = image.resize((width, height))
        resized = np.asarray(image)
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
        self.state_dim = (
            int(np.prod(self.policy.config.robot_state_feature.shape))
            if self.expect_state
            else None
        )
        self.action_dim = None
        if getattr(self.policy.config, "action_feature", None) is not None:
            self.action_dim = int(np.prod(self.policy.config.action_feature.shape))
        self.default_chunk_size = chunk_size or self.policy.config.n_action_steps

    def reset(self) -> None:
        with self.lock:
            self.policy.reset()

    def _normalize_payload(self, payload: Dict[str, object]) -> Dict[str, object]:
        if "obs" in payload and "images" not in payload:
            obs = payload.get("obs")
            if isinstance(obs, dict):
                images_dict: Dict[str, object] = {}
                for key in self.image_keys:
                    if key in obs:
                        images_dict[key] = obs[key]
                        continue
                    for alias in IMAGE_KEY_ALIASES.get(key, []):
                        if alias in obs:
                            images_dict[key] = obs[alias]
                            break
                if images_dict:
                    payload["images"] = images_dict

                if self.expect_state and "state" not in payload:
                    if "observation.state" in obs:
                        payload["state"] = obs["observation.state"]
                    elif "state" in obs:
                        payload["state"] = obs["state"]
                    else:
                        joint_pos = obs.get("robot0_joint_pos")
                        grip_joint = obs.get("robot0_gripper_qpos")
                        if joint_pos is not None and grip_joint is not None:
                            state = np.concatenate(
                                [
                                    np.asarray(joint_pos).reshape(-1),
                                    np.asarray(grip_joint).reshape(-1),
                                ],
                                axis=-1,
                            )
                            payload["state"] = state
        return payload

    def infer(self, payload: Dict[str, object]) -> Dict[str, object]:
        payload = self._normalize_payload(payload)
        images = payload.get("images")
        if not isinstance(images, dict):
            raise ValueError("Payload must include an 'images' object.")
        images = _resolve_images(images, self.image_keys)

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
            if self.state_dim is not None and state_arr.size != self.state_dim:
                raise ValueError(
                    f"State dim mismatch: got {state_arr.size}, expected {self.state_dim}."
                )
            batch["observation.state"] = torch.from_numpy(state_arr).unsqueeze(0).to(self.device)
        batch.update(image_tensors)

        chunk_size = payload.get("chunk_size") or self.default_chunk_size
        try:
            chunk_size = int(chunk_size)
        except Exception as exc:  # noqa: BLE001
            raise ValueError("chunk_size must be a positive integer.") from exc
        if chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer.")

        with self.lock:
            with torch.no_grad():
                action_chunk = self.policy.predict_action_chunk(batch).squeeze(0)

        action_chunk = action_chunk[:chunk_size].detach().cpu().float().numpy()
        return {
            "action": action_chunk.tolist(),
            "shape": list(action_chunk.shape),
            "action_dim": int(action_chunk.shape[-1]) if action_chunk.ndim >= 1 else self.action_dim,
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
                "state_dim": engine.state_dim,
                "action_dim": engine.action_dim,
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

            if self.path not in ("/infer", "/act"):
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
        help="Path to pretrained ACT directory or checkpoint root (default: ./outputs).",
    )
    parser.add_argument("--device", default=None, help="Torch device string (default: auto).")
    parser.add_argument("--image-width", type=int, default=84, help="Resize width for incoming images.")
    parser.add_argument("--image-height", type=int, default=84, help="Resize height for incoming images.")
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
