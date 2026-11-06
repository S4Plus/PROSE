#!/usr/bin/env python3
"""
OpenPI05 websocket client for robomimic / robosuite evaluation.

This script reuses `client_robomimic_act.py` evaluation logic (env creation, test-case
application, rollout statistics, saving videos / pkl), and swaps only the server transport:

- ACT HTTP API -> OpenPI websocket API (msgpack + numpy arrays)

Typical usage:
    python client_robomimic_openpi05.py \
        --server_url ws://127.0.0.1:8002 \
        --env_name can \
        --video_dir /workspace/liushuo/cupid/mimicgen/eval/tmp/can_openpi05 \
        --save_rollouts_dir /workspace/liushuo/cupid/mimicgen/eval/tmp/can_openpi05/pkl
"""

from __future__ import annotations

import base64
from io import BytesIO
import os
from typing import Any, Dict, Optional

import numpy as np

import client_robomimic_act as _act

try:
    import msgpack
except Exception:  # pragma: no cover
    msgpack = None

try:
    import websockets.sync.client as _ws_client
except Exception:  # pragma: no cover
    _ws_client = None

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None


def _pack_array(obj):
    if (isinstance(obj, (np.ndarray, np.generic))) and obj.dtype.kind in ("V", "O", "c"):
        raise ValueError(f"Unsupported dtype: {obj.dtype}")
    if isinstance(obj, np.ndarray):
        return {
            b"__ndarray__": True,
            b"data": obj.tobytes(),
            b"dtype": obj.dtype.str,
            b"shape": obj.shape,
        }
    if isinstance(obj, np.generic):
        return {
            b"__npgeneric__": True,
            b"data": obj.item(),
            b"dtype": obj.dtype.str,
        }
    return obj


def _unpack_array(obj):
    if b"__ndarray__" in obj:
        return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"])
    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])
    return obj


class _MsgpackNumpy:
    @staticmethod
    def _require_msgpack():
        if msgpack is None:
            raise RuntimeError("msgpack is required for ws transport. Install: pip install msgpack")

    @staticmethod
    def packb(data):
        _MsgpackNumpy._require_msgpack()
        return msgpack.packb(data, default=_pack_array)

    @staticmethod
    def unpackb(data):
        _MsgpackNumpy._require_msgpack()
        return msgpack.unpackb(data, object_hook=_unpack_array)


class _WebsocketPolicyClient:
    def __init__(self, uri: str):
        if _ws_client is None:
            raise RuntimeError("websockets>=11 is required for ws transport. Install: pip install 'websockets>=11'")
        self._ws = _ws_client.connect(uri, compression=None, max_size=None)
        meta = self._ws.recv()
        self._metadata = _MsgpackNumpy.unpackb(meta) if isinstance(meta, (bytes, bytearray)) else {}

    @property
    def metadata(self) -> Dict[str, Any]:
        return self._metadata if isinstance(self._metadata, dict) else {}

    def infer(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        self._ws.send(_MsgpackNumpy.packb(obs))
        response = self._ws.recv()
        if isinstance(response, str):
            raise RuntimeError(f"Error in inference server:\n{response}")
        out = _MsgpackNumpy.unpackb(response)
        if not isinstance(out, dict):
            raise RuntimeError(f"Unexpected ws response type: {type(out)}")
        return out

    def close(self) -> None:
        try:
            self._ws.close()
        except Exception:
            pass


_WS_CLIENT: Optional[_WebsocketPolicyClient] = None
_WS_URI: Optional[str] = None


def _normalize_ws_uri(url: str) -> str:
    raw = str(url).strip()
    if raw.startswith("ws://") or raw.startswith("wss://"):
        uri = raw
    elif raw.startswith("http://"):
        uri = "ws://" + raw[len("http://") :]
    elif raw.startswith("https://"):
        uri = "wss://" + raw[len("https://") :]
    else:
        uri = "ws://" + raw
    for suffix in ("/health", "/healthz", "/reset"):
        if uri.endswith(suffix):
            uri = uri[: -len(suffix)]
            break
    return uri


def _get_ws_client(url: str) -> _WebsocketPolicyClient:
    global _WS_CLIENT, _WS_URI
    uri = _normalize_ws_uri(url)
    if _WS_CLIENT is not None and _WS_URI == uri:
        return _WS_CLIENT
    if _WS_CLIENT is not None:
        _WS_CLIENT.close()
    _WS_CLIENT = _WebsocketPolicyClient(uri)
    _WS_URI = uri
    return _WS_CLIENT


def _decode_image_from_base64(image_b64: str) -> np.ndarray:
    raw = base64.b64decode(image_b64)
    if cv2 is not None:
        arr = np.frombuffer(raw, dtype=np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("Failed to decode JPEG image with cv2.")
        return bgr[..., ::-1]
    if Image is None:
        raise RuntimeError("No image decoder available. Install pillow or opencv-python.")
    return np.asarray(Image.open(BytesIO(raw)).convert("RGB"))


def _build_openpi_obs_from_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    images = payload.get("images", {}) or {}
    if not isinstance(images, dict):
        raise ValueError("Payload 'images' must be a dict.")

    agent = images.get("observation.images.agentview")
    wrist = images.get("observation.images.wrist")
    if agent is None or wrist is None:
        raise KeyError(
            "Missing required images in payload. Need keys: "
            "'observation.images.agentview' and 'observation.images.wrist'."
        )

    state = payload.get("state", None)
    if state is None:
        raise KeyError("Missing required 'state' in payload.")
    state = np.asarray(state, dtype=np.float32).reshape(-1)

    obs = {
        "observation.images.agentview": _decode_image_from_base64(agent),
        "observation.images.wrist": _decode_image_from_base64(wrist),
        "observation.state": state,
        "state": state,
    }
    # Keep optional prompt passthrough if caller provides it.
    if "prompt" in payload:
        obs["prompt"] = payload["prompt"]
    return obs


def _http_json_openpi(
    method: str,
    url: str,
    payload: Optional[Dict[str, Any]] = None,
    timeout_s: float = 10.0,  # noqa: ARG001
) -> Dict[str, Any]:
    m = str(method).upper()
    if m == "GET":
        client = _get_ws_client(url)
        metadata = client.metadata
        state_dim = metadata.get("state_dim", 9)
        if not isinstance(state_dim, int):
            state_dim = 9
        return {
            "image_keys": ["observation.images.agentview", "observation.images.wrist"],
            "expect_state": True,
            "state_dim": state_dim,
            "metadata": metadata,
        }

    if m == "POST":
        # OpenPI websocket server has no reset endpoint; treat as no-op.
        if str(url).rstrip("/").endswith("/reset"):
            return {"ok": True}
        client = _get_ws_client(url)
        obs = _build_openpi_obs_from_payload(payload or {})
        result = client.infer(obs)
        action = result.get("actions", result.get("action", result))
        action_arr = np.asarray(action, dtype=np.float32)
        out = {"action": action_arr.tolist()}
        if "server_timing" in result:
            out["server_timing"] = result["server_timing"]
        if "policy_timing" in result:
            out["policy_timing"] = result["policy_timing"]
        return out

    raise ValueError(f"Unsupported HTTP method for OpenPI websocket shim: {method}")


def _health_url_openpi(server_url: str) -> str:
    return server_url


def _reset_url_openpi(server_url: str) -> str:
    return server_url.rstrip("/") + "/reset"


_orig_build_parser = _act._build_parser


def _build_parser_openpi() -> Any:
    parser = _orig_build_parser()
    parser.description = "Evaluate OpenPI05 websocket policy server on robomimic/robosuite environments."
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Optional language prompt sent to the OpenPI server on every policy call.",
    )
    parser.set_defaults(
        server_url=os.environ.get("OPENPI_SERVER_URL", "ws://127.0.0.1:8000"),
        joint_input_type="absolute",
        joint_control_delta="false",
    )
    return parser


def _patch_act_client() -> None:
    _act._http_json = _http_json_openpi
    _act._health_url = _health_url_openpi
    _act._reset_url = _reset_url_openpi
    _act._build_parser = _build_parser_openpi


def main() -> None:
    _patch_act_client()
    try:
        _act.main()
    finally:
        global _WS_CLIENT
        if _WS_CLIENT is not None:
            _WS_CLIENT.close()
            _WS_CLIENT = None


if __name__ == "__main__":
    main()
