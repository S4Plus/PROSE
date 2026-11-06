#!/usr/bin/env python3
"""
OpenVLA websocket client for robomimic / robosuite evaluation.

简化原则：
- 只保留 OpenVLA 评测必需逻辑
- 默认不做动作后处理，直接使用服务端原始输出
- 保留输入图导出能力，便于排查观测分布问题
"""

from __future__ import annotations

import base64
import os
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from PIL import Image

import client_robomimic_act as _act

try:
    import msgpack
except Exception:  # pragma: no cover
    msgpack = None

try:
    import websockets.sync.client as _ws_client
except Exception:  # pragma: no cover
    _ws_client = None


OPENVLA_SERVER_URL = os.environ.get("OPENVLA_SERVER_URL", "ws://127.0.0.1:8000")
OPENVLA_USE_ACTION_CHUNK = os.environ.get("OPENVLA_USE_ACTION_CHUNK", "1").lower() in {"1", "true", "yes"}
OPENVLA_DEFAULT_CHUNK_SIZE = int(os.environ.get("OPENVLA_DEFAULT_CHUNK_SIZE", "8" if OPENVLA_USE_ACTION_CHUNK else "1"))
OPENVLA_JOINT_INPUT_TYPE = os.environ.get("OPENVLA_JOINT_INPUT_TYPE", "absolute").lower()
OPENVLA_JOINT_CONTROL_DELTA = "true" if OPENVLA_JOINT_INPUT_TYPE == "delta" else "false"

OPENVLA_ROTATE_180 = os.environ.get("OPENVLA_ROTATE_180", "0").lower() in {"1", "true", "yes"}
OPENVLA_FLIP_VERTICAL = os.environ.get("OPENVLA_FLIP_VERTICAL", "0").lower() in {"1", "true", "yes"}
OPENVLA_FLIP_HORIZONTAL = os.environ.get("OPENVLA_FLIP_HORIZONTAL", "0").lower() in {"1", "true", "yes"}

OPENVLA_DEBUG_INPUT_DIR = os.environ.get("OPENVLA_DEBUG_INPUT_DIR")
OPENVLA_DEBUG_INPUT_LIMIT = max(int(os.environ.get("OPENVLA_DEBUG_INPUT_LIMIT", "32")), 0)

_DEBUG_INPUT_COUNT = 0
_WS_CLIENT: Optional["_WebsocketPolicyClient"] = None
_WS_URI: Optional[str] = None


def _pack_array(obj: Any) -> Any:
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


def _unpack_array(obj: Dict[bytes, Any]) -> Any:
    if b"__ndarray__" in obj:
        return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"])
    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])
    return obj


def _pack_msg(data: Dict[str, Any]) -> bytes:
    if msgpack is None:
        raise RuntimeError("msgpack is required for OpenVLA websocket client.")
    return msgpack.packb(data, default=_pack_array)


def _unpack_msg(data: bytes) -> Dict[str, Any]:
    if msgpack is None:
        raise RuntimeError("msgpack is required for OpenVLA websocket client.")
    obj = msgpack.unpackb(data, object_hook=_unpack_array)
    if not isinstance(obj, dict):
        raise TypeError(f"Unexpected websocket payload type: {type(obj)}")
    return obj


class _WebsocketPolicyClient:
    def __init__(self, uri: str):
        if _ws_client is None:
            raise RuntimeError("websockets>=11 is required for OpenVLA websocket client.")
        self._ws = _ws_client.connect(uri, compression=None, max_size=None)
        hello = self._ws.recv()
        self._metadata = _unpack_msg(hello) if isinstance(hello, (bytes, bytearray)) else {}

    @property
    def metadata(self) -> Dict[str, Any]:
        return self._metadata if isinstance(self._metadata, dict) else {}

    def infer(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        self._ws.send(_pack_msg(obs))
        response = self._ws.recv()
        if isinstance(response, str):
            raise RuntimeError(f"OpenVLA server error:\n{response}")
        return _unpack_msg(response)

    def reset(self) -> Dict[str, Any]:
        self._ws.send(_pack_msg({"__command__": "reset"}))
        response = self._ws.recv()
        if isinstance(response, str):
            raise RuntimeError(f"OpenVLA server reset error:\n{response}")
        return _unpack_msg(response)

    def close(self) -> None:
        try:
            self._ws.close()
        except Exception:
            pass


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
    for suffix in ("/health", "/healthz", "/reset", "/act", "/infer"):
        if uri.endswith(suffix):
            return uri[: -len(suffix)]
    return uri


def _get_ws_client(url: str) -> _WebsocketPolicyClient:
    global _WS_CLIENT, _WS_URI
    uri = _normalize_ws_uri(url)
    if _WS_CLIENT is not None and _WS_URI == uri:
        return _WS_CLIENT
    _reset_ws_client()
    _WS_CLIENT = _WebsocketPolicyClient(uri)
    _WS_URI = uri
    return _WS_CLIENT


def _reset_ws_client() -> None:
    global _WS_CLIENT, _WS_URI
    if _WS_CLIENT is not None:
        _WS_CLIENT.close()
    _WS_CLIENT = None
    _WS_URI = None


def _decode_image_from_base64(image_b64: str) -> np.ndarray:
    raw = base64.b64decode(image_b64)
    return np.asarray(Image.open(BytesIO(raw)).convert("RGB"))


def _transform_image(image: np.ndarray) -> np.ndarray:
    out = np.asarray(image)
    if OPENVLA_ROTATE_180:
        out = out[::-1, ::-1]
    if OPENVLA_FLIP_VERTICAL:
        out = np.flipud(out)
    if OPENVLA_FLIP_HORIZONTAL:
        out = np.fliplr(out)
    return np.ascontiguousarray(out)


def _save_debug_image(path: Path, image: np.ndarray) -> None:
    Image.fromarray(np.asarray(image, dtype=np.uint8), mode="RGB").save(path)


def _maybe_dump_inputs(obs: Dict[str, Any], payload: Dict[str, Any]) -> None:
    global _DEBUG_INPUT_COUNT
    if not OPENVLA_DEBUG_INPUT_DIR or _DEBUG_INPUT_COUNT >= OPENVLA_DEBUG_INPUT_LIMIT:
        return

    out_dir = Path(OPENVLA_DEBUG_INPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    episode = int(payload.get("episode", -1))
    step = int(payload.get("step", _DEBUG_INPUT_COUNT))
    prefix = f"ep{episode:06d}_step{step:04d}"

    images = obs.get("images", {}) or {}
    for key, image in images.items():
        suffix = key.split(".")[-1]
        _save_debug_image(out_dir / f"{prefix}_{suffix}.png", np.asarray(image))

    state = obs.get("state")
    if state is not None:
        np.save(out_dir / f"{prefix}_state.npy", np.asarray(state, dtype=np.float32))

    prompt = obs.get("prompt")
    if prompt is not None:
        (out_dir / f"{prefix}_prompt.txt").write_text(str(prompt), encoding="utf-8")

    (out_dir / f"{prefix}_meta.txt").write_text(
        "\n".join(
            [
                f"episode: {episode}",
                f"step: {step}",
                f"chunk_size: {int(obs.get('chunk_size', OPENVLA_DEFAULT_CHUNK_SIZE))}",
                f"rotate_180: {OPENVLA_ROTATE_180}",
                f"flip_vertical: {OPENVLA_FLIP_VERTICAL}",
                f"flip_horizontal: {OPENVLA_FLIP_HORIZONTAL}",
                f"image_keys: {list(images.keys())}",
            ]
        ),
        encoding="utf-8",
    )
    _DEBUG_INPUT_COUNT += 1


def _build_obs_from_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    images = payload.get("images", {}) or {}
    if not isinstance(images, dict):
        raise ValueError("Payload 'images' must be a dict.")

    agent = images.get("observation.images.agentview")
    wrist = images.get("observation.images.wrist") or images.get("observation.images.robot0_eye_in_hand")
    if agent is None:
        raise KeyError("Missing required image 'observation.images.agentview'.")

    obs: Dict[str, Any] = {
        "images": {
            "observation.images.agentview": _transform_image(_decode_image_from_base64(agent)),
        },
        "chunk_size": int(payload.get("chunk_size", OPENVLA_DEFAULT_CHUNK_SIZE)),
    }
    if wrist is not None:
        obs["images"]["observation.images.wrist"] = _transform_image(_decode_image_from_base64(wrist))

    state = payload.get("state")
    if state is not None:
        obs["state"] = np.asarray(state, dtype=np.float32).reshape(-1)

    prompt = payload.get("prompt")
    if prompt is not None:
        obs["prompt"] = str(prompt)

    _maybe_dump_inputs(obs, payload)
    return obs


def _http_json_openvla(
    method: str,
    url: str,
    payload: Optional[Dict[str, Any]] = None,
    timeout_s: float = 10.0,  # noqa: ARG001
) -> Dict[str, Any]:
    method = str(method).upper()

    if method == "GET":
        client = _get_ws_client(url)
        metadata = client.metadata
        image_keys = metadata.get("image_keys")
        if not isinstance(image_keys, list) or not image_keys:
            image_keys = ["observation.images.agentview", "observation.images.wrist"]
        state_dim = metadata.get("state_dim")
        return {
            "image_keys": image_keys,
            "expect_state": bool(metadata.get("expect_state", True)),
            "state_dim": state_dim if isinstance(state_dim, int) else None,
            "action_dim": metadata.get("action_dim"),
            "default_chunk_size": metadata.get("action_horizon"),
            "metadata": metadata,
        }

    if method == "POST":
        if str(url).rstrip("/").endswith("/reset"):
            try:
                return _get_ws_client(url).reset()
            except Exception:
                _reset_ws_client()
                return _get_ws_client(url).reset()

        obs = _build_obs_from_payload(payload or {})
        try:
            result = _get_ws_client(url).infer(obs)
        except Exception:
            _reset_ws_client()
            result = _get_ws_client(url).infer(obs)

        action = np.asarray(result.get("action", result.get("actions", result)), dtype=np.float32)
        if not OPENVLA_USE_ACTION_CHUNK and action.ndim >= 2 and action.shape[0] > 1:
            action = action[:1]

        out = {
            "action": action.tolist(),
            "shape": list(action.shape),
            "action_dim": int(action.shape[-1]) if action.ndim >= 1 else None,
        }
        if "server_timing" in result:
            out["server_timing"] = result["server_timing"]
        if "policy_timing" in result:
            out["policy_timing"] = result["policy_timing"]
        return out

    raise ValueError(f"Unsupported method: {method}")


def _health_url_openvla(server_url: str) -> str:
    return server_url


def _reset_url_openvla(server_url: str) -> str:
    return server_url.rstrip("/") + "/reset"


_orig_build_parser = _act._build_parser
_orig_build_state_vector = _act._build_state_vector


def _build_openvla_state_vector(obs: Dict[str, Any], state_dim: Optional[int]) -> Optional[np.ndarray]:
    joint_pos = obs.get("robot0_joint_pos")
    gripper_qpos = obs.get("robot0_gripper_qpos")

    if joint_pos is None:
        sin_term = obs.get("robot0_joint_pos_sin")
        cos_term = obs.get("robot0_joint_pos_cos")
        if sin_term is not None and cos_term is not None:
            joint_pos = np.arctan2(np.asarray(sin_term), np.asarray(cos_term))

    if joint_pos is not None and gripper_qpos is not None:
        state = np.concatenate(
            [
                np.asarray(joint_pos, dtype=np.float32).reshape(-1),
                np.asarray(gripper_qpos, dtype=np.float32).reshape(-1),
            ],
            axis=0,
        )
        if state_dim is None or state.size == state_dim:
            return state.astype(np.float32)

    for key in ("observation.state", "state"):
        if key not in obs:
            continue
        try:
            state = np.asarray(obs[key], dtype=np.float32).reshape(-1)
        except Exception:
            continue
        if state_dim is None or state.size == state_dim:
            return state

    return _orig_build_state_vector(obs, state_dim)


def _build_parser_openvla() -> Any:
    parser = _orig_build_parser()
    parser.description = "Evaluate OpenVLA websocket policy server on robomimic/robosuite environments."
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="可选任务描述；若提供，则显式透传给 OpenVLA 服务端。",
    )
    parser.set_defaults(
        server_url=OPENVLA_SERVER_URL,
        joint_input_type=OPENVLA_JOINT_INPUT_TYPE,
        joint_control_delta=OPENVLA_JOINT_CONTROL_DELTA,
        test_init_mode="mixed",
        test_random_episodes=10,
        clip_action=False,
        use_action_chunk=OPENVLA_USE_ACTION_CHUNK,
        chunk_size=OPENVLA_DEFAULT_CHUNK_SIZE,
    )
    return parser


def _patch_act_client() -> None:
    _act._http_json = _http_json_openvla
    _act._health_url = _health_url_openvla
    _act._reset_url = _reset_url_openvla
    _act._build_parser = _build_parser_openvla
    _act._build_state_vector = _build_openvla_state_vector


def main() -> None:
    _patch_act_client()
    try:
        _act.main()
    finally:
        _reset_ws_client()


if __name__ == "__main__":
    main()
