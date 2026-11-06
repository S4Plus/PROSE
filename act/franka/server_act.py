#!/usr/bin/env python3
from __future__ import annotations

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
    "/workspace/liushuo/cupid/act/outputs/stack_cube_69_act/checkpoints/100000/pretrained_model"
)


def resolve_checkpoint(path_str: Optional[str]) -> Path:
    root = Path(path_str).expanduser() if path_str else DEFAULT_CHECKPOINT

    if root.is_file():
        return root.parent

    if root.is_dir():
        if (root / "config.json").is_file() and (root / "model.safetensors").is_file():
            return root
        if (root / "pretrained_model").is_dir():
            return root / "pretrained_model"

    raise FileNotFoundError(f"无法从 '{root}' 解析出 pretrained_model 目录。")


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


def rgb_to_tensor(rgb: np.ndarray, width: int, height: int, device: torch.device) -> tuple[np.ndarray, torch.Tensor]:
    resized = cv2.resize(rgb, (width, height))
    tensor = torch.from_numpy(resized).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    return resized, tensor.to(device)


def safe_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in name)


def image_stats(rgb: np.ndarray) -> Dict[str, object]:
    arr = np.asarray(rgb)
    arr01 = arr.astype(np.float32) / 255.0
    return {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "min_0_1": float(arr01.min()) if arr01.size else 0.0,
        "max_0_1": float(arr01.max()) if arr01.size else 0.0,
        "mean_0_1": float(arr01.mean()) if arr01.size else 0.0,
        "std_0_1": float(arr01.std()) if arr01.size else 0.0,
        "channel_mean_0_1": arr01.reshape(-1, arr01.shape[-1]).mean(axis=0).round(6).tolist(),
        "channel_std_0_1": arr01.reshape(-1, arr01.shape[-1]).std(axis=0).round(6).tolist(),
    }


def save_rgb(path: Path, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


class InferenceEngine:
    def __init__(
        self,
        checkpoint: Path,
        device: Optional[str],
        default_chunk_size: Optional[int],
        dump_inputs_dir: Optional[str],
        dump_inputs_every: int,
        dump_inputs_max: int,
    ):
        self.checkpoint = checkpoint
        self.model_cfg = load_model_config(checkpoint)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.dump_inputs_dir = Path(dump_inputs_dir).expanduser().resolve() if dump_inputs_dir else None
        self.dump_inputs_every = max(1, int(dump_inputs_every))
        self.dump_inputs_max = int(dump_inputs_max)
        self.dump_counter = 0
        self.dump_lock = threading.Lock()

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
            if spec.get("type") == "VISUAL":
                shape = spec.get("shape")
                if not isinstance(shape, list) or len(shape) != 3:
                    raise ValueError(f"图像特征 {key} 的 shape 非法: {shape}")
                channels, height, width = [int(x) for x in shape]
                self.image_specs[key] = {
                    "channels": channels,
                    "height": height,
                    "width": width,
                }
            elif key == "observation.state":
                shape = spec.get("shape")
                if not isinstance(shape, list) or len(shape) != 1:
                    raise ValueError(f"状态特征 {key} 的 shape 非法: {shape}")
                self.state_dim = int(shape[0])
            elif key == "observation.environment_state":
                shape = spec.get("shape")
                if not isinstance(shape, list) or len(shape) != 1:
                    raise ValueError(f"环境特征 {key} 的 shape 非法: {shape}")
                self.env_state_dim = int(shape[0])

        action_spec = output_features.get("action")
        if isinstance(action_spec, dict):
            shape = action_spec.get("shape")
            if isinstance(shape, list) and len(shape) == 1:
                self.action_dim = int(shape[0])

        policy = ACTPolicy.from_pretrained(str(checkpoint))
        policy.to(self.device)
        policy.eval()
        policy.reset()
        self.policy = policy
        self.lock = threading.Lock()
        self.state_mean, self.state_std = self._load_state_normalization_stats()

        self.image_keys = list(self.image_specs.keys())
        self.expect_state = self.state_dim is not None
        self.expect_env_state = self.env_state_dim is not None
        self.default_chunk_size = (
            int(default_chunk_size)
            if default_chunk_size is not None
            else int(self.model_cfg.get("n_action_steps", self.model_cfg.get("chunk_size", 1)))
        )

    def _load_state_normalization_stats(self) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        state = self.policy.state_dict()
        mean = state.get("normalize_inputs.buffer_observation_state.mean")
        std = state.get("normalize_inputs.buffer_observation_state.std")
        if mean is None or std is None:
            return None, None
        return (
            mean.detach().cpu().float().numpy(),
            std.detach().cpu().float().numpy(),
        )

    def _next_dump_index(self) -> Optional[int]:
        if self.dump_inputs_dir is None:
            return None
        with self.dump_lock:
            self.dump_counter += 1
            index = self.dump_counter
        if self.dump_inputs_max > 0 and index > self.dump_inputs_max:
            return None
        if (index - 1) % self.dump_inputs_every != 0:
            return None
        return index

    def _state_normalized_z(self, state_arr: Optional[np.ndarray]) -> Optional[list[float]]:
        if state_arr is None or self.state_mean is None or self.state_std is None:
            return None
        std = np.where(np.abs(self.state_std) < 1e-8, 1.0, self.state_std)
        return np.round((state_arr - self.state_mean) / std, 6).tolist()

    def _dump_infer_io(
        self,
        dump_index: int,
        image_debug: Dict[str, Dict[str, object]],
        state_arr: Optional[np.ndarray],
        env_state_arr: Optional[np.ndarray],
        action_chunk: np.ndarray,
        chunk_size: int,
    ) -> None:
        if self.dump_inputs_dir is None:
            return

        dump_dir = self.dump_inputs_dir / f"infer_{dump_index:06d}"
        dump_dir.mkdir(parents=True, exist_ok=True)

        image_summaries: Dict[str, object] = {}
        for key, values in image_debug.items():
            raw_rgb = values["raw_rgb"]
            resized_rgb = values["resized_rgb"]
            assert isinstance(raw_rgb, np.ndarray)
            assert isinstance(resized_rgb, np.ndarray)
            image_key = safe_name(key)
            raw_path = dump_dir / f"{image_key}.raw.png"
            resized_path = dump_dir / f"{image_key}.resized.png"
            save_rgb(raw_path, raw_rgb)
            save_rgb(resized_path, resized_rgb)
            image_summaries[key] = {
                "raw_png": raw_path.name,
                "resized_png": resized_path.name,
                "raw": image_stats(raw_rgb),
                "resized": image_stats(resized_rgb),
            }

        state_payload = None
        if state_arr is not None:
            state_payload = {
                "values": np.round(state_arr, 8).tolist(),
                "min": float(state_arr.min()),
                "max": float(state_arr.max()),
                "mean": float(state_arr.mean()),
                "normalized_z": self._state_normalized_z(state_arr),
            }

        action_payload = {
            "shape": list(action_chunk.shape),
            "first_action": np.round(action_chunk[0], 8).tolist() if len(action_chunk) > 0 else [],
            "values": np.round(action_chunk, 8).tolist(),
        }
        if action_chunk.ndim == 2 and action_chunk.shape[1] >= 8:
            gripper = np.asarray(action_chunk[:, 7], dtype=np.float32)
            action_payload["gripper_dim7"] = {
                "values": np.round(gripper, 8).tolist(),
                "min": float(gripper.min()),
                "max": float(gripper.max()),
                "mean": float(gripper.mean()),
            }

        payload = {
            "dump_index": dump_index,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "checkpoint": str(self.checkpoint),
            "device": str(self.device),
            "chunk_size": chunk_size,
            "image_keys": self.image_keys,
            "images": image_summaries,
            "state": state_payload,
            "env_state": None if env_state_arr is None else np.round(env_state_arr, 8).tolist(),
            "action": action_payload,
        }
        (dump_dir / "infer_io.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"[ACT Server] dumped infer input/output to {dump_dir}")

    def reset(self) -> None:
        with self.lock:
            self.policy.reset()

    def infer(self, payload: Dict[str, object]) -> Dict[str, object]:
        images = payload.get("images")
        if not isinstance(images, dict):
            raise ValueError("payload 必须包含 images 字段。")

        missing = [key for key in self.image_keys if key not in images]
        if missing:
            raise ValueError(f"缺少图像输入: {missing}")

        batch: Dict[str, torch.Tensor] = {}
        image_debug: Dict[str, Dict[str, object]] = {}
        for key in self.image_keys:
            spec = self.image_specs[key]
            raw_rgb = decode_image_rgb(str(images[key]))
            resized_rgb, tensor = rgb_to_tensor(
                rgb=raw_rgb,
                width=spec["width"],
                height=spec["height"],
                device=self.device,
            )
            batch[key] = tensor
            image_debug[key] = {
                "raw_rgb": raw_rgb,
                "resized_rgb": resized_rgb,
            }

        state = payload.get("state")
        state_arr: Optional[np.ndarray] = None
        if self.expect_state:
            if state is None:
                raise ValueError("当前模型要求提供 state。")
            state_arr = np.asarray(state, dtype=np.float32)
            if state_arr.ndim != 1 or self.state_dim is None or state_arr.shape[0] != self.state_dim:
                raise ValueError(f"state 维度错误，期望 {(self.state_dim,)}, 实际 {tuple(state_arr.shape)}")
            batch["observation.state"] = torch.from_numpy(state_arr).unsqueeze(0).to(self.device)

        env_state = payload.get("env_state")
        env_state_arr: Optional[np.ndarray] = None
        if self.expect_env_state:
            if env_state is None:
                if self.env_state_dim == 1:
                    env_state_arr = np.ones((1,), dtype=np.float32)
                else:
                    raise ValueError("当前模型要求提供 env_state。")
            else:
                env_state_arr = np.asarray(env_state, dtype=np.float32)
                if env_state_arr.ndim != 1 or self.env_state_dim is None or env_state_arr.shape[0] != self.env_state_dim:
                    raise ValueError(
                        f"env_state 维度错误，期望 {(self.env_state_dim,)}, 实际 {tuple(env_state_arr.shape)}"
                    )
            batch["observation.environment_state"] = (
                torch.from_numpy(env_state_arr).unsqueeze(0).to(self.device)
            )

        chunk_size = payload.get("chunk_size", self.default_chunk_size)
        if not isinstance(chunk_size, int) or chunk_size <= 0:
            raise ValueError("chunk_size 必须是正整数。")

        with self.lock:
            with torch.no_grad():
                action_chunk = self.policy.predict_action_chunk(batch).squeeze(0)

        action_chunk = action_chunk[:chunk_size].detach().cpu().float().numpy()
        if self.action_dim is not None and action_chunk.ndim == 2 and action_chunk.shape[1] != self.action_dim:
            raise ValueError(
                f"模型输出维度错误，期望 action_dim={self.action_dim}，实际 {action_chunk.shape}"
            )

        print(
            "[ACT Server] action chunk: "
            f"shape={action_chunk.shape}, "
            f"first_action={np.round(action_chunk[0], 6).tolist() if len(action_chunk) > 0 else []}"
        )
        if action_chunk.ndim == 2 and action_chunk.shape[1] >= 8:
            gripper_chunk = np.asarray(action_chunk[:, 7], dtype=np.float32)
            print(
                "[ACT Server] gripper chunk: "
                f"all={np.round(gripper_chunk, 6).tolist()}, "
                f"min={float(gripper_chunk.min()):.6f}, "
                f"max={float(gripper_chunk.max()):.6f}, "
                f"mean={float(gripper_chunk.mean()):.6f}"
            )

        dump_index = self._next_dump_index()
        if dump_index is not None:
            self._dump_infer_io(
                dump_index=dump_index,
                image_debug=image_debug,
                state_arr=state_arr,
                env_state_arr=env_state_arr,
                action_chunk=action_chunk,
                chunk_size=chunk_size,
            )

        return {
            "action": action_chunk.tolist(),
            "shape": list(action_chunk.shape),
            "action_dim": self.action_dim,
        }


class ACTHTTPServer(ThreadingHTTPServer):
    def __init__(self, server_address, handler_cls, engine: InferenceEngine):
        super().__init__(server_address, handler_cls)
        self.engine = engine


class RequestHandler(BaseHTTPRequestHandler):
    server_version = "FrankaACTServer/0.1"

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
                "repo_id": engine.model_cfg.get("repo_id"),
                "device": str(engine.device),
                "image_keys": engine.image_keys,
                "image_specs": engine.image_specs,
                "expect_state": engine.expect_state,
                "state_dim": engine.state_dim,
                "expect_env_state": engine.expect_env_state,
                "env_state_dim": engine.env_state_dim,
                "action_dim": engine.action_dim,
                "default_chunk_size": engine.default_chunk_size,
                "n_action_steps": engine.model_cfg.get("n_action_steps"),
                "dump_inputs_dir": str(engine.dump_inputs_dir) if engine.dump_inputs_dir else None,
                "dump_inputs_every": engine.dump_inputs_every,
                "dump_inputs_max": engine.dump_inputs_max,
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
    parser = argparse.ArgumentParser(description="Franka ACT HTTP inference server")
    parser.add_argument("--host", default="0.0.0.0", help="监听地址")
    parser.add_argument("--port", type=int, default=8002, help="监听端口")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(DEFAULT_CHECKPOINT),
        help="pretrained_model 路径，默认使用 stack_cube_mh_act/100000",
    )
    parser.add_argument("--device", default=None, help="Torch device，例如 cuda:0")
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="默认返回动作步数；不传时与训练配置 n_action_steps 保持一致",
    )
    parser.add_argument(
        "--dump-inputs-dir",
        type=str,
        default=None,
        help="可选：保存 /infer 收到的 state、图像和模型输出，便于排查输入尺度/相机错误",
    )
    parser.add_argument(
        "--dump-inputs-every",
        type=int,
        default=1,
        help="启用 dump 时每隔多少次 /infer 保存一次，默认每次都保存",
    )
    parser.add_argument(
        "--dump-inputs-max",
        type=int,
        default=20,
        help="启用 dump 时最多保存多少次；<=0 表示不限制",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = resolve_checkpoint(args.checkpoint)
    engine = InferenceEngine(
        checkpoint=checkpoint,
        device=args.device,
        default_chunk_size=args.chunk_size,
        dump_inputs_dir=args.dump_inputs_dir,
        dump_inputs_every=args.dump_inputs_every,
        dump_inputs_max=args.dump_inputs_max,
    )
    server = ACTHTTPServer((args.host, args.port), RequestHandler, engine)
    print(
        f"Franka ACT server listening on http://{args.host}:{args.port} "
        f"(checkpoint={checkpoint}, images={engine.image_specs}, "
        f"state_dim={engine.state_dim}, action_dim={engine.action_dim}, "
        f"default_chunk={engine.default_chunk_size})"
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
