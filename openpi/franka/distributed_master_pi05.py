#!/usr/bin/env python3
"""三机双臂 OpenPI Master。

该脚本运行在 Server / Master / Policy 主机上。它从左右 Host Agent 同步拉取观测，
构造与 `pi05_right_box_left_object_right_lid_lora` 对齐的三相机 16 维 state，然后调用
本机 OpenPI policy websocket，最后把左右 action chunk 分发回各自控制主机。
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
import pickle
import select
import signal
import sys
import time
from typing import Any

import einops
import numpy as np
from openpi_client import msgpack_numpy
import websockets.sync.client

DEFAULT_PROMPT = "Use the right arm to stage the box and lid, the left arm to place the object, then close the lid."
DEFAULT_ROLLOUT_DIR = Path(__file__).resolve().parents[1] / "Dataset" / "dual_host_pi05_rollouts"
ACTION_DIM = 16
GRIPPER_THRESHOLD = 0.5
GRIPPER_OPEN_COMMAND = 0
GRIPPER_CLOSE_COMMAND = 1
ACTION_ORDER = (
    "left_joint_1",
    "left_joint_2",
    "left_joint_3",
    "left_joint_4",
    "left_joint_5",
    "left_joint_6",
    "left_joint_7",
    "left_gripper_command",
    "right_joint_1",
    "right_joint_2",
    "right_joint_3",
    "right_joint_4",
    "right_joint_5",
    "right_joint_6",
    "right_joint_7",
    "right_gripper_command",
)


def _uri(host: str, port: int | None) -> str:
    result = host if host.startswith("ws") else f"ws://{host}"
    if port is not None:
        result += f":{port}"
    return result


def _gripper_command(value: float) -> int:
    return GRIPPER_CLOSE_COMMAND if float(value) >= GRIPPER_THRESHOLD else GRIPPER_OPEN_COMMAND


def _as_record_image(image: np.ndarray) -> np.ndarray:
    """把 Host 返回的 policy image 兜底转为 HWC uint8，用于 pkl 记录。"""
    img = np.asarray(image)
    if np.issubdtype(img.dtype, np.floating):
        img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    if img.ndim == 3 and img.shape[0] in (1, 3):
        img = einops.rearrange(img, "c h w -> h w c")
    return img.astype(np.uint8, copy=False)


class MsgpackWebsocketClient:
    def __init__(self, host: str, port: int, *, timeout_s: float | None, name: str):
        self.name = name
        self.uri = _uri(host, port)
        self.timeout_s = timeout_s
        self._packer = msgpack_numpy.Packer()
        self._ws = websockets.sync.client.connect(
            self.uri,
            compression=None,
            max_size=None,
            open_timeout=timeout_s,
            proxy=None,
        )
        metadata = self._ws.recv(timeout=timeout_s)
        if isinstance(metadata, str):
            raise RuntimeError(f"{name} startup error:\n{metadata}")
        self.metadata = msgpack_numpy.unpackb(metadata)

    def request(self, payload: dict[str, Any]) -> dict[str, Any]:
        self._ws.send(self._packer.pack(payload))
        response = self._ws.recv(timeout=self.timeout_s)
        if isinstance(response, str):
            raise RuntimeError(f"{self.name} returned text error:\n{response}")
        result = msgpack_numpy.unpackb(response)
        if isinstance(result, dict) and result.get("ok") is False:
            raise RuntimeError(f"{self.name} request failed:\n{result.get('error')}")
        return result

    def close(self) -> None:
        self._ws.close()


class PolicyClient:
    def __init__(self, host: str, port: int, *, timeout_s: float | None):
        self.uri = _uri(host, port)
        self.timeout_s = timeout_s
        self._packer = msgpack_numpy.Packer()
        self._ws = websockets.sync.client.connect(
            self.uri,
            compression=None,
            max_size=None,
            open_timeout=timeout_s,
            proxy=None,
        )
        metadata = self._ws.recv(timeout=timeout_s)
        if isinstance(metadata, str):
            raise RuntimeError(f"Policy startup error:\n{metadata}")
        self.metadata = msgpack_numpy.unpackb(metadata)

    def infer(self, observation: dict[str, Any]) -> dict[str, Any]:
        self._ws.send(self._packer.pack(observation))
        response = self._ws.recv(timeout=self.timeout_s)
        if isinstance(response, str):
            raise RuntimeError(f"Policy inference error:\n{response}")
        return msgpack_numpy.unpackb(response)

    def close(self) -> None:
        self._ws.close()


@dataclass
class EpisodeStep:
    state: np.ndarray
    images: dict[str, np.ndarray]
    action: np.ndarray
    raw_action: np.ndarray
    action_chunk: np.ndarray
    latency_ms: float
    left_timestamp: float
    right_timestamp: float


class PklEpisodeRecorder:
    def __init__(self, rollout_dir: Path, prompt: str):
        self.rollout_dir = rollout_dir
        self.prompt = prompt
        self.rollout_dir.mkdir(parents=True, exist_ok=True)
        self.next_episode_idx = self._discover_next_episode_idx()
        self.steps: list[EpisodeStep] = []

    def _discover_next_episode_idx(self) -> int:
        max_idx = -1
        for path in self.rollout_dir.glob("episode_*.pkl"):
            parts = path.stem.split("_")
            if len(parts) >= 2 and parts[0] == "episode" and parts[1].isdigit():
                max_idx = max(max_idx, int(parts[1]))
        return max_idx + 1

    def record_step(
        self,
        *,
        state: np.ndarray,
        images: dict[str, np.ndarray],
        action: np.ndarray,
        raw_action: np.ndarray,
        action_chunk: np.ndarray,
        latency_ms: float,
        left_timestamp: float,
        right_timestamp: float,
    ) -> None:
        self.steps.append(
            EpisodeStep(
                state=np.asarray(state, dtype=np.float32).copy(),
                images={name: np.asarray(image, dtype=np.uint8).copy() for name, image in images.items()},
                action=np.asarray(action, dtype=np.float32).copy(),
                raw_action=np.asarray(raw_action, dtype=np.float32).copy(),
                action_chunk=np.asarray(action_chunk, dtype=np.float32).copy(),
                latency_ms=float(latency_ms),
                left_timestamp=float(left_timestamp),
                right_timestamp=float(right_timestamp),
            )
        )

    @property
    def num_steps(self) -> int:
        return len(self.steps)

    def finalize(self, *, success: bool, reason: str, metadata: dict[str, Any]) -> Path | None:
        if not self.steps:
            print("[Recorder] empty episode, skip pkl save.")
            return None

        episode_idx = self.next_episode_idx
        self.next_episode_idx += 1
        actions = np.stack([step.action for step in self.steps], axis=0).astype(np.float32)
        rewards = np.zeros((actions.shape[0],), dtype=np.float32)
        dones = np.zeros((actions.shape[0],), dtype=bool)
        successes = np.zeros((actions.shape[0],), dtype=bool)
        rewards[-1] = float(success)
        dones[-1] = True
        successes[-1] = bool(success)

        image_names = sorted({name for step in self.steps for name in step.images})
        images = {
            name: np.stack([step.images[name] for step in self.steps if name in step.images], axis=0)
            for name in image_names
        }
        payload = {
            "observations": {
                "state": np.stack([step.state for step in self.steps], axis=0).astype(np.float32),
                "images": images,
            },
            "actions": actions,
            "reward": float(success),
            "rewards": rewards,
            "done": dones,
            "success": bool(success),
            "successes": successes,
            "metadata": {
                "source": "dual_host_pi05_master",
                "episode": episode_idx,
                "reason": reason,
                "prompt": self.prompt,
                "raw_actions": np.stack([step.raw_action for step in self.steps], axis=0).astype(np.float32),
                "action_chunks": [step.action_chunk for step in self.steps],
                "latencies_ms": np.asarray([step.latency_ms for step in self.steps], dtype=np.float32),
                "left_host_timestamps": np.asarray([step.left_timestamp for step in self.steps], dtype=np.float64),
                "right_host_timestamps": np.asarray([step.right_timestamp for step in self.steps], dtype=np.float64),
                "action_order": ACTION_ORDER,
                "gripper_convention": "0=open, 1=close",
                **metadata,
            },
        }
        suffix = "success" if success else "failure"
        out_path = self.rollout_dir / f"episode_{episode_idx:06d}_{suffix}.pkl"
        with out_path.open("wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        self.steps.clear()
        print(f"[Recorder] saved pkl: {out_path}")
        return out_path


class DistributedDualArmMaster:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        timeout_s = None if args.timeout <= 0.0 else float(args.timeout)
        print(f"[Connect] right host ws://{args.right_host}:{args.right_port}")
        self.right = MsgpackWebsocketClient(args.right_host, args.right_port, timeout_s=timeout_s, name="right_host")
        print(f"[Connect] right metadata={self.right.metadata}")
        print(f"[Connect] left host ws://{args.left_host}:{args.left_port}")
        self.left = MsgpackWebsocketClient(args.left_host, args.left_port, timeout_s=timeout_s, name="left_host")
        print(f"[Connect] left metadata={self.left.metadata}")
        print(f"[Connect] policy ws://{args.policy_host}:{args.policy_port}")
        self.policy = PolicyClient(args.policy_host, args.policy_port, timeout_s=timeout_s)
        print(f"[Connect] policy metadata={self.policy.metadata}")
        self.recorder = None
        if args.record:
            self.recorder = PklEpisodeRecorder(Path(args.rollout_dir).expanduser().resolve(), args.prompt)
            print(f"[Recorder] pkl dir: {self.recorder.rollout_dir}")
        self._episode_end_requested = False

    def healthcheck(self) -> None:
        for name, client in (("right", self.right), ("left", self.left)):
            reply = client.request({"type": "ping"})
            print(f"[Health] {name}: {reply}")
        obs = self._collect_observations()
        observation, _record_images = self._build_policy_observation(obs["left"], obs["right"])
        print(
            "[Health] obs ok: state_shape=%s cameras=%s"
            % (np.asarray(observation["observation.state"]).shape, sorted(observation["observation"]["images"]))
        )

    def _collect_observations(self) -> dict[str, dict[str, Any]]:
        with ThreadPoolExecutor(max_workers=2) as executor:
            left_future = executor.submit(self.left.request, {"type": "get_obs"})
            right_future = executor.submit(self.right.request, {"type": "get_obs"})
            left_obs = left_future.result()
            right_obs = right_future.result()
        return {"left": left_obs, "right": right_obs}

    def _build_policy_observation(
        self,
        left_obs: dict[str, Any],
        right_obs: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
        left_state = np.asarray(left_obs["state"], dtype=np.float32).reshape(-1)
        right_state = np.asarray(right_obs["state"], dtype=np.float32).reshape(-1)
        if left_state.shape[0] != 8 or right_state.shape[0] != 8:
            raise ValueError(f"Host state must be 8D each, got left={left_state.shape}, right={right_state.shape}.")
        state = np.asarray([*left_state.tolist(), *right_state.tolist()], dtype=np.float32)

        left_images = left_obs["images"]
        right_images = right_obs["images"]
        missing = []
        if "left_wrist" not in left_images:
            missing.append("left host image left_wrist")
        if "right_wrist" not in right_images:
            missing.append("right host image right_wrist")
        if "agent_view" not in right_images:
            missing.append("right host image agent_view")
        if missing:
            raise KeyError("missing required distributed observation fields: " + ", ".join(missing))

        policy_images = {
            "agent_view": right_images["agent_view"],
            "right_wrist": right_images["right_wrist"],
            "left_wrist": left_images["left_wrist"],
        }
        left_record = left_obs.get("record_images") or {}
        right_record = right_obs.get("record_images") or {}
        record_images = {
            "agent_view": _as_record_image(right_record.get("agent_view", policy_images["agent_view"])),
            "right_wrist": _as_record_image(right_record.get("right_wrist", policy_images["right_wrist"])),
            "left_wrist": _as_record_image(left_record.get("left_wrist", policy_images["left_wrist"])),
        }
        observation = {
            "observation": {
                "images": policy_images,
                "state": state,
            },
            "observation.images.agent_view": policy_images["agent_view"],
            "observation.images.right_wrist": policy_images["right_wrist"],
            "observation.images.left_wrist": policy_images["left_wrist"],
            "observation.state": state,
            "observation/state": state,
            "images": policy_images,
            "state": state,
            "prompt": self.args.prompt,
        }
        return observation, record_images

    def _normalize_action_chunk(self, action_chunk: np.ndarray, exec_steps: int) -> tuple[np.ndarray, np.ndarray]:
        raw = np.asarray(action_chunk[:exec_steps, :ACTION_DIM], dtype=np.float32).copy()
        commanded = raw.copy()
        commanded[:, 7] = [_gripper_command(value) for value in raw[:, 7]]
        commanded[:, 15] = [_gripper_command(value) for value in raw[:, 15]]
        return raw, commanded.astype(np.float32)

    def _execute_both(self, commanded: np.ndarray) -> tuple[dict[str, Any], dict[str, Any]]:
        execute_at = time.time() + float(self.args.execute_lead_s)
        left_payload = {
            "type": "execute_chunk",
            "joint_chunk": commanded[:, :7],
            "gripper_commands": commanded[:, 7],
            "action_hz": float(self.args.action_hz),
            "execute_at": execute_at,
        }
        right_payload = {
            "type": "execute_chunk",
            "joint_chunk": commanded[:, 8:15],
            "gripper_commands": commanded[:, 15],
            "action_hz": float(self.args.action_hz),
            "execute_at": execute_at,
        }
        with ThreadPoolExecutor(max_workers=2) as executor:
            left_future = executor.submit(self.left.request, left_payload)
            right_future = executor.submit(self.right.request, right_payload)
            left_ack = left_future.result()
            right_ack = right_future.result()
        return left_ack, right_ack

    def _poll_enter_to_end_episode(self) -> None:
        if not sys.stdin.isatty():
            return
        try:
            ready, _, _ = select.select([sys.stdin], [], [], 0.0)
        except Exception:
            return
        if ready and sys.stdin.readline().rstrip("\r\n") == "":
            print("[Episode] Enter pressed, ending current inference episode.")
            self._episode_end_requested = True

    def run_episode(self) -> None:
        executed_count = 0
        self._episode_end_requested = False
        print(
            f"[Episode] start: max_actions={self.args.max_actions}, "
            f"chunk_exec_steps={self.args.chunk_exec_steps}, execute={self.args.execute}"
        )
        print("[Episode] Press Enter to stop and save pkl.")
        while executed_count < self.args.max_actions and not self._episode_end_requested:
            self._poll_enter_to_end_episode()
            obs_pair = self._collect_observations()
            policy_observation, record_images = self._build_policy_observation(obs_pair["left"], obs_pair["right"])

            t0 = time.time()
            result = self.policy.infer(policy_observation)
            action_chunk = np.asarray(result["actions"], dtype=np.float32)
            latency_ms = (time.time() - t0) * 1000.0
            if action_chunk.ndim != 2 or action_chunk.shape[1] < ACTION_DIM:
                print(f"[Inference] expected action shape (T, >=16), got {action_chunk.shape}; skipped.")
                continue

            remaining = int(self.args.max_actions) - executed_count
            exec_steps = min(int(action_chunk.shape[0]), remaining, int(self.args.chunk_exec_steps))
            if exec_steps <= 0:
                break
            raw, commanded = self._normalize_action_chunk(action_chunk, exec_steps)
            print(
                f"[Inference] latency={latency_ms:.1f}ms chunk={action_chunk.shape} exec_steps={exec_steps} "
                f"first={np.array2string(commanded[0], precision=4)}"
            )

            left_ack: dict[str, Any] = {}
            right_ack: dict[str, Any] = {}
            if self.args.execute:
                left_ack, right_ack = self._execute_both(commanded)
                print(
                    "[Exec] left_steps=%s right_steps=%s left_start=%.6f right_start=%.6f"
                    % (
                        left_ack.get("steps"),
                        right_ack.get("steps"),
                        float(left_ack.get("started_at", 0.0)),
                        float(right_ack.get("started_at", 0.0)),
                    )
                )

            if self.recorder is not None:
                state = np.asarray(policy_observation["observation.state"], dtype=np.float32)
                left_ts = float(obs_pair["left"].get("timestamp", 0.0))
                right_ts = float(obs_pair["right"].get("timestamp", 0.0))
                for idx in range(exec_steps):
                    self.recorder.record_step(
                        state=state,
                        images=record_images,
                        action=commanded[idx],
                        raw_action=raw[idx],
                        action_chunk=action_chunk,
                        latency_ms=latency_ms,
                        left_timestamp=left_ts,
                        right_timestamp=right_ts,
                    )
            executed_count += exec_steps
            print(f"[Episode] collected_steps={executed_count}")

        self._finalize_episode(reason="manual_enter" if self._episode_end_requested else "episode_end")

    def _resolve_success_label(self) -> bool:
        if self.args.success_label is not None:
            return bool(int(self.args.success_label))
        if not sys.stdin.isatty():
            print("[Recorder] non-interactive shell; default success=False")
            return False
        while True:
            raw = input("[Recorder] result label 0=failure, 1=success: ").strip()
            if raw in ("0", "1"):
                return raw == "1"
            print("[Recorder] invalid label, input 0 or 1.")

    def _finalize_episode(self, *, reason: str) -> None:
        if self.recorder is None:
            return
        success = self._resolve_success_label()
        self.recorder.finalize(
            success=success,
            reason=reason,
            metadata={
                "execute": bool(self.args.execute),
                "policy_host": self.args.policy_host,
                "policy_port": int(self.args.policy_port),
                "right_host": self.args.right_host,
                "right_port": int(self.args.right_port),
                "left_host": self.args.left_host,
                "left_port": int(self.args.left_port),
                "action_hz": float(self.args.action_hz),
                "chunk_exec_steps": int(self.args.chunk_exec_steps),
                "execute_lead_s": float(self.args.execute_lead_s),
            },
        )

    def close(self) -> None:
        old_sigint_handler = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        try:
            self.left.close()
            self.right.close()
            self.policy.close()
        finally:
            signal.signal(signal.SIGINT, old_sigint_handler)
            print("[Master] closed.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Distributed dual-arm OpenPI PI05 master.")
    parser.add_argument("--policy-host", default="127.0.0.1")
    parser.add_argument("--policy-port", type=int, default=8000)
    parser.add_argument("--right-host", default="10.0.54.248")
    parser.add_argument("--right-port", type=int, default=9101)
    parser.add_argument("--left-host", default="10.0.54.196")
    parser.add_argument("--left-port", type=int, default=9102)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--healthcheck", action="store_true", help="只检查 policy/host/观测字段，不执行推理动作。")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--rollout-dir", default=str(DEFAULT_ROLLOUT_DIR))
    parser.add_argument("--record", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-actions", type=int, default=200)
    parser.add_argument("--chunk-exec-steps", type=int, default=32)
    parser.add_argument("--action-hz", type=float, default=30.0)
    parser.add_argument(
        "--execute-lead-s",
        type=float,
        default=0.35,
        help="Master 下发 action 后，左右 host 等待到同一个 epoch time 再开始执行。要求三台机器已做时间同步。",
    )
    parser.add_argument("--execute", dest="execute", action="store_true")
    parser.add_argument("--no-execute", dest="execute", action="store_false")
    parser.set_defaults(execute=True)
    parser.add_argument("--success-label", choices=("0", "1"), default=None)
    args = parser.parse_args()
    if args.max_actions < 1:
        parser.error("--max-actions must be >= 1")
    if args.chunk_exec_steps < 1:
        parser.error("--chunk-exec-steps must be >= 1")
    if args.action_hz <= 0.0:
        parser.error("--action-hz must be > 0")
    if args.timeout < 0.0:
        parser.error("--timeout must be >= 0")
    if args.execute_lead_s < 0.0:
        parser.error("--execute-lead-s must be >= 0")
    return args


def main() -> int:
    args = parse_args()
    master = DistributedDualArmMaster(args)
    try:
        if args.healthcheck:
            master.healthcheck()
            return 0
        master.run_episode()
    except KeyboardInterrupt:
        print("\n[Master] interrupted.")
        master._finalize_episode(reason="keyboard_interrupt")
        return 130
    finally:
        master.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
