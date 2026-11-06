#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pickle
import select
import signal
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import websockets.sync.client
from openpi_client import msgpack_numpy


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "default_franka3.yaml"
DEFAULT_ROLLOUT_DIR = ROOT / "Dataset" / "dual_arm_model_inference_rollouts"
DEFAULT_CAMERAS = "left_wrist,right_wrist,agent_view"
IMG_W = 224
IMG_H = 224
CAM_FRAME_TIMEOUT_MS = 1000
CAM_START_RETRIES = 5
CAM_START_RETRY_DELAY_S = 0.5
GRIPPER_THRESHOLD = 0.5
GRIPPER_WIDTH_THRESHOLD = 0.04
GRIPPER_OPEN_COMMAND = 0
GRIPPER_CLOSE_COMMAND = 1
ACTION_DIM = 16
FRANKA_JOINT_LIMITS = np.asarray(
    [
        [-2.8973, 2.8973],
        [-1.7628, 1.7628],
        [-2.8973, 2.8973],
        [-3.0718, -0.0698],
        [-2.8973, 2.8973],
        [-0.0175, 3.7525],
        [-2.8973, 2.8973],
    ],
    dtype=np.float64,
)
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

rs = None
Gripper = None
JointWaypoint = None
JointWaypointMotion = None
RelativeDynamicsFactor = None
Robot = None


def _load_realsense_module():
    global rs
    if rs is None:
        import pyrealsense2 as _rs

        rs = _rs
    return rs


def _load_franky_symbols() -> tuple[Any, Any, Any, Any, Any]:
    global Gripper, JointWaypoint, JointWaypointMotion, RelativeDynamicsFactor, Robot
    if Robot is None:
        from franky import (
            Gripper as _Gripper,
            JointWaypoint as _JointWaypoint,
            JointWaypointMotion as _JointWaypointMotion,
            RelativeDynamicsFactor as _RelativeDynamicsFactor,
            Robot as _Robot,
        )

        Gripper = _Gripper
        JointWaypoint = _JointWaypoint
        JointWaypointMotion = _JointWaypointMotion
        RelativeDynamicsFactor = _RelativeDynamicsFactor
        Robot = _Robot
    return Gripper, JointWaypoint, JointWaypointMotion, RelativeDynamicsFactor, Robot


@dataclass(frozen=True)
class ArmConfig:
    name: str
    robot_ip: str
    home_joint_values: list[float]
    gripper_open_width: float
    gripper_close_width: float
    gripper_speed: float
    gripper_force: float


@dataclass(frozen=True)
class CameraConfig:
    name: str
    serial: str
    width: int
    height: int
    fps: int


@dataclass(frozen=True)
class ExecutionConfig:
    joint_vel: float
    joint_acc: float
    relative_velocity: float
    relative_acceleration: float
    relative_jerk: float


def _load_yaml(path: str | Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to read dual_arm_workflow/default_franka3.yaml") from exc
    with Path(path).expanduser().resolve().open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a YAML mapping: {path}")
    return data


def _float_list(values: Any, length: int, name: str) -> list[float]:
    if not isinstance(values, list) or len(values) != length:
        raise ValueError(f"{name} must be a list of {length} numbers.")
    return [float(value) for value in values]


def _load_arm_config(config: dict[str, Any], side: str, *, ip_override: str | None) -> ArmConfig:
    arm_cfg = (config.get("arms") or {}).get(side) or {}
    if not isinstance(arm_cfg, dict):
        raise ValueError(f"arms.{side} must be a mapping.")
    return ArmConfig(
        name=str(arm_cfg.get("name") or f"{side}_arm"),
        robot_ip=str(ip_override or arm_cfg.get("fci_ip") or arm_cfg.get("robot_ip")),
        home_joint_values=_float_list(arm_cfg.get("home_joint_values"), 7, f"arms.{side}.home_joint_values"),
        gripper_open_width=float(arm_cfg.get("gripper_open_width", 0.08)),
        gripper_close_width=float(arm_cfg.get("gripper_close_width", 0.0)),
        gripper_speed=float(arm_cfg.get("gripper_open_speed", 0.10)),
        gripper_force=float(arm_cfg.get("gripper_force", 5.0)),
    )


def _load_camera_configs(config: dict[str, Any], camera_names: list[str], fps_override: int | None) -> dict[str, CameraConfig]:
    cameras_cfg = config.get("cameras") or {}
    if not isinstance(cameras_cfg, dict):
        raise ValueError("cameras must be a mapping.")
    result: dict[str, CameraConfig] = {}
    for name in camera_names:
        raw = cameras_cfg.get(name)
        if not isinstance(raw, dict):
            raise ValueError(f"Camera '{name}' not found in config.")
        result[name] = CameraConfig(
            name=name,
            serial=str(raw["serial"]),
            width=int(raw.get("width", 640)),
            height=int(raw.get("height", 480)),
            fps=int(fps_override or raw.get("fps", 30)),
        )
    return result


def _parse_names(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def _gripper_command(value: float) -> int:
    return GRIPPER_CLOSE_COMMAND if float(value) >= GRIPPER_THRESHOLD else GRIPPER_OPEN_COMMAND


def _gripper_command_from_width(width: float, threshold: float) -> int:
    return GRIPPER_OPEN_COMMAND if float(width) >= threshold else GRIPPER_CLOSE_COMMAND


def _policy_image_from_bgr(frame: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)
    return np.transpose(rgb, (2, 0, 1)).astype(np.float32) / 255.0


def _record_image_from_bgr(frame: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.uint8, copy=False)


def _policy_uri(host: str, port: int | None) -> str:
    uri = host if host.startswith("ws") else f"ws://{host}"
    if port is not None:
        uri += f":{port}"
    return uri


class TimedWebsocketClientPolicy:
    def __init__(self, host: str, port: int | None, timeout_s: float | None):
        self._uri = _policy_uri(host, port)
        self._timeout_s = timeout_s
        self._packer = msgpack_numpy.Packer()
        self._ws = websockets.sync.client.connect(
            self._uri,
            compression=None,
            max_size=None,
            open_timeout=timeout_s,
            proxy=None,
        )
        metadata = self._ws.recv(timeout=timeout_s)
        if isinstance(metadata, str):
            raise RuntimeError(f"Error from inference server during startup:\n{metadata}")
        self._server_metadata = msgpack_numpy.unpackb(metadata)

    def get_server_metadata(self) -> dict[str, Any]:
        return self._server_metadata

    def infer(self, obs: dict[str, Any]) -> dict[str, Any]:
        self._ws.send(self._packer.pack(obs))
        response = self._ws.recv(timeout=self._timeout_s)
        if isinstance(response, str):
            raise RuntimeError(f"Error in inference server:\n{response}")
        return msgpack_numpy.unpackb(response)

    def close(self) -> None:
        self._ws.close()


def _connect_policy_from_args(args: argparse.Namespace) -> TimedWebsocketClientPolicy:
    policy_timeout_s = None if args.policy_timeout <= 0.0 else float(args.policy_timeout)
    timeout_desc = "disabled" if policy_timeout_s is None else f"{policy_timeout_s:.1f}s"
    print(
        f"[Policy] connecting pi service: {args.policy_host}:{args.policy_port} "
        f"(timeout={timeout_desc}) ..."
    )
    try:
        policy = TimedWebsocketClientPolicy(
            host=args.policy_host,
            port=args.policy_port,
            timeout_s=policy_timeout_s,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to connect policy service at {args.policy_host}:{args.policy_port} "
            f"within {timeout_desc}. Check that the OpenPI server is running and reachable, "
            "or pass --policy-host/--policy-port. If this client runs on the same machine as "
            "serve_policy.py, try --policy-host 127.0.0.1."
        ) from exc
    print(f"[Policy] connected. metadata={policy.get_server_metadata()}")
    return policy


class RealSenseCamera:
    def __init__(self, config: CameraConfig):
        self.config = config
        self.pipeline = self._start_pipeline()
        self._latest_frame: np.ndarray | None = None
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._capture_loop, name=f"{config.name}Camera", daemon=True)
        self._thread.start()

        print(
            f"[Camera] {config.name} started: serial={config.serial}, "
            f"{config.width}x{config.height}@{config.fps}"
        )
        while self.get_frame() is None:
            time.sleep(0.01)
        print(f"[Camera] {config.name} first frame ready.")

    def _start_pipeline(self) -> rs.pipeline:
        last_error: Exception | None = None
        rs_module = _load_realsense_module()
        for attempt in range(1, CAM_START_RETRIES + 1):
            pipeline = rs_module.pipeline()
            cfg = rs_module.config()
            cfg.enable_device(self.config.serial)
            cfg.enable_stream(
                rs_module.stream.color,
                self.config.width,
                self.config.height,
                rs_module.format.bgr8,
                self.config.fps,
            )
            try:
                pipeline.start(cfg)
                return pipeline
            except Exception as exc:
                last_error = exc
                print(f"[Camera] {self.config.name} start failed {attempt}/{CAM_START_RETRIES}: {exc}")
                try:
                    pipeline.stop()
                except Exception:
                    pass
                if attempt < CAM_START_RETRIES:
                    time.sleep(CAM_START_RETRY_DELAY_S)
        raise RuntimeError(f"Failed to start camera {self.config.name}: {last_error}") from last_error

    def _capture_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                frames = self.pipeline.wait_for_frames(timeout_ms=CAM_FRAME_TIMEOUT_MS)
                color_frame = frames.get_color_frame()
                if color_frame:
                    with self._lock:
                        self._latest_frame = np.asanyarray(color_frame.get_data())
            except Exception as exc:
                print(f"[Camera] {self.config.name} capture error: {exc}")

    def get_frame(self) -> np.ndarray | None:
        with self._lock:
            return self._latest_frame.copy() if self._latest_frame is not None else None

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)
        try:
            self.pipeline.stop()
        except Exception as exc:
            print(f"[Camera] {self.config.name} stop failed: {exc}")


class ArmClient:
    def __init__(
        self,
        side: str,
        config: ArmConfig,
        *,
        initial_gripper_command: int,
        execution: ExecutionConfig,
    ):
        self.side = side
        self.config = config
        self.execution = execution
        gripper_cls, _, _, dynamics_cls, robot_cls = _load_franky_symbols()
        self.robot = robot_cls(config.robot_ip)
        self.gripper = gripper_cls(config.robot_ip)
        self.robot.set_collision_behavior([20.0] * 7, [20.0] * 7, [10.0] * 6, [10.0] * 6)
        self.robot.relative_dynamics_factor = dynamics_cls(
            execution.relative_velocity,
            execution.relative_acceleration,
            execution.relative_jerk,
        )
        self.last_gripper_command = int(initial_gripper_command)
        print(
            f"[Robot] {side} connected: {config.robot_ip}, initial_gripper={self.last_gripper_command}, "
            f"joint_vel={execution.joint_vel:.3f}, joint_acc={execution.joint_acc:.3f}"
        )

    def recover_if_needed(self) -> None:
        if getattr(self.robot, "has_errors", False):
            print(f"[Robot] {self.side} recovering from errors...")
            self.robot.recover_from_errors()

    def read_joints(self) -> np.ndarray:
        return np.asarray(self.robot.state.q, dtype=np.float64).reshape(7)

    def read_gripper_width(self) -> float:
        try:
            return float(self.gripper.width)
        except Exception:
            if self.last_gripper_command == GRIPPER_OPEN_COMMAND:
                return float(self.config.gripper_open_width)
            return float(self.config.gripper_close_width)

    def apply_gripper_command(self, command: int) -> None:
        command = int(command)
        if command == self.last_gripper_command:
            return
        if command == GRIPPER_OPEN_COMMAND:
            self.gripper.move(self.config.gripper_open_width, self.config.gripper_speed)
            print(f"[Gripper] {self.side} command=0 open width={self.config.gripper_open_width:.4f}")
        elif command == GRIPPER_CLOSE_COMMAND:
            try:
                self.gripper.grasp(
                    self.config.gripper_close_width,
                    self.config.gripper_speed,
                    self.config.gripper_force,
                    epsilon_outer=1.0,
                )
            except TypeError:
                self.gripper.grasp(self.config.gripper_close_width, self.config.gripper_speed, self.config.gripper_force)
            print(f"[Gripper] {self.side} command=1 close")
        else:
            raise ValueError(f"Unsupported gripper command: {command}")
        self.last_gripper_command = command

    def move_joint_chunk(self, joint_chunk: np.ndarray) -> None:
        _, waypoint_cls, motion_cls, _, _ = _load_franky_symbols()
        waypoints = [
            waypoint_cls(row.astype(float).tolist())
            for row in np.asarray(joint_chunk, dtype=np.float64)
        ]
        self.recover_if_needed()
        self.robot.joint_velocity_limit.set([float(self.execution.joint_vel)] * 7)
        self.robot.joint_acceleration_limit.set([float(self.execution.joint_acc)] * 7)
        self.robot.move(motion_cls(waypoints))
        self.recover_if_needed()


class PklEpisodeRecorder:
    def __init__(self, rollout_dir: Path, prompt: str, *, source: str):
        self.rollout_dir = rollout_dir
        self.prompt = prompt
        self.source = source
        self.rollout_dir.mkdir(parents=True, exist_ok=True)
        self.next_episode_idx = self._discover_next_episode_idx()
        self.lock = threading.Lock()
        self.states: list[np.ndarray] = []
        self.images: dict[str, list[np.ndarray]] = {}
        self.actions: list[np.ndarray] = []
        self.commanded_actions: list[np.ndarray] = []
        self.raw_actions: list[np.ndarray] = []
        self.action_chunks: list[np.ndarray] = []
        self.timestamps: list[float] = []
        self.latencies_ms: list[float] = []

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
        commanded_action: np.ndarray,
        raw_action: np.ndarray,
        action_chunk: np.ndarray,
        latency_ms: float,
    ) -> None:
        with self.lock:
            self.states.append(np.asarray(state, dtype=np.float32).copy())
            for name, image in images.items():
                self.images.setdefault(name, []).append(np.asarray(image, dtype=np.uint8).copy())
            self.actions.append(np.asarray(action, dtype=np.float32).copy())
            self.commanded_actions.append(np.asarray(commanded_action, dtype=np.float32).copy())
            self.raw_actions.append(np.asarray(raw_action, dtype=np.float32).copy())
            self.action_chunks.append(np.asarray(action_chunk, dtype=np.float32).copy())
            self.timestamps.append(time.time())
            self.latencies_ms.append(float(latency_ms))

    @property
    def num_steps(self) -> int:
        with self.lock:
            return len(self.actions)

    def _pop_episode_data(self) -> dict[str, object] | None:
        with self.lock:
            if not self.actions:
                return None
            episode_idx = self.next_episode_idx
            self.next_episode_idx += 1
            data = {
                "episode_idx": episode_idx,
                "states": np.stack(self.states, axis=0),
                "images": {name: np.stack(frames, axis=0) for name, frames in self.images.items()},
                "actions": np.stack(self.actions, axis=0),
                "commanded_actions": np.stack(self.commanded_actions, axis=0),
                "raw_actions": np.stack(self.raw_actions, axis=0),
                "action_chunks": [chunk.copy() for chunk in self.action_chunks],
                "timestamps": np.asarray(self.timestamps, dtype=np.float64),
                "latencies_ms": np.asarray(self.latencies_ms, dtype=np.float32),
            }
            self.states.clear()
            self.images.clear()
            self.actions.clear()
            self.commanded_actions.clear()
            self.raw_actions.clear()
            self.action_chunks.clear()
            self.timestamps.clear()
            self.latencies_ms.clear()
            return data

    def finalize(self, *, success: bool, reason: str, metadata: dict[str, object]) -> Path | None:
        episode = self._pop_episode_data()
        if episode is None:
            print("[Recorder] empty episode, skip pkl save.")
            return None

        actions = np.asarray(episode["actions"], dtype=np.float32)
        num_steps = int(actions.shape[0])
        rewards = np.zeros((num_steps,), dtype=np.float32)
        dones = np.zeros((num_steps,), dtype=bool)
        successes = np.zeros((num_steps,), dtype=bool)
        rewards[-1] = float(success)
        dones[-1] = True
        successes[-1] = bool(success)

        episode_idx = int(episode["episode_idx"])
        payload = {
            "observations": {
                "state": np.asarray(episode["states"], dtype=np.float32),
                "images": {name: np.asarray(value, dtype=np.uint8) for name, value in episode["images"].items()},
            },
            "actions": actions,
            "reward": float(success),
            "rewards": rewards,
            "done": dones,
            "success": bool(success),
            "successes": successes,
            "metadata": {
                "source": self.source,
                "episode": episode_idx,
                "reason": reason,
                "prompt": self.prompt,
                "timestamps": np.asarray(episode["timestamps"], dtype=np.float64),
                "latencies_ms": np.asarray(episode["latencies_ms"], dtype=np.float32),
                "commanded_actions": np.asarray(episode["commanded_actions"], dtype=np.float32),
                "raw_actions": np.asarray(episode["raw_actions"], dtype=np.float32),
                "action_chunks": [np.asarray(chunk, dtype=np.float32) for chunk in episode["action_chunks"]],
                "action_order": ACTION_ORDER,
                "gripper_convention": "0=open, 1=close",
                **metadata,
            },
        }

        suffix = "success" if success else "failure"
        out_path = self.rollout_dir / f"episode_{episode_idx:06d}_{suffix}.pkl"
        with out_path.open("wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"[Recorder] saved pkl: {out_path}")
        return out_path


class DualArmInferenceRunner:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        config = _load_yaml(args.config)
        camera_names = _parse_names(args.cameras)
        self.left_cfg = _load_arm_config(config, "left", ip_override=args.left_ip)
        self.right_cfg = _load_arm_config(config, "right", ip_override=args.right_ip)
        self.camera_cfgs = _load_camera_configs(config, camera_names, args.camera_fps)
        self.execution = ExecutionConfig(
            joint_vel=float(args.joint_vel),
            joint_acc=float(args.joint_acc),
            relative_velocity=float(args.relative_velocity),
            relative_acceleration=float(args.relative_acceleration),
            relative_jerk=float(args.relative_jerk),
        )
        self.prompt = str(args.prompt)
        self._episode_end_requested = False
        self._interrupted = False

        self.policy = _connect_policy_from_args(args)

        self.left = ArmClient(
            "left",
            self.left_cfg,
            initial_gripper_command=args.left_gripper_command,
            execution=self.execution,
        )
        self.right = ArmClient(
            "right",
            self.right_cfg,
            initial_gripper_command=args.right_gripper_command,
            execution=self.execution,
        )

        self.cameras = {name: RealSenseCamera(cfg) for name, cfg in self.camera_cfgs.items()}
        self.recorder = PklEpisodeRecorder(
            Path(args.rollout_dir).expanduser().resolve(),
            self.prompt,
            source="dual_arm_model_inference_test",
        )
        print(f"[Recorder] pkl dir: {self.recorder.rollout_dir}")

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

    def _read_state(self) -> np.ndarray:
        left_joints = self.left.read_joints()
        right_joints = self.right.read_joints()
        if self.args.state_gripper_mode == "width":
            left_gripper = self.left.read_gripper_width()
            right_gripper = self.right.read_gripper_width()
        else:
            left_gripper = float(self.left.last_gripper_command)
            right_gripper = float(self.right.last_gripper_command)
        return np.asarray(
            [
                *left_joints.tolist(),
                left_gripper,
                *right_joints.tolist(),
                right_gripper,
            ],
            dtype=np.float32,
        )

    def _read_images(self) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]] | None:
        bgr_frames: dict[str, np.ndarray] = {}
        for name, camera in self.cameras.items():
            frame = camera.get_frame()
            if frame is None:
                print(f"[Obs] missing camera frame: {name}")
                return None
            bgr_frames[name] = frame

        policy_images = {name: _policy_image_from_bgr(frame) for name, frame in bgr_frames.items()}
        record_images = {name: _record_image_from_bgr(frame) for name, frame in bgr_frames.items()}
        if "agent_view" in policy_images:
            policy_images["fixed_camera"] = policy_images["agent_view"]
            record_images["fixed_camera"] = record_images["agent_view"]
            policy_images["cam_high"] = policy_images["agent_view"]
        if "right_wrist" in policy_images:
            policy_images["wrist_camera"] = policy_images["right_wrist"]
            record_images["wrist_camera"] = record_images["right_wrist"]
            policy_images["cam_right_wrist"] = policy_images["right_wrist"]
        if "left_wrist" in policy_images:
            policy_images["cam_left_wrist"] = policy_images["left_wrist"]
        return policy_images, record_images

    def _build_observation(self) -> tuple[dict[str, object], dict[str, object]] | None:
        images_pair = self._read_images()
        if images_pair is None:
            return None
        policy_images, record_images = images_pair
        state = self._read_state()
        wrist_image = policy_images.get("wrist_camera")
        if wrist_image is None:
            wrist_image = policy_images.get("right_wrist")
        observation = {
            "observation": {
                "images": policy_images,
                "state": state,
            },
            "observation.images.agent_view": policy_images.get("agent_view"),
            "observation.images.fixed_camera": policy_images.get("fixed_camera"),
            "observation.images.right_wrist": policy_images.get("right_wrist"),
            "observation.images.wrist_camera": policy_images.get("wrist_camera"),
            "observation.images.left_wrist": policy_images.get("left_wrist"),
            "observation.state": state,
            "observation/state": state,
            "observation/wrist_image": wrist_image,
            "images": policy_images,
            "state": state,
            "prompt": self.prompt,
        }
        record_observation = {
            "state": state.copy(),
            "images": record_images,
        }
        return observation, record_observation

    def _normalize_action_chunk(self, action_chunk: np.ndarray, exec_steps: int) -> tuple[np.ndarray, np.ndarray]:
        raw = np.asarray(action_chunk[:exec_steps, :ACTION_DIM], dtype=np.float32).copy()
        commanded = raw.copy()
        if self.args.gripper_action_mode == "width":
            commanded[:, 7] = [
                _gripper_command_from_width(value, self.args.gripper_width_threshold)
                for value in raw[:, 7]
            ]
            commanded[:, 15] = [
                _gripper_command_from_width(value, self.args.gripper_width_threshold)
                for value in raw[:, 15]
            ]
        else:
            commanded[:, 7] = [_gripper_command(value) for value in raw[:, 7]]
            commanded[:, 15] = [_gripper_command(value) for value in raw[:, 15]]
        return raw, commanded.astype(np.float32)

    def _validate_joint_targets(self, side: str, current: np.ndarray, targets: np.ndarray) -> None:
        targets = np.asarray(targets, dtype=np.float64)
        if not np.isfinite(targets).all():
            raise ValueError(f"{side} joint targets contain NaN or inf.")

        margin = float(self.args.joint_limit_margin)
        lower = FRANKA_JOINT_LIMITS[:, 0] + margin
        upper = FRANKA_JOINT_LIMITS[:, 1] - margin
        below = targets < lower
        above = targets > upper
        if np.any(below | above):
            step_idx, joint_idx = np.argwhere(below | above)[0]
            value = targets[step_idx, joint_idx]
            raise ValueError(
                f"{side} joint target out of limit at step {step_idx}, joint {joint_idx + 1}: "
                f"{value:.4f} not in [{lower[joint_idx]:.4f}, {upper[joint_idx]:.4f}]"
            )

        max_delta = float(self.args.max_joint_delta)
        if max_delta <= 0.0:
            return
        trajectory = np.vstack([np.asarray(current, dtype=np.float64).reshape(1, 7), targets])
        deltas = np.abs(np.diff(trajectory, axis=0))
        observed = float(np.max(deltas))
        if observed > max_delta:
            step_idx, joint_idx = np.unravel_index(int(np.argmax(deltas)), deltas.shape)
            raise ValueError(
                f"{side} joint jump too large before step {step_idx}: joint {joint_idx + 1} "
                f"delta={observed:.4f} rad > --max-joint-delta {max_delta:.4f}"
            )

    def _validate_commanded_chunk(self, commanded: np.ndarray) -> None:
        self._validate_joint_targets("left", self.left.read_joints(), commanded[:, :7])
        self._validate_joint_targets("right", self.right.read_joints(), commanded[:, 8:15])

    def _run_gripper_events(
        self,
        arm: ArmClient,
        commands: np.ndarray,
        *,
        started_at: float,
        step_period_s: float,
    ) -> None:
        previous = int(arm.last_gripper_command)
        for step_idx, command_value in enumerate(commands.tolist()):
            command = int(command_value)
            if command == previous:
                continue
            due_at = started_at + step_idx * step_period_s
            delay_s = due_at - time.monotonic()
            if delay_s > 0.0:
                time.sleep(delay_s)
            try:
                arm.apply_gripper_command(command)
            except Exception as exc:
                print(f"[Gripper] {arm.side} command failed at step {step_idx}: {exc}")
            previous = command

    def _execute_commanded_chunk(self, commanded: np.ndarray) -> None:
        self._validate_commanded_chunk(commanded)
        left_joints = commanded[:, :7]
        right_joints = commanded[:, 8:15]
        step_period_s = 1.0 / max(float(self.args.action_hz), 1e-6)
        started_at = time.monotonic()
        left_gripper_thread = threading.Thread(
            target=self._run_gripper_events,
            args=(self.left, commanded[:, 7]),
            kwargs={"started_at": started_at, "step_period_s": step_period_s},
            daemon=True,
        )
        right_gripper_thread = threading.Thread(
            target=self._run_gripper_events,
            args=(self.right, commanded[:, 15]),
            kwargs={"started_at": started_at, "step_period_s": step_period_s},
            daemon=True,
        )
        left_gripper_thread.start()
        right_gripper_thread.start()
        with ThreadPoolExecutor(max_workers=2) as executor:
            left_task = executor.submit(self.left.move_joint_chunk, left_joints)
            right_task = executor.submit(self.right.move_joint_chunk, right_joints)
            left_task.result()
            right_task.result()
        left_gripper_thread.join(timeout=2.0)
        right_gripper_thread.join(timeout=2.0)
        self.left.last_gripper_command = int(commanded[-1, 7])
        self.right.last_gripper_command = int(commanded[-1, 15])

    def run_episode(self) -> None:
        self._episode_end_requested = False
        executed_count = 0
        print(
            f"[Episode] start: max_actions={self.args.max_actions}, "
            f"chunk_exec_steps={self.args.chunk_exec_steps}, execute={self.args.execute}"
        )
        print("[Episode] Press Enter to stop and save pkl.")
        while executed_count < self.args.max_actions and not self._episode_end_requested:
            self._poll_enter_to_end_episode()
            observation_pair = self._build_observation()
            if observation_pair is None:
                time.sleep(0.02)
                continue
            policy_observation, record_observation = observation_pair
            try:
                t0 = time.time()
                result = self.policy.infer(policy_observation)
                action_chunk = np.asarray(result["actions"], dtype=np.float32)
                latency_ms = (time.time() - t0) * 1000.0
            except Exception as exc:
                print(f"[Inference] failed: {exc}")
                time.sleep(0.05)
                continue

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

            if self.args.execute:
                try:
                    self._execute_commanded_chunk(commanded)
                except Exception as exc:
                    print(f"[Exec] failed: {exc}")
                    self.left.recover_if_needed()
                    self.right.recover_if_needed()
                    break

            state = np.asarray(record_observation["state"], dtype=np.float32)
            images = record_observation["images"]
            assert isinstance(images, dict)
            for idx in range(exec_steps):
                self.recorder.record_step(
                    state=state,
                    images=images,
                    action=commanded[idx],
                    commanded_action=commanded[idx],
                    raw_action=raw[idx],
                    action_chunk=action_chunk,
                    latency_ms=latency_ms,
                )
                executed_count += 1
            print(f"[Episode] collected_steps={executed_count}")

        success = self._resolve_success_label()
        self.recorder.finalize(
            success=success,
            reason="manual_enter" if self._episode_end_requested else "episode_end",
            metadata={
                "execute": bool(self.args.execute),
                "policy_host": self.args.policy_host,
                "policy_port": int(self.args.policy_port),
                "camera_names": list(self.camera_cfgs),
                "state_gripper_mode": self.args.state_gripper_mode,
                "gripper_action_mode": self.args.gripper_action_mode,
                "gripper_width_threshold": float(self.args.gripper_width_threshold),
                "action_hz": float(self.args.action_hz),
                "chunk_exec_steps": int(self.args.chunk_exec_steps),
                "joint_vel": float(self.execution.joint_vel),
                "joint_acc": float(self.execution.joint_acc),
                "max_joint_delta": float(self.args.max_joint_delta),
                "joint_limit_margin": float(self.args.joint_limit_margin),
                "relative_velocity": float(self.execution.relative_velocity),
                "relative_acceleration": float(self.execution.relative_acceleration),
                "relative_jerk": float(self.execution.relative_jerk),
            },
        )

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

    def close(self) -> None:
        old_sigint_handler = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        try:
            for camera in self.cameras.values():
                camera.stop()
            if hasattr(self, "policy"):
                self.policy.close()
        finally:
            signal.signal(signal.SIGINT, old_sigint_handler)
            print("[Runner] closed.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dual-arm OpenPI model inference test with pkl rollout saving.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="YAML config for arms and cameras.")
    parser.add_argument("--policy-host", default="10.0.40.113")
    parser.add_argument("--policy-port", type=int, default=8001)
    parser.add_argument(
        "--policy-timeout",
        type=float,
        default=10.0,
        help="Seconds to wait for policy websocket startup and each inference response; 0 disables timeout.",
    )
    parser.add_argument(
        "--policy-healthcheck",
        action="store_true",
        help="Only test websocket connection to the policy server, then exit before robot/camera initialization.",
    )
    parser.add_argument("--prompt", default="Use both arms to complete the task.")
    parser.add_argument("--cameras", default=DEFAULT_CAMERAS, help="Comma-separated camera names from YAML.")
    parser.add_argument("--camera-fps", type=int, default=None)
    parser.add_argument("--left-ip", default=None)
    parser.add_argument("--right-ip", default=None)
    parser.add_argument("--left-gripper-command", type=int, choices=(0, 1), default=0)
    parser.add_argument("--right-gripper-command", type=int, choices=(0, 1), default=0)
    parser.add_argument(
        "--state-gripper-mode",
        choices=("command", "width"),
        default="command",
        help="Use 0/1 command or measured gripper width in observation.state.",
    )
    parser.add_argument(
        "--gripper-action-mode",
        choices=("command", "width"),
        default="command",
        help="Interpret policy gripper action channels as 0/1 commands or gripper widths.",
    )
    parser.add_argument(
        "--gripper-width-threshold",
        type=float,
        default=GRIPPER_WIDTH_THRESHOLD,
        help="Width threshold for --gripper-action-mode width; values >= threshold open.",
    )
    parser.add_argument("--rollout-dir", default=str(DEFAULT_ROLLOUT_DIR))
    parser.add_argument("--max-actions", type=int, default=200)
    parser.add_argument("--chunk-exec-steps", type=int, default=32)
    parser.add_argument("--action-hz", type=float, default=30.0)
    parser.add_argument("--joint-vel", type=float, default=2.10)
    parser.add_argument("--joint-acc", type=float, default=5.10)
    parser.add_argument(
        "--max-joint-delta",
        type=float,
        default=0.35,
        help="Reject execution if any adjacent joint target changes by more than this radian value; <=0 disables.",
    )
    parser.add_argument(
        "--joint-limit-margin",
        type=float,
        default=0.02,
        help="Margin subtracted from Franka joint limits before executing policy actions.",
    )
    parser.add_argument("--relative-velocity", type=float, default=0.30)
    parser.add_argument("--relative-acceleration", type=float, default=0.10)
    parser.add_argument("--relative-jerk", type=float, default=0.03)
    parser.add_argument("--execute", dest="execute", action="store_true", help="Execute inferred joint targets on both arms.")
    parser.add_argument("--no-execute", dest="execute", action="store_false", help="Only infer and save pkl; do not move arms.")
    parser.set_defaults(execute=True)
    parser.add_argument("--success-label", choices=("0", "1"), default=None, help="Skip interactive label prompt.")
    args = parser.parse_args()
    if args.max_actions < 1:
        parser.error("--max-actions must be >= 1")
    if args.chunk_exec_steps < 1:
        parser.error("--chunk-exec-steps must be >= 1")
    if args.action_hz <= 0.0:
        parser.error("--action-hz must be > 0")
    if args.policy_timeout < 0.0:
        parser.error("--policy-timeout must be >= 0")
    if args.joint_vel <= 0.0:
        parser.error("--joint-vel must be > 0")
    if args.joint_acc <= 0.0:
        parser.error("--joint-acc must be > 0")
    if args.gripper_width_threshold < 0.0:
        parser.error("--gripper-width-threshold must be >= 0")
    if args.joint_limit_margin < 0.0:
        parser.error("--joint-limit-margin must be >= 0")
    if not _parse_names(args.cameras):
        parser.error("--cameras must contain at least one camera name")
    return args


def main() -> int:
    args = parse_args()
    if args.policy_healthcheck:
        policy = _connect_policy_from_args(args)
        policy.close()
        return 0

    runner = DualArmInferenceRunner(args)
    try:
        runner.run_episode()
    except KeyboardInterrupt:
        runner._interrupted = True
        print("\n[Runner] interrupted.")
        if runner.recorder.num_steps > 0:
            runner.recorder.finalize(
                success=False,
                reason="keyboard_interrupt",
                metadata={
                    "execute": bool(args.execute),
                    "policy_host": args.policy_host,
                    "policy_port": int(args.policy_port),
                    "camera_names": list(runner.camera_cfgs),
                    "state_gripper_mode": args.state_gripper_mode,
                    "gripper_action_mode": args.gripper_action_mode,
                    "gripper_width_threshold": float(args.gripper_width_threshold),
                    "action_hz": float(args.action_hz),
                    "chunk_exec_steps": int(args.chunk_exec_steps),
                    "joint_vel": float(runner.execution.joint_vel),
                    "joint_acc": float(runner.execution.joint_acc),
                    "max_joint_delta": float(args.max_joint_delta),
                    "joint_limit_margin": float(args.joint_limit_margin),
                    "relative_velocity": float(runner.execution.relative_velocity),
                    "relative_acceleration": float(runner.execution.relative_acceleration),
                    "relative_jerk": float(runner.execution.relative_jerk),
                },
            )
        return 130
    finally:
        runner.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
