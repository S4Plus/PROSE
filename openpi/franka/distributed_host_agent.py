#!/usr/bin/env python3
"""单臂控制主机 agent。

该脚本运行在 Right Host 或 Left Host 上，只负责本机 RealSense 采集、本机 Franka
状态读取和单臂动作执行。Master 通过 websocket 请求观测并下发单臂 action chunk。
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
import http
import logging
import threading
import time
import traceback
from typing import Any

import cv2
import numpy as np
from openpi_client import msgpack_numpy
import websockets
import websockets.asyncio.server as ws_server
import websockets.frames

IMG_W = 224
IMG_H = 224
CAM_FRAME_TIMEOUT_MS = 1000
CAM_START_RETRIES = 5
CAM_START_RETRY_DELAY_S = 0.5
GRIPPER_OPEN_COMMAND = 0
GRIPPER_CLOSE_COMMAND = 1
GRIPPER_THRESHOLD = 0.5
GRIPPER_WIDTH_THRESHOLD = 0.04
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
class CameraConfig:
    name: str
    serial: str
    width: int
    height: int
    fps: int


@dataclass(frozen=True)
class ArmConfig:
    side: str
    robot_ip: str
    home_joint_values: list[float]
    gripper_open_width: float
    gripper_close_width: float
    gripper_speed: float
    gripper_force: float


@dataclass(frozen=True)
class ExecutionConfig:
    joint_vel: float
    joint_acc: float
    relative_velocity: float
    relative_acceleration: float
    relative_jerk: float
    max_joint_delta: float
    joint_limit_margin: float


def _policy_image_from_bgr(frame: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)
    return np.transpose(rgb, (2, 0, 1)).astype(np.float32) / 255.0


def _record_image_from_bgr(frame: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.uint8, copy=False)


def _parse_float_list(value: str, *, length: int, name: str) -> list[float]:
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if len(parts) != length:
        raise argparse.ArgumentTypeError(f"{name} must contain {length} comma-separated numbers.")
    return [float(part) for part in parts]


def _parse_camera_specs(value: str, *, width: int, height: int, fps: int) -> dict[str, CameraConfig]:
    cameras: dict[str, CameraConfig] = {}
    for item in [part.strip() for part in value.split(",") if part.strip()]:
        if ":" not in item:
            raise argparse.ArgumentTypeError("--cameras entries must use name:serial.")
        name, serial = [part.strip() for part in item.split(":", 1)]
        if not name or not serial:
            raise argparse.ArgumentTypeError("--cameras entries must use non-empty name:serial.")
        if name in cameras:
            raise argparse.ArgumentTypeError(f"duplicate camera name: {name}")
        cameras[name] = CameraConfig(name=name, serial=serial, width=width, height=height, fps=fps)
    if not cameras:
        raise argparse.ArgumentTypeError("--cameras must contain at least one camera.")
    return cameras


def _gripper_command(value: float) -> int:
    return GRIPPER_CLOSE_COMMAND if float(value) >= GRIPPER_THRESHOLD else GRIPPER_OPEN_COMMAND


def _gripper_command_from_width(width: float, threshold: float) -> int:
    return GRIPPER_OPEN_COMMAND if float(width) >= threshold else GRIPPER_CLOSE_COMMAND


class RealSenseCamera:
    def __init__(self, config: CameraConfig):
        self.config = config
        self.pipeline = self._start_pipeline()
        self._latest_frame: np.ndarray | None = None
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._capture_loop, name=f"{config.name}Camera", daemon=True)
        self._thread.start()

        logging.info(
            "Camera %s started: serial=%s, %dx%d@%d",
            config.name,
            config.serial,
            config.width,
            config.height,
            config.fps,
        )
        while self.get_frame() is None:
            time.sleep(0.01)
        logging.info("Camera %s first frame ready.", config.name)

    def _start_pipeline(self):
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
                logging.warning("Camera %s start failed %d/%d: %s", self.config.name, attempt, CAM_START_RETRIES, exc)
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
                logging.warning("Camera %s capture error: %s", self.config.name, exc)

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
            logging.warning("Camera %s stop failed: %s", self.config.name, exc)


class ArmClient:
    def __init__(
        self,
        config: ArmConfig,
        execution: ExecutionConfig,
        *,
        initial_gripper_command: int,
        no_robot: bool,
    ):
        self.config = config
        self.execution = execution
        self.no_robot = bool(no_robot)
        self.last_gripper_command = int(initial_gripper_command)
        self._mock_joints = np.asarray(config.home_joint_values, dtype=np.float64).reshape(7)
        self.robot = None
        self.gripper = None

        if self.no_robot:
            logging.warning("Robot connection disabled by --no-robot; actions will not move hardware.")
            return

        gripper_cls, _, _, dynamics_cls, robot_cls = _load_franky_symbols()
        self.robot = robot_cls(config.robot_ip)
        self.gripper = gripper_cls(config.robot_ip)
        self.robot.set_collision_behavior([20.0] * 7, [20.0] * 7, [10.0] * 6, [10.0] * 6)
        self.robot.relative_dynamics_factor = dynamics_cls(
            execution.relative_velocity,
            execution.relative_acceleration,
            execution.relative_jerk,
        )
        logging.info(
            "Robot %s connected: %s, initial_gripper=%d, joint_vel=%.3f, joint_acc=%.3f",
            config.side,
            config.robot_ip,
            self.last_gripper_command,
            execution.joint_vel,
            execution.joint_acc,
        )

    def recover_if_needed(self) -> None:
        if self.no_robot or self.robot is None:
            return
        if getattr(self.robot, "has_errors", False):
            logging.warning("Robot %s recovering from errors...", self.config.side)
            self.robot.recover_from_errors()

    def read_joints(self) -> np.ndarray:
        if self.no_robot or self.robot is None:
            return self._mock_joints.copy()
        return np.asarray(self.robot.state.q, dtype=np.float64).reshape(7)

    def read_gripper_width(self) -> float:
        if self.no_robot or self.gripper is None:
            if self.last_gripper_command == GRIPPER_OPEN_COMMAND:
                return float(self.config.gripper_open_width)
            return float(self.config.gripper_close_width)
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
        if self.no_robot or self.gripper is None:
            self.last_gripper_command = command
            logging.info("Robot %s mock gripper command=%d", self.config.side, command)
            return
        if command == GRIPPER_OPEN_COMMAND:
            self.gripper.move(self.config.gripper_open_width, self.config.gripper_speed)
            logging.info("Gripper %s command=0 open width=%.4f", self.config.side, self.config.gripper_open_width)
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
            logging.info("Gripper %s command=1 close", self.config.side)
        else:
            raise ValueError(f"Unsupported gripper command: {command}")
        self.last_gripper_command = command

    def _validate_joint_targets(self, targets: np.ndarray) -> None:
        targets = np.asarray(targets, dtype=np.float64)
        if targets.ndim != 2 or targets.shape[1] != 7:
            raise ValueError(f"joint_chunk must have shape (T, 7), got {targets.shape}.")
        if not np.isfinite(targets).all():
            raise ValueError("joint targets contain NaN or inf.")

        margin = float(self.execution.joint_limit_margin)
        lower = FRANKA_JOINT_LIMITS[:, 0] + margin
        upper = FRANKA_JOINT_LIMITS[:, 1] - margin
        below = targets < lower
        above = targets > upper
        if np.any(below | above):
            step_idx, joint_idx = np.argwhere(below | above)[0]
            value = targets[step_idx, joint_idx]
            raise ValueError(
                f"{self.config.side} joint target out of limit at step {step_idx}, joint {joint_idx + 1}: "
                f"{value:.4f} not in [{lower[joint_idx]:.4f}, {upper[joint_idx]:.4f}]"
            )

        max_delta = float(self.execution.max_joint_delta)
        if max_delta <= 0.0:
            return
        trajectory = np.vstack([self.read_joints().reshape(1, 7), targets])
        deltas = np.abs(np.diff(trajectory, axis=0))
        observed = float(np.max(deltas))
        if observed > max_delta:
            step_idx, joint_idx = np.unravel_index(int(np.argmax(deltas)), deltas.shape)
            raise ValueError(
                f"{self.config.side} joint jump too large before step {step_idx}: joint {joint_idx + 1} "
                f"delta={observed:.4f} rad > --max-joint-delta {max_delta:.4f}"
            )

    def _run_gripper_events(self, commands: np.ndarray, *, step_period_s: float) -> None:
        previous = int(self.last_gripper_command)
        start_mono = time.monotonic()
        for step_idx, command_value in enumerate(commands.tolist()):
            command = int(command_value)
            if command == previous:
                continue
            delay_s = start_mono + step_idx * step_period_s - time.monotonic()
            if delay_s > 0.0:
                time.sleep(delay_s)
            try:
                self.apply_gripper_command(command)
            except Exception as exc:
                logging.warning("Gripper %s command failed at step %d: %s", self.config.side, step_idx, exc)
            previous = command

    def move_joint_chunk(self, joint_chunk: np.ndarray) -> None:
        self._validate_joint_targets(joint_chunk)
        if self.no_robot or self.robot is None:
            self._mock_joints = np.asarray(joint_chunk[-1], dtype=np.float64)
            logging.info("Robot %s mock moved chunk=%s", self.config.side, joint_chunk.shape)
            return
        _, waypoint_cls, motion_cls, _, _ = _load_franky_symbols()
        waypoints = [waypoint_cls(row.astype(float).tolist()) for row in np.asarray(joint_chunk, dtype=np.float64)]
        self.recover_if_needed()
        self.robot.joint_velocity_limit.set([float(self.execution.joint_vel)] * 7)
        self.robot.joint_acceleration_limit.set([float(self.execution.joint_acc)] * 7)
        self.robot.move(motion_cls(waypoints))
        self.recover_if_needed()

    def execute_chunk(
        self,
        *,
        joint_chunk: np.ndarray,
        gripper_commands: np.ndarray,
        action_hz: float,
        execute_at: float | None,
    ) -> dict[str, Any]:
        joint_chunk = np.asarray(joint_chunk, dtype=np.float64)
        gripper_commands = np.asarray(gripper_commands, dtype=np.int64).reshape(-1)
        if joint_chunk.shape[0] != gripper_commands.shape[0]:
            raise ValueError("joint_chunk and gripper_commands must have the same horizon.")
        self._validate_joint_targets(joint_chunk)

        now = time.time()
        if execute_at is not None:
            delay_s = float(execute_at) - now
            if delay_s > 0.0:
                time.sleep(delay_s)

        started_at = time.time()
        step_period_s = 1.0 / max(float(action_hz), 1e-6)
        gripper_thread = threading.Thread(
            target=self._run_gripper_events,
            args=(gripper_commands,),
            kwargs={"step_period_s": step_period_s},
            daemon=True,
        )
        gripper_thread.start()
        self.move_joint_chunk(joint_chunk)
        gripper_thread.join(timeout=2.0)
        self.last_gripper_command = int(gripper_commands[-1])
        return {
            "side": self.config.side,
            "started_at": started_at,
            "finished_at": time.time(),
            "steps": int(joint_chunk.shape[0]),
            "no_robot": bool(self.no_robot),
        }


class HostAgent:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.side = str(args.side)
        self.camera_configs = _parse_camera_specs(
            args.cameras,
            width=int(args.camera_width),
            height=int(args.camera_height),
            fps=int(args.camera_fps),
        )
        self.arm = ArmClient(
            ArmConfig(
                side=self.side,
                robot_ip=str(args.robot_ip),
                home_joint_values=_parse_float_list(args.home_joints, length=7, name="--home-joints"),
                gripper_open_width=float(args.gripper_open_width),
                gripper_close_width=float(args.gripper_close_width),
                gripper_speed=float(args.gripper_speed),
                gripper_force=float(args.gripper_force),
            ),
            ExecutionConfig(
                joint_vel=float(args.joint_vel),
                joint_acc=float(args.joint_acc),
                relative_velocity=float(args.relative_velocity),
                relative_acceleration=float(args.relative_acceleration),
                relative_jerk=float(args.relative_jerk),
                max_joint_delta=float(args.max_joint_delta),
                joint_limit_margin=float(args.joint_limit_margin),
            ),
            initial_gripper_command=int(args.initial_gripper_command),
            no_robot=bool(args.no_robot),
        )
        self.cameras = {name: RealSenseCamera(cfg) for name, cfg in self.camera_configs.items()}
        self._exec_lock = threading.Lock()

    def metadata(self) -> dict[str, Any]:
        return {
            "type": "franka_host_agent",
            "side": self.side,
            "camera_names": list(self.camera_configs),
            "state_dim": 8,
            "action_dim": 8,
            "host_time": time.time(),
            "no_robot": bool(self.args.no_robot),
        }

    def build_observation(self) -> dict[str, Any]:
        bgr_frames: dict[str, np.ndarray] = {}
        for name, camera in self.cameras.items():
            frame = camera.get_frame()
            if frame is None:
                raise RuntimeError(f"missing camera frame: {name}")
            bgr_frames[name] = frame

        if self.args.state_gripper_mode == "width":
            gripper = self.arm.read_gripper_width()
        else:
            gripper = float(self.arm.last_gripper_command)

        state = np.asarray([*self.arm.read_joints().tolist(), gripper], dtype=np.float32)
        result: dict[str, Any] = {
            "ok": True,
            "type": "obs",
            "side": self.side,
            "state": state,
            "images": {name: _policy_image_from_bgr(frame) for name, frame in bgr_frames.items()},
            "timestamp": time.time(),
        }
        if self.args.send_record_images:
            result["record_images"] = {name: _record_image_from_bgr(frame) for name, frame in bgr_frames.items()}
        return result

    def execute_chunk(self, request: dict[str, Any]) -> dict[str, Any]:
        with self._exec_lock:
            if self.args.gripper_action_mode == "width":
                gripper_commands = np.asarray(
                    [
                        _gripper_command_from_width(value, self.args.gripper_width_threshold)
                        for value in np.asarray(request["gripper_commands"]).reshape(-1)
                    ],
                    dtype=np.int64,
                )
            else:
                gripper_commands = np.asarray(
                    [_gripper_command(value) for value in np.asarray(request["gripper_commands"]).reshape(-1)],
                    dtype=np.int64,
                )
            reply = self.arm.execute_chunk(
                joint_chunk=np.asarray(request["joint_chunk"], dtype=np.float64),
                gripper_commands=gripper_commands,
                action_hz=float(request["action_hz"]),
                execute_at=None if request.get("execute_at") is None else float(request["execute_at"]),
            )
            return {"ok": True, "type": "execute_ack", **reply}

    def close(self) -> None:
        for camera in self.cameras.values():
            camera.stop()


async def _handle_client(websocket: ws_server.ServerConnection, agent: HostAgent) -> None:
    logging.info("Connection from %s opened", websocket.remote_address)
    packer = msgpack_numpy.Packer()
    await websocket.send(packer.pack(agent.metadata()))
    while True:
        try:
            request = msgpack_numpy.unpackb(await websocket.recv())
            request_type = request.get("type")
            if request_type == "ping":
                response = {"ok": True, "type": "pong", "side": agent.side, "host_time": time.time()}
            elif request_type == "get_obs":
                response = agent.build_observation()
            elif request_type == "execute_chunk":
                response = await asyncio.to_thread(agent.execute_chunk, request)
            else:
                response = {"ok": False, "error": f"unsupported request type: {request_type}"}
            await websocket.send(packer.pack(response))
        except websockets.ConnectionClosed:
            logging.info("Connection from %s closed", websocket.remote_address)
            break
        except Exception:
            await websocket.send(packer.pack({"ok": False, "error": traceback.format_exc()}))
            await websocket.close(
                code=websockets.frames.CloseCode.INTERNAL_ERROR,
                reason="Host agent internal error. Traceback included in previous frame.",
            )
            raise


def _health_check(connection: ws_server.ServerConnection, request: ws_server.Request) -> ws_server.Response | None:
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    return None


async def _run_server(args: argparse.Namespace) -> None:
    agent = HostAgent(args)
    try:
        async with ws_server.serve(
            lambda websocket: _handle_client(websocket, agent),
            args.host,
            args.port,
            compression=None,
            max_size=None,
            process_request=_health_check,
        ) as server:
            logging.info("Host agent listening on %s:%d side=%s", args.host, args.port, args.side)
            await server.serve_forever()
    finally:
        agent.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Distributed dual-arm Franka host agent.")
    parser.add_argument("--side", choices=("left", "right"), required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--robot-ip", required=True, help="本机 Franka FCI IP，不是控制主机 IP。")
    parser.add_argument(
        "--cameras",
        required=True,
        help="逗号分隔的 name:serial，例如 right_wrist:123,agent_view:456。",
    )
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--camera-fps", type=int, default=30)
    parser.add_argument(
        "--home-joints",
        default="0,-0.785,0,-2.356,0,1.571,0.785",
        help="无机器人测试或 mock state 使用的 7 维关节值。",
    )
    parser.add_argument("--initial-gripper-command", type=int, choices=(0, 1), default=0)
    parser.add_argument("--state-gripper-mode", choices=("command", "width"), default="command")
    parser.add_argument("--gripper-action-mode", choices=("command", "width"), default="command")
    parser.add_argument("--gripper-width-threshold", type=float, default=GRIPPER_WIDTH_THRESHOLD)
    parser.add_argument("--gripper-open-width", type=float, default=0.08)
    parser.add_argument("--gripper-close-width", type=float, default=0.0)
    parser.add_argument("--gripper-speed", type=float, default=0.10)
    parser.add_argument("--gripper-force", type=float, default=5.0)
    parser.add_argument("--joint-vel", type=float, default=2.10)
    parser.add_argument("--joint-acc", type=float, default=5.10)
    parser.add_argument("--max-joint-delta", type=float, default=0.35)
    parser.add_argument("--joint-limit-margin", type=float, default=0.02)
    parser.add_argument("--relative-velocity", type=float, default=0.30)
    parser.add_argument("--relative-acceleration", type=float, default=0.10)
    parser.add_argument("--relative-jerk", type=float, default=0.03)
    parser.add_argument("--send-record-images", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--no-robot", action="store_true", help="只测试相机/网络，不连接或移动 Franka。")
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s", force=True)
    args = parse_args()
    if args.port <= 0:
        raise ValueError("--port must be > 0")
    if args.camera_width <= 0 or args.camera_height <= 0 or args.camera_fps <= 0:
        raise ValueError("camera width/height/fps must be > 0")
    asyncio.run(_run_server(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
