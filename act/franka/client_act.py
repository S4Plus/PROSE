#!/usr/bin/env python3
from __future__ import annotations

"""
Franka 真机 ACT 客户端（Franka SDK + RealSense，无 ROS）。

输入:
1. 两路 RealSense 相机:
   - observation.images.fixed_camera
   - observation.images.wrist_camera
2. 机器人状态:
   - joint 模型: q(7) + gripper_width(1)
   - legacy EE 模型: TCP pose(6) + gripper_width(1) + q(7) + tau_J(7)

输出:
1. 发送 HTTP 请求到 ACT 服务
2. 根据服务端 /health 自动选择执行模式
3. 缓存整段 action chunk 供 rollout 保存

注意:
该脚本直接控制 Franka，不再发布 ROS 话题。
当 action_dim=8 时，首步动作被视为 absolute joint target:
  [joint_0, ..., joint_6, gripper_width]
当 action_dim=7 时，首步动作被视为 absolute EE target:
  [x, y, z, roll, pitch, yaw, gripper_width]
"""

import argparse
import base64
import json
import pickle
import signal
import threading
import time
from pathlib import Path
from typing import Any
from urllib import error, request

import cv2
import numpy as np
import pyrealsense2 as rs
from franky import Affine, CartesianMotion, Gripper, Robot, RobotState
from scipy.spatial.transform import Rotation

try:
    from franky import JointMotion
except ImportError:
    JointMotion = None


try:
    from lerobot.franka_inference_common import DEFAULT_CONFIG_PATH, load_inference_config, reset_robot
except ModuleNotFoundError:
    from franka_inference_common import DEFAULT_CONFIG_PATH, load_inference_config, reset_robot


CAM_FRAME_TIMEOUT_MS = 1000
CAM_START_RETRIES = 5
CAM_START_RETRY_DELAY_S = 0.5
CAM_FPS_CANDIDATES = (15, 6, 5)
CAMERA_PREVIEW_WINDOW = "ACT Front + Wrist Camera Preview"
CAMERA_PREVIEW_WIDTH = 640
CAMERA_PREVIEW_HEIGHT = 480
VIDEO_CODECS = ("MJPG", "XVID")
MAX_POS_STEP_M = 0.005
MAX_ROT_STEP_RAD = 0.05
MAX_JOINT_STEP_RAD = 0.05
_HAND_JOINT_RZ_RAD = -np.pi / 4
_PANDA_EE_Z_OFFSET_M = 0.1468


def http_get(url: str, timeout: float) -> dict[str, object]:
    with request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def http_post(url: str, payload: dict[str, object], timeout: float) -> dict[str, object]:
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    try:
        with request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except error.HTTPError as exc:
        body = exc.read().decode("utf-8") if exc.fp is not None else ""
        if body:
            try:
                return json.loads(body)
            except json.JSONDecodeError:
                pass
        raise


def encode_image(image: np.ndarray) -> str:
    array = np.asarray(image)
    if array.ndim == 2:
        array = np.stack([array] * 3, axis=-1)
    if array.ndim != 3 or array.shape[2] not in (3, 4):
        raise ValueError(f"不支持的图像 shape: {array.shape}")
    if array.shape[2] == 4:
        array = array[:, :, :3]
    if array.dtype != np.uint8:
        max_val = float(array.max()) if array.size else 1.0
        if max_val <= 1.0:
            array = np.clip(array, 0.0, 1.0) * 255.0
        else:
            array = np.clip(array, 0.0, 255.0)
        array = array.astype(np.uint8)
    ok, buf = cv2.imencode(".png", cv2.cvtColor(array, cv2.COLOR_RGB2BGR))
    if not ok:
        raise ValueError("PNG 编码失败。")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def _build_flange_to_tcp() -> np.ndarray:
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_euler("z", _HAND_JOINT_RZ_RAD).as_matrix()
    offset = np.eye(4)
    offset[2, 3] = _PANDA_EE_Z_OFFSET_M
    return transform @ offset


def _affine_to_np(affine: object) -> np.ndarray:
    return np.asarray(affine.matrix)


def pose7_to_matrix(pose7: np.ndarray | list[float]) -> np.ndarray:
    pose = np.asarray(pose7, dtype=np.float64)
    if pose.shape != (7,):
        raise ValueError(f"Expected 7D pose, got {pose.shape}")
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = Rotation.from_euler("xyz", pose[3:6], degrees=False).as_matrix()
    transform[:3, 3] = pose[:3]
    return transform


def move_gripper(gripper: Gripper, width_m: float, *, gripper_max_width: float, gripper_speed: float) -> bool:
    try:
        gripper.move(width_m, gripper_speed)
        return True
    except Exception as exc:
        print(f"  [错误] 夹爪运动失败: {exc}")
        return False


def resolve_args_with_config(args: argparse.Namespace) -> argparse.Namespace:
    config = load_inference_config(args.config)
    args.config_data = config

    if args.server_url is None:
        args.server_url = f"http://{config.policy.host}:{config.policy.port}"
    if args.robot_ip is None:
        args.robot_ip = config.robot_ip
    if args.fixed_camera_serial is None:
        args.fixed_camera_serial = config.front_camera.serial
    if args.wrist_camera_serial is None:
        args.wrist_camera_serial = config.wrist_camera.serial
    if args.fixed_camera_width is None:
        args.fixed_camera_width = args.camera_width if args.camera_width is not None else config.front_camera.width
    if args.fixed_camera_height is None:
        args.fixed_camera_height = args.camera_height if args.camera_height is not None else config.front_camera.height
    if args.fixed_camera_fps is None:
        args.fixed_camera_fps = args.camera_fps if args.camera_fps is not None else config.front_camera.fps
    if args.wrist_camera_width is None:
        args.wrist_camera_width = args.camera_width if args.camera_width is not None else config.wrist_camera.width
    if args.wrist_camera_height is None:
        args.wrist_camera_height = args.camera_height if args.camera_height is not None else config.wrist_camera.height
    if args.wrist_camera_fps is None:
        args.wrist_camera_fps = args.camera_fps if args.camera_fps is not None else config.wrist_camera.fps
    if args.velocity_rel is None:
        args.velocity_rel = config.execution.velocity_rel
    if args.acceleration_rel is None:
        args.acceleration_rel = config.execution.acceleration_rel
    if args.gripper_max_width is None:
        args.gripper_max_width = config.execution.gripper_max_width
    if args.gripper_speed is None:
        args.gripper_speed = config.execution.gripper_speed
    if args.gripper_cmd_threshold is None:
        args.gripper_cmd_threshold = config.execution.gripper_cmd_threshold
    if args.gripper_hold_time_s is None:
        args.gripper_hold_time_s = config.execution.gripper_hold_time_s
    if args.episode_max_steps == 0 and config.episode.max_actions_per_episode > 0:
        args.episode_max_steps = config.episode.max_actions_per_episode

    return args


class ThrottledLogger:
    def __init__(self) -> None:
        self._last_log_time: dict[tuple[str, str], float] = {}

    def warn(self, key: str, period_sec: float, message: str) -> None:
        self._log("warn", key, period_sec, message)

    def info(self, key: str, period_sec: float, message: str) -> None:
        self._log("info", key, period_sec, message)

    def _log(self, level: str, key: str, period_sec: float, message: str) -> None:
        now = time.monotonic()
        throttle_key = (level, key)
        last = self._last_log_time.get(throttle_key)
        if last is not None and now - last < period_sec:
            return
        self._last_log_time[throttle_key] = now
        print(message)


class RealSenseCamera:
    def __init__(self, serial_number: str, name: str, width: int, height: int, fps: int):
        self.name = name
        self.serial_number = serial_number
        self.width = width
        self.height = height
        self.fps = fps
        self.pipeline: rs.pipeline | None = None
        self._thread: threading.Thread | None = None
        self.pipeline = self._start_pipeline()

        self._latest_frame: np.ndarray | None = None
        self._latest_stamp = 0.0
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

        print(
            f"[Camera] {self.name} 已启动 "
            f"(serial={self.serial_number}, {width}x{height}@{self.fps})，等待首帧..."
        )
        while self.get_frame_with_stamp() is None:
            time.sleep(0.01)
        print(f"[Camera] {self.name} 首帧已就绪。")

    def _start_pipeline(self) -> rs.pipeline:
        last_error = None
        fps_candidates: list[int] = []
        for fps in (self.fps, *CAM_FPS_CANDIDATES):
            if fps not in fps_candidates:
                fps_candidates.append(fps)

        for candidate_fps in fps_candidates:
            for attempt in range(1, CAM_START_RETRIES + 1):
                pipeline = rs.pipeline()
                config = rs.config()
                config.enable_device(self.serial_number)
                config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, candidate_fps)
                try:
                    pipeline.start(config)
                    self.fps = candidate_fps
                    if candidate_fps != fps_candidates[0]:
                        print(f"[Camera] {self.name} 回退到 {self.width}x{self.height}@{candidate_fps}")
                    if attempt > 1:
                        print(f"[Camera] {self.name} 第 {attempt} 次启动成功。")
                    return pipeline
                except Exception as exc:
                    last_error = exc
                    print(
                        f"[Camera] {self.name} 启动失败 "
                        f"({self.width}x{self.height}@{candidate_fps})，第 {attempt}/{CAM_START_RETRIES} 次重试: {exc}"
                    )
                    try:
                        pipeline.stop()
                    except Exception:
                        pass
                    if attempt < CAM_START_RETRIES:
                        time.sleep(CAM_START_RETRY_DELAY_S)

        raise RuntimeError(
            f"{self.name} 启动失败，serial={self.serial_number}, "
            f"profile={self.width}x{self.height}@{self.fps}, error={last_error}"
        )

    def _capture_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                if self.pipeline is None:
                    return
                frames = self.pipeline.wait_for_frames(timeout_ms=CAM_FRAME_TIMEOUT_MS)
                color_frame = frames.get_color_frame()
                if color_frame:
                    bgr = np.asanyarray(color_frame.get_data())
                    with self._lock:
                        self._latest_frame = bgr
                        self._latest_stamp = time.time()
            except Exception as exc:
                print(f"[Camera] {self.name} 采帧异常: {exc}")

    def get_frame_with_stamp(self) -> tuple[np.ndarray, float] | None:
        with self._lock:
            if self._latest_frame is None:
                return None
            return self._latest_frame.copy(), self._latest_stamp

    def stop(self) -> None:
        self._stop_event.set()
        try:
            if self._thread is not None and self._thread.is_alive():
                self._thread.join(timeout=1.0)
        except Exception as exc:
            print(f"[Camera] {self.name} 等待采集线程退出失败: {exc}")

        try:
            if self.pipeline is not None:
                self.pipeline.stop()
        except Exception as exc:
            print(f"[Camera] {self.name} 停止失败: {exc}")
        else:
            if self.pipeline is not None:
                print(f"[Camera] {self.name} 已停止。")
        finally:
            self.pipeline = None


class EpisodeRecorder:
    def __init__(self, args: argparse.Namespace, logger: ThrottledLogger):
        self.args = args
        self.logger = logger
        self.enabled = bool(args.save_rollouts_dir)
        self.lock = threading.Lock()

        self.rollout_dir: Path | None = None
        self.video_dir: Path | None = None
        self.next_episode_idx = 0

        self.states: list[np.ndarray] = []
        self.fixed_images: list[np.ndarray] = []
        self.wrist_images: list[np.ndarray] = []
        self.actions: list[np.ndarray] = []
        self.action_chunks: list[np.ndarray] = []
        self.action_chunk_indices: list[int] = []
        self.timestamps: list[float] = []
        self.latencies_ms: list[float] = []

        if not self.enabled:
            return

        self.rollout_dir = Path(args.save_rollouts_dir).expanduser().resolve()
        self.rollout_dir.mkdir(parents=True, exist_ok=True)
        if args.save_video:
            self.video_dir = (
                Path(args.video_dir).expanduser().resolve()
                if args.video_dir
                else self.rollout_dir / "videos"
            )
            self.video_dir.mkdir(parents=True, exist_ok=True)
        self.next_episode_idx = self._discover_next_episode_idx()

    def _discover_next_episode_idx(self) -> int:
        assert self.rollout_dir is not None
        max_idx = -1
        for path in self.rollout_dir.glob("episode_*.pkl"):
            stem = path.stem
            parts = stem.split("_")
            if len(parts) < 2:
                continue
            if parts[0] != "episode" or not parts[1].isdigit():
                continue
            max_idx = max(max_idx, int(parts[1]))
        return max_idx + 1

    def record_step(
        self,
        observation: dict[str, object],
        executed_action: np.ndarray,
        action_chunk: np.ndarray,
        action_chunk_index: int,
        latency_ms: float,
    ) -> None:
        if not self.enabled:
            return

        state = np.asarray(observation["state"], dtype=np.float32).copy()
        images = observation["images"]
        assert isinstance(images, dict)
        fixed_image = np.asarray(images["observation.images.fixed_camera"], dtype=np.uint8).copy()
        wrist_image = np.asarray(images["observation.images.wrist_camera"], dtype=np.uint8).copy()

        with self.lock:
            self.states.append(state)
            self.fixed_images.append(fixed_image)
            self.wrist_images.append(wrist_image)
            self.actions.append(np.asarray(executed_action, dtype=np.float32).copy())
            self.action_chunks.append(np.asarray(action_chunk, dtype=np.float32).copy())
            self.action_chunk_indices.append(int(action_chunk_index))
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
            payload = {
                "episode_idx": episode_idx,
                "states": np.stack(self.states, axis=0),
                "fixed_images": np.stack(self.fixed_images, axis=0),
                "wrist_images": np.stack(self.wrist_images, axis=0),
                "actions": np.stack(self.actions, axis=0),
                "action_chunks": [chunk.copy() for chunk in self.action_chunks],
                "action_chunk_indices": np.asarray(self.action_chunk_indices, dtype=np.int32),
                "timestamps": np.asarray(self.timestamps, dtype=np.float64),
                "latencies_ms": np.asarray(self.latencies_ms, dtype=np.float32),
            }

            self.states.clear()
            self.fixed_images.clear()
            self.wrist_images.clear()
            self.actions.clear()
            self.action_chunks.clear()
            self.action_chunk_indices.clear()
            self.timestamps.clear()
            self.latencies_ms.clear()
            return payload

    def _compose_video_frames(self, fixed_images: np.ndarray, wrist_images: np.ndarray) -> list[np.ndarray]:
        frames: list[np.ndarray] = []
        for fixed_rgb, wrist_rgb in zip(fixed_images, wrist_images, strict=True):
            if self.args.video_camera == "fixed":
                frames.append(cv2.cvtColor(fixed_rgb, cv2.COLOR_RGB2BGR))
                continue
            if self.args.video_camera == "wrist":
                frames.append(cv2.cvtColor(wrist_rgb, cv2.COLOR_RGB2BGR))
                continue

            target_height = max(fixed_rgb.shape[0], wrist_rgb.shape[0])
            fixed_resized = cv2.resize(
                fixed_rgb,
                (int(round(fixed_rgb.shape[1] * target_height / fixed_rgb.shape[0])), target_height),
            )
            wrist_resized = cv2.resize(
                wrist_rgb,
                (int(round(wrist_rgb.shape[1] * target_height / wrist_rgb.shape[0])), target_height),
            )
            merged = np.concatenate([fixed_resized, wrist_resized], axis=1)
            frames.append(cv2.cvtColor(merged, cv2.COLOR_RGB2BGR))
        return frames

    def _save_video(self, episode_idx: int, success: bool, fixed_images: np.ndarray, wrist_images: np.ndarray) -> None:
        if self.video_dir is None:
            return
        frames = self._compose_video_frames(fixed_images, wrist_images)
        if not frames:
            return
        suffix = "success" if success else "failure"
        out_path = self.video_dir / f"episode_{episode_idx:06d}_{suffix}.avi"
        height, width = frames[0].shape[:2]

        writer = None
        for codec in VIDEO_CODECS:
            candidate = cv2.VideoWriter(
                str(out_path),
                cv2.VideoWriter_fourcc(*codec),
                float(self.args.video_fps),
                (width, height),
            )
            if candidate.isOpened():
                writer = candidate
                break
            candidate.release()
        if writer is None:
            print(f"[act-client] 视频写入器创建失败: {out_path}")
            return

        for frame in frames:
            writer.write(frame)
        writer.release()
        print(f"[act-client] 已保存视频: {out_path}")

    def _write_episode(self, episode: dict[str, object], success: bool, reason: str) -> None:
        episode_idx = int(episode["episode_idx"])
        reward = float(success)
        num_steps = int(np.asarray(episode["actions"]).shape[0])
        rewards = np.zeros((num_steps,), dtype=np.float32)
        dones = np.zeros((num_steps,), dtype=bool)
        successes = np.zeros((num_steps,), dtype=bool)
        rewards[-1] = reward
        dones[-1] = True
        successes[-1] = success

        payload = {
            "observations": {
                "state": np.asarray(episode["states"], dtype=np.float32),
                "images": {
                    "fixed_camera": np.asarray(episode["fixed_images"], dtype=np.uint8),
                    "wrist_camera": np.asarray(episode["wrist_images"], dtype=np.uint8),
                },
            },
            "actions": np.asarray(episode["actions"], dtype=np.float32),
            "reward": reward,
            "rewards": rewards,
            "done": dones,
            "success": success,
            "successes": successes,
            "metadata": {
                "source": "franka_sdk_act_client",
                "episode": episode_idx,
                "reason": reason,
                "timestamps": np.asarray(episode["timestamps"], dtype=np.float64),
                "latencies_ms": np.asarray(episode["latencies_ms"], dtype=np.float32),
                "action_chunks": [np.asarray(chunk, dtype=np.float32) for chunk in episode["action_chunks"]],
                "action_chunk_indices": np.asarray(episode["action_chunk_indices"], dtype=np.int32),
            },
        }

        assert self.rollout_dir is not None
        suffix = "success" if success else "failure"
        out_path = self.rollout_dir / f"episode_{episode_idx:06d}_{suffix}.pkl"
        with out_path.open("wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"[act-client] 已保存 rollout: {out_path}")

        if self.video_dir is not None:
            self._save_video(
                episode_idx=episode_idx,
                success=success,
                fixed_images=np.asarray(episode["fixed_images"], dtype=np.uint8),
                wrist_images=np.asarray(episode["wrist_images"], dtype=np.uint8),
            )

    def finalize_episode_with_label(self, success: bool, reason: str = "manual") -> None:
        if not self.enabled:
            return
        episode = self._pop_episode_data()
        if episode is None:
            self.logger.warn("empty_episode", 1.0, "[act-client] 当前 episode 为空，跳过保存。")
            return
        self._write_episode(episode, success=success, reason=reason)

    def finalize_episode_interactive(self, reason: str = "manual") -> None:
        if not self.enabled:
            return
        episode = self._pop_episode_data()
        if episode is None:
            self.logger.warn("empty_episode", 1.0, "[act-client] 当前 episode 为空，跳过保存。")
            return

        episode_idx = int(episode["episode_idx"])
        while True:
            raw = input(
                f"[act-client] episode {episode_idx:06d} 已结束 ({reason})，请输入结果 0=失败, 1=成功: "
            ).strip()
            if raw in ("0", "1"):
                self._write_episode(episode, success=(raw == "1"), reason=reason)
                return
            print("[act-client] 输入非法，只能输入 0 或 1。")

    def close(self) -> None:
        if self.enabled and self.args.save_on_exit and self.num_steps > 0:
            self.finalize_episode_interactive(reason="shutdown")


class FrankaACTClient:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.config = args.config_data
        self.server_url = args.server_url.rstrip("/")
        self.logger = ThrottledLogger()
        self._tcp_transform = _build_flange_to_tcp()
        self._tcp_inverse_transform = np.linalg.inv(self._tcp_transform)

        print(f"[Robot] 连接 Franka: {args.robot_ip} ...")
        self.robot = Robot(args.robot_ip)
        self.gripper = Gripper(args.robot_ip)
        self.robot.set_collision_behavior([20.0] * 7, [20.0] * 7, [10.0] * 6, [10.0] * 6)
        print("[Robot] 连接成功。")

        self.fixed_camera: RealSenseCamera | None = None
        self.wrist_camera: RealSenseCamera | None = None
        try:
            self.fixed_camera = RealSenseCamera(
                serial_number=args.fixed_camera_serial,
                name="fixed_camera",
                width=args.fixed_camera_width,
                height=args.fixed_camera_height,
                fps=args.fixed_camera_fps,
            )
            self.wrist_camera = RealSenseCamera(
                serial_number=args.wrist_camera_serial,
                name="wrist_camera",
                width=args.wrist_camera_width,
                height=args.wrist_camera_height,
                fps=args.wrist_camera_fps,
            )
        except Exception:
            if self.fixed_camera is not None:
                self.fixed_camera.stop()
            if self.wrist_camera is not None:
                self.wrist_camera.stop()
            raise

        self.episode_recorder = EpisodeRecorder(args, self.logger)
        self._gripper_thread: threading.Thread | None = None
        self._preview_enabled = not args.no_preview
        self.frame_idx = 0
        self._stop_requested = False
        self._episode_index = 0
        self._executed_action_count = 0
        self._ready_after_reset = False
        self.server_state_dim: int | None = None
        self.server_action_dim: int | None = None
        self.control_mode: str | None = None

    def _get_robot_state(self) -> tuple[list[float], list[float], float]:
        state: RobotState = self.robot.state
        q = list(state.q)
        tau_j = list(state.tau_J)
        gripper_width = self.gripper.state.width
        return q, tau_j, gripper_width

    def _get_tcp_pose(self) -> np.ndarray:
        base_to_flange = _affine_to_np(self.robot.current_cartesian_state.pose.end_effector_pose)
        return base_to_flange @ self._tcp_transform

    def _get_flange_pose(self) -> np.ndarray:
        return _affine_to_np(self.robot.current_cartesian_state.pose.end_effector_pose)

    def _compose_preview_frame(self, fixed_bgr: np.ndarray, wrist_bgr: np.ndarray) -> np.ndarray:
        fixed_preview = cv2.resize(fixed_bgr, (CAMERA_PREVIEW_WIDTH, CAMERA_PREVIEW_HEIGHT))
        wrist_preview = cv2.resize(wrist_bgr, (CAMERA_PREVIEW_WIDTH, CAMERA_PREVIEW_HEIGHT))
        preview = np.hstack((fixed_preview, wrist_preview))
        cv2.putText(preview, "fixed_camera", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(
            preview,
            "wrist_camera",
            (CAMERA_PREVIEW_WIDTH + 12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )
        return preview

    def _show_camera_preview(self, fixed_bgr: np.ndarray, wrist_bgr: np.ndarray) -> None:
        if not self._preview_enabled:
            return
        preview = self._compose_preview_frame(fixed_bgr, wrist_bgr)
        try:
            cv2.imshow(CAMERA_PREVIEW_WINDOW, preview)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                raise KeyboardInterrupt
        except cv2.error as exc:
            self._preview_enabled = False
            print(f"[Camera] 预览窗口不可用，已关闭图像展示: {exc}")

    def _close_camera_preview(self) -> None:
        if not self._preview_enabled:
            return
        try:
            cv2.destroyWindow(CAMERA_PREVIEW_WINDOW)
            cv2.waitKey(1)
        except cv2.error:
            pass

    def _recover_if_needed(self) -> None:
        if not self.robot.has_errors:
            return
        print("[Robot] 检测到错误状态，尝试恢复...")
        try:
            self.robot.recover_from_errors()
            print("[Robot] 恢复完成。")
        except Exception as exc:
            print(f"[Robot] 恢复失败: {exc}")

    def _reset_episode_counters(self) -> None:
        self._executed_action_count = 0

    def _wait_for_episode_command(self) -> str:
        print(
            "\n[Episode] 等待指令: "
            "`r` 先归位, `t` 开始本轮, `q` 退出。"
        )
        while True:
            try:
                command = input("[Episode] 请输入指令 [r/t/q]: ").strip().lower()
            except EOFError:
                return "q"
            if command in {"r", "t", "q"}:
                print(f"[Episode] 收到指令: {command}")
                return command
            print("[Episode] 无效指令，请输入 r / t / q。")

    def perform_reset(self) -> bool:
        print("[Reset] 状态: START")
        self._recover_if_needed()
        try:
            ok = bool(reset_robot(self.robot, self.gripper, self.config))
        except Exception as exc:
            print(f"[Reset] 执行失败: {exc}")
            ok = False
        self._recover_if_needed()
        print("[Reset] 状态: OK" if ok else "[Reset] 状态: FAILED")
        self._ready_after_reset = ok
        return ok

    def _build_observation(self) -> dict[str, object] | None:
        fixed = self.fixed_camera.get_frame_with_stamp()
        wrist = self.wrist_camera.get_frame_with_stamp()
        if fixed is None or wrist is None:
            self.logger.warn("missing_frames", 1.0, "[act-client] 未获取到 front/wrist 相机图像，跳过本轮推理。")
            return None

        fixed_bgr, fixed_stamp = fixed
        wrist_bgr, wrist_stamp = wrist
        now = time.time()
        if any(now - stamp > self.args.max_message_age for stamp in (fixed_stamp, wrist_stamp)):
            self.logger.warn("stale_inputs", 1.0, "[act-client] 输入消息过旧，跳过本轮推理。")
            return None

        self._show_camera_preview(fixed_bgr, wrist_bgr)

        q, tau_j, gripper_width = self._get_robot_state()
        if self.control_mode == "joint":
            state = np.array([*map(float, q), float(gripper_width)], dtype=np.float32)
        elif self.control_mode == "ee":
            tcp_pose = self._get_tcp_pose()
            position = tcp_pose[:3, 3]
            euler = Rotation.from_matrix(tcp_pose[:3, :3]).as_euler("xyz", degrees=False)
            state_values = [
                float(position[0]),
                float(position[1]),
                float(position[2]),
                float(euler[0]),
                float(euler[1]),
                float(euler[2]),
                float(gripper_width),
            ]
            state_values.extend(float(joint) for joint in q)
            state_values.extend(float(effort) for effort in tau_j)
            state = np.array(state_values, dtype=np.float32)
        else:
            raise RuntimeError("服务端控制模式未初始化，请先完成 /health 检查。")

        return {
            "images": {
                "observation.images.fixed_camera": cv2.cvtColor(fixed_bgr, cv2.COLOR_BGR2RGB),
                "observation.images.wrist_camera": cv2.cvtColor(wrist_bgr, cv2.COLOR_BGR2RGB),
            },
            "state": state,
        }

    def _matrix_to_pose7(self, pose: np.ndarray, gripper_width: float) -> np.ndarray:
        position = pose[:3, 3]
        euler = Rotation.from_matrix(pose[:3, :3]).as_euler("xyz", degrees=False)
        return np.array(
            [
                float(position[0]),
                float(position[1]),
                float(position[2]),
                float(euler[0]),
                float(euler[1]),
                float(euler[2]),
                float(gripper_width),
            ],
            dtype=np.float32,
        )

    def check_server(self) -> None:
        health = http_get(f"{self.server_url}/health", float(self.args.timeout))
        if health.get("status") not in ("ok", "healthy"):
            raise RuntimeError(f"服务端健康检查失败: {health}")

        image_keys = list(health.get("image_keys", []))
        expected = [
            "observation.images.fixed_camera",
            "observation.images.wrist_camera",
        ]
        if image_keys != expected:
            raise RuntimeError(
                f"服务端 image_keys={image_keys}，与当前 Franka 模型预期 {expected} 不一致。"
            )
        if not bool(health.get("expect_state", False)):
            raise RuntimeError("服务端返回 expect_state=false，但当前 Franka client 依赖 state 输入。")

        self.server_state_dim = int(health.get("state_dim", -1))
        self.server_action_dim = int(health.get("action_dim", -1))
        if self.server_state_dim == 8 and self.server_action_dim == 8:
            self.control_mode = "joint"
        elif self.server_state_dim == 21 and self.server_action_dim == 7:
            self.control_mode = "ee"
        else:
            raise RuntimeError(
                "当前 client 仅支持两种服务端协议: "
                f"(state_dim=8, action_dim=8) joint 或 "
                f"(state_dim=21, action_dim=7) legacy EE，实际收到 {health}"
            )

        print(
            "[act-client] 服务端已连接: image_keys=%s, default_chunk=%s, "
            "state_dim=%s, action_dim=%s, control_mode=%s"
            % (
                image_keys,
                health.get("default_chunk_size"),
                self.server_state_dim,
                self.server_action_dim,
                self.control_mode,
            )
        )

    def _command_gripper(self, target_gripper: float) -> float:
        target_gripper = float(target_gripper)
        current_gripper = float(self.gripper.state.width)
        gripper_delta = abs(target_gripper - current_gripper)
        print(
            "[Gripper] current=%.4f target=%.4f delta=%.4f"
            % (
                current_gripper,
                target_gripper,
                gripper_delta,
            )
        )

        # 直接发送模型输出的连续夹爪宽度，不再用 deadband 阈值吞掉小幅闭合指令。
        move_gripper(
            gripper=self.gripper,
            width_m=target_gripper,
            gripper_max_width=self.args.gripper_max_width,
            gripper_speed=self.args.gripper_speed,
        )
        time.sleep(self.args.gripper_hold_time_s)
        return target_gripper

    def _clip_joint_target(self, current_q: np.ndarray, target_q: np.ndarray) -> np.ndarray:
        max_step = float(self.args.max_joint_step_rad)
        if max_step <= 0.0:
            return target_q

        delta = target_q - current_q
        clipped_delta = np.clip(delta, -max_step, max_step)
        if np.any(np.abs(delta - clipped_delta) > 1e-9):
            print(
                "[Exec] joint target 超出单步限幅，已裁剪: "
                f"max_step={max_step:.4f}rad, "
                f"raw_delta={np.array2string(delta, precision=4)}, "
                f"clipped_delta={np.array2string(clipped_delta, precision=4)}"
            )
        return current_q + clipped_delta

    def _build_joint_motion(self, target_q: np.ndarray) -> object:
        if JointMotion is None:
            raise RuntimeError("当前环境缺少 franky.JointMotion，无法执行 8 维 joint action。")

        kwargs_candidates = [
            {
                "relative_dynamics_factor": self.robot.relative_dynamics_factor,
                "return_when_finished": True,
            },
            {"relative_dynamics_factor": self.robot.relative_dynamics_factor},
            {"return_when_finished": True},
            {},
        ]
        last_error: Exception | None = None
        target_list = target_q.astype(float).tolist()
        for kwargs in kwargs_candidates:
            try:
                return JointMotion(target_list, **kwargs)
            except TypeError as exc:
                last_error = exc
        raise RuntimeError(f"JointMotion 构造失败，请检查 franky 版本兼容性: {last_error}")

    def _send_absolute_joint_action(self, target_joint8: np.ndarray | list[float]) -> np.ndarray | None:
        target = np.asarray(target_joint8, dtype=np.float64)
        if target.shape != (8,):
            print(f"  [错误] absolute joint action 维度错误: {target.shape}")
            return None
        if np.any(~np.isfinite(target)):
            print("  [错误] absolute joint action 包含非法数值。")
            return None

        commanded_joint8 = target.copy()
        commanded_joint8[7] = self._command_gripper(float(commanded_joint8[7]))
        self.robot.relative_dynamics_factor = min(self.args.velocity_rel, self.args.acceleration_rel)

        try:
            self.robot.move(self._build_joint_motion(commanded_joint8[:7]))
            return commanded_joint8.astype(np.float32)
        except Exception as exc:
            print(f"  [错误] absolute joint 执行失败: {exc}")
            return None

    def _send_absolute_action(self, target_pose7: np.ndarray | list[float]) -> np.ndarray | None:
        target = np.asarray(target_pose7, dtype=np.float64)
        if target.shape != (7,):
            print(f"  [错误] absolute EE action 维度错误: {target.shape}")
            return None
        if np.any(~np.isfinite(target)):
            print("  [错误] absolute EE action 包含非法数值。")
            return None

        commanded_pose7 = target.copy()
        commanded_pose7[6] = self._command_gripper(float(commanded_pose7[6]))
        self.robot.relative_dynamics_factor = min(self.args.velocity_rel, self.args.acceleration_rel)

        target_base_to_tcp = pose7_to_matrix(commanded_pose7)
        target_base_to_flange = target_base_to_tcp @ self._tcp_inverse_transform
        target_flange_affine = Affine(target_base_to_flange)
        try:
            self.robot.move(
                CartesianMotion(
                    target_flange_affine,
                    relative_dynamics_factor=self.robot.relative_dynamics_factor,
                    return_when_finished=True,
                )
            )
            return commanded_pose7.astype(np.float32)
        except Exception as exc:
            print(f"  [错误] absolute EE 执行失败: {exc}")
            return None

    def _execute_action_with_observation(
        self,
        observation: dict[str, object],
        action_chunk: np.ndarray,
        action_idx: int,
        latency_ms: float,
    ) -> bool:
        state = np.asarray(observation["state"], dtype=np.float32)
        target_action = np.asarray(action_chunk[action_idx], dtype=np.float64)

        print(
            f"[Exec] 执行 chunk step {action_idx + 1}/{action_chunk.shape[0]}"
        )

        if self.control_mode == "joint":
            curr_joint8 = state[:8]
            print(f"[Exec] curr_joint8   : {np.array2string(curr_joint8, precision=4)}")
            print(f"[Exec] target_joint8 : {np.array2string(target_action, precision=4)}")
            executed_action = self._send_absolute_joint_action(target_action)
        else:
            curr_pose7 = state[:7]
            flange_pose7 = self._matrix_to_pose7(self._get_flange_pose(), float(self.gripper.state.width))
            print(f"[Exec] curr_pose7   : {np.array2string(curr_pose7, precision=4)}")
            print(f"[Exec] flange_pose7 : {np.array2string(flange_pose7, precision=4)}")
            print(f"[Exec] target_pose7 : {np.array2string(target_action, precision=4)}")
            executed_action = self._send_absolute_action(target_action)
        if executed_action is None:
            self._recover_if_needed()
            return False

        self.episode_recorder.record_step(
            observation=observation,
            executed_action=executed_action,
            action_chunk=action_chunk,
            action_chunk_index=action_idx,
            latency_ms=latency_ms,
        )
        if self.args.log_every > 0 and self.frame_idx % self.args.log_every == 0:
            print(
                "[act-client] frame=%d latency=%.2fms action=%s"
                % (self.frame_idx, latency_ms, np.array2string(target_action, precision=4))
            )
        self.frame_idx += 1
        return True

    def infer_once(self) -> int:
        observation = self._build_observation()
        if observation is None:
            return 0

        state = np.asarray(observation["state"], dtype=np.float32)
        payload = {
            "images": {key: encode_image(value) for key, value in observation["images"].items()},
            "state": state.astype(float).tolist(),
            "chunk_size": int(self.args.chunk_size),
        }

        start = time.time()
        result = http_post(f"{self.server_url}/infer", payload, float(self.args.timeout))
        latency_ms = (time.time() - start) * 1000.0
        if "error" in result:
            self.logger.warn("infer_error", 1.0, f"[act-client] 推理失败: {result['error']}")
            return 0

        action_chunk = np.asarray(result.get("action", []), dtype=np.float32)
        if action_chunk.size == 0:
            self.logger.warn("empty_action", 1.0, "[act-client] 服务端返回空动作。")
            return 0
        action_chunk = np.atleast_2d(action_chunk)
        expected_action_dim = int(self.server_action_dim or -1)
        if action_chunk.shape[1] != expected_action_dim:
            self.logger.warn("bad_action_shape", 1.0, f"[act-client] 返回动作维度错误: {action_chunk.shape}")
            return 0

        if self.control_mode == "joint" and action_chunk.shape[1] >= 8:
            gripper_chunk = action_chunk[:, 7]
            preview = np.round(gripper_chunk[:10], 6).tolist()
            print(
                "[Infer] gripper_chunk first10=%s min=%.6f max=%.6f mean=%.6f"
                % (
                    preview,
                    float(gripper_chunk.min()),
                    float(gripper_chunk.max()),
                    float(gripper_chunk.mean()),
                )
            )

        exec_steps = int(self.args.chunk_exec_steps)
        if exec_steps <= 0:
            exec_steps = int(action_chunk.shape[0])
        exec_steps = min(exec_steps, int(action_chunk.shape[0]))
        executed_steps = 0

        for action_idx in range(exec_steps):
            if action_idx == 0:
                step_observation = observation
            else:
                step_observation = self._build_observation()
                if step_observation is None:
                    self.logger.warn(
                        "missing_mid_chunk_obs",
                        1.0,
                        f"[act-client] chunk 第 {action_idx + 1} 步缺少有效观测，提前结束本次 chunk 执行。",
                    )
                    break

            ok = self._execute_action_with_observation(
                observation=step_observation,
                action_chunk=action_chunk,
                action_idx=action_idx,
                latency_ms=latency_ms,
            )
            if not ok:
                break
            executed_steps += 1

        return executed_steps

    def _run_single_episode(self) -> None:
        self._episode_index += 1
        self._reset_episode_counters()
        print(
            f"\n[Episode] 第 {self._episode_index} 轮开始，"
            f"最多执行 {self.args.episode_max_steps} 个动作。"
        )
        print("[Episode] 状态: RUNNING")

        period_s = 1.0 / self.args.frame_rate
        while not self._stop_requested:
            if self.args.episode_max_steps > 0 and self._executed_action_count >= self.args.episode_max_steps:
                break

            loop_start = time.time()
            try:
                executed_steps = self.infer_once()
            except Exception as exc:
                self.logger.warn("main_loop_exception", 1.0, f"[act-client] 主循环异常: {exc}")
                executed_steps = 0

            if executed_steps > 0:
                self._executed_action_count += executed_steps
            else:
                self._recover_if_needed()

            print(
                f"[Episode] 当前 episode 累计成功执行 {self._executed_action_count} 个动作。"
            )
            elapsed = time.time() - loop_start
            time.sleep(max(0.0, period_s - elapsed))

        print(f"[Episode] 第 {self._episode_index} 轮执行结束。")
        print("[Episode] 状态: IDLE")
        self._ready_after_reset = False
        if self.episode_recorder.enabled and self.episode_recorder.num_steps > 0:
            self.episode_recorder.finalize_episode_interactive(reason="episode_end")

    def run(self) -> None:
        self.check_server()
        print("\n[act-client] Episode 模式已启动。")
        print("[act-client] 当前为待命状态，先按 `r` 归位，再按 `t` 开始本轮。")
        print("[act-client] 按 Ctrl+C 可随时停止。")
        try:
            while not self._stop_requested:
                command = self._wait_for_episode_command()
                if command == "q":
                    print("[act-client] 收到退出指令。")
                    return
                if command == "r":
                    reset_ok = self.perform_reset()
                    print("[Episode] 已 reset，当前仍为待命状态。")
                    if not reset_ok:
                        print("[Episode] reset 失败，请检查机械臂状态后再继续。")
                    continue
                if not self._ready_after_reset:
                    print("[Episode] 当前未归位，请先按 `r` 归位，再按 `t` 开始。")
                    continue
                print("[Episode] 已完成归位，开始执行。")
                self._run_single_episode()
        except KeyboardInterrupt:
            print("\n[act-client] 收到 Ctrl+C，停止。")
        finally:
            self.close()

    def close(self) -> None:
        old_sigint_handler = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        try:
            self.episode_recorder.close()
            self._close_camera_preview()
            if self.fixed_camera is not None:
                self.fixed_camera.stop()
            if self.wrist_camera is not None:
                self.wrist_camera.stop()
        finally:
            signal.signal(signal.SIGINT, old_sigint_handler)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Franka 真机 ACT HTTP 客户端（Franka SDK + RealSense）")
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG_PATH),
        help="Franka JSON 配置文件路径，默认复用 lerobot/franka_pi0_inference_config.json",
    )
    parser.add_argument("--server-url", type=str, default=None, help="ACT HTTP 服务地址，未提供则从 config.policy 生成")
    parser.add_argument("--timeout", type=float, default=5.0, help="HTTP 请求超时时间（秒）")
    parser.add_argument("--chunk-size", type=int, default=50, help="每次请求的动作步数，真机默认建议 1")
    parser.add_argument(
        "--chunk-exec-steps",
        type=int,
        default=6,
        help="每次推理后连续执行多少个模型输出动作；设为 <=0 表示执行完整 action chunk",
    )
    parser.add_argument("--frame-rate", type=float, default=10.0, help="客户端推理频率（Hz）")
    parser.add_argument("--log-every", type=int, default=10, help="每 N 帧打印一次日志")
    parser.add_argument("--max-message-age", type=float, default=0.3, help="允许图像消息的最大陈旧时间（秒）")
    parser.add_argument("--robot-ip", type=str, default=None, help="Franka 机械臂 IP，未提供则从 config 读取")
    parser.add_argument("--fixed-camera-serial", type=str, default=None, help="固定相机序列号，未提供则从 config 读取")
    parser.add_argument("--wrist-camera-serial", type=str, default=None, help="腕部相机序列号，未提供则从 config 读取")
    parser.add_argument("--camera-width", type=int, default=None, help="兼容旧参数；若提供则默认覆盖两路相机宽度")
    parser.add_argument("--camera-height", type=int, default=None, help="兼容旧参数；若提供则默认覆盖两路相机高度")
    parser.add_argument("--camera-fps", type=int, default=None, help="兼容旧参数；若提供则默认覆盖两路相机帧率")
    parser.add_argument("--fixed-camera-width", type=int, default=None, help="固定相机图像宽度，未提供则从 config.front 读取")
    parser.add_argument("--fixed-camera-height", type=int, default=None, help="固定相机图像高度，未提供则从 config.front 读取")
    parser.add_argument("--fixed-camera-fps", type=int, default=None, help="固定相机帧率，未提供则从 config.front 读取")
    parser.add_argument("--wrist-camera-width", type=int, default=None, help="腕部相机图像宽度，未提供则从 config.wrist 读取")
    parser.add_argument("--wrist-camera-height", type=int, default=None, help="腕部相机图像高度，未提供则从 config.wrist 读取")
    parser.add_argument("--wrist-camera-fps", type=int, default=None, help="腕部相机帧率，未提供则从 config.wrist 读取")
    parser.add_argument("--no-preview", action="store_true", help="关闭双目预览窗口")
    parser.add_argument("--velocity-rel", type=float, default=None, help="Franka 相对速度系数，未提供则从 config 读取")
    parser.add_argument("--acceleration-rel", type=float, default=None, help="Franka 相对加速度系数，未提供则从 config 读取")
    parser.add_argument(
        "--max-pos-step",
        type=float,
        default=MAX_POS_STEP_M,
        help="保留参数；当前 absolute EE pose 控制路径不使用该裁剪",
    )
    parser.add_argument(
        "--max-rot-step",
        type=float,
        default=MAX_ROT_STEP_RAD,
        help="保留参数；当前 absolute EE pose 控制路径不使用该裁剪",
    )
    parser.add_argument(
        "--max-joint-step-rad",
        type=float,
        default=MAX_JOINT_STEP_RAD,
        help="joint 模型下每步最大关节变化量（弧度）；设为 <=0 可关闭限幅",
    )
    parser.add_argument("--gripper-max-width", type=float, default=None, help="夹爪最大开度（米），未提供则从 config 读取")
    parser.add_argument("--gripper-speed", type=float, default=None, help="夹爪运动速度，未提供则从 config 读取")
    parser.add_argument("--gripper-cmd-threshold", type=float, default=None, help="夹爪最小触发阈值，未提供则从 config 读取")
    parser.add_argument("--gripper-hold-time-s", type=float, default=None, help="夹爪指令后等待时间，未提供则从 config 读取")
    parser.add_argument("--save-rollouts-dir", type=str, default=None, help="保存 episode_xxxxxx_*.pkl 的目录")
    parser.add_argument("--save-video", action="store_true", help="保存每个 episode 对应的视频")
    parser.add_argument("--video-dir", type=str, default=None, help="视频输出目录，默认 <save-rollouts-dir>/videos")
    parser.add_argument("--video-fps", type=float, default=10.0, help="保存视频时使用的 FPS")
    parser.add_argument(
        "--video-camera",
        choices=["fixed", "wrist", "concat"],
        default="concat",
        help="视频使用哪一路相机，concat 表示左右拼接",
    )
    parser.add_argument(
        "--episode-max-steps",
        type=int,
        default=0,
        help="每个 episode 最多执行多少个动作；达到后会提示输入 0/1 保存",
    )
    parser.add_argument(
        "--save-on-exit",
        action="store_true",
        help="退出程序时如果当前 episode 有缓存，则提示输入 0/1 并保存",
    )
    return parser


def main() -> None:
    args = resolve_args_with_config(build_parser().parse_args())
    print(f"[Config] loaded from: {Path(args.config).expanduser().resolve()}")
    print(f"[Config] robot_ip={args.robot_ip}")
    print(f"[Config] server_url={args.server_url}")
    print(
        "[Config] cameras="
        f"front({args.fixed_camera_serial}, {args.fixed_camera_width}x{args.fixed_camera_height}@{args.fixed_camera_fps}), "
        f"wrist({args.wrist_camera_serial}, {args.wrist_camera_width}x{args.wrist_camera_height}@{args.wrist_camera_fps})"
    )
    client = FrankaACTClient(args)
    client.run()


if __name__ == "__main__":
    main()
