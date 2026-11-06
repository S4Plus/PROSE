#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#新的模型要修改prompt Pick up the blue cube and stack it on top of the red cube.
"""
Franka Robot pi0/pi05 policy inference and execution (RealSense, no ROS).
"""

from __future__ import annotations

import argparse
import pickle
import signal
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs
from franky import CartesianVelocityMotion, Duration, Gripper, Robot, RobotState, Twist
from openpi_client import websocket_client_policy
from scipy.spatial.transform import Rotation

try:
    from franky import JointMotion
except ImportError:
    JointMotion = None

from franka_inference_common import (
    DEFAULT_CONFIG_PATH,
    InferenceConfig,
    load_inference_config,
    move_gripper,
    reset_robot,
)


CAM_FRAME_TIMEOUT_MS = 1000
CAM_START_RETRIES = 5
CAM_START_RETRY_DELAY_S = 0.5
CAM_FPS_CANDIDATES = (15, 6, 5)
CAMERA_PREVIEW_WINDOW = "Front + Wrist Camera Preview"
CAMERA_PREVIEW_WIDTH = 480
CAMERA_PREVIEW_HEIGHT = 270
INFERENCE_RESULT_DIR = Path(__file__).resolve().parent / "inference_result"
VIDEO_FILENAME_PREFIX = "inference_"
VIDEO_FILENAME_SUFFIX = ".avi"
VIDEO_CODECS = ("MJPG", "XVID")
ROLLOUT_DIR = INFERENCE_RESULT_DIR / "rollouts"
ROLLOUT_VIDEO_DIR = ROLLOUT_DIR / "videos"
IMG_W, IMG_H = 224, 224
_HAND_JOINT_RZ_RAD = -np.pi / 4
_PANDA_EE_Z_OFFSET_M = 0.1468
CONTROL_CYCLE_MS = 34
MAX_JOINT_STEP_RAD = 0.05
FIXED_CHUNK_EXEC_STEPS = 20
GRIPPER_CLOSE_EPSILON_M = 1e-4
GRIPPER_GRASP_FORCE_N = 20.0
GRIPPER_GRASP_EPSILON_OUTER = 1.0


def _build_flange_to_tcp() -> np.ndarray:
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_euler("z", _HAND_JOINT_RZ_RAD).as_matrix()
    offset = np.eye(4)
    offset[2, 3] = _PANDA_EE_Z_OFFSET_M
    return transform @ offset


def _affine_to_np(affine: object) -> np.ndarray:
    return np.asarray(affine.matrix)


def abs_ee_to_delta_cmd(
    curr_pose7: np.ndarray | list[float],
    target_pose7: np.ndarray | list[float],
    *,
    max_gripper_width: float,
) -> tuple[np.ndarray, dict[str, float]]:
    curr = np.asarray(curr_pose7, dtype=np.float64)
    target = np.asarray(target_pose7, dtype=np.float64)

    if curr.shape != (7,) or target.shape != (7,):
        raise ValueError(f"Expected 7D poses, got curr={curr.shape}, target={target.shape}")

    # 保持与当前 joint 路径一致，不在客户端额外裁减模型输出。
    delta_pos = target[:3] - curr[:3]

    curr_rot = Rotation.from_euler("xyz", curr[3:6], degrees=False)
    target_rot = Rotation.from_euler("xyz", target[3:6], degrees=False)
    delta_rot = (target_rot * curr_rot.inv()).as_rotvec()

    target_gripper = float(target[6])
    delta_cmd7 = np.concatenate((delta_pos, delta_rot, [target_gripper]), axis=0)
    delta_action = {
        "delta_x": float(delta_pos[0]),
        "delta_y": float(delta_pos[1]),
        "delta_z": float(delta_pos[2]),
        "delta_roll": float(delta_rot[0]),
        "delta_pitch": float(delta_rot[1]),
        "delta_yaw": float(delta_rot[2]),
        "gripper": target_gripper,
    }
    return delta_cmd7, delta_action


def delta_cmd_to_abs_pose(curr_pose7: np.ndarray | list[float], delta_cmd7: np.ndarray | list[float]) -> np.ndarray:
    curr = np.asarray(curr_pose7, dtype=np.float64)
    delta = np.asarray(delta_cmd7, dtype=np.float64)

    if curr.shape != (7,) or delta.shape != (7,):
        raise ValueError(f"Expected 7D inputs, got curr={curr.shape}, delta={delta.shape}")

    target = curr.copy()
    target[:3] += delta[:3]
    curr_rot = Rotation.from_euler("xyz", curr[3:6], degrees=False)
    target_rot = Rotation.from_rotvec(delta[3:6]) * curr_rot
    target[3:6] = target_rot.as_euler("xyz", degrees=False)
    target[6] = delta[6]
    return target.astype(np.float32)


def preprocess_image(bgr_frame: np.ndarray, w: int = IMG_W, h: int = IMG_H) -> np.ndarray:
    rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (w, h))
    return np.transpose(rgb, (2, 0, 1)).astype(np.float32) / 255.0


def get_next_video_path() -> Path:
    INFERENCE_RESULT_DIR.mkdir(parents=True, exist_ok=True)
    existing_indices = []
    for path in INFERENCE_RESULT_DIR.glob(f"{VIDEO_FILENAME_PREFIX}*{VIDEO_FILENAME_SUFFIX}"):
        stem = path.stem
        if not stem.startswith(VIDEO_FILENAME_PREFIX):
            continue
        suffix = stem[len(VIDEO_FILENAME_PREFIX) :]
        if suffix.isdigit():
            existing_indices.append(int(suffix))

    next_index = 1 if not existing_indices else max(existing_indices) + 1
    return INFERENCE_RESULT_DIR / f"{VIDEO_FILENAME_PREFIX}{next_index:04d}{VIDEO_FILENAME_SUFFIX}"


def create_video_writer(video_path: Path, fps: int) -> cv2.VideoWriter | None:
    frame_size = (CAMERA_PREVIEW_WIDTH * 2, CAMERA_PREVIEW_HEIGHT)
    for codec in VIDEO_CODECS:
        writer = cv2.VideoWriter(
            str(video_path),
            cv2.VideoWriter_fourcc(*codec),
            fps,
            frame_size,
        )
        if writer.isOpened():
            print(f"[Recorder] 使用编码器 {codec} 保存视频")
            return writer
        writer.release()
    return None


class RealSenseCamera:
    def __init__(self, serial_number: str, name: str, width: int, height: int, fps: int):
        self.name = name
        self.serial_number = serial_number
        self.width = width
        self.height = height
        self.fps = fps
        self.pipeline = self._start_pipeline()

        self._latest_frame: np.ndarray | None = None
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

        print(
            f"[Camera] {self.name} 已启动 "
            f"(serial={self.serial_number}, {width}x{height}@{self.fps})，等待首帧..."
        )
        while self.get_frame() is None:
            time.sleep(0.01)
        print(f"[Camera] {self.name} 首帧已就绪。")

    def _start_pipeline(self) -> rs.pipeline:
        last_error = None
        fps_candidates = []
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

    def _capture_loop(self):
        while not self._stop_event.is_set():
            try:
                frames = self.pipeline.wait_for_frames(timeout_ms=CAM_FRAME_TIMEOUT_MS)
                color_frame = frames.get_color_frame()
                if color_frame:
                    bgr = np.asanyarray(color_frame.get_data())
                    with self._lock:
                        self._latest_frame = bgr
            except Exception as exc:
                print(f"[Camera] {self.name} 采帧异常: {exc}")

    def get_frame(self) -> np.ndarray | None:
        with self._lock:
            return self._latest_frame.copy() if self._latest_frame is not None else None

    def stop(self):
        self._stop_event.set()
        try:
            if self._thread.is_alive():
                self._thread.join(timeout=1.0)
        except Exception as exc:
            print(f"[Camera] {self.name} 等待采集线程退出失败: {exc}")

        try:
            self.pipeline.stop()
        except Exception as exc:
            print(f"[Camera] {self.name} 停止失败: {exc}")
        else:
            print(f"[Camera] {self.name} 已停止。")


class EpisodeRecorder:
    def __init__(self, prompt: str, video_fps: float):
        self.prompt = prompt
        self.video_fps = float(video_fps)
        self.lock = threading.Lock()

        self.rollout_dir = ROLLOUT_DIR
        self.video_dir = ROLLOUT_VIDEO_DIR
        self.rollout_dir.mkdir(parents=True, exist_ok=True)
        self.video_dir.mkdir(parents=True, exist_ok=True)
        self.next_episode_idx = self._discover_next_episode_idx()

        self.states: list[np.ndarray] = []
        self.fixed_images: list[np.ndarray] = []
        self.wrist_images: list[np.ndarray] = []
        self.actions: list[np.ndarray] = []
        self.action_chunks: list[np.ndarray] = []
        self.commanded_delta_cmds: list[np.ndarray] = []
        self.timestamps: list[float] = []
        self.latencies_ms: list[float] = []

    def _discover_next_episode_idx(self) -> int:
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
        record_observation: dict[str, object],
        executed_action: np.ndarray,
        action_chunk: np.ndarray,
        commanded_delta_cmd: np.ndarray,
        latency_ms: float,
    ) -> None:
        state = np.asarray(record_observation["state"], dtype=np.float32).copy()
        images = record_observation["images"]
        assert isinstance(images, dict)
        fixed_image = np.asarray(images["fixed_camera"], dtype=np.uint8).copy()
        wrist_image = np.asarray(images["wrist_camera"], dtype=np.uint8).copy()

        with self.lock:
            self.states.append(state)
            self.fixed_images.append(fixed_image)
            self.wrist_images.append(wrist_image)
            self.actions.append(np.asarray(executed_action, dtype=np.float32).copy())
            self.action_chunks.append(np.asarray(action_chunk, dtype=np.float32).copy())
            self.commanded_delta_cmds.append(np.asarray(commanded_delta_cmd, dtype=np.float32).copy())
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
                "commanded_delta_cmds": np.stack(self.commanded_delta_cmds, axis=0),
                "timestamps": np.asarray(self.timestamps, dtype=np.float64),
                "latencies_ms": np.asarray(self.latencies_ms, dtype=np.float32),
            }

            self.states.clear()
            self.fixed_images.clear()
            self.wrist_images.clear()
            self.actions.clear()
            self.action_chunks.clear()
            self.commanded_delta_cmds.clear()
            self.timestamps.clear()
            self.latencies_ms.clear()
            return payload

    def _compose_video_frames(self, fixed_images: np.ndarray, wrist_images: np.ndarray) -> list[np.ndarray]:
        frames: list[np.ndarray] = []
        for fixed_rgb, wrist_rgb in zip(fixed_images, wrist_images, strict=True):
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
                self.video_fps,
                (width, height),
            )
            if candidate.isOpened():
                writer = candidate
                break
            candidate.release()
        if writer is None:
            print(f"[Recorder] episode 视频写入器创建失败: {out_path}")
            return

        for frame in frames:
            writer.write(frame)
        writer.release()
        print(f"[Recorder] 已保存 episode 视频: {out_path}")

    def _write_episode(self, episode: dict[str, object], success: bool, reason: str) -> None:
        episode_idx = int(episode["episode_idx"])
        num_steps = int(np.asarray(episode["actions"]).shape[0])
        reward = float(success)
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
                "source": "franka_openpi_client",
                "episode": episode_idx,
                "reason": reason,
                "prompt": self.prompt,
                "timestamps": np.asarray(episode["timestamps"], dtype=np.float64),
                "latencies_ms": np.asarray(episode["latencies_ms"], dtype=np.float32),
                "commanded_delta_cmds": np.asarray(episode["commanded_delta_cmds"], dtype=np.float32),
                "action_chunks": [np.asarray(chunk, dtype=np.float32) for chunk in episode["action_chunks"]],
            },
        }

        suffix = "success" if success else "failure"
        out_path = self.rollout_dir / f"episode_{episode_idx:06d}_{suffix}.pkl"
        with out_path.open("wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"[Recorder] 已保存 episode rollout: {out_path}")

        self._save_video(
            episode_idx=episode_idx,
            success=success,
            fixed_images=np.asarray(episode["fixed_images"], dtype=np.uint8),
            wrist_images=np.asarray(episode["wrist_images"], dtype=np.uint8),
        )

    def finalize_episode_interactive(self, reason: str = "manual") -> None:
        episode = self._pop_episode_data()
        if episode is None:
            print("[Recorder] 当前 episode 为空，跳过保存。")
            return

        episode_idx = int(episode["episode_idx"])
        while True:
            raw = input(
                f"[Recorder] episode {episode_idx:06d} 已结束 ({reason})，请输入结果 0=失败, 1=成功: "
            ).strip()
            if raw in ("0", "1"):
                self._write_episode(episode, success=(raw == "1"), reason=reason)
                return
            print("[Recorder] 输入非法，只能输入 0 或 1。")

    def close(self) -> None:
        if self.num_steps > 0:
            self.finalize_episode_interactive(reason="shutdown")


class FrankaInferenceRunner:
    def __init__(self, config: InferenceConfig, args: argparse.Namespace):
        self.config = config
        self.args = args
        self._tcp_transform = _build_flange_to_tcp()

        print(f"[Config] robot_ip={self.config.robot_ip}")
        print(f"[Config] policy={self.config.policy.host}:{self.config.policy.port}")
        print(f"[Config] prompt={self.config.prompt}")
        print(f"[Config] reset_joints_rad={self.config.reset.joint_angles}")
        print(f"[Config] reset_gripper_width={self.config.reset.gripper_width}")

        print(f"[Robot] 连接 Franka: {self.config.robot_ip} ...")
        self.robot = Robot(self.config.robot_ip)
        self.gripper = Gripper(self.config.robot_ip)
        self.robot.set_collision_behavior([20.0] * 7, [20.0] * 7, [10.0] * 6, [10.0] * 6)
        print("[Robot] 连接成功。")

        print(f"[Policy] 连接 pi 服务: {self.config.policy.host}:{self.config.policy.port} ...")
        self.client = websocket_client_policy.WebsocketClientPolicy(
            host=self.config.policy.host,
            port=self.config.policy.port,
        )
        print("[Policy] 连接成功。")

        self.front_camera = RealSenseCamera(
            serial_number=self.config.front_camera.serial,
            name="front_camera",
            width=self.config.front_camera.width,
            height=self.config.front_camera.height,
            fps=self.config.front_camera.fps,
        )
        self.wrist_camera = RealSenseCamera(
            serial_number=self.config.wrist_camera.serial,
            name="wrist_camera",
            width=self.config.wrist_camera.width,
            height=self.config.wrist_camera.height,
            fps=self.config.wrist_camera.fps,
        )
        self._gripper_thread: threading.Thread | None = None
        self._preview_enabled = True
        self._record_fps = min(self.front_camera.fps, self.wrist_camera.fps)
        self._video_path = get_next_video_path()
        self._video_writer = create_video_writer(self._video_path, self._record_fps)
        self._recorder_stop_event = threading.Event()
        self._recorder_thread: threading.Thread | None = None
        self.episode_recorder = EpisodeRecorder(
            prompt=self.config.prompt,
            video_fps=float(self._record_fps),
        )
        if self._video_writer is not None:
            print(f"[Recorder] 保存推理视频到: {self._video_path} (fps={self._record_fps})")
            self._recorder_thread = threading.Thread(target=self._record_loop, daemon=True)
            self._recorder_thread.start()
        else:
            print(f"[Recorder] 无法创建视频文件: {self._video_path}")
        print(f"[Recorder] episode pkl 保存目录: {self.episode_recorder.rollout_dir}")
        print(f"[Recorder] episode 视频保存目录: {self.episode_recorder.video_dir}")
        self._executed_action_count = 0
        self._episode_index = 0

    def _get_robot_state(self) -> tuple[list[float], list[float], float]:
        state: RobotState = self.robot.state
        q = list(state.q)
        tau_j = list(state.tau_J)
        gripper_width = self.gripper.state.width
        return q, tau_j, gripper_width

    def _get_tcp_pose(self) -> np.ndarray:
        base_to_flange = _affine_to_np(self.robot.current_cartesian_state.pose.end_effector_pose)
        return base_to_flange @ self._tcp_transform

    def _build_observation(self) -> tuple[dict[str, object], dict[str, object]] | None:
        front_bgr_frame = self.front_camera.get_frame()
        wrist_bgr_frame = self.wrist_camera.get_frame()
        if front_bgr_frame is None or wrist_bgr_frame is None:
            print("[Obs] 未获取到 front/wrist 相机图像，跳过本次推理。")
            return None

        self._show_camera_preview(front_bgr_frame, wrist_bgr_frame)
        q, tau_j, gripper_width = self._get_robot_state()
        state_vec = np.array([*map(float, q), float(gripper_width)], dtype=np.float32)
        fixed_image = preprocess_image(front_bgr_frame)
        wrist_image = preprocess_image(wrist_bgr_frame)
        record_observation = {
            "state": state_vec.copy(),
            "images": {
                "fixed_camera": cv2.cvtColor(front_bgr_frame, cv2.COLOR_BGR2RGB),
                "wrist_camera": cv2.cvtColor(wrist_bgr_frame, cv2.COLOR_BGR2RGB),
            },
        }
        print(
            "[Obs] 发送推理输入: "
            "images.fixed_camera, "
            "images.wrist_camera, "
            "state, prompt"
        )
        policy_input = {
            "images": {
                "fixed_camera": fixed_image,
                "wrist_camera": wrist_image,
            },
            "state": state_vec,
            "prompt": self.config.prompt,
        }
        return policy_input, record_observation

    def _compose_preview_frame(self, front_bgr_frame: np.ndarray, wrist_bgr_frame: np.ndarray) -> np.ndarray:
        front_preview = cv2.resize(front_bgr_frame, (CAMERA_PREVIEW_WIDTH, CAMERA_PREVIEW_HEIGHT))
        wrist_preview = cv2.resize(wrist_bgr_frame, (CAMERA_PREVIEW_WIDTH, CAMERA_PREVIEW_HEIGHT))
        preview = np.hstack((front_preview, wrist_preview))
        cv2.putText(preview, "front_camera", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
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

    def _record_loop(self):
        frame_interval_s = 1.0 / self._record_fps
        while not self._recorder_stop_event.is_set():
            front_bgr_frame = self.front_camera.get_frame()
            wrist_bgr_frame = self.wrist_camera.get_frame()
            if front_bgr_frame is not None and wrist_bgr_frame is not None and self._video_writer is not None:
                preview = self._compose_preview_frame(front_bgr_frame, wrist_bgr_frame)
                self._video_writer.write(preview)
            if self._recorder_stop_event.wait(frame_interval_s):
                break

    def _show_camera_preview(self, front_bgr_frame: np.ndarray, wrist_bgr_frame: np.ndarray):
        preview = self._compose_preview_frame(front_bgr_frame, wrist_bgr_frame)
        if not self._preview_enabled:
            return
        try:
            cv2.imshow(CAMERA_PREVIEW_WINDOW, preview)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                raise KeyboardInterrupt
        except cv2.error as exc:
            self._preview_enabled = False
            print(f"[Camera] 预览窗口不可用，已关闭图像展示: {exc}")

    def _close_camera_preview(self):
        if not self._preview_enabled:
            return
        try:
            cv2.destroyWindow(CAMERA_PREVIEW_WINDOW)
            cv2.waitKey(1)
        except cv2.error:
            pass

    def _close_video_writer(self):
        if self._recorder_thread is not None:
            self._recorder_stop_event.set()
            try:
                if self._recorder_thread.is_alive():
                    self._recorder_thread.join(timeout=2.0)
            except Exception as exc:
                print(f"[Recorder] 等待录制线程退出失败: {exc}")
            finally:
                self._recorder_thread = None

        if self._video_writer is None:
            return
        try:
            self._video_writer.release()
            print(f"[Recorder] 视频已保存: {self._video_path}")
        except Exception as exc:
            print(f"[Recorder] 视频保存失败: {exc}")
        finally:
            self._video_writer = None

    def _cleanup(self):
        old_sigint_handler = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        try:
            self.episode_recorder.close()
            self._close_video_writer()
            self._close_camera_preview()
            self.front_camera.stop()
            self.wrist_camera.stop()
        finally:
            signal.signal(signal.SIGINT, old_sigint_handler)

    def _recover_if_needed(self):
        if not self.robot.has_errors:
            return
        print("[Robot] 检测到错误状态，尝试恢复...")
        try:
            self.robot.recover_from_errors()
            print("[Robot] 恢复完成。")
        except Exception as exc:
            print(f"[Robot] 恢复失败: {exc}")

    def _get_chunk_exec_steps(self) -> int:
        if self._executed_action_count >= self.config.execution.chunk_exec_switch_total_actions:
            return self.config.execution.chunk_exec_steps_after_switch
        return self.config.execution.chunk_exec_steps_before_switch

    def _reset_episode_counters(self):
        self._executed_action_count = 0

    def _wait_for_episode_command(self) -> str:
        print(
            "\n[Episode] 等待指令: "
            "`n` 开始下一轮(先 reset), `r` 只 reset 不开跑, `q` 退出。"
        )
        while True:
            try:
                command = input("[Episode] 请输入指令 [n/r/q]: ").strip().lower()
            except EOFError:
                return "q"
            if command in {"n", "r", "q"}:
                print(f"[Episode] 收到指令: {command}")
                return command
            print("[Episode] 无效指令，请输入 n / r / q。")

    def _run_single_episode(self):
        self._episode_index += 1
        self._reset_episode_counters()
        print(
            f"\n[Episode] 第 {self._episode_index} 轮开始，"
            f"最多执行 {self.config.episode.max_actions_per_episode} 个动作。"
        )
        print("[Episode] 状态: RUNNING")

        while self._executed_action_count < self.config.episode.max_actions_per_episode:
            observation_pair = self._build_observation()
            if observation_pair is None:
                print("[Episode] 等待有效观测...")
                continue
            policy_observation, record_observation = observation_pair
            try:
                t0 = time.time()
                result = self.client.infer(policy_observation)
                action_chunk = result["actions"]
                elapsed_ms = (time.time() - t0) * 1000
                print(f"[Inference] 推理完成，耗时 {elapsed_ms:.1f}ms，chunk shape: {action_chunk.shape}")
                if action_chunk.ndim == 2 and action_chunk.shape[1] >= 8:
                    gripper_chunk = np.asarray(action_chunk[:, 7], dtype=np.float32)
                    print(
                        "[Inference] gripper_chunk first10=%s min=%.6f max=%.6f mean=%.6f"
                        % (
                            np.round(gripper_chunk[:10], 6).tolist(),
                            float(gripper_chunk.min()),
                            float(gripper_chunk.max()),
                            float(gripper_chunk.mean()),
                        )
                    )
            except Exception as exc:
                print(f"[Inference] 推理失败: {exc}")
                continue

            remaining_steps = self.config.episode.max_actions_per_episode - self._executed_action_count
            if FIXED_CHUNK_EXEC_STEPS > 0:
                exec_steps = min(int(action_chunk.shape[0]), int(remaining_steps), FIXED_CHUNK_EXEC_STEPS)
                exec_plan = f"固定上限 {FIXED_CHUNK_EXEC_STEPS} 步"
            else:
                exec_steps = min(int(action_chunk.shape[0]), int(remaining_steps))
                exec_plan = "完整 action chunk"
            print(
                f"[Inference] 当前 episode 已执行 {self._executed_action_count} 个动作，"
                f"服务实际提供 {action_chunk.shape[0]} 步，本次将顺序执行 {exec_steps} 步"
                f"({exec_plan})。"
            )

            executed_steps = 0
            for action_idx in range(exec_steps):
                if action_idx == 0:
                    step_policy_observation = policy_observation
                    step_record_observation = record_observation
                else:
                    step_observation_pair = self._build_observation()
                    if step_observation_pair is None:
                        print(
                            f"[Inference] chunk 第 {action_idx + 1} 步缺少有效观测，"
                            "提前结束本次 chunk 执行。"
                        )
                        break
                    step_policy_observation, step_record_observation = step_observation_pair

                ok, executed_action, commanded_delta_cmd = self._execute_policy_step(
                    step_policy_observation,
                    action_chunk,
                    action_idx=action_idx,
                )
                if not ok:
                    self._recover_if_needed()
                    break

                executed_steps += 1
                self._executed_action_count += 1
                assert executed_action is not None
                assert commanded_delta_cmd is not None
                self.episode_recorder.record_step(
                    record_observation=step_record_observation,
                    executed_action=executed_action,
                    action_chunk=action_chunk,
                    commanded_delta_cmd=commanded_delta_cmd,
                    latency_ms=elapsed_ms,
                )

            print(
                f"[Inference] 本次执行 {executed_steps} 个动作，"
                f"当前 episode 累计 {self._executed_action_count} 个动作。"
            )

        print(f"[Episode] 第 {self._episode_index} 轮执行结束。")
        if self.episode_recorder.num_steps > 0:
            self.episode_recorder.finalize_episode_interactive(reason="episode_end")
        print("[Episode] 状态: IDLE")

    def _execute_policy_step(
        self,
        observation: dict[str, object],
        action_chunk: np.ndarray,
        *,
        action_idx: int = 0,
    ) -> tuple[bool, np.ndarray | None, np.ndarray | None]:
        if action_chunk.ndim != 2:
            print(f"  [错误] 动作 chunk 维度异常: {action_chunk.shape}")
            return False, None, None
        if action_chunk.shape[0] <= 0:
            print("  [错误] 动作 chunk 为空。")
            return False, None, None
        if action_idx < 0 or action_idx >= int(action_chunk.shape[0]):
            print(f"  [错误] 动作索引越界: action_idx={action_idx}, chunk_shape={action_chunk.shape}")
            return False, None, None
        observation_state = observation["state"]
        state_vec = np.asarray(observation_state, dtype=np.float64)
        print(f"[Exec] 执行 chunk step {action_idx + 1}/{action_chunk.shape[0]}")

        if int(action_chunk.shape[1]) == 8:
            if state_vec.shape != (8,):
                print(f"  [错误] joint 观测 state 维度异常: {state_vec.shape}")
                return False, None, None
            curr_joint8 = state_vec
            target_joint8 = np.asarray(action_chunk[action_idx], dtype=np.float64)
            print(f"[Exec] curr_joint8   : {np.array2string(curr_joint8, precision=4)}")
            print(f"[Exec] target_joint8 : {np.array2string(target_joint8, precision=4)}")
            executed_joint8 = self._send_absolute_joint_action(curr_joint8, target_joint8)
            if executed_joint8 is None:
                return False, None, None
            return True, executed_joint8, target_joint8.astype(np.float32)

        if int(action_chunk.shape[1]) == 7:
            if state_vec.shape != (7,):
                print(f"  [错误] EE 观测 state 维度异常: {state_vec.shape}")
                return False, None, None
            curr_pose7 = state_vec
            target_pose7 = np.asarray(action_chunk[action_idx], dtype=np.float64)
            delta_cmd7, delta_action = abs_ee_to_delta_cmd(
                curr_pose7,
                target_pose7,
                max_gripper_width=self.config.execution.gripper_max_width,
            )
            commanded_pose7 = delta_cmd_to_abs_pose(curr_pose7, delta_cmd7)
            print(f"[Exec] curr_pose7   : {np.array2string(curr_pose7, precision=4)}")
            print(f"[Exec] target_pose7 : {np.array2string(target_pose7, precision=4)}")
            print(f"[Exec] command_pose7: {np.array2string(commanded_pose7, precision=4)}")
            print(f"[Exec] delta_cmd7   : {np.array2string(delta_cmd7, precision=4)}")
            ok = self._send_delta_action(delta_action)
            if not ok:
                return False, None, None
            return True, commanded_pose7, delta_cmd7.astype(np.float32)

        print(f"  [错误] 只支持 8 维 joint 或 7 维 EE action，实际收到: {action_chunk.shape}")
        return False, None, None

    def _clip_joint_target(self, current_q: np.ndarray, target_q: np.ndarray) -> np.ndarray:
        max_step = float(MAX_JOINT_STEP_RAD)
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

    def _clip_gripper_width(self, target_gripper: float) -> float:
        max_width = float(self.config.execution.gripper_max_width)
        return float(np.clip(target_gripper, 0.0, max_width))

    def _send_absolute_joint_action(
        self,
        current_joint8: np.ndarray | list[float],
        target_joint8: np.ndarray | list[float],
    ) -> np.ndarray | None:
        current = np.asarray(current_joint8, dtype=np.float64)
        target = np.asarray(target_joint8, dtype=np.float64)
        if current.shape != (8,):
            print(f"  [错误] current joint state 维度错误: {current.shape}")
            return None
        if target.shape != (8,):
            print(f"  [错误] absolute joint action 维度错误: {target.shape}")
            return None
        if np.any(~np.isfinite(current)) or np.any(~np.isfinite(target)):
            print("  [错误] absolute joint action 包含非法数值。")
            return None

        commanded_joint8 = current.copy()
        raw_target_q = target[:7]
        commanded_joint8[:7] = self._clip_joint_target(current[:7], raw_target_q)
        commanded_joint8[7] = self._clip_gripper_width(float(target[7]))

        print(f"[Exec] raw_target_q  : {np.array2string(raw_target_q, precision=4)}")
        print(f"[Exec] command_joint8: {np.array2string(commanded_joint8, precision=4)}")
        self._command_gripper(float(commanded_joint8[7]))
        self.robot.relative_dynamics_factor = min(
            self.config.execution.velocity_rel,
            self.config.execution.acceleration_rel,
        )

        try:
            self.robot.move(self._build_joint_motion(commanded_joint8[:7]))
            return commanded_joint8.astype(np.float32)
        except Exception as exc:
            print(f"  [错误] absolute joint 执行失败: {exc}")
            return None

    def _command_gripper(self, target_gripper: float) -> None:
        # 训练数据中的 gripper 维使用连续开口宽度（米），部署侧保持同样语义，
        # 避免再额外做 open/close/hold 三态离散化导致夹爪行为偏离训练分布。
        raw_target_gripper = float(target_gripper)
        current_gripper = float(self.gripper.state.width)
        target_gripper = self._clip_gripper_width(raw_target_gripper)
        gripper_delta = abs(target_gripper - current_gripper)

        if abs(target_gripper - raw_target_gripper) > 1e-9:
            print(
                "[Gripper] 目标宽度越界，已裁剪: raw=%.4f clipped=%.4f"
                % (raw_target_gripper, target_gripper)
            )

        print(
            "[Gripper] current=%.4f target=%.4f delta=%.4f"
            % (
                current_gripper,
                target_gripper,
                gripper_delta,
            )
        )

        is_closing = target_gripper < current_gripper - GRIPPER_CLOSE_EPSILON_M
        try:
            if is_closing and hasattr(self.gripper, "grasp"):
                grasp_attempts = (
                    {
                        "args": (
                            target_gripper,
                            self.config.execution.gripper_speed,
                            GRIPPER_GRASP_FORCE_N,
                        ),
                        "kwargs": {"epsilon_outer": GRIPPER_GRASP_EPSILON_OUTER},
                    },
                    {
                        "args": (
                            target_gripper,
                            self.config.execution.gripper_speed,
                            GRIPPER_GRASP_FORCE_N,
                        ),
                        "kwargs": {},
                    },
                )
                last_error: Exception | None = None
                grasp_success = False
                for attempt in grasp_attempts:
                    try:
                        grasp_result = self.gripper.grasp(*attempt["args"], **attempt["kwargs"])
                        grasp_success = bool(grasp_result) if grasp_result is not None else True
                        print(
                            "[Gripper] 使用 grasp() 闭合: success=%s force=%.1f"
                            % (grasp_success, GRIPPER_GRASP_FORCE_N)
                        )
                        break
                    except TypeError as exc:
                        last_error = exc
                else:
                    raise RuntimeError(f"grasp() 参数不兼容: {last_error}")

                if not grasp_success:
                    print("[Gripper] grasp() 未确认成功，回退到 move()。")
                    self.gripper.move(target_gripper, self.config.execution.gripper_speed)
            else:
                self.gripper.move(target_gripper, self.config.execution.gripper_speed)
        except Exception as exc:
            print(f"  [错误] 夹爪运动失败: {exc}")
        time.sleep(self.config.execution.gripper_hold_time_s)

    def _send_delta_action(self, delta_action: dict[str, float]) -> bool:
        target_gripper = float(delta_action["gripper"])
        self._command_gripper(target_gripper)
        self.robot.relative_dynamics_factor = min(
            self.config.execution.velocity_rel,
            self.config.execution.acceleration_rel,
        )
        duration_ms = CONTROL_CYCLE_MS
        dt = duration_ms / 1000.0
        lin_vel = np.array(
            [
                float(delta_action["delta_x"]),
                float(delta_action["delta_y"]),
                float(delta_action["delta_z"]),
            ],
            dtype=np.float64,
        ) / dt
        ang_vel = np.array(
            [
                float(delta_action["delta_roll"]),
                float(delta_action["delta_pitch"]),
                float(delta_action["delta_yaw"]),
            ],
            dtype=np.float64,
        ) / dt
        if np.any(~np.isfinite(lin_vel)) or np.any(~np.isfinite(ang_vel)):
            print("  [错误] delta action 包含非法速度。")
            return False

        try:
            self.robot.move(
                CartesianVelocityMotion(
                    Twist(linear_velocity=lin_vel, angular_velocity=ang_vel),
                    Duration(duration_ms),
                ),
                asynchronous=True,
            )
            return True
        except Exception as exc:
            print(f"  [错误] delta EE 执行失败: {exc}")
            return False

    def perform_reset(self) -> bool:
        print("[Reset] 状态: START")
        self._recover_if_needed()
        ok = reset_robot(self.robot, self.gripper, self.config)
        self._recover_if_needed()
        print("[Reset] 状态: OK" if ok else "[Reset] 状态: FAILED")
        return ok

    def run(self):
        print("\n[Inference] Episode 模式已启动。")
        print("[Inference] 当前为待命状态，输入 `n` 后才会开始首轮执行。")
        print("[Inference] 按 Ctrl+C 可随时停止。")
        try:
            while True:
                command = self._wait_for_episode_command()
                if command == "q":
                    print("[Inference] 收到退出指令。")
                    return
                reset_ok = self.perform_reset()
                if command == "r":
                    print("[Episode] 已 reset，当前仍为待命状态。")
                    if not reset_ok:
                        print("[Episode] reset 失败，请检查机械臂状态后再继续。")
                    continue
                if not reset_ok:
                    print("[Episode] reset 失败，本轮未启动，请重新指令。")
                    continue
                print("[Episode] reset 成功，开始执行。")
                self._run_single_episode()
        except KeyboardInterrupt:
            print("\n[Inference] 收到 Ctrl+C，停止。")
        finally:
            self._cleanup()
            print("[Runner] 程序结束。")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Franka OpenPI inference runner")
    parser.add_argument(
        "--config",
        default="/home/embint/ls/cupid/lerobot/franka_pi0_inference_config.json",
        help="Path to JSON config file.",
    )
    parser.add_argument(
        "--reset-only",
        action="store_true",
        help="Reset robot and gripper to configured start pose, then exit.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_inference_config(args.config)
    print(f"[Config] loaded from: {Path(args.config).expanduser().resolve()}")
    if args.reset_only:
        print(f"[Robot] 连接 Franka: {config.robot_ip} ...")
        robot = Robot(config.robot_ip)
        gripper = Gripper(config.robot_ip)
        robot.set_collision_behavior([20.0] * 7, [20.0] * 7, [10.0] * 6, [10.0] * 6)
        if getattr(robot, "has_errors", False):
            print("[Robot] 检测到错误状态，尝试恢复...")
            robot.recover_from_errors()
        ok = reset_robot(robot, gripper, config)
        print("[Reset] reset-only 完成。" if ok else "[Reset] reset-only 有失败。")
        return
    runner = FrankaInferenceRunner(config, args)
    runner.run()


if __name__ == "__main__":
    main()
