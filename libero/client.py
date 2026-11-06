#!/usr/bin/env python3
"""在 mimiclabs 环境中运行 LIBERO 仿真，并通过 HTTP 请求远端 ACT server。

典型用法：
    conda activate mimiclabs
    python /workspace/liushuo/cupid/libero/client.py \
        --server-url http://127.0.0.1:8000 \
        --suite libero_object \
        --task-id 3 \
        --episodes 10 \
        --controller JOINT_POSITION \
        --task-map-json /workspace/liushuo/cupid/output/act/libero_object_multitask/task_index_map.json \
        --out-dir /workspace/liushuo/cupid/output/eval/libero_http/bbq
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib import error, request

import cv2
import imageio.v2 as imageio
import numpy as np
import torch


DEFAULT_LIBERO_ROOT = Path("/workspace/liushuo/mimiclabs/LIBERO")
DEFAULT_TASK_MAP = Path("/workspace/liushuo/cupid/output/act/libero_object_multitask/task_index_map.json")


def ensure_runtime_paths(libero_root: Path) -> None:
    """确保 LIBERO 仓库可导入，并给 numba 提供可写缓存目录。"""
    os.environ.setdefault("NUMBA_CACHE_DIR", str(Path("/tmp/numba-cache")))
    try:
        Path(os.environ["NUMBA_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)
    except OSError:
        pass

    if str(libero_root) not in sys.path:
        sys.path.insert(0, str(libero_root))


def http_get(url: str, timeout: float) -> Dict[str, object]:
    with request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def http_post(url: str, payload: Dict[str, object], timeout: float) -> Dict[str, object]:
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


class ACTHTTPClient:
    def __init__(self, server_url: str, timeout: float) -> None:
        self.server_url = server_url.rstrip("/")
        self.timeout = timeout

    def health(self) -> Dict[str, object]:
        return http_get(f"{self.server_url}/health", self.timeout)

    def reset(self) -> Dict[str, object]:
        return http_post(f"{self.server_url}/reset", {}, self.timeout)

    def infer(self, payload: Dict[str, object]) -> Dict[str, object]:
        return http_post(f"{self.server_url}/infer", payload, self.timeout)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LIBERO HTTP client evaluator")
    parser.add_argument("--server-url", type=str, required=True, help="ACT server 地址，例如 http://127.0.0.1:8000")
    parser.add_argument(
        "--libero-root",
        type=Path,
        default=DEFAULT_LIBERO_ROOT,
        help="LIBERO 仓库根目录，内部应包含 libero/libero",
    )
    parser.add_argument(
        "--suite",
        type=str,
        default="libero_object",
        choices=["libero_object", "libero_goal", "libero_spatial", "libero_10"],
        help="LIBERO benchmark suite",
    )
    parser.add_argument("--task-id", type=int, required=True, help="benchmark 内的 task_id")
    parser.add_argument("--episodes", type=int, default=10, help="评估 episode 数")
    parser.add_argument("--max-steps", type=int, default=400, help="每条 episode 的最大步数")
    parser.add_argument("--seed", type=int, default=0, help="起始随机种子")
    parser.add_argument("--image-size", type=int, default=128, help="评估图像分辨率")
    parser.add_argument("--control-freq", type=int, default=20, help="环境控制频率")
    parser.add_argument("--num-steps-wait", type=int, default=10, help="reset 后等待物体稳定的步数")
    parser.add_argument("--render", type=int, default=2, help="最多保存多少条视频")
    parser.add_argument(
        "--controller",
        type=str,
        default="JOINT_POSITION",
        help="LIBERO 控制器，8 维 joint policy 推荐 JOINT_POSITION",
    )
    parser.add_argument(
        "--joint-input-type",
        type=str,
        default="absolute",
        choices=["absolute", "delta"],
        help="policy 前 7 维 joint 动作的语义；当前 ACT 数据默认是 absolute",
    )
    parser.add_argument(
        "--joint-delta-scale",
        type=float,
        default=0.05,
        help="JOINT_POSITION 控制器的单步 joint delta 缩放上限，应与 robosuite controller config 对齐",
    )
    parser.add_argument(
        "--gripper-force-open-steps",
        type=int,
        default=0,
        help="实验开关：前 N 个执行步强制保持夹爪张开，用于验证是否闭合过早",
    )
    parser.add_argument(
        "--task-conditioning",
        type=str,
        default="auto",
        choices=["auto", "on", "off"],
        help="是否启用 task-conditioned 输入；auto 会根据 server /health 自动判断",
    )
    parser.add_argument(
        "--task-map-json",
        type=Path,
        default=None,
        help="task_index_map.json，用于把当前任务映射到训练时的 one-hot 索引",
    )
    parser.add_argument("--timeout", type=float, default=30.0, help="HTTP 请求超时")
    parser.add_argument("--chunk-size", type=int, default=None, help="覆盖 server 默认返回 chunk 长度")
    parser.add_argument("--debug-log-actions", action="store_true", help="是否记录每条 episode 前若干步的动作调试信息")
    parser.add_argument("--debug-log-steps", type=int, default=20, help="每条 episode 最多记录多少个执行步")
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="输出目录，将写入 eval_info.json 和可选视频",
    )
    return parser.parse_args()


def normalize_task_name(task: str) -> str:
    return " ".join(task.strip().lower().replace("_", " ").split())


def load_task_map(task_map_path: Path) -> tuple[dict[str, int], int]:
    payload = json.loads(task_map_path.read_text(encoding="utf-8"))
    mapping = payload.get("task_to_task_index")
    total_tasks = int(payload.get("total_tasks", 0))
    if not isinstance(mapping, dict) or total_tasks <= 0:
        raise ValueError(f"非法 task map 文件: {task_map_path}")

    normalized = {normalize_task_name(str(k)): int(v) for k, v in mapping.items()}
    return normalized, total_tasks


def resolve_task_index(task_name: str, task_language: str, mapping: dict[str, int]) -> int:
    candidates = [normalize_task_name(task_name), normalize_task_name(task_language)]
    for cand in candidates:
        if cand in mapping:
            return mapping[cand]
    raise KeyError(
        f"任务 '{task_name}' / '{task_language}' 未出现在 task_index_map.json 中。"
    )


def encode_png_base64(image_rgb: np.ndarray) -> str:
    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".png", bgr)
    if not ok:
        raise ValueError("PNG 编码失败。")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def build_payload(
    obs: dict[str, Any],
    health: dict[str, object],
    task_index: Optional[int],
    num_tasks: Optional[int],
    chunk_size: Optional[int],
) -> Dict[str, object]:
    image_keys = list(health.get("image_keys", []))
    expect_state = bool(health.get("expect_state", False))
    expect_env_state = bool(health.get("expect_env_state", False))

    front = np.flipud(np.asarray(obs["agentview_image"]))
    wrist = np.flipud(np.asarray(obs["robot0_eye_in_hand_image"]))
    images: Dict[str, str] = {}
    for key in image_keys:
        suffix = key.split(".")[-1]
        if suffix in {"front", "agentview"}:
            images[key] = encode_png_base64(front)
        elif suffix in {"wrist", "robot0_eye_in_hand"}:
            images[key] = encode_png_base64(wrist)
        else:
            raise KeyError(f"未知图像键: {key}")

    payload: Dict[str, object] = {"images": images}

    if expect_state:
        joint = np.asarray(obs["robot0_joint_pos"], dtype=np.float32).reshape(-1)
        gripper = np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32).reshape(-1)
        state = np.concatenate([joint, gripper], axis=0)
        payload["state"] = state.astype(float).tolist()

    if expect_env_state:
        if task_index is None or num_tasks is None:
            raise ValueError("server 需要 env_state，但当前未提供 task map 解析结果。")
        env_state = np.zeros((num_tasks,), dtype=np.float32)
        env_state[task_index] = 1.0
        payload["env_state"] = env_state.astype(float).tolist()

    if chunk_size is not None:
        payload["chunk_size"] = int(chunk_size)

    return payload


def current_joint_gripper_action(obs: dict[str, Any]) -> np.ndarray:
    joint = np.asarray(obs["robot0_joint_pos"], dtype=np.float32).reshape(-1)
    gripper_qpos = np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32).reshape(-1)
    if joint.shape[0] != 7:
        raise ValueError(f"robot0_joint_pos 维度异常: {joint.shape}")
    grip_scalar = float(np.mean(gripper_qpos)) if gripper_qpos.size > 0 else 0.0
    return np.concatenate([joint, np.array([grip_scalar], dtype=np.float32)], axis=0)


def extract_joint_gripper_state(obs: dict[str, Any]) -> dict[str, Any]:
    joint = np.asarray(obs["robot0_joint_pos"], dtype=np.float32).reshape(-1)
    gripper_qpos = np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32).reshape(-1)
    return {
        "joint": joint.astype(float).tolist(),
        "gripper_qpos": gripper_qpos.astype(float).tolist(),
        "gripper_mean": float(np.mean(gripper_qpos)) if gripper_qpos.size > 0 else 0.0,
    }


def choose_primary_object_name(object_names: List[str]) -> Optional[str]:
    if not object_names:
        return None
    low_priority = ("basket", "tray", "plate", "drawer", "cabinet", "microwave", "stove")
    for name in object_names:
        if not any(token in name.lower() for token in low_priority):
            return name
    return object_names[0]


def extract_task_geometry(env, obs: dict[str, Any]) -> dict[str, Any]:
    """提取抓取相关几何量，便于判断初始化和抓取窗口是否异常。"""
    geometry: Dict[str, Any] = {}

    eef_pos = None
    if "robot0_eef_pos" in obs:
        eef_pos = np.asarray(obs["robot0_eef_pos"], dtype=np.float32).reshape(-1)
        geometry["eef_pos"] = eef_pos.astype(float).tolist()
    if "robot0_eef_quat" in obs:
        eef_quat = np.asarray(obs["robot0_eef_quat"], dtype=np.float32).reshape(-1)
        geometry["eef_quat"] = eef_quat.astype(float).tolist()

    object_names = list(getattr(env, "obj_of_interest", []) or [])
    geometry["obj_of_interest"] = object_names

    object_positions: Dict[str, List[float]] = {}
    rel_positions: Dict[str, List[float]] = {}
    raw_env = getattr(env, "env", None)
    obj_body_id = getattr(raw_env, "obj_body_id", None)
    sim = getattr(env, "sim", None)

    if sim is not None and isinstance(obj_body_id, dict):
        for name in object_names:
            if name not in obj_body_id:
                continue
            pos = np.asarray(sim.data.body_xpos[obj_body_id[name]], dtype=np.float32).reshape(-1)
            object_positions[name] = pos.astype(float).tolist()
            if eef_pos is not None and pos.shape == eef_pos.shape:
                rel_positions[name] = (pos - eef_pos).astype(float).tolist()

    geometry["object_positions"] = object_positions
    geometry["eef_to_object"] = rel_positions

    primary_name = choose_primary_object_name(list(object_positions.keys()))
    geometry["primary_object_name"] = primary_name
    if primary_name is not None:
        geometry["primary_object_pos"] = object_positions.get(primary_name)
        geometry["eef_to_primary_object"] = rel_positions.get(primary_name)

    return geometry


def convert_policy_action_to_env_action(
    policy_action: np.ndarray,
    obs: dict[str, Any],
    controller: str,
    joint_input_type: str,
    joint_delta_scale: float,
) -> np.ndarray:
    """将 policy 输出动作转换成环境实际需要的控制指令。"""
    action = np.asarray(policy_action, dtype=np.float32).reshape(-1).copy()
    if controller.upper() != "JOINT_POSITION":
        return action
    if action.shape[0] != 8:
        raise ValueError(f"JOINT_POSITION 期望 8 维动作，实际得到: {action.shape}")

    if joint_input_type == "absolute":
        current_joint = np.asarray(obs["robot0_joint_pos"], dtype=np.float32).reshape(-1)
        joint_target = action[:7]
        joint_delta = joint_target - current_joint
        action[:7] = np.clip(joint_delta / float(joint_delta_scale), -1.0, 1.0)
    elif joint_input_type == "delta":
        action[:7] = np.clip(action[:7], -1.0, 1.0)
    else:
        raise ValueError(f"未知 joint_input_type: {joint_input_type}")

    action[7] = np.clip(action[7], -1.0, 1.0)
    return action


def apply_gripper_schedule(env_action: np.ndarray, step_index: int, force_open_steps: int) -> tuple[np.ndarray, bool]:
    """按实验设定覆盖夹爪命令。当前数据中 -1 更像张开。"""
    action = np.asarray(env_action, dtype=np.float32).reshape(-1).copy()
    applied = False
    if force_open_steps > 0 and step_index < force_open_steps and action.shape[0] >= 1:
        action[-1] = -1.0
        applied = True
    return action, applied


def render_frame(obs: dict[str, Any]) -> np.ndarray:
    front = np.flipud(np.asarray(obs["agentview_image"]))
    wrist = np.flipud(np.asarray(obs["robot0_eye_in_hand_image"]))
    if front.shape[0] != wrist.shape[0]:
        wrist = cv2.resize(wrist, (front.shape[1], front.shape[0]))
    return np.concatenate([front, wrist], axis=1)


def save_video(video_path: Path, frames: List[np.ndarray], fps: int) -> None:
    if not frames:
        return
    video_path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    normalized_frames: List[np.ndarray] = []
    for frame in frames:
        if frame.shape[:2] != (h, w):
            frame = cv2.resize(frame, (w, h))
        normalized_frames.append(np.asarray(frame, dtype=np.uint8))

    try:
        # 优先写成 H.264，浏览器和常见播放器兼容性更好。
        imageio.mimwrite(
            str(video_path),
            normalized_frames,
            fps=float(fps),
            codec="libx264",
            format="FFMPEG",
            pixelformat="yuv420p",
            macro_block_size=1,
        )
        return
    except Exception:
        pass

    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (w, h))
    try:
        for frame in normalized_frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()


def step_env_safely(env, action: np.ndarray) -> tuple[dict[str, Any], bool, dict[str, Any]]:
    """执行一步环境交互，并把 terminated-episode 异常转成正常结束信号。"""
    try:
        obs, _, done, info = env.step(action)
        info = info or {}
        return obs, bool(done), info
    except ValueError as exc:
        if "terminated episode" not in str(exc):
            raise
        info: dict[str, Any] = {"terminated_episode_exception": True}
        success = False
        if hasattr(env, "check_success"):
            try:
                success = bool(env.check_success())
            except Exception:
                success = False
        info["success"] = success
        return {}, True, info


def main() -> None:
    args = parse_args()
    ensure_runtime_paths(args.libero_root)

    from libero.libero.benchmark import get_benchmark  # type: ignore
    from libero.libero.envs.env_wrapper import OffScreenRenderEnv  # type: ignore

    if not args.libero_root.exists():
        raise FileNotFoundError(f"libero_root 不存在: {args.libero_root}")

    client = ACTHTTPClient(args.server_url, args.timeout)
    health = client.health()

    server_action_dim = int(health.get("action_dim", -1))
    if server_action_dim <= 0:
        raise ValueError(f"server /health 返回的 action_dim 非法: {health}")
    if server_action_dim == 8 and args.controller.upper() != "JOINT_POSITION":
        raise ValueError("当前模型输出 8 维 joint action，必须使用 --controller JOINT_POSITION。")

    if args.task_conditioning == "auto":
        use_task_conditioning = bool(health.get("expect_env_state", False))
    else:
        use_task_conditioning = args.task_conditioning == "on"

    task_map_path = args.task_map_json
    if task_map_path is None and DEFAULT_TASK_MAP.exists():
        task_map_path = DEFAULT_TASK_MAP

    benchmark = get_benchmark(args.suite.upper())(0)
    if args.task_id < 0 or args.task_id >= benchmark.n_tasks:
        raise ValueError(f"task_id 超出范围: {args.task_id}, suite={args.suite}, n_tasks={benchmark.n_tasks}")

    task = benchmark.get_task(args.task_id)
    bddl_file = args.libero_root / "libero" / "libero" / "bddl_files" / task.problem_folder / task.bddl_file
    init_states_file = args.libero_root / "libero" / "libero" / "init_files" / task.problem_folder / task.init_states_file
    if not bddl_file.exists():
        raise FileNotFoundError(f"bddl 文件不存在: {bddl_file}")
    if not init_states_file.exists():
        raise FileNotFoundError(f"init state 文件不存在: {init_states_file}")

    task_index = None
    num_tasks = None
    if use_task_conditioning:
        if task_map_path is None or not task_map_path.exists():
            raise FileNotFoundError("server 需要 env_state，但未找到 task_index_map.json。")
        task_mapping, num_tasks = load_task_map(task_map_path)
        task_index = resolve_task_index(task.name, task.language, task_mapping)

    # LIBERO 的 init_states 是普通 pickle 数据，不是纯 tensor state_dict。
    init_states = torch.load(init_states_file, map_location="cpu", weights_only=False)

    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl_file),
        camera_heights=args.image_size,
        camera_widths=args.image_size,
        controller=args.controller,
        control_freq=args.control_freq,
        horizon=args.max_steps,
        camera_names=["agentview", "robot0_eye_in_hand"],
    )
    try:
        env.seed(args.seed)
    except Exception:
        pass

    args.out_dir.mkdir(parents=True, exist_ok=True)
    videos_dir = args.out_dir / "videos"
    debug_dir = args.out_dir / "debug_actions"
    per_episode: List[dict[str, Any]] = []
    success_count = 0

    try:
        for ep_idx in range(args.episodes):
            env.reset()
            client.reset()
            init_state = init_states[ep_idx % len(init_states)]
            obs = env.set_init_state(init_state)
            initial_geometry_post_set_state = extract_task_geometry(env, obs)

            # 对 JOINT_POSITION 模型，持续发送当前关节位置，让物体有时间落稳。
            for _ in range(args.num_steps_wait):
                dummy_policy_action = current_joint_gripper_action(obs)
                dummy_env_action = convert_policy_action_to_env_action(
                    policy_action=dummy_policy_action,
                    obs=obs,
                    controller=args.controller,
                    joint_input_type=args.joint_input_type,
                    joint_delta_scale=args.joint_delta_scale,
                )
                obs_next, done, _ = step_env_safely(env, dummy_env_action)
                if done:
                    break
                if obs_next:
                    obs = obs_next
            initial_geometry_post_wait = extract_task_geometry(env, obs)

            frames: List[np.ndarray] = []
            if ep_idx < args.render:
                frames.append(render_frame(obs))

            success = False
            episode_done = False
            step_count = 0
            debug_steps: List[dict[str, Any]] = []
            while step_count < args.max_steps and not episode_done:
                payload = build_payload(
                    obs=obs,
                    health=health,
                    task_index=task_index,
                    num_tasks=num_tasks,
                    chunk_size=args.chunk_size,
                )
                result = client.infer(payload)
                if "error" in result:
                    raise RuntimeError(f"server infer 失败: {result['error']}")

                action_chunk = np.asarray(result.get("action", []), dtype=np.float32)
                if action_chunk.ndim == 1:
                    action_chunk = action_chunk[None, :]
                if action_chunk.ndim != 2 or action_chunk.shape[1] != server_action_dim:
                    raise ValueError(f"server 返回动作维度异常: {action_chunk.shape}")

                for action in action_chunk:
                    obs_before = obs
                    env_action = convert_policy_action_to_env_action(
                        policy_action=action,
                        obs=obs_before,
                        controller=args.controller,
                        joint_input_type=args.joint_input_type,
                        joint_delta_scale=args.joint_delta_scale,
                    )
                    env_action, gripper_override_applied = apply_gripper_schedule(
                        env_action=env_action,
                        step_index=step_count,
                        force_open_steps=args.gripper_force_open_steps,
                    )
                    obs_next, done, info = step_env_safely(env, env_action)
                    step_count += 1

                    if obs_next:
                        obs = obs_next

                    if ep_idx < args.render and obs_next:
                        frames.append(render_frame(obs))

                    env_success = False
                    if hasattr(env, "check_success"):
                        try:
                            env_success = bool(env.check_success())
                        except Exception:
                            env_success = False
                    success = bool(env_success or info.get("success", False))
                    episode_done = bool(done or success)

                    if args.debug_log_actions and len(debug_steps) < args.debug_log_steps:
                        record: Dict[str, Any] = {
                            "step_index": step_count - 1,
                            "state_before": extract_joint_gripper_state(obs_before),
                            "geometry_before": extract_task_geometry(env, obs_before),
                            "policy_action_raw": np.asarray(action, dtype=np.float32).astype(float).tolist(),
                            "env_action_sent": np.asarray(env_action, dtype=np.float32).astype(float).tolist(),
                            "action_last_dim": float(np.asarray(action, dtype=np.float32).reshape(-1)[-1]),
                            "env_action_last_dim": float(np.asarray(env_action, dtype=np.float32).reshape(-1)[-1]),
                            "gripper_override_applied": bool(gripper_override_applied),
                            "done": bool(done),
                            "env_success": bool(env_success),
                            "info_success": bool(info.get("success", False)),
                            "terminated_episode_exception": bool(info.get("terminated_episode_exception", False)),
                        }
                        if obs_next:
                            record["state_after"] = extract_joint_gripper_state(obs_next)
                            record["geometry_after"] = extract_task_geometry(env, obs_next)
                        debug_steps.append(record)

                    if episode_done or step_count >= args.max_steps:
                        break

            success_count += int(success)
            per_episode.append(
                {
                    "episode_index": ep_idx,
                    "task_id": args.task_id,
                    "task_name": task.name,
                    "task_language": task.language,
                    "task_index_for_env_state": task_index,
                    "success": bool(success),
                    "steps": step_count,
                }
            )

            if args.debug_log_actions:
                debug_dir.mkdir(parents=True, exist_ok=True)
                debug_payload = {
                    "episode_index": ep_idx,
                    "task_id": args.task_id,
                    "task_name": task.name,
                    "task_language": task.language,
                    "task_index_for_env_state": task_index,
                    "server_action_dim": server_action_dim,
                    "num_steps_wait": args.num_steps_wait,
                    "joint_input_type": args.joint_input_type,
                    "joint_delta_scale": args.joint_delta_scale,
                    "gripper_force_open_steps": args.gripper_force_open_steps,
                    "initial_geometry_post_set_state": initial_geometry_post_set_state,
                    "initial_geometry_post_wait": initial_geometry_post_wait,
                    "success": bool(success),
                    "steps": step_count,
                    "logged_steps": debug_steps,
                }
                (debug_dir / f"episode_{ep_idx:03d}.json").write_text(
                    json.dumps(debug_payload, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )

            if ep_idx < args.render and frames:
                suffix = "success" if success else "failure"
                save_video(videos_dir / f"episode_{ep_idx:03d}_{suffix}.mp4", frames, fps=args.control_freq)
    finally:
        env.close()

    payload = {
        "server_url": args.server_url,
        "suite": args.suite,
        "task_id": args.task_id,
        "task_name": task.name,
        "task_language": task.language,
        "episodes": args.episodes,
        "successes": success_count,
        "success_rate": success_count / args.episodes if args.episodes else 0.0,
        "controller": args.controller,
        "num_steps_wait": args.num_steps_wait,
        "joint_input_type": args.joint_input_type,
        "joint_delta_scale": args.joint_delta_scale,
        "gripper_force_open_steps": args.gripper_force_open_steps,
        "task_conditioning": use_task_conditioning,
        "task_index_for_env_state": task_index,
        "per_episode": per_episode,
    }
    (args.out_dir / "eval_info.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
