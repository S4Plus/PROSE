#!/usr/bin/env python3
"""在 mimiclabs 环境中运行 LIBERO 仿真，并通过 OpenPI05 websocket server 评测。"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List

import cv2
import imageio.v2 as imageio
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
OPENPI_ROOT = ROOT / "openpi"
OPENPI_CLIENT_SRC = OPENPI_ROOT / "packages" / "openpi-client" / "src"
if str(OPENPI_CLIENT_SRC) not in sys.path:
    sys.path.insert(0, str(OPENPI_CLIENT_SRC))

from openpi_client import image_tools  # noqa: E402
from openpi_client import websocket_client_policy as _websocket_client_policy  # noqa: E402


DEFAULT_LIBERO_ROOT = Path("/workspace/liushuo/mimiclabs/LIBERO")
DEFAULT_TASKS_JSONL = Path("/workspace/liushuo/cupid/datasets/local/libero_object_multitask_act/meta/tasks.jsonl")
DEFAULT_MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
}


def ensure_runtime_paths(libero_root: Path) -> None:
    """确保 LIBERO 仓库可导入，并给 numba 提供可写缓存目录。"""
    os.environ.setdefault("NUMBA_CACHE_DIR", str(Path("/tmp/numba-cache")))
    try:
        Path(os.environ["NUMBA_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)
    except OSError:
        pass

    if str(libero_root) not in sys.path:
        sys.path.insert(0, str(libero_root))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LIBERO OpenPI05 websocket client evaluator")
    parser.add_argument("--server-url", type=str, required=True, help="OpenPI05 websocket 地址，例如 ws://127.0.0.1:8000")
    parser.add_argument("--libero-root", type=Path, default=DEFAULT_LIBERO_ROOT, help="LIBERO 仓库根目录")
    parser.add_argument(
        "--suite",
        type=str,
        default="libero_object",
        choices=["libero_object", "libero_goal", "libero_spatial", "libero_10"],
        help="LIBERO benchmark suite",
    )
    parser.add_argument("--task-id", type=int, required=True, help="benchmark 内的 task_id")
    parser.add_argument("--episodes", type=int, default=10, help="评估 episode 数")
    parser.add_argument("--max-steps", type=int, default=None, help="每条 episode 的最大步数，默认按 suite 自动选择")
    parser.add_argument("--seed", type=int, default=7, help="随机种子")
    parser.add_argument("--image-size", type=int, default=224, help="发送给 OpenPI05 的图像尺寸")
    parser.add_argument("--render-resolution", type=int, default=256, help="LIBERO 渲染分辨率")
    parser.add_argument("--controller", type=str, default="JOINT_POSITION", help="环境控制器；当前自训练 LIBERO OpenPI05 默认用 JOINT_POSITION")
    parser.add_argument(
        "--joint-input-type",
        type=str,
        default="absolute",
        choices=["absolute", "delta"],
        help="policy 前 7 维 joint 动作语义；当前数据默认是 absolute",
    )
    parser.add_argument(
        "--joint-delta-scale",
        type=float,
        default=0.05,
        help="JOINT_POSITION 控制器的单步 joint delta 缩放上限",
    )
    parser.add_argument(
        "--rotate-input-180",
        action="store_true",
        help="是否对模型输入图像做 180 度旋转；官方 pi05_libero 示例需要，自训练正常朝向数据通常不要开",
    )
    parser.add_argument("--control-freq", type=int, default=20, help="环境控制频率")
    parser.add_argument("--num-steps-wait", type=int, default=10, help="reset 后等待物体稳定的步数")
    parser.add_argument("--replan-steps", type=int, default=10, help="每次从 action chunk 执行多少步后重规划")
    parser.add_argument("--timeout", type=float, default=30.0, help="websocket 推理超时保留参数")
    parser.add_argument("--render", type=int, default=2, help="最多保存多少条视频")
    parser.add_argument("--save-rollout-pkl", action="store_true", help="是否额外保存 rollout pkl，供 influence score 使用")
    parser.add_argument("--pkl-dir", type=Path, default=None, help="rollout pkl 输出目录；默认写到 out-dir/pkl")
    parser.add_argument(
        "--prompt-source",
        type=str,
        default="auto",
        choices=["auto", "language", "name", "tasks_jsonl"],
        help="prompt 来源；auto 优先使用 tasks.jsonl，再回退到 benchmark language",
    )
    parser.add_argument("--tasks-jsonl", type=Path, default=None, help="训练数据中的 meta/tasks.jsonl，可用于复用 prompt")
    parser.add_argument("--default-prompt", type=str, default=None, help="覆盖当前 task 的 prompt")
    parser.add_argument("--out-dir", type=Path, required=True, help="输出目录，将写入 eval_info.json 和可选视频")
    return parser.parse_args()


def parse_server_url(server_url: str) -> tuple[str, int]:
    url = server_url.strip()
    if "://" in url:
        url = url.split("://", 1)[1]
    if "/" in url:
        url = url.split("/", 1)[0]
    if ":" in url:
        host, port_str = url.rsplit(":", 1)
        return host, int(port_str)
    return url, 8000


def normalize_text(text: str) -> str:
    return " ".join(text.strip().lower().replace("_", " ").split())


def load_tasks_jsonl(path: Path) -> dict[int, str]:
    mapping: dict[int, str] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            mapping[int(item["task_index"])] = str(item["task"])
    return mapping


def build_task_name_to_index(task_mapping: dict[int, str]) -> dict[str, int]:
    return {normalize_text(task): idx for idx, task in task_mapping.items()}


def resolve_training_task_index(task_name: str, task_language: str, tasks_jsonl: Path | None) -> int | None:
    """将 benchmark task 名称映射到训练时的 task_index。"""
    if tasks_jsonl is None or not tasks_jsonl.exists():
        return None
    task_mapping = load_tasks_jsonl(tasks_jsonl)
    name_to_index = build_task_name_to_index(task_mapping)
    for candidate in (normalize_text(task_name), normalize_text(task_language)):
        if candidate in name_to_index:
            return int(name_to_index[candidate])
    return None


def resolve_prompt(
    task_id: int,
    task_name: str,
    task_language: str,
    prompt_source: str,
    tasks_jsonl: Path | None,
    default_prompt: str | None,
) -> str:
    if default_prompt:
        return default_prompt

    task_prompts: dict[int, str] = {}
    resolved_training_index = None
    if tasks_jsonl is not None and tasks_jsonl.exists():
        task_prompts = load_tasks_jsonl(tasks_jsonl)
        resolved_training_index = resolve_training_task_index(task_name, task_language, tasks_jsonl)

    if prompt_source == "tasks_jsonl":
        if resolved_training_index is None or resolved_training_index not in task_prompts:
            raise KeyError(f"tasks.jsonl 中缺少任务 '{task_name}' / '{task_language}' 的 prompt。")
        return task_prompts[resolved_training_index]
    if prompt_source == "name":
        return normalize_text(task_name)
    if prompt_source == "language":
        return str(task_language)
    if resolved_training_index is not None and resolved_training_index in task_prompts:
        return task_prompts[resolved_training_index]
    return str(task_language)


def build_env_state_vector(task_name: str, task_language: str, tasks_jsonl: Path | None) -> np.ndarray | None:
    """根据 tasks.jsonl 构造 one-hot environment state。"""
    if tasks_jsonl is None or not tasks_jsonl.exists():
        return None
    mapping = load_tasks_jsonl(tasks_jsonl)
    if not mapping:
        return None
    total_tasks = max(mapping.keys()) + 1
    task_index = resolve_training_task_index(task_name, task_language, tasks_jsonl)
    if task_index is None or task_index < 0 or task_index >= total_tasks:
        return None
    env_state = np.zeros((total_tasks,), dtype=np.float32)
    env_state[task_index] = 1.0
    return env_state


def resolve_env_action_dim(env) -> int:
    """兼容 LIBERO wrapper，优先从底层 robosuite env 读取动作维度。"""
    for candidate in (env, getattr(env, "env", None)):
        if candidate is None:
            continue
        try:
            value = int(getattr(candidate, "action_dim"))
        except Exception:
            continue
        if value > 0:
            return value
    return -1


def quat2axisangle(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64).reshape(-1).copy()
    if quat.shape[0] != 4:
        raise ValueError(f"四元数维度异常: {quat.shape}")
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    den = math.sqrt(max(1.0 - quat[3] * quat[3], 0.0))
    if math.isclose(den, 0.0):
        return np.zeros(3, dtype=np.float64)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def current_joint_gripper_action(obs: dict[str, Any]) -> np.ndarray:
    """在等待物体稳定阶段，持续发送“保持当前关节位置”的 dummy 动作。"""
    joint = np.asarray(obs["robot0_joint_pos"], dtype=np.float32).reshape(-1)
    gripper_qpos = np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32).reshape(-1)
    grip_scalar = float(np.mean(gripper_qpos)) if gripper_qpos.size > 0 else 0.0
    return np.concatenate([joint, np.array([grip_scalar], dtype=np.float32)], axis=0)


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


def preprocess_policy_image(image: np.ndarray, image_size: int, rotate_input_180: bool) -> np.ndarray:
    image = np.flipud(np.asarray(image))
    if rotate_input_180:
        image = image[:, ::-1]
    image = np.ascontiguousarray(image)
    return image_tools.convert_to_uint8(image_tools.resize_with_pad(image, image_size, image_size))


def preprocess_render_image(image: np.ndarray, image_size: int) -> np.ndarray:
    image = np.ascontiguousarray(np.flipud(np.asarray(image)))
    return image_tools.convert_to_uint8(image_tools.resize_with_pad(image, image_size, image_size))


def build_observation(
    obs: dict[str, Any],
    prompt: str,
    image_size: int,
    rotate_input_180: bool,
) -> dict[str, Any]:
    base_image = preprocess_policy_image(obs["agentview_image"], image_size, rotate_input_180)
    wrist_image = preprocess_policy_image(obs["robot0_eye_in_hand_image"], image_size, rotate_input_180)
    joint = np.asarray(obs["robot0_joint_pos"], dtype=np.float32).reshape(-1)
    gripper = np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32).reshape(-1)
    state = np.concatenate((joint, gripper)).astype(np.float32)

    return {
        "observation/image": base_image,
        "observation/wrist_image": wrist_image,
        "observation/state": state,
        "prompt": prompt,
    }


def render_frame(obs: dict[str, Any], image_size: int) -> np.ndarray:
    base_image = preprocess_render_image(obs["agentview_image"], image_size)
    wrist_image = preprocess_render_image(obs["robot0_eye_in_hand_image"], image_size)
    if base_image.shape[:2] != wrist_image.shape[:2]:
        wrist_image = cv2.resize(wrist_image, (base_image.shape[1], base_image.shape[0]))
    return np.concatenate([base_image, wrist_image], axis=1)


def extract_rollout_row(
    obs: dict[str, Any],
    action: np.ndarray,
    env_state: np.ndarray | None,
    image_size: int,
    rotate_input_180: bool,
) -> dict[str, Any]:
    """提取 influence 计算所需的最小 rollout 行结构。"""
    front = preprocess_policy_image(obs["agentview_image"], image_size=image_size, rotate_input_180=rotate_input_180)
    wrist = preprocess_policy_image(
        obs["robot0_eye_in_hand_image"], image_size=image_size, rotate_input_180=rotate_input_180
    )
    joint = np.asarray(obs["robot0_joint_pos"], dtype=np.float32).reshape(-1)
    gripper = np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32).reshape(-1)
    state = np.concatenate((joint, gripper)).astype(np.float32)
    row: dict[str, Any] = {
        "observation.images.front": front.copy(),
        "observation.images.wrist": wrist.copy(),
        "observation.state": state.copy(),
        "action": np.asarray(action, dtype=np.float32).reshape(-1).copy(),
    }
    if env_state is not None:
        row["observation.environment_state"] = env_state.astype(np.float32).copy()
    return row


def save_rollout_pkl(
    pkl_path: Path,
    episode_rows: List[dict[str, Any]],
    success: bool,
    task_id: int,
    task_name: str,
    episode_index: int,
) -> None:
    """把单条 rollout 序列保存成 influence pipeline 可读取的 pkl。"""
    pkl_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "rows": episode_rows,
        "success": bool(success),
        "task_id": int(task_id),
        "task_name": str(task_name),
        "episode_index": int(episode_index),
    }
    with pkl_path.open("wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)


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

    benchmark = get_benchmark(args.suite.upper())(0)
    if args.task_id < 0 or args.task_id >= benchmark.n_tasks:
        raise ValueError(f"task_id 超出范围: {args.task_id}, suite={args.suite}, n_tasks={benchmark.n_tasks}")

    task = benchmark.get_task(args.task_id)
    tasks_jsonl = args.tasks_jsonl if args.tasks_jsonl is not None else (DEFAULT_TASKS_JSONL if DEFAULT_TASKS_JSONL.exists() else None)
    prompt = resolve_prompt(
        task_id=args.task_id,
        task_name=task.name,
        task_language=task.language,
        prompt_source=args.prompt_source,
        tasks_jsonl=tasks_jsonl,
        default_prompt=args.default_prompt,
    )
    task_index_for_env_state = resolve_training_task_index(task.name, task.language, tasks_jsonl)
    env_state_vector = build_env_state_vector(
        task_name=task.name,
        task_language=task.language,
        tasks_jsonl=tasks_jsonl,
    )

    max_steps = args.max_steps if args.max_steps is not None else DEFAULT_MAX_STEPS[args.suite]
    bddl_file = args.libero_root / "libero" / "libero" / "bddl_files" / task.problem_folder / task.bddl_file
    init_states_file = args.libero_root / "libero" / "libero" / "init_files" / task.problem_folder / task.init_states_file
    if not bddl_file.exists():
        raise FileNotFoundError(f"bddl 文件不存在: {bddl_file}")
    if not init_states_file.exists():
        raise FileNotFoundError(f"init state 文件不存在: {init_states_file}")

    init_states = torch.load(init_states_file, map_location="cpu", weights_only=False)

    host, port = parse_server_url(args.server_url)
    client = _websocket_client_policy.WebsocketClientPolicy(host=host, port=port)
    metadata = dict(getattr(client, "metadata", {}) or {})

    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl_file),
        camera_heights=args.render_resolution,
        camera_widths=args.render_resolution,
        control_freq=args.control_freq,
        controller=args.controller,
        horizon=max_steps,
        camera_names=["agentview", "robot0_eye_in_hand"],
    )
    env_action_dim = resolve_env_action_dim(env)
    if env_action_dim <= 0:
        raise ValueError(f"环境动作维度异常: {env_action_dim}")
    try:
        env.seed(args.seed)
    except Exception:
        pass

    args.out_dir.mkdir(parents=True, exist_ok=True)
    videos_dir = args.out_dir / "videos"
    pkl_dir = args.pkl_dir if args.pkl_dir is not None else (args.out_dir / "pkl")
    per_episode: List[dict[str, Any]] = []
    success_count = 0

    try:
        for ep_idx in range(args.episodes):
            env.reset()
            obs = env.set_init_state(init_states[ep_idx % len(init_states)])

            frames: List[np.ndarray] = []
            episode_rows: List[dict[str, Any]] = []
            if ep_idx < args.render:
                frames.append(render_frame(obs, args.image_size))

            success = False
            episode_done = False
            step_count = 0
            pending_actions: List[np.ndarray] = []

            while step_count < max_steps + args.num_steps_wait and not episode_done:
                if step_count < args.num_steps_wait:
                    dummy_policy_action = current_joint_gripper_action(obs)
                    dummy_env_action = convert_policy_action_to_env_action(
                        policy_action=dummy_policy_action,
                        obs=obs,
                        controller=args.controller,
                        joint_input_type=args.joint_input_type,
                        joint_delta_scale=args.joint_delta_scale,
                    )
                    obs_next, done, info = step_env_safely(env, dummy_env_action)
                    step_count += 1
                    if obs_next:
                        obs = obs_next
                    if ep_idx < args.render and obs_next:
                        frames.append(render_frame(obs, args.image_size))
                    success = bool(info.get("success", False))
                    episode_done = bool(done or success)
                    continue

                if not pending_actions:
                    policy_obs = build_observation(obs, prompt, args.image_size, args.rotate_input_180)
                    result = client.infer(policy_obs)
                    action_chunk = np.asarray(result["actions"], dtype=np.float32)
                    if action_chunk.ndim == 1:
                        action_chunk = action_chunk[None, :]
                    if action_chunk.ndim != 2 or action_chunk.shape[1] < env_action_dim:
                        raise ValueError(f"server 返回动作维度异常: {action_chunk.shape}")
                    if action_chunk.shape[0] < args.replan_steps:
                        raise ValueError(
                            f"replan_steps={args.replan_steps}，但 server 仅返回 {action_chunk.shape[0]} 步动作。"
                        )
                    pending_actions = [
                        np.asarray(a[:env_action_dim], dtype=np.float32)
                        for a in action_chunk[: args.replan_steps]
                    ]

                policy_action = pending_actions.pop(0)
                episode_rows.append(
                    extract_rollout_row(obs, policy_action, env_state_vector, args.image_size, args.rotate_input_180)
                )
                action = convert_policy_action_to_env_action(
                    policy_action=policy_action,
                    obs=obs,
                    controller=args.controller,
                    joint_input_type=args.joint_input_type,
                    joint_delta_scale=args.joint_delta_scale,
                )
                obs_next, done, info = step_env_safely(env, action)
                step_count += 1

                if obs_next:
                    obs = obs_next
                if ep_idx < args.render and obs_next:
                    frames.append(render_frame(obs, args.image_size))

                env_success = False
                if hasattr(env, "check_success"):
                    try:
                        env_success = bool(env.check_success())
                    except Exception:
                        env_success = False
                success = bool(env_success or info.get("success", False))
                episode_done = bool(done or success)

            success_count += int(success)
            per_episode.append(
                {
                    "episode_index": ep_idx,
                    "task_id": args.task_id,
                    "task_name": task.name,
                    "task_language": task.language,
                    "prompt": prompt,
                    "success": bool(success),
                    "steps": step_count,
                }
            )

            if ep_idx < args.render and frames:
                suffix = "success" if success else "failure"
                save_video(videos_dir / f"episode_{ep_idx:03d}_{suffix}.mp4", frames, fps=args.control_freq)
            if args.save_rollout_pkl and episode_rows:
                suffix = "success" if success else "failure"
                save_rollout_pkl(
                    pkl_path=pkl_dir / f"task{args.task_id:02d}_ep{ep_idx:03d}_{suffix}.pkl",
                    episode_rows=episode_rows,
                    success=bool(success),
                    task_id=args.task_id,
                    task_name=task.name,
                    episode_index=ep_idx,
                )
    finally:
        try:
            env.close()
        finally:
            close_fn = getattr(client, "close", None)
            if callable(close_fn):
                close_fn()

    payload = {
        "server_url": args.server_url,
        "suite": args.suite,
        "task_id": args.task_id,
        "task_name": task.name,
        "task_language": task.language,
        "prompt": prompt,
        "episodes": args.episodes,
        "successes": success_count,
        "success_rate": success_count / args.episodes if args.episodes else 0.0,
        "replan_steps": args.replan_steps,
        "num_steps_wait": args.num_steps_wait,
        "image_size": args.image_size,
        "controller": args.controller,
        "joint_input_type": args.joint_input_type,
        "joint_delta_scale": args.joint_delta_scale,
        "rotate_input_180": bool(args.rotate_input_180),
        "env_action_dim": env_action_dim,
        "server_metadata": metadata,
        "task_index_for_env_state": task_index_for_env_state,
        "env_state_vector": env_state_vector.astype(int).tolist() if env_state_vector is not None else None,
        "save_rollout_pkl": bool(args.save_rollout_pkl),
        "per_episode": per_episode,
    }
    (args.out_dir / "eval_info.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
