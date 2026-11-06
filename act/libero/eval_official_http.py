#!/usr/bin/env python3
"""使用 LIBERO 官方 benchmark 协议，通过 HTTP ACT server 评测模型。

这个脚本的目标不是复用之前自定义的 episode 循环，而是尽量贴近
LIBERO 官方示例 `/workspace/liushuo/cupid/openpi/examples/libero/main.py`
的评测流程：

1. 通过 `benchmark.get_benchmark_dict()` 构造官方 task suite
2. 通过 `task_suite.get_task_init_states(task_id)` 获取官方初始状态
3. 通过官方 task/bddl 创建 `OffScreenRenderEnv`
4. reset 后先等待 `num_steps_wait` 步让物体落稳
5. 按 chunk/replan 周期向远端 ACT server 请求动作

同时保留 ACT 所需的两点适配：

- `observation.state = joint_pos(7) + gripper_qpos(2)`
- policy 输出的前 7 维若为 absolute joint target，则需转换成
  robosuite `JOINT_POSITION` 控制器需要的 delta action
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import pickle
import sys
from collections import deque
from pathlib import Path
from typing import Any
from urllib import error, request

import cv2
import imageio.v2 as imageio
import numpy as np
import torch


DEFAULT_LIBERO_ROOT = Path("/workspace/liushuo/mimiclabs/LIBERO")
DEFAULT_TASK_MAP = Path("/workspace/liushuo/cupid/output/act/libero_object_multitask/task_index_map.json")
DEFAULT_OUT_ROOT = Path("/workspace/liushuo/cupid/output/eval/libero_official_http")


def ensure_runtime_paths(libero_root: Path) -> None:
    """确保 LIBERO 仓库可导入，并为 numba 提供可写缓存目录。"""
    os.environ.setdefault("NUMBA_CACHE_DIR", str(Path("/tmp/numba-cache")))
    try:
        Path(os.environ["NUMBA_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)
    except OSError:
        pass

    if str(libero_root) not in sys.path:
        sys.path.insert(0, str(libero_root))


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


class ACTHTTPClient:
    """对接 ACT 推理 server 的极简 HTTP client。"""

    def __init__(self, server_url: str, timeout: float) -> None:
        self.server_url = server_url.rstrip("/")
        self.timeout = timeout

    def health(self) -> dict[str, object]:
        return http_get(f"{self.server_url}/health", self.timeout)

    def reset(self) -> dict[str, object]:
        return http_post(f"{self.server_url}/reset", {}, self.timeout)

    def infer(self, payload: dict[str, object]) -> dict[str, object]:
        return http_post(f"{self.server_url}/infer", payload, self.timeout)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="基于 LIBERO 官方 benchmark 协议评测 ACT HTTP 模型")
    parser.add_argument("--server-url", type=str, required=True, help="ACT server 地址，例如 http://127.0.0.1:8000")
    parser.add_argument("--libero-root", type=Path, default=DEFAULT_LIBERO_ROOT, help="LIBERO 仓库根目录")
    parser.add_argument(
        "--suite",
        type=str,
        default="libero_object",
        choices=["libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"],
        help="LIBERO 官方 task suite 名称",
    )
    parser.add_argument("--task-id", type=int, default=None, help="只评测单个 task；留空则跑整个 suite")
    parser.add_argument("--episodes", type=int, default=10, help="每个 task 评测多少条 episode")
    parser.add_argument("--seed", type=int, default=7, help="随机种子")
    parser.add_argument("--env-image-size", type=int, default=128, help="环境渲染分辨率")
    parser.add_argument("--control-freq", type=int, default=20, help="环境控制频率")
    parser.add_argument("--num-steps-wait", type=int, default=10, help="reset 后等待物体稳定的步数")
    parser.add_argument("--replan-steps", type=int, default=1, help="每次只执行 server 返回 chunk 的前多少步，然后重规划")
    parser.add_argument("--timeout", type=float, default=30.0, help="HTTP 请求超时")
    parser.add_argument("--render", type=int, default=1, help="每个 task 最多保存多少条视频")
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT, help="评测结果输出根目录")
    parser.add_argument("--save-rollout-pkl", action="store_true", help="是否额外保存 rollout pkl，供 influence score 使用")
    parser.add_argument(
        "--pkl-dir",
        type=Path,
        default=None,
        help="rollout pkl 输出目录；默认写到 out-root/pkl",
    )
    parser.add_argument("--controller", type=str, default="JOINT_POSITION", help="LIBERO 控制器类型")
    parser.add_argument(
        "--joint-input-type",
        type=str,
        default="absolute",
        choices=["absolute", "delta"],
        help="policy 前 7 维 joint 动作语义；ACT 数据默认是 absolute",
    )
    parser.add_argument(
        "--joint-delta-scale",
        type=float,
        default=0.05,
        help="JOINT_POSITION 控制器的单步 joint delta 缩放上限",
    )
    parser.add_argument(
        "--task-conditioning",
        type=str,
        default="auto",
        choices=["auto", "on", "off"],
        help="是否向 server 发送 env_state one-hot；auto 根据 /health 判断",
    )
    parser.add_argument(
        "--task-map-json",
        type=Path,
        default=None,
        help="训练时生成的 task_index_map.json，用于构造多任务 one-hot",
    )
    parser.add_argument("--chunk-size", type=int, default=None, help="覆盖 server 默认返回 chunk 长度")
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
    for candidate in candidates:
        if candidate in mapping:
            return mapping[candidate]
    raise KeyError(f"任务 '{task_name}' / '{task_language}' 未出现在 task_index_map.json 中。")


def encode_png_base64(image_rgb: np.ndarray) -> str:
    """把 RGB 图像编码成 base64 PNG，便于 HTTP 传输。"""
    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".png", bgr)
    if not ok:
        raise ValueError("PNG 编码失败。")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def build_payload(
    obs: dict[str, Any],
    health: dict[str, object],
    task_index: int | None,
    num_tasks: int | None,
    chunk_size: int | None,
) -> dict[str, object]:
    """构造送给 ACT server 的观测字典。"""
    image_keys = list(health.get("image_keys", []))
    expect_state = bool(health.get("expect_state", False))
    expect_env_state = bool(health.get("expect_env_state", False))

    front = np.flipud(np.asarray(obs["agentview_image"]))
    wrist = np.flipud(np.asarray(obs["robot0_eye_in_hand_image"]))

    images: dict[str, str] = {}
    for key in image_keys:
        suffix = key.split(".")[-1]
        if suffix in {"front", "agentview"}:
            images[key] = encode_png_base64(front)
        elif suffix in {"wrist", "robot0_eye_in_hand"}:
            images[key] = encode_png_base64(wrist)
        else:
            raise KeyError(f"未知图像键: {key}")

    payload: dict[str, object] = {"images": images}

    if expect_state:
        joint = np.asarray(obs["robot0_joint_pos"], dtype=np.float32).reshape(-1)
        gripper = np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32).reshape(-1)
        state = np.concatenate([joint, gripper], axis=0)
        payload["state"] = state.astype(float).tolist()

    if expect_env_state:
        if task_index is None or num_tasks is None:
            raise ValueError("server 需要 env_state，但当前未提供 task map 解析结果。")
        env_state = build_env_state_vector(task_index, num_tasks)
        payload["env_state"] = env_state.astype(float).tolist()

    if chunk_size is not None:
        payload["chunk_size"] = int(chunk_size)

    return payload


def build_env_state_vector(task_index: int, num_tasks: int) -> np.ndarray:
    """根据 task 索引构造 one-hot env_state。"""
    env_state = np.zeros((num_tasks,), dtype=np.float32)
    env_state[task_index] = 1.0
    return env_state


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
    """把 policy 动作转换成 robosuite 真正接收的控制指令。"""
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


def render_frame(obs: dict[str, Any]) -> np.ndarray:
    """把主视角与手腕视角拼成一个可回放帧。"""
    front = np.flipud(np.asarray(obs["agentview_image"]))
    wrist = np.flipud(np.asarray(obs["robot0_eye_in_hand_image"]))
    if wrist.shape[:2] != front.shape[:2]:
        wrist = cv2.resize(wrist, (front.shape[1], front.shape[0]))
    return np.concatenate([front, wrist], axis=1)


def extract_rollout_row(
    obs: dict[str, Any],
    policy_action: np.ndarray,
    env_state_vector: np.ndarray | None,
) -> dict[str, Any]:
    """提取 influence 计算所需的最小 rollout 行结构。"""
    front = np.flipud(np.asarray(obs["agentview_image"], dtype=np.uint8))
    wrist = np.flipud(np.asarray(obs["robot0_eye_in_hand_image"], dtype=np.uint8))
    joint = np.asarray(obs["robot0_joint_pos"], dtype=np.float32).reshape(-1)
    gripper = np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32).reshape(-1)
    state = np.concatenate([joint, gripper], axis=0)

    row: dict[str, Any] = {
        "observation.images.front": front.copy(),
        "observation.images.wrist": wrist.copy(),
        "observation.state": state.astype(np.float32),
        "action": np.asarray(policy_action, dtype=np.float32).reshape(-1).copy(),
    }
    if env_state_vector is not None:
        row["observation.environment_state"] = env_state_vector.astype(np.float32).copy()
    return row


def save_video(video_path: Path, frames: list[np.ndarray], fps: int) -> None:
    """优先保存成 H.264，失败时再回退到 mp4v。"""
    if not frames:
        return

    video_path.parent.mkdir(parents=True, exist_ok=True)
    normalized = [np.asarray(frame, dtype=np.uint8) for frame in frames]
    try:
        imageio.mimwrite(
            str(video_path),
            normalized,
            fps=float(fps),
            codec="libx264",
            format="FFMPEG",
            pixelformat="yuv420p",
            macro_block_size=1,
        )
        return
    except Exception:
        pass

    h, w = normalized[0].shape[:2]
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (w, h))
    try:
        for frame in normalized:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()


def step_env_safely(env, action: np.ndarray) -> tuple[dict[str, Any], bool, dict[str, Any]]:
    """兼容 LIBERO/robosuite 在已终止 episode 上继续 step 时抛异常的行为。"""
    try:
        obs, _, done, info = env.step(action.tolist())
        return obs, bool(done), info or {}
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


def get_max_steps(suite: str) -> int:
    """沿用官方示例里按 suite 给出的 horizon。"""
    max_steps_by_suite = {
        "libero_spatial": 220,
        "libero_object": 280,
        "libero_goal": 300,
        "libero_10": 520,
        "libero_90": 400,
    }
    if suite not in max_steps_by_suite:
        raise ValueError(f"未知 suite: {suite}")
    return max_steps_by_suite[suite]


def get_libero_env(task, resolution: int, seed: int, controller: str, control_freq: int, horizon: int):
    """按官方 task/bddl 路径创建环境，同时显式对齐 ACT 所需 controller。"""
    from libero.libero import get_libero_path  # type: ignore
    from libero.libero.envs import OffScreenRenderEnv  # type: ignore

    task_bddl_file = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(
        bddl_file_name=str(task_bddl_file),
        camera_heights=resolution,
        camera_widths=resolution,
        controller=controller,
        control_freq=control_freq,
        horizon=horizon,
        camera_names=["agentview", "robot0_eye_in_hand"],
    )
    try:
        env.seed(seed)
    except Exception:
        pass
    return env


def load_task_init_states(task) -> Any:
    """按 LIBERO 官方路径加载 init states，并兼容 PyTorch 2.6 的 weights_only 默认变更。"""
    from libero.libero import get_libero_path  # type: ignore

    init_states_path = Path(get_libero_path("init_states")) / task.problem_folder / task.init_states_file
    return torch.load(init_states_path, map_location="cpu", weights_only=False)


def save_rollout_pkl(
    pkl_path: Path,
    episode_rows: list[dict[str, Any]],
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


def evaluate_task(
    task_id: int,
    task,
    task_suite,
    client: ACTHTTPClient,
    health: dict[str, object],
    args: argparse.Namespace,
    task_map: dict[str, int] | None,
    total_tasks: int | None,
) -> dict[str, Any]:
    """评测单个 task，并返回统计结果。"""
    max_steps = get_max_steps(args.suite)
    init_states = load_task_init_states(task)
    if args.episodes > len(init_states):
        raise ValueError(
            f"--episodes={args.episodes} 超过该 task 可用 init states 数量 {len(init_states)}。"
        )

    task_index = None
    env_state_vector = None
    if bool(health.get("expect_env_state", False)):
        if task_map is None or total_tasks is None:
            raise ValueError("server 需要 env_state，但未提供 task_index_map.json。")
        task_index = resolve_task_index(task.name, task.language, task_map)
        env_state_vector = build_env_state_vector(task_index, total_tasks)

    if env_state_vector is not None:
        print(
            f"[task {task_id}] task_id={task_id}, task_name={task.name}, "
            f"task_index_for_env_state={task_index}, "
            f"env_state={env_state_vector.astype(int).tolist()}",
            flush=True,
        )
    else:
        print(
            f"[task {task_id}] task_id={task_id}, task_name={task.name}, env_state=disabled",
            flush=True,
        )

    env = get_libero_env(
        task=task,
        resolution=args.env_image_size,
        seed=args.seed,
        controller=args.controller,
        control_freq=args.control_freq,
        horizon=max_steps,
    )
    if int(getattr(env, "action_dim", -1)) not in {-1, 8} and args.controller.upper() == "JOINT_POSITION":
        raise ValueError(f"环境动作维度异常：controller={args.controller}, env.action_dim={env.action_dim}")
    task_dir = args.out_root / f"{task_id:02d}_{task.name}"
    task_dir.mkdir(parents=True, exist_ok=True)
    videos_dir = task_dir / "videos"
    pkl_dir = args.pkl_dir if args.pkl_dir is not None else (args.out_root / "pkl")

    per_episode: list[dict[str, Any]] = []
    success_count = 0

    try:
        for episode_idx in range(args.episodes):
            env.reset()
            client.reset()
            obs = env.set_init_state(init_states[episode_idx])
            frames: list[np.ndarray] = []
            episode_rows: list[dict[str, Any]] = []
            if episode_idx < args.render:
                frames.append(render_frame(obs))

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
                if episode_idx < args.render and obs_next:
                    frames.append(render_frame(obs))

            action_plan: deque[np.ndarray] = deque()
            step_count = 0
            success = False
            done = False

            while step_count < max_steps and not done:
                if not action_plan:
                    payload = build_payload(
                        obs=obs,
                        health=health,
                        task_index=task_index,
                        num_tasks=total_tasks,
                        chunk_size=args.chunk_size,
                    )
                    result = client.infer(payload)
                    if "error" in result:
                        raise RuntimeError(f"server infer 失败: {result['error']}")

                    action_chunk = np.asarray(result.get("action", []), dtype=np.float32)
                    if action_chunk.ndim == 1:
                        action_chunk = action_chunk[None, :]
                    if action_chunk.ndim != 2:
                        raise ValueError(f"server 返回动作维度异常: {action_chunk.shape}")
                    if action_chunk.shape[0] < args.replan_steps:
                        raise ValueError(
                            f"server 返回 chunk 长度 {action_chunk.shape[0]} 小于 replan_steps={args.replan_steps}"
                        )
                    action_plan.extend(action_chunk[: args.replan_steps])

                policy_action = action_plan.popleft()
                episode_rows.append(extract_rollout_row(obs, policy_action, env_state_vector))
                env_action = convert_policy_action_to_env_action(
                    policy_action=policy_action,
                    obs=obs,
                    controller=args.controller,
                    joint_input_type=args.joint_input_type,
                    joint_delta_scale=args.joint_delta_scale,
                )
                obs_next, done_signal, info = step_env_safely(env, env_action)
                step_count += 1

                if obs_next:
                    obs = obs_next
                if episode_idx < args.render and obs_next:
                    frames.append(render_frame(obs))

                env_success = False
                if hasattr(env, "check_success"):
                    try:
                        env_success = bool(env.check_success())
                    except Exception:
                        env_success = False
                success = bool(env_success or info.get("success", False))
                done = bool(done_signal or success)

            success_count += int(success)
            per_episode.append(
                {
                    "episode_index": episode_idx,
                    "success": bool(success),
                    "steps": step_count,
                }
            )

            if episode_idx < args.render and frames:
                suffix = "success" if success else "failure"
                save_video(videos_dir / f"episode_{episode_idx:03d}_{suffix}.mp4", frames, fps=args.control_freq)
            if args.save_rollout_pkl and episode_rows:
                suffix = "success" if success else "failure"
                save_rollout_pkl(
                    pkl_path=pkl_dir / f"task{task_id:02d}_ep{episode_idx:03d}_{suffix}.pkl",
                    episode_rows=episode_rows,
                    success=bool(success),
                    task_id=task_id,
                    task_name=task.name,
                    episode_index=episode_idx,
                )
    finally:
        env.close()

    task_summary = {
        "task_id": task_id,
        "task_name": task.name,
        "task_language": task.language,
        "episodes": args.episodes,
        "successes": success_count,
        "success_rate": success_count / args.episodes if args.episodes else 0.0,
        "num_steps_wait": args.num_steps_wait,
        "replan_steps": args.replan_steps,
        "controller": args.controller,
        "joint_input_type": args.joint_input_type,
        "joint_delta_scale": args.joint_delta_scale,
        "task_conditioning": bool(health.get("expect_env_state", False)),
        "task_index_for_env_state": task_index,
        "env_state_vector": env_state_vector.astype(int).tolist() if env_state_vector is not None else None,
        "save_rollout_pkl": bool(args.save_rollout_pkl),
        "per_episode": per_episode,
    }
    (task_dir / "eval_info.json").write_text(json.dumps(task_summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return task_summary


def main() -> None:
    args = parse_args()
    ensure_runtime_paths(args.libero_root)

    from libero.libero import benchmark  # type: ignore

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

    task_map = None
    total_tasks = None
    if use_task_conditioning:
        if task_map_path is None or not task_map_path.exists():
            raise FileNotFoundError("server 需要 env_state，但未找到 task_index_map.json。")
        task_map, total_tasks = load_task_map(task_map_path)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.suite]()
    task_ids = [args.task_id] if args.task_id is not None else list(range(task_suite.n_tasks))

    for task_id in task_ids:
        if task_id < 0 or task_id >= task_suite.n_tasks:
            raise ValueError(f"task_id 超出范围: {task_id}, suite={args.suite}, n_tasks={task_suite.n_tasks}")

    args.out_root.mkdir(parents=True, exist_ok=True)
    task_summaries: list[dict[str, Any]] = []
    total_successes = 0
    total_episodes = 0

    for task_id in task_ids:
        task = task_suite.get_task(task_id)
        summary = evaluate_task(
            task_id=task_id,
            task=task,
            task_suite=task_suite,
            client=client,
            health=health,
            args=args,
            task_map=task_map,
            total_tasks=total_tasks,
        )
        task_summaries.append(summary)
        total_successes += int(summary["successes"])
        total_episodes += int(summary["episodes"])
        print(
            f"[task {task_id}] {task.name}: "
            f"{summary['successes']}/{summary['episodes']} "
            f"({summary['success_rate']:.3f})"
        )

    suite_summary = {
        "server_url": args.server_url,
        "suite": args.suite,
        "task_ids": task_ids,
        "episodes_per_task": args.episodes,
        "total_tasks": len(task_ids),
        "total_episodes": total_episodes,
        "total_successes": total_successes,
        "overall_success_rate": total_successes / total_episodes if total_episodes else 0.0,
        "num_steps_wait": args.num_steps_wait,
        "replan_steps": args.replan_steps,
        "controller": args.controller,
        "joint_input_type": args.joint_input_type,
        "joint_delta_scale": args.joint_delta_scale,
        "task_conditioning": use_task_conditioning,
        "tasks": task_summaries,
    }
    (args.out_root / "summary.json").write_text(json.dumps(suite_summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(suite_summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
