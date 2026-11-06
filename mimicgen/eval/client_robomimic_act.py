#!/usr/bin/env python3
"""
HTTP client for evaluating a LeRobot ACT policy server on robomimic/robosuite tasks.

Typical usage:
    python client_robomimic_act.py \
        --server_url http://127.0.0.1:8000/act \
        --env_name can \
        --n_rollouts 100 \
        --horizon 400 \
        --video_dir /tmp/can_eval_videos


python /workspace/liushuo/cupid/mimicgen/eval/client_robomimic_act.py \
    --env_name can \
    --video_dir /workspace/liushuo/cupid/mimicgen/eval/tmp/can\
    --controller JOINT_POSITION    \
    --joint_input_type absolute   \
    --robots Panda       \
    --save_rollouts_dir /workspace/liushuo/cupid/mimicgen/eval/tmp/can/episodes


python /workspace/liushuo/cupid/mimicgen/eval/client_robomimic_act.py \
    --env_name can \
    --video_dir /workspace/liushuo/cupid/mimicgen/eval/tmp/can_0.8\
    --controller JOINT_POSITION    \
    --joint_input_type absolute   \
    --robots Panda       \
    --save_rollouts_dir /workspace/liushuo/cupid/mimicgen/eval/tmp/can_0.8/episodes


    
"""

from __future__ import annotations

import argparse
import base64
import copy
import difflib
import json
import os
import pickle
import random
import re
import sys
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit, urlunsplit
from urllib.request import Request, urlopen

# Prefer EGL offscreen in headless setup unless user already set these.
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
# Some bundled robosuite / numba combinations need an explicit writable cache dir.
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba-cache")

import imageio
import numpy as np
import robosuite as suite
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.obs_utils as ObsUtils
from robomimic.envs.env_base import EnvType
from PIL import Image

try:
    from robosuite.controllers import load_part_controller_config
except Exception:  # pragma: no cover - older robosuite without part-controller helper
    load_part_controller_config = None

# Optional env registries (MimicGen / legacy).
try:  # pragma: no cover
    import mimicgen  # noqa: F401
except Exception:
    pass
try:  # pragma: no cover
    import mimicgen_envs  # noqa: F401
except Exception:
    pass

try:
    import cv2
except Exception:  # pragma: no cover - optional fallback to PIL
    cv2 = None

try:
    from robosuite.controllers import load_controller_config as _load_controller_config_legacy
except Exception:  # pragma: no cover - not available in robosuite>=1.5
    _load_controller_config_legacy = None

try:
    from robosuite.controllers.composite.composite_controller_factory import refactor_composite_controller_config
except Exception:  # pragma: no cover - older robosuite without composite helper
    refactor_composite_controller_config = None

try:
    import pandas as pd
except Exception:  # pragma: no cover - optional for rollout persistence
    pd = None


TASK_NAME_CANDIDATES = {
    "lift": ["Lift"],
    "can": ["Can", "PickPlaceCan"],
    "square": ["Square", "NutAssemblySquare"],
    "transport": ["Transport", "TwoArmTransport"],
    "tool_hang": ["ToolHang"],
}

IMAGE_KEY_ALIASES = {
    "observation.images.agentview": ["agentview_image", "agentview"],
    "observation.images.wrist": ["robot0_eye_in_hand_image", "robot0_eye_in_hand", "wrist"],
    "observation.images.robot0_eye_in_hand": [
        "robot0_eye_in_hand_image",
        "robot0_eye_in_hand",
        "wrist",
        "observation.images.wrist",
    ],
    "observation.images.head_camera": ["agentview_image", "agentview", "head_camera"],
    "observation.images.right_wrist_camera": [
        "robot0_eye_in_hand_image",
        "robot0_eye_in_hand",
        "right_wrist_camera",
    ],
}

JOINT_KEY_RE = re.compile(r"robot(\d+)_joint_pos$")
GRIPPER_KEY_RE = re.compile(r"robot(\d+)_gripper_qpos$")


def _to_json_bytes(payload: Dict[str, Any]) -> bytes:
    return json.dumps(payload).encode("utf-8")


def _http_json(
    method: str,
    url: str,
    payload: Optional[Dict[str, Any]] = None,
    timeout_s: float = 10.0,
) -> Dict[str, Any]:
    body = _to_json_bytes(payload) if payload is not None else None
    req = Request(
        url=url,
        method=method,
        data=body,
        headers={"Content-Type": "application/json"} if body is not None else {},
    )
    try:
        with urlopen(req, timeout=timeout_s) as resp:
            content = resp.read().decode("utf-8")
        return json.loads(content)
    except HTTPError as exc:
        detail = exc.read().decode("utf-8") if hasattr(exc, "read") else str(exc)
        raise RuntimeError(f"HTTPError {exc.code} from {url}: {detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"URLError from {url}: {exc}") from exc


def _health_url(server_url: str) -> str:
    if server_url.endswith("/act"):
        return server_url[: -len("/act")] + "/health"
    if server_url.endswith("/infer"):
        return server_url[: -len("/infer")] + "/health"
    return server_url.rstrip("/") + "/health"


def _reset_url(server_url: str) -> str:
    if server_url.endswith("/act"):
        return server_url[: -len("/act")] + "/reset"
    if server_url.endswith("/infer"):
        return server_url[: -len("/infer")] + "/reset"
    return server_url.rstrip("/") + "/reset"


def _override_server_port(server_url: str, port: Optional[int]) -> str:
    if port is None:
        return server_url
    parts = urlsplit(server_url)
    hostname = parts.hostname or "127.0.0.1"
    auth = ""
    if parts.username:
        auth = parts.username
        if parts.password:
            auth += f":{parts.password}"
        auth += "@"
    netloc = f"{auth}{hostname}:{int(port)}"
    return urlunsplit((parts.scheme or "http", netloc, parts.path, parts.query, parts.fragment))


def _parse_task_name(task: str) -> str:
    key = task.strip()
    available = set(getattr(suite, "ALL_ENVIRONMENTS", []))

    lookup_keys: List[str] = []

    def _add_lookup(candidate: str) -> None:
        value = str(candidate).strip()
        if value and value not in lookup_keys:
            lookup_keys.append(value)

    _add_lookup(key)

    # Accept common suffix variants like square_d1 / ToolHang-test by
    # progressively stripping the last hyphen/underscore segment.
    reduced = key
    while True:
        m = re.match(r"^(.*?)[_-][^_-]+$", reduced)
        if m is None:
            break
        reduced = m.group(1).strip()
        if not reduced:
            break
        _add_lookup(reduced)

    for candidate in lookup_keys:
        if candidate in available:
            return candidate

    for lowered in [candidate.lower() for candidate in lookup_keys]:
        if lowered not in TASK_NAME_CANDIDATES:
            continue
        for candidate in TASK_NAME_CANDIDATES[lowered]:
            if not available or candidate in available:
                return candidate
        raise ValueError(
            f"Task alias '{task}' has no compatible environment in this robosuite build. "
            f"Tried: {TASK_NAME_CANDIDATES[lowered]}. "
            f"Available: {sorted(available)}"
        )

    # Also allow direct canonical names even if ALL_ENVIRONMENTS cannot be read.
    for candidates in TASK_NAME_CANDIDATES.values():
        for lookup in lookup_keys:
            if lookup in candidates:
                return lookup

    supported_aliases = sorted(TASK_NAME_CANDIDATES.keys())
    suggestions: List[str] = []
    for lowered in [candidate.lower() for candidate in lookup_keys]:
        for match in difflib.get_close_matches(lowered, supported_aliases, n=2, cutoff=0.75):
            if match not in suggestions:
                suggestions.append(match)
    suggestion_text = f" Did you mean: {suggestions}?" if suggestions else ""

    raise ValueError(
        f"Unsupported task '{task}'. "
        f"Supported aliases: {sorted(TASK_NAME_CANDIDATES.keys())}. "
        f"Available envs: {sorted(available)}."
        f"{suggestion_text}"
    )


def _parse_robots(robots_arg: str, env_name: str) -> str | List[str]:
    robots = [r.strip() for r in robots_arg.split(",") if r.strip()]
    if not robots:
        robots = ["Panda"]
    if env_name.lower() in {"transport", "twoarmtransport"} and len(robots) == 1:
        robots = [robots[0], robots[0]]
    return robots[0] if len(robots) == 1 else robots


def _encode_image_to_base64(image: np.ndarray) -> str:
    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    if arr.dtype != np.uint8:
        if np.issubdtype(arr.dtype, np.floating) and np.max(arr) <= 1.0:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if cv2 is not None:
        bgr = arr[..., ::-1]
        ok, buffer = cv2.imencode(".jpg", bgr)
        if not ok:
            raise ValueError("Failed to encode image")
        raw = buffer.tobytes()
    else:
        buf = BytesIO()
        Image.fromarray(arr).save(buf, format="JPEG")
        raw = buf.getvalue()
    return base64.b64encode(raw).decode("ascii")


def _candidate_obs_image_keys(required_key: str) -> List[str]:
    candidates = [required_key]
    candidates.extend(IMAGE_KEY_ALIASES.get(required_key, []))
    # Generic fallback based on the trailing key, e.g. observation.images.foo -> foo_image / foo
    tail = required_key.split(".")[-1]
    candidates.append(f"{tail}_image")
    candidates.append(tail)
    # Deduplicate while preserving order.
    deduped: List[str] = []
    seen = set()
    for key in candidates:
        if key in seen:
            continue
        seen.add(key)
        deduped.append(key)
    return deduped


def _extract_images_for_policy(obs: Dict[str, Any], required_image_keys: List[str]) -> Dict[str, str]:
    images: Dict[str, str] = {}
    for required_key in required_image_keys:
        img = None
        for candidate in _candidate_obs_image_keys(required_key):
            if candidate in obs:
                img = obs[candidate]
                break
        if img is None:
            raise KeyError(
                f"Observation missing required image key '{required_key}'. "
                f"Available keys: {sorted(list(obs.keys()))}"
            )
        images[required_key] = _encode_image_to_base64(np.asarray(img))
    return images


def _build_joint_gripper_state(obs: Dict[str, Any]) -> Optional[np.ndarray]:
    joint_keys: List[Tuple[int, str]] = []
    for key in obs.keys():
        m = JOINT_KEY_RE.match(key)
        if m:
            joint_keys.append((int(m.group(1)), key))
    if not joint_keys:
        return None
    joint_keys = sorted(joint_keys, key=lambda x: x[0])

    gripper_keys: Dict[int, str] = {}
    for key in obs.keys():
        m = GRIPPER_KEY_RE.match(key)
        if m:
            gripper_keys[int(m.group(1))] = key

    parts: List[np.ndarray] = []
    for robot_idx, joint_key in joint_keys:
        parts.append(np.asarray(obs[joint_key], dtype=np.float32).reshape(-1))
        gkey = gripper_keys.get(robot_idx)
        if gkey is not None:
            parts.append(np.asarray(obs[gkey], dtype=np.float32).reshape(-1))
    return np.concatenate(parts, axis=0).astype(np.float32)


def _is_numeric_array(value: Any) -> bool:
    try:
        arr = np.asarray(value)
    except Exception:
        return False
    return np.issubdtype(arr.dtype, np.number)


def _is_likely_image_key(key: str, arr: np.ndarray) -> bool:
    key_l = key.lower()
    if key.startswith("observation.images.") or "image" in key_l:
        return True
    # Keys like "agentview" / "robot0_eye_in_hand" can still be images.
    if arr.ndim >= 2 and arr.size >= 1024 and arr.shape[-1] in (1, 3, 4):
        if ("view" in key_l) or ("camera" in key_l) or ("eye_in_hand" in key_l) or ("wrist" in key_l):
            return True
    return False


def _collect_numeric_lowdim(obs: Dict[str, Any]) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for key, value in obs.items():
        if not _is_numeric_array(value):
            continue
        arr = np.asarray(value)
        if _is_likely_image_key(key, arr):
            continue
        # Filter obvious high-dim tensors while keeping vectors / small matrices.
        if arr.ndim >= 3:
            continue
        if arr.size > 512:
            continue
        out[key] = arr.astype(np.float32).reshape(-1)
    return out


def _concat_ordered(parts: List[np.ndarray]) -> Optional[np.ndarray]:
    if not parts:
        return None
    return np.concatenate(parts, axis=0).astype(np.float32)


def _build_state_candidates(obs: Dict[str, Any]) -> List[Tuple[str, np.ndarray]]:
    candidates: List[Tuple[str, np.ndarray]] = []

    # Candidate 1: explicit state already provided.
    for key in ("observation.state", "state"):
        if key in obs and _is_numeric_array(obs[key]):
            candidates.append((key, np.asarray(obs[key], dtype=np.float32).reshape(-1)))
            break

    # Candidate 2: historical fallback used earlier.
    joint_state = _build_joint_gripper_state(obs)
    if joint_state is not None:
        candidates.append(("joint_gripper", joint_state))

    lowdim = _collect_numeric_lowdim(obs)
    if lowdim:
        # Candidate 3: robot-prefixed terms plus object features.
        preferred_order = [
            "robot0_joint_pos",
            "robot0_gripper_qpos",
            "robot0_eef_pos",
            "robot0_eef_quat",
            "robot0_eef_vel_lin",
            "robot0_eef_vel_ang",
            "object",
            "object-state",
        ]
        used = set()
        parts_pref: List[np.ndarray] = []
        for key in preferred_order:
            if key in lowdim:
                parts_pref.append(lowdim[key])
                used.add(key)
        for key in sorted(lowdim.keys()):
            if key in used:
                continue
            if key.startswith("robot0_"):
                parts_pref.append(lowdim[key])
                used.add(key)
        # Keep object if present and not yet included.
        for key in ("object", "object-state"):
            if key in lowdim and key not in used:
                parts_pref.append(lowdim[key])
                used.add(key)
        pref = _concat_ordered(parts_pref)
        if pref is not None:
            candidates.append(("robot_object", pref))

        # Candidate 4: all low-dim numeric (deterministic sorted order).
        full = _concat_ordered([lowdim[k] for k in sorted(lowdim.keys())])
        if full is not None:
            candidates.append(("all_lowdim", full))

    # Deduplicate by shape and values identity in order.
    uniq: List[Tuple[str, np.ndarray]] = []
    seen = set()
    for name, arr in candidates:
        sig = (arr.size, arr.shape)
        if sig in seen:
            continue
        seen.add(sig)
        uniq.append((name, arr))
    return uniq


def _build_state_vector(obs: Dict[str, Any], state_dim: Optional[int]) -> Optional[np.ndarray]:
    candidates = _build_state_candidates(obs)
    if not candidates:
        return None
    if state_dim is None:
        return candidates[0][1]
    for _, arr in candidates:
        if arr.size == state_dim:
            return arr
    # No exact match: choose closest and let _adapt_state_dim handle padding / trim policy.
    return min(candidates, key=lambda x: abs(x[1].size - state_dim))[1]


def _summarize_state_candidates(obs: Dict[str, Any], state_dim: Optional[int]) -> str:
    candidates = _build_state_candidates(obs)
    if not candidates:
        return "state candidates: none"
    parts = [f"{name}:{arr.size}" for name, arr in candidates]
    msg = "state candidates: " + ", ".join(parts)
    if state_dim is not None:
        msg += f" | expected:{state_dim}"
    return msg


def _unwrap_env(env: Any) -> Any:
    current = env
    seen = set()
    while True:
        obj_id = id(current)
        if obj_id in seen:
            break
        seen.add(obj_id)
        nxt = None
        if hasattr(current, "base_env"):
            try:
                nxt = current.base_env
            except Exception:
                nxt = None
        if nxt is None and hasattr(current, "env"):
            try:
                nxt = current.env
            except Exception:
                nxt = None
        if nxt is None or nxt is current:
            break
        current = nxt
    return current


def _linspace_centers(min_val: float, max_val: float, step: float) -> List[float]:
    if step <= 0:
        return []
    start = min_val + step * 0.5
    end = max_val - step * 0.5
    if end + 1e-12 < start:
        return []
    vals = np.arange(start, end + 1e-12, step)
    return [float(v) for v in vals]


def _linspace_inclusive(min_val: float, max_val: float, step: float) -> List[float]:
    if step <= 0:
        return [float(min_val), float(max_val)]
    vals = list(np.arange(min_val, max_val + 1e-12, step))
    if not vals:
        vals = [min_val, max_val]
    elif vals[-1] < max_val - 1e-12:
        vals.append(max_val)
    return [float(v) for v in vals]


def _dedupe_points(points: List[Tuple[float, float]], ndigits: int = 8) -> List[Tuple[float, float]]:
    seen = set()
    out: List[Tuple[float, float]] = []
    for x_val, y_val in points:
        key = (round(float(x_val), ndigits), round(float(y_val), ndigits))
        if key in seen:
            continue
        seen.add(key)
        out.append((float(x_val), float(y_val)))
    return out


def _axis_angle_quat(axis: str, angle: float) -> np.ndarray:
    c = float(np.cos(angle / 2.0))
    s = float(np.sin(angle / 2.0))
    axis = str(axis).lower()
    if axis == "x":
        return np.array([c, s, 0.0, 0.0], dtype=np.float64)
    if axis == "y":
        return np.array([c, 0.0, s, 0.0], dtype=np.float64)
    return np.array([c, 0.0, 0.0, s], dtype=np.float64)


def _quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )


def _resolve_reference(env: Any, reference: Any) -> np.ndarray:
    if reference is None:
        return np.zeros(3, dtype=np.float64)
    if isinstance(reference, (list, tuple, np.ndarray)):
        arr = np.asarray(reference, dtype=np.float64).reshape(-1)
        if arr.size >= 3:
            return arr[:3]
        return np.zeros(3, dtype=np.float64)
    if isinstance(reference, str):
        if reference == "table_offset" and hasattr(env, "table_offset"):
            try:
                return np.asarray(env.table_offset, dtype=np.float64).reshape(-1)[:3]
            except Exception:
                pass
    return np.zeros(3, dtype=np.float64)


def _extract_case_objects(case: Dict[str, Any], default_object: str) -> Dict[str, Dict[str, Any]]:
    if "objects" in case and isinstance(case["objects"], dict):
        out = {}
        for k, v in case["objects"].items():
            if isinstance(v, dict):
                out[str(k)] = v
        if out:
            return out

    out = {}
    for k, v in case.items():
        if isinstance(v, dict):
            out[str(k)] = v
    if out:
        return out

    # Support flat case schema: {"x":..., "y":..., ...}
    flat_keys = {"x", "y", "z", "rot", "z_rot", "rotation", "angle", "quat", "rotation_axis"}
    if any(k in case for k in flat_keys):
        return {default_object: case}
    return {}


def _find_object_by_name(env: Any, object_name: str) -> Optional[Any]:
    key = object_name.lower()
    if not hasattr(env, "objects"):
        return None

    prefer = []
    fallback = []
    for obj in env.objects:
        name = str(getattr(obj, "name", ""))
        root = str(getattr(obj, "root_body", ""))
        text = (name + " " + root).lower()
        if key in text:
            if "visual" in text:
                fallback.append(obj)
            else:
                prefer.append(obj)
    if prefer:
        return prefer[0]
    if fallback:
        return fallback[0]
    return None


def _set_object_pose(env: Any, object_name: str, spec: Dict[str, Any], reference: Any) -> bool:
    target = _find_object_by_name(env, object_name)
    if target is None or not hasattr(env, "sim"):
        return False

    ref = _resolve_reference(env, reference)
    x_val = spec.get("x", None)
    y_val = spec.get("y", None)
    z_val = spec.get("z", None)
    if x_val is not None:
        x_val = float(x_val) + ref[0]
    if y_val is not None:
        y_val = float(y_val) + ref[1]
    if z_val is not None:
        z_val = float(z_val) + ref[2]

    quat = None
    if "quat" in spec:
        quat = np.asarray(spec["quat"], dtype=np.float64).reshape(-1)
        if quat.size != 4:
            quat = None
    else:
        angle = None
        for k in ("z_rot", "rot", "rotation", "angle"):
            if k in spec:
                angle = float(spec[k])
                break
        if angle is not None:
            axis = str(spec.get("rotation_axis", "z"))
            quat = _axis_angle_quat(axis, angle)

    if quat is not None and hasattr(target, "init_quat"):
        try:
            quat = _quat_multiply(quat, np.asarray(target.init_quat, dtype=np.float64))
        except Exception:
            pass

    try:
        if hasattr(target, "joints") and target.joints:
            joint_name = target.joints[0]
            qpos = np.array(env.sim.data.get_joint_qpos(joint_name), dtype=np.float64).copy()
            if qpos.size < 7:
                return False
            if x_val is not None:
                qpos[0] = x_val
            if y_val is not None:
                qpos[1] = y_val
            if z_val is not None:
                qpos[2] = z_val
            if quat is not None:
                qpos[3:7] = quat
            env.sim.data.set_joint_qpos(joint_name, qpos)
            env.sim.forward()
            return True

        root_body = getattr(target, "root_body", None)
        if root_body is None:
            return False
        body_id = env.sim.model.body_name2id(root_body)
        pos = np.array(env.sim.model.body_pos[body_id], dtype=np.float64).copy()
        if x_val is not None:
            pos[0] = x_val
        if y_val is not None:
            pos[1] = y_val
        if z_val is not None:
            pos[2] = z_val
        env.sim.model.body_pos[body_id] = pos
        if quat is not None:
            env.sim.model.body_quat[body_id] = quat
        env.sim.forward()
        return True
    except Exception:
        return False


def _apply_test_case(env: Any, case: Dict[str, Any], reference: Any, default_object: str) -> bool:
    raw_env = _unwrap_env(env)
    objects = _extract_case_objects(case, default_object=default_object)
    ok_any = False
    for obj_name, spec in objects.items():
        if not isinstance(spec, dict):
            continue
        ok_any = _set_object_pose(raw_env, obj_name, spec, reference) or ok_any
    return ok_any


def _build_cases_from_bounds(
    bounds: Dict[str, Any],
    step: float,
    include_boundary: bool,
    seed: Optional[int],
) -> List[Dict[str, Any]]:
    if not isinstance(bounds, dict) or not bounds:
        return []

    object_name = next(iter(bounds.keys()))
    spec = bounds[object_name]
    if not isinstance(spec, dict):
        return []
    if "x" not in spec or "y" not in spec:
        return []

    x_min, x_max = float(spec["x"][0]), float(spec["x"][1])
    y_min, y_max = float(spec["y"][0]), float(spec["y"][1])
    rot_axis = str(spec.get("rotation_axis", "z"))
    rng = np.random.default_rng(seed if seed is not None else 0)

    def sample_rot() -> Optional[float]:
        if "z_rot" in spec and isinstance(spec["z_rot"], (list, tuple)) and len(spec["z_rot"]) >= 2:
            lo, hi = float(spec["z_rot"][0]), float(spec["z_rot"][1])
            if abs(hi - lo) < 1e-12:
                return lo
            return float(rng.uniform(lo, hi))
        for k in ("rot", "rotation", "angle"):
            if k in spec:
                try:
                    return float(spec[k])
                except Exception:
                    return None
        return None

    centers_x = _linspace_centers(x_min, x_max, step)
    centers_y = _linspace_centers(y_min, y_max, step)
    grid_points = [(xv, yv) for xv in centers_x for yv in centers_y]

    boundary_points: List[Tuple[float, float]] = []
    if include_boundary:
        edge_x = _linspace_inclusive(x_min, x_max, step)
        edge_y = _linspace_inclusive(y_min, y_max, step)
        for xv in edge_x:
            boundary_points.append((xv, y_min))
            boundary_points.append((xv, y_max))
        for yv in edge_y:
            boundary_points.append((x_min, yv))
            boundary_points.append((x_max, yv))
        boundary_points = _dedupe_points(boundary_points)

    cases: List[Dict[str, Any]] = []
    next_id = 0
    for tag, points in (("grid", _dedupe_points(grid_points)), ("boundary", boundary_points)):
        if tag == "boundary" and not include_boundary:
            continue
        for x_val, y_val in points:
            obj = {"x": float(x_val), "y": float(y_val)}
            rot = sample_rot()
            if rot is not None:
                obj["z_rot"] = float(rot)
                obj["rotation_axis"] = rot_axis
            case = {"id": next_id, "tag": tag, object_name: obj}
            cases.append(case)
            next_id += 1
    return cases


def _load_test_positions(path: str) -> Tuple[List[Dict[str, Any]], Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data, None
    if not isinstance(data, dict):
        return [], None

    reference = data.get("reference", None)
    cases = data.get("cases") or data.get("test_cases") or data.get("positions") or []
    if isinstance(cases, list) and len(cases) > 0:
        return cases, reference

    # Fallback: generate from bounds section.
    bounds = data.get("bounds", {})
    step = float(data.get("step", 0.0) or 0.0)
    include_boundary = bool(data.get("include_boundary", True))
    seed = data.get("seed", None)
    if step > 0 and isinstance(bounds, dict) and bounds:
        cases = _build_cases_from_bounds(bounds, step=step, include_boundary=include_boundary, seed=seed)
    return cases, reference


def _resolve_test_positions_path(args: argparse.Namespace, env_name: str) -> Optional[str]:
    if args.test_positions_path:
        p = Path(args.test_positions_path).expanduser()
        return str(p) if p.is_file() else None
    if not args.test_positions_dir:
        return None

    test_dir = Path(args.test_positions_dir).expanduser()
    if not test_dir.is_dir():
        return None

    json_files = sorted(test_dir.glob("*.json"))
    if not json_files:
        return None

    # Build candidate names from canonical env, user input alias, and known task aliases.
    candidate_names: List[str] = []

    def _add_candidate(name: Any) -> None:
        if name is None:
            return
        s = str(name).strip()
        if s and s not in candidate_names:
            candidate_names.append(s)

    _add_candidate(env_name)
    _add_candidate(getattr(args, "env_name", None))

    env_l = str(env_name).lower()
    arg_l = str(getattr(args, "env_name", "")).lower()
    for alias, names in TASK_NAME_CANDIDATES.items():
        names_l = [str(n).lower() for n in names]
        if env_l == alias or env_l in names_l or arg_l == alias or arg_l in names_l:
            _add_candidate(alias)
            for n in names:
                _add_candidate(n)

    # 1) Exact stem match (case-insensitive), e.g. PickPlaceCan -> PickPlaceCan.json
    lower_stem_to_path: Dict[str, str] = {}
    for p in json_files:
        lower_stem_to_path.setdefault(p.stem.lower(), str(p))
    for name in candidate_names:
        hit = lower_stem_to_path.get(str(name).lower())
        if hit is not None:
            return hit

    # 2) Normalized match: remove separators / symbols, e.g. nut_assembly_square ~= NutAssemblySquare
    def _norm(s: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", str(s).lower())

    norm_stem_to_path: Dict[str, str] = {}
    for p in json_files:
        norm_stem_to_path.setdefault(_norm(p.stem), str(p))
    for name in candidate_names:
        hit = norm_stem_to_path.get(_norm(name))
        if hit is not None:
            return hit

    # 3) Prefix match, e.g. square -> Square_D1.json
    lowered_candidates = [str(name).lower() for name in candidate_names]
    for cand in lowered_candidates:
        for p in json_files:
            stem_l = p.stem.lower()
            if stem_l.startswith(cand + "_") or stem_l.startswith(cand + "-") or stem_l.startswith(cand):
                return str(p)

    return None


def _infer_bounds_for_object_from_env(env: Any, object_name: str) -> Tuple[Optional[Dict[str, Any]], Any]:
    raw = _unwrap_env(env)
    sampler = getattr(raw, "placement_initializer", None)
    if sampler is None or not hasattr(sampler, "samplers"):
        return None, None

    target = object_name.lower()
    for s in sampler.samplers.values():
        if not hasattr(s, "x_range") or not hasattr(s, "y_range"):
            continue
        objs = getattr(s, "mujoco_objects", []) or []
        has_obj = False
        for obj in objs:
            name = str(getattr(obj, "name", "")).lower()
            if target in name and "visual" not in name:
                has_obj = True
                break
        if not has_obj and objs:
            continue

        bounds = {
            object_name: {
                "x": [float(s.x_range[0]), float(s.x_range[1])],
                "y": [float(s.y_range[0]), float(s.y_range[1])],
                "rotation_axis": str(getattr(s, "rotation_axis", "z")),
            }
        }
        reference = getattr(s, "reference_pos", None)
        return bounds, reference
    return None, None


def _prepare_test_cases(
    args: argparse.Namespace,
    env: Any,
    canonical_env_name: str,
) -> Tuple[List[Dict[str, Any]], Any, Optional[str]]:
    test_init_mode = str(getattr(args, "test_init_mode", "auto")).lower()
    if test_init_mode == "random":
        if args.debug_test_case:
            print("test_init_mode=random: skipping test json / generated cases.")
        return [], None, None

    path = _resolve_test_positions_path(args, canonical_env_name)
    if path is not None:
        cases, reference = _load_test_positions(path)
        if args.test_use_all_cases and cases:
            args.n_rollouts = len(cases) + _get_extra_random_rollouts(args)
        if args.debug_test_case:
            print(f"loaded test positions: {path} (cases={len(cases)})")
        return cases, reference, path

    if test_init_mode == "json":
        raise FileNotFoundError(
            "test_init_mode=json but no test position json was found. "
            "Check --test_positions_path / --test_positions_dir / --env_name."
        )

    # If no json specified, optionally generate from env bounds.
    if args.test_grid_step is None or args.test_grid_step <= 0:
        return [], None, None
    bounds, reference = _infer_bounds_for_object_from_env(env, args.test_case_object)
    if not bounds:
        return [], None, None
    cases = _build_cases_from_bounds(
        bounds=bounds,
        step=float(args.test_grid_step),
        include_boundary=bool(args.test_grid_include_boundary),
        seed=args.seed,
    )
    if args.test_use_all_cases and cases:
        args.n_rollouts = len(cases) + _get_extra_random_rollouts(args)
    if args.debug_test_case:
        print(f"generated test positions from env bounds (cases={len(cases)})")
    return cases, reference, None


def _get_extra_random_rollouts(args: argparse.Namespace) -> int:
    if str(getattr(args, "test_init_mode", "auto")).lower() != "mixed":
        return 0
    return max(int(getattr(args, "test_random_episodes", 0) or 0), 0)


def _format_progress_line(completed: int, total: int, success_sum: float, bar_width: int = 30) -> str:
    if total <= 0:
        ratio = 1.0
    else:
        ratio = min(max(completed / float(total), 0.0), 1.0)
    filled = int(round(bar_width * ratio))
    bar = "#" * filled + "-" * (bar_width - filled)
    success_rate = (success_sum / float(completed) * 100.0) if completed > 0 else 0.0
    return f"[{bar}] {completed}/{total} ({ratio * 100.0:5.1f}%) | success {success_rate:5.1f}%"


def _adapt_state_dim(state: np.ndarray, state_dim: Optional[int], strict: bool) -> np.ndarray:
    if state_dim is None:
        return state
    if state.size == state_dim:
        return state
    if strict:
        raise ValueError(f"State dim mismatch: got {state.size}, expected {state_dim}")
    # Pad / trim to tolerate slight obs schema differences.
    if state.size > state_dim:
        return state[:state_dim]
    padded = np.zeros((state_dim,), dtype=np.float32)
    padded[: state.size] = state
    return padded


def _render_frame(env: Any, camera_name: str, width: int, height: int) -> Optional[np.ndarray]:
    frame = None
    try:
        frame = env.render(mode="rgb_array", camera_name=camera_name, width=width, height=height)
    except Exception:
        pass
    sim = None
    if hasattr(env, "sim"):
        sim = env.sim
    elif hasattr(env, "base_env") and hasattr(env.base_env, "sim"):
        sim = env.base_env.sim
    elif hasattr(env, "env") and hasattr(env.env, "sim"):
        sim = env.env.sim
    if frame is None and sim is not None:
        try:
            frame = sim.render(camera_name=camera_name, width=width, height=height)
            frame = np.flipud(frame)  # mujoco offscreen frame
        except Exception:
            return None
    return frame if isinstance(frame, np.ndarray) else None


def _is_success(env: Any, info: Dict[str, Any]) -> bool:
    def _as_success_flag(value: Any) -> Optional[bool]:
        if isinstance(value, dict):
            for key in ("task", "success", "is_success"):
                if key in value:
                    return _as_success_flag(value[key])
            return None
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        if isinstance(value, (int, float, np.integer, np.floating)):
            return bool(value)
        return None

    if hasattr(env, "_check_success"):
        try:
            flag = _as_success_flag(env._check_success())
            if flag is not None:
                return flag
        except Exception:
            pass
    if isinstance(info, dict):
        for key in ("is_success", "success"):
            if key in info:
                flag = _as_success_flag(info[key])
                if flag is not None:
                    return flag
    if hasattr(env, "is_success"):
        try:
            result = env.is_success()
            flag = _as_success_flag(result)
            if flag is not None:
                return flag
        except Exception:
            pass
    return False


def _resolve_bool_mode(mode: str, default: bool) -> bool:
    m = str(mode).lower()
    if m == "true":
        return True
    if m == "false":
        return False
    return bool(default)


def _should_require_release(args: argparse.Namespace) -> bool:
    env_name = str(getattr(args, "canonical_env_name", "")).lower()
    default = env_name in {"pickplacecan", "can", "pickplace"}
    return _resolve_bool_mode(args.require_release, default=default)


def _should_require_low_height(args: argparse.Namespace) -> bool:
    env_name = str(getattr(args, "canonical_env_name", "")).lower()
    default = env_name in {"pickplacecan", "can", "pickplace"}
    return _resolve_bool_mode(args.require_low_height, default=default)


def _get_eval_target_object(env: Any, default_object_name: str = "can") -> Optional[Any]:
    raw = _unwrap_env(env)
    try:
        if not hasattr(raw, "objects") or not raw.objects:
            return None

        target_obj = None
        obj_to_use = getattr(raw, "obj_to_use", None)
        if obj_to_use is not None:
            target_name = str(obj_to_use).lower()
            for obj in raw.objects:
                name = str(getattr(obj, "name", "")).lower()
                if target_name == name or target_name in name:
                    target_obj = obj
                    break
        if target_obj is None and hasattr(raw, "object_id"):
            try:
                idx = int(raw.object_id)
                if 0 <= idx < len(raw.objects):
                    target_obj = raw.objects[idx]
            except Exception:
                pass
        if target_obj is None and len(raw.objects) == 1:
            target_obj = raw.objects[0]
        if target_obj is None:
            key = str(default_object_name).lower()
            for obj in raw.objects:
                name = str(getattr(obj, "name", "")).lower()
                if key in name and "visual" not in name:
                    target_obj = obj
                    break
        return target_obj
    except Exception:
        return None


def _is_object_released(env: Any, default_object_name: str = "can") -> bool:
    raw = _unwrap_env(env)
    try:
        if not hasattr(raw, "_check_grasp") or not hasattr(raw, "robots"):
            return True
        target_obj = _get_eval_target_object(env, default_object_name=default_object_name)
        if target_obj is None:
            return True

        geoms = list(getattr(target_obj, "contact_geoms", []) or [])
        if not geoms:
            return True
        gripper = raw.robots[0].gripper
        grasping = bool(raw._check_grasp(gripper=gripper, object_geoms=geoms))
        return not grasping
    except Exception:
        return True


def _get_eval_object_z(env: Any, default_object_name: str = "can") -> Optional[float]:
    raw = _unwrap_env(env)
    target_obj = _get_eval_target_object(env, default_object_name=default_object_name)
    if target_obj is None or not hasattr(raw, "sim"):
        return None
    try:
        if hasattr(raw, "obj_body_id"):
            obj_name = getattr(target_obj, "name", None)
            if obj_name in raw.obj_body_id:
                body_id = raw.obj_body_id[obj_name]
                return float(raw.sim.data.body_xpos[body_id][2])
        root_body = getattr(target_obj, "root_body", None)
        if root_body is None:
            return None
        body_id = raw.sim.model.body_name2id(root_body)
        return float(raw.sim.data.body_xpos[body_id][2])
    except Exception:
        return None


def _get_success_height_threshold(env: Any, args: argparse.Namespace) -> Optional[float]:
    if args.success_max_obj_height is not None:
        return float(args.success_max_obj_height)
    raw = _unwrap_env(env)
    try:
        if hasattr(raw, "bin2_pos"):
            return float(np.asarray(raw.bin2_pos).reshape(-1)[2]) + float(args.success_max_height_offset)
    except Exception:
        pass
    return None


def _extract_action(response: Dict[str, Any], use_action_chunk: bool) -> np.ndarray:
    raw = response.get("action", response)
    arr = np.asarray(raw, dtype=np.float32)
    if arr.ndim == 1:
        return arr.reshape(1, -1) if use_action_chunk else arr
    if arr.ndim >= 2:
        return arr
    raise ValueError(f"Unsupported action shape from server: {arr.shape}")


def _infer_robot_arms(robot: str) -> List[str]:
    try:
        from robosuite.models.robots.robot_model import REGISTERED_ROBOTS

        if robot in REGISTERED_ROBOTS:
            arms = list(REGISTERED_ROBOTS[robot].arms)
            if arms:
                return arms
    except Exception:
        pass
    # Conservative fallback that also works for bimanual robots.
    return ["right", "left"]


def _load_controller_config_compat(controller_name: str, robots: str | List[str]) -> Dict[str, Any]:
    if _load_controller_config_legacy is not None:
        return _load_controller_config_legacy(default_controller=controller_name)

    part_cfg = load_part_controller_config(default_controller=controller_name)
    robot = robots[0] if isinstance(robots, list) else robots
    arms = _infer_robot_arms(robot)

    if refactor_composite_controller_config is not None:
        return refactor_composite_controller_config(part_cfg, robot, arms)

    # Minimal fallback if refactor helper is unavailable.
    composite_cfg: Dict[str, Any] = {"type": "BASIC", "body_parts": {}}
    for arm in arms:
        arm_cfg = copy.deepcopy(part_cfg)
        arm_cfg.setdefault("gripper", {"type": "GRIP"})
        composite_cfg["body_parts"][arm] = arm_cfg
    return composite_cfg


def _override_joint_control_delta(controller_cfg: Dict[str, Any], control_delta: bool) -> None:
    if "body_parts" in controller_cfg and isinstance(controller_cfg["body_parts"], dict):
        for part_cfg in controller_cfg["body_parts"].values():
            if isinstance(part_cfg, dict):
                part_cfg["control_delta"] = bool(control_delta)
        return
    controller_cfg["control_delta"] = bool(control_delta)


def _override_joint_input_type(controller_cfg: Dict[str, Any], input_type: str) -> None:
    if input_type not in {"delta", "absolute"}:
        raise ValueError(f"Unsupported joint input_type '{input_type}'")
    if "body_parts" in controller_cfg and isinstance(controller_cfg["body_parts"], dict):
        for part_cfg in controller_cfg["body_parts"].values():
            if isinstance(part_cfg, dict) and str(part_cfg.get("type", "")).upper() == "JOINT_POSITION":
                part_cfg["input_type"] = input_type
        return
    if str(controller_cfg.get("type", "")).upper() == "JOINT_POSITION":
        controller_cfg["input_type"] = input_type


def _initialize_obs_utils(camera_names: List[str]) -> None:
    # EnvRobosuite expects ObsUtils mappings to be initialized before reset()/step().
    if ObsUtils.OBS_KEYS_TO_MODALITIES is not None:
        return

    rgb_keys: List[str] = []
    for cam in camera_names:
        rgb_keys.append(cam)
        rgb_keys.append(f"{cam}_image")
    # Preserve order and dedupe.
    seen = set()
    rgb_keys = [k for k in rgb_keys if not (k in seen or seen.add(k))]

    ObsUtils.initialize_obs_utils_with_obs_specs(
        {
            "obs": {
                "low_dim": [],
                "rgb": rgb_keys,
            }
        }
    )


def _make_env(args: argparse.Namespace, camera_names: List[str]) -> Any:
    _initialize_obs_utils(camera_names)
    env_name = _parse_task_name(args.env_name)
    robots = _parse_robots(args.robots, env_name)

    controller_cfg = _load_controller_config_compat(controller_name=args.controller, robots=robots)
    if str(args.controller).upper().startswith("JOINT"):
        # Legacy compatibility: map deprecated control-delta flag to input_type semantics.
        if args.joint_control_delta.lower() == "true":
            args.joint_input_type = "delta"
        elif args.joint_control_delta.lower() == "false":
            args.joint_input_type = "absolute"

        if str(args.controller).upper() == "JOINT_POSITION":
            if args.joint_input_type in {"delta", "absolute"}:
                _override_joint_input_type(controller_cfg, args.joint_input_type)
        else:
            # For non-position JOINT controllers, keep backward control_delta override path.
            if args.joint_control_delta.lower() == "true":
                _override_joint_control_delta(controller_cfg, True)
            elif args.joint_control_delta.lower() == "false":
                _override_joint_control_delta(controller_cfg, False)

    env = EnvUtils.create_env(
        env_type=EnvType.ROBOSUITE_TYPE,
        env_name=env_name,
        render=args.render,
        render_offscreen=True,
        use_image_obs=True,
        use_depth_obs=False,
        robots=robots,
        controller_configs=controller_cfg,
        ignore_done=args.ignore_done,
        use_object_obs=True,
        camera_names=camera_names,
        camera_heights=args.camera_height,
        camera_widths=args.camera_width,
        control_freq=args.control_freq,
        horizon=args.horizon,
        reward_shaping=args.reward_shaping,
    )
    return env


def _get_action_spec(env: Any) -> Tuple[np.ndarray, np.ndarray]:
    if hasattr(env, "action_spec"):
        try:
            low, high = env.action_spec
            return np.asarray(low), np.asarray(high)
        except Exception:
            pass
    if hasattr(env, "base_env") and hasattr(env.base_env, "action_spec"):
        low, high = env.base_env.action_spec
        return np.asarray(low), np.asarray(high)
    if hasattr(env, "env") and hasattr(env.env, "action_spec"):
        low, high = env.env.action_spec
        return np.asarray(low), np.asarray(high)
    raise RuntimeError("Could not read action_spec from environment.")


def _save_episode_rollout(
    args: argparse.Namespace,
    episode_idx: int,
    success: bool,
    rows: List[Dict[str, Any]],
    policy_calls: Optional[List[Dict[str, Any]]] = None,
) -> None:
    if args.save_rollouts_dir is None:
        return
    save_dir = Path(args.save_rollouts_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    suffix = "success" if success else "failure"
    out_path = save_dir / f"episode_{episode_idx:06d}_{suffix}.pkl"

    for row in rows:
        row["success"] = success

    if args.save_full_policy_io:
        obj = {
            "schema_version": "robomimic_eval_rollout_v2",
            "episode": int(episode_idx),
            "success": bool(success),
            "rows": rows,
            "policy_calls": policy_calls or [],
        }
    elif pd is not None:
        obj = pd.DataFrame(rows)
    else:
        obj = rows

    with open(out_path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def _rollout_episode(
    env: Any,
    args: argparse.Namespace,
    required_image_keys: List[str],
    expect_state: bool,
    state_dim: Optional[int],
    episode_idx: int,
    video_writer: Any,
    test_cases: Optional[List[Dict[str, Any]]] = None,
    test_reference: Any = None,
) -> Dict[str, Any]:
    # Clear ACT internal action queue between episodes.
    _http_json("POST", _reset_url(args.server_url), payload={}, timeout_s=args.timeout_s)

    obs = env.reset()
    # Follow robomimic run_trained_agent.py for deterministic rollout start.
    try:
        state_dict = env.get_state()
        obs_reset = env.reset_to(state_dict)
        if obs_reset is not None:
            obs = obs_reset
    except Exception:
        pass

    applied_case_id = None
    applied_case_tag = None
    applied_case_ok = False
    extra_random_rollouts = _get_extra_random_rollouts(args)
    use_random_after_cases = bool(test_cases) and extra_random_rollouts > 0
    if test_cases:
        case_index: Optional[int]
        if use_random_after_cases and episode_idx >= len(test_cases):
            case_index = None
        else:
            case_index = episode_idx % len(test_cases)

        case = test_cases[case_index] if case_index is not None else None
        if isinstance(case, dict):
            applied_case_id = case.get("id", case_index)
            applied_case_tag = case.get("tag", None)
            applied_case_ok = _apply_test_case(
                env=env,
                case=case,
                reference=test_reference,
                default_object=args.test_case_object,
            )
            if args.debug_test_case:
                print(
                    f"episode={episode_idx} test_case_id={applied_case_id} "
                    f"tag={applied_case_tag} applied={applied_case_ok}"
                )
            if applied_case_ok:
                try:
                    obs = env.get_observation()
                except Exception:
                    pass
        elif args.debug_test_case and use_random_after_cases:
            print(f"episode={episode_idx} using environment random reset after exhausting test json cases")

    action_cache: Optional[np.ndarray] = None
    policy_calls: List[Dict[str, Any]] = []
    active_policy_call_id: Optional[int] = None
    next_chunk_index = 0
    total_reward = 0.0
    success = False
    first_success_step: Optional[int] = None
    rows: List[Dict[str, Any]] = []
    success_streak = 0
    require_release = _should_require_release(args)
    require_low_height = _should_require_low_height(args)
    success_height_threshold = getattr(args, "success_height_threshold", None)
    if require_low_height and success_height_threshold is None:
        success_height_threshold = _get_success_height_threshold(env, args)
    if require_low_height and success_height_threshold is None:
        # Fallback: if threshold cannot be inferred, do not hard-fail every rollout.
        require_low_height = False

    action_low, action_high = _get_action_spec(env)

    for step_i in range(args.horizon):
        policy_source = "server"
        policy_call_id: Optional[int] = None
        policy_chunk_index: Optional[int] = None

        if action_cache is not None and action_cache.shape[0] > 0:
            action = action_cache[0]
            action_cache = action_cache[1:] if action_cache.shape[0] > 1 else None
            policy_source = "cache"
            policy_call_id = active_policy_call_id
            policy_chunk_index = int(next_chunk_index)
            next_chunk_index += 1
        else:
            payload = {
                "episode": episode_idx,
                "step": step_i,
                "images": _extract_images_for_policy(obs, required_image_keys),
            }
            if hasattr(args, "prompt") and args.prompt is not None:
                payload["prompt"] = str(args.prompt)
            if expect_state:
                state = _build_state_vector(obs, state_dim=state_dim)
                if state is None:
                    raise ValueError("Policy expects state, but no state could be built from env observation.")
                state = _adapt_state_dim(state, state_dim, args.strict_state_dim)
                payload["state"] = state.tolist()
                if args.debug_state and step_i == 0:
                    print(_summarize_state_candidates(obs, state_dim=state_dim))
                    print(f"selected state dim: {state.size}")
            if not args.use_action_chunk:
                payload["chunk_size"] = 1
            elif args.chunk_size is not None:
                payload["chunk_size"] = int(args.chunk_size)

            response = _http_json("POST", args.server_url, payload=payload, timeout_s=args.timeout_s)
            action_chunk = _extract_action(response, args.use_action_chunk)
            if args.debug_action and step_i == 0:
                flat = action_chunk.reshape(-1)
                print(
                    "action stats: shape={} min={:.4f} max={:.4f} mean={:.4f} std={:.4f}".format(
                        list(action_chunk.shape),
                        float(np.min(flat)),
                        float(np.max(flat)),
                        float(np.mean(flat)),
                        float(np.std(flat)),
                    )
                )
            if action_chunk.ndim == 1:
                action_chunk = action_chunk.reshape(1, -1)

            action_chunk = np.asarray(action_chunk, dtype=np.float32)
            active_policy_call_id = len(policy_calls)
            policy_calls.append(
                {
                    "call_id": int(active_policy_call_id),
                    "episode": int(episode_idx),
                    "query_step": int(step_i),
                    "request": payload,
                    "response": response,
                    "action_chunk_raw": action_chunk,
                }
            )
            policy_call_id = int(active_policy_call_id)
            policy_chunk_index = 0
            next_chunk_index = 1

            action = action_chunk[0]
            action_cache = action_chunk[1:] if action_chunk.shape[0] > 1 else None

        action_raw = np.asarray(action, dtype=np.float32).copy()
        action = action * args.action_scale + args.action_bias
        if args.clip_action:
            action = np.clip(action, action_low, action_high)

        next_obs, reward, done, info = env.step(action)
        total_reward += float(reward)

        task_success = _is_success(env, info if isinstance(info, dict) else {})
        released = _is_object_released(env, default_object_name=args.test_case_object) if require_release else True
        obj_z = _get_eval_object_z(env, default_object_name=args.test_case_object)
        low_height = True
        if require_low_height:
            if (obj_z is None) or (success_height_threshold is None):
                low_height = False
            else:
                low_height = bool(obj_z <= float(success_height_threshold))
        step_success = bool(task_success and released and low_height)
        if step_success:
            success_streak += 1
        else:
            success_streak = 0
        if (not success) and success_streak >= args.success_hold_steps:
            success = True
            if first_success_step is None:
                first_success_step = step_i + 1

        row_state = _build_state_vector(obs, state_dim=state_dim)
        row: Dict[str, Any] = {
            "obs": {
                "state": row_state.astype(np.float32) if row_state is not None else None
            },
            "policy_source": policy_source,
            "policy_call_id": int(policy_call_id) if policy_call_id is not None else None,
            "policy_chunk_index": int(policy_chunk_index) if policy_chunk_index is not None else None,
            "policy_input": (
                policy_calls[policy_call_id]["request"] if policy_call_id is not None else None
            ),
            "policy_output": (
                policy_calls[policy_call_id]["response"] if policy_call_id is not None else None
            ),
            "policy_action_chunk_raw": (
                policy_calls[policy_call_id]["action_chunk_raw"] if policy_call_id is not None else None
            ),
            "action_raw": action_raw,
            "action": np.asarray(action, dtype=np.float32),
            "reward": float(reward),
            "done": bool(done),
            "success": False,
            "task_success_raw": bool(task_success),
            "released_raw": bool(released),
            "object_z_raw": float(obj_z) if obj_z is not None else None,
            "low_height_raw": bool(low_height),
            "success_height_threshold": float(success_height_threshold) if success_height_threshold is not None else None,
            "success_qualified": bool(step_success),
        }
        if args.save_images_in_pkl:
            try:
                row["obs"]["agentview"] = np.asarray(obs.get("agentview_image", obs.get("agentview")))
                row["obs"]["robot0_eye_in_hand"] = np.asarray(
                    obs.get("robot0_eye_in_hand_image", obs.get("robot0_eye_in_hand"))
                )
            except Exception:
                pass
        rows.append(row)

        if video_writer is not None and (step_i % args.video_skip == 0):
            frames: List[np.ndarray] = []
            for cam_name in args.camera_names:
                frame = _render_frame(env, cam_name, args.video_width, args.video_height)
                if frame is not None:
                    frames.append(frame)
            if frames:
                frame = np.concatenate(frames, axis=1) if len(frames) > 1 else frames[0]
                video_writer.append_data(np.ascontiguousarray(frame))

        obs = next_obs
        if success:
            break
        if done and not args.ignore_done:
            break

    _save_episode_rollout(args, episode_idx, success, rows, policy_calls=policy_calls)

    horizon = step_i + 1
    return {
        "episode": episode_idx,
        "return": total_reward,
        "horizon": horizon,
        "success": float(success),
        "success_first_step": first_success_step,
        "test_case_id": applied_case_id,
        "test_case_tag": applied_case_tag,
        "test_case_applied": bool(applied_case_ok),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate LeRobot ACT server on robomimic/robosuite environments.")
    parser.add_argument("--server_url", type=str, default="http://127.0.0.1:8000/act")
    parser.add_argument("--port", type=int, default=None, help="Optional server port override, e.g. 8001.")
    parser.add_argument("--timeout_s", type=float, default=10.0)

    parser.add_argument(
        "--env_name",
        "--env",
        dest="env_name",
        type=str,
        default="can",
        help="robomimic task: lift/can/square/transport/tool_hang",
    )
    parser.add_argument("--robots", type=str, default="Panda", help="Comma-separated robots, e.g. Panda or Panda,Panda")
    parser.add_argument("--controller", type=str, default="JOINT_POSITION")
    parser.add_argument("--control_freq", type=int, default=20)
    parser.add_argument("--horizon", type=int, default=400)
    parser.add_argument("--ignore_done", action="store_true")
    parser.add_argument("--reward_shaping", action="store_true")

    parser.add_argument("--n_rollouts", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--camera_names", type=str, default="agentview,robot0_eye_in_hand")
    parser.add_argument("--camera_height", type=int, default=84)
    parser.add_argument("--camera_width", type=int, default=84)
    parser.add_argument("--render", action="store_true")

    parser.add_argument("--use_action_chunk", action="store_true", default=True)
    parser.add_argument("--no_action_chunk", dest="use_action_chunk", action="store_false")
    parser.add_argument("--chunk_size", type=int, default=None)
    parser.add_argument("--action_scale", type=float, default=1.0)
    parser.add_argument("--action_bias", type=float, default=0.0)
    parser.add_argument("--clip_action", action="store_true", default=False)
    parser.add_argument("--no_clip_action", dest="clip_action", action="store_false")
    parser.add_argument(
        "--joint_input_type",
        type=str,
        default="auto",
        choices=["auto", "delta", "absolute"],
        help="For JOINT_POSITION: keep config as-is (auto), or force arm input_type delta/absolute.",
    )
    parser.add_argument(
        "--joint_control_delta",
        type=str,
        default="auto",
        choices=["auto", "true", "false"],
        help="Deprecated compatibility flag. For JOINT_POSITION it maps to input_type (true->delta, false->absolute).",
    )
    parser.add_argument("--strict_state_dim", action="store_true", default=False)
    parser.add_argument("--debug_state", action="store_true", default=False)
    parser.add_argument("--debug_action", action="store_true", default=False)
    parser.add_argument("--debug_test_case", action="store_true", default=False)
    parser.add_argument(
        "--success_hold_steps",
        type=int,
        default=3,
        help="Qualified success must hold for this many consecutive env steps.",
    )
    parser.add_argument(
        "--require_release",
        type=str,
        default="auto",
        choices=["auto", "true", "false"],
        help="Whether success additionally requires object to be released from gripper.",
    )
    parser.add_argument(
        "--require_low_height",
        type=str,
        default="auto",
        choices=["auto", "true", "false"],
        help="Whether success additionally requires target object z-height under a threshold.",
    )
    parser.add_argument(
        "--success_max_obj_height",
        type=float,
        default=None,
        help="Absolute world-z threshold for success (if require_low_height enabled).",
    )
    parser.add_argument(
        "--success_max_height_offset",
        type=float,
        default=0.08,
        help="If absolute threshold is unset, use bin2_pos.z + this offset as threshold.",
    )

    parser.add_argument("--test_positions_path", type=str, default=None, help="Optional JSON path with test cases.")
    parser.add_argument(
        "--test_positions_dir",
        type=str,
        default="/workspace/liushuo/cupid/mimicgen/eval/test",
        help="Optional directory to auto-resolve test json by env name.",
    )
    parser.add_argument(
        "--test_init_mode",
        type=str,
        default="auto",
        choices=["auto", "json", "random", "mixed"],
        help=(
            "How to choose rollout initial states: "
            "auto=prefer json then fallback, "
            "json=use test json only, "
            "random=always use env random reset, "
            "mixed=run json/generated test cases first, then append random reset episodes."
        ),
    )
    parser.add_argument(
        "--test_random_episodes",
        type=int,
        default=0,
        help="When --test_init_mode=mixed, append this many extra random-reset episodes after test json/generated cases.",
    )
    parser.add_argument(
        "--test_case_object",
        type=str,
        default="can",
        help="Default object name for flat test case schema or auto-generated bounds.",
    )
    parser.add_argument(
        "--test_grid_step",
        type=float,
        default=None,
        help="If no JSON is found, auto-generate test cases from env bounds using this grid step.",
    )
    parser.add_argument("--test_grid_include_boundary", action="store_true", default=True)
    parser.add_argument("--no_test_grid_include_boundary", dest="test_grid_include_boundary", action="store_false")
    parser.add_argument(
        "--test_use_all_cases",
        action="store_true",
        default=True,
        help="If test cases are available, override n_rollouts to len(test_cases).",
    )
    parser.add_argument(
        "--no_test_use_all_cases",
        dest="test_use_all_cases",
        action="store_false",
        help="Do not override n_rollouts even if test cases are available.",
    )

    parser.add_argument("--video_dir", type=str, default=None)
    parser.add_argument("--video_skip", type=int, default=5)
    parser.add_argument("--video_width", type=int, default=512)
    parser.add_argument("--video_height", type=int, default=512)
    parser.add_argument("--video_macro_block_size", type=int, default=1)

    parser.add_argument("--save_rollouts_dir", type=str, default=None, help="Optional directory to save per-episode .pkl")
    parser.add_argument("--save_images_in_pkl", action="store_true", default=False)
    parser.add_argument(
        "--save_full_policy_io",
        action="store_true",
        default=True,
        help="When saving rollouts to pkl, include full server request / response payloads.",
    )
    parser.add_argument(
        "--no_save_full_policy_io",
        dest="save_full_policy_io",
        action="store_false",
        help="Disable full server request / response payload storage.",
    )
    parser.add_argument("--summary_path", type=str, default=None, help="Optional JSON path to save aggregate metrics")
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    args.server_url = _override_server_port(args.server_url, args.port)

    random.seed(args.seed)
    np.random.seed(args.seed)

    args.camera_names = [x.strip() for x in args.camera_names.split(",") if x.strip()]
    if not args.camera_names:
        args.camera_names = ["agentview", "robot0_eye_in_hand"]

    health = _http_json("GET", _health_url(args.server_url), payload=None, timeout_s=args.timeout_s)
    required_image_keys = list(health.get("image_keys", []))
    expect_state = bool(health.get("expect_state", False))
    state_dim = health.get("state_dim")
    state_dim = int(state_dim) if state_dim is not None else None

    canonical_env_name = _parse_task_name(args.env_name)
    args.canonical_env_name = canonical_env_name
    env = _make_env(args, args.camera_names)
    if hasattr(env, "seed"):
        try:
            env.seed(args.seed)
        except Exception:
            pass

    test_cases, test_reference, test_positions_path = _prepare_test_cases(
        args,
        env=env,
        canonical_env_name=canonical_env_name,
    )
    extra_random_rollouts = _get_extra_random_rollouts(args)
    if test_cases:
        source = test_positions_path if test_positions_path is not None else "generated-from-env-bounds"
        if extra_random_rollouts > 0:
            print(
                "Using {} test initial positions from: {} (reference={}), plus {} random-reset episodes.".format(
                    len(test_cases),
                    source,
                    test_reference,
                    extra_random_rollouts,
                )
            )
        else:
            print(f"Using {len(test_cases)} test initial positions from: {source} (reference={test_reference})")
    elif args.test_init_mode == "random":
        print("Using environment default random reset (--test_init_mode=random).")
    elif args.test_init_mode == "mixed":
        print("No test position json found for env; mixed mode fell back to environment default random reset.")
    else:
        print("No test position json found for env; using environment default random reset.")

    require_release = _should_require_release(args)
    require_low_height = _should_require_low_height(args)
    args.success_height_threshold = _get_success_height_threshold(env, args) if require_low_height else None

    print(
        "Success criterion: hold_steps={} require_release={} require_low_height={} max_obj_z={}".format(
            args.success_hold_steps,
            require_release,
            require_low_height,
            args.success_height_threshold,
        )
    )
    print(f"Total eval episodes: {args.n_rollouts}")

    if args.video_dir is not None:
        Path(args.video_dir).mkdir(parents=True, exist_ok=True)

    stats: List[Dict[str, Any]] = []
    success_sum = 0.0
    completed = 0
    print(_format_progress_line(0, args.n_rollouts, 0.0), end="", flush=True)
    try:
        for episode_idx in range(args.n_rollouts):
            video_writer = None
            video_path = None
            if args.video_dir is not None:
                video_path = Path(args.video_dir) / f"episode_{episode_idx:06d}.mp4"
                video_writer = imageio.get_writer(
                    str(video_path),
                    fps=args.control_freq,
                    macro_block_size=args.video_macro_block_size,
                )

            ep_stats = _rollout_episode(
                env=env,
                args=args,
                required_image_keys=required_image_keys,
                expect_state=expect_state,
                state_dim=state_dim,
                episode_idx=episode_idx,
                video_writer=video_writer,
                test_cases=test_cases,
                test_reference=test_reference,
            )
            stats.append(ep_stats)
            success_sum += float(ep_stats.get("success", 0.0))
            completed += 1

            sys.stdout.write("\r" + _format_progress_line(completed, args.n_rollouts, success_sum))
            sys.stdout.flush()

            if video_writer is not None:
                video_writer.close()
                suffix = "success" if ep_stats["success"] > 0.0 else "failure"
                final_path = video_path.with_name(f"{video_path.stem}_{suffix}{video_path.suffix}")
                try:
                    os.replace(video_path, final_path)
                except Exception:
                    pass

            if args.debug_test_case:
                print("\n" + json.dumps(ep_stats, indent=2))
    finally:
        sys.stdout.write("\n")
        sys.stdout.flush()
        try:
            env.close()
        except Exception:
            pass

    success_rate = float(np.mean([s["success"] for s in stats])) if stats else 0.0
    actual_random_rollouts = 0
    if str(args.test_init_mode).lower() == "mixed":
        if test_cases:
            if extra_random_rollouts > 0:
                actual_random_rollouts = max(len(stats) - len(test_cases), 0)
        else:
            actual_random_rollouts = len(stats)
    summary = {
        "avg_return": float(np.mean([s["return"] for s in stats])) if stats else 0.0,
        "avg_horizon": float(np.mean([s["horizon"] for s in stats])) if stats else 0.0,
        "success_rate": success_rate,
        "success_count": int(np.sum([s["success"] for s in stats])) if stats else 0,
        "total_episodes": len(stats),
        "avg_success_first_step": float(
            np.mean([s["success_first_step"] for s in stats if s["success_first_step"] is not None])
        )
        if any(s["success_first_step"] is not None for s in stats)
        else 0.0,
        "server_url": args.server_url,
        "env_name": canonical_env_name,
        "test_cases_total": len(test_cases),
        "test_random_episodes_configured": extra_random_rollouts,
        "test_random_episodes_actual": actual_random_rollouts,
        "test_positions_path": test_positions_path,
    }
    print(json.dumps(summary, indent=2))

    if args.summary_path:
        out_path = Path(args.summary_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
