#!/usr/bin/env python3
"""
MimicGen evaluation client for ACT.
Sends observations to a policy server and steps the simulation with returned actions.
"""
import argparse
import base64
import json
import os
import pickle
import socket
import struct
import sys
from typing import Any, Dict, List, Optional, Tuple

import imageio
import cv2
try:
    import msgpack
except Exception:  # pragma: no cover - optional dependency for ws transport
    msgpack = None
import numpy as np
import robosuite as suite
from robosuite.controllers import load_controller_config
from robosuite.utils.input_utils import choose_robots

import mimicgen  # register MimicGen environments


DEFAULT_SERVER_URL = os.environ.get("MIMICGEN_ACT_SERVER_URL", "http://127.0.0.1:8000/act")
DEFAULT_TRANSPORT = os.environ.get("MIMICGEN_ACT_TRANSPORT", "http")


def _pack_array(obj):
    if (isinstance(obj, (np.ndarray, np.generic))) and obj.dtype.kind in ("V", "O", "c"):
        raise ValueError("Unsupported dtype: {}".format(obj.dtype))
    if isinstance(obj, np.ndarray):
        return {
            b"__ndarray__": True,
            b"data": obj.tobytes(),
            b"dtype": obj.dtype.str,
            b"shape": obj.shape,
        }
    if isinstance(obj, np.generic):
        return {
            b"__npgeneric__": True,
            b"data": obj.item(),
            b"dtype": obj.dtype.str,
        }
    return obj


def _unpack_array(obj):
    if b"__ndarray__" in obj:
        return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"])
    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])
    return obj


class MsgpackNumpy:
    @staticmethod
    def _require_msgpack():
        if msgpack is None:
            raise RuntimeError("msgpack is required for ws transport. Install msgpack to use ws.")

    @staticmethod
    def Packer():
        MsgpackNumpy._require_msgpack()
        return msgpack.Packer(default=_pack_array)

    @staticmethod
    def packb(data):
        MsgpackNumpy._require_msgpack()
        return msgpack.packb(data, default=_pack_array)

    @staticmethod
    def unpackb(data):
        MsgpackNumpy._require_msgpack()
        return msgpack.unpackb(data, object_hook=_unpack_array)


class WebsocketPolicyClient:
    def __init__(self, uri: str, api_key: Optional[str] = None):
        try:
            import websockets.sync.client as ws_client
        except Exception as exc:
            raise RuntimeError(
                "websockets is required for ws transport. Install websockets>=11 to use ws."
            ) from exc
        headers = {"Authorization": "Api-Key {}".format(api_key)} if api_key else None
        self._ws = ws_client.connect(uri, compression=None, max_size=None, additional_headers=headers)
        self._packer = MsgpackNumpy.Packer()
        meta = self._ws.recv()
        if isinstance(meta, (bytes, bytearray)):
            self._metadata = MsgpackNumpy.unpackb(meta)
        else:
            self._metadata = {}

    def infer(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        payload = self._packer.pack(obs)
        self._ws.send(payload)
        response = self._ws.recv()
        if isinstance(response, str):
            raise RuntimeError("Error in inference server:\n{}".format(response))
        return MsgpackNumpy.unpackb(response)

    def close(self) -> None:
        try:
            self._ws.close()
        except Exception:
            pass


def choose_mimicgen_environment() -> str:
    """Return a chosen MimicGen environment name (interactive)."""
    # try to import robosuite task zoo to include those envs in the robosuite registry
    try:
        import robosuite_task_zoo  # noqa: F401
    except Exception:
        pass

    # all base robosuite environments (and maybe robosuite task zoo)
    robosuite_envs = set(suite.ALL_ENVIRONMENTS)

    # all environments including mimicgen environments
    all_envs = set(suite.ALL_ENVIRONMENTS)

    # get only mimicgen envs
    only_mimicgen = sorted(all_envs - robosuite_envs)

    # keep only envs that correspond to the different reset distributions from the paper
    envs = [x for x in only_mimicgen if x[-1].isnumeric()]

    # Select environment to run
    print("Here is a list of environments in the suite:\n")

    for k, env in enumerate(envs):
        print("[{}] {}".format(k, env))
    print()
    try:
        s = input("Choose an environment to run (enter a number from 0 to {}): ".format(len(envs) - 1))
        # parse input into a number within range
        k = min(max(int(s), 0), len(envs) - 1)
    except Exception:
        k = 0
        print("Input is not valid. Use {} by default.\n".format(envs[k]))

    return envs[k]


def _encode_array(arr: np.ndarray) -> Dict[str, Any]:
    return {
        "__ndarray__": True,
        "dtype": str(arr.dtype),
        "shape": list(arr.shape),
        "data": base64.b64encode(arr.tobytes()).decode("ascii"),
    }


def _decode_array(obj: Dict[str, Any]) -> np.ndarray:
    data = base64.b64decode(obj["data"])
    arr = np.frombuffer(data, dtype=np.dtype(obj["dtype"]))
    return arr.reshape(obj["shape"])


def encode_item(obj: Any, mode: str) -> Any:
    if isinstance(obj, np.ndarray):
        if mode == "raw":
            return obj.tolist()
        return _encode_array(obj)
    if isinstance(obj, (np.floating, np.integer, np.bool_)):
        return obj.item()
    if isinstance(obj, dict):
        return {k: encode_item(v, mode) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [encode_item(v, mode) for v in obj]
    return obj


def decode_item(obj: Any) -> Any:
    if isinstance(obj, dict) and obj.get("__ndarray__"):
        return _decode_array(obj)
    if isinstance(obj, dict):
        return {k: decode_item(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [decode_item(v) for v in obj]
    return obj


def decode_action(obj: Any) -> np.ndarray:
    if isinstance(obj, np.ndarray):
        return obj
    if isinstance(obj, dict) and obj.get("__ndarray__"):
        return _decode_array(obj)
    if isinstance(obj, list):
        return np.asarray(obj, dtype=np.float32)
    if isinstance(obj, (int, float, np.integer, np.floating)):
        return np.asarray([obj], dtype=np.float32)
    if isinstance(obj, dict):
        # allow nested dict response
        maybe = decode_item(obj)
        if isinstance(maybe, np.ndarray):
            return maybe
    raise ValueError("Unsupported action format: {}".format(type(obj)))


def _get_test_object_bounds(env: Any) -> Optional[Tuple[str, Dict[str, Any]]]:
    if not hasattr(env, "_get_initial_placement_bounds"):
        return None
    try:
        bounds = env._get_initial_placement_bounds()
        # print(bounds)
    except Exception:
        return None
    if not isinstance(bounds, dict) or not bounds:
        return None
    for key in ("nut", "square_nut", "square"):
        if key in bounds:
            return key, bounds[key]
    for key, value in bounds.items():
        if "nut" in key:
            return key, value
    for key in ("hammer",):
        if key in bounds:
            return key, bounds[key]
    for key, value in bounds.items():
        if "hammer" in key:
            return key, value
    for key in ("object", "sorting_object"):
        if key in bounds:
            return key, bounds[key]
    first_key = next(iter(bounds))
    return first_key, bounds[first_key]


def _linspace_centers(min_val: float, max_val: float, step: float) -> List[float]:
    if step <= 0:
        return []
    start = min_val + step * 0.5
    end = max_val - step * 0.5
    if end < start:
        return []
    values = np.arange(start, end + 1e-9, step)
    return [float(v) for v in values]


def _linspace_inclusive(min_val: float, max_val: float, step: float) -> List[float]:
    if step <= 0:
        return []
    values = list(np.arange(min_val, max_val + 1e-9, step))
    if not values or values[-1] < max_val - 1e-9:
        values.append(max_val)
    return [float(v) for v in values]


def _dedupe_points(points: List[Tuple[float, float]], ndigits: int = 6) -> List[Tuple[float, float]]:
    seen = set()
    unique = []
    for x_val, y_val in points:
        key = (round(x_val, ndigits), round(y_val, ndigits))
        if key in seen:
            continue
        seen.add(key)
        unique.append((x_val, y_val))
    return unique


def _build_test_positions(
    env: Any, step: float, include_boundary: bool
) -> Tuple[List[Tuple[float, float]], Optional[str]]:
    bounds_pair = _get_test_object_bounds(env)
    if not bounds_pair:
        return [], None
    object_name, bounds = bounds_pair
    x_min, x_max = bounds["x"]
    y_min, y_max = bounds["y"]
    ref = bounds.get("reference", np.zeros(3))
    centers_x = _linspace_centers(x_min, x_max, step)
    centers_y = _linspace_centers(y_min, y_max, step)
    points: List[Tuple[float, float]] = []
    for x_val in centers_x:
        for y_val in centers_y:
            points.append((x_val + ref[0], y_val + ref[1]))
    if include_boundary:
        edge_x = _linspace_inclusive(x_min, x_max, step)
        edge_y = _linspace_inclusive(y_min, y_max, step)
        for x_val in edge_x:
            points.append((x_val + ref[0], y_min + ref[1]))
            points.append((x_val + ref[0], y_max + ref[1]))
        for y_val in edge_y:
            points.append((x_min + ref[0], y_val + ref[1]))
            points.append((x_max + ref[0], y_val + ref[1]))
    return _dedupe_points(points), object_name


def _find_object_by_name(env: Any, object_name: Optional[str]) -> Optional[Any]:
    if object_name is None or not hasattr(env, "objects"):
        return None
    for obj in env.objects:
        name = getattr(obj, "name", None)
        if name is None and hasattr(obj, "root_body"):
            name = obj.root_body
        if name and object_name in name:
            return obj
    return None


def _get_active_object(env: Any, object_name: Optional[str]) -> Optional[Any]:
    if object_name in (None, "nut", "square_nut", "square"):
        if hasattr(env, "nuts"):
            if hasattr(env, "nut_id"):
                try:
                    return env.nuts[int(env.nut_id)]
                except Exception:
                    pass
            if hasattr(env, "nut_to_id"):
                for key in ("square", "square_nut"):
                    if key in env.nut_to_id:
                        return env.nuts[int(env.nut_to_id[key])]
            try:
                return env.nuts[0]
            except Exception:
                pass

    if object_name is not None and "hammer" in object_name:
        if hasattr(env, "sorting_object"):
            return env.sorting_object
        if hasattr(env, "hammer"):
            return env.hammer

    if object_name is not None and object_name in ("drawer", "cabinet", "cabinet_object"):
        if hasattr(env, "cabinet_object"):
            return env.cabinet_object

    obj = _find_object_by_name(env, object_name)
    if obj is not None:
        return obj

    return None


def _set_object_xy(env: Any, x_val: float, y_val: float, object_name: Optional[str]) -> bool:
    obj = _get_active_object(env, object_name)
    if obj is None or not hasattr(env, "sim"):
        return False
    try:
        if not hasattr(obj, "joints") or not obj.joints:
            return False
        joint_name = obj.joints[0]
        qpos = np.array(env.sim.data.get_joint_qpos(joint_name), dtype=np.float64).copy()
        if qpos.size < 2:
            return False
        qpos[0] = x_val
        qpos[1] = y_val
        env.sim.data.set_joint_qpos(joint_name, qpos)
        env.sim.forward()
        return True
    except Exception:
        return False


def _axis_angle_quat(axis: str, angle: float) -> np.ndarray:
    c = float(np.cos(angle / 2.0))
    s = float(np.sin(angle / 2.0))
    if axis == "x":
        return np.array([c, s, 0.0, 0.0], dtype=np.float64)
    if axis == "y":
        return np.array([c, 0.0, s, 0.0], dtype=np.float64)
    if axis == "z":
        return np.array([c, 0.0, 0.0, s], dtype=np.float64)
    raise ValueError("Invalid rotation axis: {}".format(axis))


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
        arr = np.asarray(reference, dtype=np.float64)
        if arr.size == 3:
            return arr
        return np.zeros(3, dtype=np.float64)
    if isinstance(reference, str):
        if reference == "table_offset" and hasattr(env, "table_offset"):
            return np.asarray(env.table_offset, dtype=np.float64)
    return np.zeros(3, dtype=np.float64)


def _set_object_pose(env: Any, object_name: Optional[str], spec: Dict[str, Any], reference: Any) -> bool:
    obj = _get_active_object(env, object_name)
    if obj is None or not hasattr(env, "sim"):
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
        quat = np.asarray(spec["quat"], dtype=np.float64)
    else:
        angle = None
        for key in ("rot", "z_rot", "rotation", "angle"):
            if key in spec:
                angle = float(spec[key])
                break
        if angle is not None:
            axis = str(spec.get("rotation_axis", "z"))
            quat = _axis_angle_quat(axis, angle)

    if quat is not None and hasattr(obj, "init_quat"):
        quat = _quat_multiply(quat, np.asarray(obj.init_quat, dtype=np.float64))

    try:
        if hasattr(obj, "joints") and obj.joints:
            joint_name = obj.joints[0]
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

        if not hasattr(obj, "root_body"):
            return False
        body_id = env.sim.model.body_name2id(obj.root_body)
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


def _extract_case_objects(case: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    if "objects" in case and isinstance(case["objects"], dict):
        return case["objects"]
    objects = {}
    for key, value in case.items():
        if isinstance(value, dict):
            objects[key] = value
    return objects


def _apply_test_case(env: Any, case: Dict[str, Any], reference: Any) -> bool:
    ok_any = False
    for object_name, spec in _extract_case_objects(case).items():
        if not isinstance(spec, dict):
            continue
        ok = _set_object_pose(env, object_name, spec, reference)
        ok_any = ok_any or ok
    return ok_any


def _load_test_positions(path: str) -> Tuple[List[Dict[str, Any]], Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        cases = data
        reference = None
    else:
        cases = data.get("cases") or data.get("test_cases") or data.get("positions") or []
        reference = data.get("reference", None)
    if not isinstance(cases, list):
        cases = []
    return cases, reference


def _resolve_test_positions_path(args: argparse.Namespace, env_name: str) -> Optional[str]:
    if getattr(args, "test_positions_path", None):
        return args.test_positions_path
    test_dir = getattr(args, "test_positions_dir", None)
    if not test_dir:
        return None
    candidate = os.path.join(test_dir, "{}.json".format(env_name))
    if os.path.isfile(candidate):
        return candidate
    return None


def _ensure_video_extension(path: str, default_ext: str = ".mp4") -> str:
    base, ext = os.path.splitext(path)
    if ext:
        return path
    return base + default_ext


def _append_video_suffix(path: str, suffix: str) -> str:
    base, ext = os.path.splitext(path)
    if not ext:
        return base + "_" + suffix
    return base + "_" + suffix + ext


def _format_progress_line(completed: int, total: int, success_sum: float, bar_width: int = 30) -> str:
    if total <= 0:
        ratio = 1.0
    else:
        ratio = min(max(completed / float(total), 0.0), 1.0)
    filled = int(round(bar_width * ratio))
    bar = "#" * filled + "-" * (bar_width - filled)
    success_rate = (success_sum / float(completed)) if completed > 0 else 0.0
    return "[{}] {}/{} ({:5.1f}%) | success {:5.1f}%".format(
        bar,
        completed,
        total,
        ratio * 100.0,
        success_rate * 100.0,
    )


def _print_progress(completed: int, total: int, success_sum: float) -> None:
    line = _format_progress_line(completed, total, success_sum).ljust(90)
    if sys.stderr.isatty():
        end = "\n" if completed >= total else "\r"
        sys.stderr.write(line + end)
    else:
        sys.stderr.write(line + "\n")
    sys.stderr.flush()


def recvall(sock: socket.socket, size: int) -> bytes:
    data = b""
    while len(data) < size:
        chunk = sock.recv(size - len(data))
        if not chunk:
            raise RuntimeError("Socket closed while receiving data")
        data += chunk
    return data


def request_action_http(url: str, payload: Dict[str, Any], timeout_s: float) -> Dict[str, Any]:
    from urllib.request import Request, urlopen

    body = json.dumps(payload).encode("utf-8")
    req = Request(url, data=body, headers={"Content-Type": "application/json"})
    with urlopen(req, timeout=timeout_s) as resp:
        resp_body = resp.read().decode("utf-8")
    return json.loads(resp_body)


def request_action_tcp(host: str, port: int, payload: Dict[str, Any], timeout_s: float) -> Dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    header = struct.pack("!I", len(body))
    with socket.create_connection((host, port), timeout=timeout_s) as sock:
        sock.sendall(header + body)
        resp_len = struct.unpack("!I", recvall(sock, 4))[0]
        resp_body = recvall(sock, resp_len).decode("utf-8")
    return json.loads(resp_body)


def get_action_from_server(
    transport: str,
    server_url: str,
    server_host: str,
    server_port: int,
    payload: Dict[str, Any],
    timeout_s: float,
) -> Dict[str, Any]:
    if transport == "http":
        return request_action_http(server_url, payload, timeout_s)
    if transport == "tcp":
        return request_action_tcp(server_host, server_port, payload, timeout_s)
    if transport == "ws":
        raise RuntimeError("Websocket transport requires a persistent client.")
    raise ValueError("Unsupported transport: {}".format(transport))


def parse_camera_names(camera_names: Optional[str]) -> List[str]:
    if not camera_names:
        return ["agentview"]
    return [name.strip() for name in camera_names.split(",") if name.strip()]


def select_obs_keys(obs: Dict[str, Any], keys: Optional[List[str]]) -> Dict[str, Any]:
    if not keys:
        return obs
    return {k: obs[k] for k in keys if k in obs}


def _encode_image_to_base64(image: np.ndarray) -> str:
    array = np.asarray(image)
    if array.ndim == 2:
        array = np.stack([array] * 3, axis=-1)
    if array.shape[-1] == 4:
        array = array[..., :3]
    if array.dtype != np.uint8:
        if np.issubdtype(array.dtype, np.floating) and array.max() <= 1.0:
            array = array * 255.0
        array = np.clip(array, 0, 255).round().astype(np.uint8)
    bgr = array[..., ::-1]
    ok, buf = cv2.imencode(".jpg", bgr)
    if not ok:
        raise ValueError("Failed to encode image as JPEG.")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def _format_act_payload(
    obs: Dict[str, Any],
    flip_vertical: bool,
    flip_horizontal: bool,
    bgr_to_rgb: bool,
) -> Dict[str, Any]:
    if "observation.images.agentview" in obs and "observation.images.wrist" in obs:
        agentview = obs["observation.images.agentview"]
        wrist = obs["observation.images.wrist"]
        state = obs.get("observation.state", obs.get("state"))
    else:
        agentview = obs.get("agentview_image")
        if agentview is None:
            agentview = obs.get("agentview")
        wrist = obs.get("robot0_eye_in_hand_image")
        if wrist is None:
            wrist = obs.get("robot0_eye_in_hand")
        if wrist is None:
            wrist = obs.get("wrist")
        if agentview is None or wrist is None:
            raise KeyError("Missing camera images in observation; enable --use_camera_obs and camera names.")

        joint_pos = obs.get("robot0_joint_pos")
        if joint_pos is None:
            sin_term = obs.get("robot0_joint_pos_sin")
            cos_term = obs.get("robot0_joint_pos_cos")
            if sin_term is not None and cos_term is not None:
                joint_pos = np.arctan2(sin_term, cos_term)
        if joint_pos is None:
            joint_pos = np.zeros(7, dtype=np.float32)

        grip_joint = obs.get("robot0_gripper_qpos")
        if grip_joint is None:
            grip_joint = np.zeros(2, dtype=np.float32)

        state = np.concatenate([np.asarray(joint_pos).reshape(-1), np.asarray(grip_joint).reshape(-1)], axis=-1)

    if flip_vertical:
        agentview = np.flipud(agentview)
        wrist = np.flipud(wrist)
    if flip_horizontal:
        agentview = np.fliplr(agentview)
        wrist = np.fliplr(wrist)
    if bgr_to_rgb:
        agentview = agentview[..., ::-1]
        wrist = wrist[..., ::-1]

    payload: Dict[str, Any] = {
        "images": {
            "observation.images.agentview": _encode_image_to_base64(agentview),
            "observation.images.wrist": _encode_image_to_base64(wrist),
        }
    }
    if state is not None:
        state_arr = np.asarray(state).astype(np.float32).reshape(-1)
        payload["state"] = state_arr.tolist()
    return payload


def _extract_action_from_response(
    response: Any,
    action_field: str,
    use_action_chunk: bool,
    cached_actions: Optional[np.ndarray],
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    if cached_actions is not None and cached_actions.size > 0:
        return cached_actions[0], cached_actions[1:] if cached_actions.shape[0] > 1 else None

    action_obj = response
    if isinstance(response, dict):
        if "actions" in response:
            action_obj = response["actions"]
        elif "action" in response:
            action_obj = response["action"]
        elif action_field in response:
            action_obj = response[action_field]

    action = decode_action(action_obj)
    if use_action_chunk and action.ndim >= 2:
        return action[0], action[1:] if action.shape[0] > 1 else None
    if not use_action_chunk and action.ndim >= 2 and action.shape[0] == 1:
        return action[0], None
    return action, None


def get_success(env: Any) -> float:
    if hasattr(env, "_check_success"):
        try:
            return float(env._check_success())
        except Exception:
            pass
    if hasattr(env, "is_success"):
        try:
            result = env.is_success()
            if isinstance(result, dict):
                if "task" in result:
                    return float(result["task"])
                if "success" in result:
                    return float(result["success"])
            return float(result)
        except Exception:
            return 0.0
    return 0.0


def _render_frame(
    env: Any,
    camera_name: str,
    height: int,
    width: int,
    flip_vertical: bool,
    flip_horizontal: bool,
    bgr_to_rgb: bool,
) -> Optional[np.ndarray]:
    if hasattr(env, "sim"):
        try:
            frame = env.sim.render(camera_name=camera_name, height=height, width=width)
        except Exception:
            frame = None
    try:
        if frame is None:
            frame = env.render()
    except Exception:
        frame = None
    if not isinstance(frame, np.ndarray):
        return None
    if flip_vertical:
        frame = np.flipud(frame)
    if flip_horizontal:
        frame = np.fliplr(frame)
    if bgr_to_rgb:
        frame = frame[..., ::-1]
    return frame


def make_env(args: argparse.Namespace, camera_names: List[str]):
    env_name = args.env_name or choose_mimicgen_environment()
    robots = args.robots or choose_robots(exclude_bimanual=True)
    controller_configs = load_controller_config(default_controller=args.controller)
    if str(args.controller).upper().startswith("JOINT"):
        controller_configs["control_delta"] = False

    env = suite.make(
        env_name=env_name,
        robots=robots,
        controller_configs=controller_configs,
        has_renderer=args.render,
        has_offscreen_renderer=(args.video_path is not None or args.video_dir is not None or args.use_camera_obs),
        ignore_done=args.ignore_done,
        use_camera_obs=args.use_camera_obs,
        camera_names=camera_names if args.use_camera_obs else None,
        camera_heights=args.camera_height,
        camera_widths=args.camera_width,
        control_freq=args.control_freq,
    )
    return env


def rollout_episode(
    env: Any,
    args: argparse.Namespace,
    camera_names: List[str],
    episode_idx: int,
    video_writer: Optional[Any],
    ws_client: Optional[WebsocketPolicyClient],
    nut_test_xy: Optional[Tuple[float, float]] = None,
    test_object_name: Optional[str] = None,
    test_case: Optional[Dict[str, Any]] = None,
    test_reference: Any = None,
) -> Dict[str, Any]:
    obs = env.reset()
    applied = False
    if test_case is not None:
        applied = _apply_test_case(env, test_case, test_reference)
    elif nut_test_xy is not None:
        applied = _set_object_xy(env, nut_test_xy[0], nut_test_xy[1], test_object_name)
    if applied and hasattr(env, "_get_observations"):
        try:
            obs = env._get_observations(force_update=True)
        except Exception:
            pass
    total_reward = 0.0
    success = 0.0
    first_success_step: Optional[int] = None

    action_low, action_high = env.action_spec
    cached_actions = None

    video_flip_vertical = args.flip_vertical if args.video_flip_vertical is None else args.video_flip_vertical
    video_flip_horizontal = args.flip_horizontal if args.video_flip_horizontal is None else args.video_flip_horizontal
    video_bgr_to_rgb = args.bgr_to_rgb if args.video_bgr_to_rgb is None else args.video_bgr_to_rgb

    # 用于保存 rollout 数据
    rollout_observations = []
    rollout_actions = []
    rollout_rewards = []

    for step_i in range(args.horizon):
        action = None
        if args.use_action_chunk and cached_actions is not None:
            action, cached_actions = _extract_action_from_response(
                response=None,
                action_field=args.action_field,
                use_action_chunk=args.use_action_chunk,
                cached_actions=cached_actions,
            )

        if action is None:
            obs_payload = select_obs_keys(obs, args.obs_keys)

            if args.transport == "ws":
                if ws_client is None:
                    raise RuntimeError("Websocket client is not initialized.")
                response = ws_client.infer(obs_payload)
            else:
                act_payload = _format_act_payload(
                    obs_payload,
                    args.flip_vertical,
                    args.flip_horizontal,
                    args.bgr_to_rgb,
                )
                if not args.use_action_chunk:
                    act_payload["chunk_size"] = 1
                payload = {
                    "images": act_payload["images"],
                    "episode": episode_idx,
                    "step": step_i,
                    "env_name": getattr(env, "name", args.env_name),
                }
                if "state" in act_payload:
                    payload["state"] = act_payload["state"]
                if "chunk_size" in act_payload:
                    payload["chunk_size"] = act_payload["chunk_size"]
                response = get_action_from_server(
                    transport=args.transport,
                    server_url=args.server_url,
                    server_host=args.server_host,
                    server_port=args.server_port,
                    payload=payload,
                    timeout_s=args.timeout_s,
                )

            action, cached_actions = _extract_action_from_response(
                response=response,
                action_field=args.action_field,
                use_action_chunk=args.use_action_chunk,
                cached_actions=cached_actions,
            )

        if args.action_scale != 1.0 or args.action_bias != 0.0:
            action = action * args.action_scale + args.action_bias

        if args.clip_action:
            action = np.clip(action, action_low, action_high)

        # 保存当前观测和动作(在执行前)
        if hasattr(args, 'save_rollouts_dir') and args.save_rollouts_dir is not None:
            rollout_observations.append({k: np.array(v) if not isinstance(v, np.ndarray) else v for k, v in obs.items()})
            rollout_actions.append(np.array(action))

        obs, reward, done, _ = env.step(action)
        total_reward += float(reward)
        step_success = get_success(env)
        if step_success > 0.0 and first_success_step is None:
            first_success_step = step_i + 1
        success = max(success, step_success)

        # 保存reward
        if hasattr(args, 'save_rollouts_dir') and args.save_rollouts_dir is not None:
            rollout_rewards.append(float(reward))

        if args.render:
            env.render()

        if video_writer is not None and (step_i % args.video_skip == 0):
            frames = []
            for cam in camera_names:
                frame = _render_frame(
                    env,
                    cam,
                    args.video_height,
                    args.video_width,
                    video_flip_vertical,
                    video_flip_horizontal,
                    video_bgr_to_rgb,
                )
                if frame is not None:
                    frames.append(frame)
            if not frames:
                continue
            frame = np.concatenate(frames, axis=1) if len(frames) > 1 else frames[0]
            video_writer.append_data(np.ascontiguousarray(frame))

        if step_success > 0.0:
            break

        if done and not args.ignore_done:
            break

    # 保存 rollout 数据
    if hasattr(args, 'save_rollouts_dir') and args.save_rollouts_dir is not None:
        os.makedirs(args.save_rollouts_dir, exist_ok=True)
        suffix = "success" if success > 0.0 else "failure"
        rollout_file = os.path.join(args.save_rollouts_dir, f"episode_{episode_idx:06d}_{suffix}.pkl")
        
        # 组织数据为 dict 格式 (与 save_eval_rollouts.py 兼容)
        rollout_data = {
            "observations": {},
            "actions": np.stack(rollout_actions) if rollout_actions else np.array([]),
            "reward": total_reward,
            "success": success > 0.0,
            "rewards": np.array(rollout_rewards) if rollout_rewards else np.array([]),
        }
        
        # 聚合观测数据
        if rollout_observations:
            # 假设所有观测都有相同的 keys
            obs_keys = rollout_observations[0].keys()
            for key in obs_keys:
                try:
                    # 尝试 stack 所有时间步的观测
                    values = [obs[key] for obs in rollout_observations]
                    
                    # 检查是否为嵌套 dict (如 images)
                    if isinstance(values[0], dict):
                        rollout_data["observations"][key] = {}
                        for sub_key in values[0].keys():
                            sub_values = [v[sub_key] for v in values]
                            rollout_data["observations"][key][sub_key] = np.stack(sub_values)
                    else:
                        rollout_data["observations"][key] = np.stack(values)
                except Exception as e:
                    # 如果 stack 失败,保存为列表
                    rollout_data["observations"][key] = [obs[key] for obs in rollout_observations]
        
        # 保存为 pkl
        with open(rollout_file, "wb") as f:
            pickle.dump(rollout_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    episode_length = step_i + 1
    return {
        "episode": episode_idx,
        "return": total_reward,
        "horizon": episode_length,
        "trajectory_length": episode_length,
        "success_first_step": first_success_step,
        "success_trajectory_length": episode_length if success > 0.0 else None,
        "success": success,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MimicGen ACT evaluation client")

    # server config
    parser.add_argument("--transport", choices=["http", "tcp", "ws"], default=DEFAULT_TRANSPORT)
    parser.add_argument("--server_url", default=DEFAULT_SERVER_URL)
    parser.add_argument("--server_host", default="127.0.0.1")
    parser.add_argument("--server_port", type=int, default=8000)
    parser.add_argument("--timeout_s", type=float, default=10.0)
    parser.add_argument("--action_field", default="action")
    parser.add_argument("--api_key", default=None)

    # env config
    parser.add_argument("--env_name", default="Square_D1")
    parser.add_argument("--robots", nargs="+", default="Panda")
    parser.add_argument("--controller", default="JOINT_POSITION")
    parser.add_argument("--control_freq", type=int, default=20)
    parser.add_argument("--ignore_done", action="store_true")

    # obs config
    parser.add_argument("--use_camera_obs", action="store_true")
    parser.add_argument("--camera_names", default=None)
    parser.add_argument("--camera_height", type=int, default=84)
    parser.add_argument("--camera_width", type=int, default=84)
    parser.add_argument("--obs_keys", nargs="+", default=None)
    parser.add_argument("--flip_vertical", action="store_true", default=True)
    parser.add_argument("--no_flip_vertical", dest="flip_vertical", action="store_false")
    parser.add_argument("--flip_horizontal", action="store_true", default=False)
    parser.add_argument("--video_flip_vertical", dest="video_flip_vertical", action="store_true", default=None)
    parser.add_argument("--video_no_flip_vertical", dest="video_flip_vertical", action="store_false")
    parser.add_argument("--video_flip_horizontal", dest="video_flip_horizontal", action="store_true", default=None)
    parser.add_argument("--video_no_flip_horizontal", dest="video_flip_horizontal", action="store_false")
    parser.add_argument("--video_bgr_to_rgb", dest="video_bgr_to_rgb", action="store_true", default=None)
    parser.add_argument("--video_no_bgr_to_rgb", dest="video_bgr_to_rgb", action="store_false")
    parser.add_argument("--bgr_to_rgb", action="store_true", default=False)

    # rollout config
    parser.add_argument("--n_rollouts", type=int, default=1)
    parser.add_argument("--num_episodes", type=int, default=20)
    parser.add_argument("--num_tests", type=int, default=None)
    parser.add_argument("--horizon", type=int, default=400)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--nut_test_grid", dest="nut_test_grid", action="store_true", default=True)
    parser.add_argument("--no_nut_test_grid", dest="nut_test_grid", action="store_false")
    parser.add_argument("--nut_test_step", type=float, default=0.05)
    parser.add_argument("--nut_test_include_boundary", dest="nut_test_include_boundary", action="store_true", default=True)
    parser.add_argument("--no_nut_test_include_boundary", dest="nut_test_include_boundary", action="store_false")
    parser.add_argument("--test_positions_path", default=None, help="JSON file with explicit test cases")
    parser.add_argument("--test_positions_dir", default="test", help="Directory containing <env_name>.json test cases")

    # video config
    parser.add_argument("--video_path", default=None)
    parser.add_argument("--video_dir", default=None)
    parser.add_argument("--video_skip", type=int, default=5)
    parser.add_argument("--video_height", type=int, default=None)
    parser.add_argument("--video_width", type=int, default=None)
    parser.add_argument("--video_macro_block_size", type=int, default=1)

    # action post-processing
    parser.add_argument("--action_scale", type=float, default=1.0)
    parser.add_argument("--action_bias", type=float, default=0.0)
    parser.add_argument("--clip_action", dest="clip_action", action="store_true", default=None)
    parser.add_argument("--no_clip_action", dest="clip_action", action="store_false")
    parser.add_argument("--use_action_chunk", action="store_true", default=True)
    parser.add_argument("--no_action_chunk", dest="use_action_chunk", action="store_false")

    # rollout saving
    parser.add_argument("--save_rollouts_dir", default=None, help="Directory to save rollout pkl files for influence calculation")
    parser.add_argument("--save_rollouts_format", choices=["dict", "pandas"], default="dict", help="Format to save rollouts (dict or pandas)")

    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.num_episodes is not None:
        args.n_rollouts = args.num_episodes
    if args.num_tests is not None:
        args.n_rollouts = args.num_tests

    if args.seed is not None:
        np.random.seed(args.seed)

    if args.use_camera_obs and not args.camera_names:
        camera_names = ["agentview", "robot0_eye_in_hand"]
    else:
        camera_names = parse_camera_names(args.camera_names)
    if args.video_height is None:
        args.video_height = args.camera_height
    if args.video_width is None:
        args.video_width = args.camera_width

    env = make_env(args, camera_names)
    env_name = getattr(env, "name", args.env_name)

    if args.clip_action is None:
        if str(args.controller).upper().startswith("JOINT"):
            args.clip_action = False
        else:
            args.clip_action = True

    ws_client = None
    if args.transport == "ws":
        uri = args.server_url
        if not uri.startswith("ws"):
            uri = "ws://{}:{}".format(args.server_host, args.server_port)
        ws_client = WebsocketPolicyClient(uri=uri, api_key=args.api_key)

    if args.seed is not None and hasattr(env, "seed"):
        try:
            env.seed(args.seed)
        except Exception:
            pass

    test_cases: List[Dict[str, Any]] = []
    test_reference = None
    test_positions_path = _resolve_test_positions_path(args, env_name)
    if test_positions_path:
        test_cases, test_reference = _load_test_positions(test_positions_path)
        if test_cases:
            args.n_rollouts = len(test_cases)
            args.nut_test_grid = False
            sys.stderr.write(
                "Using {} test cases from {}.\n".format(len(test_cases), test_positions_path)
            )
            sys.stderr.flush()

    test_positions: List[Tuple[float, float]] = []
    test_object_name: Optional[str] = None
    if args.nut_test_grid and not test_cases:
        test_positions, test_object_name = _build_test_positions(
            env,
            args.nut_test_step,
            args.nut_test_include_boundary,
        )
        if test_positions:
            args.n_rollouts = len(test_positions)
            sys.stderr.write(
                "Using {} test positions for object '{}' (grid + boundary).\n".format(
                    len(test_positions), test_object_name
                )
            )
            sys.stderr.flush()

    stats = []
    success_sum = 0.0
    if args.n_rollouts > 0:
        _print_progress(0, args.n_rollouts, success_sum)
    try:
        for ep in range(args.n_rollouts):
            video_writer = None
            video_path = None
            if args.video_path is not None:
                video_path = _ensure_video_extension(args.video_path)
                video_writer = imageio.get_writer(
                    video_path,
                    fps=args.control_freq,
                    macro_block_size=args.video_macro_block_size,
                )
            elif args.video_dir is not None:
                os.makedirs(args.video_dir, exist_ok=True)
                video_path = os.path.join(args.video_dir, "episode_{:03d}.mp4".format(ep))
                video_writer = imageio.get_writer(
                    video_path,
                    fps=args.control_freq,
                    macro_block_size=args.video_macro_block_size,
                )

            nut_test_xy = test_positions[ep % len(test_positions)] if test_positions else None
            if nut_test_xy is not None:
                sys.stderr.write(
                    "Episode {:03d} test {} xy = ({:.4f}, {:.4f})\n".format(
                        ep, test_object_name, nut_test_xy[0], nut_test_xy[1]
                    )
                )
                sys.stderr.flush()
            ep_stats = rollout_episode(
                env,
                args,
                camera_names,
                ep,
                video_writer,
                ws_client,
                nut_test_xy=nut_test_xy,
                test_object_name=test_object_name,
                test_case=test_cases[ep] if test_cases else None,
                test_reference=test_reference,
            )
            stats.append(ep_stats)
            success_sum += float(ep_stats.get("success", 0.0))
            _print_progress(ep + 1, args.n_rollouts, success_sum)

            if video_writer is not None:
                video_writer.close()
                if video_path is not None:
                    suffix = "success" if float(ep_stats.get("success", 0.0)) > 0.0 else "failure"
                    final_path = _append_video_suffix(video_path, suffix)
                    if final_path != video_path:
                        try:
                            os.replace(video_path, final_path)
                        except Exception:
                            pass

            print(json.dumps(ep_stats, indent=2))

        if stats:
            avg_return = float(np.mean([s["return"] for s in stats]))
            avg_traj_len = float(np.mean([s["horizon"] for s in stats]))
            success_count = int(np.sum([s["success"] for s in stats]))
            total_episodes = len(stats)
            success_rate = float(success_count) / float(total_episodes)
            success_horizons = [s["horizon"] for s in stats if s.get("success", 0.0) > 0.0]
            avg_success_traj_len = float(np.mean(success_horizons)) if success_horizons else 0.0
            success_first_steps = [s["success_first_step"] for s in stats if s.get("success_first_step") is not None]
            avg_success_first_step = float(np.mean(success_first_steps)) if success_first_steps else 0.0
            print(
                json.dumps(
                    {
                        "avg_return": avg_return,
                        "avg_trajectory_length": avg_traj_len,
                        "avg_success_trajectory_length": avg_success_traj_len,
                        "avg_success_first_step": avg_success_first_step,
                        "success_count": success_count,
                        "total_episodes": total_episodes,
                        "success_rate": success_rate,
                    },
                    indent=2,
                )
            )
    finally:
        if ws_client is not None:
            ws_client.close()
        try:
            env.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
