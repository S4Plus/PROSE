#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import io
import json
import pickle
import time
from collections import deque
from pathlib import Path
from typing import Dict, Iterable, Optional
from urllib import error, request

import numpy as np
from PIL import Image

try:
    import rospy
    import cv2
    from cv_bridge import CvBridge
    from sensor_msgs.msg import Image as RosImage, CompressedImage, JointState
    from std_msgs.msg import Header

    ROS_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    rospy = None
    cv2 = None
    CvBridge = None
    RosImage = None
    CompressedImage = None
    JointState = None
    Header = None
    ROS_AVAILABLE = False


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


def resolve_server_url(args: argparse.Namespace) -> str:
    if not args.server_url:
        raise ValueError("Missing --server-url")
    return args.server_url.rstrip("/")


def encode_image_bytes(raw: bytes) -> str:
    return base64.b64encode(raw).decode("ascii")


def _image_to_uint8_rgb(image: object) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim == 4 and array.shape[0] == 1:
        array = array[0]
    if array.ndim == 2:
        array = np.stack([array] * 3, axis=-1)
    if array.ndim == 3 and array.shape[0] in (1, 3, 4) and array.shape[2] not in (1, 3, 4):
        array = np.transpose(array, (1, 2, 0))
    if array.ndim != 3:
        raise ValueError(f"Unsupported image shape: {array.shape}")
    if array.shape[2] == 1:
        array = np.repeat(array, 3, axis=2)
    if array.shape[2] == 4:
        array = array[:, :, :3]

    if array.dtype != np.uint8:
        max_val = float(array.max()) if array.size else 1.0
        if max_val <= 1.0:
            array = np.clip(array, 0.0, 1.0) * 255.0
        else:
            array = np.clip(array, 0.0, 255.0)
        array = array.astype(np.uint8)
    return array


def encode_image_array(image: object, resize: Optional[tuple[int, int]]) -> str:
    array = _image_to_uint8_rgb(image)
    img = Image.fromarray(array)
    if resize:
        img = img.resize(resize)
    with io.BytesIO() as buffer:
        img.save(buffer, format="PNG")
        return encode_image_bytes(buffer.getvalue())


def encode_image_file(path: Path, resize: Optional[tuple[int, int]]) -> str:
    if resize:
        img = Image.open(path).convert("RGB").resize(resize)
        with io.BytesIO() as buffer:
            img.save(buffer, format="PNG")
            return encode_image_bytes(buffer.getvalue())
    return encode_image_bytes(path.read_bytes())


def resolve_image_path(value: str, image_dir: Optional[str], base_dir: Path) -> Optional[Path]:
    candidate = Path(value)
    if candidate.is_file():
        return candidate
    if not candidate.is_absolute():
        if image_dir:
            alt = Path(image_dir).expanduser() / candidate
            if alt.is_file():
                return alt
        alt = base_dir / candidate
        if alt.is_file():
            return alt
    return None


def load_sample_from_json(
    data_path: str,
    sample_index: int,
    image_dir: Optional[str],
) -> Dict[str, object]:
    data_file = Path(data_path).expanduser()
    payload = json.loads(data_file.read_text(encoding="utf-8"))

    if isinstance(payload, list):
        samples = payload
    elif isinstance(payload, dict) and isinstance(payload.get("samples"), list):
        samples = payload["samples"]
    elif isinstance(payload, dict):
        samples = None
        sample = payload
    else:
        raise ValueError("data-file must contain a JSON object or list.")

    if samples is not None:
        if sample_index < 0 or sample_index >= len(samples):
            raise IndexError(f"sample_index {sample_index} out of range for {len(samples)} samples")
        sample = samples[sample_index]

    images = sample.get("images") or sample.get("observation", {}).get("images")
    if not isinstance(images, dict):
        raise ValueError("data-file must contain an 'images' object.")
    base_dir = data_file.parent
    resolved_images: Dict[str, str] = {}
    for key, value in images.items():
        if not isinstance(value, str):
            raise ValueError(f"Image value for {key} must be a string path or base64.")
        path = resolve_image_path(value, image_dir, base_dir)
        if path:
            resolved_images[key] = str(path)
        else:
            resolved_images[key] = value

    sample["images"] = resolved_images
    for field in ("state", "action", "env_state"):
        value = sample.get(field)
        if isinstance(value, str):
            path = Path(value)
            if not path.is_absolute():
                sample[field] = str(base_dir / path)
    return sample


def _load_vector_field(sample: Dict[str, object], *keys: str) -> Optional[list[float]]:
    value: object = None
    for key in keys:
        if "." in key:
            root, child = key.split(".", 1)
            nested = sample.get(root)
            if isinstance(nested, dict) and child in nested:
                value = nested[child]
                break
        elif key in sample:
            value = sample[key]
            break

    if value is None:
        return None
    if isinstance(value, str):
        value = json.loads(Path(value).read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError(f"{keys[0]} 必须是 JSON list。")
    return [float(x) for x in value]


def resolve_env_state_payload(
    sample: Dict[str, object],
    expect_env_state: bool,
    env_state_dim: Optional[int],
) -> Optional[list[float]]:
    env_state = _load_vector_field(sample, "env_state", "observation.environment_state")
    if env_state is not None:
        if env_state_dim is not None and len(env_state) != env_state_dim:
            raise ValueError(f"env_state 维度错误，期望 {env_state_dim}，实际 {len(env_state)}")
        return env_state

    if not expect_env_state:
        return None
    if env_state_dim == 1:
        return [1.0]
    raise ValueError("样本缺少 env_state，但服务端要求该字段。")


def build_payload_from_sample(
    sample: Dict[str, object],
    image_keys: list[str],
    expect_state: bool,
    expect_env_state: bool,
    env_state_dim: Optional[int],
    resize: Optional[tuple[int, int]],
    chunk_size: Optional[int],
) -> Dict[str, object]:
    images = sample.get("images", {})
    if not isinstance(images, dict):
        raise ValueError("Sample missing images.")
    missing = [key for key in image_keys if key not in images]
    if missing:
        raise ValueError(f"Sample missing image keys: {missing}")

    encoded_images: Dict[str, str] = {}
    for key in image_keys:
        value = images[key]
        if not isinstance(value, str):
            raise ValueError(f"Image value for {key} must be a string path or base64.")
        path = Path(value)
        if path.is_file():
            encoded_images[key] = encode_image_file(path, resize)
        else:
            encoded_images[key] = value

    payload: Dict[str, object] = {"images": encoded_images}
    if expect_state:
        state = _load_vector_field(sample, "state", "observation.state")
        if state is None:
            raise ValueError("Sample missing state for a stateful policy.")
        payload["state"] = state

    env_state = resolve_env_state_payload(sample, expect_env_state, env_state_dim)
    if env_state is not None:
        payload["env_state"] = env_state

    if chunk_size is not None:
        payload["chunk_size"] = int(chunk_size)
    elif isinstance(sample.get("chunk_size"), int):
        payload["chunk_size"] = int(sample["chunk_size"])
    return payload


def print_action_summary(result: Dict[str, object]) -> None:
    action = result.get("action", [])
    shape = result.get("shape", [])
    print("Prediction ok")
    print(f"  action shape: {shape if shape else 'N/A'}")
    # print(f"  action: {action}")
    if isinstance(action, list) and action:
        flat = [x for row in action for x in row] if isinstance(action[0], list) else action
        if flat:
            print(f"  action range: [{min(flat):.3f}, {max(flat):.3f}]")


def save_capture_frame(
    output_root: Path,
    frame_idx: int,
    images: Dict[str, np.ndarray],
    qpos: Optional[np.ndarray],
) -> Dict[str, object]:
    frame_dir = output_root / f"frame_{frame_idx:06d}"
    frame_dir.mkdir(parents=True, exist_ok=True)

    images_rel: Dict[str, str] = {}
    for key, image in images.items():
        filename = f"{key}.png"
        Image.fromarray(_image_to_uint8_rgb(image)).save(frame_dir / filename)
        images_rel[key] = str(Path(frame_dir.name) / filename)

    state_rel = None
    if qpos is not None and qpos.size:
        state_path = frame_dir / "state.json"
        state_path.write_text(json.dumps(qpos.astype(float).tolist()), encoding="utf-8")
        state_rel = str(Path(frame_dir.name) / "state.json")

    return {
        "frame_index": frame_idx,
        "images": images_rel,
        "state": state_rel,
    }


def save_rollout(
    output_root: Path,
    frames: list[Dict[str, object]],
    save_pkl: bool,
    video_path: Optional[Path],
    video_fps: float,
    video_camera_key: Optional[str],
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    if save_pkl:
        pkl_path = output_root / "rollout.pkl"
        with pkl_path.open("wb") as handle:
            pickle.dump({"frames": frames}, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Saved rollout pkl to {pkl_path}")

    if video_path is None:
        return
    if cv2 is None:
        print("cv2 not available; skip video export.")
        return
    if not frames:
        print("No frames to write; skip video export.")
        return

    first_images = frames[0].get("images", {})
    if not isinstance(first_images, dict) or not first_images:
        print("Missing images in rollout frames; skip video export.")
        return

    if not video_camera_key or video_camera_key not in first_images:
        video_camera_key = next(iter(first_images))
    first_frame = first_images[video_camera_key]
    height, width = first_frame.shape[:2]
    video_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(video_fps),
        (width, height),
    )
    for frame in frames:
        images = frame.get("images", {})
        image = images.get(video_camera_key)
        if image is None:
            continue
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        writer.write(bgr)
    writer.release()
    print(f"Saved rollout video to {video_path}")


class RosOperator:
    def __init__(
        self,
        args: argparse.Namespace,
        image_key_topics: Dict[str, str],
        expect_state: bool,
    ) -> None:
        self.args = args
        self.image_key_topics = image_key_topics
        self.expect_state = expect_state
        self.bridge = CvBridge()
        self.image_deques = {
            key: deque(maxlen=args.queue_size) for key in image_key_topics
        }
        self.left_deque = deque(maxlen=args.queue_size)
        self.right_deque = deque(maxlen=args.queue_size)
        self._init_ros()

    def _init_ros(self) -> None:
        if not ROS_AVAILABLE:
            raise RuntimeError("ROS is not available; cannot use --from-ros.")

        for key, topic in self.image_key_topics.items():
            msg_type = RosImage if self.args.ros_image_type == "raw" else CompressedImage
            rospy.Subscriber(
                topic,
                msg_type,
                lambda msg, k=key: self.image_deques[k].append(msg),
                queue_size=self.args.queue_size,
                tcp_nodelay=True,
            )

        if self.expect_state:
            rospy.Subscriber(
                self.args.puppet_arm_left_topic,
                JointState,
                lambda msg: self.left_deque.append(msg),
                queue_size=self.args.queue_size,
                tcp_nodelay=True,
            )
            rospy.Subscriber(
                self.args.puppet_arm_right_topic,
                JointState,
                lambda msg: self.right_deque.append(msg),
                queue_size=self.args.queue_size,
                tcp_nodelay=True,
            )

    def _decode_image(self, msg: object) -> np.ndarray:
        if self.args.ros_image_type == "compressed":
            raw = np.frombuffer(msg.data, np.uint8)
            bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR)
            if bgr is None:
                raise ValueError("Failed to decode compressed image.")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            return rgb
        bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def get_frame(self) -> Optional[tuple[Dict[str, np.ndarray], np.ndarray]]:
        if any(len(dq) == 0 for dq in self.image_deques.values()):
            return None

        frame_time = min(dq[-1].header.stamp.to_sec() for dq in self.image_deques.values())
        if self.expect_state:
            if not self.left_deque or not self.right_deque:
                return None
            if self.left_deque[-1].header.stamp.to_sec() < frame_time:
                return None
            if self.right_deque[-1].header.stamp.to_sec() < frame_time:
                return None

        images: Dict[str, np.ndarray] = {}
        for key, dq in self.image_deques.items():
            while dq and dq[0].header.stamp.to_sec() < frame_time:
                dq.popleft()
            if not dq:
                return None
            images[key] = self._decode_image(dq.popleft())

        if not self.expect_state:
            return images, np.zeros((0,), dtype=np.float32)

        while self.left_deque and self.left_deque[0].header.stamp.to_sec() < frame_time:
            self.left_deque.popleft()
        while self.right_deque and self.right_deque[0].header.stamp.to_sec() < frame_time:
            self.right_deque.popleft()
        if not self.left_deque or not self.right_deque:
            return None
        left = self.left_deque.popleft()
        right = self.right_deque.popleft()

        left_pos = np.array(left.position, dtype=np.float32)
        right_pos = np.array(right.position, dtype=np.float32)
        if self.args.arm_order == "left_right":
            qpos = np.concatenate([left_pos, right_pos], axis=0)
        else:
            qpos = np.concatenate([right_pos, left_pos], axis=0)
        return images, qpos

    def process(self) -> Iterable[tuple[Dict[str, np.ndarray], np.ndarray]]:
        rate = rospy.Rate(self.args.frame_rate)
        while not rospy.is_shutdown():
            result = self.get_frame()
            if result is None:
                rospy.sleep(0.03)
                continue
            yield result
            rate.sleep()


def map_image_keys(image_keys: Iterable[str], args: argparse.Namespace) -> Dict[str, str]:
    mapping = {}
    for key in image_keys:
        lowered = key.lower()
        if "left" in lowered:
            mapping[key] = args.img_left_topic
        elif "right" in lowered:
            mapping[key] = args.img_right_topic
        else:
            mapping[key] = args.img_front_topic
    return mapping


def publish_actions(
    actions: np.ndarray,
    left_pub,
    right_pub,
    joint_names: list[str],
    arm_order: str,
    pub_rate: float,
    pub_steps: int,
) -> None:
    actions = np.atleast_2d(actions)
    steps = min(actions.shape[0], pub_steps)
    rate = rospy.Rate(pub_rate)
    msg = JointState()
    msg.name = joint_names
    msg.header = Header()

    for i in range(steps):
        msg.header.stamp = rospy.Time.now()
        if arm_order == "left_right":
            left = actions[i][:7]
            right = actions[i][7:]
        else:
            right = actions[i][:7]
            left = actions[i][7:]
        msg.position = left.tolist()
        left_pub.publish(msg)
        msg.position = right.tolist()
        right_pub.publish(msg)
        rate.sleep()


def run_once(args: argparse.Namespace) -> bool:
    server_url = resolve_server_url(args)
    print(f"Server: {server_url}")
    health = http_get(f"{server_url}/health", float(args.timeout))
    status = health.get("status")
    if status not in ("ok", "healthy"):
        print(f"Health check failed: status={status}")
        return False

    image_keys = list(health.get("image_keys", []))
    expect_state = bool(health.get("expect_state", False))
    expect_env_state = bool(health.get("expect_env_state", False))
    env_state_dim = health.get("env_state_dim")
    if env_state_dim is not None:
        env_state_dim = int(env_state_dim)
    print("Server info received")
    print(f"  image_keys: {image_keys}")
    print(f"  expect_state: {expect_state}")
    print(f"  expect_env_state: {expect_env_state}")
    print(f"  env_state_dim: {env_state_dim if env_state_dim is not None else 'N/A'}")
    print(
        "  image_size: "
        f"{health.get('image_width', 'N/A')}x{health.get('image_height', 'N/A')}"
    )
    print(f"  default_chunk_size: {health.get('default_chunk_size', 'N/A')}")

    if args.test_mode and not args.predict and not args.from_ros and not args.data_file:
        return True

    resize = tuple(args.resize) if args.resize else None

    if args.from_ros:
        if not ROS_AVAILABLE:
            print("ROS not available. Install ROS (rospy) or run without --from-ros.")
            return False
        rospy.init_node("act_http_client", anonymous=True)
        image_key_topics = map_image_keys(image_keys, args)
        ros_operator = RosOperator(args, image_key_topics, expect_state)

        left_pub = right_pub = None
        joint_names = args.joint_names.split(",") if args.joint_names else [f"joint{i}" for i in range(7)]
        if args.pub_actions:
            print("Publishing actions to ROS")
            input("Press Enter to continue...")
            left_pub = rospy.Publisher(args.master_arm_left_topic, JointState, queue_size=10)
            right_pub = rospy.Publisher(args.master_arm_right_topic, JointState, queue_size=10)

        rollout_root = Path(args.rollout_dir).expanduser() if args.rollout_dir else None
        if rollout_root and not args.rollout_pkl and not args.rollout_video:
            args.rollout_pkl = True
            args.rollout_video = True
        rollout_enabled = bool(rollout_root and (args.rollout_pkl or args.rollout_video))
        if rollout_enabled:
            rollout_root = rollout_root / time.strftime("rollout_%Y%m%d_%H%M%S")
        rollout_frames: list[Dict[str, object]] = []
        rollout_limit = max(0, int(args.rollout_max_frames))
        rollout_stride = max(1, int(args.rollout_stride))
        video_path = None
        if rollout_enabled and args.rollout_video:
            video_path = rollout_root / "rollout.mp4"

        capture_root = Path(args.capture_dir).expanduser() if args.capture_dir else None
        capture_entries: list[Dict[str, object]] = []
        capture_every = max(1, int(args.capture_every))
        capture_limit = max(0, int(args.capture_limit))
        if capture_root:
            capture_root.mkdir(parents=True, exist_ok=True)

        max_steps = int(args.max_steps)
        frame_idx = 0
        for images, qpos in ros_operator.process():
            if rollout_enabled and frame_idx % rollout_stride == 0:
                rollout_frames.append(
                    {
                        "frame_index": frame_idx,
                        "timestamp": float(rospy.get_time()),
                        "images": {key: _image_to_uint8_rgb(img) for key, img in images.items()},
                        "state": qpos.copy() if expect_state else None,
                        "action": None,
                    }
                )
                if rollout_limit and len(rollout_frames) >= rollout_limit:
                    save_rollout(
                        output_root=rollout_root,
                        frames=rollout_frames,
                        save_pkl=args.rollout_pkl,
                        video_path=video_path,
                        video_fps=args.rollout_fps or args.frame_rate,
                        video_camera_key=args.rollout_camera,
                    )
                    return True

            if capture_root and frame_idx % capture_every == 0:
                entry = save_capture_frame(capture_root, frame_idx, images, qpos if expect_state else None)
                capture_entries.append(entry)
                if capture_limit and len(capture_entries) >= capture_limit:
                    if capture_root:
                        manifest = {"samples": capture_entries}
                        (capture_root / "samples.json").write_text(
                            json.dumps(manifest, ensure_ascii=False, indent=2),
                            encoding="utf-8",
                        )
                        print(f"Saved capture samples to {capture_root}")
                    return True

            if args.capture_only:
                frame_idx += 1
                if max_steps > 0 and frame_idx >= max_steps:
                    break
                continue

            payload = {
                "images": {
                    key: encode_image_array(img, resize) for key, img in images.items()
                }
            }
            if expect_state:
                payload["state"] = qpos.astype(float).tolist()
            if expect_env_state:
                if env_state_dim == 1:
                    payload["env_state"] = [1.0]
                else:
                    rospy.logwarn("Server requires env_state but generic ROS client cannot infer it automatically.")
                    continue
            if args.chunk_size is not None:
                payload["chunk_size"] = int(args.chunk_size)

            try:
                result = http_post(f"{server_url}/infer", payload, float(args.timeout))
            except Exception as exc:  # noqa: BLE001
                rospy.logwarn(f"Request failed: {exc}")
                continue

            if "error" in result:
                rospy.logwarn(f"Infer error: {result.get('error')}")
                continue

            if args.log_every > 0 and frame_idx % args.log_every == 0:
                print_action_summary(result)

            if rollout_enabled and frame_idx % rollout_stride == 0 and rollout_frames:
                rollout_frames[-1]["action"] = result.get("action")

            if args.pub_actions and left_pub is not None and right_pub is not None:
                actions = np.asarray(result.get("action", []), dtype=np.float32)
                if actions.size:
                    publish_actions(
                        actions=actions,
                        left_pub=left_pub,
                        right_pub=right_pub,
                        joint_names=joint_names,
                        arm_order=args.arm_order,
                        pub_rate=args.pub_rate,
                        pub_steps=args.pub_steps,
                    )
            frame_idx += 1
            if max_steps > 0 and frame_idx >= max_steps:
                break
        if capture_root and capture_entries:
            manifest = {"samples": capture_entries}
            (capture_root / "samples.json").write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"Saved capture samples to {capture_root}")
        if rollout_enabled:
            save_rollout(
                output_root=rollout_root,
                frames=rollout_frames,
                save_pkl=args.rollout_pkl,
                video_path=video_path if args.rollout_video else None,
                video_fps=args.rollout_fps or args.frame_rate,
                video_camera_key=args.rollout_camera,
            )
        return True

    if not args.data_file and not args.predict:
        print("No prediction requested. Use --data-file or --from-ros.")
        return False


    if not args.data_file:
        print("Missing --data-file for prediction.")
        return False

    sample = load_sample_from_json(args.data_file, args.sample_index, args.image_dir)
    payload = build_payload_from_sample(
        sample,
        image_keys,
        expect_state,
        expect_env_state,
        env_state_dim,
        resize,
        args.chunk_size,
    )
    result = http_post(f"{server_url}/infer", payload, float(args.timeout))
    if "error" in result:
        print("Prediction failed")
        print(f"  error: {result.get('error')}")
        return False

    print_action_summary(result)
    if args.output_file:
        output_path = Path(args.output_file).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved prediction to {output_path}")
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ACT HTTP client")
    parser.add_argument("--server-url", type=str, required=True,
                       help="服务器URL，例如: http://172.16.111.13:8000")
    parser.add_argument("--timeout", type=int, default=30,
                       help="请求超时时间（秒）")
    parser.add_argument("--chunk-size", type=int, default=30,
                       help="请求动作步数（默认使用服务端配置）")
    parser.add_argument("--test-mode", action="store_true",
                       help="测试模式：检查服务器连接和信息")
    parser.add_argument("--predict", action="store_true",
                       help="执行离线预测（配合 --data-file）")
    parser.add_argument("--data-file", type=str,
                       help="包含观察数据的JSON文件路径")
    parser.add_argument("--sample-index", type=int, default=0,
                       help="当输入JSON为样本列表时选择第几个样本 (默认0)")
    parser.add_argument("--image-dir", type=str,
                       help="图像文件目录路径")
    parser.add_argument("--output-file", type=str,
                       help="预测结果输出文件路径")
    parser.add_argument("--resize", type=int, nargs=2, metavar=("W", "H"), default=None,
                       help="将输入图像缩放到 WxH")

    parser.add_argument("--from-ros", action="store_true",
                       help="从 ROS 话题连续采集并推理")
    parser.add_argument("--capture-dir", type=str, default=None,
                       help="在线采集时保存图片与state的目录")
    parser.add_argument("--capture-limit", type=int, default=0,
                       help="在线采集保存的最大帧数（0 表示不限制）")
    parser.add_argument("--capture-every", type=int, default=1,
                       help="每 N 帧保存一次（默认每帧保存）")
    parser.add_argument("--capture-only", action="store_true",
                       help="仅采集保存，不进行推理或发布动作")
    parser.add_argument("--ros-image-type", choices=["raw", "compressed"], default="raw",
                       help="ROS 图像消息类型（raw=Image, compressed=CompressedImage）")
    parser.add_argument("--img-front-topic", type=str, default="/camera_f/color/image_raw",
                       help="前/高位相机图像话题")
    parser.add_argument("--img-left-topic", type=str, default="/camera_l/color/image_raw",
                       help="左腕相机图像话题")
    parser.add_argument("--img-right-topic", type=str, default="/camera_r/color/image_raw",
                       help="右腕相机图像话题")
    parser.add_argument("--puppet-arm-left-topic", type=str, default="/puppet/joint_left",
                       help="左臂关节 JointState 话题")
    parser.add_argument("--puppet-arm-right-topic", type=str, default="/puppet/joint_right",
                       help="右臂关节 JointState 话题")
    parser.add_argument("--frame-rate", type=int, default=30,
                       help="ROS 采集帧率")
    parser.add_argument("--queue-size", type=int, default=2000,
                       help="ROS 消息缓存长度")
    parser.add_argument("--arm-order", choices=["right_left", "left_right"], default="left_right",
                       help="state/action 14维的左右臂顺序")
    parser.add_argument("--log-every", type=int, default=10,
                       help="每 N 帧打印一次 action（<=0 关闭）")
    parser.add_argument("--max-steps", type=int, default=250,
                       help="在线推理最大步数（<=0 表示不限制）")

    parser.add_argument("--pub-actions", action="store_true",
                       help="将预测动作发布到 ROS")
    parser.add_argument("--master-arm-left-topic", type=str, default="/master/joint_left",
                       help="发布左臂 JointState 的话题")
    parser.add_argument("--master-arm-right-topic", type=str, default="/master/joint_right",
                       help="发布右臂 JointState 的话题")
    parser.add_argument("--pub-rate", type=float, default=15,
                       help="动作发布频率 Hz")
    parser.add_argument("--pub-steps", type=int, default=1,
                       help="每帧发布动作的步数（默认发布第一步）")
    parser.add_argument("--joint-names", type=str, default="",
                       help="发布 JointState 时的关节名，逗号分隔（默认 joint0..joint6）")
    parser.add_argument("--rollout-dir", type=str, default=None,
                       help="保存 rollout 的目录（pkl + video）")
    parser.add_argument("--rollout-max-frames", type=int, default=0,
                       help="rollout 最大帧数（0 表示不限制）")
    parser.add_argument("--rollout-stride", type=int, default=1,
                       help="rollout 每 N 帧保存一次（默认每帧）")
    parser.add_argument("--rollout-video", action="store_true",
                       help="保存 rollout 视频（mp4）")
    parser.add_argument("--rollout-pkl", action="store_true",
                       help="保存 rollout pkl")
    parser.add_argument("--rollout-camera", type=str, default="",
                       help="视频使用的相机 key（默认使用第一路）")
    parser.add_argument("--rollout-fps", type=float, default=15,
                       help="rollout 视频 FPS（默认使用 --frame-rate）")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    success = run_once(args)
    if not success:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
