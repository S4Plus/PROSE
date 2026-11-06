#!/usr/bin/env python3
"""
Simple HTTP client test for the ACT server.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
from urllib import error, request

import numpy as np
import torch
from PIL import Image


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

    def health_check(self) -> Dict[str, object]:
        return http_get(f"{self.server_url}/health", self.timeout)

    def get_server_info(self) -> Dict[str, object]:
        return self.health_check()

    def reset(self) -> Dict[str, object]:
        return http_post(f"{self.server_url}/reset", {}, self.timeout)

    def predict_action(self, observation: Dict[str, object]) -> Dict[str, object]:
        return http_post(f"{self.server_url}/infer", observation, self.timeout)


IMAGE_ALIASES = {
    "head_camera": "observation.images.head_camera",
    "right_wrist_camera": "observation.images.right_wrist_camera",
    "left_wrist_camera": "observation.images.left_wrist_camera",
}
DEFAULT_DATASET_PATH = Path("/ext_workspace/liushuo/swk/CB81")


def find_default_image() -> Optional[str]:
    repo_root = Path(__file__).resolve().parent.parent
    for name in ("nut_heatmap.png", "yaw_heatmap.png", "nut_heatmap_en.png", "yaw_heatmap_en.png"):
        candidate = repo_root / name
        if candidate.is_file():
            return str(candidate)
    return None


def resolve_default_dataset_path(path: Optional[str]) -> Optional[str]:
    if path:
        return path
    if DEFAULT_DATASET_PATH.is_dir():
        return str(DEFAULT_DATASET_PATH)
    return None


def encode_image(path: str) -> str:
    image_path = Path(path)
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")
    return base64.b64encode(image_path.read_bytes()).decode("ascii")


def _to_numpy(value: object) -> np.ndarray:
    try:
        import torch
    except ImportError:  # pragma: no cover - optional dependency
        torch = None

    if torch is not None and isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _image_to_uint8_rgb(image: object) -> np.ndarray:
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("Pillow is required to encode dataset images.") from exc

    if isinstance(image, str):
        return np.array(Image.open(image).convert("RGB"))
    if isinstance(image, Image.Image):
        return np.array(image.convert("RGB"))

    array = _to_numpy(image)
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


class RunningStats:
    def __init__(self) -> None:
        self.count = 0
        self.nan_count = 0
        self.sum = 0.0
        self.sumsq = 0.0
        self.min: Optional[float] = None
        self.max: Optional[float] = None

    def update(self, values: object) -> None:
        arr = _to_numpy(values)
        arr = np.asarray(arr, dtype=np.float64).reshape(-1)
        if arr.size == 0:
            return
        finite_mask = np.isfinite(arr)
        self.nan_count += int(arr.size - np.count_nonzero(finite_mask))
        arr = arr[finite_mask]
        if arr.size == 0:
            return
        self.count += int(arr.size)
        self.sum += float(arr.sum())
        self.sumsq += float((arr**2).sum())
        val_min = float(arr.min())
        val_max = float(arr.max())
        self.min = val_min if self.min is None else min(self.min, val_min)
        self.max = val_max if self.max is None else max(self.max, val_max)

    def mean(self) -> Optional[float]:
        if self.count == 0:
            return None
        return self.sum / self.count

    def std(self) -> Optional[float]:
        if self.count == 0:
            return None
        mean = self.mean()
        if mean is None:
            return None
        var = max(0.0, self.sumsq / self.count - mean**2)
        return var**0.5


def encode_image_data(image: object) -> str:
    array = _image_to_uint8_rgb(image)
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("Pillow is required to encode dataset images.") from exc

    with io.BytesIO() as buffer:
        Image.fromarray(array).save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("ascii")


def parse_image_map(entries: Iterable[str]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for entry in entries:
        if "=" not in entry:
            raise ValueError(f"Invalid image mapping '{entry}', expected KEY=PATH.")
        key, path = entry.split("=", 1)
        key = key.strip()
        path = path.strip()
        if not key or not path:
            raise ValueError(f"Invalid image mapping '{entry}', empty key/path.")
        mapping[key] = path
    return mapping


def parse_key_map(entries: Iterable[str], label: str) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for entry in entries:
        if "=" not in entry:
            raise ValueError(f"Invalid {label} mapping '{entry}', expected KEY=VALUE.")
        key, value = entry.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or not value:
            raise ValueError(f"Invalid {label} mapping '{entry}', empty key/value.")
        mapping[key] = value
    return mapping


def parse_state(state_arg: Optional[str], state_file: Optional[str]) -> Optional[List[float]]:
    if state_file:
        data = json.loads(Path(state_file).read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError("State file must contain a JSON list.")
        return [float(x) for x in data]
    if not state_arg:
        return None
    try:
        data = json.loads(state_arg)
        if isinstance(data, list):
            return [float(x) for x in data]
    except json.JSONDecodeError:
        pass
    parts = [part.strip() for part in state_arg.split(",") if part.strip()]
    if not parts:
        return None
    return [float(x) for x in parts]


def ensure_lerobot_on_path(lerobot_repo: Optional[str]) -> None:
    if lerobot_repo:
        root = Path(lerobot_repo).expanduser()
    else:
        root = Path(os.environ.get("LEROBOT_REPO", "/workspace/liushuo/cupid/lerobot_2.1")).expanduser()
    lerobot_src = root / "src"
    if lerobot_src.is_dir() and str(lerobot_src) not in sys.path:
        sys.path.insert(0, str(lerobot_src))


def resolve_lerobot_location(
    dataset_path: Optional[str],
    lerobot_home: Optional[str],
    repo_id: Optional[str],
) -> Tuple[Path, str]:
    if dataset_path:
        dataset_root = Path(dataset_path).expanduser()
        meta_dir = dataset_root / "meta"
        if (meta_dir / "stats.json").is_file() or (meta_dir / "info.json").is_file():
            resolved_repo_id = repo_id or dataset_root.name
            return dataset_root, resolved_repo_id
        if repo_id:
            candidate = dataset_root / repo_id
            meta_dir = candidate / "meta"
            if (meta_dir / "stats.json").is_file() or (meta_dir / "info.json").is_file():
                return candidate, repo_id
        raise ValueError(f"Could not find a LeRobot dataset under '{dataset_root}'.")

    if not repo_id:
        raise ValueError("Missing --repo-id for LeRobot dataset.")
    if lerobot_home:
        resolved_home = Path(lerobot_home).expanduser()
    else:
        env_home = os.environ.get("HF_LEROBOT_HOME")
        if not env_home:
            raise ValueError("Missing --lerobot-home or HF_LEROBOT_HOME for LeRobot dataset.")
        resolved_home = Path(env_home).expanduser()
    if not resolved_home.exists():
        raise ValueError(f"LeRobot home does not exist: {resolved_home}")
    return resolved_home, repo_id


def load_lerobot_dataset(args: argparse.Namespace):
    ensure_lerobot_on_path(args.lerobot_repo)
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    lerobot_home, repo_id = resolve_lerobot_location(args.dataset_path, args.lerobot_home, args.repo_id)
    os.environ.setdefault("HF_LEROBOT_HOME", str(lerobot_home))

    dataset = LeRobotDataset(repo_id=repo_id, root=lerobot_home, download_videos=False)
    if dataset.num_episodes <= 0:
        raise ValueError("LeRobot dataset has no episodes.")
    return dataset, repo_id, lerobot_home


def resolve_first_episode_range(dataset, demo_step: int) -> Tuple[int, int, int, int]:
    episode_ids = sorted(getattr(dataset, "_episode_ids", list(dataset.meta.episodes.keys())))
    if not episode_ids:
        raise ValueError("Failed to resolve episode indices from dataset metadata.")
    episode_idx = episode_ids[0]
    ep_pos = dataset._episode_position[episode_idx]
    ep_start = int(dataset.episode_data_index["from"][ep_pos])
    ep_end = int(dataset.episode_data_index["to"][ep_pos])
    if ep_end <= ep_start:
        raise ValueError(f"Episode {episode_idx} is empty.")
    demo_step = max(0, min(int(demo_step), ep_end - ep_start - 1))
    frame_idx = ep_start + demo_step
    return episode_idx, ep_start, ep_end, frame_idx


def resolve_dataset_image_key(
    server_key: str,
    sample_keys: Iterable[str],
    overrides: Dict[str, str],
) -> Optional[str]:
    if server_key in overrides:
        return overrides[server_key]
    if server_key in sample_keys:
        return server_key
    alias = IMAGE_ALIASES.get(server_key)
    if alias and alias in sample_keys:
        return alias
    if not server_key.startswith("observation.images."):
        candidate = f"observation.images.{server_key}"
        if candidate in sample_keys:
            return candidate
    else:
        suffix = server_key.split("observation.images.", 1)[1]
        if suffix in sample_keys:
            return suffix
    return None


def resolve_dataset_state_key(sample_keys: Iterable[str], preferred: Optional[str]) -> Optional[str]:
    if preferred and preferred in sample_keys:
        return preferred
    for key in ("observation.state", "state", "observation.joint_state", "joint_state"):
        if key in sample_keys:
            return key
    return None


def build_observation(
    image_keys: List[str],
    image_default: Optional[str],
    image_map: Dict[str, str],
    state: Optional[List[float]],
    chunk_size: Optional[int],
) -> Dict[str, object]:
    images: Dict[str, str] = {}
    for key in image_keys:
        path = image_map.get(key) or image_default
        if not path:
            raise ValueError(f"Missing image for key '{key}'. Provide --image or --image-map.")
        images[key] = encode_image(path)

    payload: Dict[str, object] = {"images": images}
    if state is not None:
        payload["state"] = state
    if chunk_size is not None:
        payload["chunk_size"] = int(chunk_size)
    return payload


def build_lerobot_observation(
    args: argparse.Namespace,
    image_keys: List[str],
    expect_state: bool,
) -> Tuple[Dict[str, object], Optional[np.ndarray], Dict[str, object]]:
    dataset, repo_id, lerobot_home = load_lerobot_dataset(args)
    episode_idx, ep_start, ep_end, frame_idx = resolve_first_episode_range(dataset, args.demo_step)
    sample = dataset[frame_idx]
    payload, action_array = build_lerobot_observation_from_sample(
        sample=sample,
        image_keys=image_keys,
        expect_state=expect_state,
        args=args,
    )
    info = {
        "repo_id": repo_id,
        "lerobot_home": str(lerobot_home),
        "episode_index": episode_idx,
        "frame_index": frame_idx,
        "frame_offset": frame_idx - ep_start,
        "episode_start": ep_start,
        "episode_end": ep_end,
    }
    return payload, action_array, info


def build_lerobot_observation_from_sample(
    sample: Dict[str, object],
    image_keys: List[str],
    expect_state: bool,
    args: argparse.Namespace,
) -> Tuple[Dict[str, object], Optional[np.ndarray]]:
    sample_keys = set(sample.keys())
    image_overrides = parse_key_map(args.dataset_image_map, "dataset image")

    images: Dict[str, str] = {}
    for key in image_keys:
        sample_key = resolve_dataset_image_key(key, sample_keys, image_overrides)
        if not sample_key:
            raise ValueError(f"Missing dataset image key for server key '{key}'.")
        images[key] = encode_image_data(sample[sample_key])

    payload: Dict[str, object] = {"images": images}
    if expect_state:
        state_key = resolve_dataset_state_key(sample_keys, args.dataset_state_key)
        if not state_key:
            raise ValueError("Server expects state but dataset does not provide it.")
        payload["state"] = _to_numpy(sample[state_key]).reshape(-1).astype(float).tolist()
    if args.chunk_size is not None:
        payload["chunk_size"] = int(args.chunk_size)

    action = sample.get("action")
    action_array = _to_numpy(action) if action is not None else None
    return payload, action_array


def save_dataset_samples(
    dataset,
    args: argparse.Namespace,
    image_keys: List[str],
    expect_state: bool,
    output_dir: str,
) -> List[Dict[str, object]]:
    output_root = Path(output_dir).expanduser()
    output_root.mkdir(parents=True, exist_ok=True)

    episode_idx, ep_start, ep_end, frame_idx = resolve_first_episode_range(dataset, args.demo_step)
    num_samples = max(1, int(args.num_samples))
    stride = max(1, int(args.sample_stride))
    start_offset = frame_idx - ep_start

    image_overrides = parse_key_map(args.dataset_image_map, "dataset image")
    samples: List[Dict[str, object]] = []
    for idx in range(num_samples):
        frame_offset = start_offset + idx * stride
        frame_pos = ep_start + frame_offset
        if frame_pos >= ep_end:
            break
        sample = dataset[frame_pos]
        sample_keys = set(sample.keys())

        sample_dir = output_root / f"frame_{frame_pos:06d}"
        sample_dir.mkdir(parents=True, exist_ok=True)

        images: Dict[str, str] = {}
        for key in image_keys:
            sample_key = resolve_dataset_image_key(key, sample_keys, image_overrides)
            if not sample_key:
                raise ValueError(f"Missing dataset image key for server key '{key}'.")
            array = _image_to_uint8_rgb(sample[sample_key])
            filename = f"{key}.png"
            Image.fromarray(array).save(sample_dir / filename)
            images[key] = str(Path(sample_dir.name) / filename)

        state_path = None
        if expect_state:
            state_key = resolve_dataset_state_key(sample_keys, args.dataset_state_key)
            if not state_key:
                raise ValueError("Server expects state but dataset does not provide it.")
            state = _to_numpy(sample[state_key]).reshape(-1).astype(float).tolist()
            state_file = sample_dir / "state.json"
            state_file.write_text(json.dumps(state), encoding="utf-8")
            state_path = str(Path(sample_dir.name) / "state.json")

        action_path = None
        action = sample.get("action")
        if action is not None:
            action_list = _to_numpy(action).reshape(-1).astype(float).tolist()
            action_file = sample_dir / "action.json"
            action_file.write_text(json.dumps(action_list), encoding="utf-8")
            action_path = str(Path(sample_dir.name) / "action.json")

        samples.append(
            {
                "episode_index": episode_idx,
                "frame_index": frame_pos,
                "frame_offset": frame_offset,
                "images": images,
                "state": state_path,
                "action": action_path,
            }
        )

    manifest = {"episode_index": episode_idx, "samples": samples}
    (output_root / "samples.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return samples


def load_saved_samples(sample_dir: Path) -> List[Dict[str, object]]:
    manifest_path = sample_dir / "samples.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing samples.json in {sample_dir}")
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    samples = data.get("samples", data)
    if not isinstance(samples, list):
        raise ValueError("samples.json must contain a list or {'samples': [...]} structure.")

    normalized: List[Dict[str, object]] = []
    for entry in samples:
        images = {}
        for key, rel_path in entry.get("images", {}).items():
            path = Path(rel_path)
            if not path.is_absolute():
                path = sample_dir / path
            images[key] = str(path)
        normalized_entry = dict(entry)
        normalized_entry["images"] = images
        for field in ("state", "action"):
            value = entry.get(field)
            if value:
                path = Path(value)
                if not path.is_absolute():
                    path = sample_dir / path
                normalized_entry[field] = str(path)
        normalized.append(normalized_entry)
    return normalized


def build_observation_from_saved_sample(
    sample: Dict[str, object],
    expect_state: bool,
    chunk_size: Optional[int],
) -> Tuple[Dict[str, object], Optional[np.ndarray]]:
    images: Dict[str, str] = {}
    for key, path in sample.get("images", {}).items():
        images[key] = encode_image(str(path))
    payload: Dict[str, object] = {"images": images}

    if expect_state:
        state_path = sample.get("state")
        if not state_path:
            raise ValueError("Sample is missing state for a stateful policy.")
        state_data = json.loads(Path(state_path).read_text(encoding="utf-8"))
        payload["state"] = [float(x) for x in state_data]

    if chunk_size is not None:
        payload["chunk_size"] = int(chunk_size)

    action_array = None
    action_path = sample.get("action")
    if action_path:
        action_data = json.loads(Path(action_path).read_text(encoding="utf-8"))
        action_array = _to_numpy(action_data)
    return payload, action_array


def compare_actions(pred: object, ref: object) -> Tuple[Optional[Dict[str, float]], Tuple[int, ...], Tuple[int, ...]]:
    if pred is None or ref is None:
        return None, tuple(), tuple()
    pred_arr = _to_numpy(pred)
    ref_arr = _to_numpy(ref)
    pred_shape = tuple(pred_arr.shape)
    ref_shape = tuple(ref_arr.shape)
    if pred_arr.size == 0 or ref_arr.size == 0:
        return None, pred_shape, ref_shape
    if pred_arr.ndim == 2 and ref_arr.ndim == 1:
        pred_arr = pred_arr[0]
    elif pred_arr.ndim == 1 and ref_arr.ndim == 2:
        ref_arr = ref_arr[0]
    if pred_arr.shape != ref_arr.shape:
        return None, pred_shape, ref_shape
    diff = pred_arr.astype(np.float32) - ref_arr.astype(np.float32)
    return (
        {
            "mae": float(np.mean(np.abs(diff))),
            "rmse": float(np.sqrt(np.mean(diff**2))),
            "max_abs": float(np.max(np.abs(diff))),
            "pred_min": float(np.min(pred_arr)),
            "pred_max": float(np.max(pred_arr)),
            "ref_min": float(np.min(ref_arr)),
            "ref_max": float(np.max(ref_arr)),
        },
        tuple(pred_arr.shape),
        tuple(ref_arr.shape),
    )


def test_client(args: argparse.Namespace) -> bool:
    print(f"Testing server: {args.server_url}")
    print("=" * 50)

    client = ACTHTTPClient(args.server_url, timeout=args.timeout)

    print("1. Health check...")
    try:
        health_result = client.health_check()
    except error.URLError as exc:
        print(f"Health check failed: {exc}")
        return False

    status = health_result.get("status")
    if status not in ("ok", "healthy"):
        print(f"Health check failed: status={status}")
        return False
    print("Health check ok")

    print("\n2. Server info...")
    info_result = client.get_server_info()
    image_keys = info_result.get("image_keys", [])
    expect_state = info_result.get("expect_state", False)
    print("Server info received")
    print(f"  image_keys: {image_keys}")
    print(f"  expect_state: {expect_state}")
    print(
        "  image_size: "
        f"{info_result.get('image_width', 'N/A')}x{info_result.get('image_height', 'N/A')}"
    )
    print(f"  default_chunk_size: {info_result.get('default_chunk_size', 'N/A')}")

    if args.reset:
        print("\n2.1 Reset server...")
        try:
            client.reset()
            print("Reset ok")
        except error.URLError as exc:
            print(f"Reset failed: {exc}")
            return False

    print("\n3. Action prediction...")

    args.dataset_path = resolve_default_dataset_path(args.dataset_path)
    use_dataset = bool(args.dataset_path or args.repo_id or args.lerobot_home)
    use_saved_samples = bool(args.sample_dir)
    dataset = None
    repo_id = None
    lerobot_home = None
    episode_idx = None
    ep_start = None
    ep_end = None
    frame_idx = None

    if args.save_samples_dir:
        if not use_dataset:
            print("Saving samples requires a dataset. Provide --dataset-path or --repo-id.")
            return False
        try:
            dataset, repo_id, lerobot_home = load_lerobot_dataset(args)
            save_dataset_samples(
                dataset=dataset,
                args=args,
                image_keys=list(image_keys),
                expect_state=expect_state,
                output_dir=args.save_samples_dir,
            )
            print(f"  saved samples: {args.save_samples_dir}")
        except Exception as exc:  # noqa: BLE001
            print(f"Failed to save samples: {exc}")
            return False

    if use_saved_samples:
        try:
            sample_dir = Path(args.sample_dir).expanduser()
            samples = load_saved_samples(sample_dir)
            if not samples:
                print("No samples found in sample directory.")
                return False
            sample_keys = set(samples[0].get("images", {}).keys())
            missing = [key for key in image_keys if key not in sample_keys]
            if missing:
                print(f"Saved samples are missing image keys required by server: {missing}")
                return False
        except Exception as exc:  # noqa: BLE001
            print(f"Failed to load saved samples: {exc}")
            return False
    elif use_dataset:
        try:
            if dataset is None:
                dataset, repo_id, lerobot_home = load_lerobot_dataset(args)
            episode_idx, ep_start, ep_end, frame_idx = resolve_first_episode_range(
                dataset, args.demo_step
            )
            print("  dataset input:")
            print(f"    repo_id: {repo_id}")
            print(f"    lerobot_home: {lerobot_home}")
            print(
                f"    episode_index: {episode_idx}, "
                f"frame_index: {frame_idx}, "
                f"frame_offset: {frame_idx - ep_start}"
            )
        except Exception as exc:  # noqa: BLE001
            print(f"Failed to build dataset observation: {exc}")
            return False
    else:
        image_default = args.image or find_default_image()
        image_map = parse_image_map(args.image_map)
        state = parse_state(args.state, args.state_file)

        if expect_state and state is None:
            print("Server expects state. Provide --state or --state-file.")
            return False

        try:
            observation = build_observation(
                image_keys=list(image_keys),
                image_default=image_default,
                image_map=image_map,
                state=state,
                chunk_size=args.chunk_size,
            )
        except (ValueError, FileNotFoundError) as exc:
            print(f"Failed to build observation: {exc}")
            return False

    if use_saved_samples or use_dataset:
        if args.chunk_size is None:
            args.chunk_size = 1
        pred_stats = RunningStats()
        ref_stats = RunningStats()
        metric_sums = {"mae": 0.0, "rmse": 0.0, "max_abs": 0.0}
        metric_count = 0
        shape_mismatch = 0

        num_samples = max(1, int(args.num_samples))
        sample_stride = max(1, int(args.sample_stride))
        used_samples = 0

        if use_saved_samples:
            selected_samples = samples[::sample_stride][:num_samples]
            total_requested = len(selected_samples)
            for idx, sample in enumerate(selected_samples):
                observation, demo_action = build_observation_from_saved_sample(
                    sample=sample,
                    expect_state=expect_state,
                    chunk_size=args.chunk_size,
                )
                used_samples += 1
                if idx == 0:
                    print(f"  observation keys: {list(observation.keys())}")
                try:
                    result = client.predict_action(observation)
                except error.URLError as exc:
                    print(f"Prediction failed: {exc}")
                    return False

                if "error" in result:
                    print("Prediction failed")
                    print(f"  error: {result.get('error')}")
                    return False

                action = result.get("action", [])
                shape = result.get("shape", [])
                if idx == 0:
                    print("Prediction ok")
                    print(f"  action shape: {shape if shape else 'N/A'}")
                    print("action", action)
                    if isinstance(action, list) and action:
                        flat = [x for row in action for x in row] if isinstance(action[0], list) else action
                        if flat:
                            print(f"  action range: [{min(flat):.3f}, {max(flat):.3f}]")

                if "action" in result:
                    pred_stats.update(result.get("action"))
                if demo_action is not None:
                    ref_stats.update(demo_action)
                    metrics, _, _ = compare_actions(result.get("action"), demo_action)
                    if metrics is None:
                        shape_mismatch += 1
                    else:
                        metric_count += 1
                        metric_sums["mae"] += metrics["mae"]
                        metric_sums["rmse"] += metrics["rmse"]
                        metric_sums["max_abs"] += metrics["max_abs"]
        else:
            start_offset = frame_idx - ep_start
            total_requested = num_samples
            for idx in range(num_samples):
                frame_offset = start_offset + idx * sample_stride
                frame_pos = ep_start + frame_offset
                if frame_pos >= ep_end:
                    break
                sample = dataset[frame_pos]
                observation, demo_action = build_lerobot_observation_from_sample(
                    sample=sample,
                    image_keys=list(image_keys),
                    expect_state=expect_state,
                    args=args,
                )
                used_samples += 1
                if idx == 0:
                    print(f"  observation keys: {list(observation.keys())}")
                    images = observation.get("images")
                    if isinstance(images, dict):
                        for key, value in images.items():
                            if not isinstance(value, str):
                                continue
                            try:
                                raw = base64.b64decode(value)
                                with io.BytesIO(raw) as buffer:
                                    Image.open(buffer).convert("RGB").save(f"{key}.png")
                            except Exception as exc:  # noqa: BLE001
                                print(f"    failed to save image {key}: {exc}")
                    else:
                        for key, value in observation.items():
                            if isinstance(value, torch.Tensor):
                                value = value.cpu().numpy()
                            if isinstance(value, np.ndarray):
                                if value.ndim == 3 and value.shape[0] in (3, 4):
                                    value = value.transpose(1, 2, 0)
                                Image.fromarray(value).save(f"{key}.png")
                try:
                    result = client.predict_action(observation)
                except error.URLError as exc:
                    print(f"Prediction failed: {exc}")
                    return False

                if "error" in result:
                    print("Prediction failed")
                    print(f"  error: {result.get('error')}")
                    return False

                action = result.get("action", [])
                shape = result.get("shape", [])
                if idx == 0:
                    print("Prediction ok")
                    print(f"  action shape: {shape if shape else 'N/A'}")
                    print("action", action)
                    if isinstance(action, list) and action:
                        flat = [x for row in action for x in row] if isinstance(action[0], list) else action
                        if flat:
                            print(f"  action range: [{min(flat):.3f}, {max(flat):.3f}]")

                if "action" in result:
                    pred_stats.update(result.get("action"))
                if demo_action is not None:
                    ref_stats.update(demo_action)
                    metrics, _, _ = compare_actions(result.get("action"), demo_action)
                    if metrics is None:
                        shape_mismatch += 1
                    else:
                        metric_count += 1
                        metric_sums["mae"] += metrics["mae"]
                        metric_sums["rmse"] += metrics["rmse"]
                        metric_sums["max_abs"] += metrics["max_abs"]

        if used_samples == 0:
            print("Failed to sample any frames from dataset/saved samples.")
            return False

        print("\n4. Summary...")
        print(f"  samples: {used_samples} (requested {total_requested}, stride {sample_stride})")
        if pred_stats.count:
            pred_mean = pred_stats.mean()
            pred_std = pred_stats.std()
            print(
                "  pred values: count={}, nan_inf={}, min={:.3f}, max={:.3f}, mean={:.3f}, std={:.3f}".format(
                    pred_stats.count,
                    pred_stats.nan_count,
                    pred_stats.min if pred_stats.min is not None else float("nan"),
                    pred_stats.max if pred_stats.max is not None else float("nan"),
                    pred_mean if pred_mean is not None else float("nan"),
                    pred_std if pred_std is not None else float("nan"),
                )
            )
        if ref_stats.count:
            ref_mean = ref_stats.mean()
            ref_std = ref_stats.std()
            print(
                "  demo values: count={}, min={:.3f}, max={:.3f}, mean={:.3f}, std={:.3f}".format(
                    ref_stats.count,
                    ref_stats.min if ref_stats.min is not None else float("nan"),
                    ref_stats.max if ref_stats.max is not None else float("nan"),
                    ref_mean if ref_mean is not None else float("nan"),
                    ref_std if ref_std is not None else float("nan"),
                )
            )
        if metric_count:
            print(
                "  compare mean: mae={:.4f}, rmse={:.4f}, max_abs={:.4f}".format(
                    metric_sums["mae"] / metric_count,
                    metric_sums["rmse"] / metric_count,
                    metric_sums["max_abs"] / metric_count,
                )
            )
        if shape_mismatch:
            print(f"  compare skipped: {shape_mismatch} (shape mismatch or empty)")

        reasonable = True
        reasons = []
        if pred_stats.nan_count > 0:
            reasonable = False
            reasons.append("pred has NaN/Inf")

        ref_range = None
        if ref_stats.min is not None and ref_stats.max is not None:
            ref_range = ref_stats.max - ref_stats.min

        if metric_count and ref_range and ref_range > 0:
            mean_mae = metric_sums["mae"] / metric_count
            mae_limit = args.mae_ratio * ref_range
            if mean_mae > mae_limit:
                reasonable = False
                reasons.append(f"mae {mean_mae:.4f} > {mae_limit:.4f} (ratio {args.mae_ratio})")

            pred_min = pred_stats.min
            pred_max = pred_stats.max
            if pred_min is not None and pred_max is not None:
                margin = args.range_margin_ratio * ref_range
                if pred_min < ref_stats.min - margin or pred_max > ref_stats.max + margin:
                    reasonable = False
                    reasons.append(
                        "pred range out of demo range "
                        f"(margin ratio {args.range_margin_ratio})"
                    )

        print("  judgement: {}".format("reasonable" if reasonable else "not reasonable"))
        if reasons:
            print("  reasons: " + "; ".join(reasons))
        if not reasonable:
            return False
    else:
        print(f"  observation keys: {list(observation.keys())}")
        try:
            result = client.predict_action(observation)
        except error.URLError as exc:
            print(f"Prediction failed: {exc}")
            return False

        if "error" in result:
            print("Prediction failed")
            print(f"  error: {result.get('error')}")
            return False

        action = result.get("action", [])
        print("action", action)
        shape = result.get("shape", [])
        print("Prediction ok")
        print(f"  action shape: {shape if shape else 'N/A'}")
        if isinstance(action, list) and action:
            flat = [x for row in action for x in row] if isinstance(action[0], list) else action
            if flat:
                print(f"  action range: [{min(flat):.3f}, {max(flat):.3f}]")

    print("\n" + "=" * 50)
    print("All tests passed")
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ACT HTTP client test")
    parser.add_argument("--server_url", help="Server URL, e.g. http://127.0.0.1:8000")
    parser.add_argument("--timeout", type=float, default=30.0, help="Request timeout in seconds")
    parser.add_argument("--image", type=str, default=None, help="Image file used for all image keys")
    parser.add_argument(
        "--image-map",
        action="append",
        default=[],
        metavar="KEY=PATH",
        help="Per-key image path mapping (repeatable)",
    )
    parser.add_argument("--state", type=str, default=None, help="State as JSON list or comma-separated values")
    parser.add_argument("--state-file", type=str, default=None, help="Path to JSON list state file")
    parser.add_argument("--chunk-size", type=int, default=None, help="Override action chunk size")
    parser.add_argument("--reset", action="store_true", help="Reset the server before inference")
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help="Local LeRobot dataset root path (defaults to /ext_workspace/liushuo/swk/CB81 when available)",
    )
    parser.add_argument(
        "--save-samples-dir",
        type=str,
        default=None,
        help="Save first-episode samples to this directory",
    )
    parser.add_argument(
        "--sample-dir",
        type=str,
        default=None,
        help="Use saved samples from this directory instead of reading the dataset",
    )
    parser.add_argument("--lerobot-home", type=str, default=None, help="HF_LEROBOT_HOME override")
    parser.add_argument("--repo-id", type=str, default=None, help="LeRobot dataset repo_id")
    parser.add_argument("--lerobot-repo", type=str, default=None, help="Path to lerobot source repo")
    parser.add_argument("--demo-step", type=int, default=0, help="Frame offset in first episode")
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10,
        help="Number of frames sampled from the first episode for stats",
    )
    parser.add_argument(
        "--sample-stride",
        type=int,
        default=1,
        help="Frame stride when sampling multiple frames from the first episode",
    )
    parser.add_argument(
        "--dataset-image-map",
        action="append",
        default=[],
        metavar="KEY=DATASET_KEY",
        help="Map server image key to dataset key (repeatable)",
    )
    parser.add_argument("--dataset-state-key", type=str, default=None, help="Dataset state key override")
    parser.add_argument(
        "--mae-ratio",
        type=float,
        default=0.5,
        help="Fail if mean MAE exceeds this ratio of demo action range",
    )
    parser.add_argument(
        "--range-margin-ratio",
        type=float,
        default=0.2,
        help="Fail if pred range exceeds demo range by this ratio",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    success = test_client(args)

    if not success:
        print("\nTest failed. Check server status and input data.")
        sys.exit(1)
