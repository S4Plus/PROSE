#!/usr/bin/env python3
"""
用 LeRobot 数据集样本直接请求 OpenVLA server，并与数据集真实 action 对比。

说明：
- 该脚本不做客户端动作后处理，直接比较服务端原始输出与数据集 action。
- 建议在 `openvla-oft` 环境中运行，因为需要 `pyarrow` 和 `imageio`。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import imageio.v2 as imageio
import msgpack
import numpy as np
import pyarrow.parquet as pq
from PIL import Image
import websockets.sync.client as ws_client


def _pack_array(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        if obj.dtype.kind in ("V", "O", "c"):
            raise ValueError(f"Unsupported dtype: {obj.dtype}")
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


def _unpack_array(obj: Dict[bytes, Any]) -> Any:
    if b"__ndarray__" in obj:
        return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"])
    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])
    return obj


def _pack_msg(data: Dict[str, Any]) -> bytes:
    return msgpack.packb(data, default=_pack_array)


def _unpack_msg(data: bytes) -> Dict[str, Any]:
    obj = msgpack.unpackb(data, object_hook=_unpack_array)
    if not isinstance(obj, dict):
        raise TypeError(f"Unexpected response type: {type(obj)}")
    return obj


def _normalize_ws_uri(url: str) -> str:
    raw = str(url).strip()
    if raw.startswith("ws://") or raw.startswith("wss://"):
        uri = raw
    elif raw.startswith("http://"):
        uri = "ws://" + raw[len("http://") :]
    elif raw.startswith("https://"):
        uri = "wss://" + raw[len("https://") :]
    else:
        uri = "ws://" + raw
    for suffix in ("/health", "/healthz", "/reset", "/act", "/infer"):
        if uri.endswith(suffix):
            uri = uri[: -len(suffix)]
            break
    return uri


def _transform_image(image: np.ndarray, rotate_180: bool, flip_vertical: bool, flip_horizontal: bool) -> np.ndarray:
    out = np.asarray(image)
    if rotate_180:
        out = out[::-1, ::-1]
    if flip_vertical:
        out = np.flipud(out)
    if flip_horizontal:
        out = np.fliplr(out)
    return np.ascontiguousarray(out)


def _load_task_map(dataset_root: Path) -> Dict[int, str]:
    task_map: Dict[int, str] = {}
    tasks_path = dataset_root / "meta" / "tasks.jsonl"
    with tasks_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            task_map[int(item["task_index"])] = str(item["task"])
    return task_map


def _read_episode_table(dataset_root: Path, episode_index: int) -> Dict[str, List[Any]]:
    parquet_path = dataset_root / "data" / "chunk-000" / f"episode_{episode_index:06d}.parquet"
    table = pq.read_table(parquet_path)
    return table.to_pydict()


def _video_path(dataset_root: Path, key: str, episode_index: int) -> Path:
    return dataset_root / "videos" / "chunk-000" / key / f"episode_{episode_index:06d}.mp4"


def _read_video_frame(video_path: Path, frame_index: int) -> np.ndarray:
    reader = imageio.get_reader(str(video_path))
    try:
        frame = np.asarray(reader.get_data(int(frame_index)), dtype=np.uint8)
    finally:
        reader.close()
    return frame


def _save_image(path: Path, image: np.ndarray) -> None:
    Image.fromarray(np.asarray(image, dtype=np.uint8), mode="RGB").save(path)


def _parse_frame_indices(frame_indices: str) -> List[int]:
    items = []
    for part in str(frame_indices).split(","):
        part = part.strip()
        if not part:
            continue
        items.append(int(part))
    if not items:
        raise ValueError("frame_indices 不能为空")
    return items


def _future_gt_chunk(actions: List[Any], start: int, chunk_size: int) -> np.ndarray:
    end = min(start + chunk_size, len(actions))
    rows = [np.asarray(actions[i], dtype=np.float32).reshape(-1) for i in range(start, end)]
    return np.stack(rows, axis=0)


def _summarize_diff(pred: np.ndarray, gt: np.ndarray) -> Dict[str, Any]:
    common = min(pred.shape[0], gt.shape[0])
    pred = pred[:common]
    gt = gt[:common]
    diff = np.abs(pred - gt)
    return {
        "steps_compared": int(common),
        "mean_abs_diff": float(diff.mean()),
        "max_abs_diff": float(diff.max()),
        "first_step_pred": pred[0].tolist(),
        "first_step_gt": gt[0].tolist(),
        "first_step_abs_diff": np.abs(pred[0] - gt[0]).tolist(),
    }


def _infer_once(
    server_url: str,
    prompt: str,
    state: np.ndarray,
    agentview: np.ndarray,
    wrist: np.ndarray,
    chunk_size: int,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    uri = _normalize_ws_uri(server_url)
    with ws_client.connect(uri, compression=None, max_size=None) as ws:
        meta = ws.recv()
        metadata = _unpack_msg(meta) if isinstance(meta, (bytes, bytearray)) else {}
        payload = {
            "images": {
                "observation.images.agentview": np.asarray(agentview, dtype=np.uint8),
                "observation.images.wrist": np.asarray(wrist, dtype=np.uint8),
            },
            "state": np.asarray(state, dtype=np.float32).reshape(-1),
            "prompt": str(prompt),
            "chunk_size": int(chunk_size),
        }
        ws.send(_pack_msg(payload))
        response = ws.recv()
        if isinstance(response, str):
            raise RuntimeError(f"Server error:\n{response}")
        result = _unpack_msg(response)
        return metadata, result


def main() -> None:
    parser = argparse.ArgumentParser(description="对 LeRobot 数据集样本做 OpenVLA server sanity check。")
    parser.add_argument("--server_url", type=str, default="ws://127.0.0.1:8000")
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--episode_index", type=int, default=0)
    parser.add_argument("--frame_indices", type=str, default="0,10,20")
    parser.add_argument("--chunk_size", type=int, default=None)
    parser.add_argument("--prompt", type=str, default=None, help="若不传，则使用 tasks.jsonl 对应 task。")
    parser.add_argument("--rotate_180", action="store_true", default=False)
    parser.add_argument("--flip_vertical", action="store_true", default=False)
    parser.add_argument("--flip_horizontal", action="store_true", default=False)
    parser.add_argument("--output_dir", type=str, default=None, help="可选输出目录，保存输入图和结果 json。")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root).resolve()
    task_map = _load_task_map(dataset_root)
    episode = _read_episode_table(dataset_root, args.episode_index)
    frame_rows = len(episode["frame_index"])
    frame_indices = _parse_frame_indices(args.frame_indices)

    output_dir = Path(args.output_dir).resolve() if args.output_dir else None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    agentview_video = _video_path(dataset_root, "observation.images.agentview", args.episode_index)
    wrist_video = _video_path(dataset_root, "observation.images.robot0_eye_in_hand", args.episode_index)

    metadata_printed = False
    all_results: List[Dict[str, Any]] = []

    for row_idx in frame_indices:
        if row_idx < 0 or row_idx >= frame_rows:
            raise IndexError(f"frame row {row_idx} 超出范围，当前 episode 共 {frame_rows} 帧")

        dataset_frame_index = int(episode["frame_index"][row_idx])
        state = np.asarray(episode["observation.state"][row_idx], dtype=np.float32).reshape(-1)
        task_index = int(episode["task_index"][row_idx])
        prompt = args.prompt or task_map[task_index]

        agentview = _transform_image(
            _read_video_frame(agentview_video, dataset_frame_index),
            rotate_180=args.rotate_180,
            flip_vertical=args.flip_vertical,
            flip_horizontal=args.flip_horizontal,
        )
        wrist = _transform_image(
            _read_video_frame(wrist_video, dataset_frame_index),
            rotate_180=args.rotate_180,
            flip_vertical=args.flip_vertical,
            flip_horizontal=args.flip_horizontal,
        )

        metadata, result = _infer_once(
            server_url=args.server_url,
            prompt=prompt,
            state=state,
            agentview=agentview,
            wrist=wrist,
            chunk_size=int(args.chunk_size or 8),
        )
        if not metadata_printed:
            print("server_metadata:")
            print(json.dumps(metadata, ensure_ascii=False, indent=2))
            metadata_printed = True

        action_raw = result.get("action", result.get("actions", result))
        pred = np.asarray(action_raw, dtype=np.float32)
        if pred.ndim == 1:
            pred = pred.reshape(1, -1)

        gt = _future_gt_chunk(episode["action"], row_idx, pred.shape[0])
        summary = _summarize_diff(pred=pred, gt=gt)
        item = {
            "episode_index": int(args.episode_index),
            "row_index": int(row_idx),
            "dataset_frame_index": dataset_frame_index,
            "prompt": prompt,
            "state": state.tolist(),
            "server_timing": result.get("server_timing"),
            "prediction_summary": summary,
        }
        all_results.append(item)

        print(f"\nframe_row={row_idx} dataset_frame={dataset_frame_index}")
        print(f"prompt: {prompt}")
        print("pred_first_step:", np.round(pred[0], 6).tolist())
        print("gt_first_step:  ", np.round(gt[0], 6).tolist())
        print("abs_diff_first: ", np.round(np.abs(pred[0] - gt[0]), 6).tolist())
        print(
            "chunk_diff: steps_compared={} mean_abs_diff={:.6f} max_abs_diff={:.6f}".format(
                summary["steps_compared"],
                summary["mean_abs_diff"],
                summary["max_abs_diff"],
            )
        )

        if output_dir is not None:
            prefix = output_dir / f"ep{args.episode_index:06d}_row{row_idx:04d}"
            _save_image(prefix.with_name(prefix.name + "_agentview.png"), agentview)
            _save_image(prefix.with_name(prefix.name + "_wrist.png"), wrist)
            np.save(prefix.with_name(prefix.name + "_state.npy"), state)
            np.save(prefix.with_name(prefix.name + "_pred.npy"), pred)
            np.save(prefix.with_name(prefix.name + "_gt.npy"), gt)
            prefix.with_name(prefix.name + "_summary.json").write_text(
                json.dumps(item, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    if output_dir is not None:
        (output_dir / "all_results.json").write_text(
            json.dumps(all_results, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
