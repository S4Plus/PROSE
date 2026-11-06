#!/usr/bin/env python3
"""
单视频端到端：视频 -> YOLO 检测 -> 简单关联 -> 按 failure.md 输出失败标签。

依赖：
  pip install ultralytics opencv-python
  (无网络环境请提前安装好 wheel)

用法示例：
  python video_label_failures.py /workspace/liushuo/cupid/outputs/eval_cli/2026-01-06/17-34-29/videos/eval_episode_1_fail.mp4 \
    --model yolov8n.pt \
    --object-cls 0 --gripper-cls 1 --hole-cls 2 \
    --outfile results.csv \
    --dump-json demo_tracks.json

若模型不含“hole”类别，可用 --hole-rect x1 y1 x2 y2 手动提供 ROI。

流程：
  1) 对每帧做检测（仅保留各类别得分最高的框）
  2) 生成与 failure_label_from_tracks.py 兼容的 JSON
  3) 直接调用其中的 detect_events 输出标签
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2

try:
    from ultralytics import YOLO
except Exception as e:  # pragma: no cover
    YOLO = None
    _yolo_import_error = e
else:
    _yolo_import_error = None

from failure_label_from_tracks import (
    detect_events,
    extract_series,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Video -> YOLO detections -> failure label (semi-auto).")
    p.add_argument("video", type=Path, help="输入视频路径")
    p.add_argument("--model", type=Path, required=True, help="YOLO/RT-DETR 权重文件 (.pt)")
    p.add_argument("--object-cls", type=int, required=True, help="物体类别 id")
    p.add_argument("--gripper-cls", type=int, required=True, help="夹爪类别 id")
    p.add_argument("--hole-cls", type=int, default=None, help="孔/ROI 类别 id，可空")
    p.add_argument("--hole-rect", type=float, nargs=4, default=None, help="手动孔 ROI: x1 y1 x2 y2（像素）")
    p.add_argument("--stride", type=int, default=1, help="帧抽样间隔，1 表示逐帧")
    p.add_argument("--conf", type=float, default=0.25, help="检测置信度阈值")
    p.add_argument("--outfile", type=Path, default=None, help="输出 CSV（含 label 与事件帧/时间）")
    p.add_argument("--dump-json", type=Path, default=None, help="如需保存检测结果 JSON，提供路径")

    # 事件检测阈值（透传给 detect_events）
    p.add_argument("--lift-rel", type=float, default=0.03)
    p.add_argument("--lift-min-frames", type=int, default=3)
    p.add_argument("--couple-rel", type=float, default=0.08)
    p.add_argument("--uncouple-rel", type=float, default=0.12)
    p.add_argument("--drop-vel-rel", type=float, default=0.04)
    p.add_argument("--insert-roi-expand", type=float, default=0.4)
    p.add_argument("--insert-min-frames", type=int, default=3)
    p.add_argument("--stuck-delta-rel", type=float, default=0.015)
    return p.parse_args()


def best_by_cls(result, cls_id: int) -> Optional[Tuple[List[float], float]]:
    """返回给定类别的最佳 bbox 与 score。"""
    if result.boxes is None:
        return None
    boxes = result.boxes
    mask = boxes.cls.int() == cls_id
    if mask.sum() == 0:
        return None
    scores = boxes.conf[mask]
    b = boxes.xyxy[mask]
    best_idx = int(scores.argmax())
    return b[best_idx].tolist(), float(scores[best_idx])


def run_detection(args: argparse.Namespace) -> Dict:
    if YOLO is None:
        raise SystemExit(f"未安装 ultralytics: {_yolo_import_error}")
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise SystemExit(f"无法打开视频: {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    model = YOLO(str(args.model))

    frames = []
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % args.stride != 0:
            frame_idx += 1
            continue
        # ultralytics 接受 BGR/np.ndarray
        res = model.predict(frame, verbose=False, conf=args.conf, imgsz=max(frame.shape[:2]))
        if not res:
            break
        det = res[0]
        objs = []
        for cls_id, label in [
            (args.object_cls, "object"),
            (args.gripper_cls, "gripper"),
            (args.hole_cls, "hole"),
        ]:
            if cls_id is None:
                continue
            best = best_by_cls(det, cls_id)
            if best:
                bbox, score = best
                objs.append({"label": label, "bbox": bbox, "score": score})
        frames.append({"frame_index": frame_idx, "objects": objs})
        frame_idx += 1

    cap.release()

    # 若未检测到 hole 且提供了 hole-rect，填充静态 ROI。
    if args.hole_rect and not any(
        any(o["label"] == "hole" for o in f["objects"]) for f in frames
    ):
        hole_bbox = [float(x) for x in args.hole_rect]
        for f in frames:
            f["objects"].append({"label": "hole", "bbox": hole_bbox, "score": 1.0})

    return {"fps": fps, "frames": frames}


def label_from_frames(data: Dict, args: argparse.Namespace) -> Dict[str, object]:
    label_map = {"object": "object", "gripper": "gripper", "hole": "hole"}
    obj_boxes, grip_boxes, hole_boxes, _, max_h = extract_series(data["frames"], label_map)
    img_h = max_h if max_h > 0 else 1.0
    events = detect_events(
        obj_boxes=obj_boxes,
        grip_boxes=grip_boxes,
        hole_boxes=hole_boxes,
        img_h=img_h,
        lift_rel=args.lift_rel,
        lift_min_frames=args.lift_min_frames,
        couple_rel=args.couple_rel,
        uncouple_rel=args.uncouple_rel,
        drop_vel_rel=args.drop_vel_rel,
        insert_expand=args.insert_roi_expand,
        insert_min_frames=args.insert_min_frames,
        stuck_delta_rel=args.stuck_delta_rel,
    )

    fps = float(data.get("fps") or 30.0)
    to_time = lambda f: (f / fps) if f is not None else ""
    return {
        "file": str(args.video),
        "label": events.label,
        "note": events.note,
        "lift_frame": events.lift_frame if events.lift_frame is not None else "",
        "drop_frame": events.drop_frame if events.drop_frame is not None else "",
        "insert_frame": events.insertion_start if events.insertion_start is not None else "",
        "lift_time": to_time(events.lift_frame),
        "drop_time": to_time(events.drop_frame),
        "insert_time": to_time(events.insertion_start),
    }


def main() -> None:
    args = parse_args()
    data = run_detection(args)
    if args.dump_json:
        args.dump_json.expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
        with args.dump_json.open("w") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"已写检测 JSON: {args.dump_json}")

    row = label_from_frames(data, args)
    print(json.dumps(row, ensure_ascii=False))

    if args.outfile:
        args.outfile.expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
        with args.outfile.open("w") as f:
            f.write("file,label,note,lift_frame,drop_frame,insert_frame,lift_time,drop_time,insert_time\n")
            f.write(
                f"{row['file']},{row['label']},{row['note']},"
                f"{row['lift_frame']},{row['drop_frame']},{row['insert_frame']},"
                f"{row['lift_time']},{row['drop_time']},{row['insert_time']}\n"
            )
        print(f"已写 CSV: {args.outfile}")


if __name__ == "__main__":
    main()
