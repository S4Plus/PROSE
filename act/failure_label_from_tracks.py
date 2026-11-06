#!/usr/bin/env python3
"""
Semi-automatic failure labeling using 2D detections/tracks, following failure.md.

Input format (JSON):
{
  "fps": 30,  # optional
  "frames": [
    {
      "frame_index": 0,               # or "frame"
      "objects": [
        {"label": "gripper", "bbox": [x1, y1, x2, y2], "score": 0.93},
        {"label": "object",  "bbox": [x1, y1, x2, y2]},
        {"label": "hole",    "bbox": [x1, y1, x2, y2]}
      ]
    },
    ...
  ]
}

Only 2D bboxes are required; the script applies simple temporal rules:
- Detect Lift: object bottom rises by a threshold for several frames.
- Detect Drop: gripper-object coupling breaks with a downward jump.
- Detect Insertion attempt: object enters the hole ROI.

Outputs a CSV with label + event frames for each tracks file:
  grasp_fail / drop_during_transport / insertion_drop_near_hole /
  insertion_stuck_on_top / success_guess / uncertain

This is a lightweight scaffold—you can swap in your own detector/tracker
as long as it emits the JSON above.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


BBox = Tuple[float, float, float, float]  # x1, y1, x2, y2


@dataclass
class Events:
    lift_frame: Optional[int]
    drop_frame: Optional[int]
    insertion_start: Optional[int]
    label: str
    note: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Label failure types from tracked bboxes (see failure.md).")
    p.add_argument("tracks", nargs="+", type=Path, help="Paths to JSON tracks (see docstring for format).")
    p.add_argument("--outfile", type=Path, default=None, help="CSV to write; default prints to stdout.")
    p.add_argument("--fps", type=float, default=None, help="FPS fallback if JSON lacks 'fps'.")
    p.add_argument("--object-label", default="object", help="Label name for the target object.")
    p.add_argument("--gripper-label", default="gripper", help="Label name for the gripper/end-effector.")
    p.add_argument("--hole-label", default="hole", help="Label name/ROI for insertion region.")

    # Thresholds are relative to image height; tweak per dataset.
    p.add_argument("--lift-rel", type=float, default=0.03, help="Relative upward motion to count as Lift (fraction of height).")
    p.add_argument("--lift-min-frames", type=int, default=3, help="Consecutive frames needed to accept Lift.")
    p.add_argument("--couple-rel", type=float, default=0.08, help="Gripper-object distance (rel height) treated as coupled.")
    p.add_argument("--uncouple-rel", type=float, default=0.12, help="Distance treated as uncoupled (hysteresis > couple-rel).")
    p.add_argument("--drop-vel-rel", type=float, default=0.04, help="Downward delta (rel height) to flag a drop.")
    p.add_argument("--insert-roi-expand", type=float, default=0.4, help="Expand hole bbox by this fraction on each side.")
    p.add_argument("--insert-min-frames", type=int, default=3, help="Frames inside hole ROI to mark insertion attempt.")
    p.add_argument("--stuck-delta-rel", type=float, default=0.015, help="Vertical progress after insertion below this => stuck.")
    return p.parse_args()


def load_tracks(path: Path) -> Dict:
    with path.open("r") as f:
        data = json.load(f)
    if not isinstance(data, dict) or "frames" not in data:
        raise ValueError(f"{path} missing top-level 'frames'")
    return data


def bbox_center(b: BBox) -> Tuple[float, float]:
    return ((b[0] + b[2]) * 0.5, (b[1] + b[3]) * 0.5)


def bbox_bottom(b: BBox) -> float:
    return b[3]


def bbox_size(b: BBox) -> Tuple[float, float]:
    return (b[2] - b[0], b[3] - b[1])


def dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def pick_best(dets: List[Dict]) -> Optional[Dict]:
    if not dets:
        return None
    return max(dets, key=lambda d: d.get("score", 0.0))


def extract_series(
    frames: Sequence[Dict],
    label_map: Dict[str, str],
) -> Tuple[List[Optional[BBox]], List[Optional[BBox]], List[Optional[BBox]], float, float]:
    """Return per-frame bbox series for object/gripper/hole + max width/height seen."""
    max_w = 0.0
    max_h = 0.0
    by_frame: Dict[int, Dict[str, Optional[BBox]]] = {}
    for f in frames:
        idx = int(f.get("frame_index") or f.get("frame") or f.get("frame_id") or 0)
        dets = f.get("objects") or f.get("detections") or []
        slot = by_frame.setdefault(idx, {"object": None, "gripper": None, "hole": None})
        bucket: Dict[str, List[Dict]] = {"object": [], "gripper": [], "hole": []}
        for d in dets:
            label = d.get("label") or d.get("name") or d.get("cls")
            mapped = label_map.get(str(label))
            if mapped is None:
                continue
            bucket[mapped].append(d)
        for key in ("object", "gripper", "hole"):
            chosen = pick_best(bucket[key])
            if chosen and "bbox" in chosen:
                b = tuple(float(x) for x in chosen["bbox"])
                slot[key] = (b[0], b[1], b[2], b[3])
                max_w = max(max_w, b[2])
                max_h = max(max_h, b[3])
    if not by_frame:
        raise ValueError("No frames parsed from detections.")
    max_idx = max(by_frame.keys())
    objs: List[Optional[BBox]] = []
    grps: List[Optional[BBox]] = []
    holes: List[Optional[BBox]] = []
    for i in range(max_idx + 1):
        slot = by_frame.get(i, {"object": None, "gripper": None, "hole": None})
        objs.append(slot["object"])
        grps.append(slot["gripper"])
        holes.append(slot["hole"])
    return objs, grps, holes, max_w, max_h


def moving_average(seq: List[Optional[float]], window: int = 3) -> List[Optional[float]]:
    out: List[Optional[float]] = []
    buf: List[float] = []
    for v in seq:
        if v is None:
            buf.clear()
            out.append(None)
            continue
        buf.append(v)
        if len(buf) > window:
            buf.pop(0)
        out.append(sum(buf) / len(buf))
    return out


def detect_events(
    obj_boxes: List[Optional[BBox]],
    grip_boxes: List[Optional[BBox]],
    hole_boxes: List[Optional[BBox]],
    img_h: float,
    lift_rel: float,
    lift_min_frames: int,
    couple_rel: float,
    uncouple_rel: float,
    drop_vel_rel: float,
    insert_expand: float,
    insert_min_frames: int,
    stuck_delta_rel: float,
) -> Events:
    obj_centers = [bbox_center(b) if b else None for b in obj_boxes]
    grip_centers = [bbox_center(b) if b else None for b in grip_boxes]
    obj_bottoms = [bbox_bottom(b) if b else None for b in obj_boxes]
    hole_boxes_full = hole_boxes

    # Smooth bottoms to reduce flicker noise.
    obj_bottoms_smooth = moving_average(obj_bottoms, window=3)

    # Coupling based on center distance.
    couple_thresh = couple_rel * img_h
    uncouple_thresh = uncouple_rel * img_h
    coupled: List[bool] = []
    for oc, gc in zip(obj_centers, grip_centers):
        if oc is None or gc is None:
            coupled.append(False)
            continue
        d = dist(oc, gc)
        prev = coupled[-1] if coupled else False
        if prev and d < uncouple_thresh:
            coupled.append(True)
        elif d < couple_thresh:
            coupled.append(True)
        else:
            coupled.append(False)

    # Lift detection: bottom rises (y decreases) for several frames.
    baseline_bottoms = [b for b in obj_bottoms_smooth[:10] if b is not None]
    base = sum(baseline_bottoms) / len(baseline_bottoms) if baseline_bottoms else None
    lift_thresh = lift_rel * img_h
    lift_frame = None
    if base is not None:
        streak = 0
        for i, b in enumerate(obj_bottoms_smooth):
            if b is None:
                streak = 0
                continue
            if (base - b) >= lift_thresh:
                streak += 1
                if streak >= lift_min_frames:
                    lift_frame = i - streak + 1
                    break
            else:
                streak = 0

    # Drop: coupled -> uncoupled with downward jump.
    drop_frame = None
    drop_vel_thresh = drop_vel_rel * img_h
    prev_b = None
    for i, (c, b) in enumerate(zip(coupled, obj_bottoms_smooth)):
        if b is None or prev_b is None:
            prev_b = b
            continue
        dy = b - prev_b  # positive => moving down
        uncoupled_now = not c
        coupled_prev = coupled[i - 1] if i > 0 else False
        if coupled_prev and uncoupled_now and dy > drop_vel_thresh:
            drop_frame = i
            break
        prev_b = b

    # Insertion attempt: object enters expanded hole ROI.
    insert_roi_frames: List[int] = []
    for i, (obj, hole) in enumerate(zip(obj_boxes, hole_boxes_full)):
        if obj is None or hole is None:
            insert_roi_frames.append(0)
            continue
        hx1, hy1, hx2, hy2 = hole
        hw, hh = bbox_size(hole)
        expand_x = hw * insert_expand
        expand_y = hh * insert_expand
        roi = (hx1 - expand_x, hy1 - expand_y, hx2 + expand_x, hy2 + expand_y)
        oc = bbox_center(obj)
        inside = (roi[0] <= oc[0] <= roi[2]) and (roi[1] <= oc[1] <= roi[3])
        insert_roi_frames.append(1 if inside else 0)

    insertion_start = None
    streak = 0
    for i, inside in enumerate(insert_roi_frames):
        if inside:
            streak += 1
            if streak >= insert_min_frames:
                insertion_start = i - streak + 1
                break
        else:
            streak = 0

    # Heuristic "stuck on top": little downward progress after insertion start.
    stuck = False
    if insertion_start is not None:
        start_bottom = obj_bottoms_smooth[insertion_start]
        if start_bottom is not None:
            post = [b for b in obj_bottoms_smooth[insertion_start + 1 :] if b is not None]
            if post:
                min_after = min(post)
                progress = start_bottom - min_after  # upward progress; negative => went down
                stuck = progress < (stuck_delta_rel * img_h)

    # Classify.
    label = "uncertain"
    note = ""
    if lift_frame is None:
        label = "grasp_fail"
        note = "no_lift_detected"
    elif drop_frame is not None and (insertion_start is None or drop_frame < insertion_start):
        label = "drop_during_transport"
        note = f"drop_frame={drop_frame}"
    elif insertion_start is not None:
        if drop_frame is not None and drop_frame >= insertion_start:
            label = "insertion_drop_near_hole"
            note = f"drop_frame={drop_frame}"
        elif stuck:
            label = "insertion_stuck_on_top"
            note = "little_height_progress"
        else:
            label = "success_guess"
            note = "insertion_no_drop"
    elif drop_frame is not None:
        label = "drop_during_transport"
        note = f"drop_frame={drop_frame}"
    else:
        label = "uncertain"
        note = "no_insert_and_no_drop"

    return Events(
        lift_frame=lift_frame,
        drop_frame=drop_frame,
        insertion_start=insertion_start,
        label=label,
        note=note,
    )


def process_file(path: Path, args: argparse.Namespace) -> Dict[str, object]:
    data = load_tracks(path)
    fps = float(data.get("fps") or args.fps or 30.0)
    label_map = {
        args.object_label: "object",
        args.gripper_label: "gripper",
        args.hole_label: "hole",
    }
    obj_boxes, grip_boxes, hole_boxes, max_w, max_h = extract_series(data["frames"], label_map)
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
    to_time = lambda f: (f / fps) if f is not None else ""
    return {
        "file": str(path),
        "label": events.label,
        "note": events.note,
        "lift_frame": events.lift_frame if events.lift_frame is not None else "",
        "drop_frame": events.drop_frame if events.drop_frame is not None else "",
        "insert_frame": events.insertion_start if events.insertion_start is not None else "",
        "lift_time": to_time(events.lift_frame),
        "drop_time": to_time(events.drop_frame),
        "insert_time": to_time(events.insertion_start),
    }


def write_csv(rows: List[Dict[str, object]], outfile: Path) -> None:
    if not rows:
        return
    outfile.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "file",
        "label",
        "note",
        "lift_frame",
        "drop_frame",
        "insert_frame",
        "lift_time",
        "drop_time",
        "insert_time",
    ]
    with outfile.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {outfile}")


def main() -> None:
    args = parse_args()
    rows = []
    for p in args.tracks:
        try:
            row = process_file(p, args)
        except Exception as e:
            row = {
                "file": str(p),
                "label": "error",
                "note": str(e),
                "lift_frame": "",
                "drop_frame": "",
                "insert_frame": "",
                "lift_time": "",
                "drop_time": "",
                "insert_time": "",
            }
        rows.append(row)

    if args.outfile:
        write_csv(rows, args.outfile.expanduser().resolve())
    else:
        for r in rows:
            print(json.dumps(r, ensure_ascii=False))


if __name__ == "__main__":
    main()
