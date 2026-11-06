#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import cv2
import numpy as np


def load_joint_names(dataset_root: Path) -> List[str]:
    info = json.loads((dataset_root / "meta" / "info.json").read_text())
    return info["features"]["action"]["names"][0]


def draw_legend_panel(
    panel: np.ndarray, lines: List[str], font_scale: float = 0.6
) -> np.ndarray:
    h, w = panel.shape[:2]
    margin = 10
    y = margin + 20
    color = (235, 235, 235)
    shadow = (20, 20, 20)
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 1

    for line in lines:
        if y > h - margin:
            break
        cv2.putText(panel, line, (margin + 1, y + 1), font, font_scale, shadow, 2)
        cv2.putText(panel, line, (margin, y), font, font_scale, color, thickness)
        y += int(22 * font_scale + 8)
    return panel


def annotate_video(
    video_path: Path,
    output_path: Path,
    joint_names: List[str],
    panel_width: int = 420,
) -> None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_w = width + panel_width
    out_h = height

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (out_w, out_h))

    lines = ["Joint index -> name"] + [
        f"{i:02d} {name}" for i, name in enumerate(joint_names)
    ]

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        panel = np.zeros((height, panel_width, 3), dtype=np.uint8)
        panel[:] = (35, 35, 35)
        panel = draw_legend_panel(panel, lines, font_scale=0.6)
        combined = np.concatenate([frame, panel], axis=1)
        writer.write(combined)

    cap.release()
    writer.release()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Annotate a video with joint legend on the right."
    )
    parser.add_argument(
        "--dataset",
        default="/workspace/liushuo/cupid/act/swk/six_object_pick_conveyor_first2",
        help="Dataset root (to read joint names).",
    )
    parser.add_argument("--video", required=True, help="Input video path.")
    parser.add_argument("--out", required=True, help="Output video path.")
    args = parser.parse_args()

    joint_names = load_joint_names(Path(args.dataset))
    annotate_video(Path(args.video), Path(args.out), joint_names)
    print(f"Saved annotated video to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
