#!/usr/bin/env python3
"""
test_pipeline.py

Two-pass pipeline on a single video:
  Pass 1 – run YOLO/CSRT/MediaPipe, collect valid frame indices + features
            then interpolate gaps (same logic as extract_features.py)
  Pass 2 – re-read the video and render an annotated output

Skeleton colours in the output:
  Green  – frame passed visibility check (directly detected)
  Orange – feature values were interpolated from neighbouring valid frames
  Grey   – frame could not be recovered (NaN); labelled "dropped"

Usage:
    python test_pipeline.py                           # default video below
    python test_pipeline.py data/beginner/beginner_1.mp4
"""

import sys
import cv2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from ultralytics import YOLO
import numpy as np
from pathlib import Path

from model_logic.pipeline_utils import (
    YOLO_INTERVAL, MAX_INTERPOLATION_GAP, FEATURE_NAMES,
    create_csrt_tracker, largest_person_box, select_subject,
    process_frame, interpolate_gaps,
    interpolate_landmark_coords, draw_skeleton,
)

DEFAULT_VIDEO = "data/intermediate/intermediate_7.mp4"

FEATURE_UNITS = ["deg", "x", ""] + [""] * 24

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Model init ──────────────────────────────────────────────────────────────────
yolo = YOLO("yolov8n.pt")

base_options = python.BaseOptions(model_asset_path="data/pose_landmarker_heavy.task")
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    output_segmentation_masks=False,
    min_pose_detection_confidence=0.5,
    min_pose_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)
detector = vision.PoseLandmarker.create_from_options(options)


# ── Pass 1: collect detections ─────────────────────────────────────────────────
def collect_detections(video_path: Path):
    """
    Pass 1: run YOLO/CSRT/MediaPipe on every frame and collect raw detections.

    Returns (valid_indices, valid_features, landmark_coords, bbox_per_frame,
             total_frames, fps, width, height).
    """
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    valid_indices, valid_features = [], []
    landmark_coords = {}
    bbox_per_frame  = {}

    success, first_frame = cap.read()
    if not success:
        cap.release()
        return np.array([], dtype=np.int32), np.array([], dtype=np.float32), {}, {}, 0, fps, w, h

    fh, fw   = first_frame.shape[:2]
    yolo_out = yolo(first_frame, classes=[0], verbose=False)
    boxes    = yolo_out[0].boxes
    n_people = len(boxes)

    if n_people == 1:
        x1, y1, x2, y2 = largest_person_box(boxes)
        print("  1 person detected on frame 0 — auto-selected.")
    else:
        print(f"  {n_people} {'people' if n_people != 1 else 'person'} detected on frame 0 "
              f"— opening selection window...")
        x1, y1, x2, y2 = select_subject(first_frame, boxes)

    tracker = create_csrt_tracker()
    tracker.init(first_frame, (x1, y1, x2 - x1, y2 - y1))

    frame_count = 1
    bbox_per_frame[0] = (x1, y1, x2, y2)
    process_frame(first_frame, 0, x1, y1, x2, y2, fw, fh,
                  detector, valid_indices, valid_features, landmark_coords)

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break

        fh, fw = image.shape[:2]
        idx     = frame_count
        frame_count += 1

        run_yolo = (frame_count % YOLO_INTERVAL == 1)

        if not run_yolo:
            ok, bbox = tracker.update(image)
            if ok:
                tx, ty, tw, th = (int(v) for v in bbox)
                x1, y1, x2, y2 = tx, ty, tx + tw, ty + th
            else:
                run_yolo = True

        if run_yolo:
            yolo_out = yolo(image, classes=[0], verbose=False)
            box = largest_person_box(yolo_out[0].boxes) if len(yolo_out[0].boxes) > 0 else None
            if box:
                x1, y1, x2, y2 = box
                tracker = create_csrt_tracker()
                tracker.init(image, (x1, y1, x2 - x1, y2 - y1))
            else:
                x1, y1, x2, y2 = 0, 0, fw, fh

        bbox_per_frame[idx] = (x1, y1, x2, y2)
        process_frame(image, idx, x1, y1, x2, y2, fw, fh,
                      detector, valid_indices, valid_features, landmark_coords)

    cap.release()
    return (
        np.array(valid_indices,  dtype=np.int32),
        np.array(valid_features, dtype=np.float32),
        landmark_coords,
        bbox_per_frame,
        frame_count,
        fps, w, h,
    )


# ── Pass 2: render annotated video ─────────────────────────────────────────────
def render_video(video_path, out_path, feature_arr, is_interpolated,
                 all_landmark_coords, bbox_per_frame, fps, w, h):
    """
    Pass 2: re-read the video and write an annotated mp4 with skeleton overlays and per-frame status labels.

    Returns (n_detected, n_interpolated, n_dropped) frame counts.
    """
    cap    = cv2.VideoCapture(str(video_path))
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    frame_count = 0
    n_detected = n_interp = n_dropped = 0

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break

        idx = frame_count
        frame_count += 1

        if idx in bbox_per_frame:
            bx1, by1, bx2, by2 = bbox_per_frame[idx]
            cv2.rectangle(image, (bx1, by1), (bx2, by2), (255, 100, 0), 2)

        feat     = feature_arr[idx] if idx < len(feature_arr) else None
        has_feat = feat is not None and not np.isnan(feat).any()

        if has_feat:
            if is_interpolated[idx]:
                colour = (0, 165, 255)   # orange — interpolated
                status = "interpolated"
                n_interp += 1
            else:
                colour = (0, 220, 0)     # green — directly detected
                status = "OK"
                n_detected += 1

            draw_skeleton(image, all_landmark_coords.get(idx, {}), colour)

            for j, (name, unit, val) in enumerate(zip(FEATURE_NAMES[:3], FEATURE_UNITS[:3], feat[:3])):
                suffix = f" {unit}" if unit else ""
                cv2.putText(image, f"{name}: {val:.2f}{suffix}", (12, 30 + j * 26),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        else:
            status = "dropped"
            n_dropped += 1

        cv2.putText(image, f"frame {frame_count}  [{status}]", (12, h - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1, cv2.LINE_AA)

        writer.write(image)

    cap.release()
    writer.release()
    return n_detected, n_interp, n_dropped


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    """CLI entry point: run both pipeline passes on one video and print a detection summary."""
    video_path = Path(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_VIDEO)
    if not video_path.exists():
        print(f"Video not found: {video_path}")
        sys.exit(1)

    out_path = OUTPUT_DIR / "test_output_interp.mp4"
    print(f"Processing: {video_path.name}")
    print(f"Output:     {out_path}")

    print("Pass 1/2 — detection...")
    valid_idx, valid_feat, lm_coords, bbox_map, total, fps, w, h = collect_detections(video_path)

    feat_arr, is_interp = interpolate_gaps(valid_idx, valid_feat, total)
    is_nan = np.isnan(feat_arr[:, 0])
    all_lm_coords = interpolate_landmark_coords(lm_coords, valid_idx, total, is_nan)

    n_interp_frames = int(is_interp.sum())
    print(f"  {len(valid_idx)} detected  +  {n_interp_frames} interpolated  "
          f"(gaps ≤ {MAX_INTERPOLATION_GAP} frames filled)")

    print("Pass 2/2 — rendering...")
    n_det, n_int, n_drop = render_video(
        video_path, out_path, feat_arr, is_interp, all_lm_coords, bbox_map, fps, w, h
    )

    print(f"\nDone.")
    print(f"  Green  (detected):     {n_det}")
    print(f"  Orange (interpolated): {n_int}")
    print(f"  Grey   (dropped):      {n_drop}")
    print(f"  Saved: {out_path}")


if __name__ == "__main__":
    main()
