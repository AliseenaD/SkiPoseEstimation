#!/usr/bin/env python3
"""
extract_features.py

Pipeline per video:
  YOLO largest-box detection → CSRT tracker → MediaPipe pose →
  visibility filter → gap interpolation → temporal smoothing →
  sliding window

Features per frame (27 total):
  Computed (3):
    upper_lower_rotation  – angle between shoulder axis and hip axis
    stance_width          – ankle spread normalised by hip width
    arm_height            – mean wrist height above hip line, normalised by torso height
  Body-centred relative positions (24):
    (x, y) of each of 12 landmarks relative to hip midpoint, normalised by torso height

On the first frame of each video, an interactive subject-selection prompt
appears in the terminal whenever YOLO detects more than one person.

Dropped frames are recovered via linear interpolation from the nearest valid
neighbours, provided the gap is ≤ MAX_INTERPOLATION_GAP.  Larger gaps and
leading/trailing edges are left as NaN and excluded from sliding windows.

Outputs (written to ./output/):
  X.npy          (N, WINDOW_SIZE, 27) float32  – LSTM input
  y.npy          (N,)                 int64    – class labels 0/1/2
  features.csv                                 – one row per non-NaN frame
"""

import cv2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from ultralytics import YOLO
import numpy as np
import pandas as pd
from pathlib import Path

from model_logic.pipeline_utils import (
    YOLO_INTERVAL, WINDOW_SIZE, FEATURE_NAMES,
    create_csrt_tracker, largest_person_box, select_subject,
    process_frame, interpolate_gaps, smooth, sliding_windows,
)

# ── Config ──────────────────────────────────────────────────────────────────────
DATA_DIR   = Path("data")
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

LABEL_MAP  = {"beginner": 0, "intermediate": 1, "advanced": 2}

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


# ── Per-video processing ─────────────────────────────────────────────────────────
def process_video(video_path: Path):
    """
    Returns (valid_indices, valid_features, total_frames).

    On the first frame, shows an interactive subject-selection prompt whenever
    YOLO detects more than one person.  Single-person frames auto-confirm.
    """
    cap = cv2.VideoCapture(str(video_path))
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    indices, features = [], []

    success, first_frame = cap.read()
    if not success:
        cap.release()
        return np.array([], dtype=np.int32), np.array([], dtype=np.float32), 0

    fh, fw   = first_frame.shape[:2]
    yolo_out = yolo(first_frame, classes=[0], verbose=False)
    boxes    = yolo_out[0].boxes
    n_people = len(boxes)

    if n_people == 1:
        x1, y1, x2, y2 = largest_person_box(boxes)
    else:
        print(f"    {n_people} {'people' if n_people != 1 else 'person'} detected — opening selection window...")
        x1, y1, x2, y2 = select_subject(first_frame, boxes)

    tracker = create_csrt_tracker()
    tracker.init(first_frame, (x1, y1, x2 - x1, y2 - y1))

    frame_count = 1
    process_frame(first_frame, 0, x1, y1, x2, y2, fw, fh, detector, indices, features)

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

        process_frame(image, idx, x1, y1, x2, y2, fw, fh, detector, indices, features)

    cap.release()
    return (
        np.array(indices,  dtype=np.int32),
        np.array(features, dtype=np.float32),
        frame_count,
    )


# ── Main ─────────────────────────────────────────────────────────────────────────
def main():
    """Iterate over all labelled videos, extract sliding-window features, and save X.npy / y.npy / features.csv."""
    all_X, all_y = [], []
    csv_rows = []

    for level, label in LABEL_MAP.items():
        video_dir = DATA_DIR / level
        videos    = sorted(video_dir.glob("*.mp4"))
        print(f"\n[{level}]  {len(videos)} videos")

        for vid in videos:
            print(f"  {vid.name} ...", end=" ", flush=True)

            valid_idx, valid_feat, total = process_video(vid)
            arr, is_interp = interpolate_gaps(valid_idx, valid_feat, total)
            arr = smooth(arr)

            usable_mask = ~np.isnan(arr[:, 0])
            n_usable    = usable_mask.sum()

            if n_usable < WINDOW_SIZE:
                print(f"skipped  ({n_usable} usable frames, need ≥ {WINDOW_SIZE})")
                continue

            for i in np.where(usable_mask)[0]:
                csv_rows.append({
                    "video":        vid.name,
                    "level":        level,
                    "label_id":     label,
                    "frame_idx":    int(i),
                    "interpolated": bool(is_interp[i]),
                    **dict(zip(FEATURE_NAMES, arr[i].tolist())),
                })

            wins = list(sliding_windows(arr))
            for w in wins:
                all_X.append(w)
                all_y.append(label)

            n_interp = int(is_interp.sum())
            print(
                f"{len(valid_idx)} detected  +  {n_interp} interpolated"
                f"  =  {n_usable} usable frames  →  {len(wins)} windows"
            )

    X = np.array(all_X, dtype=np.float32)
    y = np.array(all_y, dtype=np.int64)

    np.save(OUTPUT_DIR / "X.npy", X)
    np.save(OUTPUT_DIR / "y.npy", y)

    df = pd.DataFrame(csv_rows)
    df.to_csv(OUTPUT_DIR / "features.csv", index=False)

    print(f"\nDone.")
    print(f"  X.npy        {X.shape}   (samples × frames × features)")
    print(f"  y.npy        {y.shape}")
    print(f"  features.csv {len(df)} frame rows  "
          f"({df['interpolated'].sum()} interpolated, "
          f"{(~df['interpolated']).sum()} detected)")
    print(f"\n  Class distribution in windows:")
    for lv, lb in LABEL_MAP.items():
        print(f"    {lv:12s}  {int((y == lb).sum())} windows")


if __name__ == "__main__":
    main()
