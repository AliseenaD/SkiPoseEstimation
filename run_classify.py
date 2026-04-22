#!/usr/bin/env python3
"""
run_classify.py

Headless runner called by the Vite dev server plugin.
Writes an annotated video to --output-path and prints a JSON result to stdout.

Usage:
  python run_classify.py --input <video.mp4> --output-path <output/classified_x.mp4>

Must be run from the SkiProject root directory.
"""

import os
import sys
import json
import argparse
from pathlib import Path

# classify_video.py uses relative paths (data/, output/, yolov8n.pt) that are
# all relative to the model_logic/ directory, so we switch there before any
# of that code runs. Imports are resolved from the project root (sys.path),
# which is set by the caller, so they are unaffected by the chdir.
_PROJECT_ROOT = Path(__file__).parent.resolve()
os.chdir(_PROJECT_ROOT / "model_logic")

import cv2
import numpy as np

from model_logic.classify_video import _extract_and_infer
from model_logic.pipeline_utils import draw_skeleton


def write_annotated_video(
    video_path: Path,
    all_lm_coords: dict,
    feature_arr: np.ndarray,
    is_interp: np.ndarray,
    bbox_per_frame: dict,
    fps: float,
    vid_w: int,
    vid_h: int,
    output_path: Path,
) -> None:
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (vid_w, vid_h),
    )
    cap = cv2.VideoCapture(str(video_path))
    frame_i = 0
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        feat = feature_arr[frame_i] if feature_arr is not None and frame_i < len(feature_arr) else None
        has_feat = feat is not None and not np.isnan(feat).any()
        if has_feat:
            colour = (0, 165, 255) if is_interp[frame_i] else (0, 220, 0)
            draw_skeleton(frame, all_lm_coords.get(frame_i, {}), colour)
        if frame_i in bbox_per_frame:
            bx1, by1, bx2, by2 = bbox_per_frame[frame_i]
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), (255, 100, 0), 2)
        writer.write(frame)
        frame_i += 1
    cap.release()
    writer.release()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to the input ski video")
    parser.add_argument("--output-path", required=True, help="Path to write the annotated video")
    args = parser.parse_args()

    video_path = Path(args.input)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    result, all_lm_coords, feature_arr, is_interp, bbox_per_frame, fps, vid_w, vid_h, _ = \
        _extract_and_infer(video_path)

    if result.get("error"):
        print(json.dumps({"error": result["error"]}), flush=True)
        sys.exit(1)

    write_annotated_video(
        video_path, all_lm_coords, feature_arr, is_interp,
        bbox_per_frame, fps, vid_w, vid_h, output_path,
    )

    # Print JSON last so the plugin can find it after ML library noise
    print(json.dumps({
        "level":       result["predicted_class"],
        "confidence":  result["confidence"],
        "tips":        result["tips"],
        "class_probs": result["class_probs"],
        "n_windows":   result["n_windows"],
    }), flush=True)


if __name__ == "__main__":
    main()
