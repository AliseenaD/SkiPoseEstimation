#!/usr/bin/env python3
"""
classify_video.py

Accepts a ski video, runs it through the full feature extraction pipeline,
classifies it as beginner / intermediate / advanced, and displays an annotated
playback window showing the pose skeleton alongside the classification result
and coaching tips.

Usage:
  python3 classify_video.py                    # opens a file picker
  python3 classify_video.py path/to/video.mp4  # pass path directly

predict_video(path) is a clean API function that returns a result dict —
call it directly from a future backend (e.g. FastAPI) without the display.
"""

import sys
from pathlib import Path

import cv2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from ultralytics import YOLO
import numpy as np
import torch
import torch.nn as nn

from model_logic.pipeline_utils import (
    YOLO_INTERVAL, WINDOW_SIZE,
    create_csrt_tracker, largest_person_box, select_subject,
    process_frame, interpolate_gaps, smooth, sliding_windows,
    interpolate_landmark_coords, draw_skeleton,
)
from model_logic.coaching_tips import COACHING_TIPS

# ── Inference config ─────────────────────────────────────────────────────────────
OUTPUT_DIR = Path("output")
MODEL_PATH = OUTPUT_DIR / "ski_classifier_kfold.pt"
MEAN_PATH  = OUTPUT_DIR / "scaler_mean_kfold.npy"
STD_PATH   = OUTPUT_DIR / "scaler_std_kfold.npy"
POSE_MODEL = "data/pose_landmarker_heavy.task"

CLASS_NAMES   = ["beginner", "intermediate", "advanced"]
LEVEL_COLOURS = {                        # BGR
    "beginner":     (80,  200, 80),
    "intermediate": (60,  200, 240),
    "advanced":     (60,  80,  220),
}

PANEL_W       = 400                      # width of the results sidebar (px)
MAX_DISPLAY_H = 700                      # max height to show video (px)


# ── Model definition (must match train_model.py) ─────────────────────────────────
class SkiClassifier(nn.Module):
    def __init__(self, n_features: int = 27, n_classes: int = 3) -> None:
        """BiLSTM → Dropout → Linear(ReLU) → Dropout → Linear classifier."""
        super().__init__()
        self.lstm  = nn.LSTM(n_features, 32, batch_first=True, bidirectional=True)
        self.drop1 = nn.Dropout(0.5)
        self.fc1   = nn.Linear(64, 32)
        self.relu  = nn.ReLU()
        self.drop2 = nn.Dropout(0.4)
        self.fc2   = nn.Linear(32, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Concatenate final hidden states from both LSTM directions and pass through the classifier head."""
        _, (h, _) = self.lstm(x)
        h = torch.cat([h[0], h[1]], dim=1)
        x = self.drop1(h)
        x = self.relu(self.fc1(x))
        x = self.drop2(x)
        return self.fc2(x)


# ── Internal pipeline (extraction + inference) ────────────────────────────────────
def _extract_and_infer(video_path: Path):
    """
    Runs the full detection and classification pipeline.

    Returns (result_dict, landmark_coords, feature_arr, is_interpolated,
             bbox_per_frame, fps, vid_w, vid_h, frame_count)
    so the caller can optionally render the annotated video.
    """
    yolo         = YOLO("yolov8n.pt")
    base_options = python.BaseOptions(model_asset_path=POSE_MODEL)
    detector     = vision.PoseLandmarker.create_from_options(
        vision.PoseLandmarkerOptions(
            base_options=base_options,
            output_segmentation_masks=False,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {"error": f"Could not open video: {video_path}",
                "predicted_class": None, "confidence": None,
                "class_probs": None, "tips": None, "n_windows": 0}, \
               {}, None, None, {}, 30, 0, 0, 0

    fps  = cap.get(cv2.CAP_PROP_FPS) or 30
    fw   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh_v = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    indices, features   = [], []
    landmark_coords     = {}
    bbox_per_frame      = {}

    success, first_frame = cap.read()
    if not success:
        cap.release()
        return {"error": "Video is empty.", "predicted_class": None,
                "confidence": None, "class_probs": None, "tips": None, "n_windows": 0}, \
               {}, None, None, {}, fps, fw, fh_v, 0

    h_px, w_px = first_frame.shape[:2]
    yolo_out   = yolo(first_frame, classes=[0], verbose=False)
    boxes      = yolo_out[0].boxes
    n_people   = len(boxes)

    if n_people == 0:
        x1, y1, x2, y2 = 0, 0, w_px, h_px
    elif n_people == 1:
        x1, y1, x2, y2 = largest_person_box(boxes)
    else:
        x1, y1, x2, y2 = select_subject(first_frame, boxes)

    tracker = create_csrt_tracker()
    tracker.init(first_frame, (x1, y1, x2 - x1, y2 - y1))
    frame_count = 1
    bbox_per_frame[0] = (x1, y1, x2, y2)
    process_frame(first_frame, 0, x1, y1, x2, y2, w_px, h_px,
                  detector, indices, features, landmark_coords)

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break
        h_px, w_px = image.shape[:2]
        idx         = frame_count
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
                x1, y1, x2, y2 = 0, 0, w_px, h_px

        bbox_per_frame[idx] = (x1, y1, x2, y2)
        process_frame(image, idx, x1, y1, x2, y2, w_px, h_px,
                      detector, indices, features, landmark_coords)

    cap.release()

    _err_result = {"predicted_class": None, "confidence": None,
                   "class_probs": None, "tips": None, "n_windows": 0}
    if len(indices) == 0:
        return {**_err_result, "error": "No pose detected in video."}, \
               {}, None, None, bbox_per_frame, fps, fw, fh_v, frame_count

    arr, is_interp = interpolate_gaps(
        np.array(indices,  dtype=np.int32),
        np.array(features, dtype=np.float32),
        frame_count,
    )
    arr     = smooth(arr)
    windows = list(sliding_windows(arr))

    if len(windows) == 0:
        return {**_err_result,
                "error": f"Not enough valid frames to form a window (need {WINDOW_SIZE})."}, \
               {}, arr, is_interp, bbox_per_frame, fps, fw, fh_v, frame_count

    X = np.array(windows, dtype=np.float32)

    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    model = SkiClassifier(n_features=checkpoint["n_features"],
                          n_classes=checkpoint["n_classes"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    mean = np.load(MEAN_PATH)
    std  = np.load(STD_PATH)
    X    = (X - mean) / std

    with torch.no_grad():
        probs = torch.softmax(model(torch.tensor(X)), dim=1).numpy()

    avg_probs       = probs.mean(axis=0)
    predicted_idx   = int(avg_probs.argmax())
    predicted_class = CLASS_NAMES[predicted_idx]

    result = {
        "predicted_class": predicted_class,
        "confidence":      float(avg_probs[predicted_idx]),
        "class_probs":     {CLASS_NAMES[i]: float(avg_probs[i]) for i in range(len(CLASS_NAMES))},
        "tips":            COACHING_TIPS[predicted_class],
        "n_windows":       len(windows),
        "error":           None,
    }

    is_nan      = np.isnan(arr[:, 0])
    all_lm_coords = interpolate_landmark_coords(
        landmark_coords,
        np.array(indices, dtype=np.int32),
        frame_count,
        is_nan,
    )

    return result, all_lm_coords, arr, is_interp, bbox_per_frame, fps, fw, fh_v, frame_count


# ── Results panel ─────────────────────────────────────────────────────────────────
def _build_results_panel(result: dict, height: int) -> np.ndarray:
    """Render a dark sidebar showing classification result and coaching tips."""
    panel = np.full((height, PANEL_W, 3), 28, dtype=np.uint8)

    def txt(s, x, y, scale, colour, bold=False):
        cv2.putText(panel, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, colour, 2 if bold else 1, cv2.LINE_AA)

    def hline(y):
        cv2.line(panel, (14, y), (PANEL_W - 14, y), (58, 58, 58), 1)

    lc  = LEVEL_COLOURS.get(result["predicted_class"], (220, 220, 220))
    dim = (160, 160, 160)
    y   = 36

    txt("CLASSIFICATION", 14, y, 0.48, dim);            y += 44
    txt(result["predicted_class"].upper(), 14, y, 1.15, lc, bold=True); y += 36
    txt(f"Confidence:  {result['confidence']:.1%}", 14, y, 0.50, (215, 215, 215)); y += 24
    txt(f"Windows:     {result['n_windows']}", 14, y, 0.44, dim);       y += 16
    hline(y);                                                             y += 18

    txt("Probabilities", 14, y, 0.44, dim);             y += 20
    bar_max = PANEL_W - 28
    for cls, prob in result["class_probs"].items():
        c = LEVEL_COLOURS.get(cls, (100, 100, 100))
        txt(cls, 14, y, 0.40, (200, 200, 200))
        txt(f"{prob:.0%}", PANEL_W - 52, y, 0.40, (200, 200, 200))
        y += 12
        bw = int(bar_max * prob)
        if bw > 0:
            cv2.rectangle(panel, (14, y), (14 + bw, y + 9), c, -1)
        cv2.rectangle(panel, (14, y), (14 + bar_max, y + 9), (58, 58, 58), 1)
        y += 18

    y += 4; hline(y); y += 18
    txt("Coaching tips", 14, y, 0.44, dim); y += 22

    max_chars = (PANEL_W - 42) // 7   # approx chars that fit per line at scale 0.37
    for i, tip in enumerate(result["tips"], 1):
        if y > height - 22:
            break
        txt(f"{i}.", 14, y, 0.39, (185, 185, 185))
        words = tip.split()
        line  = ""
        first = True
        for word in words:
            if len(line) + len(word) + 1 > max_chars:
                if y < height - 18:
                    txt(line.rstrip(), 36 if first else 36, y, 0.37, (175, 175, 175))
                    y += 17
                    first = False
                line = word + " "
            else:
                line += word + " "
        if line and y < height - 18:
            txt(line.rstrip(), 36, y, 0.37, (175, 175, 175))
            y += 21

    return panel


# ── Display window ────────────────────────────────────────────────────────────────
def display_results_window(
    video_path: Path,
    result: dict,
    all_lm_coords: dict,
    feature_arr: np.ndarray,
    is_interp: np.ndarray,
    bbox_per_frame: dict,
    fps: float,
    vid_w: int,
    vid_h: int,
):
    """
    Re-read the video and play it in a composite window:
      left  — annotated frame (skeleton in green/orange, dropped frames in grey label)
      right — static results panel (classification + probabilities + tips)

    Controls: SPACE to pause/resume, Q or ESC to close.
    The annotated video (without panel) is also saved to output/classified_<name>.mp4.
    """
    scale     = min(1.0, MAX_DISPLAY_H / vid_h) if vid_h > 0 else 1.0
    disp_w    = int(vid_w * scale)
    disp_h    = int(vid_h * scale)
    panel     = _build_results_panel(result, max(disp_h, MAX_DISPLAY_H))
    win_h     = max(disp_h, panel.shape[0])
    win_title = f"Ski Classifier — {video_path.name}"

    # Video writer for the annotated output (original resolution, no panel)
    out_path = OUTPUT_DIR / f"classified_{video_path.stem}.mp4"
    writer   = cv2.VideoWriter(
        str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (vid_w, vid_h)
    )

    cap     = cv2.VideoCapture(str(video_path))
    delay   = max(1, int(1000 / fps))
    paused  = False
    frame_i = 0

    while cap.isOpened():
        if not paused:
            success, frame = cap.read()
            if not success:
                break

            # ── Draw skeleton and bbox on the original-resolution frame ──────────
            feat     = feature_arr[frame_i] if feature_arr is not None and frame_i < len(feature_arr) else None
            has_feat = feat is not None and not np.isnan(feat).any()

            if has_feat:
                colour = (0, 165, 255) if is_interp[frame_i] else (0, 220, 0)
                draw_skeleton(frame, all_lm_coords.get(frame_i, {}), colour)

            if frame_i in bbox_per_frame:
                bx1, by1, bx2, by2 = bbox_per_frame[frame_i]
                cv2.rectangle(frame, (bx1, by1), (bx2, by2), (255, 100, 0), 2)

            writer.write(frame)   # save original-res annotated frame

            # ── Build composite display frame ─────────────────────────────────────
            disp_frame = cv2.resize(frame, (disp_w, disp_h))

            # Pad video vertically if panel is taller
            if disp_h < win_h:
                pad = np.zeros((win_h - disp_h, disp_w, 3), dtype=np.uint8)
                disp_frame = np.vstack([disp_frame, pad])

            # Status label on video
            if has_feat:
                status = "interpolated" if is_interp[frame_i] else "detected"
            else:
                status = "dropped"
            cv2.putText(disp_frame, f"frame {frame_i}  [{status}]",
                        (10, win_h - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.48, (160, 160, 160), 1, cv2.LINE_AA)

            composite = np.hstack([disp_frame, panel])
            frame_i  += 1

        cv2.imshow(win_title, composite)
        key = cv2.waitKey(1 if paused else delay) & 0xFF
        if key in (ord("q"), 27):     # Q or ESC — quit
            break
        if key == ord(" "):           # SPACE — pause / resume
            paused = not paused

    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    print(f"Annotated video saved → {out_path}")


# ── Public API ────────────────────────────────────────────────────────────────────
def predict_video(video_path: str | Path) -> dict:
    """
    Runs the full pipeline and returns a classification result dict:
    {
        "predicted_class":  str,
        "confidence":       float,
        "class_probs":      dict[str, float],
        "tips":             list[str],
        "n_windows":        int,
        "error":            str | None,
    }
    No windows are opened — suitable for use in a backend / API context.
    """
    result, *_ = _extract_and_infer(Path(video_path))
    return result


# ── CLI ───────────────────────────────────────────────────────────────────────────
def pick_file() -> Path | None:
    """Open a Tkinter file-picker dialog and return the selected Path, or None if cancelled."""
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        path = filedialog.askopenfilename(
            title="Select a ski video",
            filetypes=[("Video files", "*.mp4 *.mov *.avi *.mkv"), ("All files", "*.*")],
        )
        root.destroy()
        return Path(path) if path else None
    except Exception:
        return None


def main() -> None:
    """CLI entry point: resolve video path, run the full pipeline, and open the results window."""
    if len(sys.argv) > 1:
        video_path = Path(sys.argv[1])
    else:
        print("Opening file picker …")
        video_path = pick_file()
        if video_path is None:
            print("No file selected. Exiting.")
            return

    if not video_path.exists():
        print(f"File not found: {video_path}")
        return

    print(f"Video:  {video_path.name}")
    print("Running extraction pipeline …")

    result, all_lm_coords, feature_arr, is_interp, bbox_per_frame, fps, vid_w, vid_h, _ = \
        _extract_and_infer(video_path)

    if result["error"]:
        print(f"Error: {result['error']}")
        return

    print(f"Done — classified as {result['predicted_class'].upper()} "
          f"({result['confidence']:.1%} confidence)")
    print("Opening results window … (SPACE = pause, Q/ESC = close)\n")

    display_results_window(
        video_path, result, all_lm_coords,
        feature_arr, is_interp, bbox_per_frame,
        fps, vid_w, vid_h,
    )


if __name__ == "__main__":
    main()
