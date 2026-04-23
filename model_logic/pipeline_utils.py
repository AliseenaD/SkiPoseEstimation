"""
pipeline_utils.py

Shared constants, geometry helpers, and pipeline functions used by
extract_features.py, test_pipeline.py, and classify_video.py.

Any change to feature definitions, visibility thresholds, or window
parameters must be made here — all three scripts inherit them automatically.
"""

import math
import sys

import cv2
import mediapipe as mp
import numpy as np

# ── Pipeline config ─────────────────────────────────────────────────────────────
YOLO_INTERVAL          = 5
VISIBILITY_THRESH      = 0.45   # applied to core landmarks
VISIBILITY_THRESH_SOFT = 0      # applied to wrists (0 = disabled)
SMOOTHING_WINDOW       = 5
WINDOW_SIZE            = 30
STRIDE                 = 15
MAX_INTERPOLATION_GAP  = 30

REQUIRED_HARD = {11, 12, 23, 24, 27, 28}   # shoulders, hips, ankles
REQUIRED_SOFT = {15, 16}                    # wrists

POSITION_LANDMARKS = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
DRAW_LANDMARKS     = set(POSITION_LANDMARKS)

_LM_NAMES = {
    11: "l_shoulder", 12: "r_shoulder",
    13: "l_elbow",    14: "r_elbow",
    15: "l_wrist",    16: "r_wrist",
    23: "l_hip",      24: "r_hip",
    25: "l_knee",     26: "r_knee",
    27: "l_ankle",    28: "r_ankle",
}

FEATURE_NAMES = (
    ["upper_lower_rotation", "stance_width", "arm_height"]
    + [f"{_LM_NAMES[i]}_{ax}" for i in POSITION_LANDMARKS for ax in ("x", "y")]
)
N_FEATURES = len(FEATURE_NAMES)   # 27


# ── Geometry helpers ─────────────────────────────────────────────────────────────
def line_angle(p1, p2):
    """Return the angle in degrees of the line from p1 to p2, measured from the positive x-axis."""
    return math.degrees(math.atan2(p2[1] - p1[1], p2[0] - p1[0]))

def angular_diff(a, b):
    """Return the shortest unsigned difference between two angles in degrees (0–180)."""
    return abs((a - b + 180) % 360 - 180)

def dist(p1, p2):
    """Return the Euclidean distance between two (x, y) points."""
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def mid(p1, p2):
    """Return the midpoint between two (x, y) points."""
    return ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)


# ── Feature extraction ───────────────────────────────────────────────────────────
def extract_features(landmarks, cw, ch):
    """Compute 27 features from a MediaPipe pose landmark list."""
    def px(i):
        lm = landmarks[i]
        return (lm.x * cw, lm.y * ch)

    ls, rs = px(11), px(12)
    lw, rw = px(15), px(16)
    lh, rh = px(23), px(24)
    la, ra = px(27), px(28)

    upper_lower_rotation = angular_diff(line_angle(ls, rs), line_angle(lh, rh))
    stance_width         = dist(la, ra) / max(dist(lh, rh), 1e-6)

    hip_mid     = mid(lh, rh)
    sho_mid     = mid(ls, rs)
    torso_h     = max(dist(sho_mid, hip_mid), 1e-6)
    avg_wrist_y = (lw[1] + rw[1]) / 2
    arm_height  = (hip_mid[1] - avg_wrist_y) / torso_h

    positions = []
    for lm_id in POSITION_LANDMARKS:
        x, y = px(lm_id)
        positions.append((x - hip_mid[0]) / torso_h)
        positions.append((y - hip_mid[1]) / torso_h)

    return [upper_lower_rotation, stance_width, arm_height] + positions


# ── YOLO / tracker helpers ───────────────────────────────────────────────────────
def create_csrt_tracker():
    """Instantiate a CSRT tracker, trying multiple OpenCV API locations for compatibility."""
    for factory in [
        lambda: cv2.legacy.TrackerCSRT_create(),
        lambda: cv2.TrackerCSRT_create(),
        lambda: cv2.TrackerCSRT.create(),
    ]:
        try:
            return factory()
        except AttributeError:
            continue
    raise RuntimeError(
        "CSRT tracker not available.\n  pip install opencv-contrib-python"
    )

def largest_person_box(boxes):
    """Return the bounding box with the largest area (closest person heuristic)."""
    best, best_area = None, 0
    for box in boxes:
        bx1, by1, bx2, by2 = map(int, box.xyxy[0])
        area = (bx2 - bx1) * (by2 - by1)
        if area > best_area:
            best_area = area
            best = (bx1, by1, bx2, by2)
    return best

def select_subject(image, yolo_boxes):
    """
    Show numbered YOLO boxes in a preview window and prompt the user to pick
    one via the terminal.  Also supports 'r' (draw custom ROI) and 'q' (quit).

    Returns (x1, y1, x2, y2).
    """
    boxes = [tuple(map(int, b.xyxy[0])) for b in yolo_boxes]
    n     = len(boxes)

    display = image.copy()
    for i, (bx1, by1, bx2, by2) in enumerate(boxes):
        cv2.rectangle(display, (bx1, by1), (bx2, by2), (0, 220, 220), 2)
        cv2.putText(display, str(i + 1), (bx1 + 6, by1 + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 220, 220), 2, cv2.LINE_AA)
    cv2.imshow("Select subject — see terminal for prompt", display)
    cv2.waitKey(1)

    print()
    for i, (bx1, by1, bx2, by2) in enumerate(boxes):
        print(f"  [{i + 1}]  box at ({bx1}, {by1}) → ({bx2}, {by2})")
    print("  [r]  draw a custom ROI with the mouse")
    print("  [q]  quit")

    while True:
        if n == 0:
            prompt = "\nNo person detected. Enter r to draw ROI, or q to quit: "
        elif n == 1:
            prompt = "\n1 person detected. Enter 1 to confirm, or r/q: "
        else:
            prompt = f"\n{n} people detected. Enter 1-{n}, r, or q: "

        choice = input(prompt).strip().lower()

        if choice == "q":
            cv2.destroyAllWindows()
            sys.exit(0)

        if choice == "r":
            cv2.destroyAllWindows()
            roi = cv2.selectROI("Draw ROI around subject — press ENTER when done",
                                image, fromCenter=False, showCrosshair=True)
            cv2.destroyAllWindows()
            rx, ry, rw, rh = map(int, roi)
            if rw > 0 and rh > 0:
                return (rx, ry, rx + rw, ry + rh)
            print("Empty ROI, try again.")
            cv2.imshow("Select subject — see terminal for prompt", display)
            cv2.waitKey(1)
            continue

        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < n:
                cv2.destroyAllWindows()
                return boxes[idx]

        print(f"  Invalid — enter a number between 1 and {n}, 'r', or 'q'.")


# ── Per-frame pose processing ────────────────────────────────────────────────────
def process_frame(
    image, idx, x1, y1, x2, y2, fw, fh,
    detector, indices, features,
    landmark_coords=None,
):
    """
    Crop the subject region, run MediaPipe, apply visibility checks, and
    append results to the caller's accumulators.

    If landmark_coords (dict) is provided, pixel positions for every landmark
    in DRAW_LANDMARKS are stored at landmark_coords[idx] — used by test_pipeline
    for skeleton rendering.
    """
    pad = 40
    cx1 = max(0, x1 - pad);  cy1 = max(0, y1 - pad)
    cx2 = min(fw, x2 + pad);  cy2 = min(fh, y2 + pad)

    crop   = image[cy1:cy2, cx1:cx2].copy()
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=crop)
    result = detector.detect(mp_img)

    if not result.pose_landmarks:
        return

    lms = result.pose_landmarks[0]
    if (any(lms[i].visibility < VISIBILITY_THRESH      for i in REQUIRED_HARD) or
            any(lms[i].visibility < VISIBILITY_THRESH_SOFT for i in REQUIRED_SOFT)):
        return

    ch_px, cw_px = crop.shape[:2]
    indices.append(idx)
    features.append(extract_features(lms, cw_px, ch_px))

    if landmark_coords is not None:
        landmark_coords[idx] = {
            i: (cx1 + int(lms[i].x * cw_px), cy1 + int(lms[i].y * ch_px))
            for i in DRAW_LANDMARKS
        }


# ── Gap interpolation ────────────────────────────────────────────────────────────
def interpolate_gaps(valid_indices, valid_features, total_frames):
    """
    Linearly interpolate across dropped frames.  Gaps longer than
    MAX_INTERPOLATION_GAP frames and leading/trailing edges are left as NaN.

    Returns (arr, is_interpolated):
      arr              (total_frames, N_FEATURES) float32, NaN where unrecoverable
      is_interpolated  (total_frames,) bool, True for filled-in frames
    """
    arr = np.full((total_frames, N_FEATURES), np.nan, dtype=np.float32)

    if len(valid_indices) == 0:
        return arr, np.zeros(total_frames, dtype=bool)

    all_idx = np.arange(total_frames, dtype=np.float64)
    for f in range(N_FEATURES):
        arr[:, f] = np.interp(all_idx, valid_indices, valid_features[:, f])

    arr[:valid_indices[0]]      = np.nan
    arr[valid_indices[-1] + 1:] = np.nan

    for i in range(len(valid_indices) - 1):
        if valid_indices[i + 1] - valid_indices[i] - 1 > MAX_INTERPOLATION_GAP:
            arr[valid_indices[i] + 1 : valid_indices[i + 1]] = np.nan

    detected        = np.zeros(total_frames, dtype=bool)
    detected[valid_indices] = True
    is_interpolated = ~detected & ~np.isnan(arr[:, 0])

    return arr, is_interpolated


# ── Temporal smoothing ───────────────────────────────────────────────────────────
def smooth(arr, k=SMOOTHING_WINDOW):
    """Centred rolling mean over k frames, NaN-aware."""
    out = arr.copy()
    for i in range(len(arr)):
        if np.isnan(arr[i]).any():
            continue
        s      = max(0, i - k // 2)
        e      = min(len(arr), i + k // 2 + 1)
        window = arr[s:e]
        valid  = window[~np.isnan(window).any(axis=1)]
        if len(valid) > 0:
            out[i] = valid.mean(axis=0)
    return out


# ── Skeleton drawing ────────────────────────────────────────────────────────────
POSE_CONNECTIONS = [
    (11, 12), (11, 23), (12, 24), (23, 24),   # torso
    (11, 13), (13, 15),                         # left arm
    (12, 14), (14, 16),                         # right arm
    (23, 25), (25, 27),                         # left leg
    (24, 26), (26, 28),                         # right leg
]

def draw_skeleton(image, coords, colour):
    """Draw pose connections and joint dots onto image in-place."""
    for start, end in POSE_CONNECTIONS:
        if start in coords and end in coords:
            cv2.line(image, coords[start], coords[end], colour, 2)
    for pt in coords.values():
        cv2.circle(image, pt, 5, colour, -1)


# ── Landmark coord interpolation ─────────────────────────────────────────────────
def interpolate_landmark_coords(landmark_coords, valid_indices, total_frames, is_nan):
    """
    Linearly interpolate (x, y) pixel positions for every landmark in
    DRAW_LANDMARKS across all usable frames.

    Returns {frame_idx: {landmark_id: (x, y)}} for detected + interpolated frames.
    """
    if len(valid_indices) == 0:
        return {}

    all_idx = np.arange(total_frames, dtype=np.float64)
    result  = {}

    for lm_id in DRAW_LANDMARKS:
        xs = np.array([landmark_coords[vi][lm_id][0] for vi in valid_indices], dtype=np.float64)
        ys = np.array([landmark_coords[vi][lm_id][1] for vi in valid_indices], dtype=np.float64)
        all_x = np.interp(all_idx, valid_indices, xs)
        all_y = np.interp(all_idx, valid_indices, ys)

        for frame_idx in range(total_frames):
            if is_nan[frame_idx]:
                continue
            result.setdefault(frame_idx, {})[lm_id] = (
                int(round(all_x[frame_idx])),
                int(round(all_y[frame_idx])),
            )

    return result


# ── Sliding window ───────────────────────────────────────────────────────────────
def sliding_windows(arr):
    """
    Yield non-overlapping WINDOW_SIZE windows from arr with STRIDE step.
    Windows containing any NaN frame are skipped.
    """
    for start in range(0, len(arr) - WINDOW_SIZE + 1, STRIDE):
        window = arr[start : start + WINDOW_SIZE]
        if not np.isnan(window).any():
            yield window
