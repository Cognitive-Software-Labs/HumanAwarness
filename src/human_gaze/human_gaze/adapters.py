"""Data-conversion adapters, named per-arrow - same convention as
human_head_pose/adapters.py and the design doc's "Adaptoare" principle
(docs/context/gaze.md): swapping a model only ever touches its adapter.

Two arrows get one here:
- `crop_relative_to_frame_xyxy`: `MediaPipeGate`'s output is in head-crop-
  relative pixel coordinates (it only ever sees the head crop, not the
  full frame) - this converts a box back to frame-absolute coordinates
  using the head crop's own offset in the frame (from
  `human_head_pose.adapters.extract_head_crop`).
- `extract_face_crop`: the frame-absolute face bbox -> an actual pixel
  crop taken from the *original* frame, not the head-crop sub-image, to
  avoid compounding the head crop's own resolution loss - matching the
  design doc's "crop fata din cadrul original" (not from the head crop).
"""

import numpy as np


def crop_relative_to_frame_xyxy(
    bbox_in_crop: tuple[float, float, float, float],
    crop_bbox_in_frame: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    crop_x1, crop_y1, _crop_x2, _crop_y2 = crop_bbox_in_frame
    bx1, by1, bx2, by2 = bbox_in_crop
    return (bx1 + crop_x1, by1 + crop_y1, bx2 + crop_x1, by2 + crop_y1)


def point_relative_to_frame_xy(
    point_in_crop: tuple[float, float],
    crop_bbox_in_frame: tuple[float, float, float, float],
) -> tuple[float, float]:
    """Unused by the current flat-angle pipeline - kept for the phase-2
    geometric decision rule (docs/context/gaze.md), which will need the
    iris centers MediaPipeGate already extracts in frame coordinates."""
    crop_x1, crop_y1, _crop_x2, _crop_y2 = crop_bbox_in_frame
    return (point_in_crop[0] + crop_x1, point_in_crop[1] + crop_y1)


def extract_face_crop(
    frame: np.ndarray,
    face_bbox_in_frame: tuple[float, float, float, float],
    margin_frac: float = 0.1,
) -> np.ndarray | None:
    """A small margin is added since MediaPipe's landmark-derived bbox is
    tight to the face outline, while MobileGaze was trained/demoed on
    detector boxes that are typically a bit looser (see
    docs/context/gaze.md) - `None` if the result would be empty (the face
    bbox fell entirely outside the frame)."""
    x1, y1, x2, y2 = face_bbox_in_frame
    w, h = x2 - x1, y2 - y1
    mx, my = w * margin_frac, h * margin_frac
    frame_h, frame_w = frame.shape[:2]
    cx1, cy1 = max(0, int(x1 - mx)), max(0, int(y1 - my))
    cx2, cy2 = min(frame_w, int(x2 + mx)), min(frame_h, int(y2 + my))
    if cx2 <= cx1 or cy2 <= cy1:
        return None
    return frame[cy1:cy2, cx1:cx2]
