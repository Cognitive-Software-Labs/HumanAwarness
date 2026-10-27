"""MediaPipe Face Landmarker: runs on the same head crop human_head_pose
already derives (see human_gaze/adapters.py), and gates whether the result
is trustworthy enough to hand to the gaze model - matching the design doc's
"MediaPipe pregateste, L2CS-Net estimeaza" framing (docs/context/gaze.md):
MediaPipe's job here is gating + a face bbox, not gaze itself.

Model: the official `face_landmarker` Task bundle (478 3D landmarks
including iris, 5 points per eye) - a plain HTTPS download from Google's
model bucket, cached locally on first use, same pattern as this project's
other model weights.

Known library gotcha, confirmed empirically (not assumed) and worth
documenting since it silently returns zero faces rather than erroring:
`FaceLandmarkerOptions(num_faces=1)` spuriously detects 0 faces on frames
where `num_faces=2` correctly finds the same single face - true in both
mediapipe 0.10.21 and 1.1.1. Worked around by always requesting at least 2
and using the first/highest-confidence result - see docs/context/gaze.md.
"""

import math
import os
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"

# COCO 478-point face mesh indices (Google's canonical topology - see
# docs/context/gaze.md for how these were cross-checked before trusting
# them, same discipline as this project's other keypoint-index code).
_NUM_FACE_FACES_WORKAROUND = 2  # see module docstring - never actually 1
_RIGHT_EYE_IRIS_CENTER = 468
_LEFT_EYE_IRIS_CENTER = 473
# 6-point EAR sets (Soukupova & Cech 2016 style): [outer, upper1, upper2, inner, lower1, lower2].
_RIGHT_EYE_EAR = (33, 160, 158, 133, 153, 144)
_LEFT_EYE_EAR = (362, 385, 387, 263, 373, 380)


def _default_cache_path() -> Path:
    cache_dir = Path.home() / ".cache" / "human_gaze"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / "face_landmarker.task"


def _ensure_model(model_path: str | None) -> str:
    path = Path(model_path) if model_path else _default_cache_path()
    if path.exists():
        return str(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_name = tempfile.mkstemp(dir=path.parent, suffix=".part")
    os.close(tmp_fd)
    tmp_path = Path(tmp_name)
    try:
        urllib.request.urlretrieve(_MODEL_URL, str(tmp_path))
        tmp_path.replace(path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
    return str(path)


def _eye_aspect_ratio(points: list[tuple[float, float]]) -> float:
    """Soukupova & Cech (2016): (|p2-p6| + |p3-p5|) / (2 * |p1-p4|) - drops
    toward 0 as the eyelid closes, stays roughly constant (~0.2-0.3) while
    open, largely independent of face scale/in-plane rotation."""
    p1, p2, p3, p4, p5, p6 = points
    vertical = math.dist(p2, p6) + math.dist(p3, p5)
    horizontal = math.dist(p1, p4)
    if horizontal <= 0:
        return 0.0
    return vertical / (2.0 * horizontal)


@dataclass
class EyeGazeGateResult:
    """Everything downstream (the gaze model, the fixation/filter stage)
    needs from one frame's gating pass."""

    passed: bool
    reason: str  # why gating failed, or "ok" - for diagnostics/display, not branching on
    face_bbox_xyxy: tuple[float, float, float, float] | None = None
    eye_width_px: float = 0.0
    ear: float = 0.0
    blur_score: float = 0.0
    # Both unused by the current flat-angle pipeline - kept for the
    # phase-2 geometric decision rule (docs/context/gaze.md), paired with
    # adapters.point_relative_to_frame_xy to place the iris in frame
    # coordinates.
    right_iris_xy: tuple[float, float] | None = None
    left_iris_xy: tuple[float, float] | None = None


class MediaPipeGate:
    def __init__(
        self,
        model_path: str | None = None,
        min_face_detection_confidence: float = 0.3,
        min_eye_px: float = 12.0,
        ear_min: float = 0.2,
        blur_min: float = 0.0,
    ):
        resolved_path = _ensure_model(model_path)
        base_options = mp_python.BaseOptions(model_asset_path=resolved_path)
        options = mp_vision.FaceLandmarkerOptions(
            base_options=base_options,
            num_faces=_NUM_FACE_FACES_WORKAROUND,
            min_face_detection_confidence=min_face_detection_confidence,
        )
        self._landmarker = mp_vision.FaceLandmarker.create_from_options(options)
        self.min_eye_px = min_eye_px
        self.ear_min = ear_min
        self.blur_min = blur_min

    def process(self, head_crop_bgr: np.ndarray) -> EyeGazeGateResult:
        """Runs Face Landmarker on `head_crop_bgr` and applies every gate
        except the head-yaw and crop-size ones (those use B1's head pose
        and are checked by the caller before this is even invoked - see
        human_gaze/gaze_node.py)."""
        if head_crop_bgr is None or head_crop_bgr.size == 0:
            return EyeGazeGateResult(passed=False, reason="empty crop")

        h, w = head_crop_bgr.shape[:2]
        rgb = cv2.cvtColor(head_crop_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        result = self._landmarker.detect(mp_image)
        if not result.face_landmarks:
            return EyeGazeGateResult(passed=False, reason="no face")

        landmarks = result.face_landmarks[0]
        xs = [p.x * w for p in landmarks]
        ys = [p.y * h for p in landmarks]
        face_bbox = (min(xs), min(ys), max(xs), max(ys))

        right_eye_pts = [(landmarks[i].x * w, landmarks[i].y * h) for i in _RIGHT_EYE_EAR]
        left_eye_pts = [(landmarks[i].x * w, landmarks[i].y * h) for i in _LEFT_EYE_EAR]
        right_width = math.dist(right_eye_pts[0], right_eye_pts[3])
        left_width = math.dist(left_eye_pts[0], left_eye_pts[3])
        eye_width_px = max(right_width, left_width)

        right_ear = _eye_aspect_ratio(right_eye_pts)
        left_ear = _eye_aspect_ratio(left_eye_pts)
        ear = max(right_ear, left_ear)  # one eye can be foreshortened in profile - trust whichever reads more open

        right_iris = (landmarks[_RIGHT_EYE_IRIS_CENTER].x * w, landmarks[_RIGHT_EYE_IRIS_CENTER].y * h)
        left_iris = (landmarks[_LEFT_EYE_IRIS_CENTER].x * w, landmarks[_LEFT_EYE_IRIS_CENTER].y * h)

        eye_region = head_crop_bgr[
            max(0, int(min(ys))) : min(h, int(max(ys))),
            max(0, int(min(xs))) : min(w, int(max(xs))),
        ]
        blur_score = (
            float(cv2.Laplacian(cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
            if eye_region.size > 0
            else 0.0
        )

        if eye_width_px < self.min_eye_px:
            reason = "eyes too small"
        elif ear < self.ear_min:
            reason = "eyes closed (blink)"
        elif blur_score < self.blur_min:
            reason = "blurry"
        else:
            reason = "ok"

        return EyeGazeGateResult(
            passed=(reason == "ok"),
            reason=reason,
            face_bbox_xyxy=face_bbox,
            eye_width_px=eye_width_px,
            ear=ear,
            blur_score=blur_score,
            right_iris_xy=right_iris,
            left_iris_xy=left_iris,
        )
