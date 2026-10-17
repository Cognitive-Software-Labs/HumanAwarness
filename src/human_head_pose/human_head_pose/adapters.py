"""Data-conversion adapters between this package's neighbors and its own
model/component contracts - kept separate and named per-arrow, matching
the project's pipeline design doc (~/Downloads/Pipeline awareness
robot2.pdf, "Adaptoare" section): "the output of one component is rarely
exactly the next one's input," and an adapter is a pure function, tested on
its own, so that swapping a model or a message format only ever touches its
adapter, not the components on either side of it.

Two arrows get adapters here:
- `person_detection_to_bbox_keypoints`: a ROS `PersonDetection` message (or
  anything shaped like one - see its docstring) -> the plain
  `(bbox_xyxy, keypoints)` tuples the rest of this package works in. Used by
  `head_pose_node`; not needed by the test scripts, which already get plain
  tuples straight from Ultralytics or from a `human_tracking.Track`.
- `extract_head_crop`: keypoints + the source frame -> an actual pixel crop
  ready for `HeadPoseEstimator`, combining `head_crop.head_crop_bbox`'s
  geometry with the frame slicing every caller otherwise had to repeat.
"""

import numpy as np

from human_head_pose.head_crop import head_crop_bbox


def person_detection_to_bbox_keypoints(
    detection,
) -> tuple[tuple[float, float, float, float], list[tuple[float, float, float]]]:
    """Adapts a `human_awareness_msgs/PersonDetection` (or any object with
    the same `bbox_x/y/width/height` + `keypoints[].x/y/confidence` fields)
    into this package's plain `(bbox_xyxy, keypoints)` representation."""
    x1, y1 = detection.bbox_x, detection.bbox_y
    bbox_xyxy = (x1, y1, x1 + detection.bbox_width, y1 + detection.bbox_height)
    keypoints = [(kp.x, kp.y, kp.confidence) for kp in detection.keypoints]
    return bbox_xyxy, keypoints


def extract_head_crop(
    frame: np.ndarray,
    keypoints: list[tuple[float, float, float]],
    min_keypoint_conf: float = 0.3,
    head_k_h: float = 0.8,
    head_scale_alpha: float = 1.7,
    crop_margin: float = 0.2,
    min_head_px: float = 35.0,
) -> tuple[np.ndarray, tuple[float, float, float, float]] | tuple[None, None]:
    """`head_crop_bbox`'s geometry, realized as an actual pixel crop of
    `frame`. Returns `(None, None)` wherever `head_crop_bbox` would (no
    confident neck, or the computed crop is smaller than `min_head_px`), or
    where the crop falls entirely outside the frame - keypoints aren't
    guaranteed clamped to the image bounds, and a negative crop coordinate
    used directly as a slice stop would otherwise wrap around and silently
    select from the other end of the frame instead of producing no crop."""
    crop_bbox = head_crop_bbox(keypoints, min_keypoint_conf, head_k_h, head_scale_alpha, crop_margin, min_head_px)
    if crop_bbox is None:
        return None, None
    x1, y1, x2, y2 = crop_bbox
    frame_h, frame_w = frame.shape[:2]
    cx1, cy1 = max(0, int(x1)), max(0, int(y1))
    cx2, cy2 = min(frame_w, int(x2)), min(frame_h, int(y2))
    if cx2 <= cx1 or cy2 <= cy1:
        return None, None
    return frame[cy1:cy2, cx1:cx2], crop_bbox
