"""Monocular distance estimation for the phase-2 geometric decision rule
(docs/context/gaze.md) - the design doc's own three-tier fallback
("PARAMETRI · GEOMETRIE + TRAIECTORIE" slide): ground-plane projection from
an ankle keypoint when feet are visible, a shoulder-width size prior when
they aren't, and a person-height size prior as the last resort. Each tier
is only ever a fallback for the one before it - ground-plane projection is
the only tier that's actually measuring this specific person rather than
assuming they match an average adult's proportions, so it's preferred
whenever it's available.
"""

import math

from human_gaze.camera_model import CameraIntrinsics

_LEFT_ANKLE, _RIGHT_ANKLE = 15, 16
_LEFT_SHOULDER, _RIGHT_SHOULDER = 5, 6


def ground_plane_distance_m(
    ankle_v: float,
    intrinsics: CameraIntrinsics,
    camera_height_m: float,
    camera_pitch_deg: float,
) -> float | None:
    """Projects a ground-contact pixel (the ankle) onto the ground plane
    via the camera's known height and downward tilt - standard pinhole
    ground-plane triangulation. Returns the straight-line (euclidean)
    distance along the viewing ray, not the horizontal ground distance
    (`camera_height_m / sin(total_deg)`, not `/ tan`) - this is what
    `camera_model.person_position_camera_frame` needs to place the person
    at the correct point along their actual viewing ray, since the two
    differ once the robot (and so the final distance-to-robot the cone
    test actually cares about) isn't assumed co-located with the camera
    - see `camera_model.CameraRobotOffset`. `None` if the ray doesn't point
    at the ground at all given the current tilt (e.g. the ankle is above
    the horizon, which shouldn't happen for a real ankle keypoint but is
    possible with a bad tilt value or a jittery detection)."""
    ray_deg = math.degrees(math.atan((ankle_v - intrinsics.cy) / intrinsics.fy))
    total_deg = camera_pitch_deg + ray_deg
    if total_deg <= 0.0:
        return None
    return camera_height_m / math.sin(math.radians(total_deg))


def size_based_distance_m(pixel_size: float, real_size_m: float, focal_px: float) -> float | None:
    """Standard similar-triangles depth-from-known-size - shared by the
    shoulder-width and person-height fallbacks below."""
    if pixel_size <= 0.0:
        return None
    return (real_size_m * focal_px) / pixel_size


def estimate_distance_m(
    keypoints: list[tuple[float, float, float]],
    intrinsics: CameraIntrinsics,
    camera_height_m: float,
    camera_pitch_deg: float,
    bbox_height_px: float | None = None,
    shoulder_width_prior_m: float = 0.40,
    person_height_prior_m: float = 1.70,
    min_keypoint_conf: float = 0.3,
) -> tuple[float | None, str]:
    """Returns `(distance_m, method)` - `method` is one of "ground_plane",
    "shoulder_width", "height", or "none", so callers/diagnostics can tell
    which tier actually fired rather than only seeing the final number."""
    for idx in (_LEFT_ANKLE, _RIGHT_ANKLE):
        if idx < len(keypoints) and keypoints[idx][2] >= min_keypoint_conf:
            d = ground_plane_distance_m(keypoints[idx][1], intrinsics, camera_height_m, camera_pitch_deg)
            if d is not None:
                return d, "ground_plane"

    if max(_LEFT_SHOULDER, _RIGHT_SHOULDER) < len(keypoints):
        left, right = keypoints[_LEFT_SHOULDER], keypoints[_RIGHT_SHOULDER]
        if left[2] >= min_keypoint_conf and right[2] >= min_keypoint_conf:
            pixel_width = math.hypot(left[0] - right[0], left[1] - right[1])
            d = size_based_distance_m(pixel_width, shoulder_width_prior_m, intrinsics.fx)
            if d is not None:
                return d, "shoulder_width"

    if bbox_height_px is not None:
        d = size_based_distance_m(bbox_height_px, person_height_prior_m, intrinsics.fy)
        if d is not None:
            return d, "height"

    return None, "none"
