"""Minimal pinhole camera model shared by `distance_estimator.py` and
`geometric_decision.py` (phase 2 of B2 - docs/context/gaze.md). Derives
intrinsics from `camera_hfov_deg` + image size, the same approximation
`human_head_pose.geometry.bearing_deg` already makes, rather than requiring
real calibration (`camera_info`/`K`/`D`) this project doesn't have outside a
real robot deployment - see docs/context/gaze.md for the calibration gap
this leaves open.
"""

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float


def intrinsics_from_hfov(image_width: float, image_height: float, hfov_deg: float) -> CameraIntrinsics:
    """Square-pixel assumption (fx == fy) - reasonable for any normal
    camera/phone sensor, and the only option without a real calibration."""
    fx = (image_width / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
    return CameraIntrinsics(fx=fx, fy=fx, cx=image_width / 2.0, cy=image_height / 2.0)


def pixel_ray_camera_frame(u: float, v: float, intrinsics: CameraIntrinsics) -> np.ndarray:
    """Unit 3D ray from the camera's optical center through pixel (u, v),
    in the same camera-optical-frame convention as
    `gaze_estimator.gaze_yaw_pitch_to_vector` (+x right, +y down, +z
    forward) - so the two are directly comparable with a plain dot
    product, no extra frame conversion needed."""
    x = (u - intrinsics.cx) / intrinsics.fx
    y = (v - intrinsics.cy) / intrinsics.fy
    vec = np.array([x, y, 1.0], dtype=np.float64)
    return vec / np.linalg.norm(vec)


def person_position_camera_frame(
    u: float, v: float, distance_m: float, distance_method: str, intrinsics: CameraIntrinsics
) -> np.ndarray:
    """Reconstructs a person's full 3D position in camera-optical-frame
    coordinates from their reference pixel + an estimated distance -
    reconstructed differently depending on what `distance_method`
    (`distance_estimator.estimate_distance_m`) actually measured:

    - `"ground_plane"`'s distance is a true straight-line (euclidean)
      distance along the viewing ray - scale the unit ray by it directly.
    - `"shoulder_width"`/`"height"`'s distance is a depth (forward/z)
      estimate from focal-length/known-size triangulation - the standard
      approximation for monocular size-based distance (treats apparent
      size as governed by z-distance alone, ignoring the small additional
      foreshortening an off-axis/off-center person would have) - so it's
      the ray's *unnormalized* direction scaled so its z-component lands
      exactly on that depth, not the unit ray.

    Needed (rather than just a scalar distance) so the phase-2 geometric
    decision can compare the person's position against the robot's own
    position - not the camera's - once they differ (`camera_model.
    CameraRobotOffset` below)."""
    if distance_method == "ground_plane":
        return pixel_ray_camera_frame(u, v, intrinsics) * distance_m
    x = (u - intrinsics.cx) / intrinsics.fx * distance_m
    y = (v - intrinsics.cy) / intrinsics.fy * distance_m
    return np.array([x, y, distance_m], dtype=np.float64)


@dataclass
class CameraRobotOffset:
    """Fixed, rigid offset from the camera's own optical center to the
    robot body's center, measured along the *camera's own* axes (right /
    down / forward from the lens) - not the robot's own body frame.

    This is a deliberate simplification: it sidesteps ever needing to know
    which way the robot's body itself is facing (the cone test below
    treats the robot as a plain sphere of radius `robot_radius_m`, with no
    orientation at all - matching the design doc's own model), and it's
    arguably the easier thing for a real person to measure too - stand at
    the camera, measure forward/right/down to the robot's body center,
    done. It's correct as long as the camera is rigidly bolted to the
    robot body; a camera on an independently-moving gimbal/pan-tilt unit
    would need this offset to vary with the gimbal's angle, which isn't
    supported - see docs/context/gaze.md."""

    right_m: float = 0.0
    down_m: float = 0.0
    forward_m: float = 0.0


def robot_center_camera_frame(offset: CameraRobotOffset) -> np.ndarray:
    """All-zero `offset` (the project's old, now-default assumption) means
    the robot's center is exactly at the camera's own optical center."""
    return np.array([offset.right_m, offset.down_m, offset.forward_m], dtype=np.float64)
