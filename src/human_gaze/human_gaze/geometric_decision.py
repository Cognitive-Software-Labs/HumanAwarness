"""Phase 2 of B2 (docs/context/gaze.md): replaces the flat
yaw-minus-bearing approximation (phase 1's `aware-threshold-deg` demo) with
the design doc's actual geometric test - whether the person's real 3D gaze
direction falls inside a cone around the robot's physical size at their
actual distance, per the doc's own parameters ("PARAMETRI · B PRIVIRE ÎN
TIMP + UNGHIURI" slide: theta_contact_deg, robot_radius_m,
peripheral_buffer_deg, head_only_extra_deg).

Two separate outputs, not one threshold, matching the doc's "Vede robotul?"
slide: `sees_robot` is the peripheral-vision case (robot merely in view,
not necessarily the direct target of gaze) and `direct_contact` is the
stricter, distance-independent "actually looking right at it" case.
"""

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class GeometricGazeResult:
    distance_m: float | None  # distance from the PERSON to the ROBOT - not to the camera, once they differ
    distance_method: str  # which tier produced the underlying camera-to-person measurement: "ground_plane" | "shoulder_width" | "height" | "none"
    theta_to_robot_deg: float | None  # angle between the gaze vector and the true direction to the robot
    robot_cone_deg: float | None  # atan(robot_radius_m / distance_m) at this distance
    sees_robot: bool
    direct_contact: bool


def robot_cone_half_angle_deg(distance_m: float, robot_radius_m: float) -> float:
    return math.degrees(math.atan(robot_radius_m / distance_m))


def angle_between_deg(v1: np.ndarray, v2: np.ndarray) -> float:
    cos_theta = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    return math.degrees(math.acos(np.clip(cos_theta, -1.0, 1.0)))


def evaluate(
    gaze_vector: np.ndarray,
    person_position_camera_frame: np.ndarray | None,
    robot_center_camera_frame: np.ndarray,
    distance_method: str,
    source: str,
    robot_radius_m: float = 0.3,
    theta_contact_deg: float = 12.0,
    peripheral_buffer_deg: float = 5.0,
    head_only_extra_deg: float = 15.0,
) -> GeometricGazeResult:
    """Compares the person's and the robot's actual 3D positions (both in
    camera-optical-frame coordinates) rather than a pre-baked direction -
    so this is correct whether or not the robot is co-located with the
    camera (`camera_model.CameraRobotOffset`; the project's old assumption
    is just the zero-offset special case). `source` is "iris" or "head"
    (`GazeTrackResult.source`) - `head_only_extra_deg` widens only
    `sees_robot`, not `direct_contact`: a widened peripheral allowance
    makes sense for head pose's coarser precision, but "direct eye
    contact" shouldn't get easier to claim just because the more precise
    (iris) signal wasn't available this frame."""
    if person_position_camera_frame is None:
        return GeometricGazeResult(None, distance_method, None, None, False, False)

    vector_to_robot = robot_center_camera_frame - person_position_camera_frame
    distance_m = float(np.linalg.norm(vector_to_robot))
    if distance_m <= 0.0:
        # Degenerate (person's estimated position exactly at the robot's
        # center) - can't form a direction, but trivially "sees" it.
        return GeometricGazeResult(0.0, distance_method, 0.0, 0.0, True, True)

    direction_to_robot = vector_to_robot / distance_m
    theta_deg = angle_between_deg(gaze_vector, direction_to_robot)
    cone_deg = robot_cone_half_angle_deg(distance_m, robot_radius_m)
    extra_deg = head_only_extra_deg if source == "head" else 0.0
    sees_robot = theta_deg <= cone_deg + peripheral_buffer_deg + extra_deg
    direct_contact = theta_deg <= theta_contact_deg
    return GeometricGazeResult(distance_m, distance_method, theta_deg, cone_deg, sees_robot, direct_contact)
