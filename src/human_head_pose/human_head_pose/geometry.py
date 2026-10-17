"""Combines a person's head pose with their position in frame to estimate
whether they're actually facing the robot - not just facing the camera's
optical axis, which is only the same thing when the person happens to be
dead-center in the image.

A head-pose network trained on face crops alone (no position info - the crop
is centered/normalized before it ever reaches the network) can only ever
report orientation relative to the camera-to-face viewing ray, not relative
to the camera's fixed optical axis. For a person off to the side of the
frame, those two references differ by exactly that person's bearing angle,
so this correction (standard in the gaze-target-estimation literature, e.g.
work on detecting "looking at the camera" for off-center subjects) is applied
explicitly here rather than assuming yaw alone already answers "are they
looking at us":

    gaze_robot_deg = head_yaw_deg - bearing_deg

Caveat (see docs/context/head_pose.md): the sign of this correction depends
on 6DRepNet360's own yaw sign convention, which hasn't been empirically
checked against a real photo yet - treat the magnitude as more trustworthy
than the sign until that's done.
"""

import math


def wrap_deg(angle_deg: float) -> float:
    """Wrap an angle into (-180, 180]."""
    return (angle_deg + 180.0) % 360.0 - 180.0


def bearing_deg(bbox_center_x: float, image_width: float, camera_hfov_deg: float) -> float:
    """Azimuth of a point at image x-coordinate `bbox_center_x`, relative to
    the camera's optical axis (0 = dead center), under a simple pinhole
    model with the given horizontal field of view. Does not need depth/
    distance - azimuth from a horizontal pixel offset is depth-independent
    for a pinhole camera.
    """
    principal_x = image_width / 2.0
    focal_length_px = principal_x / math.tan(math.radians(camera_hfov_deg / 2.0))
    return math.degrees(math.atan2(bbox_center_x - principal_x, focal_length_px))


def gaze_robot_deg(head_yaw_deg: float, bearing_deg_: float) -> float:
    """~0 => the head is pointed back at the robot; ~180 => facing directly away."""
    return wrap_deg(head_yaw_deg - bearing_deg_)
