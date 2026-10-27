"""Combines the gaze (iris) signal with B1's head pose into one
"facing the robot" angle, by source priority - "alege sursa dupa ierarhie"
per the design doc (docs/context/gaze.md): iris, when a confirmed fixation
passed every gate and is plausible against head pose; otherwise head,
reusing human_head_pose's own `gaze_robot_deg` unchanged; "body" (a
track's overall position/orientation with no head or iris signal at all)
is listed in the design doc too but isn't built anywhere in this project
yet, so it's not a fallback option here - see docs/context/gaze.md.

Kept as a 2D, yaw-only correction (`gaze_yaw_deg - bearing_deg`), the same
simplification `human_head_pose.geometry` already documents as short of
the design doc's full 3D treatment (which folds in pitch too) - not
re-litigated here, just carried forward consistently.
"""

from dataclasses import dataclass

from human_head_pose.geometry import gaze_robot_deg as head_only_gaze_robot_deg
from human_head_pose.geometry import wrap_deg

SOURCE_IRIS = "iris"
SOURCE_HEAD = "head"


@dataclass
class FusedGaze:
    source: str
    gaze_robot_deg: float


def plausible_against_head(iris_yaw_deg: float, head_yaw_deg: float, max_eye_head_deg: float) -> bool:
    """Design doc's `max_eye_head_deg` check: an iris reading that disagrees
    with the head's own facing direction by more than this is more likely a
    bad gaze estimate than a real extreme eye-only glance, and is discarded
    rather than trusted - "privire - cap peste asta -> privirea se arunca"."""
    return abs(wrap_deg(iris_yaw_deg - head_yaw_deg)) <= max_eye_head_deg


def fuse(
    head_yaw_deg: float,
    bearing_deg_: float,
    iris_yaw_deg: float | None,
    iris_is_fixating: bool,
    max_eye_head_deg: float,
) -> FusedGaze:
    """`iris_yaw_deg` should already be the fixation-confirmed, One-Euro-
    filtered angle (or None if gating/fixation didn't clear this frame) -
    see human_gaze/gaze_node.py for where those are applied before this is
    called. Falls back to head-only whenever iris isn't both fixating and
    plausible against head pose - never partially trusts a rejected iris
    reading."""
    if iris_yaw_deg is not None and iris_is_fixating and plausible_against_head(
        iris_yaw_deg, head_yaw_deg, max_eye_head_deg
    ):
        return FusedGaze(source=SOURCE_IRIS, gaze_robot_deg=wrap_deg(iris_yaw_deg - bearing_deg_))
    return FusedGaze(source=SOURCE_HEAD, gaze_robot_deg=head_only_gaze_robot_deg(head_yaw_deg, bearing_deg_))
