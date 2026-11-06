"""PersonDetection (from human_gaze) -> AwarenessCue: the "All cues -> fusion"
adapter from the design doc, for the one cue awareness uses so far."""

from human_awareness.awareness import SOURCE_HEAD, SOURCE_IRIS, AwarenessCue

# human_tracking's `state` value for a track matched to a detection this frame.
# LOST_KALMAN/LOST_GALLERY tracks are still published, but with the keypoints
# of their last match, so a head pose computed for them describes a stale crop.
ACTIVE = "ACTIVE"

# `distance_method` when the phase-2 decision didn't run ("") or found no
# distance ("none"): `sees_robot` is then False without having been tested.
_NO_DECISION = ("", "none")


def person_detection_to_cue(det) -> AwarenessCue | None:
    """`None` (unknown) unless this frame really looked at the person: ACTIVE,
    a valid head pose, and a phase-2 `sees_robot` decision that ran. The
    iris-position veto needs no special case: over ROS it already arrives as
    `sees_robot=False`."""
    if det.state != ACTIVE or not det.head_pose_valid or det.distance_method in _NO_DECISION:
        return None
    if det.gaze_source not in (SOURCE_IRIS, SOURCE_HEAD):
        return None
    return AwarenessCue(sees_robot=bool(det.sees_robot), source=det.gaze_source)
