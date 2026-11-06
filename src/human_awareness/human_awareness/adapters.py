"""The design doc's "All cues -> fusion" adapter, for the one cue awareness
uses so far: PersonDetection (from human_gaze) -> AwarenessCue, plus how a
message's header stamp fits the stream before its cues are applied."""

from human_awareness.awareness import SOURCE_HEAD, SOURCE_IRIS, AwarenessCue

# human_tracking's `state` value for a track matched to a detection this frame.
# LOST_KALMAN/LOST_GALLERY tracks are still published, but with the keypoints
# of their last match, so a head pose computed for them describes a stale crop.
ACTIVE = "ACTIVE"

# `distance_method` when the phase-2 decision didn't run ("") or found no
# distance ("none"): `sees_robot` is then False without having been tested.
_NO_DECISION = ("", "none")

# A stamp more than this far behind the newest one means the source restarted
# (a looping bag, a reset simulation). Anything closer is a message delivered
# out of order.
RESTART_JUMP_S = 1.0

STAMP_ADVANCE = "advance"
STAMP_REORDERED = "reordered"
STAMP_RESTART = "restart"


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


def classify_stamp(stamp_s: float, newest_s: float | None, restart_jump_s: float = RESTART_JUMP_S) -> str:
    """Where a message stamped `stamp_s` falls in a stream whose newest stamp
    so far is `newest_s`: STAMP_ADVANCE (in order, or the same instant
    again), STAMP_REORDERED (slightly older: skip it, the newer state already
    covers that moment) or STAMP_RESTART (more than `restart_jump_s` older:
    the source restarted and the old state belongs to another timeline)."""
    if newest_s is None or stamp_s >= newest_s:
        return STAMP_ADVANCE
    if newest_s - stamp_s > restart_jump_s:
        return STAMP_RESTART
    return STAMP_REORDERED
