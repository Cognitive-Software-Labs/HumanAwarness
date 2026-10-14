"""Keypoint-derived body orientation - cheap, viewpoint-explaining signal for ReID.

Not head pose: head pose models (e.g. 6DRepNet) need a detected face crop to
run at all, so they have nothing to say exactly when it matters most here - a
person showing their back to the camera. This instead reads the same COCO-17
keypoints human_tracking already has (no new model/dependency) and asks a
much cruder question: is the face side or the back of the head toward the
camera right now?

Why this helps ReID (see tracker.py's fusion in Track.best_similarity): OSNet
was trained on pedestrian crops of all viewpoints, but was never told to make
the embedding viewpoint-*invariant* - a real front-view and back-view sample
of the same person can legitimately score low on pure appearance similarity,
which looks identical to "different person" to a bare threshold. Knowing
*both* samples' orientation lets the tracker tell those two cases apart and
grant a small, targeted similarity relief only when there's direct evidence
(a known orientation mismatch) explaining why the score is low - rather than
lowering the threshold for everyone, which is what caused the real crowd-scene
false merges already documented in docs/context/human_tracking.md.
"""

FRONTAL = "FRONTAL"
BACK = "BACK"
PROFILE = "PROFILE"
UNKNOWN = "UNKNOWN"

# COCO-17 indices.
_NOSE, _LEYE, _REYE, _LEAR, _REAR = 0, 1, 2, 3, 4

# Mismatch severity between orientation labels, as a fraction of the base
# relief: a full front/back flip is the largest, best-evidenced appearance
# change; a profile is only a partial rotation away from either. Same label
# (or either side UNKNOWN) gets no relief - there's no confident evidence to
# justify one.
_MISMATCH_SEVERITY = {
    frozenset({FRONTAL, BACK}): 1.0,
    frozenset({FRONTAL, PROFILE}): 0.5,
    frozenset({BACK, PROFILE}): 0.5,
}


def estimate_orientation(
    keypoints: list[tuple[float, float, float]],
    min_conf: float = 0.4,
) -> str:
    """Classify which way a person is facing, from head/shoulder keypoints alone.

    - FRONTAL: the face is actually visible (nose, or both eyes, confident).
    - BACK: the head is visible but the face isn't - at least one ear is
      confident while the nose and both eyes are not - i.e. the back of the
      head is toward the camera, not an occluded/undetected head.
    - PROFILE: exactly one eye or one ear confident, not both - a side view.
    - UNKNOWN: not enough confident head keypoints to tell (occluded,
      cropped out, or the detector just didn't find them) - callers must not
      grant any orientation-based relief in this case, there's no evidence.

    Args:
        keypoints: per-keypoint (x, y, confidence) tuples in COCO-17 order
            (any length - only indices 0-4 are used).
    """
    if len(keypoints) <= _REAR:
        return UNKNOWN

    nose_conf = keypoints[_NOSE][2]
    leye_conf, reye_conf = keypoints[_LEYE][2], keypoints[_REYE][2]
    lear_conf, rear_conf = keypoints[_LEAR][2], keypoints[_REAR][2]

    face_visible = nose_conf >= min_conf or (leye_conf >= min_conf and reye_conf >= min_conf)
    if face_visible:
        return FRONTAL

    any_eye = leye_conf >= min_conf or reye_conf >= min_conf
    any_ear = lear_conf >= min_conf or rear_conf >= min_conf
    if any_eye:
        return PROFILE  # one eye but not both, and no clean nose - side-on

    if any_ear:
        return BACK  # head visible (ear), face not - back of the head

    return UNKNOWN


def orientation_mismatch_relief(orientation_a: str, orientation_b: str, base_relief: float) -> float:
    """Similarity bonus to grant a cross-orientation comparison, scaled by how
    severe the known mismatch is. Zero whenever either side is UNKNOWN or
    they match - no evidence, no relief."""
    if orientation_a == UNKNOWN or orientation_b == UNKNOWN or orientation_a == orientation_b:
        return 0.0
    severity = _MISMATCH_SEVERITY.get(frozenset({orientation_a, orientation_b}), 0.0)
    return base_relief * severity
