"""Head-region crop, derived from keypoints rather than a fixed fraction of
the bbox - the same approach human_tracking's torso_crop_bbox uses, for the
same reason: a fixed fraction doesn't adapt to how much of the person is
actually in frame, but a keypoint-derived region does.

Geometry matches the project's design doc ("Pipeline · B1 Orientarea
capului" / "Parametri · B Cap + Gating" in
~/Downloads/Pipeline awareness robot2.pdf): a square crop anchored at the
neck, sized from body scale rather than the bbox, specifically *because* the
bbox itself is an unreliable size reference - it changes with raised arms,
a leg out of frame, or a lean, while the neck-to-shoulder geometry stays
stable and (unlike the face) is still visible from behind.

There's no "neck" keypoint in COCO-17, so it's approximated as the midpoint
of the two shoulder keypoints - the usual stand-in when only body keypoints
(not a dedicated face/head landmark set) are available.

One simplification vs. the design doc: the doc places the crop "along the
hip-to-neck axis" (so a leaning/tilted body gets a correspondingly tilted
crop); this only ever places it along the vertical image axis. Revisit if
leaning turns out to matter in practice - see docs/context/head_pose.md.
"""

import math

# COCO-17 indices.
_NOSE = 0
_LEFT_SHOULDER, _RIGHT_SHOULDER = 5, 6


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def neck_point(
    keypoints: list[tuple[float, float, float]],
    min_keypoint_conf: float = 0.3,
) -> tuple[float, float] | None:
    """Midpoint of the shoulder keypoints when both are confident. Falls
    back to whichever single shoulder is confident, when only one is - a
    turned/three-quarter body stance commonly foreshortens the far shoulder
    below confidence while the face stays clearly visible and gaze-readable;
    empirically confirmed (see docs/context/head_pose.md) to be costing real
    recall by skipping pose entirely in exactly that case. In that fallback,
    the single shoulder's raw x is offset to one side of the actual head
    (it's not the body's center the way a two-shoulder midpoint is) - pulled
    halfway toward the nose's x when the nose is confident too, since the
    nose sits roughly above the true head center regardless of which
    shoulder dropped out, which empirically produces a better-centered (and
    so more reliable) crop than the raw shoulder position alone. None only
    when neither shoulder clears the bar."""
    if len(keypoints) <= _RIGHT_SHOULDER:
        return None
    left, right = keypoints[_LEFT_SHOULDER], keypoints[_RIGHT_SHOULDER]
    left_confident = left[2] >= min_keypoint_conf
    right_confident = right[2] >= min_keypoint_conf
    if left_confident and right_confident:
        return (left[0] + right[0]) / 2.0, (left[1] + right[1]) / 2.0
    if not left_confident and not right_confident:
        return None
    shoulder = left if left_confident else right
    nose = keypoints[_NOSE] if len(keypoints) > _NOSE else None
    if nose is not None and nose[2] >= min_keypoint_conf:
        return (shoulder[0] + nose[0]) / 2.0, shoulder[1]
    return shoulder[0], shoulder[1]


def _head_scale(
    keypoints: list[tuple[float, float, float]],
    neck: tuple[float, float],
    min_keypoint_conf: float,
    head_scale_alpha: float,
) -> float:
    """max(shoulder width, alpha * neck-to-nose distance) - the neck-to-nose
    term is what keeps the scale from collapsing in profile, where shoulder
    width visually shrinks but the neck is still roughly as far from the
    (now side-on) nose. Shoulder width only counts when BOTH shoulders are
    confident - when `neck_point` fell back to a single shoulder, the other
    shoulder's own (unconfident) position is not a trustworthy distance
    endpoint, so this leans entirely on the neck-to-nose term in that case."""
    left, right = keypoints[_LEFT_SHOULDER], keypoints[_RIGHT_SHOULDER]
    if left[2] >= min_keypoint_conf and right[2] >= min_keypoint_conf:
        shoulder_width = _distance((left[0], left[1]), (right[0], right[1]))
    else:
        shoulder_width = 0.0

    nose = keypoints[_NOSE] if len(keypoints) > _NOSE else None
    if nose is not None and nose[2] >= min_keypoint_conf:
        neck_to_nose = _distance(neck, (nose[0], nose[1]))
        return max(shoulder_width, head_scale_alpha * neck_to_nose)
    return shoulder_width


def head_crop_bbox(
    keypoints: list[tuple[float, float, float]],
    min_keypoint_conf: float = 0.3,
    head_k_h: float = 0.8,
    head_scale_alpha: float = 1.7,
    crop_margin: float = 0.2,
    min_head_px: float = 35.0,
) -> tuple[float, float, float, float] | None:
    """Square crop from the neck upward, sized `head_k_h * scale` (+ a
    `crop_margin` fraction of padding on each side), where `scale` is
    `_head_scale` above. None if the neck can't be estimated, or the result
    would be smaller than `min_head_px` - too small to trust a pose estimate
    from (the model itself wasn't evaluated on crops that degraded)."""
    neck = neck_point(keypoints, min_keypoint_conf)
    if neck is None:
        return None

    scale = _head_scale(keypoints, neck, min_keypoint_conf, head_scale_alpha)
    side = head_k_h * scale * (1.0 + 2.0 * crop_margin)
    if side < min_head_px:
        return None

    neck_x, neck_y = neck
    half = side / 2.0
    return (neck_x - half, neck_y - side, neck_x + half, neck_y)
