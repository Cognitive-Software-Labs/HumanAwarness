"""Edge-crop appearance gate: a box cut off by the image border is still
motion-tracked, but its embedding never enters a gallery and never acts as a
ReID/reconciliation query (see human_tracking.track.touches_image_edge)."""

import numpy as np
import pytest

from human_tracking.track import (
    ACTIVE,
    EDGE_SIDES_ALL,
    EDGE_SIDES_HORIZONTAL,
    EDGE_SIDES_OFF,
    LOST_GALLERY,
    touches_image_edge,
)
from human_tracking.tracker import Detection, Tracker

FRAME_W, FRAME_H = 640, 480
FRAME_SIZE = (FRAME_W, FRAME_H)
FRAME_AREA = FRAME_W * FRAME_H
DT = 1.0 / 30.0


def _unit(seed: int) -> np.ndarray:
    v = np.random.default_rng(seed).normal(size=512)
    return v / np.linalg.norm(v)


def _keypoints(bbox):
    """17 confident keypoints spread over the box's full height, so the
    detection passes is_plausible_person_crop."""
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2.0
    return [(cx, y1 + (y2 - y1) * (i + 0.5) / 17.0, 0.9) for i in range(17)]


def _det(bbox, emb):
    return Detection(bbox_xyxy=bbox, confidence=0.9, keypoints=_keypoints(bbox), embedding=emb)


def _tracker(sides):
    return Tracker(edge_embedding_sides=sides, edge_embedding_margin_px=2.0)


def _confirm(tracker, bbox, emb, frames=5):
    for _ in range(frames):
        published = tracker.update([_det(bbox, emb)], FRAME_AREA, DT, frame_size=FRAME_SIZE)
    assert len(published) == 1 and published[0].state == ACTIVE
    return published[0]


@pytest.mark.parametrize(
    "bbox,sides,expected",
    [
        ((100, 100, 200, 400), EDGE_SIDES_HORIZONTAL, False),
        ((1, 100, 100, 400), EDGE_SIDES_HORIZONTAL, True),  # left edge
        ((500, 100, 639, 400), EDGE_SIDES_HORIZONTAL, True),  # right edge
        ((100, 0, 200, 479), EDGE_SIDES_HORIZONTAL, False),  # top/bottom not checked
        ((100, 0, 200, 400), EDGE_SIDES_ALL, True),
        ((100, 100, 200, 479), EDGE_SIDES_ALL, True),
        ((1, 100, 100, 400), EDGE_SIDES_OFF, False),
        ((5, 100, 100, 400), EDGE_SIDES_HORIZONTAL, False),  # outside the 2px margin
    ],
)
def test_touches_image_edge(bbox, sides, expected):
    assert touches_image_edge(bbox, FRAME_SIZE, 2.0, sides) is expected


def test_unknown_frame_size_never_gates():
    assert touches_image_edge((0, 0, 640, 480), None, 2.0, EDGE_SIDES_ALL) is False


def test_invalid_sides_rejected():
    with pytest.raises(ValueError):
        touches_image_edge((0, 0, 10, 10), FRAME_SIZE, 2.0, "left")


def test_edge_detection_is_motion_tracked_but_not_added_to_gallery():
    tracker = _tracker(EDGE_SIDES_HORIZONTAL)
    track = _confirm(tracker, (100, 100, 200, 400), _unit(0), frames=3)
    gallery_before = len(track.gallery)
    # Same person drifts into the left border: IoU still matches it.
    for x1 in (60, 20, 1, 0):
        published = tracker.update(
            [_det((x1, 100, x1 + 100, 400), _unit(1))], FRAME_AREA, DT, frame_size=FRAME_SIZE
        )
        assert [t.track_id for t in published] == [track.track_id]
        assert published[0].state == ACTIVE
    # 60 and 20 are inside the frame and added; 1 and 0 touch the edge and are withheld.
    assert len(track.gallery) == gallery_before + 2
    assert tracker.edge_gated_detections == 2


def _lose_then_reappear(sides, reappear_bbox):
    """Confirm a person, lose them long enough for LOST_GALLERY, then show
    the same appearance at `reappear_bbox` (spatially plausible, but far
    enough that IoU can't match it). Returns (original_track, published)."""
    tracker = _tracker(sides)
    emb = _unit(7)
    track = _confirm(tracker, (300, 100, 400, 400), emb)
    for _ in range(int(3.0 / DT)):
        tracker.update([], FRAME_AREA, DT, frame_size=FRAME_SIZE)
    assert track.state == LOST_GALLERY
    published = tracker.update([_det(reappear_bbox, emb)], FRAME_AREA, DT, frame_size=FRAME_SIZE)
    return track, published


def test_edge_reappearance_is_not_reid_matched():
    track, _ = _lose_then_reappear(EDGE_SIDES_HORIZONTAL, (540, 100, 639, 400))
    assert track.state == LOST_GALLERY  # appearance unavailable -> no recovery from this crop


def test_same_reappearance_off_edge_is_reid_matched():
    track, _ = _lose_then_reappear(EDGE_SIDES_HORIZONTAL, (520, 100, 620, 400))
    assert track.state == ACTIVE and track.last_match_via_reid


def test_gate_off_keeps_old_behavior():
    track, _ = _lose_then_reappear(EDGE_SIDES_OFF, (540, 100, 639, 400))
    assert track.state == ACTIVE and track.last_match_via_reid


def test_edge_entry_reconciles_once_fully_inside():
    """A person re-entering at the edge spawns a young track (no appearance
    yet); once their box is fully inside the frame, reconciliation can still
    hand the detection back to the original lost identity."""
    tracker = _tracker(EDGE_SIDES_HORIZONTAL)
    emb = _unit(11)
    original = _confirm(tracker, (300, 100, 400, 400), emb)
    for _ in range(int(3.0 / DT)):
        tracker.update([], FRAME_AREA, DT, frame_size=FRAME_SIZE)
    assert original.state == LOST_GALLERY
    for x2 in (639, 639, 639, 630, 625):
        tracker.update([_det((x2 - 100, 100, x2, 400), emb)], FRAME_AREA, DT, frame_size=FRAME_SIZE)
    assert original.state == ACTIVE
    assert len(tracker.tracks) == 1


@pytest.mark.parametrize("defer,expect_reclaimed", [(True, True), (False, False)])
def test_long_edge_entry_needs_deferred_reconcile_window(defer, expect_reclaimed):
    """Re-entering person whose box hugs the border for 2 s (> reconcile_window_s
    1.5 s) before stepping fully inside: only with the deferred window does the
    young track still get to hand its detection back to the original identity."""
    tracker = Tracker(edge_embedding_sides=EDGE_SIDES_HORIZONTAL, edge_defers_reconcile_window=defer)
    emb = _unit(13)
    original = _confirm(tracker, (300, 100, 400, 400), emb)
    for _ in range(int(3.0 / DT)):
        tracker.update([], FRAME_AREA, DT, frame_size=FRAME_SIZE)
    for _ in range(int(2.0 / DT)):
        tracker.update([_det((539, 100, 639, 400), emb)], FRAME_AREA, DT, frame_size=FRAME_SIZE)
    for x2 in (630, 625, 620):
        tracker.update([_det((x2 - 100, 100, x2, 400), emb)], FRAME_AREA, DT, frame_size=FRAME_SIZE)
    assert (original.state == ACTIVE) is expect_reclaimed
    assert len(tracker.tracks) == (1 if expect_reclaimed else 2)
