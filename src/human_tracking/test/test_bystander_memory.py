"""Synthetic tests for the co-occurrence ("bystander") constraints in Tracker.

No model weights or video needed: detections carry hand-made unit
embeddings with controlled cosine similarities, and plausible full-body
keypoints so the crop-plausibility gate lets the embeddings through.
"""

import numpy as np

from human_tracking.track import ACTIVE
from human_tracking.tracker import Detection, Tracker

DT = 1.0 / 30.0
FRAME_AREA = 1280.0 * 720.0


def _unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


def _emb_with_similarity(base: np.ndarray, sim: float, axis: int) -> np.ndarray:
    """Unit vector at cosine `sim` from `base` (base must be axis-aligned on a different axis)."""
    other = np.zeros_like(base)
    other[axis] = 1.0
    return _unit(sim * base + np.sqrt(1.0 - sim**2) * other)


def _det(bbox, emb):
    x1, y1, x2, y2 = bbox
    # 17 confident keypoints spread over the full box height: passes
    # is_plausible_person_crop, orientation reads FRONTAL (nose visible).
    kps = [((x1 + x2) / 2.0, y1 + (y2 - y1) * (i + 0.5) / 17.0, 0.9) for i in range(17)]
    return Detection(bbox_xyxy=bbox, confidence=0.9, keypoints=kps, embedding=emb)


def _tracker(**kwargs):
    return Tracker(**kwargs)


E_A = _unit([1, 0, 0, 0, 0, 0])
BOX_A = (100.0, 100.0, 200.0, 400.0)
BOX_B = (330.0, 100.0, 430.0, 400.0)  # 230px from A's center: within the gate after ~0.3s lost


def _run_lookalike_bystander_scenario(tracker: Tracker, sim_b_to_a: float):
    """A is tracked; B (a different person who looks like A) appears while A is
    still visible; then A leaves. Returns (id_of_A, published tracks at the end)."""
    e_b = _emb_with_similarity(E_A, sim_b_to_a, axis=1)
    # frames 0-9: A alone
    for _ in range(10):
        tracker.update([_det(BOX_A, E_A)], FRAME_AREA, DT)
    # frames 10-19: A and B together (B's track is young, co-visible with A)
    for _ in range(10):
        tracker.update([_det(BOX_A, E_A), _det(BOX_B, e_b)], FRAME_AREA, DT)
    id_a = next(t.track_id for t in tracker.tracks if t.last_bbox == BOX_A)
    # frames 20-29: A gone, B still there and still young (< reconcile_window_s)
    published = []
    for _ in range(10):
        published = tracker.update([_det(BOX_B, e_b)], FRAME_AREA, DT)
    return id_a, published


def test_baseline_reconciliation_falsely_merges_a_lookalike_bystander():
    """Documents the failure mode being fixed: without the constraint, a young
    track that was visible *at the same time* as the lost identity is still
    reclaimed by it once the lost one disappears."""
    tracker = _tracker(cooccurrence_veto=False)
    id_a, published = _run_lookalike_bystander_scenario(tracker, sim_b_to_a=0.85)
    assert [t.track_id for t in published if t.state == ACTIVE] == [id_a]
    assert len(tracker.merged_ids) == 1


def test_cooccurrence_veto_blocks_merging_two_simultaneously_seen_tracks():
    tracker = _tracker(cooccurrence_veto=True)
    id_a, published = _run_lookalike_bystander_scenario(tracker, sim_b_to_a=0.85)
    active_ids = [t.track_id for t in published if t.state == ACTIVE]
    assert len(active_ids) == 1 and active_ids[0] != id_a
    assert tracker.merged_ids == {}
    assert any(ev["kind"] == "cooccurrence" for ev in tracker.veto_events)


def test_cooccurrence_veto_still_allows_a_genuine_reappearance():
    """Same person leaves and comes back (never co-visible with themselves):
    reconciliation must still reclaim the original id."""
    tracker = _tracker(cooccurrence_veto=True)
    for _ in range(10):
        tracker.update([_det(BOX_A, E_A)], FRAME_AREA, DT)
    id_a = tracker.tracks[0].track_id
    for _ in range(10):
        tracker.update([], FRAME_AREA, DT)
    e_back = _emb_with_similarity(E_A, 0.8, axis=2)
    # first reappearance frames too degraded to embed, so a young track claims
    # the detection; reconciliation must hand it back once embeddings arrive
    for _ in range(3):
        tracker.update([_det(BOX_B, None)], FRAME_AREA, DT)
    published = []
    for _ in range(3):
        published = tracker.update([_det(BOX_B, e_back)], FRAME_AREA, DT)
    assert [t.track_id for t in published if t.state == ACTIVE] == [id_a]
    assert tracker.veto_events == []


def test_overlapping_boxes_do_not_record_cooccurrence():
    """Two heavily overlapping boxes (a fragment the containment suppression
    missed, or two people crossing) are not evidence of two different people."""
    tracker = _tracker(cooccurrence_veto=True, cooccurrence_max_overlap=0.5)
    # 80% of the shifted box lies inside BOX_A: below the 0.9 containment
    # suppression (both detections survive), above the 0.5 co-occurrence cap
    shifted = (120.0, 100.0, 220.0, 400.0)
    for _ in range(5):
        tracker.update([_det(BOX_A, E_A), _det(shifted, _emb_with_similarity(E_A, 0.3, 1))], FRAME_AREA, DT)
    a, b = tracker.tracks
    assert b.uid not in a.co_visible and a.uid not in b.co_visible


def test_separate_boxes_record_cooccurrence_symmetrically():
    tracker = _tracker(cooccurrence_veto=True)
    e_b = _emb_with_similarity(E_A, 0.2, axis=1)
    for _ in range(3):
        tracker.update([_det(BOX_A, E_A), _det(BOX_B, e_b)], FRAME_AREA, DT)
    a, b = tracker.tracks
    assert b.uid in a.co_visible and a.uid in b.co_visible


def test_reclaimed_identity_inherits_young_tracks_cooccurrence():
    """After young track Y is merged into lost identity L, everything Y was
    seen alongside is also known to be a different person from L."""
    tracker = _tracker(cooccurrence_veto=True)
    for _ in range(10):
        tracker.update([_det(BOX_A, E_A)], FRAME_AREA, DT)
    track_a = tracker.tracks[0]
    for _ in range(10):
        tracker.update([], FRAME_AREA, DT)
    e_back = _emb_with_similarity(E_A, 0.8, axis=2)
    e_c = _emb_with_similarity(E_A, 0.1, axis=3)
    box_c = (700.0, 100.0, 800.0, 400.0)
    for _ in range(3):  # A comes back unembeddable next to a new person C
        tracker.update([_det(BOX_B, None), _det(box_c, e_c)], FRAME_AREA, DT)
    # C leaves before A's young track becomes recognizable, so A and C are
    # never matched in the same frame - the A/C link can only come from Y
    for _ in range(3):
        tracker.update([_det(BOX_B, e_back)], FRAME_AREA, DT)
    track_c = next(t for t in tracker.tracks if t.last_bbox == box_c)
    assert track_a in tracker.tracks and track_a.state == ACTIVE
    assert tracker.merged_ids  # Y really was merged into A
    assert track_c.uid in track_a.co_visible and track_a.uid in track_c.co_visible


def _run_stage2_with_visible_lookalike(tracker: Tracker):
    """A and C are both tracked; A disappears; a new detection then appears that
    resembles A's gallery (0.72) but resembles C - who is visibly elsewhere this
    same frame, so is certainly not this detection - even more (0.80)."""
    box_c = (700.0, 100.0, 800.0, 400.0)
    e_c = _emb_with_similarity(E_A, 0.55, axis=1)
    for _ in range(10):
        tracker.update([_det(BOX_A, E_A), _det(box_c, e_c)], FRAME_AREA, DT)
    id_a = next(t.track_id for t in tracker.tracks if t.last_bbox == BOX_A)
    for _ in range(10):
        tracker.update([_det(box_c, e_c)], FRAME_AREA, DT)
    # q: sim 0.72 to E_A; build it in the (E_A, e_c-orthogonal-part) plane so
    # that its similarity to e_c is high
    c_perp = _unit(e_c - 0.55 * E_A)
    q = _unit(0.72 * E_A + np.sqrt(1 - 0.72**2) * c_perp)
    assert float(np.dot(q, e_c)) > 0.75
    published = tracker.update([_det(box_c, e_c), _det(BOX_B, q)], FRAME_AREA, DT)
    return id_a, published


def test_stage2_without_negative_check_revives_lost_track_on_lookalike():
    tracker = _tracker(bystander_negative_check=False)
    id_a, published = _run_stage2_with_visible_lookalike(tracker)
    assert id_a in [t.track_id for t in published if t.state == ACTIVE]


def test_stage2_negative_check_blocks_detection_resembling_a_visible_bystander():
    tracker = _tracker(bystander_negative_check=True, bystander_margin=0.0)
    id_a, published = _run_stage2_with_visible_lookalike(tracker)
    assert id_a not in [t.track_id for t in published if t.state == ACTIVE]
    assert any(ev["kind"] == "bystander" for ev in tracker.veto_events)
