import itertools
import random

import pytest

from human_tracking.tracker import (
    ASSIGNMENT_GREEDY,
    ASSIGNMENT_HUNGARIAN,
    Detection,
    Tracker,
    compute_iou,
    greedy_assignment,
    optimal_assignment,
)

NO_KEYPOINTS = [(0.0, 0.0, 0.0)] * 17  # implausible crop -> no embedding, motion-only


def _det(bbox):
    return Detection(bbox_xyxy=bbox, confidence=0.9, keypoints=NO_KEYPOINTS, embedding=None)


def _brute_force_best_total(pairs):
    """Exhaustive maximum-total-score matching, for checking small cases."""
    best = 0.0
    for k in range(len(pairs) + 1):
        for subset in itertools.combinations(pairs, k):
            rows = [r for _s, r, _c in subset]
            cols = [c for _s, _r, c in subset]
            if len(set(rows)) == len(rows) and len(set(cols)) == len(cols):
                best = max(best, sum(s for s, _r, _c in subset))
    return best


def test_empty_pairs():
    assert optimal_assignment([]) == []
    assert greedy_assignment([]) == []


def test_no_conflict_greedy_and_optimal_agree():
    pairs = [(0.9, 0, 0), (0.7, 1, 1), (0.4, 2, 3)]
    assert set(greedy_assignment(pairs)) == set(optimal_assignment(pairs)) == {(0, 0), (1, 1), (2, 3)}


def test_crossing_case_optimal_keeps_both_matches():
    # Greedy grabs the single best pair (0,0) and strands row 1 and col 1;
    # the optimal matching uses both off-diagonal pairs for a higher total.
    pairs = [(0.6, 0, 0), (0.5, 0, 1), (0.55, 1, 0)]
    assert set(greedy_assignment(pairs)) == {(0, 0)}
    assert set(optimal_assignment(pairs)) == {(0, 1), (1, 0)}


def test_optimal_returns_matches_in_descending_score_order():
    pairs = [(0.6, 0, 0), (0.5, 0, 1), (0.55, 1, 0)]
    assert optimal_assignment(pairs) == [(1, 0), (0, 1)]


def test_optimal_never_uses_absent_pairs():
    # Row 1 only allows col 5; col 5 also wanted by row 0 with a lower score.
    pairs = [(0.9, 0, 4), (0.3, 0, 5), (0.35, 1, 5)]
    out = optimal_assignment(pairs)
    assert set(out) == {(0, 4), (1, 5)}
    assert all((r, c) in {(r2, c2) for _s, r2, c2 in pairs} for r, c in out)


def test_optimal_matches_brute_force_on_random_small_problems():
    rng = random.Random(0)
    for _ in range(200):
        n_rows, n_cols = rng.randint(1, 4), rng.randint(1, 4)
        pairs = [
            (rng.uniform(0.3, 1.0), r, c)
            for r in range(n_rows)
            for c in range(n_cols)
            if rng.random() < 0.6
        ]
        out = optimal_assignment(pairs)
        score = {(r, c): s for s, r, c in pairs}
        assert len({r for r, _ in out}) == len(out) and len({c for _, c in out}) == len(out)
        assert sum(score[rc] for rc in out) == pytest.approx(_brute_force_best_total(pairs))
        assert sum(score[rc] for rc in out) >= sum(score[rc] for rc in greedy_assignment(pairs)) - 1e-9


def test_invalid_assignment_method_rejected():
    with pytest.raises(ValueError):
        Tracker(assignment_method="auction")


def _crossing_scene():
    track_a, track_b = (0.0, 0.0, 100.0, 100.0), (50.0, 0.0, 150.0, 100.0)
    det_1, det_2 = (20.0, 0.0, 120.0, 100.0), (-40.0, 0.0, 60.0, 100.0)
    # Precondition: greedy's single best pair (A, d1) leaves B with no valid
    # partner, while (A, d2) + (B, d1) are both above the 0.3 threshold.
    assert compute_iou(track_a, det_1) > compute_iou(track_b, det_1) > 0.3
    assert compute_iou(track_a, det_2) > 0.3 > compute_iou(track_b, det_2)
    return [track_a, track_b], [det_1, det_2]


@pytest.mark.parametrize("method,expected_new_tracks", [(ASSIGNMENT_GREEDY, 1), (ASSIGNMENT_HUNGARIAN, 0)])
def test_tracker_stage1_crossing(method, expected_new_tracks):
    first, second = _crossing_scene()
    tracker = Tracker(assignment_method=method, collect_assignment_stats=True)
    tracker.update([_det(b) for b in first], frame_area=1e6, dt_seconds=0.1)
    original = list(tracker.tracks)
    tracker.update([_det(b) for b in second], frame_area=1e6, dt_seconds=0.1)

    survivors = [t for t in tracker.tracks if t in original]
    spawned = [t for t in tracker.tracks if t not in original]
    assert len(spawned) == expected_new_tracks
    assert len(survivors) == 2 - expected_new_tracks
    # Diagnostics see the same disagreement regardless of which method ran.
    assert tracker.assignment_stats["stage1_differ"] == 1
    if method == ASSIGNMENT_HUNGARIAN:
        by_track = {id(t): t.last_bbox for t in survivors}
        assert by_track[id(original[0])] == second[1]
        assert by_track[id(original[1])] == second[0]


def test_stats_off_by_default():
    assert Tracker().assignment_stats is None
