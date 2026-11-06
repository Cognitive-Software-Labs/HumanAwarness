import math
from types import SimpleNamespace

import pytest

from human_awareness.adapters import (
    STAMP_ADVANCE,
    STAMP_REORDERED,
    STAMP_RESTART,
    classify_stamp,
    person_detection_to_cue,
)
from human_awareness.awareness import SOURCE_HEAD, SOURCE_IRIS, AwarenessAggregator, AwarenessCue

SEES_HEAD = AwarenessCue(True, SOURCE_HEAD)
SEES_IRIS = AwarenessCue(True, SOURCE_IRIS)
AWAY_HEAD = AwarenessCue(False, SOURCE_HEAD)
AWAY_IRIS = AwarenessCue(False, SOURCE_IRIS)


def feed(agg, track_id, cue, t_start, t_end, fps):
    """One `observe` per frame at t_start, t_start + 1/fps, ... t_end (inclusive)."""
    n = round((t_end - t_start) * fps)
    est = None
    for i in range(n + 1):
        est = agg.observe(track_id, t_start + i / fps, cue)
    return est


def test_new_track_starts_unknown():
    est = AwarenessAggregator().observe(1, 10.0, SEES_HEAD)
    assert (est.value, est.confidence) == (0.5, 0.0)


def test_missing_cues_stay_unknown():
    agg = AwarenessAggregator()
    est = feed(agg, 1, None, 0.0, 30.0, fps=30)
    assert est.value == pytest.approx(0.5)
    assert est.confidence == 0.0


def test_looking_rises_with_head_time_constant():
    agg = AwarenessAggregator(rise_time_s=0.5, weight_head=0.5)
    est = feed(agg, 1, SEES_HEAD, 0.0, 1.0, fps=30)
    # 1 s of head-only evidence at rate 0.5 / 0.5 s = one time constant.
    assert est.value == pytest.approx(1.0 - 0.5 * math.exp(-1.0))
    assert est.confidence == pytest.approx(1.0 - math.exp(-0.5 * 1.0 / 2.0))


def test_iris_counts_more_than_head():
    head, iris = AwarenessAggregator(), AwarenessAggregator()
    assert feed(iris, 1, SEES_IRIS, 0.0, 0.5, 30).value > feed(head, 1, SEES_HEAD, 0.0, 0.5, 30).value
    head_away = feed(head, 2, AWAY_HEAD, 0.0, 60.0, 30)
    iris_away = feed(iris, 2, AWAY_IRIS, 0.0, 60.0, 30)
    # Watching someone not look: iris settles near 0, head pose only halfway to the prior.
    assert iris_away.value == pytest.approx(0.0, abs=0.01)
    assert head_away.value == pytest.approx(0.25, abs=0.01)
    assert iris_away.confidence > head_away.confidence


def test_one_frame_flicker_barely_moves_it():
    agg = AwarenessAggregator()
    before = feed(agg, 1, AWAY_HEAD, 0.0, 10.0, 30)
    agg.observe(1, 10.0 + 1 / 30, SEES_HEAD)
    peak = max(agg.observe(1, 10.0 + i / 30, AWAY_HEAD).value for i in range(2, 30))
    assert peak - before.value < 0.03


def test_fades_over_tau_after_looking():
    agg = AwarenessAggregator(tau_s=8.0)
    feed(agg, 1, SEES_HEAD, 0.0, 3.0, 30)
    start = agg.observe(1, 3.0 + 1 / 30, AWAY_HEAD)  # the held SEES cue ends here
    end = feed(agg, 1, AWAY_HEAD, 3.0 + 1 / 30, 11.0 + 1 / 30, 30)
    assert end.value - 0.25 == pytest.approx((start.value - 0.25) * math.exp(-1.0))
    assert end.value < start.value


def test_unseen_person_relaxes_to_unknown_not_to_zero():
    agg = AwarenessAggregator(tau_s=8.0, cue_max_age_s=0.5)
    feed(agg, 1, SEES_HEAD, 0.0, 3.0, 30)
    held = agg.observe(1, 3.5, None)  # last cue still holds for 0.5 s
    later = agg.observe(1, 11.5, None)
    assert later.value - 0.5 == pytest.approx((held.value - 0.5) * math.exp(-1.0))
    assert later.confidence == pytest.approx(held.confidence * math.exp(-1.0))
    assert feed(agg, 1, None, 11.5, 200.0, 1).value == pytest.approx(0.5, abs=1e-6)


def test_cue_counts_for_at_most_cue_max_age():
    agg = AwarenessAggregator(tau_s=8.0, cue_max_age_s=0.5, rise_time_s=0.5, weight_head=0.5)
    agg.observe(1, 0.0, SEES_HEAD)
    est = agg.observe(1, 2.0, None)
    after_hold = 1.0 - 0.5 * math.exp(-0.5 * 0.5 / 0.5)
    assert est.value == pytest.approx(0.5 + (after_hold - 0.5) * math.exp(-1.5 / 8.0))


def test_same_result_at_any_frame_rate():
    def run(fps):
        agg = AwarenessAggregator()
        feed(agg, 1, SEES_HEAD, 0.0, 2.0, fps)
        feed(agg, 1, AWAY_HEAD, 2.1, 5.0, fps)
        feed(agg, 1, None, 5.1, 9.0, fps)
        return feed(agg, 1, SEES_IRIS, 9.1, 10.0, fps)

    reference = run(10)
    for fps in (30, 60):
        est = run(fps)
        assert est.value == pytest.approx(reference.value, abs=1e-9)
        assert est.confidence == pytest.approx(reference.confidence, abs=1e-9)


def test_state_survives_lost_and_resumes():
    agg = AwarenessAggregator()
    peak = feed(agg, 7, SEES_HEAD, 0.0, 2.0, 30)
    lost = feed(agg, 7, None, 2.0 + 1 / 30, 5.0, 30)  # LOST_KALMAN/LOST_GALLERY: published, no cue
    agg.prune({7})
    assert agg.get(7) == lost
    assert 0.5 < lost.value < peak.value
    assert 0.0 < lost.confidence < peak.confidence
    back = feed(agg, 7, SEES_HEAD, 5.0 + 1 / 30, 5.5, 30)
    assert back.value > lost.value


def test_prune_forgets_tracks_no_longer_published():
    agg = AwarenessAggregator()
    feed(agg, 1, SEES_HEAD, 0.0, 2.0, 30)
    feed(agg, 2, SEES_HEAD, 0.0, 2.0, 30)
    agg.prune({2})
    assert agg.get(1) is None
    est = agg.observe(1, 3.0, SEES_HEAD)
    assert (est.value, est.confidence) == (0.5, 0.0)
    assert agg.get(2).value > 0.5


def test_out_of_order_stamp_changes_nothing():
    agg = AwarenessAggregator()
    now = feed(agg, 1, SEES_HEAD, 0.0, 1.0, 30)
    assert agg.observe(1, 0.5, AWAY_IRIS) == now
    assert agg.observe(1, 1.0 + 1 / 30, SEES_HEAD).value > now.value  # the stale AWAY cue wasn't recorded


def test_unknown_source_rejected():
    with pytest.raises(ValueError):
        AwarenessAggregator().observe(1, 0.0, AwarenessCue(True, "body"))


@pytest.mark.parametrize(
    "kwargs",
    [{"tau_s": 0.0}, {"cue_max_age_s": 0.0}, {"rise_time_s": -1.0}, {"weight_head": 0.0}, {"weight_iris": 1.5},
     {"init_value": 1.2}],
)
def test_rejects_bad_parameters(kwargs):
    with pytest.raises(ValueError):
        AwarenessAggregator(**kwargs)


def _det(**overrides):
    fields = dict(
        track_id=3, state="ACTIVE", head_pose_valid=True, distance_method="shoulder_width",
        gaze_source="head", sees_robot=True,
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


def test_adapter_builds_cue_from_a_fresh_decision():
    assert person_detection_to_cue(_det()) == SEES_HEAD
    assert person_detection_to_cue(_det(gaze_source="iris", sees_robot=False)) == AWAY_IRIS


@pytest.mark.parametrize(
    "overrides",
    [{"state": "LOST_KALMAN"}, {"state": "LOST_GALLERY"}, {"head_pose_valid": False}, {"distance_method": "none"},
     {"distance_method": ""}, {"gaze_source": ""}],
)
def test_adapter_treats_anything_else_as_unknown(overrides):
    assert person_detection_to_cue(_det(**overrides)) is None


@pytest.mark.parametrize(
    "stamp_s, newest_s, expected",
    [
        (5.0, None, STAMP_ADVANCE),  # first message
        (5.0, 5.0, STAMP_ADVANCE),  # same instant again
        (5.1, 5.0, STAMP_ADVANCE),
        (4.99, 5.0, STAMP_REORDERED),  # a few ms out of order: skip, don't reset
        (4.0, 5.0, STAMP_REORDERED),  # exactly restart_jump_s back is still a reorder
        (3.9, 5.0, STAMP_RESTART),
        (0.5, 600.0, STAMP_RESTART),  # a looping bag starting over
    ],
)
def test_classify_stamp(stamp_s, newest_s, expected):
    assert classify_stamp(stamp_s, newest_s, restart_jump_s=1.0) == expected
