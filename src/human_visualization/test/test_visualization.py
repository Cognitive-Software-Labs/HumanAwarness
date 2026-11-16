from types import SimpleNamespace

import numpy as np
import pytest

from human_visualization.adapters import ImageBuffer, is_looking, person_detection_to_overlay
from human_visualization.overlay import (
    ACTIVE_COLOR,
    PREDICTED_COLOR,
    PersonOverlay,
    draw_info,
    draw_person,
    looking_label,
)


def det(**overrides):
    """A PersonDetection as gaze_node + awareness_node publish it, by attribute."""
    fields = dict(
        track_id=3, state="ACTIVE", bbox_x=100.0, bbox_y=120.0, bbox_width=80.0, bbox_height=200.0,
        head_pose_valid=True, gaze_robot_deg=10.0, distance_method="ground_plane", sees_robot=True,
        awareness_value=0.8, awareness_confidence=0.6,
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


# --- is_looking: the same rule as the video script ---------------------------------------


def test_looking_follows_phase2_decision_when_it_ran():
    assert is_looking(det(sees_robot=True, gaze_robot_deg=170.0), 45.0)
    assert not is_looking(det(sees_robot=False, gaze_robot_deg=0.0), 45.0)


@pytest.mark.parametrize("method", ["", "none"])
def test_looking_falls_back_to_flat_threshold_without_phase2(method):
    assert is_looking(det(distance_method=method, sees_robot=False, gaze_robot_deg=-30.0), 45.0)
    assert not is_looking(det(distance_method=method, sees_robot=True, gaze_robot_deg=60.0), 45.0)


def test_not_looking_without_head_pose():
    assert not is_looking(det(head_pose_valid=False, sees_robot=True), 45.0)


# --- person_detection_to_overlay -----------------------------------------------------------


def test_overlay_carries_bbox_id_label_and_awareness():
    o = person_detection_to_overlay(det())
    assert o.bbox_xyxy == (100.0, 120.0, 180.0, 320.0)
    assert o.track_id == 3 and not o.predicted
    assert o.head_pose_valid is True and o.is_looking
    assert o.awareness == (0.8, 0.6)


def test_lost_kalman_is_drawn_as_predicted_and_lost_gallery_is_skipped():
    assert person_detection_to_overlay(det(state="LOST_KALMAN")).predicted
    assert person_detection_to_overlay(det(state="LOST_GALLERY")) is None


def test_untracked_detection_gets_box_only():
    o = person_detection_to_overlay(det(track_id=-1, state=""))
    assert o.track_id == -1
    assert o.head_pose_valid is None and o.awareness is None


def test_show_flags_hide_label_and_awareness():
    o = person_detection_to_overlay(det(), show_gaze=False, show_awareness=False)
    assert o.head_pose_valid is None and o.awareness is None


# --- ImageBuffer ------------------------------------------------------------------------------


def test_buffer_pops_exact_stamp_and_drops_older_frames():
    buf = ImageBuffer(10)
    for s in (10, 20, 30, 40):
        buf.add(s, f"img{s}")
    assert buf.pop_match(30) == "img30"
    assert len(buf) == 1 and buf.oldest_ns() == 40  # 10, 20 and 30 can't be asked for again


def test_buffer_miss_leaves_frames_in_place():
    buf = ImageBuffer(10)
    for s in (10, 20, 30):
        buf.add(s, f"img{s}")
    assert buf.pop_match(25) is None  # a dropped frame
    assert buf.pop_match(5) is None  # older than anything buffered
    assert len(buf) == 3
    assert buf.pop_match(20) == "img20"


def test_buffer_forgets_beyond_maxlen():
    buf = ImageBuffer(3)
    for s in range(1, 6):
        buf.add(s, s)
    assert buf.oldest_ns() == 3 and buf.newest() == 5
    assert buf.pop_match(1) is None


def test_empty_buffer():
    buf = ImageBuffer(3)
    assert buf.pop_match(1) is None and buf.oldest_ns() is None and buf.newest() is None


# --- drawing ------------------------------------------------------------------------------------


def blank():
    return np.zeros((480, 640, 3), dtype=np.uint8)


def test_box_color_marks_active_vs_predicted():
    for predicted, color in ((False, ACTIVE_COLOR), (True, PREDICTED_COLOR)):
        img = blank()
        draw_person(img, PersonOverlay(bbox_xyxy=(100, 150, 200, 400), track_id=1, predicted=predicted))
        assert tuple(img[300, 100]) == color  # left edge of the box


def test_drawing_near_the_frame_edge_does_not_crash():
    img = blank()
    draw_person(
        img,
        PersonOverlay(
            bbox_xyxy=(600, 0, 700, 500), track_id=7, head_crop_bbox=(-10, -5, 50, 40),
            face_bbox=(610, 5, 660, 60), head_pose_valid=True, is_looking=True, awareness=(1.0, 1.0),
        ),
    )
    draw_info(img, ["people: 1", "lag: 120 ms"])
    assert img.any()


def test_labels_of_a_person_at_the_right_edge_stay_in_frame():
    img = blank()
    draw_person(img, PersonOverlay(bbox_xyxy=(630, 200, 700, 400), track_id=12, head_pose_valid=True))
    id_row = img[175:192]  # the "id 12" text row, above the box
    cols = np.nonzero(id_row.any(axis=(0, 2)))[0]
    assert cols.min() < 600  # shifted left rather than drawn off the edge at x=630


def test_looking_label_text():
    assert looking_label(True, False)[0] == "LOOKING AT ROBOT"
    assert looking_label(False, True)[0] == "NOT LOOKING (eyes away)"
    assert looking_label(False, False)[0] == "NOT LOOKING"
