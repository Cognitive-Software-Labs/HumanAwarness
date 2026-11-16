import json
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from human_run_recorder.run_files import RunWriter, debug_lines, detection_to_row, estimate_fps


def det(**overrides):
    """A PersonDetection as awareness_node publishes it, by attribute."""
    fields = dict(
        track_id=3, state="ACTIVE", confidence=0.9,
        bbox_x=100.0, bbox_y=120.0, bbox_width=80.0, bbox_height=200.0,
        head_pose_valid=True, head_yaw_deg=5.0, head_pitch_deg=-10.0, head_roll_deg=1.0,
        bearing_deg=2.0, gaze_robot_deg=3.0, gaze_source="iris", gaze_gate_reason="ok",
        iris_valid=True, iris_fixating=True, iris_yaw_deg=4.0, iris_offset=0.2, eyes_averted=True,
        distance_m=1.2, distance_method="shoulder_width", theta_to_robot_deg=25.0,
        sees_robot=False, direct_contact=False, awareness_value=0.8, awareness_confidence=0.6,
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


def frame(value=0):
    return np.full((48, 64, 3), value, np.uint8)


def test_row_has_the_export_fields_the_review_tools_read():
    row = detection_to_row(det(), frame=7, timestamp_s=1.5, stamp_ns=123, aware_threshold_deg=45.0)
    assert row["frame"] == 7 and row["track_id"] == 3 and row["bbox"] == [100, 120, 180, 320]
    assert row["head_pose_valid"] is True and row["is_looking"] is False  # phase 2 ran: sees_robot decides
    assert row["iris_offset"] == pytest.approx(0.2) and row["eyes_averted"] is True
    assert row["gaze_gate_reason"] == "ok" and row["awareness_value"] == pytest.approx(0.8)


def test_row_uses_none_where_nothing_was_measured():
    row = detection_to_row(
        det(head_pose_valid=False, iris_offset=-1.0, iris_fixating=False, distance_method="", gaze_source=""),
        0, 0.0, 0, 45.0,
    )
    assert row["head_yaw_deg"] is None and row["iris_offset"] is None and row["iris_yaw_deg"] is None
    assert row["distance_m"] is None and row["theta_to_robot_deg"] is None and row["gaze_source"] is None
    assert debug_lines(row) == []


def test_lost_gallery_is_not_written():
    assert detection_to_row(det(state="LOST_GALLERY"), 0, 0.0, 0, 45.0) is None


def test_untracked_detection_has_no_awareness():
    row = detection_to_row(det(track_id=-1, state=""), 0, 0.0, 0, 45.0)
    assert row["awareness_value"] is None and row["awareness_confidence"] is None


def test_debug_lines_show_why_it_decided():
    text = " | ".join(debug_lines(detection_to_row(det(), 0, 0.0, 0, 45.0)))
    assert "src iris" in text and "theta 25.0" in text and "iris off 0.20 AVERTED" in text


@pytest.mark.parametrize(
    "stamps, expected",
    [([0.0, 0.1, 0.2, 0.3], 10.0), ([5.0], 15.0), ([1.0, 1.0, 1.0], 15.0), ([0.0, 100.0], 1.0)],
)
def test_estimate_fps(stamps, expected):
    assert estimate_fps(stamps, fallback_fps=15.0) == pytest.approx(expected)


def test_run_folder_round_trip(tmp_path):
    run = RunWriter(tmp_path / "run", probe_frames=3, fallback_fps=15.0)
    for i in range(5):
        stamp_ns = 1_000_000_000 + i * 100_000_000  # 10 fps
        run.add_frame(frame(i * 40), [det(), det(state="LOST_GALLERY", track_id=4)], stamp_ns)
        if i == 2:
            marker = run.add_marker("said looking, eyes were away", "2026-11-16T12:00:00")
    run.add_log_line("1.0 frame=5 WARN [gaze_node] something")
    run.close()

    assert marker["frame"] == 2 and marker["note"] == "said looking, eyes were away"
    rows = [json.loads(line) for line in (tmp_path / "run" / "predictions.jsonl").read_text().splitlines()]
    assert [r["frame"] for r in rows] == [0, 1, 2, 3, 4]  # LOST_GALLERY skipped
    assert rows[4]["timestamp_s"] == pytest.approx(0.4)
    assert run.summary()["video_fps"] == pytest.approx(10.0)
    for name in ("camera.mkv", "annotated.mkv"):
        cap = cv2.VideoCapture(str(tmp_path / "run" / name))
        n = 0
        while cap.read()[0]:
            n += 1
        assert n == 5, name  # frame N of the video is "frame" N in predictions.jsonl
    assert "WARN [gaze_node]" in (tmp_path / "run" / "rosout.log").read_text()


def test_short_run_still_writes_its_video(tmp_path):
    run = RunWriter(tmp_path / "run", probe_frames=15, record_annotated=False)
    run.add_frame(frame(), [det()], 1)
    run.close()
    assert (tmp_path / "run" / "camera.mkv").stat().st_size > 0
    assert not (tmp_path / "run" / "annotated.mkv").exists()


def test_existing_run_folder_is_never_overwritten(tmp_path):
    (tmp_path / "run").mkdir()
    with pytest.raises(FileExistsError):
        RunWriter(tmp_path / "run")
